#!/usr/bin/env python3
import os, time, argparse
import cv2
import torch
import clip
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# -----------------------
# Cargar objetos y embeddings
# -----------------------
def load_objects_embeddings(objects_folder, model, preprocess, device):
    objects = {}
    for folder in os.listdir(objects_folder):
        path = os.path.join(objects_folder, folder)
        if os.path.isdir(path):
            imgs = [os.path.join(path, f) for f in os.listdir(path)
                    if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
            if imgs:
                embeddings = []
                for img_path in imgs:
                    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(img)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        embeddings.append(emb)
                objects[folder] = embeddings
    return objects

# -----------------------
# Dec YOLO + CLIP + tiles en crops
# -----------------------
def detect_objects_yolo_clip(frame, objects, model, preprocess, device,
                             threshold=0.7, yolo_model=None, save_crops=True,
                             tile_size=(128,128), stride=64):
    detected = []
    results = yolo_model(frame)[0]

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        h_crop, w_crop, _ = crop.shape

        # Sliding window dentro del crop
        for top in range(0, h_crop, stride):
            for left in range(0, w_crop, stride):
                bottom = min(top + tile_size[1], h_crop)
                right = min(left + tile_size[0], w_crop)
                tile = crop[top:bottom, left:right]
                if tile.size == 0:
                    continue

                img_input = preprocess(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(img_input)
                    emb = emb / emb.norm(dim=-1, keepdim=True)

                # Comparar con objetos internos
                for name, emb_list in objects.items():
                    for e in emb_list:
                        sim = (emb @ e.T).item()
                        if sim > threshold:
                            tx1, ty1 = x1 + left, y1 + top
                            tx2, ty2 = x1 + right, y1 + bottom
                            detected.append((name, tx1, ty1, tx2, ty2, sim))

                            print(f"[DETECTED] Objeto: {name}, pos: ({tx1},{ty1},{tx2},{ty2}), similitud: {sim:.2f}")

                            if save_crops:
                                save_dir = os.path.join("detections", name)
                                os.makedirs(save_dir, exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                crop_path = os.path.join(save_dir, f"{timestamp}.jpg")
                                cv2.imwrite(crop_path, tile)

    return detected

# -----------------------
# Abrir camara
# -----------------------
def open_camera(cam_source):
    try:
        cam_idx = int(cam_source)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print("No se pudo abrir la cámara:", cam_source)
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# -----------------------
# Main
# -----------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando device:", device)

    # Cargar CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Cargar objetos
    objects_folder = args.objects_folder
    if not os.path.exists(objects_folder):
        print("Carpeta de objetos no encontrada:", objects_folder)
        return
    objects = load_objects_embeddings(objects_folder, model, preprocess, device)
    if not objects:
        print("No hay objetos en", objects_folder)
        return
    print("Objetos cargados:", list(objects.keys()))

    cap = open_camera(args.camera)
    if cap is None:
        return

    # Cargar YOLO
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.overrides['verbose'] = False

    # Ventana UI
    cv2.namedWindow("Object Finder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Finder", 900, 600)

    # Control de busqueda periodica
    search_interval = args.interval
    last_search_time = 0
    detected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        current_time = time.time()
        key = cv2.waitKey(1) & 0xFF
        do_search = False
        if current_time - last_search_time > search_interval:
            do_search = True
        if key == ord('s'):
            do_search = True

        if do_search:
            detected = detect_objects_yolo_clip(frame, objects, model, preprocess, device,
                                                threshold=args.threshold, yolo_model=yolo_model,
                                                save_crops=True)
            last_search_time = current_time

        # Dibujar resultados
        for name, x1, y1, x2, y2, sim in detected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"pos: {cx},{cy}", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Object Finder", frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-folder", default="objects",
                        help="Carpeta con subcarpetas de objetos")
    parser.add_argument("--camera", default="0",
                        help="Cámara a usar: número de índice o URL")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similitud mínima para detección CLIP")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Intervalo en segundos entre búsquedas CLIP")
    args = parser.parse_args()
    main(args)
