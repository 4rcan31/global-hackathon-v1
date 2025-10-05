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
# Utilidades CLIP
# -----------------------
def encode_clip(img, model, preprocess, device):
    img_input = preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

def match_clip(emb, objects, threshold=0.7):
    best_match = None
    best_sim = 0
    for name, emb_list in objects.items():
        for e in emb_list:
            sim = (emb @ e.T).item()
            if sim > threshold and sim > best_sim:
                best_match = (name, sim)
                best_sim = sim
    return best_match

# -----------------------
# Deteccion con crop completo
# -----------------------
def analyze_crop_full(crop, objects, model, preprocess, device, threshold):
    emb = encode_clip(crop, model, preprocess, device)
    match = match_clip(emb, objects, threshold)
    return match

# -----------------------
# Deteccion multi-scale
# -----------------------
def analyze_crop_multiscale(crop, objects, model, preprocess, device, threshold, scales=[128, 224, 256]):
    h, w, _ = crop.shape
    for scale in scales:
        if h < 20 or w < 20:
            continue
        resized = cv2.resize(crop, (scale, scale))
        match = analyze_crop_full(resized, objects, model, preprocess, device, threshold)
        if match:
            return match
    return None

# -----------------------
# Deteccion con tiles adaptativos
# -----------------------
def analyze_crop_tiles(crop, objects, model, preprocess, device, threshold, tile_size=(128,128), stride=64):
    detected = []
    h_crop, w_crop, _ = crop.shape
    adaptive_tile_size = (min(tile_size[0], w_crop), min(tile_size[1], h_crop))
    adaptive_stride = min(stride, adaptive_tile_size[0]//2, adaptive_tile_size[1]//2)

    for top in range(0, h_crop, adaptive_stride):
        for left in range(0, w_crop, adaptive_stride):
            bottom = min(top + adaptive_tile_size[1], h_crop)
            right = min(left + adaptive_tile_size[0], w_crop)
            tile = crop[top:bottom, left:right]
            if tile.size == 0 or tile.shape[0] < 20 or tile.shape[1] < 20:
                continue
            match = analyze_crop_full(tile, objects, model, preprocess, device, threshold)
            if match:
                name, sim = match
                detected.append((name, left, top, right, bottom, sim))
    return detected

# -----------------------
# Funcion principal de Deteccion
# -----------------------
def detect_objects_yolo_clip(frame, objects, model, preprocess, device,
                             threshold=0.7, yolo_model=None,
                             save_crops=True, tile_size=(128,128), stride=64):

    detected = []
    results = yolo_model(frame)[0]

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Azul YOLO

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        h_crop, w_crop, _ = crop.shape
        aspect_ratio = w_crop / h_crop

        # 1. Crop completo
        match = analyze_crop_full(crop, objects, model, preprocess, device, threshold)
        if match:
            name, sim = match
            detected.append((name, x1, y1, x2, y2, sim))
            if save_crops:
                save_dir = os.path.join("detections", name)
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(os.path.join(save_dir, f"full_{timestamp}.jpg"), crop)
            continue

        # 2. Decidir estrategia según tamaño y forma
        if max(h_crop, w_crop) < 150:
            continue  # Crop muy pequeño, no hacemos mas
        elif max(h_crop, w_crop) < 300 and 0.5 <= aspect_ratio <= 2:
            # Mediano y cuadrado-ish → multi-scale
            match = analyze_crop_multiscale(crop, objects, model, preprocess, device, threshold)
            if match:
                name, sim = match
                detected.append((name, x1, y1, x2, y2, sim))
                if save_crops:
                    save_dir = os.path.join("detections", name)
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(os.path.join(save_dir, f"multiscale_{timestamp}.jpg"), crop)
                continue
        else:
            # Crop grande o muy alargado → multi-scale primero, fallback tiles
            match = analyze_crop_multiscale(crop, objects, model, preprocess, device, threshold)
            if match:
                name, sim = match
                detected.append((name, x1, y1, x2, y2, sim))
                if save_crops:
                    save_dir = os.path.join("detections", name)
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(os.path.join(save_dir, f"multiscale_{timestamp}.jpg"), crop)
            else:
                # Tiles adaptativos como fallback
                tile_detections = analyze_crop_tiles(crop, objects, model, preprocess, device, threshold, tile_size, stride)
                for t in tile_detections:
                    name, left, top, right, bottom, sim = t
                    detected.append((name, x1+left, y1+top, x1+right, y1+bottom, sim))
                    if save_crops:
                        save_dir = os.path.join("detections", name)
                        os.makedirs(save_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(os.path.join(save_dir, f"tile_{timestamp}.jpg"),
                                    crop[top:bottom, left:right])

    return detected



# -----------------------
# Abrir camara
# -----------------------
def open_camera(cam_source):
    # Usar camara por defecto
    if not cam_source:
        cam_source = 0
    
    # es numero (camara local) o string (URL)
    if isinstance(cam_source, str) and cam_source.isdigit():
        cam_source = int(cam_source)
    
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

    yolo_model = YOLO("yolov8n.pt")
    yolo_model.overrides['verbose'] = False

    # Ventana UI
    cv2.namedWindow("Object Finder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Finder", 900, 600)

    # Control de busqueda periodica
    search_interval = args.interval
    last_search_time = 0
    debug_mode = True
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
        if key == ord('d'):
            debug_mode = not debug_mode
            print("Debug mode:", "ON" if debug_mode else "OFF", "(d)")

        if do_search:
            if debug_mode:
                cv2.putText(frame, "Buscando...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            detected = detect_objects_yolo_clip(frame, objects, model, preprocess, device,
                                                threshold=args.threshold, yolo_model=yolo_model,
                                                save_crops=True)
            last_search_time = current_time

        # Dibujar rectángulos YOLO si debug activado
        if debug_mode:
            results = yolo_model(frame)[0]
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # azul = YOLO

        # Dibujar resultados CLIP
        for name, x1, y1, x2, y2, sim in detected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"pos: {cx},{cy}", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Mostrar estado debug
        cv2.putText(frame, f"Debug: {'ON' if debug_mode else 'OFF'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Similitud mínima para Deteccion CLIP")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Intervalo en segundos entre búsquedas CLIP")
    args = parser.parse_args()
    main(args)
