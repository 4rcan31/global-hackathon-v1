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

    # ==============================================================
    # CARGAR CLIP
    # ==============================================================

    # CLIP (Contrastive Language–Image Pretraining)
    # ----------------------------------------------
    # clip.load(model_name, device) carga un modelo preentrenado de CLIP.
    # Devuelve una tupla: (modelo, transformador_preprocesamiento)
    # - `model`: es el modelo de CLIP (permite obtener embeddings de texto e imagen).
    # - `preprocess`: es la función de transformación de imágenes compatible con el modelo.

    # Parámetros:
    #   model_name: nombre del modelo, por ejemplo "ViT-B/32".
    #   device: "cpu" o "cuda" (si hay GPU compatible con CUDA).
    #
    # Ejemplo:
    #   model, preprocess = clip.load("ViT-B/32", device=device)

    # Modelos disponibles y diferencias:
    # ----------------------------------
    # 1. ViT-B/32
    #     - Base (más ligero y rápido)
    #     - Resolución: 224x224
    #     -  Visual Transformer con patch size de 32
    #     - Recomendado para CPU o GPU modesta (poca VRAM)
    #
    # 2. ViT-B/16
    #     - Misma arquitectura base pero con patch size más pequeño (16)
    #     - Más precisión en los embeddings (mejor calidad)
    #     - Requiere más VRAM (~2x ViT-B/32)
    #
    # 3. ViT-L/14
    #     - Modelo Large, mucho más preciso
    #     - Recomendado solo si tienes GPU con >6GB VRAM
    #
    # 4. ViT-L/14@336px
    #     - Versión de mayor resolución (336x336)
    #     - Aún más exacta para detalles finos
    #     - Requiere bastante VRAM (>8GB)
    #
    # 5. RN50 / RN101 / RN50x4 / RN50x16 / RN50x64
    #     - Basados en ResNet (no Vision Transformer)
    #     - Menos eficientes que los ViT en tareas semánticas
    #     - Útiles si quieres compatibilidad más amplia o si el entorno no soporta ViT

    #  Nota: Para cada uno se necesita una descarga diferente

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

    # ==============================================================
    # CARGAR YOLO
    # ==============================================================

    # YOLOv8 (Ultralytics)
    # ----------------------------------------------
    # Carga un modelo de detección de objetos preentrenado.
    # La clase YOLO permite:
    #   - Detectar objetos
    #   - Entrenar nuevos modelos
    #   - Exportar modelos a otros formatos (ONNX, TensorRT, etc.)
    #
    # Ejemplo:
    #   yolo_model = YOLO("yolov8n.pt")

    # Modelos disponibles y diferencias:
    # ----------------------------------
    # Los sufijos representan el tamaño del modelo:
    #   n  = nano     (más rápido, menos preciso)
    #   s  = small    (rápido, buena precisión)
    #   m  = medium   (equilibrado)
    #   l  = large    (más preciso, más pesado)
    #   x  = xlarge   (máxima precisión, mucho consumo)
    #
    # Detalle de cada modelo (YOLOv8):
    # 1. yolov8n.pt  →  3-4 MB  |  velocidad máxima  |  baja precisión
    # 2. yolov8s.pt  →  ~10 MB  |  muy rápido        |  buena precisión
    # 3. yolov8m.pt  →  ~25 MB  |  balanceado        |  precisión media-alta
    # 4. yolov8l.pt  →  ~45 MB  |  más lento         |  alta precisión
    # 5. yolov8x.pt  →  ~90 MB  |  muy lento         |  máxima precisión
    #
    # Requisitos estimados:
    # - n/s → puede correr en CPU o GPU de gama baja (2GB VRAM)
    # - m/l/x → se recomienda GPU NVIDIA >= 4GB VRAM
    #
    #  Camara en tiempo real, (empezar) yolov8n o yolov8s.
    #  Para procesamiento por lotes o imágenes de alta resolución, usa m o l.
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
