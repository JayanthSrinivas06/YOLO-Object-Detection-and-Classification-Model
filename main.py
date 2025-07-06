# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from ultralytics import YOLO
import numpy as np
import os

# ---------------------- FastAPI Initialization ----------------------
app = FastAPI()

# ---------------------- CORS Configuration ----------------------
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "file://",  # Allows access when index.html is opened directly from file system
    "null"      # Chrome uses 'null' for file:// origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Model Loading ----------------------
models_dir = "models"
detection_model_path = os.path.join(models_dir, "detection_best.pt")
classification_model_path = os.path.join(models_dir, "classification_best.pt")

detection_model = None
classification_model = None

try:
    if os.path.exists(detection_model_path):
        detection_model = YOLO(detection_model_path)
        print(f"Detection model loaded from {detection_model_path}")
    else:
        print(f"Warning: Detection model not found at {detection_model_path}.")

    if os.path.exists(classification_model_path):
        classification_model = YOLO(classification_model_path)
        print(f"Classification model loaded from {classification_model_path}")
    else:
        print(f"Warning: Classification model not found at {classification_model_path}.")

    if detection_model and classification_model:
        print("All models loaded successfully!")
    else:
        print("Some models could not be loaded. Check paths.")

except Exception as e:
    print(f"Error loading models: {e}")

# ---------------------- Utility Functions ----------------------
def read_image_from_bytes(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

def encode_image_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def enhance_grayscale_image(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(0.5)
    img = ImageEnhance.Sharpness(img).enhance(3.0)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    return img

# ---------------------- Detection Endpoint ----------------------
@app.post("/detect-device/")
async def detect_device(file: UploadFile = File(...)):
    if detection_model is None:
        raise HTTPException(status_code=500, detail="Detection model not loaded.")

    image_bytes = await file.read()
    original_image = read_image_from_bytes(image_bytes)

    results = detection_model.predict(source=original_image, save=False, imgsz=640, conf=0.25, iou=0.7)

    if not results or not results[0].boxes or len(results[0].boxes) == 0:
        return {
            "label": "no_device_detected",
            "bbox": [],
            "cropped_image_base64": ""
        }

    best_box = results[0].boxes[0]
    xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
    conf = best_box.conf[0].item()
    cls_id = best_box.cls[0].item()

    label = detection_model.names[int(cls_id)] if 0 <= int(cls_id) < len(detection_model.names) else "unknown_device"

    xmin, ymin, xmax, ymax = xyxy.tolist()
    img_width, img_height = original_image.size
    xmin_clamped = max(0, xmin)
    ymin_clamped = max(0, ymin)
    xmax_clamped = min(img_width, xmax)
    ymax_clamped = min(img_height, ymax)

    if xmax_clamped <= xmin_clamped or ymax_clamped <= ymin_clamped:
        print(f"Warning: Invalid crop coordinates for {label}: {xmin}, {ymin}, {xmax}, {ymax}")
        cropped_image_base64 = ""
    else:
        cropped_image = original_image.crop((xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped))
        cropped_image_base64 = encode_image_to_base64(cropped_image)

    return {
        "label": label,
        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
        "cropped_image_base64": cropped_image_base64
    }

# ---------------------- Classification Endpoint ----------------------
@app.post("/classify-device/")
async def classify_device(file: UploadFile = File(...)):
    if classification_model is None:
        raise HTTPException(status_code=500, detail="Classification model not loaded.")

    image_bytes = await file.read()
    input_image = read_image_from_bytes(image_bytes)

    np_img = np.array(input_image)
    avg_pixel = np.mean(np_img)

    gray_image = input_image.convert("L")
    gray_array = np.array(gray_image)
    black_pixel_ratio = np.sum(gray_array < 50) / gray_array.size
    print(black_pixel_ratio)

    if avg_pixel < 150 or black_pixel_ratio > 0.05:
        input_image = input_image.convert("L")
        input_image = enhance_grayscale_image(input_image)

    if black_pixel_ratio > 0.40:
        input_image = input_image.rotate(200, expand=True)

    results = classification_model.predict(source=input_image, save=False, imgsz=224)

    if not results or not results[0].probs:
        raise HTTPException(status_code=500, detail="Classification prediction failed.")

    probs = results[0].probs
    top1_idx = probs.top1
    confidence = probs.top1conf.item()
    print(results[0].names, probs)

    label = classification_model.names[top1_idx] if 0 <= top1_idx < len(classification_model.names) else "unknown_class"

    return {
        "label": label,
        "confidence": confidence
    }
