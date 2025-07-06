# YOLO Object Detection and Classification Model

This project presents a dual-model system using **YOLOv8** and **YOLOv8-cls** to accurately detect and classify highly similar-looking objects. It is built using **FastAPI** and trained on 5 visually identical **medical device** classes as a demonstration.

> ⚠️ While the example focuses on medical devices, the architecture is **flexible** and applicable to any domain where objects are visually similar, but need to modify some part of .ipynb file (e.g., industrial tools, food items, product variants).

---

## 📸 Dataset Preparation & Preprocessing

### 🗂️ 1. Data Collection
Begin by collecting a raw dataset of the objects you want to detect and classify. This project was demonstrated using 5 visually similar medical device classes, but the architecture is adaptable to any object category.

### 🏷️ 2. Data Annotation (Object Detection Labels) and Data Argumentation 
Use [Roboflow](https://roboflow.com/) or any other annotation tool to label the collected images with bounding boxes for each object class. This creates the detection dataset needed to train the YOLOv8 object detection model.

### 💾 3. Exporting & Organizing Annotations
Export the annotated dataset in YOLO format and place it into a folder named:

```
annotate_data/
```

This will serve as the source for both the detection and classification datasets.

### 🔁 4. Dataset Preprocessing for Multi-Stage Inference
Run the notebook `medical_device.ipynb` to process the annotations and generate two separate datasets:

- `detect_dataset/` – for training the **YOLOv8 object detector**.
- `classify_dataset/` – for training the **YOLOv8-cls classifier**, using cropped object images and their respective labels.

I used 5 medical device images: Weighting scale, Glucose meter, SPO2, BP Monitor, HBA1C (Digital devices) used dataset of 2000 images and made 5000 images after Data Argumentation.

---

## 🧠 Two-Stage Model Architecture

### 🎯 Stage 1 – Object Detection (YOLOv8)
A YOLOv8 detection model is trained on `detect_dataset/` to localize objects and generate bounding boxes from input images. It is responsible for identifying *where* the objects are in the image.

### 🧪 Stage 2 – Fine-Grained Image Classification (YOLOv8-cls)
Each detected object is cropped and passed to a second YOLOv8-cls model trained on `classify_dataset/`, which performs high-accuracy classification of visually similar items. This classifier determines *what* each detected object is, based on subtle features.

This two-stage design is highly effective for scenarios where object classes share similar shapes, textures, or colors, but require nuanced differentiation.

---

## 🖼️ Workflow Summary

1. Input image is uploaded.
2. **YOLOv8** detects objects.
3. Detected regions are cropped.
4. **YOLOv8-cls** classifies each cropped region.
5. Final image with bounding boxes and labels is returned.

---

## ✅ Use Cases

- Differentiating between **similar medical tools**
- **Product variant** recognition in retail/inventory
- Classifying **identical industrial parts**
- Any visual task requiring **fine-grained recognition**

---

## 📦 Requirements

```txt
fastapi
uvicorn
ultralytics
pillow
numpy
python-multipart
```

---

## 👨‍💻 Author

**Jayanth Srinivas Bommisetty**  
Python Developer | AI/ML | Deep Learning | Computer Vision  
[GitHub](https://github.com/JayanthSrinivas06) | [LinkedIn](https://www.linkedin.com/in/jayanth-srinivas-b-0b7911269/)
