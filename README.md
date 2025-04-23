# Object Detection with Faster R-CNN and RetinaNet (Detectron2)

This repository contains experiments and implementations for object detection using **YOLO**,**Faster R-CNN** and **RetinaNet**, both built on **Detectron2**. The dataset is prepared and preprocessed via a data cleaning pipeline, followed by model training and validation to detect objects in retail environments or structured product layouts.

---

## 🎯 Objective

The goal is to compare the performance of two leading object detection architectures on a cleaned dataset:
-  **YOLO (You Only Look Once)**- Fast, real-time single stage object detection (Adapted from Ultralytics)
-  **Faster R-CNN** – region proposal-based model, good for accuracy
-  **RetinaNet** – single-shot detector with Focal Loss for class imbalance

Additionally, we explore the impact of **data augmentation** on RetinaNet’s performance using Detectron2’s built-in transformations.

---

## 📁 Repository Structure

```
.
├── yolo/
│   ├── Data cleaning.ipynb                  # 🔧 Preprocess and clean the raw dataset
│   ├── RD2022_train_f.ipynb                 # 🧪 Utility script to visualize/train final dataset using YOLO
│   ├── RD2022_validation_f.ipynb            # 📊 Model evaluation and validation metrics
│
├── faster_rcnn/
│   └── 610 Faster R-CNN (1).ipynb           # 🐢 Training Faster R-CNN using Detectron2
│
├── retinanet/
│   ├── RetinaNet_Detectron2_Final.ipynb     # ⚡ RetinaNet training script with no augmentation
│   └── RetinaNet_Detectron2_Final_Augmentation.ipynb  # 🎨 RetinaNet training with augmentations
│
└── README.md                                # 📘 This file
```



---

##  Models Used

###  YOLO (You Only Look Once)
- Fast, real-time object detection
- Runs as a single-stage end-to-end model
- Adapted here via Ultralytics or custom PyTorch model
  
###  Faster R-CNN
- Two-stage detector
- Good for precision and small object localization
- Requires more training time

###  RetinaNet (Detectron2)
- One-stage detector with Focal Loss
- Handles class imbalance effectively
- Faster inference speed

---

##  Data Pipeline

1. **`Data cleaning.ipynb`**
   - Cleans bounding box coordinates
   - Handles missing labels or incorrect annotations
   - Converts data to COCO format (if needed)

2. **Augmentation**
   - Applied in `RetinaNet_Detectron2_Final_Augmentation.ipynb`
   - Includes random flip, resize, brightness adjustments, and more

---

##  Evaluation Metrics

- **Mean Average Precision (mAP)**
- **Precision-Recall curves**
- **Loss curves (classification + bbox regression)**

Each model’s results are visualized using COCO-style evaluation metrics and plotted with Detectron2’s inbuilt tools.

---

## 🚀 How to Run

### 1. Set up the environment
```bash
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python matplotlib seaborn
2. Open notebooks in Jupyter or VSCode and run:
Data cleaning.ipynb → preprocess data

610 Faster R-CNN (1).ipynb or RetinaNet_Detectron2_Final.ipynb  or RD2022_train_f.ipynb  → train models

RD2022_validation_f.ipynb → visualize and evaluate performance

