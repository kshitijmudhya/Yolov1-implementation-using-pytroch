
# 🦾 YOLOv1 with ResNet50 Backbone - Object Detection from Scratch

This project is a complete implementation of the YOLOv1 (You Only Look Once) object detection architecture using a ResNet50 backbone in PyTorch. It trains on the Pascal VOC dataset and supports real-time object detection using a webcam stream.

---

## 📁 Project Structure

```
.
├── backbone.py      # Feature extractor using pretrained ResNet50
├── model.py         # YOLO head with the ResNet50 backbone
├── loss.py          # Custom YOLOv1 loss function with IoU calculation
├── utils.py         # Image preprocessing and VOC label generation
├── train.py         # Training loop for the model
├── stream.py        # Real-time object detection via webcam
└── README.md        # This file
```

---

## 📚 Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy opencv-python pillow
```

Make sure you have access to a CUDA-compatible GPU and the Pascal VOC 2012 dataset.

---

## 🏗️ Model Overview

- **Backbone**: ResNet50 (up to final convolutional layers)
- **YOLO Head**:
  - Linear layer reducing ResNet output to 2048 features
  - Final layer projecting to a tensor of shape `(7, 7, 30)`
  - Outputs bounding boxes, object confidence, and class probabilities

---

## 🏋️‍♂️ Training (`train.py`)

Train the YOLOv1 model on Pascal VOC:

```bash
python train.py
```

- Uses a custom dataloader from `utils.py`
- Saves model checkpoints after each epoch (e.g., `yolo_pytorch_epoch3.pt`)
- Loss combines localization, confidence, and classification errors

📁 **Expected directory structure** for training:
```
VOC2012_train_val/
├── Annotations/
├── JPEGImages/
```

---

## 🧪 Real-Time Detection (`stream.py`)

Run webcam detection using the trained model:

```bash
python stream.py
```

- Captures live webcam frames
- Preprocesses and runs inference
- Draws predicted bounding box with class label
- Press `q` to quit

---

## 🧠 Label Format (VOC)

The model predicts:
- Two bounding boxes per cell
- Confidence scores
- 20 VOC classes:
  - aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, ...
  - See `utils.py` for full list

---

## ⚠️ Notes

- The model uses sigmoid activation on all outputs.
- Bounding box drawing selects the box with higher confidence.
- YOLOv1 divides the image into a `7x7` grid and outputs predictions per cell.

---

## 🧑‍💻 Author

A modular, educational implementation of YOLOv1, great for learning object detection from the ground up.
