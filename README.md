
# 🎨 Sketch2Rig: Automating Houdini Facial Controls from 2D Expressions using Machine Learning

This project uses a PyTorch-based supervised learning model to **automatically infer facial rig slider values in Houdini** based on hand-drawn 2D expressions.

---

## 🔍 Overview

The pipeline is built to convert simple 2D sketch drawings of eyes and eyebrows into a 3D facial rig pose in Houdini using a trained regression model.

![Demo](Sketch2RigML_06-20-2025.gif)

---

## 🧠 Machine Learning Model

### Training
- `expression_train_01.py`: Trains a CNN model to regress 18 facial rig parameters from grayscale eye+eyebrow sketch images.
- The model is trained on custom sketch samples, labeled with corresponding Houdini slider values stored in a `.csv`.

### Inference
- `expression_infer_01.py`: Loads the trained model and performs inference on new sketch images located in the `inference/` folder.
- Outputs predictions as both console prints and a `.csv` file (`inferred_results.csv`).

---

## 📎 Houdini Integration

- `expresssion_insert_to_houdini_pythonshell_01.py`: A Python script to be pasted into Houdini's Python Shell.
- Reads `inferred_results.csv` and updates the `Eye_UI` controller node with predicted values.

---

## 🗂 Folder Structure

```
📁 inference/             # Contains test sketches for inference
📁 images/                # Training images (e.g., happy, sad, angry expressions)
📄 labels_01.csv          # CSV file with rig slider values
📄 inferred_results.csv   # Output of ML model during inference
```

---

## 🖼️ Visuals

![Sketch2Rig Demo](Documentation/Sketch2RigML_06-27-2025.gif)

---

## 🚀 Getting Started

1. Run `expression_train_01.py` to train the model on your labeled sketch dataset.
2. Add test sketches to `inference/` and run `expression_infer_01.py`.
3. Paste the Houdini insert script into the Python Shell to update the rig.

---
