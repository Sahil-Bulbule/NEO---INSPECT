# 🚀 Neo-Inspect

## Automated Surface Defect Detection using CNN Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-red)
![Accuracy](https://img.shields.io/badge/Accuracy-80%25-success)

## 📌 Project Overview

**Neo-Inspect** is a Deep Learning based automated surface defect inspection system developed using **Convolutional Neural Networks (CNN)**.

The main objective of this project is to automatically detect and classify industrial surface defects from images. The CNN model learns important visual features from defect images and predicts the category of the surface defect with high efficiency.

This project implements core **Computer Vision and Deep Learning concepts** including image preprocessing, CNN architecture, feature extraction, training, validation, and image classification.

---

# 🧠 Concepts Implemented

* Convolutional Neural Network (CNN)
* Image Classification
* Feature Extraction
* Image Preprocessing
* Data Augmentation
* Training & Validation Pipeline
* Deep Learning Model Deployment using Streamlit

---

# 🎯 Problem Statement

Manual quality inspection in industries is time-consuming and can lead to human errors.

Neo-Inspect provides an automated AI-based inspection solution that can identify different types of surface defects from images.

---

# 📂 Dataset Structure

```
NeoInspect
│
├── train
│   ├── crazing
│   ├── inclusion
│   ├── patches
│   ├── pitted_surface
│   ├── rolled_in_scale
│   └── scratches
│
├── validation
│   ├── crazing
│   ├── inclusion
│   ├── patches
│   ├── pitted_surface
│   ├── rolled_in_scale
│   └── scratches
│
├── visioninspect_cnn.keras
│
├── app.py
│
└── README.md
```

---

# 🔍 Defect Classes

The model can classify the following surface defects:

| Class           | Description               |
| --------------- | ------------------------- |
| Crazing         | Surface cracks pattern    |
| Inclusion       | Foreign material defects  |
| Patches         | Irregular surface patches |
| Pitted Surface  | Small holes or pits       |
| Rolled In Scale | Rolling process defects   |
| Scratches       | Surface scratches         |

---

# 🏗️ CNN Model Workflow

```
Input Image
      |
      ↓
Image Preprocessing
      |
      ↓
CNN Layers
      |
      ↓
Feature Extraction
      |
      ↓
Fully Connected Layers
      |
      ↓
Defect Classification
```

---

# 📊 Model Performance

## Accuracy Achieved

✅ **Training & Validation Accuracy: ~80%**

The CNN model successfully learned surface defect patterns and achieved reliable classification performance.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* CNN
* OpenCV
* NumPy
* Streamlit

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <https://github.com/Sahil-Bulbule/NEO---INSPECT>
```

Navigate to project folder:

```bash
cd NeoInspect
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

Start Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖥️ Application Features

✔ Upload surface defect image
✔ AI-based defect prediction
✔ CNN trained model integration
✔ Real-time classification
✔ User-friendly Streamlit interface

---

# 📸 Output

The system takes an input surface image and predicts the defect category:

Example:

```
Input:
Steel Surface Image

Prediction:
Scratches

Confidence:
High
```

---

# 📈 Future Improvements

* Increase dataset size
* Improve accuracy using Transfer Learning
* Implement advanced CNN architectures
* Add defect severity detection
* Deploy model on cloud platforms

---

# 👨‍💻 Author

**Sahil Jain**

Deep Learning | Computer Vision | CNN Projects

---

⭐ If you found this project useful, consider giving it a star!
