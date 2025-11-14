# 🎤 Speech-Based Stress Detection Using Deep Learning (Keras)

This repository contains a complete **speech-based stress detection system** built using a **Keras-only CNN–LSTM pipeline**.  
The model analyzes short audio recordings, extracts MFCC features, and predicts whether the speaker is **Stressed** or **Non-Stressed**.

The system includes:
- Full training notebook  
- Data preprocessing & augmentation pipeline  
- CNN–LSTM model in Keras  
- Evaluation metrics & visualizations  
- A real-time **Gradio interface** (`app.py`)  
- Saved trained model (`stress_detection_keras.h5`)

---

## 🚀 Project Overview

Stress significantly affects mental and physical health. Traditional assessment methods (surveys, physiological sensors) are often intrusive or impractical.  
Speech, however, naturally reflects emotional and physiological states.

This project demonstrates that **deep learning can detect stress directly from voice** using time–frequency features and hybrid neural architectures.

---

## 🧠 Model Architecture

The model uses:

- **MFCC extraction**  
- **CNN layers** for spectral feature extraction  
- **LSTM layer** for temporal modeling  
- **Dense layers** for binary classification
