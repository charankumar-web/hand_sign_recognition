# Hand Sign Recognition System (1–9)

A real-time hand sign recognition system that detects and classifies hand gestures representing numbers **1 to 9** using OpenCV, MediaPipe, and TensorFlow.

## 🔍 Project Overview

This project uses computer vision and deep learning to recognize static hand gestures corresponding to numeric signs (1–9). The system leverages **MediaPipe** for hand tracking and a **custom-trained CNN model** for gesture classification.

## 🚀 Features

- Real-time hand tracking using MediaPipe.
- Finger gesture recognition for numbers 1 to 9.
- Lightweight and fast TensorFlow model (.tflite) for efficient performance.
- Webcam-based live gesture prediction with bounding box and class label.

## 📷 Demo

![Demo Screenshot](demo/demo.png)  
*Live prediction of hand signs from webcam feed.*

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / TensorFlow Lite**
- **NumPy**

## 📁 Project Structure

hand_sign_recognition/
│
├── model/ # Contains the trained TFLite model
│
├── utils/ # Utility scripts (e.g., FPS calculator, drawing)
│
├── dataset/ # Dataset used for training (optional)
│
├── hand_sign_recognition.py # Main script for running real-time prediction
│
├── keypoint_classifier.py # Custom classifier logic
│
├── README.md # Project documentation

## 🧠 How It Works

1. **MediaPipe** detects 21 keypoints of the hand in real-time.
2. The coordinates of these keypoints are extracted and preprocessed.
3. The processed data is passed to a **TensorFlow Lite** model trained to classify numeric hand signs (1–9).
4. The predicted class is overlaid on the live webcam feed.

## 🧪 Setup Instructions

### 🔧 Prerequisites

- Python 3.7+
- Install dependencies:
pip install opencv-python mediapipe tensorflow numpy

▶️ Run the Project
python app.py
Make sure your webcam is connected and accessible.

📦 Model Info
Input: 21 hand keypoints (42 values – x and y for each point)
Output: One of 9 classes (digits 1–9)
Format: .tflite model

📚 References
MediaPipe Hands
MNIST (for baseline ideas)

🙋‍♂️ Author
 P.Charan Kumar
🔗 GitHub Profile

🌐 Repository Link
https://github.com/charankumar-web/hand_sign_recognition
