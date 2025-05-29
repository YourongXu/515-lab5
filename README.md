# TECHIN 515 – Lab 5: From Edge to Cloud

## 📋 Overview

This lab demonstrates a complete end-to-end pipeline from data collection on an ESP32 device to cloud-based model inference and deployment using Azure ML. The project is structured to simulate how IoT sensor data can be used to train machine learning models and integrate them into edge-cloud applications.

The lab is divided into the following major components:

- **ESP32_to_cloud**: Arduino sketch that captures motion data from an MPU6050 sensor and sends it to a local Flask server.
- **app**: Python-based Flask server that receives and processes the sensor data, then uses a pre-trained ML model (`wand_model.h5`) for real-time inference.
- **data**: Contains zipped sensor datasets used for training the gesture recognition model.
- **trainer_scripts**: Jupyter and Python scripts for training a Keras model and registering it with Azure ML.
- **README.md**: This file provides a comprehensive guide to the lab setup and contents.

## 🧠 ML Model Output

- A neural network model trained to classify gesture data into three classes: `O`, `V`, and `A`.
- The final trained model is saved as `wand_model.h5` and served by the Flask app.
- Model registration and versioning are handled via Azure ML using `register_model.ipynb`.

## 📁 Repository Structure

├── ESP32_to_cloud/
│ └── ESP32_to_cloud.ino # Arduino code for collecting sensor data
│
├── app/
│ ├── app.py # Flask server for receiving sensor data and serving model inference
│ ├── requirements.txt # Python dependencies
│ └── wand_model.h5 # Trained Keras model for inference
│
├── data/
│ ├── A.zip # Gesture data for class A
│ ├── O.zip # Gesture data for class O
│ ├── V.zip # Gesture data for class V
│ └── Some_class.zip # Optional/extra data
│
├── trainer_scripts/
│ ├── train.py # Script to train the ML model
│ └── register_model.ipynb # Azure ML registration notebook
│
└── README.md # Lab summary and usage instructions
