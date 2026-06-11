# Driver Monitoring AI

An AI-powered Driver Monitoring System (DMS) designed to detect driver drowsiness and distraction in real time using Computer Vision, MediaPipe Face Mesh, and Machine Learning.

## Overview

This project monitors a driver's facial behavior through a webcam and analyzes key indicators of fatigue and inattentiveness, including eye closure, yawning, and head pose movements.

The system extracts facial landmarks using MediaPipe Face Mesh, computes behavioral features such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Pitch, and Yaw, and then uses a trained Machine Learning model (Random Forest / XGBoost) to classify the driver's state.

When signs of drowsiness or distraction are detected, an alarm is triggered immediately to warn the driver.

## Key Features

### Eye Closure Detection

* Real-time eye tracking using facial landmarks.
* Computes Eye Aspect Ratio (EAR).
* Detects prolonged eye closure and microsleep events.

### Yawning Detection

* Monitors mouth opening using Mouth Aspect Ratio (MAR).
* Identifies frequent or extended yawning behavior.

### Head Pose Monitoring

* Estimates head orientation using facial landmarks and OpenCV Pose Estimation.
* Calculates:

  * Pitch (head up/down movement)
  * Yaw (head left/right movement)
* Detects distraction caused by looking away from the road.

### Machine Learning Classification

* Extracts behavioral features:

  * EAR
  * MAR
  * Pitch
  * Yaw
* Uses trained Machine Learning models to classify driver states:

  * Awake
  * Drowsy

### Real-Time Alert System

* Audio warning system using Pygame.
* Immediate alarm activation when dangerous behavior is detected.


## System Architecture

Webcam → Face Mesh Detection → Feature Extraction → Machine Learning Model → Driver State Prediction → Alarm System


## Technologies Used

### Computer Vision

* OpenCV
* MediaPipe Face Mesh

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* XGBoost Classifier

### Data Processing

* NumPy
* Pandas

### Alert System

* Pygame


## Extracted Features

| Feature | Description                                |
| ------- | ------------------------------------------ |
| EAR     | Eye Aspect Ratio for eye closure detection |
| MAR     | Mouth Aspect Ratio for yawning detection   |
| Pitch   | Head up/down angle                         |
| Yaw     | Head left/right angle                      |


## Driver States

The system classifies the driver's condition into:

### Awake

* Eyes open normally
* No excessive yawning
* Head facing forward

### Drowsy

* Eyes closed for an extended period
* Frequent yawning
* Head drooping downward

### Distracted

* Looking away from the road
* Significant head rotation


## Dataset Collection

The system supports real-time dataset generation.

Collected features:

* EAR
* MAR
* Pitch
* Yaw
* Driver state label

Data is automatically stored in CSV format for training and evaluation.

## Future Improvements

* Infrared camera support for nighttime driving
* Mobile deployment on Raspberry Pi or Jetson Nano
* Deep Learning models (CNN, LSTM, Transformer)
* Multi-driver adaptation
* Driver identity recognition
* Real vehicle integration

## Author

Developed as an AI-based Driver Drowsiness Detection and Monitoring Project using Computer Vision and Machine Learning.

