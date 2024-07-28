# Real-time-Traffic-Sign-Recognition-for-Autonomous-Vehicles

### Project Overview
Traffic sign recognition is a crucial component for autonomous vehicles, helping them navigate safely by recognizing and responding to traffic signs. This project involves developing a real-time traffic sign recognition system using deep learning.

### Project Goals
    Data Collection and Preprocessing: Gather and preprocess a dataset of traffic sign images.
    Model Development: Create a convolutional neural network (CNN) to classify traffic signs.
    Model Training: Train the model on the labeled dataset of traffic sign images.
    Model Evaluation: Evaluate the model's performance using appropriate metrics.
    Deployment: Develop a real-time recognition system that processes video feeds from a car's camera.

### Steps for Implementation
1. Data Collection
Use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of labeled traffic sign images.

2. Data Preprocessing
    ###### Normalization: Normalize pixel values to a range of 0 to 1.
    ###### Resizing: Resize images to a consistent size (e.g., 32x32).
3. Model Development
Develop a CNN using TensorFlow and Keras.

4. Model Training
Split the dataset into training and validation sets, then train the model.

5. Model Evaluation
Evaluate the model using metrics like accuracy, precision, recall, and F1 score.

6. Deployment
Deploy the model using OpenCV to process real-time video feeds.

