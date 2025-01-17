🐔 Chicken Disease Detection Application

This project presents a state-of-the-art GUI-based application that uses deep learning to detect chicken diseases. The application allows users to upload chicken images for real-time detection and evaluates the model's performance on a test dataset.

📋 Table of Contents

🌟 Project Overview

📦 Dependencies

📚 Training Process

🚀 Usage

✨ Features

📊 Results

🌟 Project Overview

Chicken health is critical for poultry farming, and early disease detection can significantly improve outcomes. This project leverages a fine-tuned InceptionV3 model to classify chicken images as either Healthy or Unhealthy. With an intuitive graphical interface, this application simplifies the process of disease detection for poultry farmers and researchers.

📦 Dependencies

Before running the application, ensure the following libraries are installed:

numpy: For numerical computations.

tkinter: Built-in Python library for GUI.

Pillow: Image processing.

tensorflow: Deep learning framework.

scikit-learn: Evaluation metrics.

matplotlib: Data visualization.

Install dependencies with:

pip install numpy pillow tensorflow scikit-learn matplotlib

📚 Training Process

Data Preparation:

The dataset is organized into training and validation sets using the ImageDataGenerator class for preprocessing:

Training set: 80% of the data

Validation set: 20% of the data

Training Workflow:

Base Model:
The InceptionV3 model is used as a feature extractor with pre-trained weights from ImageNet.

Feature Extraction:

Extracted features from both the training and validation datasets using the base model.

Flattened the extracted features for use in the classifier.

Classifier:

A fully connected neural network with the following architecture:

Dense layer with 1024 units and ReLU activation.

Dropout layer with a 50% rate to reduce overfitting.

Output layer with a single unit and sigmoid activation.

Training:

The model was trained for 50 epochs using the Adam optimizer and binary cross-entropy loss.

Validation accuracy and loss were monitored during training.

Model Saving:

The trained model was saved to:

C:/Users/sanja/OneDrive/Desktop/poultry/path_to_save_model/inceptionv310.h5

🚀 Usage

Running the Application:

Place the trained model my_model_inceptionv310.h5 in the path_to_save_model/ directory.

Organize your test dataset within the testing/ directory.

Run the application with:

python app.py

Using the GUI:

Upload an Image: Click "Upload Image" to select a chicken image. The application processes the image and displays the detection result (Healthy/Unhealthy) along with the image preview.

Evaluate Model: Click "Evaluate Model" to calculate and display the model's accuracy on the test dataset in the terminal.

✨ Features

Real-Time Image Classification:

Upload an image and receive an immediate classification result.

Model Evaluation:

Evaluate the model on a labeled test dataset to compute accuracy.

Interactive GUI:

User-friendly interface designed with Tkinter.

📊 Results

Detection:

Outputs real-time predictions with a visual preview of the uploaded image.

Model Accuracy:

Provides detailed evaluation metrics for test performance.

Example Output:

Test Accuracy: 95.47%

Confusion Matrix:

Visualizes the classification performance.

🌟 Future Enhancements:

Add multi-class disease detection.

Implement batch image processing.

Incorporate cloud storage for large-scale datasets.

This project showcases the powerful intersection of machine learning and practical applications for the agricultural sector. 🐓

