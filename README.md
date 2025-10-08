CIFAR-10 Image Classification with CNN
Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Using a combination of data augmentation, batch normalization, dropout, and adaptive learning rates, the model achieves robust performance and generalizes well to unseen images.

Test Accuracy: 86.71%

This project demonstrates how CNNs can effectively extract hierarchical features from small RGB images, making it a solid foundation for exploring deeper architectures like ResNet, VGG, or custom CNN models.

Key Features

Deep CNN architecture with 3 convolutional blocks and fully connected layers

Batch Normalization for stable and faster training

Dropout layers to prevent overfitting

Data Augmentation to increase dataset variability and robustness

Adaptive Learning Rate Scheduler for efficient optimization

Dataset

CIFAR-10: 60,000 color images (32×32) in 10 classes

50,000 training images

10,000 test images

Dataset is automatically loaded using tensorflow.keras.datasets.

Tech Stack

Python 3.x

TensorFlow & Keras

NumPy, Matplotlib

Project Structure
cnn_cifar10/
│
├── cnn_cifar10_model.h5       # Trained CNN model
├── cifar10_cnn.ipynb          # Jupyter/Colab notebook with full training code
├── README.md                  # Project description
└── requirements.txt           # Python dependencies

How It Works

Load and normalize CIFAR-10 images

Apply data augmentation (rotation, shifts, flips) to training data

Build the CNN:

Convolution + BatchNorm + MaxPooling + Dropout blocks

Flatten layer

Dense layers with ReLU + Dropout

Softmax output layer for classification

Compile using Adam optimizer and categorical crossentropy loss

Train the model for 40 epochs with validation monitoring

Evaluate on the test set → achieve 80.21% accuracy

Save the trained model for future inference
