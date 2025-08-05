# Handwritten Digit Recognition using Neural Networks

This project implements a handwritten digit recognition system using a simple neural network built with TensorFlow and Keras. The model is trained on the widely used MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

## Project Overview

The goal of this project is to classify handwritten digits with high accuracy. The process involves loading the dataset, preprocessing the images, building and training a neural network model, evaluating its performance, and finally, creating a predictive system to recognize digits from new images.

## Techniques Used

*   **Neural Networks:** A multi-layer perceptron (MLP) is used for classifying the digits.
*   **TensorFlow and Keras:** These libraries are used for building, training, and evaluating the neural network model.
*   **MNIST Dataset:** The standard dataset for handwritten digit recognition is utilized.
*   **Image Preprocessing:**
    *   **Grayscale Conversion:** Color images are converted to grayscale.
    *   **Resizing:** Images are resized to a consistent 28x28 resolution.
    *   **Scaling:** Pixel values are scaled to a range between 0 and 1.
*   **Confusion Matrix:** A confusion matrix is used to visualize the performance of the model and identify misclassifications.
*   **Matplotlib and Seaborn:** These libraries are used for data visualization, including displaying images and the confusion matrix.
*   **OpenCV (cv2):** Used for image loading, grayscale conversion, and resizing.

## Implementation Steps

1.  **Load the Dataset:** The MNIST dataset is loaded using `keras.datasets.mnist.load_data()`.
2.  **Explore the Data:** The shape and type of the dataset are examined, and examples of the images and their corresponding labels are displayed.
3.  **Preprocess the Data:**
    *   The pixel values of the training and testing images are scaled to the range [0, 1] by dividing by 255.
4.  **Build the Neural Network Model:**
    *   A sequential Keras model is created with a `Flatten` layer to convert the 28x28 images into a 1D array.
    *   Two dense layers with ReLU activation functions are added as hidden layers.
    *   An output dense layer with a sigmoid activation function is used for the 10 classes (digits 0-9).
5.  **Compile the Model:** The model is compiled with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric.
6.  **Train the Model:** The model is trained on the scaled training data for a specified number of epochs.
7.  **Evaluate the Model:** The model's performance is evaluated on the scaled test data, and the loss and accuracy are reported.
8.  **Make Predictions:** The trained model is used to predict the digit for individual images from the test set.
9.  **Visualize Predictions:** The predicted labels are compared with the true labels, and a confusion matrix is generated and visualized using a heatmap to understand the model's performance across different classes.
10. **Build a Predictive System:** A system is created to take a new image as input, preprocess it, and predict the handwritten digit using the trained model.

## How to Run the Project

1.  Clone the repository to your local machine.
2.  Make sure you have the necessary libraries installed (TensorFlow, Keras, NumPy, Matplotlib, Seaborn, OpenCV). You can install them using pip:
