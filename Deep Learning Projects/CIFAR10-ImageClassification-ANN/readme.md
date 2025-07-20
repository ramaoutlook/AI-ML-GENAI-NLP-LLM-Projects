# CIFAR-10 Image Classification with TensorFlow/Keras

This project demonstrates a simple image classification model built using TensorFlow and Keras to classify images from the CIFAR-10 dataset. I have developed this project in Google Colab with GPU activation

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The classes are:

*   airplane
*   automobile
*   bird
*   cat
*   deer
*   dog
*   frog
*   horse
*   ship
*   truck

## Project Steps

1.  **Load the dataset:** The CIFAR-10 dataset is loaded using `tf.keras.datasets.cifar10.load_data()`.
2.  **Data Exploration:** Basic exploration of the dataset is performed to understand the shape and content of the images and labels.
3.  **Data Preprocessing:** The image data is scaled by dividing by 255 to normalize the pixel values between 0 and 1. The labels are converted to categorical format using `keras.utils.to_categorical`.
4.  **Model Definition:** A simple sequential model is defined using Keras. The model consists of:
    *   A `Flatten` layer to flatten the 32x32x3 images into a 1D array.
    *   Two `Dense` layers with 3000 neurons and ReLU activation.
    *   A final `Dense` layer with 10 neurons (one for each class) and sigmoid activation.
5.  **Model Compilation:** The model is compiled with the SGD optimizer, categorical crossentropy loss function, and accuracy as the evaluation metric.
6.  **Model Training:** The model is trained on the scaled training data for 50 epochs.
7.  **Model Evaluation:** (This step was not explicitly shown in the provided code, but would typically follow training to evaluate the model's performance on the test set.)
8.  **Prediction:** The trained model is used to make predictions on the test set.

## Algorithms Used

*   **Deep Neural Network (DNN):** The core of the model is a simple feedforward deep neural network with multiple dense layers.
*   **Stochastic Gradient Descent (SGD):** This optimization algorithm is used to train the model by iteratively updating the model's weights based on the gradient of the loss function.
*   **Categorical Crossentropy:** This is the loss function used for multi-class classification problems like CIFAR-10. It measures the difference between the predicted class probabilities and the true class labels.
*   **ReLU Activation:** The Rectified Linear Unit activation function is used in the hidden layers to introduce non-linearity into the model.
*   **Sigmoid Activation:** The sigmoid activation function is used in the output layer to produce probability values for each class.

## How to Run

1.  Ensure you have TensorFlow and Keras installed.
2.  Run the code cells in the provided notebook sequentially.
3.  You need GPU Setup to run this project.
