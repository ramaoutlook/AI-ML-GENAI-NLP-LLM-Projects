# Breast Cancer Prediction

## Project Overview

This project focuses on building a machine learning model to predict whether a breast tumor is malignant or benign based on features extracted from digitized images of fine needle aspirate (FNA) of a breast mass. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Dataset

The dataset is the Breast Cancer Wisconsin (Diagnostic) Dataset, available through scikit-learn. It contains 569 observations and 30 features describing the characteristics of the cell nuclei. The target variable indicates whether the tumor is malignant (0) or benign (1).

## Methodology

1.  **Data Loading and Exploration:** The dataset is loaded using `sklearn.datasets.load_breast_cancer` and converted into a pandas DataFrame for easier manipulation. Initial data exploration includes checking the shape, information, descriptive statistics, and the distribution of the target variable.
2.  **Data Preprocessing:** The data is split into training and testing sets. The features are standardized using `StandardScaler` to ensure that each feature contributes equally to the model training.
3.  **Model Building:** A simple Sequential Neural Network model is built using TensorFlow and Keras. The model consists of:
    *   A Flatten layer to flatten the input data.
    *   A Dense layer with ReLU activation.
    *   A Dense output layer with Sigmoid activation for binary classification.
4.  **Model Compilation:** The model is compiled using the 'adam' optimizer and 'sparse_categorical_crossentropy' as the loss function, with 'accuracy' as the evaluation metric.
5.  **Model Training:** The model is trained on the standardized training data for a specified number of epochs. A validation split is used to monitor performance during training.
6.  **Model Evaluation:** The trained model is evaluated on the standardized test data to assess its performance on unseen data.
7.  **Predictive System:** A system is built to take new data as input, preprocess it, and use the trained model to make predictions.

## Results

The model achieved an accuracy of [Insert Accuracy Here] on the test data. The training and validation accuracy and loss curves are visualized to understand the training process.

## How to Run the Project

1.  Clone the repository.
2.  Ensure you have the necessary libraries installed (numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow). You can install them using pip:
