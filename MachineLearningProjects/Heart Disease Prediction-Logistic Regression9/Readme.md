Heart Disease Prediction Project

This project aims to build a predictive model to determine whether a person has heart disease based on various health indicators.
Dataset

The dataset used for this project is heart_disease_data.csv. It contains the following features:

    age: Age of the individual
    sex: Gender of the individual (1 = male; 0 = female)
    cp: Chest pain type (0-3)
    trestbps: Resting blood pressure (in mm Hg)
    chol: Serum cholestoral in mg/dl
    fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    restecg: Resting electrocardiographic results (0-2)
    thalach: Maximum heart rate achieved
    exang: Exercise induced angina (1 = yes; 0 = no)
    oldpeak: ST depression induced by exercise relative to rest
    slope: The slope of the peak exercise ST segment
    ca: Number of major vessels (0-3) colored by flourosopy
    thal: Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)
    target: Prediction target (0 = healthy; 1 = defective heart)

The dataset has 303 entries and 14 columns.
Model

A Logistic Regression model was used for classification. The data was split into training and testing sets with a test size of 0.2 and stratified sampling based on the target variable.
Results

The model achieved the following accuracy scores:

    Accuracy on Training Data: 85.12%
    Accuracy on Test Data: 81.97%

Predictive System

The trained model can be used to predict the presence of heart disease for a new data point. The input data should be a tuple of 13 features corresponding to the columns in the dataset (excluding the 'target' column). The model will output a prediction of 0 (healthy heart) or 1 (defective heart).
