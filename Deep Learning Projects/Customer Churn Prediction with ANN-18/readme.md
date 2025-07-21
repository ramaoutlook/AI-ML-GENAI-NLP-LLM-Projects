# Customer Churn Prediction using Deep Learning

This project demonstrates a end-to-end machine learning pipeline to predict customer churn based on various customer attributes and service usage data. The goal is to identify customers who are likely to churn so that the business can take proactive measures to retain them.

This project was developed as part of my learning journey in machine learning and deep learning, and it is uploaded to GitHub to showcase my skills and attract potential recruiters.

## Project Objective

The main objectives of this project are:

1.  **Data Loading and Exploration**: Load the customer churn data and perform initial data exploration to understand the structure and characteristics of the dataset.
2.  **Data Preprocessing**: Clean and preprocess the data for machine learning, including handling missing values, converting categorical features into numerical representations, and scaling numerical features.
3.  **Model Building**: Build a deep learning model using TensorFlow and Keras to predict customer churn.
4.  **Model Evaluation**: Evaluate the performance of the deep learning model using appropriate metrics like accuracy, precision, recall, and F1-score, and visualize the results with a confusion matrix.

## Data Source

The dataset used in this project is the `customer_churn.csv` file. It contains information about customers, including their demographics, services they use, contract details, and whether they have churned or not.

## Steps Taken

The following steps were performed in the analysis:

1.  **Loading Data**: The data was loaded into a pandas DataFrame.
2.  **Data Cleaning**:
    *   The 'customerID' column was dropped as it is not relevant for the prediction task.
    *   Missing values in the 'TotalCharges' column (represented by spaces) were identified and handled by removing the corresponding rows.
    *   The 'TotalCharges' column was converted from object type to numeric (float64).
3.  **Exploratory Data Analysis (EDA)**:
    *   Histograms were plotted to visualize the distribution of 'tenure' and 'MonthlyCharges' for churned and non-churned customers to understand their relationship with churn.
    *   Unique values in object type columns were printed to identify categorical features.
4.  **Feature Engineering and Encoding**:
    *   'No internet service' and 'No phone service' values in relevant columns were replaced with 'No' for consistency.
    *   Binary categorical columns (with 'Yes'/'No' or 'Female'/'Male') were converted to numerical representation (1/0).
    *   One-hot encoding was applied to multi-category nominal features like 'InternetService', 'Contract', and 'PaymentMethod' to convert them into numerical format.
5.  **Feature Scaling**: Numerical features ('tenure', 'MonthlyCharges', 'TotalCharges') were scaled using `MinMaxScaler` to bring them within a similar range, which is important for the performance of deep learning models.
6.  **Model Development**:
    *   The data was split into training and testing sets.
    *   A sequential deep learning model was built using TensorFlow Keras with a dense layer and a sigmoid output layer for binary classification.
    *   The model was compiled with the 'adam' optimizer and 'binary_crossentropy' loss function.
    *   The model was trained on the training data for 100 epochs.
7.  **Model Evaluation**:
    *   The model's performance was evaluated on the test set using `model.evaluate()`.
    *   Predictions were made on the test set.
    *   A classification report and a confusion matrix were generated to assess the model's precision, recall, F1-score, and accuracy.

## Results

The model achieved an accuracy of [Insert Accuracy from model.evaluate() output] on the test set. The classification report and confusion matrix provide further insights into the model's performance in predicting churn.

[You can elaborate on the precision, recall, and F1-score for each class (Churn=Yes and Churn=No) based on the classification report and discuss what these metrics indicate in the context of churn prediction.]

## Technologies Used

*   Python
*   Pandas (for data manipulation and analysis)
*   NumPy (for numerical operations)
*   Matplotlib and Seaborn (for data visualization)
*   Scikit-learn (for data splitting and preprocessing)
*   TensorFlow and Keras (for building and training the deep learning model)

## How to Run the Project

1.  Clone this repository to your local machine.
2.  Make sure you have the required libraries installed (you can use `pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`).
3.  Download the `customer_churn.csv` dataset and place it in the appropriate directory (or update the file path in the code).
4.  Run the Jupyter Notebook or Python script (`your_notebook_name.ipynb` or `your_script_name.py`) to execute the code.

## Future Improvements

*   Experiment with different deep learning model architectures (e.g., adding more layers, changing the number of neurons).
*   Tune hyperparameters of the model and optimizer to improve performance.
*   Explore other feature engineering techniques.
*   Investigate more advanced techniques for handling imbalanced datasets (if applicable).
*   Consider deploying the model for real-time predictions.
