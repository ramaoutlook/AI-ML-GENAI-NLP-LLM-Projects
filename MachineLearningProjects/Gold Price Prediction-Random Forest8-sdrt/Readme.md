# Gold Price Prediction ML Project

This project focuses on predicting gold prices using a Random Forest Regressor model based on historical data including the price of SPX, USO, SLV, and the EUR/USD exchange rate.

## Dataset

The dataset used in this project is sourced from Kaggle: [Gold Price Data](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data). It contains historical data of gold prices and related financial instruments.

## Project Steps

1.  **Import Libraries:** Import necessary libraries for data manipulation, visualization, and machine learning (pandas, numpy, matplotlib, seaborn, sklearn).
2.  **Data Collection and Processing:**
    *   Download the dataset from Kaggle using the provided API.
    *   Unzip the downloaded dataset.
    *   Load the data into a pandas DataFrame.
    *   Perform basic data exploration (check head, tail, shape, info, and missing values).
3.  **Data Analysis:**
    *   Calculate and visualize the correlation matrix of the features using a heatmap.
    *   Analyze the correlation of each feature with the target variable (GLD price).
    *   Visualize the distribution of the GLD price.
4.  **Splitting Features and Target:** Separate the features (independent variables) from the target variable (GLD price).
5.  **Splitting into Training and Test Data:** Split the dataset into training and testing sets to train and evaluate the model.
6.  **Model Training:** Train a Random Forest Regressor model on the training data.
7.  **Model Evaluation:**
    *   Make predictions on the test data.
    *   Evaluate the model's performance using metrics like R-squared error.
    *   Visualize the comparison between actual and predicted gold prices.

## Results

The Random Forest Regressor model achieved an R-squared error of approximately {{error_score}} on the test data, indicating a good fit to the data and strong predictive power.

The plot below shows the comparison between the actual gold prices and the prices predicted by the model on the test dataset.
