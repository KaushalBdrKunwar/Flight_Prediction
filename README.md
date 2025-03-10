# Flight Price Prediction:

This project aims to predict flight prices using a dataset containing various features related to flights. The project involves data cleaning, exploratory data analysis, feature engineering, and machine learning model building. The final model is a RandomForestRegressor, which is used to predict flight prices based on the given features.

# Table of Contents
1.Introduction

2.Dataset

3.Data Cleaning

4.Exploratory Data Analysis (EDA)

5.Feature Engineering

6.Model Building

7.Model Evaluation

8.Model Deployment

9.Conclusion

10.Future Work

# Introduction
Predicting flight prices is a common problem in the travel industry. This project uses a dataset containing various features such as airline, source, destination, departure time, arrival time, duration, and more. The goal is to build a machine learning model that can accurately predict flight prices based on these features.

# Dataset
The dataset used in this project is flightdata.csv, which contains the following columns:

Airline: The name of the airline.

Date_of_Journey: The date of the flight.

Source: The source city of the flight.

Destination: The destination city of the flight.

Route: The route taken by the flight.

Dep_Time: The departure time of the flight.

Arrival_Time: The arrival time of the flight.

Duration: The duration of the flight.

Total_Stops: The total number of stops in the flight.

Additional_Info: Additional information about the flight.

Price: The price of the flight (target variable).

# Data Cleaning
The data cleaning process involves handling missing values, converting data types, and extracting useful features from the existing columns. The following steps were taken:

Handling missing values by dropping rows with missing data.

Converting date and time columns to datetime format.

Extracting day, month, and year from the Date_of_Journey column.

Extracting hour and minute from the Dep_Time and Arrival_Time columns.

Preprocessing the Duration column to extract hours and minutes.

# Exploratory Data Analysis (EDA)
The EDA process involves analyzing the data to understand the distribution of flight prices, the relationship between different features, and identifying any outliers. The following analyses were performed:

Analyzing the distribution of flight prices.

Analyzing the relationship between flight duration and price.

Analyzing the most common departure times.

Visualizing the distribution of flight prices using histograms and box plots.

# Feature Engineering
Feature engineering involves creating new features from the existing data to improve the performance of the machine learning model. The following steps were taken:

Encoding categorical variables using One-Hot Encoding and Target Guided Encoding.

Removing irrelevant columns such as Additional_Info, Route, and Date_of_Journey.

Handling outliers in the Price column using the Interquartile Range (IQR) method.

# Model Building
The machine learning model used in this project is a RandomForestRegressor. The following steps were taken to build and evaluate the model:

Splitting the data into training and testing sets.

Training the RandomForestRegressor on the training data.

Making predictions on the test data.

Evaluating the model using metrics such as R2 score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

# Model Evaluation
The model was evaluated using the following metrics:

R2 Score: 0.81

MAE: 1176.93

MSE: 3705315.86

RMSE: 1924.92

MAPE: 13.17%

The model achieved an R2 score of 0.81, indicating that it explains 81% of the variance in the flight prices.

# Model Deployment
The final model was saved using the pickle library for future use. The model can be loaded and used to make predictions on new data.

python
Copy
import pickle

Save the model
file = open('rf_random.pkl', 'wb')
pickle.dump(ml_model, file)

Load the model
model = open('rf_random.pkl', 'rb')
forest = pickle.load(model)

Make predictions
y_pred2 = forest.predict(X_test)

# Conclusion
This project successfully built a machine learning model to predict flight prices with an R2 score of 0.81. The model can be further improved by tuning hyperparameters and exploring other machine learning algorithms.

# Future Work
Hyperparameter Tuning: Use techniques like RandomizedSearchCV or GridSearchCV to find the best hyperparameters for the RandomForestRegressor.

Feature Selection: Explore other feature selection techniques to improve model performance.

Other Algorithms: Experiment with other regression algorithms such as Gradient Boosting, XGBoost, or Neural Networks.

Deployment: Deploy the model as a web application or API for real-time predictions.
