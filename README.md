# Customer Churn Prediction Project

## Overview

This project aims to build and optimize a model to predict customer churn. By analyzing customer data, the model helps businesses identify customers who are likely to leave, enabling them to take proactive measures to retain these customers.

## Table of Contents

1. [Project Objective](#project-objective)
2. [Data Preparation and Cleaning](#data-preparation-and-cleaning)
3. [Feature Engineering and Preprocessing](#feature-engineering-and-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Deployment](#model-deployment)
7. [File Structure](#file-structure)
8. [How to Run](#how-to-run)
9. [License](#license)

## Project Objective

The goal of this project is to develop a predictive model to determine if a customer will churn based on their attributes and behavior.

## Data Preparation and Cleaning

- **Data Loading**: Import the dataset from a CSV file into a DataFrame.
- **Handling Missing Values**: Convert non-numeric values to numeric where needed and fill missing values with appropriate statistics (e.g., median for numerical data).
- **Target Variable Encoding**: Convert categorical target values ('Yes'/'No') into binary format (1/0).

## Feature Engineering and Preprocessing

- **Feature Selection**: Identify and separate feature columns from the target variable.
- **Categorical vs. Numerical Features**:
  - **Categorical Features**: e.g., gender, internet service.
  - **Numerical Features**: e.g., tenure, monthly charges.
- **Preprocessing Pipelines**:
  - **Numerical Pipeline**: Includes steps for imputation and scaling.
  - **Categorical Pipeline**: Includes steps for imputation and one-hot encoding.

## Model Training and Evaluation

- **Data Splitting**: Divide data into training and test sets to evaluate model performance.
- **Model Training**: Train machine learning models such as Random Forest and Logistic Regression.
- **Model Evaluation**: Use metrics like accuracy, precision, recall, and classification report to assess model performance.

## Hyperparameter Tuning

- **Grid Search**: Find the optimal settings for model hyperparameters by evaluating combinations of parameters.

## Model Deployment

- **Interactive Web Application**: Use Streamlit to create a web interface for real-time predictions.
- **Label Encoding**: Save and load label encoders to ensure consistent encoding of categorical variables in the deployed application.


