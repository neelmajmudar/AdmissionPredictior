Project Overview
This project implements a machine learning model to predict a candidate's chance of admission to a graduate program based on various academic and profile parameters. The goal is to provide prospective students with an estimation of their admission likelihood, helping them make informed decisions about their applications.

The predictor leverages a dataset containing attributes commonly considered in university admissions, applying different machine learning algorithms to learn the patterns and make accurate predictions. This repository includes the data preprocessing steps, model training, evaluation, and inference.

Features
Data Preprocessing: Handles data cleaning, feature scaling, and preparation for model training.

Multiple Model Exploration: Implements and compares different machine learning algorithms (e.g., Logistic Regression, XGBoost) for prediction.

Model Training & Evaluation: Trains the chosen models on historical data and evaluates their performance using relevant metrics (e.g., accuracy, R-squared, MAE).

Prediction: Provides a mechanism to predict the "Chance of Admit" for new, unseen student profiles.

Jupyter Notebooks: Comprehensive notebooks detailing each step from data loading to model evaluation.

Dataset
The project utilizes the "Graduate Admissions" dataset, which contains various features influencing admission decisions.

Source: The dataset is typically available on platforms like Kaggle or UCI Machine Learning Repository. In this project, it is provided as Admission_Predict.csv.

Key Features (Columns):

GRE Score: Graduate Record Examination (GRE) score (out of 340)

TOEFL Score: Test of English as a Foreign Language (TOEFL) score (out of 120)

University Rating: Rating of the university (1 to 5)

SOP: Statement of Purpose strength (out of 5)

LOR: Letter of Recommendation strength (out of 5)

CGPA: Cumulative Grade Point Average (out of 10)

Research: Whether the applicant has research experience (0 or 1)

Chance of Admit: The target variable, representing the probability of admission (0 to 1)

Machine Learning Models
This project explores and implements several machine learning models to identify the best fit for the prediction task:

Logistic Regression: A linear model used for binary classification, but adaptable for regression tasks when predicting probabilities.

XGBoost (Extreme Gradient Boosting): A powerful and efficient open-source library that provides a gradient boosting framework for C++, Python, R, Java, and other languages. It's known for its speed and performance on structured data.

Other Models (Potential): Depending on future iterations, other models like Support Vector Machines (SVM), Random Forest, or Neural Networks could be integrated and compared.

Each model's implementation, training, and evaluation are detailed within their respective Jupyter notebooks.

App Demo: 

<img src='https://i.imgur.com/qseq0MH.gif' title='Video Demo' width='' alt='Video Demo' />
