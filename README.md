#Predicting Salary from LinkedIn Job Postings 2023
This project focuses on developing predictive models to estimate salaries based on various job listing features from LinkedIn job postings. Using data sourced from Kaggle, this project explores a variety of machine learning models, feature selection methods, and data preprocessing techniques to derive insights into the factors influencing salary levels in today's job market.

Table of Contents
Introduction
Data Description
Data Preprocessing
Model Building
Feature Selection
Hyperparameter Tuning
Additional Models
Results
Resources

Introduction
The main goal of this project is to utilize machine learning techniques to create a model that can predict salaries from job listings on LinkedIn. Various algorithms were tested, including linear regression, K-Nearest Neighbors (KNN), Random Forest, Support Vector Machines (SVM), XGBoost, and deep learning models. By exploring different models, preprocessing steps, and feature selection methods, the project aims to identify key salary determinants and provide stakeholders with a reliable tool for salary estimation.

Data Description
The dataset used in this project contains information on LinkedIn job postings, including:

Job Information: Job title, description, salary (min, max, median), location, work type, experience level, etc.
Company Details: Company size, location, description, employee count, follower count, etc.
Additional Information: Job benefits, application URLs, listing times, etc.
Data Preprocessing
The preprocessing steps include:

Data Collection: Gathering job posting data from Kaggle.
Data Cleaning: Handling missing values, outliers, and inconsistent patterns.
Data Integration: Merging multiple datasets to create a consolidated data source.
Data Reduction: Using techniques like Principal Component Analysis (PCA) to optimize the dataset for computational efficiency and enhance model performance.
Model Building
Various models were implemented to predict salaries based on job features:

Linear Regression
KNN Regressor
RandomForest Regressor
Linear SVM and Nonlinear SVM
Gradient Boosting
Model performances were evaluated using metrics such as Mean Squared Error (MSE) and R-squared.

Feature Selection
LASSO regression was employed to select the most impactful features for salary prediction. This method penalizes less significant features, promoting model sparsity and enhancing predictive accuracy. The final model's coefficients provide insights into the influence of various features on the salary.

Hyperparameter Tuning
Hyperparameter tuning was conducted using GridSearchCV to optimize the models. This exhaustive search across predefined hyperparameter values ensures that the best combination is selected to maximize model performance.

Additional Models
XGBoost: Used for its high efficiency and strong handling of missing data.
Extreme Machine Learning Model (ELM): A neural network model with quick training times, ideal for high-dimensional data.
Basic Deep Learning Model: A simple neural network with two layers to capture complex non-linear relationships in the data.
Ensemble Model: Combines the top three models (Random Forest, Gradient Boosting, and Deep Learning) for improved accuracy.

Results
The models' performances were evaluated, with Random Forest showing the lowest MSE, indicating superior accuracy in salary prediction. The final ensemble model provided the best metrics by combining the strengths of individual models.

Resources
Scikit-learn Documentation
GridSearchCV Documentation
LASSO Regression
