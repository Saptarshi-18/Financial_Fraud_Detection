 
# Financial Fraud Detection Model Analysis

## Overview
This project focuses on detecting financial fraud using a variety of machine learning and deep learning models. The dataset used contains transactional data with features such as transaction type, amount, and balance changes. The goal is to develop effective models that accurately classify transactions as fraudulent or non-fraudulent.

Link to the dataset: [Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) as it exceeds github max limit

## Dataset
The dataset used in this project consists of transactional data containing the following features:
- Transaction type (e.g., debit, credit)
- Transaction amount
- Balance changes
- Other relevant transaction details

## Methodology
### Preprocessing
- Data cleaning: One-hot encoding and data normalization.
- Feature engineering: Dropping irrelevant features.

### Model Selection
The project explores both traditional machine learning models and deep learning architectures for classification:
#### Machine Learning Models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. Naive Bayes
6. XGBoost

#### Deep Learning Models:
1. Artificial Neural Network (ANN)
2. Convolutional Neural Network (CNN)

### Model Training and Evaluation
- Splitting the dataset into training and testing sets.
- Training each model using the training data.
- Evaluating model performance using accuracy, precision, recall, and F1-score metrics.
- Comparing the performance of different models to identify the most effective ones for fraud detection.

## Results
- Among the machine learning models, the Random Forest classifier demonstrated the best performance based on evaluation metrics.
- Within the deep learning models, the Convolutional Neural Network (CNN) showed promising results for fraud detection.
- Detailed model performance analysis and interpretation, including confusion matrices and ROC curves.


## Deployed Model
- The trained model has been deployed using Flask for inference. Access the deployment repository [here](https://github.com/StrangeCoder1729/FinancialFraudDetector).

 
 
