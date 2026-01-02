# Rainfall-_prediction
Rainfall Prediction Using Machine Learning
Project Overview
This project predicts whether it will rain tomorrow using historical weather data
from Australian weather stations. The task is treated as a binary classification
problem.

Models Used
- Logistic Regression
- Random Forest Classifier

Techniques Applied
- Data cleaning and preprocessing
- One-hot encoding for categorical features
- Feature scaling
- Pipeline and ColumnTransformer
- Stratified K-Fold cross-validation
- Hyperparameter tuning using GridSearchCV

Evaluation Metrics
- Accuracy
- Precision
- Recall (True Positive Rate)
- Confusion Matrix

Key Findings
- Random Forest achieved higher overall accuracy
- Logistic Regression had comparable recall for rainfall detection
- Dataset was imbalanced, with more non-rain days
