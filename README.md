Rainfall Prediction in Melbourne using Machine Learning

## Overview
This project predicts whether it will rain tomorrow in the Melbourne area using historical weather data. 
It demonstrates a complete machine learning workflow including data preprocessing, feature engineering, 
model training, hyperparameter tuning, and evaluation.

## Dataset
Australian weather dataset filtered to:
- Melbourne
- Melbourne Airport
- Watsonia

Target variable: **RainToday** (Yes / No)

## Feature Engineering
- Converted date into a seasonal feature (Summer, Autumn, Winter, Spring)
- One-hot encoded categorical variables
- Standardized numerical features
- Handled class imbalance

## Models Used
- Logistic Regression (baseline, interpretable)
- Random Forest Classifier (nonlinear, higher capacity)

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

Special attention was paid to **recall for rainy days**, as missing rain events is more costly than false alarms.

## Key Findings
- Random Forest achieved higher overall accuracy
- Logistic Regression offered better interpretability
- Dataset is moderately imbalanced; naive accuracy is misleading
- Feature correlation limits interpretation of feature importance

## Limitations
- Correlated features may inflate importance scores
- No temporal cross-validation
- Limited to Melbourne-area stations

## Future Improvements
- Use time-series validation
- Optimize explicitly for rain recall
- Try gradient boosting (XGBoost, LightGBM)

## Technologies
Python, Pandas, Scikit-learn, Matplotlib
