from sklearn.model_selection import train_test_split
from data_loader import load_and_clean_data
from feature_engineering import add_season_feature
from models import train_random_forest, train_logistic_regression
from evaluation import evaluate_model

def main():
    df = load_and_clean_data()
    df = add_season_feature(df)

    X = df.drop(columns='RainToday')
    y = df['RainToday']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)

if __name__ == "__main__":
    main()
