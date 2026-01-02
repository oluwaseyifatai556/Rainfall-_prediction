from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include='number').columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor

def train_random_forest(X_train, y_train):
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1_macro',
        cv=cv
    )

    grid.fit(X_train, y_train)
    return grid

def train_logistic_regression(X_train, y_train):
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
