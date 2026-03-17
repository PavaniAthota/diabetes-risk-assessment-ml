import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load processed dataset
df = pd.read_csv("data/processed_data/diabetes_clean.csv")

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline (scaling + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# Hyperparameter grid
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_split": [2, 5, 10]
}

# Grid search
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, prob))

# Save model
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to models/diabetes_model.pkl")