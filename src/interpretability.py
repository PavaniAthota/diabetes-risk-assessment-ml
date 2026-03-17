import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# Load trained pipeline
model = pickle.load(open("models/diabetes_model.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/processed_data/diabetes_clean.csv")

X = df.drop("Outcome", axis=1)

print("Shape of X:", X.shape)
print("Features:", X.columns)

# Use SHAP Explainer on full pipeline
explainer = shap.Explainer(model.predict, X)

# Compute SHAP values
shap_values = explainer(X)

# Beeswarm plot
plt.figure(figsize=(10,6))
plt.subplots_adjust(left=0.35)  # add space for feature names
shap.plots.beeswarm(shap_values, show=False)

# Feature importance bar plot
plt.figure(figsize=(8,5))
plt.subplots_adjust(left=0.35)
shap.plots.bar(shap_values, show=False)

plt.show()