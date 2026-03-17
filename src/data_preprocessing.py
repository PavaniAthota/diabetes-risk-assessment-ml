import pandas as pd

# Load dataset
df = pd.read_csv("data/raw_data/diabetes.csv")

print("Dataset shape:", df.shape)

# Replace invalid zeros with missing values
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)

print("\nMissing values before imputation:")
print(df.isnull().sum())

# Fill missing values using median
df.fillna(df.median(), inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/processed_data/diabetes_clean.csv", index=False)

print("\nClean dataset saved to data/processed_data/")