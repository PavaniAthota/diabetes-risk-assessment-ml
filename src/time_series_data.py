import pandas as pd
import numpy as np

# Load original data
df = pd.read_csv(r"C:\Users\Pavani\Desktop\diabetes-risk-assessment\data\processed_data\diabetes_clean.csv")

# Create time steps (3 per patient)
time_steps = 3

data_list = []

for _, row in df.iterrows():
    for t in range(time_steps):
        new_row = row.copy()

        # simulate change over time
        new_row["Glucose"] = row["Glucose"] + np.random.randint(-10, 15)
        new_row["BMI"] = row["BMI"] + np.random.uniform(-1, 1)
        new_row["Age"] = row["Age"] + t

        new_row["time_step"] = t

        data_list.append(new_row)

ts_df = pd.DataFrame(data_list)

# Save
ts_df.to_csv(r"C:\Users\Pavani\Desktop\diabetes-risk-assessment\data\processed_data\diabetes_timeseries.csv", index=False)

print("Time-series dataset created")