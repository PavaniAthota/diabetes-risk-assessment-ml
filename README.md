# Diabetes Risk Assessment using Machine Learning

This project presents a machine learning system for predicting diabetes risk using clinical health data. It also includes an interactive dashboard for visualization and basic decision support.

---

## Overview

The goal of this project is to build a predictive system that can estimate the likelihood of diabetes while also providing simple and understandable insights.

The system combines:
- Machine learning for prediction  
- Basic clinical rules for interpretation  
- An interactive dashboard for visualization  

Key capabilities include:
- Predicting diabetes risk from patient data  
- Providing clinically meaningful interpretation  
- Visualizing results through a dashboard  
- Simulating risk progression using time-series data  

---

## Methodology

### Data Preprocessing

- Cleaned and structured the dataset  
- Handled missing and zero values  
- Prepared features for model training  

---

### Machine Learning Model

- Algorithm: Random Forest Classifier  
- Input features:
  - Glucose  
  - BMI  
  - Age  
  - Blood Pressure  
- Output: Probability of diabetes  

The model was selected because it provides a good balance between performance and interpretability.

---

### Model Persistence

- Model saved using `pickle`  
- File location: `models/diabetes_model.pkl`  
- Allows reuse without retraining  

---

## Interactive Dashboard (Streamlit)

The dashboard allows users to interact with the model in real time.

### Prediction

- Users can input patient data  
- The system returns a diabetes risk prediction  

---

### Visualization

- Donut chart for class distribution  
- Glucose vs Age trend  
- Risk gauge visualization  

---

### Clinical Interpretation

- Glucose thresholds:
  - ≥ 200 → High Risk  
  - 140–199 → Prediabetic  

A simple hybrid approach is used by combining model predictions with clinical rules.

---

### Recommendations

- Provides basic lifestyle suggestions based on predicted risk  

---

## Time-Series Analysis

A simulated time-series component is included to explore how risk may change over time.

- Tracks glucose variation  
- Estimates risk progression  
- Visualized using line plots  

This is intended to resemble basic longitudinal monitoring.

---

## Project Structure

diabetes-risk-assessment/

├── data/
│ ├── raw_data/
│ │ └── diabetes.csv
│ ├── processed_data/
│ │ ├── diabetes_clean.csv
│ │ └── diabetes_timeseries.csv

├── models/
│ └── diabetes_model.pkl

├── notebooks/
│ └── exploratory_analysis.ipynb

├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── interpretability.py
│ └── time_series_data.py

├── dashboard/
│ └── app.py

├── results/
│ └── figures/

├── requirements.txt
└── README.md


---

## How to Run

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Model Performance

- Accuracy: ~73–75%

Evaluation metrics:
- Precision  
- Recall  
- F1-score  

---

## Future Work

- Use real longitudinal clinical data  
- Explore deep learning models such as LSTM  
- Add SHAP-based interpretability to the dashboard  
- Deploy as a web application  

---

## Author

Pavani Athota