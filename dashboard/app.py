import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load data
df = pd.read_csv(r"C:\Users\Pavani\Desktop\diabetes-risk-assessment\data\processed_data\diabetes_clean.csv")

# Load model
model = pickle.load(open(r"C:\Users\Pavani\Desktop\diabetes-risk-assessment\models\diabetes_model.pkl","rb"))

st.set_page_config(layout="wide")

# PDF function
def create_pdf(pregnancies, glucose, blood_pressure, bmi, age, prob, risk_level):
    doc = SimpleDocTemplate("diabetes_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Diabetes Risk Report", styles['Title']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Pregnancies: {pregnancies}", styles['Normal']))
    content.append(Paragraph(f"Glucose: {glucose}", styles['Normal']))
    content.append(Paragraph(f"Blood Pressure: {blood_pressure}", styles['Normal']))
    content.append(Paragraph(f"BMI: {bmi}", styles['Normal']))
    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Risk Percentage: {prob*100:.2f}%", styles['Normal']))
    content.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))

    doc.build(content)

# TITLE
st.title("🩺 Diabetes Analytics Dashboard")

# METRICS
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Patients", len(df))
c2.metric("Average BMI", round(df["BMI"].mean(),2))
c3.metric("Average Glucose", round(df["Glucose"].mean(),2))
c4.metric("Average Pregnancies", round(df["Pregnancies"].mean(),2))

st.divider()

# DATA OVERVIEW
st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

# Donut chart
with col1:
    counts = df["Outcome"].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=["No Diabetes","Diabetes"],
        values=counts,
        hole=0.55
    )])

    st.plotly_chart(fig, use_container_width=True)

# Avg glucose by age
with col2:
    st.subheader("Average Glucose by Age")
    avg_glucose = df.groupby("Age")["Glucose"].mean()
    st.line_chart(avg_glucose)

# PREDICTION
st.header("Predict Diabetes Risk")

left_col, right_col = st.columns(2)

with left_col:
    pregnancies = st.number_input("Pregnancies",0,20,1)
    glucose = st.number_input("Glucose",0,300,120)
    blood_pressure = st.number_input("Blood Pressure",0,150,70)
    skin_thickness = st.number_input("Skin Thickness",0,100,20)
    insulin = st.number_input("Insulin",0,900,80)
    bmi = st.number_input("BMI",10.0,50.0,25.0)
    dpf = st.number_input("Diabetes Pedigree Function",0.0,2.5,0.5)
    age = st.number_input("Age",18,90,40)

    predict_button = st.button("Predict Diabetes Risk")

with right_col:
    st.header("Insights: Factors Affecting Diabetes Risk")
    st.markdown("""
### 🔬 Key Risk Factors
- High Glucose → strongest predictor  
- High BMI → obesity increases risk  
- Age → risk increases  
- Genetics → family history  

### 📌 Interpretation
Higher glucose + BMI → higher diabetes risk
""")

# OUTPUT
if predict_button:

    input_data = np.array([[pregnancies,glucose,blood_pressure,
                            skin_thickness,insulin,bmi,dpf,age]])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Diabetes Risk Analysis")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk %", f"{prob*100:.2f}%")

    with col2:
        st.subheader("Lifestyle Recommendations")

        if prob > 0.7:
            risk_level = "High"
            st.error("🔴 Reduce sugar, exercise daily, consult doctor")
        elif prob > 0.4:
            risk_level = "Moderate"
            st.warning("🟡 Improve diet, exercise regularly")
        else:
            risk_level = "Low"
            st.success("🟢 Maintain healthy lifestyle")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Clinical Glucose Assessment")

        if glucose >= 200:
            st.error("🔴 High Diabetes Risk (≥ 200)")
        elif glucose >= 140:
            st.warning("🟡 Prediabetic (140–199)")
        else:
            st.success("🟢 Normal")

    with col4:
        st.subheader("Final Risk Interpretation")

        if prob > 0.7 or glucose >= 200:
            st.error("🚨 High Risk")
        elif prob > 0.4 or glucose >= 140:
            st.warning("⚠️ Moderate Risk")
        else:
            st.success("✅ Low Risk")

    # Gauge 
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Scaled Target Risk"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': "#5DADE2"},
            'steps': [
                {'range':[0,40],'color':"#D5F5E3"},
                {'range':[40,70],'color':"#FCF3CF"},
                {'range':[70,100],'color':"#FADBD8"}
            ]
        }
    ))

    gauge.update_layout(height=300)

    st.plotly_chart(gauge, use_container_width=True)

# TIME SERIES
st.header("Time-Series Analysis")


time_steps = 10
data_list = []

for _, row in df.head(1).iterrows():
    for t in range(time_steps):
        new_row = row.copy()
        new_row["Glucose"] = row["Glucose"] + (t * 5) + np.random.randint(-5,5)
        new_row["BMI"] = row["BMI"] + np.random.uniform(-1,1)
        new_row["Age"] = row["Age"] + t
        new_row["time_step"] = t
        data_list.append(new_row)

ts_df = pd.DataFrame(data_list)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Glucose Trend")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(ts_df["time_step"], ts_df["Glucose"], marker='o')
    st.pyplot(fig)

with col2:
    st.subheader("Risk Trend")

    risk_values = []

    for _, row in ts_df.iterrows():
        input_data = np.array([[row["Pregnancies"],
                                row["Glucose"],
                                row["BloodPressure"],
                                row["SkinThickness"],
                                row["Insulin"],
                                row["BMI"],
                                row["DiabetesPedigreeFunction"],
                                row["Age"]]])

        prob = model.predict_proba(input_data)[0][1]
        risk_values.append(prob)

    ts_df["Risk"] = risk_values

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.plot(ts_df["time_step"], ts_df["Risk"], marker='o')
    st.pyplot(fig2)

# DATA PREVIEW
st.subheader("Dataset Preview")
rows = st.slider("Number of rows", 5, 100, 20)
st.dataframe(df.head(rows))