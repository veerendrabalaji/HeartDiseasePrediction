import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler= pickle.load(open('scaler.pkl','rb'))

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("---")

st.sidebar.info("""
### About

This application predicts whether a patient is at risk of heart disease using **Machine Learning**.

### Model

- Logistic Regression

### Dataset

UCI Heart Disease Dataset

### Developed With

- Python
- Streamlit
- Scikit-Learn
""")

st.sidebar.markdown("---")

st.sidebar.success("Mini Project")

# ----------------------------
# Title
# ----------------------------
st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the patient's clinical information below and click **Predict**."
)

st.markdown("---")

# ----------------------------
# Two Columns
# ----------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "👤 Age",
        min_value=1,
        max_value=100,
        value=45
    )

    sex = st.selectbox(
        "🚻 Gender",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "💔 Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "🩸 Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "🧪 Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120 mg/dL",
        ["No", "Yes"]
    )

with col2:

    restecg = st.selectbox(
        "📈 Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    thalach = st.number_input(
        "❤️ Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exang = st.selectbox(
        "🏃 Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "📉 Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "📊 ST Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.selectbox(
        "🫀 Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "🩺 Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )

# ----------------------------
# Encoding
# ----------------------------

sex = 1 if sex == "Male" else 0

cp = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}[cp]

fbs = 1 if fbs == "Yes" else 0

restecg = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}[restecg]

exang = 1 if exang == "Yes" else 0

slope = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}[slope]

thal = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}[thal]

# ----------------------------
# Input Array
# ----------------------------

input_data = np.array([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------

if st.button("❤️ Predict", use_container_width=True):
    scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.markdown("---")

    st.subheader("Prediction Confidence")

    healthy = probability[0][0]
    disease = probability[0][1]

    st.write(f"🟢 Healthy : **{healthy*100:.2f}%**")
    st.progress(float(healthy))

    st.write(f"🔴 Heart Disease : **{disease*100:.2f}%**")
    st.progress(float(disease))

    st.markdown("---")

    st.subheader("Risk Level")

    if disease < 0.30:
        st.success("🟢 Low Risk")

    elif disease < 0.70:
        st.warning("🟡 Moderate Risk")

    else:
        st.error("🔴 High Risk")

    st.markdown("---")

    st.subheader("📋 Patient Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Age", age)
        st.metric("Gender", "Male" if sex else "Female")

    with c2:
        st.metric("Blood Pressure", trestbps)
        st.metric("Heart Rate", thalach)

    with c3:
        st.metric("Cholesterol", chol)
        st.metric("Major Vessels", ca)

st.markdown("---")

st.warning(
"""
**Disclaimer**

This application is intended for educational purposes only.
It should **not** be used as a substitute for professional medical diagnosis or treatment.
Always consult a qualified healthcare provider.
"""
)