import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and preprocessing objects
with open("credit_risk_model.pkl", "rb") as file:
    deployment_objects = pickle.load(file)

model = deployment_objects["model"]
one_hot = deployment_objects["one_hot"]
scaler = deployment_objects["scaler"]
cat_cols = deployment_objects["cat_cols"]
encoded_cols = deployment_objects["encoded_cols"]
scale_cols = deployment_objects["scale_cols"]
train_columns = deployment_objects["train_columns"]

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Credit Risk Prediction")
st.subheader("Loan Applicant Risk Classification")

st.image("images/modern_fintech_dashboard_illustration_banner.png", width="stretch")

st.write(
    "This app predicts whether a loan applicant is risky or non-risky "
    "based on financial, employment, loan, and credit-related information."
)

st.info(
    "Note: This app is built for educational and portfolio purposes. "
    "It should not be used as the only basis for real financial decisions."
)

st.divider()

st.subheader("Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

income = st.number_input("Annual Income", min_value=0, value=60000)

home = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=60.0,
    value=5.0
)

intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
)

amount = st.number_input("Loan Amount", min_value=0, value=10000)

rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=12.5
)

percent_income = st.number_input(
    "Loan Amount as Percentage of Income",
    min_value=0.0,
    max_value=2.0,
    value=0.17
)

default = st.selectbox(
    "Previous Default History",
    ["N", "Y"]
)

cred_length = st.number_input(
    "Credit History Length (Years)",
    min_value=0,
    max_value=100,
    value=4
)

if st.button("Predict Credit Risk"):
    new_applicant = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Home": [home],
        "Emp_length": [emp_length],
        "Intent": [intent],
        "Amount": [amount],
        "Rate": [rate],
        "Percent_income": [percent_income],
        "Default": [default],
        "Cred_length": [cred_length]
    })

    # Same feature engineering used during training
    new_applicant["Income_Category"] = pd.cut(
        new_applicant["Income"],
        bins=[0, 25000, 50000, 100000, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    )

    new_applicant["Income_log"] = np.log1p(new_applicant["Income"])
    new_applicant["Amount_log"] = np.log1p(new_applicant["Amount"])

    new_applicant = new_applicant.drop(["Income", "Amount"], axis=1)

    # Encode categorical columns using fitted encoder
    encoded_input = one_hot.transform(new_applicant[cat_cols])

    encoded_input_df = pd.DataFrame(
        encoded_input,
        columns=encoded_cols,
        index=new_applicant.index
    )

    new_applicant = pd.concat(
        [new_applicant.drop(cat_cols, axis=1), encoded_input_df],
        axis=1
    )

    # Match training column order
    new_applicant = new_applicant.reindex(columns=train_columns, fill_value=0)

    # Scale numerical columns using fitted scaler
    new_applicant[scale_cols] = scaler.transform(new_applicant[scale_cols])

    prediction = model.predict(new_applicant)
    prediction_probability = model.predict_proba(new_applicant)

    non_risky_probability = prediction_probability[0][0]
    risky_probability = prediction_probability[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Result: Risky Applicant")
    else:
        st.success("Result: Non-Risky Applicant")

    st.write(f"Non-Risky Probability: **{non_risky_probability:.2%}**")
    st.write(f"Risky Probability: **{risky_probability:.2%}**")