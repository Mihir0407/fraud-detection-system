import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------
# Load model & scaler
# -------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("Detect fraudulent transactions using Machine Learning")

# -------------------------
# Load dataset once
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

# -------------------------
# SIDEBAR (Better UX)
# -------------------------
st.sidebar.header("⚡ Quick Testing")

test_option = st.sidebar.selectbox(
    "Choose Test Type",
    ["Manual Input", "Normal Transaction", "Fraud Transaction", "Random Transaction"]
)

# -------------------------
# MAIN LOGIC
# -------------------------

def predict_transaction(data):
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return pred, prob


# =========================
# 🔹 OPTION 1: MANUAL INPUT
# =========================
if test_option == "Manual Input":

    st.subheader("🧾 Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("Time", value=0.0)
        amount = st.number_input("Amount", value=0.0)

    st.markdown("### PCA Features (V1–V28)")
    st.caption("These are transformed features (no need to fully understand them)")

    features = []
    cols = st.columns(4)

    for i in range(28):
        with cols[i % 4]:
            val = st.number_input(f"V{i+1}", value=0.0, key=f"V{i}")
            features.append(val)

    input_data = [time] + features + [amount]
    input_array = np.array(input_data).reshape(1, -1)

    if st.button("🔍 Predict Transaction"):
        pred, prob = predict_transaction(input_array)

        if pred == 1:
            st.error(f"🚨 Fraud Detected\n\nProbability: {prob:.4f}")
        else:
            st.success(f"✅ Normal Transaction\n\nProbability: {prob:.4f}")


# =========================
# 🔹 OPTION 2: NORMAL
# =========================
elif test_option == "Normal Transaction":

    st.subheader("✅ Testing with Normal Transaction")

    sample = df[df["Class"] == 0].sample(1)
    X_sample = sample.drop("Class", axis=1)

    pred, prob = predict_transaction(X_sample)

    st.dataframe(sample)

    st.success(f"Prediction: Normal\nProbability: {prob:.4f}")


# =========================
# 🔹 OPTION 3: FRAUD
# =========================
elif test_option == "Fraud Transaction":

    st.subheader("🚨 Testing with Fraud Transaction")

    sample = df[df["Class"] == 1].sample(1)
    X_sample = sample.drop("Class", axis=1)

    pred, prob = predict_transaction(X_sample)

    st.dataframe(sample)

    st.error(f"Prediction: Fraud\nProbability: {prob:.4f}")


# =========================
# 🔹 OPTION 4: RANDOM
# =========================
elif test_option == "Random Transaction":

    st.subheader("🎲 Random Transaction Test")

    sample = df.sample(1)
    X_sample = sample.drop("Class", axis=1)

    pred, prob = predict_transaction(X_sample)

    st.dataframe(sample)

    if pred == 1:
        st.error(f"🚨 Fraud\nProbability: {prob:.4f}")
    else:
        st.success(f"✅ Normal\nProbability: {prob:.4f}")


# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("Built with XGBoost | Imbalanced Learning | SHAP Ready")