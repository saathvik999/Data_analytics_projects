
import pandas as pd
import numpy as np
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank (1).csv")
    return df

df = load_data()

def preprocess(df):
    df = df.copy()

    df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df


@st.cache_resource
def train_model():
    data = preprocess(df)

    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    return model, scaler, X.columns, auc


model, scaler, feature_cols, auc_score = train_model()


# STREAMLIT UI
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🏦 Bank Customer Churn Prediction App")
st.write("Predict whether a customer will churn using Machine Learning")

st.subheader(f"📊 Model ROC-AUC Score: {auc_score:.3f}")

st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 80, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", value=50000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", value=50000.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])


def predict():
    input_data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Gender": gender,
        "Geography": geography
    }

    input_df = pd.DataFrame([input_data])

    # Apply same preprocessing
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    return prob

if st.button("Predict Churn"):

    probability = predict()

    st.subheader("🔮 Prediction Result")

    st.write(f"**Churn Probability:** {probability:.2f}")

    if probability > 0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    # Progress bar
    st.progress(float(probability))

st.subheader("📌 Feature Importance")

importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False).head(10)

st.bar_chart(importance)

if st.checkbox("Show Raw Data"):
    st.write(df.head())