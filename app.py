import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_churn_model():
    with open("churn_best_pipeline.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]

model, best_threshold = load_churn_model()
preprocessor = model.named_steps["preprocessor"]

# ---------------- TITLE ----------------
st.title("Customer Churn Prediction App")
st.write("Fill in customer details and click **Predict** to see churn risk and explanation.")

# ---------------- FORM ----------------
with st.form("prediction_form"):

    st.subheader("Enter Customer Details")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    threshold = st.slider(
        "Decision Threshold (Recommended: 0.35 based on profit curve analysis)",
        0.0,
        1.0,
        float(round(best_threshold, 3)),
        step=0.01,
    )
    
    submit = st.form_submit_button("Predict Churn")

# ---------------- PREDICTION BLOCK ----------------
if submit:

    # Create input dataframe
    input_df = pd.DataFrame(
        {
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "TechSupport": [tech_support],
            "PaymentMethod": [payment],
        }
    )

    # Prediction
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{proba:.2f}**")

    if proba >= threshold:
        st.error(
            f"High risk ({proba:.2f}) → Recommend retention action (e.g., discount or call)."
        )
    else:
        st.success(f"Low risk ({proba:.2f}) → No immediate action required.")

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("Why is this customer predicted to churn?")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform the single customer's input into the numeric format the model understands
    input_transformed = preprocessor.transform(input_df)

    # Recover the feature names that the preprocessor created (e.g. after one-hot encoding)
    # and strip the internal prefixes that ColumnTransformer adds automatically
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    # CHANGED: Detect which model was selected during training and use the correct
    # SHAP explainer for it. Different model families need different explainers,
    # and they also return SHAP values in different shapes — so we handle each separately.
    if isinstance(classifier, RandomForestClassifier):
        # TreeExplainer uses the actual tree structure to compute exact SHAP values.
        # For binary classification it returns shape (1, n_features, 2) — one value
        # per feature per class. We slice [0, :, 1] to get class 1 (churn) for
        # sample 0, giving a clean 1D array of shape (n_features,).
        explainer = shap.TreeExplainer(classifier)
        shap_values_all = explainer.shap_values(input_transformed)  # shape: (1, 19, 2)
        shap_values_single = shap_values_all[0, :, 1]               # shape: (19,)
        base_value = explainer.expected_value[1]                     # baseline for class 1

    elif isinstance(classifier, LogisticRegression):
        # LinearExplainer uses the model's coefficients to compute SHAP values.
        # It returns shape (1, n_features) — already just the positive class —
        # so we only need sample 0, no class slicing required.
        explainer = shap.LinearExplainer(classifier, input_transformed)
        shap_values_all = explainer.shap_values(input_transformed)  # shape: (1, 19)
        shap_values_single = shap_values_all[0]                     # shape: (19,)
        base_value = explainer.expected_value

    # shap_values_single is now a clean 1D array of shape (19,) in both cases —
    # one SHAP value per feature, showing how much each feature pushed the
    # prediction up or down from the baseline churn probability.

    fig = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single,   # how much each feature contributed
            base_values=base_value,      # the model's average prediction (starting point)
            data=input_transformed[0],   # the actual feature values for this customer
            feature_names=feature_names  # human-readable names for each feature
        ),
        show=False,
    )
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- TOP FEATURES ----------------
    # CHANGED: was shap_values[0] — now correctly uses shap_values_single
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Impact": shap_values_single,
        }
    )

    shap_df["AbsImpact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("AbsImpact", ascending=False).head(10)

    st.subheader("Top Factors Affecting Churn")
    st.bar_chart(shap_df.set_index("Feature")["Impact"])