
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Estimator",
    layout="wide"
)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "insurance_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

# Reference average charges
REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000


# Load trained model and feature columns
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None

    if not os.path.exists(FEATURES_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    return model, feature_cols


# BMI category
def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


# Prepare user input
def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    category = bmi_category(bmi)

    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category_obese": 1 if category == "obese" else 0,
        "bmi_category_overweight": 1 if category == "overweight" else 0,
        "bmi_category_underweight": 1 if category == "underweight" else 0,
    }

    encoded = pd.DataFrame([row])

    return encoded.reindex(columns=feature_cols, fill_value=0)


# Load artifacts
model, feature_cols = load_artifacts()

st.title("Insurance Cost Estimator")
st.write("Predict your estimated annual medical insurance charges.")

if model is None:
    st.error("Model or feature file not found.")
    st.info(
        "Check that insurance_model.pkl and model_features.pkl "
        "are present in the same folder as app.py."
    )
    st.stop()

# Layout
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Patient Details")

    age = st.slider("Age", 18, 64, 30)

    sex = st.radio(
        "Sex",
        ["male", "female"],
        horizontal=True
    )

    bmi = st.slider(
        "BMI",
        15.0, 55.0, 26.0, 0.1
    )

    children = st.slider("Children", 0, 5, 0)

    smoker = st.radio(
        "Smoker",
        ["no", "yes"],
        horizontal=True
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    predict_clicked = st.button(
        "Estimate Charges",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("Estimated Charges")

    if predict_clicked:
        try:
            input_row = build_input_row(
                age,
                sex,
                bmi,
                children,
                smoker,
                region,
                feature_cols
            )

            # Predict charges
            prediction = model.predict(input_row)[0]

            # Use expm1 only if the model was trained on log(charges)
            pred_charges = np.expm1(prediction)

            st.metric(
                "Predicted Annual Charge",
                f"${pred_charges:,.2f}"
            )

            st.divider()

            # Comparison chart
            compare_df = pd.DataFrame({
                "Group": [
                    "Your prediction",
                    "Non-smoker average",
                    "Smoker average"
                ],
                "Charges": [
                    pred_charges,
                    REF_AVG_NON_SMOKER,
                    REF_AVG_SMOKER
                ]
            }).set_index("Group")

            st.bar_chart(
                compare_df,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.info(
            "Adjust the patient details and click "
            "**Estimate Charges**."
        )
