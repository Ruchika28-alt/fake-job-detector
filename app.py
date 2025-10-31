# app.py
import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load trained models
# -------------------------------
@st.cache_resource
def load_models():
    model_lr = joblib.load("models/pipe_logreg_tfidf.joblib")
    model_rf = joblib.load("models/pipe_rf_tfidf.joblib")
    return model_lr, model_rf

model_lr, model_rf = load_models()

# -------------------------------
# Streamlit app configuration
# -------------------------------
st.set_page_config(page_title="Fake Job Posting Detector", layout="centered")
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.write(
    "Detect whether a job posting is **real or fake** using machine learning models."
)
st.caption("Models used: Logistic Regression and Random Forest")

# -------------------------------
# Choose mode
# -------------------------------
mode = st.radio("Select mode:", ["Single Prediction", "Batch (CSV Upload)"])
label_map = {0: "Real", 1: "Fake"}

# -------------------------------
# Single prediction section
# -------------------------------
if mode == "Single Prediction":
    st.subheader("🔹 Single Job Prediction")

    sample_placeholder = (
        "Eg : Work from home and earn $10,000 per week! No experience required. Click here to apply now!"
    )

    text = st.text_area(
        "Enter job posting details below:",
        height=250,
        placeholder=sample_placeholder
    )

    model_choice = st.selectbox("Select Model", ("Logistic Regression", "Random Forest"))

    if st.button("🔍 Predict"):
        if not text.strip():
            st.warning("Please enter job posting text.")
        else:
            model = model_lr if model_choice == "Logistic Regression" else model_rf
            pred = model.predict([text])[0]
            proba = None
            try:
                proba = model.predict_proba([text])[0][1]
            except Exception:
                pass

            st.success(f"Prediction: **{label_map[int(pred)]}** (Label ID: {int(pred)})")
            if proba is not None:
                st.write(f"Probability (Fake): `{proba:.3f}`")

# -------------------------------
# Batch prediction section
# -------------------------------
else:
    st.subheader("📂 Batch Prediction (CSV Upload)")
    st.write("Upload a CSV file with columns like: `title`, `company_profile`, `description`, `requirements`, `benefits`.")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    model_choice = st.selectbox("Select Model (Batch)", ("Logistic Regression", "Random Forest"), key="batch_model")

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Combine text columns if needed
        if 'text' not in df.columns:
            df['text'] = (
                df.get('title', '').fillna('') + ". " +
                df.get('company_profile', '').fillna('') + ". " +
                df.get('description', '').fillna('') + ". " +
                df.get('requirements', '').fillna('') + ". " +
                df.get('benefits', '').fillna('')
            )

        model = model_lr if model_choice == "Logistic Regression" else model_rf

        # Make predictions
        preds = model.predict(df['text'].astype(str).tolist())

        try:
            probs = model.predict_proba(df['text'].astype(str).tolist())[:, 1]
        except Exception:
            probs = [None] * len(preds)

        df['predicted_fraudulent'] = preds
        df['predicted_label'] = df['predicted_fraudulent'].map(label_map)
        df['prob_fake'] = probs

        st.write("### 🧾 Prediction Results (First 20 Rows)")
        st.dataframe(df.head(20))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )



