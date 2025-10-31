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
# Streamlit Configuration
# -------------------------------
st.set_page_config(page_title="Fake Job Posting Detector", layout="centered")
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.write("Detect whether a job posting is **real or fake** using trained ML models.")
st.caption("Models available: Logistic Regression & Random Forest")

mode = st.radio("Select Mode:", ["Single Prediction", "Batch (CSV Upload)"])
label_map = {0: "Real", 1: "Fake"}

# -------------------------------
# SINGLE PREDICTION MODE
# -------------------------------
if mode == "Single Prediction":
    st.subheader("🔹 Enter Job Posting Details")

    # Text inputs
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job Title", placeholder="e.g. Data Analyst")
        company_profile = st.text_area("Company Profile", placeholder="Brief about the company...", height=100)
    with col2:
        description = st.text_area("Job Description", placeholder="Describe job responsibilities...", height=100)
        requirements = st.text_area("Requirements", placeholder="List required skills, experience...", height=100)

    benefits = st.text_area("Benefits", placeholder="Mention benefits like WFH, insurance, etc.", height=80)

    # Dropdowns (optional)
    employment_type = st.selectbox("Employment Type", ["", "Full-time", "Part-time", "Internship", "Contract", "Other"])
    required_experience = st.selectbox("Experience Level", ["", "Entry Level", "Mid-Senior level", "Director", "Executive"])
    industry = st.selectbox("Industry", ["", "IT", "Finance", "Healthcare", "Education", "Marketing", "Other"])

    model_choice = st.selectbox("Select Model", ("Logistic Regression", "Random Forest"))

    if st.button("🔍 Predict"):
        # Combine all inputs into one text string
        combined_text = (
            f"Title: {title}. "
            f"Company Profile: {company_profile}. "
            f"Description: {description}. "
            f"Requirements: {requirements}. "
            f"Benefits: {benefits}. "
            f"Employment Type: {employment_type}. "
            f"Experience: {required_experience}. "
            f"Industry: {industry}."
        )

        if not title and not description and not requirements:
            st.warning("Please fill in at least the Title, Description, or Requirements fields.")
        else:
            model = model_lr if model_choice == "Logistic Regression" else model_rf
            pred = model.predict([combined_text])[0]
            proba = None
            try:
                proba = model.predict_proba([combined_text])[0][1]
            except Exception:
                pass

            st.success(f"Prediction: **{label_map[int(pred)]}** (Label ID: {int(pred)})")
            if proba is not None:
                st.write(f"Probability (Fake): `{proba:.3f}`")

# -------------------------------
# BATCH (CSV UPLOAD) MODE
# -------------------------------
else:
    st.subheader("📂 Batch Prediction (CSV Upload)")
    st.write("Upload a CSV file with columns like: `title`, `company_profile`, `description`, `requirements`, `benefits`")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    model_choice = st.selectbox("Select Model (Batch)", ("Logistic Regression", "Random Forest"), key="batch_model")

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Combine text columns if 'text' not present
        if 'text' not in df.columns:
            df['text'] = (
                df.get('title', '').fillna('') + ". " +
                df.get('company_profile', '').fillna('') + ". " +
                df.get('description', '').fillna('') + ". " +
                df.get('requirements', '').fillna('') + ". " +
                df.get('benefits', '').fillna('')
            )

        model = model_lr if model_choice == "Logistic Regression" else model_rf
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
