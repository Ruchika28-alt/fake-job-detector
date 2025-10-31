# app.py
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    model_lr = joblib.load("models/pipe_logreg_tfidf.joblib")
    model_rf = joblib.load("models/pipe_rf_tfidf.joblib")
    return model_lr, model_rf

model_lr, model_rf = load_models()
st.set_page_config(page_title="Fake Job Posting Detector", layout="centered")

st.title("Fake Job Posting Detector")
st.write("Upload a CSV or paste a job posting to get predictions. Models: Logistic Regression, Random Forest")

mode = st.radio("Mode", ["Single prediction", "Batch (CSV upload)"])

label_map = {0: "Real", 1: "Fake"}

if mode == "Single prediction":
    st.subheader("Paste job posting text ")
    text = st.text_area("Job text", height=250)
    model_choice = st.selectbox("Model", ("Logistic Regression", "Random Forest"))
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please provide job posting text.")
        else:
            model = model_lr if model_choice == "Logistic Regression" else model_rf
            pred = model.predict([text])[0]
            proba = None
            try:
                proba = model.predict_proba([text])[0][1]
            except:
                pass
            st.success(f"Prediction: **{label_map[int(pred)]}**  (label id: {int(pred)})")
            if proba is not None:
                st.write(f"Probability (fake): {proba:.3f}")

else:
    st.subheader("Upload CSV (must contain columns: title, company_profile, description, requirements, benefits) or at least 'text' column")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_choice = st.selectbox("Model (batch)", ("Logistic Regression", "Random Forest"), key="batch_model")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # If text column not present, try to combine known columns
        if 'text' not in df.columns:
            df['text'] = (
                df.get('title', '').fillna('') + " . " +
                df.get('company_profile', '').fillna('') + " . " +
                df.get('description', '').fillna('') + " . " +
                df.get('requirements', '').fillna('') + " . " +
                df.get('benefits', '').fillna('')
            )
        model = model_lr if model_choice == "Logistic Regression" else model_rf
        preds = model.predict(df['text'].astype(str).tolist())
        probs = None
        try:
            probs = model.predict_proba(df['text'].astype(str).tolist())[:,1]
        except:
            probs = [None]*len(preds)
        df['predicted_fraudulent'] = preds
        df['predicted_label'] = df['predicted_fraudulent'].map({0:'Real',1:'Fake'})
        df['prob_fake'] = probs
        st.write("Prediction results (first 20 rows):")
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

