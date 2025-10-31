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

st.title("🕵️‍♀️ Fake Job Posting Detector")
st.write("Predict whether a job posting is **real** or **fake** using NLP models trained on real-world job data.")

mode = st.radio("Select Mode", ["Single Prediction", "Batch (CSV Upload)"])
label_map = {0: "✅ Real", 1: "🚨 Fake"}

# --- SINGLE TEXT MODE ---
if mode == "Single Prediction":
    st.subheader("Enter or select a job posting")

    # Example options
    examples = {
        "Example 1 (Fake)": "Work from home and earn $10,000 per week! No experience required. Click here to apply now!",
        "Example 2 (Real)": "We are hiring a Data Analyst with 3+ years of experience in Python, SQL, and Power BI. Bachelor's degree required.",
        "Example 3 (Fake)": "Online job opportunity! Send your bank details to start receiving payments today!",
        "Example 4 (Real)": "Marketing Intern needed at a reputed food-tech startup. Must have strong communication skills and creativity."
    }

    example_choice = st.selectbox("Choose an example or write your own:", ["-- Custom Input --"] + list(examples.keys()))
    text = ""
    if example_choice != "-- Custom Input --":
        text = examples[example_choice]
    else:
        text = st.text_area("Paste a job description below:", height=200, placeholder="Type or paste job description here...")

    model_choice = st.selectbox("Choose Model", ("Logistic Regression", "Random Forest"))

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter or select a job posting to predict.")
        else:
            model = model_lr if model_choice == "Logistic Regression" else model_rf
            pred = model.predict([text])[0]
            try:
                prob = model.predict_proba([text])[0][1]
            except:
                prob = None

            st.success(f"Prediction: **{label_map[int(pred)]}** (Label ID: {int(pred)})")
            if prob is not None:
                st.write(f"Probability of being Fake: **{prob:.3f}**")

# --- BATCH MODE ---
else:
    st.subheader("Upload a CSV for Batch Prediction")
    st.markdown("CSV must contain at least one column named **`text`** (or columns like title, description, etc.)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_choice = st.selectbox("Model (Batch)", ("Logistic Regression", "Random Forest"))

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Auto combine columns if 'text' not found
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

        try:
            probs = model.predict_proba(df['text'].astype(str).tolist())[:, 1]
        except:
            probs = [None] * len(preds)

        df['predicted_fraudulent'] = preds
        df['predicted_label'] = df['predicted_fraudulent'].map({0: 'Real', 1: 'Fake'})
        df['prob_fake'] = probs

        st.write("### Prediction Results (first 20 rows):")
        st.dataframe(df.head(20))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
