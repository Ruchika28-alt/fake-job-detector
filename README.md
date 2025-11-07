# 🧠 Fake Job Posting Detection using NLP and Machine Learning

## 📌 Overview
The rise of online job portals has brought convenience to job seekers, but it has also created opportunities for fraudsters to exploit unsuspecting applicants. Fake job postings, often designed to steal personal information or solicit money, have become a growing concern in today’s digital employment landscape.

This project focuses on building an automated system that can detect fraudulent job advertisements using Natural Language Processing (NLP) and Machine Learning (ML). By analyzing textual features of job postings, the system aims to classify them as either real or fake, thereby helping users identify potentially suspicious listings before engaging with them.

The project compares two models — **Logistic Regression** and **Random Forest** — to determine which performs better in identifying suspicious job postings.  

The final solution is deployed using **Streamlit**, allowing users to paste or upload job descriptions and instantly check whether they appear *real* or *fake*.
 
🔗 Live Demo: Fake Job Posting Detector : https://fake-job-detector-ncnyxwvhhuospq9sjkbvrp.streamlit.app

---

## 📂 Dataset Source
The dataset used is derived from the **Employment Scam Aegean Dataset (EMSCAD)**, a popular benchmark for job fraud detection tasks.  
Additionally, a **synthetic labeled dataset (1,000 samples)** was created to evaluate and visualize model performance more effectively.

### Dataset Summary
| Property | Description |
|-----------|--------------|
| Training samples | ~18,000 |
| Test samples | 1,000 |
| Features | `title`, `company_profile`, `description`, `requirements`, `benefits` |
| Target variable | `fraudulent` → 0 = Real, 1 = Fake |

## ⚙️ Data Preprocessing

Before feeding the text data into machine learning models, several preprocessing steps were applied to ensure data quality and consistency:

### 1. Data Cleaning
- Removed missing entries and duplicates.  
- Dropped irrelevant columns to focus on textual features.

### 2. Text Consolidation
- Combined multiple text columns (`title`, `description`, `requirements`, etc.) into a single `text` feature to capture the full context of the job posting.

### 3. Text Normalization
- Converted all text to lowercase.  
- Removed punctuation, special characters, and excessive whitespace.

### 4. Vectorization
- Transformed textual data into numerical representations using **TF-IDF Vectorization**, which reflects the importance of words in the context of all documents.

---

## ⚙️ Methods

The overall process is based on text classification using two pipelines — **Logistic Regression** and **Random Forest** — built with **scikit-learn**.

### 🧩 Workflow
Raw Data → Cleaning → TF-IDF Vectorization → ML Model (LR / RF) → Evaluation
### Model Details
| Step | Description |
|------|--------------|
| **Feature Engineering** | TF-IDF to represent text numerically |
| **Model 1** | Logistic Regression (interpretable and fast) |
| **Model 2** | Random Forest (handles feature interactions) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Visualization Tools** | Matplotlib and Seaborn |

### Why These Models?
- **Logistic Regression** offers strong performance on sparse TF-IDF data and is easy to interpret.  
- **Random Forest** captures non-linear patterns and works well when features interact in complex ways.  

Alternative algorithms such as SVMs or XGBoost were considered but not included to keep the focus on interpretability and efficiency.

---
## 📊 Model Evaluation

Models are evaluated using standard metrics:  

- **Accuracy:** Overall correctness.  
- **Precision:** Fraction of predicted fakes that are actually fake.  
- **Recall:** Fraction of actual fakes correctly identified.  
- **F1-Score:** Harmonic mean of precision and recall.  
- **Confusion Matrix:** Visual summary of true/false positives and negatives.  

Visualizations are done using **Matplotlib** and **Seaborn** to analyze model performance.

---
## 🚀 Deployment

The trained models are deployed via **Streamlit**, allowing real-time predictions.  

**Features:**
- Paste or upload job descriptions.
- Get instant predictions: **Real** or **Fake**.
- Highlight suspicious keywords (Logistic Regression).

### How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector
```
2. Install dependencies
```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app
```bash
streamlit run app.py
```  
