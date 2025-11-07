# 🧠 Fake Job Posting Detection using NLP and Machine Learning

## 📌 Overview
The rise of online job portals has brought convenience to job seekers, but it has also created opportunities for fraudsters to exploit unsuspecting applicants. Fake job postings, often designed to steal personal information or solicit money, have become a growing concern in today’s digital employment landscape.

This project focuses on building an automated system that can detect fraudulent job advertisements using Natural Language Processing (NLP) and Machine Learning (ML). By analyzing textual features of job postings, the system aims to classify them as either real or fake, thereby helping users identify potentially suspicious listings before engaging with them.

The project compares two models — **Logistic Regression** and **Random Forest** — to determine which performs better in identifying suspicious job postings.  

The final solution is deployed using **Streamlit**, allowing users to paste or upload job descriptions and instantly check whether they appear *real* or *fake*.
 

🔗 Live Demo: Fake Job Posting Detector
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

### Preprocessing Steps
1. Removed missing and duplicate text entries  
2. Combined multiple textual columns into a single feature (`text`)  
3. Converted text to lowercase and removed punctuation  
4. Transformed text into numeric vectors using **TF-IDF Vectorization**

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

## ▶️ Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector 
