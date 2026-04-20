


# 📧 Email Spam Classification Using NLP and Machine Learning

> A machine learning pipeline for automated spam detection, built with SVM and TF-IDF vectorisation — trained on the SpamAssassin Public Corpus.

---

## 🧾 Project Overview

This project was developed as part of the **Predictive Analytics** course at **Digital University Kerala**. It implements a complete end-to-end spam email classification system — from raw text preprocessing through model training and evaluation, to a fully interactive web application for real-time inference.

The classifier distinguishes between legitimate (*ham*) and unsolicited (*spam*) emails using classical NLP preprocessing techniques combined with a high-performance Support Vector Machine (SVM) model.

---

## 🎯 Problem Statement & Motivation

Email spam is one of the most pervasive problems in digital communication. Unsolicited emails clog inboxes, reduce productivity, and often serve as vectors for phishing attacks, malware, and fraud. Despite decades of filtering technology, spam still accounts for a significant portion of global email traffic, making robust, accurate detection systems critically important.

The goal of this project is to build an intelligent, interpretable spam detection system using classical NLP and machine learning techniques — specifically TF-IDF feature extraction combined with Naive Bayes and Support Vector Machine classifiers. The motivation is to achieve high precision (minimising false positives, i.e., legitimate emails being flagged as spam) while maintaining strong recall, using a well-established public benchmark dataset.

---

## 🎯 Objectives

- Build a robust, interpretable spam detection pipeline using classical NLP and ML techniques
- Compare the performance of Multinomial Naive Bayes and Linear SVM classifiers
- Deploy the final model as an interactive Streamlit web application
- Achieve high precision to minimise false positives (legitimate emails flagged as spam)

---

## 📁 Repository Structure

```
├── app.py                    # Streamlit web application
├── code.ipynb                # Model training and evaluation notebook
├── models/
│   ├── spam_model.pkl        # Trained LinearSVC model (with Platt scaling)
│   └── vectorizer.pkl        # Fitted TF-IDF vectoriser
├── assets/
│   ├── model_comparison.png  # Accuracy & F1 score comparison chart
│   ├── confusion_matrix.png  # Confusion matrix — Optimised SVM
│   └── confusion_matrices.png# Side-by-side Naive Bayes vs SVM matrices
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [SpamAssassin Public Corpus](https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus) |
| **Total Emails** | 4,198 |
| **Ham (Legitimate)** | 2,801 (66.7%) |
| **Spam** | 1,397 (33.3%) |
| **Format** | Raw email files (headers + body) |
| **Train / Test Split** | 80% training · 20% test (3,358 train / 840 test) |
| **Features** | TF-IDF vectors (10,000 features, unigrams + bigrams) |

---

## 🔧 Methodology

### 1. Text Preprocessing

Each raw email is processed through a multi-step pipeline:

- **Header stripping** — isolates the email body from metadata headers
- **Lowercasing** — normalises case
- **URL/email/number tokenisation** — replaces URLs with `url`, email addresses with `email`, and digits with `num`
- **Punctuation removal**
- **Stopword removal** — using NLTK English stopword list
- **Lemmatisation** — reduces words to their base form using WordNet Lemmatizer

### 2. Feature Extraction

A **TF-IDF Vectoriser** is fitted on the training corpus with the following configuration:

- **Vocabulary size:** 10,000 features
- **N-gram range:** Unigrams + Bigrams `(1, 2)`
- **Sublinear TF scaling** for improved performance on high-frequency terms

### 3. Model Training & Comparison

Two classifiers were trained and evaluated:

| Model | Accuracy | F1 Score |
|---|---|---|
| Multinomial Naive Bayes | ~93.7% | ~90.8% |
| **Linear SVM (LinearSVC)** | **~98.6%** | **~97.9%** |

The **SVM model significantly outperforms Naive Bayes** on both metrics. Platt scaling was applied to the SVM via `CalibratedClassifierCV` to enable probability estimates for confidence scoring.

### 4. Evaluation

The optimised SVM model was evaluated on a held-out test set:

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 550 ✅ | 10 ❌ |
| **Actual Spam** | 2 ❌ | 278 ✅ |

- **False Positive Rate (Ham → Spam):** Only 10 legitimate emails misclassified — critical for user trust
- **False Negative Rate (Spam → Ham):** Only 2 spam emails missed
- **Overall Test Accuracy:** ~98.6%

---

## 📈 Results Summary

| Metric | Naive Bayes | SVM (Final) |
|---|---|---|
| Accuracy | 93.7% | **98.6%** |
| F1 Score | 90.8% | **97.9%** |
| False Positives | 33 | **10** |
| False Negatives | 20 | **2** |

The LinearSVC model with TF-IDF features delivers near state-of-the-art performance for classical ML on this task, with particularly strong precision — essential for a spam filter where misclassifying legitimate emails is costly.

---

## 🖥️ Web Application

The project ships with an interactive **Streamlit** web app (`app.py`) that provides real-time spam detection.

**Features:**
- Paste any raw email text (including headers) for instant classification
- Displays spam/ham probability with a visual confidence bar
- One-click sample emails for quick testing (normal & spam)
- Expandable view of preprocessed text for transparency

### Screenshots

**Home — Input Interface**

![App Home](https://github.com/user-attachments/assets/6c19c255-6a32-4594-bb6f-2f8a66602e36)

**Spam Detection — Suspicious Email**

![Spam Result](https://github.com/user-attachments/assets/b91a73d0-db1f-437a-9a84-0ffaa80c31a1)

**Ham Detection — Legitimate Email**

![Ham Result](https://github.com/user-attachments/assets/5df00db5-aeb5-4ea2-867c-e1cad400c9be)


> 🔗 **Live Demo:** [https://email-classification-using-nlp-and-machine-learning-enmusnrjkc.streamlit.app/](https://email-classification-using-nlp-and-machine-learning-enmusnrjkc.streamlit.app/)

---

## ⚙️ Setup & Running Locally

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Email-Classification-Using-NLP-and-Machine-Learning.git
cd Email-Classification-Using-NLP-and-Machine-Learning
```

**2. Install dependencies**

```bash
pip install scikit-learn nltk joblib streamlit
```

**3. Download NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

**4. Place model files**

```bash
mkdir models
cp spam_model.pkl models/
cp vectorizer.pkl models/
```

**5. Launch the app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

```
Python >= 3.8
scikit-learn
nltk
joblib
streamlit
```

---

## 🏫 Academic Context

| Field | Detail |
|---|---|
| **Institution** | Digital University Kerala |
| **Course** | Predictive Analytics |
| **Topic** | Email Classification Using NLP and Machine Learning |

---

## 👥 Team Members

| Name |
|---|
| Abikrishnan M S |
| Mohammed Yazin N |
| Athulya Mannambath |

---

## 📄 License

This project is intended for academic and educational purposes.
