"""
============================================================
  SPAM EMAIL CLASSIFIER — Streamlit App
============================================================
Run:  streamlit run app.py
Requires: models/spam_model.pkl and models/vectorizer.pkl
============================================================
"""

import re
import os
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
)

# ── NLTK setup ───────────────────────────────────────────
@st.cache_resource
def init_nlp():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',  quiet=True)
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = init_nlp()

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "spam_model.pkl")
    vec_path   = os.path.join("models", "vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("Model files not found! Run spam_classifier_improved.py first.")
        st.stop()
    return joblib.load(model_path), joblib.load(vec_path)

model, vectorizer = load_model()

# ── Preprocessing ────────────────────────────────────────
def extract_body(raw):
    parts = re.split(r'\n\n', raw, maxsplit=1)
    return parts[1] if len(parts) > 1 else raw

def clean_text(text):
    text = extract_body(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ',   text)
    text = re.sub(r'\S+@\S+',           ' email ', text)
    text = re.sub(r'\d+',               ' num ',   text)
    text = re.sub(r'[^\w\s]',           ' ',       text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 1]
    return " ".join(words)

def predict(raw_text):
    vec       = vectorizer.transform([clean_text(raw_text)])
    probs     = model.predict_proba(vec)[0]
    spam_prob = float(probs[1])
    return spam_prob > 0.5, spam_prob

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
.result-spam {
    background:#ffe0e0; border-left:5px solid #e53935;
    padding:1rem 1.2rem; border-radius:8px; margin-top:1rem;
}
.result-ham {
    background:#e0f4e0; border-left:5px solid #43a047;
    padding:1rem 1.2rem; border-radius:8px; margin-top:1rem;
}
.result-title { font-size:1.5rem; font-weight:700; margin-bottom:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── UI ───────────────────────────────────────────────────
st.title("📧 Spam Email Detector")
st.markdown(
    "Powered by an **SVM + TF-IDF** pipeline trained on the "
    "[SpamAssassin Public Corpus](https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus)."
)
st.divider()

col1, col2 = st.columns([3, 1], gap="small")

with col1:
    email_input = st.text_area(
        "Paste your email text below:",
        height=220,
        placeholder="Type or paste email content here…",
    )

with col2:
    st.markdown("#### Try a sample")
    if st.button("📬 Normal email"):
        st.session_state["sample"] = (
            "Hi John,\n\nJust a quick reminder about our team meeting "
            "tomorrow at 10 AM in Conference Room B. Please bring the "
            "Q3 project report.\n\nBest regards,\nSarah"
        )
        st.rerun()
    if st.button("🚨 Spam email"):
        st.session_state["sample"] = (
            "CONGRATULATIONS!!! You've WON a $1,000,000 prize!\n\n"
            "Click here IMMEDIATELY to claim your FREE reward before it expires. "
            "Limited time offer! Act NOW! Call 1-800-SCAM-123"
        )
        st.rerun()

if "sample" in st.session_state:
    email_input = st.session_state.pop("sample")

if st.button("🔍 Analyse Email", type="primary", use_container_width=True):
    if not email_input.strip():
        st.warning("Please enter some email text first.")
    else:
        with st.spinner("Analysing …"):
            is_spam, spam_prob = predict(email_input)
        ham_prob = 1 - spam_prob

        if is_spam:
            st.markdown(
                f'<div class="result-spam">'
                f'<div class="result-title">🚫 SPAM Detected</div>'
                f'Spam confidence: <strong>{spam_prob:.1%}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-ham">'
                f'<div class="result-title">✅ Legitimate Email</div>'
                f'Ham confidence: <strong>{ham_prob:.1%}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

        col_h, col_s = st.columns(2)
        col_h.metric("✅ Ham",  f"{ham_prob:.1%}")
        col_s.metric("🚫 Spam", f"{spam_prob:.1%}")
        st.progress(spam_prob, text=f"Spam probability: {spam_prob:.1%}")

        with st.expander("🔬 Show preprocessed text"):
            st.code(clean_text(email_input), language=None)

st.divider()
st.caption(
    "Model: LinearSVC with Platt scaling  •  "
    "Features: TF-IDF (10 000 unigrams + bigrams)  •  "
    "Training accuracy ≈ 99.6%"
)