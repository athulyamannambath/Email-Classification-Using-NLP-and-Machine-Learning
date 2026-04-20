import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

SAMPLES = {
    "💰 Lottery winner":     ("spam", "Subject: Congratulations! You've Won $1,000,000!!\n\nDear Lucky Winner,\n\nYou have been selected as our GRAND PRIZE WINNER in our annual international lottery! Click here NOW to claim your FREE cash reward. This offer expires in 24 hours!\n\nCall 1-800-WIN-CASH or visit www.claimprize.net immediately.\n\n— Prize Committee"),
    "💊 Miracle pill":       ("spam", "Subject: Lose 30 lbs in 30 days — GUARANTEED!\n\nHi,\n\nDoctors HATE this one weird trick! Our miracle fat-burning pill melts belly fat overnight. No diet, no exercise needed. LIMITED STOCK — order now and get 3 bottles FREE!\n\nClick here: www.miraclepill.biz\n\n— Health Solutions Team"),
    "🏦 Bank alert":         ("spam", "Subject: URGENT: Your bank account has been suspended\n\nDear Customer,\n\nWe have detected suspicious activity on your account. Your account has been TEMPORARILY SUSPENDED. Verify your details immediately.\n\nwww.secure-bankverify.com/login\n\nFailure to verify within 12 hours will result in permanent closure.\n\n— Security Team"),
    "🎁 Free gift offer":    ("spam", "Subject: You've been selected for a FREE iPhone 15 Pro!\n\nCongratulations! You are today's lucky visitor chosen to receive a FREE iPhone 15 Pro. Complete our 30-second survey to claim. Only 3 left!\n\nClaim here: www.freeiphonegift.co\n\n— Rewards Center"),
    "🔐 Account verify":     ("spam", "Subject: Action Required: Verify Your PayPal Account\n\nDear User,\n\nYour PayPal account has been limited due to unusual activity. Verify now or your account will be permanently disabled within 48 hours.\n\nVerify Now: www.paypa1-secure.com/verify\n\n— PayPal Support"),
    "📅 Meeting reschedule": ("ham",  "Subject: Team sync moved to Thursday 3pm\n\nHi Sarah,\n\nJust a quick note — our weekly sync has been moved from Wednesday to Thursday at 3pm due to a client call. Could you please update the calendar invite and let the team know?\n\nAlso, don't forget to bring the Q1 report.\n\nThanks,\nMichael"),
    "📦 Order confirmed":    ("ham",  "Subject: Your order #ORD-847291 has been confirmed\n\nHi Rahul,\n\nThank you for your purchase! Your order has been confirmed and is being processed.\n\nItem: Noise-cancelling headphones\nAmount: ₹4,999\nEstimated delivery: April 14–16\n\n— Shop Team"),
    "👨‍💼 Job interview":      ("ham",  "Subject: Interview Invitation — Software Engineer Role\n\nDear Arjun,\n\nThank you for applying to the Software Engineer position at TechCorp. We were impressed with your profile and would like to invite you for an interview.\n\nDate: April 15, 2026\nTime: 11:00 AM IST\nFormat: Video call (Google Meet link to follow)\n\nPlease confirm your availability.\n\nBest regards,\nPriya Nair\nTalent Acquisition, TechCorp"),
    "🧾 Invoice reminder":   ("ham",  "Subject: Invoice #INV-2041 — Payment Due April 20\n\nHi,\n\nThis is a friendly reminder that Invoice #INV-2041 for ₹12,500 is due on April 20, 2026.\n\nServices: Web development — March 2026\nAmount: ₹12,500\n\nPlease process at your earliest convenience.\n\nThanks,\nDevStudio"),
    "🏫 Class update":       ("ham",  "Subject: CS301 — Lab session rescheduled\n\nDear Students,\n\nThe CS301 lab session on Friday April 12 has been moved to Saturday April 13 at 10:00 AM in Lab 204. Bring your laptops with required software. Attendance is mandatory.\n\nRegards,\nDr. Anand Krishnan\nDepartment of Computer Science"),
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'subject:\s*', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

st.markdown("""
<style>
    /* ── Full dark background ── */
    .stApp, .stApp > div, [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d !important;
    }
    [data-testid="stMain"], .main, section.main {
        background-color: #0d0d0d !important;
    }

    /* ── Header bar ── */
    header[data-testid="stHeader"] {
        background-color: #111111 !important;
        border-bottom: 1px solid #222222 !important;
    }
    header[data-testid="stHeader"] * { color: #e0def4 !important; }
    button[data-testid="baseButton-header"] {
        background-color: #378ADD !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222222 !important;
    }
    [data-testid="stSidebar"] * { color: #b0aec8 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e0def4 !important; }
    [data-testid="stSidebar"] hr { border-color: #222222 !important; }

    .sidebar-label {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #555577 !important; margin-bottom: 0.4rem;
    }
    .sidebar-desc {
        font-size: 0.8rem; color: #7a7890 !important;
        line-height: 1.5; margin-bottom: 0.8rem;
    }
    .spam-section {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: #e24b4a !important; margin: 0.6rem 0 0.3rem;
    }
    .ham-section {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: #1D9E75 !important; margin: 0.6rem 0 0.3rem;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #1a1a1a !important;
        color: #b0aec8 !important;
        border: 0.5px solid #2a2a2a !important;
        border-left: 2px solid transparent !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 4px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #222233 !important;
        border-left-color: #378ADD !important;
        color: #e0def4 !important;
    }

    /* ── Main content area ── */
    .block-container {
        padding-top: 2rem;
        max-width: 820px;
        background-color: #0d0d0d !important;
    }

    /* All text in main area */
    .stApp p, .stApp label, .stApp span,
    .stApp div, .stApp li { color: #c0bedd !important; }
    h1, h2, h3, h4 { color: #e0def4 !important; }

    /* ── Title bar ── */
    .title-bar {
        border-left: 4px solid #378ADD;
        padding-left: 14px;
        margin-bottom: 1rem;
        border-radius: 0;
    }
    .title-bar h1 { margin: 0; font-size: 1.7rem; color: #e0def4 !important; }
    .title-bar p  { margin: 0.2rem 0 0; color: #7a7890 !important; font-size: 0.88rem; }

    /* ── Text area ── */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1.5px solid #2a3a4a !important;
        background-color: #1a1a1a !important;
        font-size: 0.9rem !important;
        padding: 0.8rem !important;
        color: #e0def4 !important;
    }
    .stTextArea textarea:focus {
        border-color: #378ADD !important;
        box-shadow: 0 0 0 3px #378ADD22 !important;
    }
    .stTextArea textarea::placeholder { color: #555577 !important; }

    /* ── Predict button ── */
    .predict-btn > button {
        background: #378ADD !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        width: 100% !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .predict-btn > button:hover {
        background: #185FA5 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(55,138,221,0.4) !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #1a1a1a !important;
        border-radius: 10px !important;
        border: 0.5px solid #2a2a2a !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricValue"] { color: #e0def4 !important; }
    [data-testid="stMetricLabel"] { color: #7a7890 !important; }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border-width: 1.5px !important;
        background-color: #1a1a1a !important;
    }

    /* ── Divider ── */
    hr { border-color: #222222 !important; }

    /* ── Bar chart ── */
    [data-testid="stVegaLiteChart"] {
        background: #1a1a1a !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] * { color: #378ADD !important; }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #378ADD; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📧 Spam Classifier")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">🧪 Try a Sample</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Pick any email to auto-fill the text area and run a prediction.</div>', unsafe_allow_html=True)

    st.markdown('<div class="spam-section">🚫 Spam emails</div>', unsafe_allow_html=True)
    for name, (kind, body) in SAMPLES.items():
        if kind == "spam":
            if st.button(name, key=f"btn_{name}", use_container_width=True):
                st.session_state["email_input"] = body

    st.markdown('<div class="ham-section">✅ Not spam emails</div>', unsafe_allow_html=True)
    for name, (kind, body) in SAMPLES.items():
        if kind == "ham":
            if st.button(name, key=f"btn_{name}", use_container_width=True):
                st.session_state["email_input"] = body

    st.markdown("---")
    st.markdown('<div class="sidebar-label">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Uses a trained ML model with TF-IDF vectorization to detect spam emails.</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────
st.markdown("""
    <div class="title-bar">
        <h1>📧 Email Spam Classifier</h1>
        <p>Paste any email below to instantly check if it's spam or not.</p>
    </div>
""", unsafe_allow_html=True)

if "email_input" not in st.session_state:
    st.session_state["email_input"] = ""

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("Model loaded successfully ✅")

    user_input = st.text_area(
        "Enter email text",
        value=st.session_state["email_input"],
        height=220,
        placeholder="Paste your email content here, or pick a sample from the left...",
    )

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some email text first.")
        else:
            with st.spinner("Analysing email..."):
                cleaned = clean_text(user_input)
                transformed = vectorizer.transform([cleaned])
                prediction = model.predict(transformed)[0]
                proba = model.predict_proba(transformed)[0]
                not_spam_prob = float(proba[0])
                spam_prob = float(proba[1])

            st.markdown("---")

            if prediction == 1:
                st.error("🚫 **Spam Detected** — This email looks suspicious.")
            else:
                st.success("✅ **Not Spam** — This email looks legitimate.")

            st.markdown("#### 📊 Confidence Scores")
            col1, col2 = st.columns(2)
            col1.metric("✅ Not Spam", f"{not_spam_prob * 100:.1f}%")
            col2.metric("🚫 Spam", f"{spam_prob * 100:.1f}%")

            chart_data = pd.DataFrame(
                {"Probability": [not_spam_prob, spam_prob]},
                index=["Not Spam", "Spam"]
            )
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"❌ Error loading model: {e}")