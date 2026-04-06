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

st.title("📧 Email Spam Classifier")

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("Model loaded successfully ✅")

    user_input = st.text_area("Enter email text")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            cleaned = clean_text(user_input)
            transformed = vectorizer.transform([cleaned])

            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.error("🚫 Spam")
            else:
                st.success("✅ Not Spam")

            # ✅ Works because SVC has probability=True
            proba = model.predict_proba(transformed)[0]
            not_spam_prob = float(proba[0])
            spam_prob = float(proba[1])

            st.subheader("Confidence Score")
            chart_data = pd.DataFrame(
                {"Probability": [not_spam_prob, spam_prob]},
                index=["Not Spam", "Spam"]
            )
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Error loading model: {e}")