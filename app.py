import streamlit as st
import pickle

st.title("📧 Email Spam Classifier")

# Try loading model
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    
    st.success("Model loaded successfully ✅")

    user_input = st.text_area("Enter email text")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)[0]
            proba = model.predict_proba(transformed)

            if prediction == 1:
                st.error("🚫 Spam")
            else:
                st.success("✅ Not Spam")

            st.subheader("Confidence Score")
            st.bar_chart(proba)

except Exception as e:
    st.error(f"Error loading model: {e}")