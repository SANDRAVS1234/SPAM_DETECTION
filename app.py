import streamlit as st
import joblib
import re

# Load the vectorizer and model
vectorizer, model = joblib.load("spam_classifier.pkl")

st.title("Spam Detection App")

user_input = st.text_area("Enter a message to check if it's spam:")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if st.button("Predict"):
    if user_input:
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])  # Use vectorizer here
        prediction = model.predict(vectorized_text)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message.")
