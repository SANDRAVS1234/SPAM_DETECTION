import streamlit as st
import pickle
import re

# Load the trained model
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Add title
st.title("Spam Detection App")

# Input text
user_input = st.text_area("Enter a message to check if it's spam:")

# Preprocess text (optional: match your notebook's preprocessing)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

if st.button("Predict"):
    if user_input:
        processed_text = preprocess_text(user_input)
        prediction = model.predict([processed_text])
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.success(f"Prediction: {result}"
        )
        
    else:
        st.warning("Please enter a message.")
