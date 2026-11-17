import streamlit as st
import joblib

# Set page layout and design
st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load the pre-trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("spam_classifier_pipeline.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'spam_classifier_pipeline.pkl' not found! Please run the training code first.")
    st.stop()

# App Header & Styling
st.title("🛡️ AI-Based Email Spam Detection System")
st.markdown("""
Welcome to the AI Spam Classifier. Enter an email's text below to analyze 
whether it is **Spam** or **Legitimate (Ham)** using our trained Natural Language Processing model.
""")

# Quick Sample buttons to help test the system
col1, col2 = st.columns(2)
sample_spam = "URGENT! You have won a $1,000 Walmart gift card. Click here to claim your prize now before it expires! Call 0800-555-0199."
sample_ham = "Hi team, please find attached the minutes from yesterday's project meeting. Let me know if you have any edits before Friday."

if "email_input" not in st.session_state:
    st.session_state.email_input = ""

with col1:
    if st.button("📋 Load Sample Spam"):
        st.session_state.email_input = sample_spam
with col2:
    if st.button("📋 Load Sample Ham"):
        st.session_state.email_input = sample_ham

# User Input Text Area
user_input = st.text_area(
    "Email Content:",
    value=st.session_state.email_input,
    height=180,
    placeholder="Paste email subject or body here..."
)

# Classification logic
if st.button("Analyze Email", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Perform prediction
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        confidence = probabilities[prediction] * 100
        
        st.subheader("Analysis Results:")
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.2f}%)")
            st.write("This message exhibits critical markers typical of phishing, advertising scams, or unsolicited bulk messages.")
        else:
            st.success(f"✅ **LEGITIMATE (HAM)** (Confidence: {confidence:.2f}%)")
            st.write("This message appears clean and matches safe, standard communication patterns.")
