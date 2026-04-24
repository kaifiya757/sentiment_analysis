import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Predict
def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()
    
    return prediction, confidence
# ---------------- UI ---------------- #

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.markdown("Analyze whether a review is **Positive 😊** or **Negative 😡**")

# Input box
user_input = st.text_area("✍️ Enter your review here:")

# Button
result, confidence = predict(user_input)

if result == "Positive":
    st.success(f"😊 Positive Sentiment")
else:
    st.error(f"😡 Negative Sentiment")

st.info(f"Confidence Score: {round(confidence*100, 2)}%")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using NLP + Streamlit")