import streamlit as st
import joblib
import re

# Load trained models
category_vectorizer = joblib.load("models/category_tfidf.pkl")
category_model = joblib.load("models/category_model.pkl")

severity_vectorizer = joblib.load("models/severity_tfidf.pkl")
severity_model = joblib.load("models/severity_model.pkl")


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Prediction
def predict_complaint(complaint):

    processed = clean_text(complaint)

    category_features = category_vectorizer.transform(
        [processed]
    )

    severity_features = severity_vectorizer.transform(
        [processed]
    )

    category = category_model.predict(
        category_features
    )[0]

    severity = severity_model.predict(
        severity_features
    )[0]

    severity_probabilities = severity_model.predict_proba(
        severity_features
    )[0]

    confidence = severity_probabilities.max()

    return category, severity, confidence


# Streamlit UI
st.set_page_config(
    page_title="Campus Complaint Intelligence",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Campus Complaint Intelligence")

st.write(
    "Enter a campus complaint and the ML model will "
    "predict its category and severity."
)

complaint = st.text_area(
    "Enter your complaint:",
    placeholder="Example: WiFi is not working properly in the CSE lab."
)

if st.button("Analyze Complaint"):

    if complaint.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        category, severity, confidence = predict_complaint(
            complaint
        )

        st.subheader("Prediction")

        st.write("**Category:**", category)
        st.write("**Severity:**", severity)
        st.write(
            "**Severity Confidence:**",
            f"{confidence * 100:.2f}%"
        )
