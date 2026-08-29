import streamlit as st
import joblib
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Campus Complaint Intelligence",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    category_vectorizer = joblib.load(
        "models/category_tfidf.pkl"
    )

    category_model = joblib.load(
        "models/category_model.pkl"
    )

    severity_vectorizer = joblib.load(
        "models/severity_tfidf.pkl"
    )

    severity_model = joblib.load(
        "models/severity_model.pkl"
    )

    return (
        category_vectorizer,
        category_model,
        severity_vectorizer,
        severity_model
    )


(
    category_vectorizer,
    category_model,
    severity_vectorizer,
    severity_model
) = load_models()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

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

    probabilities = severity_model.predict_proba(
        severity_features
    )[0]

    confidence = probabilities.max()

    return category, severity, confidence


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Campus Complaint Intelligence")

st.markdown(
    """
    **An NLP-based Machine Learning system for automatically
    classifying campus complaints by category and severity.**
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        """
        This system uses Natural Language Processing (NLP)
        and Machine Learning to analyze student complaints.
        """
    )

    st.subheader("🤖 Models Used")

    st.write("**Category:** Linear SVM")
    st.write("**Severity:** Logistic Regression")

    st.subheader("📊 Model Performance")

    st.metric(
        "Category Accuracy",
        "94.55%"
    )

    st.metric(
        "Severity Accuracy",
        "72.73%"
    )

    st.divider()

    st.write(
        "Dataset: 273 training complaints"
    )

    st.write(
        "Categories: 5"
    )

    st.write(
        "Severity Levels: 4"
    )


# =========================================================
# MAIN INPUT
# =========================================================

st.subheader("📝 Submit a Complaint")

st.write(
    "Describe your campus-related issue below."
)

complaint = st.text_area(
    "Complaint",
    placeholder=(
        "Example: The WiFi is not working properly "
        "in the CSE computer lab."
    ),
    height=150
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Complaint",
    use_container_width=True
):

    if not complaint.strip():

        st.warning(
            "Please enter a complaint before analyzing."
        )

    else:

        category, severity, confidence = predict_complaint(
            complaint
        )

        st.divider()

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Complaint Category",
                category
            )

        with col2:

            st.metric(
                "Predicted Severity",
                severity
            )

        with col3:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.progress(
            float(confidence)
        )

        if confidence < 0.50:

            st.info(
                "ℹ️ The model has relatively low confidence "
                "in this severity prediction. This is expected "
                "for some ambiguous complaints."
            )

        elif confidence < 0.75:

            st.info(
                "ℹ️ The model has moderate confidence "
                "in this severity prediction."
            )

        else:

            st.success(
                "✅ The model has high confidence "
                "in this severity prediction."
            )


# =========================================================
# EXAMPLES
# =========================================================

st.divider()

st.subheader("💡 Example Complaints")

examples = [
    "The WiFi is not working in the CSE lab.",
    "There is no drinking water on the sixth floor.",
    "My fee payment is not showing in the student portal.",
    "The projector in our classroom is unclear.",
    "The cafeteria should provide larger tea cups."
]

for example in examples:

    st.write("•", example)


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader("⚙️ How the System Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**1️⃣ Complaint**")
    st.write("Student enters a complaint.")

with col2:
    st.write("**2️⃣ Preprocessing**")
    st.write("Text is cleaned and processed.")

with col3:
    st.write("**3️⃣ TF-IDF**")
    st.write("Important words are converted into numerical features.")

with col4:
    st.write("**4️⃣ Prediction**")
    st.write("ML models predict category and severity.")
        )
