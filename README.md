# 🎓 Campus Complaint Intelligence

An NLP-based Machine Learning project that automatically analyzes student complaints and predicts their **category** and **severity**.

## 🚀 Live Demo

The project is deployed using Streamlit Community Cloud.

## 🎯 Objective

The goal of this project is to reduce the manual effort involved in handling campus complaints by automatically classifying complaints based on their content.

## 📊 Dataset

The dataset contains **332 total complaints**:

- 273 training complaints
- 59 testing complaints
- 12 original features

The complaint data covers different campus-related issues reported by students.

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Missing-value checking
- Duplicate/overlap checking
- Text normalization
- Lowercase conversion
- Punctuation removal
- Stopword removal
- Text cleaning
- TF-IDF feature extraction

## 🤖 Machine Learning Models

### Category Classification

Different models were evaluated.

The best-performing model was:

**Linear SVM**

Accuracy:

**94.55%**

Categories:

- Academic
- Administrative
- Finance
- Infrastructure
- Technical

### Severity Classification

Different approaches were evaluated.

The best-performing model was:

**Logistic Regression**

Accuracy:

**72.73%**

Severity levels:

- Low
- Medium
- High
- Urgent

## 🔄 Machine Learning Pipeline

```text
Student Complaint
       ↓
Text Cleaning
       ↓
Stopword Removal
       ↓
TF-IDF Vectorization
       ↓
Machine Learning Model
       ↓
Category + Severity

## 🖥️ Application

The Streamlit application allows a user to enter a campus complaint and receive:

- Predicted category
- Predicted severity
- Severity confidence

## 📁 Project Structure

```text
campus-complaint-intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── complaints_raw.xlsx.numbers
│
├── models/
│   ├── category_model.pkl
│   ├── category_tfidf.pkl
│   ├── severity_model.pkl
│   └── severity_tfidf.pkl
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── requirements.txt
│
└── README.md

📈 Results
| Task                    | Model               |   Accuracy |
| ----------------------- | ------------------- | ---------: |
| Category Classification | Linear SVM          | **94.55%** |
| Severity Classification | Logistic Regression | **72.73%** |

⚠️ Limitations

The dataset is relatively small, with only 273 training complaints. Severity prediction is more difficult because the distinction between severity levels is sometimes subjective.

The model therefore provides an automated prediction rather than a guaranteed assessment.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Linear SVM
Streamlit
Joblib
Google Colab
GitHub

👩‍💻 Author
Manashvini Dave


