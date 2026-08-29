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
