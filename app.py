import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import re

st.set_page_config(
    page_title="Djinni Job Classifier",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Djinni Job Classifier")
st.markdown("Paste a job description and the model will predict the job category.")

@st.cache_resource
def load_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/linear_svc.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return vectorizer, model, le

def clean_text(text):
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

try:
    vectorizer, model, le = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("⚠️ Models not found. Please train and save models first.")

text_input = st.text_area(
    "Job Description",
    height=300,
    placeholder="Paste job description here..."
)

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.error("Please enter a job description.")
    elif not model_loaded:
        st.error("Models not loaded. Run the notebook first.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        
        prediction = model.predict(vectorized)[0]
        category = le.inverse_transform([prediction])[0]
        
        # Get decision scores for top 5
        scores = model.decision_function(vectorized)[0]
        top5_idx = np.argsort(scores)[::-1][:5]
        top5_categories = le.inverse_transform(top5_idx)
        top5_scores = scores[top5_idx]
        
        # Normalize scores to 0-1
        top5_scores_norm = (top5_scores - top5_scores.min()) / (top5_scores.max() - top5_scores.min() + 1e-8)
        
        st.success(f"### Predicted Category: **{category}**")
        
        st.markdown("#### Top 5 Predictions")
        for cat, score in zip(top5_categories, top5_scores_norm):
            st.progress(float(score), text=f"{cat}: {score:.2f}")

st.markdown("---")
st.markdown(
    "**Dataset:** [lang-uk/recruitment-dataset-job-descriptions-ukrainian](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-ukrainian) "
    "| **Model:** TF-IDF + LinearSVC | **Accuracy:** 0.81"
)
