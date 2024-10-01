# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:30:02 2024

@author: 2105208
"""

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import is_classifier
import nltk

# Set up the Streamlit app
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Ensure NLTK stopwords are downloaded
def download_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        st.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        st.success("NLTK stopwords downloaded successfully!")

# Call the function to download resources if not already available
download_nltk_resources()

# Load the trained model and vectorizer
def load_model_and_vectorizer():
    try:
        with open(r'model/model1.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open(r'model/vectorizer1.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        if not is_classifier(model):
            raise TypeError("Loaded model is not a classifier.")
        if not isinstance(vectorizer, TfidfVectorizer):
            raise TypeError("Loaded vectorizer is not a TfidfVectorizer.")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# Preprocess and predict
def predict_news(text, model, vectorizer):
    if model is None or vectorizer is None:
        return "Error: Model or vectorizer not loaded."
    
    # Obtain additional stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    text_cleaned = re.sub(r'\W', ' ', text)
    text_cleaned = text_cleaned.lower()
    text_cleaned = ' '.join([PorterStemmer().stem(word) for word in text_cleaned.split() if word not in stop_words])
    
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return 'FAKE' if prediction[0] == 0 else 'REAL'

# Apply custom CSS styling
st.markdown("""
    <style>
    :root {
        --background-light: #87CEFA; /* Light Sky Blue */
        --background-dark: #1e1e1e;
        --text-light: #333;
        --text-dark: #ffffff;
        --primary-light: #3e7b99;
        --primary-dark: #2a4d61;
        --input-background: #ffffff;
        --input-background-dark: #2c2c2c;
        --output-success: #d4edda;
        --output-danger: #f8d7da;
    }

    body {
        background-color: var(--background-light);
        color: var(--text-light);
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .streamlit-expanderHeader {
        color: var(--text-light);
    }

    .title {
        color: var(--primary-light);
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
    }
    .input-section {
        background-color: var(--input-background);
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-top: 30px;
    }
    .predict-btn {
        background-color: var(--primary-light);
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-size: 1.2em;
        width: 100%;
        transition: background-color 0.3s ease;
        border: none;
        cursor: pointer;
    }
    .predict-btn:hover {
        background-color: var(--primary-dark);
    }
    .output {
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
    }
    .output-icon {
        font-size: 2em;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.write("<h1 class='title'>📰 Fake News Detection</h1>", unsafe_allow_html=True)

# Input Section
st.markdown("""
    <div class='input-section'>
    <h2>📝 Input:</h2>
    <p>Enter your text below to predict whether it's fake or real news.</p>
    """, unsafe_allow_html=True)

# Text Area for User Input
text = st.text_area(
    label="Enter your text to try it.",
    placeholder="Enter your text to predict whether this is fake or not.",
    height=200
)

st.write(f'You wrote {len(text.split())} words.')

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Predict Button
if st.button("Predict", key='predict-btn'):
    st.markdown("<div class='output'>## Output:</div>", unsafe_allow_html=True)
    result = predict_news(text, model, vectorizer)
    
    # Display Prediction Result
    if result == "REAL":
        st.markdown("<div class='output'><span class='output-icon'>📰✅</span> Looking Real News 📰</div>", unsafe_allow_html=True)
    elif result == "FAKE":
        st.markdown("<div class='output'><span class='output-icon'>⚠️📰</span> Looking Fake News ⚠️📰</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='output'>{result}</div>", unsafe_allow_html=True)
