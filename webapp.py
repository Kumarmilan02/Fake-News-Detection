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

# Load the trained model and vectorizer
def load_model_and_vectorizer():
    try:
        with open(r'C:/Users/KIIT/Downloads/model1.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open(r'C:/Users/KIIT/Downloads/vectorizer1.pkl', 'rb') as vectorizer_file:
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
    
    text_cleaned = re.sub(r'\W', ' ', text)
    text_cleaned = text_cleaned.lower()
    text_cleaned = ' '.join([PorterStemmer().stem(word) for word in text_cleaned.split() if word not in stopwords.words('english')])
    
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return 'FAKE' if prediction[0] == 0 else 'REAL'

# Set up the Streamlit app
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="C:/Users/KIIT/Downloads/logo.jpg",
    layout="wide"
)

# Apply custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        color: #333;
    }
    .title {
        color: #1f77b4;
        font-size: 2.5em;
    }
    .input-section {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .predict-btn {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 1.2em;
    }
    .predict-btn:hover {
        background-color: #155a8a;
    }
    .output {
        font-size: 1.5em;
    }
    </style>
    """, unsafe_allow_html=True)

st.write("<h1 class='title'>Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='input-section'>
    <h2>Input:</h2>
    <p>Enter your text below to predict whether it's fake or real news.</p>
    </div>
    """, unsafe_allow_html=True
)

text = st.text_area(
    label="Enter your text to try it.",
    placeholder="Enter your text to predict whether this is fake or not.",
    height=200
)

st.write(f'You wrote {len(text.split())} words.')

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Predict button
if st.button("Predict", key='predict-btn'):
    st.markdown("<div class='output'>## Output:</div>", unsafe_allow_html=True)
    result = predict_news(text, model, vectorizer)
    if result == "REAL":
        st.markdown("<div class='output'>#### Looking Real News 📰</div>", unsafe_allow_html=True)
    elif result == "FAKE":
        st.markdown("<div class='output'>#### Looking Fake News ⚠️📰</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='output'>{result}</div>", unsafe_allow_html=True)
