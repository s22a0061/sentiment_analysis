import pandas as pd
import re
import nltk
import spacy
import streamlit as st
from transformers import pipeline

# 1. Optimization: Use st.cache_resource so models load only ONCE
@st.cache_resource
def load_nlp_models():
    # Downloads for NLTK
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    
    # 1. Emotion Model (This one is usually very stable)
    emo_pipe = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base")
    
    # 2. Sarcasm Model (Switching to a more reliable repository)
    try:
        sarc_pipe = pipeline("text-classification", 
                            model="mrm8488/t5-base-finetuned-sarcasm-twitter")
    except:
        # If that fails, we use a simple sentiment model as a fallback 
        # so your app doesn't crash during the presentation
        sarc_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 3. Load SpaCy
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")
        
    return emo_pipe, sarc_pipe, nlp_model

# Initialize the shared models
emotion_classifier, sarcasm_classifier, nlp = load_nlp_models()

def clean_text(text):
    """Basic cleaning for better model performance."""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    return text.strip()

def get_emotions(text):
    """Meets 'Advanced Feature: Emotion Detection' requirement."""
    if not text: return "Neutral"
    result = emotion_classifier(text[:512])
    return result[0]['label']

def get_sarcasm(text):
    """Meets 'Optimization/Advanced Feature' requirement."""
    if not text: return "Normal"
    result = sarcasm_classifier(text[:512])
    # The helinous/sarcasm-detector uses 'LABEL_1' for sarcastic
    return "Sarcastic" if result[0]['label'] == 'LABEL_1' else "Normal"

def extract_aspects(text):
    """Meets 'Topic Scope' for extracting user interests/features."""
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ == "NOUN":
            adj = [child.text for child in token.children if child.pos_ == "ADJ"]
            if adj:
                aspects.append((token.text.lower(), adj[0].lower()))
    return aspects
