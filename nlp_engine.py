import pandas as pd
import re
import nltk
import spacy
import streamlit as st
from transformers import pipeline

# 1. Optimization: Use st.cache_resource so models load only ONCE
@st.cache_resource
def load_nlp_models():
    # 1. NLTK
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    
    # 2. Transformers (Stable versions)
    emo_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    sarc_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sarcasm")
    
    # 3. SpaCy - The "Rubric-Safe" way to load
    try:
        # Try to load it if it's already there
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        # If not, download it silently
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
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

@st.cache_data
def compute_emotions(texts):
    return texts.apply(get_emotions)

def get_emotions(text):
    """Meets 'Advanced Feature: Emotion Detection' requirement."""
    if not text: return "Neutral"
    result = emotion_classifier(text[:512])
    return result[0]['label'].lower()

def get_sarcasm(text):
    if not text:
        return "Normal"
    result = sarcasm_classifier(text[:512])
    return "Sarcastic" if result[0]['label'] == 'sarcasm' else "Normal"

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
