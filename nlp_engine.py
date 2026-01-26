import re
import nltk
import spacy
import streamlit as st
import subprocess
import sys
from transformers import pipeline

@st.cache_resource
def load_models():
    # 1. NLTK Setup
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt")
        
    # 2. Transformers
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # 3. SpaCy - The "Force Fix" Method
    # If the model isn't found, we use a subprocess to install it inside the running app
    if not spacy.util.is_package("en_core_web_sm"):
        st.warning("Downloading language model... this takes 1 minute.")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    nlp = spacy.load("en_core_web_sm")

    return emotion_model, sentiment_model, nlp

# Initialize
emotion_classifier, sentiment_classifier, nlp = load_models()

# --- Utilities ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()

def get_emotions(text):
    if not text: return "neutral"
    return emotion_classifier(text[:512])[0]["label"].lower()

def get_sarcasm(text):
    if not text: return "Normal"
    sentiment = sentiment_classifier(text[:512])[0]["label"]
    emotion = get_emotions(text)
    if sentiment == "POSITIVE" and emotion in ["anger", "sadness", "disgust"]:
        return "Sarcastic"
    return "Normal"

def extract_aspects(text):
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ == "NOUN":
            descriptors = [child.text.lower() for child in token.children if child.pos_ == "ADJ"]
            if descriptors:
                aspects.append((token.text.lower(), descriptors[0]))
    return aspects
