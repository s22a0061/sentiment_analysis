import re
import nltk
import spacy
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_models():
    # 1. NLTK Setup
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt")
        
    # 2. Emotion Model
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

    # 3. Sentiment Model (for Sarcasm Logic)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # 4. SpaCy Model (Loaded via requirements.txt)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback if the link fails, though unlikely
        import en_core_web_sm
        nlp = en_core_web_sm.load()

    return emotion_model, sentiment_model, nlp

# Initialize
emotion_classifier, sentiment_classifier, nlp = load_models()

# --- Utilities ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()

def get_emotions(text):
    if not text:
        return "neutral"
    # Truncate to 512 tokens to prevent model crash on long text
    result = emotion_classifier(text[:512])
    return result[0]["label"].lower()

def get_sarcasm(text):
    """
    Sarcasm Logic:
    If text is Positively phrased but contains Negative emotions (Anger/Disgust),
    it is likely Sarcastic.
    """
    if not text:
        return "Normal"

    sentiment = sentiment_classifier(text[:512])[0]["label"]
    emotion = get_emotions(text)

    # If it sounds positive but the emotion is angry/sad/disgust -> Sarcasm
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
