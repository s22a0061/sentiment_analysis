import re
import nltk
import spacy
import streamlit as st
from transformers import pipeline

# ---------------- Load Models Once ----------------
@st.cache_resource
def load_models():
    nltk.download("punkt")

    # Emotion model (cloud stable)
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

    # Sentiment model (used for sarcasm proxy)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # spaCy model (pre-installed via requirements.txt)
    nlp = spacy.load("en_core_web_sm")

    return emotion_model, sentiment_model, nlp


emotion_classifier, sentiment_classifier, nlp = load_models()

# ---------------- Utilities ----------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()


def get_emotions(text):
    if not text:
        return "neutral"
    result = emotion_classifier(text[:512])
    return result[0]["label"].lower()


def get_sarcasm(text):
    """
    Sarcasm proxy:
    Positive sentiment + negative emotion = sarcasm
    """
    if not text:
        return "Normal"

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
            descriptors = [
                child.text.lower()
                for child in token.children
                if child.pos_ == "ADJ"
            ]
            if descriptors:
                aspects.append((token.text.lower(), descriptors[0]))

    return aspects
