import re
import nltk
import spacy
import streamlit as st
from transformers import pipeline

# ---------------- Load Models Once ----------------
@st.cache_resource
def load_models():
    nltk.download("punkt")

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

    sarcasm_model = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sarcasm"
    )

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    return emotion_model, sarcasm_model, nlp


emotion_classifier, sarcasm_classifier, nlp = load_models()

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
    if not text:
        return "Normal"
    result = sarcasm_classifier(text[:512])
    return "Sarcastic" if result[0]["label"] == "sarcasm" else "Normal"


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
