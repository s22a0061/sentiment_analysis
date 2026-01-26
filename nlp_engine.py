import pandas as pd
import re
import nltk
import spacy
from transformers import pipeline

# Load SpaCy for Aspect Extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize HuggingFace Pipelines (Lightweight models)
# These will download on first run
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
sarcasm_classifier = pipeline("text-classification", model="helinous/sarcasm-detector", top_k=1)

def clean_text(text):
    """Basic cleaning: remove URLs, Mentions, and special chars but keep emojis."""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove Mentions
    text = text.replace('\n', ' ')
    return text.strip()

def get_emotions(text):
    """Detects primary emotion: joy, anger, fear, etc."""
    result = emotion_classifier(text[:512]) # Truncate to 512 for BERT
    return result[0][0]['label']

def get_sarcasm(text):
    """Detects if text is sarcastic."""
    result = sarcasm_classifier(text[:512])
    # Assuming label 'LABEL_1' is sarcastic based on most sarcasm models
    return "Sarcastic" if result[0][0]['label'] == 'LABEL_1' else "Normal"

def extract_aspects(text):
    """
    Extracts Nouns (Aspects) and Adjectives (Sentiments).
    Example: 'The battery is great' -> ('battery', 'great')
    """
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ == "NOUN":
            # Look for adjectives connected to the noun
            adj = [child.text for child in token.children if child.pos_ == "ADJ"]
            if adj:
                aspects.append((token.text.lower(), adj[0].lower()))
    return aspects
