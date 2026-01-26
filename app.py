import streamlit as st
import pandas as pd
import plotly.express as px
import os
from nlp_engine import clean_text, get_emotions, extract_aspects, get_sarcasm

st.set_page_config(page_title="NLP Sentiment Dashboard", layout="wide")

st.title("📊 Multi-Domain Sentiment & Emotion Intelligence")
st.markdown("Comparing **Product Reviews (Amazon)** vs **Social Media (Twitter)**")

# ---------------- Sidebar ----------------
st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Amazon Products", "Twitter Social"]
)

# ---------------- Load Data ----------------
# This logic checks if the file actually exists before crashing
try:
    if dataset_choice == "Amazon Products":
        file_path = "data/7817_1.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_col = "reviews.text"
        else:
            st.error(f"File not found: {file_path}. Is it in the 'data' folder?")
            st.stop()
    else:
        file_path = "data/twitter_sentiment_small.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            text_col = "text"
        else:
            st.error(f"File not found: {file_path}. Is it in the 'data' folder?")
            st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sample for performance (Important for Free Cloud Tier)
df = df.sample(min(500, len(df)), random_state=42)

# ---------------- Preprocessing ----------------
st.write(f"Processing {len(df)} samples...")
df["clean_text"] = df[text_col].apply(clean_text)

# We compute emotions here so we can use it in charts later
# Using a spinner so the user knows something is happening
with st.spinner("Analyzing emotions..."):
    df["emotion"] = df["clean_text"].apply(get_emotions)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(
    ["Sentiment Overview", "Emotion & Sarcasm", "Aspect Explorer"]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Sentiment Distribution")
    if dataset_choice == "Amazon Products":
        fig = px.histogram(
            df, x="reviews.rating", 
            title="Amazon Rating Distribution (1–5 Stars)",
            color_discrete_sequence=['#FF4B4B']
        )
    else:
        fig = px.pie(
            df, names="target", 
            title="Twitter Polarity (0 = Negative, 4 = Positive)"
        )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Emotion & Sarcasm Analysis")
    emo_counts = df["emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]
    
    fig_emo = px.bar(
        emo_counts, x="Emotion", y="Count", 
        title="Detected Emotions", color="Emotion"
    )
    st.plotly_chart(fig_emo, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Aspect-Based Feature Extraction")
    all_aspects = []
    for text in df["clean_text"]:
        all_aspects.extend(extract_aspects(text))

    if all_aspects:
        aspect_df = pd.DataFrame(all_aspects, columns=["Feature", "Descriptor"])
        top_aspects = aspect_df["Feature"].value_counts().head(10).reset_index()
        top_aspects.columns = ["Feature", "Mentions"]

        fig_aspect = px.bar(
            top_aspects, x="Feature", y="Mentions",
            title="Top 10 Mentioned Features"
        )
        st.plotly_chart(fig_aspect, use_container_width=True)
        st.dataframe(aspect_df.head(10))
    else:
        st.info("No aspects detected in this sample.")

# ---------------- Single Text Tester ----------------
st.divider()
st.subheader("Test a Sentence")
user_input = st.text_input("Enter text (e.g., 'I loved the phone but the battery is dead'):")

if user_input:
    emo = get_emotions(user_input)
    sarc = get_sarcasm(user_input)
    asp = extract_aspects(user_input)

    col1, col2, col3 = st.columns(3)
    col1.metric("Emotion", emo)
    col2.metric("Sarcasm Check", sarc)
    col3.write("**Detected Aspects:**")
    col3.write(asp)
