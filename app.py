import streamlit as st
import pandas as pd
import plotly.express as px
from nlp_engine import clean_text, get_emotions, extract_aspects, get_sarcasm

st.set_page_config(page_title="NLP Sentiment Dashboard", layout="wide")

st.title("📊 Multi-Domain Sentiment & Emotion Intelligence")
st.markdown("Comparing **Product Reviews** (Amazon) vs. **Social Media** (Twitter)")

# --- Sidebar ---
st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Amazon Products", "Twitter Social"])

# Load Data
try:
    if dataset_choice == "Amazon Products":
        df = pd.read_csv("data/7817_1.csv") 
        text_col, date_col = "reviews.text", "reviews.date"
    else:
        df = pd.read_csv("data/twitter_sentiment_small.csv")
        text_col, date_col = "text", "date"
except FileNotFoundError:
    st.error("CSV files not found in the 'data' folder. Please check your GitHub structure.")
    st.stop()

# Sample data for speed in demo
df = df.sample(min(500, len(df))) 

# --- Process Data ---
st.write(f"Analyzing {len(df)} samples...")
df['clean_text'] = df[text_col].apply(clean_text)

# --- Layout Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Sentiment Trends", "🎭 Emotion & Sarcasm", "🔍 Aspect Explorer"])

with tab1:
    st.subheader("Sentiment Distribution")
    # If Amazon, we use ratings. If Twitter, we use the 'target' column.
    if dataset_choice == "Amazon Products":
        fig = px.histogram(df, x="reviews.rating", title="Rating Distribution (1-5 Stars)", color_discrete_sequence=['#FF4B4B'])
    else:
        fig = px.pie(df, names="target", title="Twitter Polarity (0=Neg, 4=Pos)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Deep Emotion Analysis")
    # Apply emotion detection (this might take a second)
    with st.spinner('Detecting emotions...'):
        df['emotion'] = df['clean_text'].apply(get_emotions)
    
    emo_counts = df['emotion'].value_counts().reset_index()
    fig_emo = px.bar(emo_counts, x='emotion', y='count', color='emotion', title="Dominant Emotions Found")
    st.plotly_chart(fig_emo, use_container_width=True)

with tab3:
    st.subheader("Aspect-Based Analysis (Service/Product Features)")
    all_aspects = []
    for text in df['clean_text']:
        all_aspects.extend(extract_aspects(text))
    
    aspect_df = pd.DataFrame(all_aspects, columns=['Feature', 'Sentiment'])
    top_aspects = aspect_df['Feature'].value_counts().head(10).reset_index()
    
    fig_aspect = px.bar(top_aspects, x='Feature', y='count', title="Top 10 Mentioned Features")
    st.plotly_chart(fig_aspect, use_container_width=True)
    
    st.write("Recent Feature Mentions:")
    st.dataframe(aspect_df.head(10))

# --- Single Text Tester ---
st.divider()
st.subheader("Test Your Own Text")
user_input = st.text_input("Enter a sentence to analyze:")
if user_input:
    emo = get_emotions(user_input)
    asp = extract_aspects(user_input)
    sarc = get_sarcasm(user_input)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Emotion", emo)
    col2.metric("Style", sarc) # Shows "Sarcastic" or "Normal"
    col3.write("**Detected Aspects:**")
    col3.write(asp)

# Add this inside your app.py to satisfy the "Measurement" requirement 
with st.expander("📊 Technical Performance Metrics"):
    if dataset_choice == "Twitter Social":
        # Let's see how often our emotion model aligns with the 'target' labels
        # target 0 = Negative, target 4 = Positive
        st.write("Model Accuracy vs Ground Truth Labels")
        
        # Example calculation logic
        correct_predictions = df[df['emotion'].isin(['sadness', 'anger']) & (df['target'] == 0)].shape[0]
        total_negatives = df[df['target'] == 0].shape[0]
        
        if total_negatives > 0:
            accuracy = (correct_predictions / total_negatives) * 100
            st.metric("Negative Sentiment Recall", f"{accuracy:.2f}%")
            st.caption("This measures how well the model identifies actual negative tweets.")
