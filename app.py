import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    # Base URL for your repo
    base_url = "https://raw.githubusercontent.com/nurulaina02/melis-nlp/main/"
    
    try:
        # 1. Load Twitter (The new cleaned file)
        twitter_df = pd.read_csv(base_url + "twitter_cleaned.csv")
        twitter_df['source'] = "Twitter" # Tag it for charts later
        
        # 2. Load Reviews
        reviews_df = pd.read_csv(base_url + "reviews_cleaned.csv")
        reviews_df['source'] = "Amazon"
        
        # 3. Combine
        full_df = pd.concat([twitter_df, reviews_df], ignore_index=True)
        return full_df
        
    except Exception as e:
        st.error(f"Error loading data. Check your filenames on GitHub! Error: {e}")
        return pd.DataFrame()

df = load_data()

# ================== DATA PRE-PROCESSING ==================
if not df.empty:
    # Ensure all text is string and lower case
    df = df.dropna(subset=["clean_text", "sentiment"])
    df["clean_text"] = df["clean_text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    
    # Filter only valid labels
    df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Controls")
st.sidebar.info("Model: TF-IDF + Logistic Regression")

# Debug info for your presentation (Show that data is real)
if st.sidebar.checkbox("Show Data Debug Info"):
    st.sidebar.write(df['sentiment'].value_counts())

# ================== MAIN DASHBOARD ==================
st.title("📊 Multi-Domain Sentiment Dashboard")
st.markdown("Analyzing **Social Media (Twitter)** vs **Product Reviews (Amazon)**")

# --- ROW 1: METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Points", len(df))
col2.metric("Positive Samples", len(df[df['sentiment']=='positive']))
col3.metric("Neutral Samples", len(df[df['sentiment']=='neutral']))
col4.metric("Negative Samples", len(df[df['sentiment']=='negative']))

st.divider()

# --- ROW 2: CHARTS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("⚖️ Class Distribution")
    # This chart proves your data is now balanced!
    fig_bal = px.bar(
        df['sentiment'].value_counts().reset_index(),
        x='sentiment', y='count',
        color='sentiment',
        title="Balanced Dataset for Training",
        color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"}
    )
    st.plotly_chart(fig_bal, use_container_width=True)

with c2:
    st.subheader("🌍 Data Sources")
    # Shows you are using both files
    fig_src = px.pie(
        df, names='source', 
        title="Composition: Twitter vs Amazon",
        hole=0.4
    )
    st.plotly_chart(fig_src, use_container_width=True)

# ================== MODEL ENGINE ==================
# We train this live because it's fast (Logistic Regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english')),
    ("clf", LogisticRegression(class_weight='balanced', max_iter=1000))
])

X = df["clean_text"]
y = df["sentiment"]
model.fit(X, y)
acc = model.score(X, y)

# ================== INTERACTIVE DEMO ==================
st.divider()
st.subheader("🧠 Artificial Intelligence Demo")

# Two columns: Input and Result
i1, i2 = st.columns([2, 1])

with i1:
    user_input = st.text_area("Enter text to analyze:", placeholder="e.g., I bought this phone but the screen is cracked.")
    
    # Preset examples for your Video Presentation
    st.caption("Try these examples:")
    if st.button("Example 1 (Positive)"): 
        user_input = "I absolutely love this product! Best purchase ever."
    if st.button("Example 2 (Negative)"): 
        user_input = "This is the worst service I have ever received. Terrible."
    if st.button("Example 3 (Neutral)"): 
        user_input = "It arrived on time but the packaging was okay."

with i2:
    st.write("### Prediction")
    if user_input:
        # Prediction Logic
        prediction = model.predict([user_input])[0]
        # Probability (Confidence)
        probs = model.predict_proba([user_input])[0]
        confidence = np.max(probs)
        
        # Display Result
        if prediction == "positive":
            st.success(f"**POSITIVE** ({confidence:.2%})")
        elif prediction == "negative":
            st.error(f"**NEGATIVE** ({confidence:.2%})")
        else:
            st.warning(f"**NEUTRAL** ({confidence:.2%})")
            
        st.progress(float(confidence))
    else:
        st.info("Waiting for input...")

# ================== TECHNICAL REPORT ==================
with st.expander("Show Technical Performance Report"):
    st.write(f"**Overall Model Accuracy:** {acc:.2%}")
    report = classification_report(y, model.predict(X), output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
