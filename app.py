import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    twitter_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/melis-nlp/main/twitter_cleaned.csv"
    )
    reviews_df = pd.read_csv(
        "https://raw.githubusercontent.com/nurulaina02/melis-nlp/main/reviews_cleaned.csv"
    )
    return pd.concat([twitter_df, reviews_df], ignore_index=True)


df = load_data()

# ================== DATA CLEANING (CRITICAL FIX) ==================
df = df.dropna(subset=["clean_text", "sentiment"])
df["clean_text"] = df["clean_text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower()
df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Dashboard Controls")
search_text = st.sidebar.text_input("🔍 Search text")

# ================== TITLE ==================
st.title("📊 Sentiment Analysis Dashboard")
st.caption("Interactive NLP Dashboard using TF-IDF and Logistic Regression")

# ================== CLASS BALANCE ==================
st.subheader("⚖️ Class Balance Distribution")

class_counts = df["sentiment"].value_counts().reset_index()
class_counts.columns = ["Sentiment", "Count"]

fig_balance = px.bar(
    class_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    title="Sentiment Class Distribution"
)

st.plotly_chart(fig_balance, use_container_width=True)

# ================== KPI METRICS ==================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Positive", (df["sentiment"] == "positive").sum())
col3.metric("Neutral", (df["sentiment"] == "neutral").sum())
col4.metric("Negative", (df["sentiment"] == "negative").sum())

# ================== MODEL TRAINING ==================
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression())
])

X = df["clean_text"]
y = df["sentiment"]

model.fit(X, y)
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

# ================== MODEL PERFORMANCE ==================
st.subheader("✅ Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy:.3f}")

with col2:
    st.info("Model: TF-IDF + Logistic Regression")

# ================== SENTIMENT DISTRIBUTION ==================
st.subheader("📊 Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig_sentiment = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    hole=0.4,
    title="Overall Sentiment Breakdown"
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# ================== DATA EXPLORER ==================
st.subheader("🔎 Explore Text Data")

if search_text:
    filtered_df = df[
        df["clean_text"].str.contains(search_text, case=False, na=False)
    ]
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.dataframe(df.head(50), use_container_width=True)

# ================== CLASSIFICATION REPORT (TABLE) ==================
st.subheader("📑 Classification Report")

report_dict = classification_report(
    y,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose().round(3)

st.dataframe(report_df, use_container_width=True)

# ================== LIVE PREDICTION ==================
st.subheader("✍️ Live Sentiment Prediction")

user_input = st.text_area("Enter a sentence to analyze sentiment:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.success(f"Predicted Sentiment: **{prediction.upper()}**")
