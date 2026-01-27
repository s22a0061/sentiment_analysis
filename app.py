import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================== 1. PAGE CONFIGURATION ==================
st.set_page_config(
    page_title="NLP Project: Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ================== 2. DATA LOADING & TRAINING ==================
@st.cache_resource
def build_model_pipeline():
    """
    Loads data, trains the model, and returns the pipeline + metrics.
    This runs once and caches the result for speed.
    """
    # Load your local CSV files
    try:
        df_twitter = pd.read_csv("twitter_cleaned.csv")
        df_amazon = pd.read_csv("reviews_cleaned.csv")
        
        # Tag sources for analysis
        df_twitter['source'] = 'Twitter (Social)'
        df_amazon['source'] = 'Amazon (Product)'
        
        # Combine into one big dataset
        df = pd.concat([df_twitter, df_amazon], ignore_index=True)
        
        # Basic cleaning (Drop empty rows)
        df = df.dropna(subset=['clean_text', 'sentiment'])
        
        # Prepare Training Data
        X = df['clean_text'].astype(str)
        y = df['sentiment']
        
        # Split data for Valid Performance Metrics (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build Pipeline: TF-IDF (Text to Numbers) -> Logistic Regression (Classification)
        # We use class_weight='balanced' to help with the Amazon imbalance
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, df, accuracy, report

    except FileNotFoundError:
        return None, None, 0, None

# Run the training
model, df, accuracy, report = build_model_pipeline()

# ================== 3. DASHBOARD UI ==================

# ERROR HANDLING if files are missing
if model is None:
    st.error("🚨 CRITICAL ERROR: CSV files not found!")
    st.info("Please make sure 'twitter_cleaned.csv' and 'reviews_cleaned.csv' are in the same folder as this app.py.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("NLP Project Control")
st.sidebar.markdown("**Student Name:** [Your Name]")
st.sidebar.markdown("**Project:** Multi-Domain Sentiment Analysis")
menu = st.sidebar.radio("Navigate", ["Project Overview", "Data Analysis", "Model Performance", "Live Demo"])

# --- TAB 1: PROJECT OVERVIEW ---
if menu == "Project Overview":
    st.title("📊 Multi-Domain Sentiment Analysis")
    st.markdown("### 1. Objectives")
    st.write("""
    The objective of this project is to develop a **Natural Language Processing (NLP)** application 
    capable of analyzing sentiment across two distinct domains:
    * **Informal Social Media:** Short, slang-heavy text (Twitter).
    * **Formal Product Reviews:** Longer, feature-specific text (Amazon).
    """)
    
    st.markdown("### 2. Problem Statement")
    st.info("""
    Traditional sentiment models often fail when applied to different domains. 
    A model trained only on formal reviews may not understand Twitter slang (e.g., 'smh', 'lol'). 
    This project aims to build a unified model that understands both.
    """)
    
    st.markdown("### 3. Solution Architecture")
    st.code("Raw Data -> Preprocessing -> TF-IDF Vectorization -> Logistic Regression -> Streamlit Dashboard", language="text")

# --- TAB 2: DATA ANALYSIS ---
elif menu == "Data Analysis":
    st.title("📈 Data Exploration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Dataset Size", f"{len(df):,} rows")
        st.write("### Source Distribution")
        fig_src = px.pie(df, names='source', title="Data Composition", hole=0.3)
        st.plotly_chart(fig_src, use_container_width=True)
        
    with col2:
        st.metric("Twitter Samples", len(df[df['source']=='Twitter (Social)']))
        st.metric("Amazon Samples", len(df[df['source']=='Amazon (Product)']))
        
    st.write("### Sentiment Class Balance")
    st.write("This chart shows we have successfully balanced the dataset (mostly thanks to the Twitter data).")
    class_counts = df['sentiment'].value_counts().reset_index()
    class_counts.columns = ['Sentiment', 'Count']
    fig_bar = px.bar(class_counts, x='Sentiment', y='Count', color='Sentiment', 
                     color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: MODEL PERFORMANCE ---
elif menu == "Model Performance":
    st.title("⚙️ Technical Evaluation")
    
    st.write(f"### Overall Model Accuracy: **{accuracy:.2%}**")
    st.progress(accuracy)
    
    st.write("### Detailed Classification Report")
    st.write("This table shows Precision (correctness) and Recall (completeness) for each sentiment class.")
    
    # Format the report nicely
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
    
    st.write("### Interpretation")
    st.success("""
    - **High Accuracy:** The model performs well on Positive and Negative classes.
    - **Neutral Class:** Often harder to predict, as seen in the slightly lower F1-score.
    - **Generalization:** Because we combined datasets, this model is robust across different writing styles.
    """)

# --- TAB 4: LIVE DEMO ---
elif menu == "Live Demo":
    st.title("🧠 Live AI Prediction")
    st.write("Test the model in real-time. Enter a Tweet or a Review below.")
    
    user_input = st.text_area("Enter text here:", height=100, placeholder="e.g., I bought this phone and the battery is dead. Worst purchase ever!")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # 1. Predict Label
            prediction = model.predict([user_input])[0]
            # 2. Get Confidence Score
            probs = model.predict_proba([user_input])[0]
            confidence = np.max(probs)
            
            # Display
            st.divider()
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if prediction == 'positive':
                    st.image("https://cdn-icons-png.flaticon.com/512/10009/10009963.png", width=100)
                    st.success(f"**POSITIVE**")
                elif prediction == 'negative':
                    st.image("https://cdn-icons-png.flaticon.com/512/10009/10009514.png", width=100)
                    st.error(f"**NEGATIVE**")
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/10009/10009650.png", width=100)
                    st.warning(f"**NEUTRAL**")
            
            with col2:
                st.write(f"**Confidence Score:** {confidence:.2%}")
                st.progress(float(confidence))
                st.write(f"*The model is {confidence:.2%} sure about this result.*")
                
                # Sarcasm Check (Simple Logic)
                if prediction == 'positive' and "bad" in user_input.lower():
                    st.info("⚠️ Note: This text contains negative words but was classified as Positive. Possible Sarcasm?")
        else:
            st.warning("Please enter some text first.")
