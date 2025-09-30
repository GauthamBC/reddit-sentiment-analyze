import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import io
import time

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="Reddit Comment Sentiment Analyzer", layout="wide")
st.title("📊 Reddit Comment Sentiment Analyzer")

# ==============================
# Top Navigation Tabs
# ==============================
tabs = st.tabs(["URLs Fetcher", "Comment scraper", "Sentiment / Emotion Analyzer"])

# ==============================
# Tab 1: URLs Fetcher
# ==============================
with tabs[0]:
    st.subheader("🔗 URLs Fetcher")
    url = st.text_input("URL:", placeholder="Paste your Google search url here")
    if st.button("Fetch URLS", use_container_width=True):
        st.info(f"Fetching URLs from: {url}")
        # 👉 Insert your URL fetching logic here

# ==============================
# Tab 2: Comment scraper
# ==============================
with tabs[1]:
    st.subheader("💬 Comment Scraper")
    urls = st.text_area("URLs:", placeholder="Paste Reddit urls, one per line")
    if st.button("Scrape Comments", use_container_width=True):
        st.info(f"Scraping comments from {len(urls.splitlines())} URLs...")
        # 👉 Insert your scraping logic here

# ==============================
# Tab 3: Sentiment / Emotion Analyzer
# ==============================
with tabs[2]:
    st.subheader("📊 Sentiment / Emotion Analyzer")

    uploaded_file = st.file_uploader("Upload your Reddit CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=0, on_bad_lines="skip")
        st.success(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

        col_to_analyze = st.selectbox("Select column to analyze:", df.columns)
        start = st.number_input("Start row (0-indexed)", min_value=0, max_value=len(df), value=0)
        end = st.number_input("End row (exclusive)", min_value=1, max_value=len(df), value=len(df))

        texts = df[col_to_analyze].iloc[start:end].astype(str).tolist()

        if "active_analysis" not in st.session_state:
            st.session_state.active_analysis = None

        # Buttons side by side
        col1, col2 = st.columns([1, 1])
        with col1:
            run_sentiment = st.button("🚀 Run Sentiment Analysis", use_container_width=True, key="btn_sent")
        with col2:
            run_emotion = st.button("🎭 Run Emotion Analysis", use_container_width=True, key="btn_emo")

        # Prevent running both simultaneously
        if run_sentiment and st.session_state.active_analysis == "emotion":
            st.warning("⚠️ Please clear the Emotion Analysis table before running Sentiment Analysis.")
            run_sentiment = False

        if run_emotion and st.session_state.active_analysis == "sentiment":
            st.warning("⚠️ Please clear the Sentiment Analysis table before running Emotion Analysis.")
            run_emotion = False

        # 👉 Paste your full analyzer logic here (from the last working version),
        # including saving results in st.session_state, download + clear buttons,
        # and the 3 result tabs.
