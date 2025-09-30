import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import io
import time

# ==============================
# 1) Load Models
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
# 2) Streamlit UI
# ==============================
st.set_page_config(page_title="Reddit Analyzer", layout="wide")
st.title("📊 Reddit Comment Analyzer")

uploaded_file = st.file_uploader("Upload your Reddit CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=0, on_bad_lines="skip")
    st.success(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

    col_to_analyze = st.selectbox("Select column to analyze:", df.columns)
    start = st.number_input("Start row (0-indexed)", min_value=0, max_value=len(df), value=0)
    end = st.number_input("End row (exclusive)", min_value=1, max_value=len(df), value=len(df))

    texts = df[col_to_analyze].iloc[start:end].astype(str).tolist()

    # --- Sentiment Analysis ---
    if st.button("🚀 Run Sentiment Analysis"):
        st.info(f"Running sentiment analysis on {len(texts)} comments... ⏳")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            results.extend(sentiment_model(batch, truncation=True, max_length=512))
            percent = int(((i+len(batch)) / len(texts)) * 100)
            progress_bar.progress(percent)
            status_text.text(f"Processed {i+len(batch)} / {len(texts)} comments ({percent}%)")
            time.sleep(0.01)

        # Map labels
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        df_results = df.iloc[start:end].copy()
        df_results["sentiment_label"] = [label_map[r["label"]] for r in results]
        df_results["sentiment_score"] = [r["score"] for r in results]

        # Summary
        sentiment_counts = Counter(df_results["sentiment_label"])
        total = sum(sentiment_counts.values())
        df_summary = pd.DataFrame([
            {"Category": k, "Count": v, "Percentage": round((v/total)*100, 2)}
            for k, v in sentiment_counts.items()
        ])

        st.success("✅ Sentiment analysis complete!")
        tab1, tab2 = st.tabs(["📄 Per-Comment Sentiment", "📊 Sentiment Breakdown"])
        with tab1: st.dataframe(df_results, use_container_width=True)
        with tab2: st.table(df_summary)

    # --- Emotion Analysis ---
    if st.button("🎭 Run Emotion Analysis"):
        st.info(f"Running emotion analysis on {len(texts)} comments... ⏳")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            results.extend(emotion_model(batch, truncation=True, max_length=512))
            percent = int(((i+len(batch)) / len(texts)) * 100)
            progress_bar.progress(percent)
            status_text.text(f"Processed {i+len(batch)} / {len(texts)} comments ({percent}%)")
            time.sleep(0.01)

        # Pick dominant emotion
        df_results = df.iloc[start:end].copy()
        df_results["dominant_emotion"] = [max(r, key=lambda x: x["score"])["label"] for r in results]
        df_results["emotion_score"] = [max(r, key=lambda x: x["score"])["score"] for r in results]

        # Summary
        emotion_counts = Counter(df_results["dominant_emotion"])
        total = sum(emotion_counts.values())
        df_summary = pd.DataFrame([
            {"Emotion": k, "Count": v, "Percentage": round((v/total)*100, 2)}
            for k, v in emotion_counts.items()
        ])

        st.success("✅ Emotion analysis complete!")
        tab1, tab2 = st.tabs(["📄 Per-Comment Emotion", "📊 Emotion Breakdown"])
        with tab1: st.dataframe(df_results, use_container_width=True)
        with tab2: st.table(df_summary)
