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

        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Per-Comment Sentiment", index=False)
            df_summary.to_excel(writer, sheet_name="Sentiment Breakdown", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Sentiment Results",
            data=output,
            file_name="reddit_sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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

        # Pick dominant emotion (skip Neutral if top-1)
        dominant_emotions, dominant_scores = [], []
        for r in results:
            sorted_emotions = sorted(r, key=lambda x: x["score"], reverse=True)
            if sorted_emotions[0]["label"].lower() == "neutral":
                dominant_emotions.append(sorted_emotions[1]["label"])
                dominant_scores.append(sorted_emotions[1]["score"])
            else:
                dominant_emotions.append(sorted_emotions[0]["label"])
                dominant_scores.append(sorted_emotions[0]["score"])

        df_results = df.iloc[start:end].copy()
        df_results["dominant_emotion"] = dominant_emotions
        df_results["emotion_score"] = dominant_scores

        # Full summary
        emotion_counts = Counter(dominant_emotions)
        total = sum(emotion_counts.values())
        df_summary_full = pd.DataFrame([
            {"Emotion": k, "Count": v, "Percentage": round((v/total)*100, 2)}
            for k, v in emotion_counts.items()
        ])

        # Option to exclude neutral
        exclude_neutral = st.checkbox("Exclude Neutral and Renormalize Breakdown")

        if exclude_neutral and "neutral" in [e.lower() for e in emotion_counts.keys()]:
            emotion_counts_no_neutral = {k: v for k, v in emotion_counts.items() if k.lower() != "neutral"}
            total_no_neutral = sum(emotion_counts_no_neutral.values())
            df_summary = pd.DataFrame([
                {"Emotion": k, "Count": v, "Percentage": round((v/total_no_neutral)*100, 2)}
                for k, v in emotion_counts_no_neutral.items()
            ])
        else:
            df_summary = df_summary_full

        st.success("✅ Emotion analysis complete!")
        tab1, tab2 = st.tabs(["📄 Per-Comment Emotion", "📊 Emotion Breakdown"])
        with tab1: st.dataframe(df_results, use_container_width=True)
        with tab2: st.table(df_summary)

        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Per-Comment Emotion", index=False)
            df_summary_full.to_excel(writer, sheet_name="Emotion Breakdown (Full)", index=False)
            if exclude_neutral:
                df_summary.to_excel(writer, sheet_name="Emotion Breakdown (No Neutral)", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Emotion Results",
            data=output,
            file_name="reddit_emotion_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
