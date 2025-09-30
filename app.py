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
  
    # --- Track which analysis is active ---
    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None

    # ==============================
    # Buttons side by side
    # ==============================
    col1, col2 = st.columns([1, 1])
    with col1:
        run_sentiment = st.button("🚀 Run Sentiment Analysis", use_container_width=True)
    with col2:
        run_emotion = st.button("🎭 Run Emotion Analysis", use_container_width=True)

    # --- Sentiment Analysis ---
    if run_sentiment:
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

        # Summaries
        sentiment_counts_all = Counter(df_results["sentiment_label"])
        total_all = sum(sentiment_counts_all.values())
        df_summary_all = pd.DataFrame([
            {"Sentiment": k, "Count": v, "Percentage": round((v/total_all)*100, 2)}
            for k, v in sentiment_counts_all.items()
        ])
        df_summary_all.loc[len(df_summary_all)] = ["Total", total_all, 100.0]

        sentiment_counts_wo = {k: v for k, v in sentiment_counts_all.items() if k.lower() != "neutral"}
        total_wo = sum(sentiment_counts_wo.values())
        df_summary_wo = pd.DataFrame([
            {"Sentiment": k, "Count": v, "Percentage": round((v/total_wo)*100, 2)}
            for k, v in sentiment_counts_wo.items()
        ])
        df_summary_wo.loc[len(df_summary_wo)] = ["Total", total_wo, 100.0]

        # Save in session state
        st.session_state["sentiment_results"] = {
            "df_results": df_results,
            "df_summary_all": df_summary_all,
            "df_summary_wo": df_summary_wo
        }
        st.success("✅ Sentiment analysis complete!")

    # Render Sentiment Results if available
    if "sentiment_results" in st.session_state:
        res = st.session_state["sentiment_results"]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            res["df_results"].to_excel(writer, sheet_name="Per-Comment Sentiment", index=False)
            res["df_summary_all"].to_excel(writer, sheet_name="Breakdown All Sentiments", index=False)
            res["df_summary_wo"].to_excel(writer, sheet_name="Breakdown Excl Neutral", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Sentiment Results",
            data=output,
            file_name="reddit_sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        tab1, tab2, tab3 = st.tabs([
            "📄 Per-Comment Sentiment",
            "📊 Sentiment Breakdown (All Sentiments)",
            "📊 Sentiment Breakdown (Excluding Neutral, Renormalized)"
        ])
        with tab1: st.dataframe(res["df_results"], use_container_width=True)
        with tab2: st.table(res["df_summary_all"])
        with tab3: st.table(res["df_summary_wo"])

    # --- Emotion Analysis ---
    if run_emotion:
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

        dominant_emotions, dominant_scores = [], []
        for r in results:
            top = max(r, key=lambda x: x["score"])
            dominant_emotions.append(top["label"])
            dominant_scores.append(top["score"])

        df_results = df.iloc[start:end].copy()
        df_results["dominant_emotion"] = dominant_emotions
        df_results["emotion_score"] = dominant_scores

        # Summaries
        emotion_counts_all = Counter(dominant_emotions)
        total_all = sum(emotion_counts_all.values())
        df_summary_all = pd.DataFrame([
            {"Emotion": k, "Count": v, "Percentage": round((v/total_all)*100, 2)}
            for k, v in emotion_counts_all.items()
        ])
        df_summary_all.loc[len(df_summary_all)] = ["Total", total_all, 100.0]

        emotion_counts_wo = {k: v for k, v in emotion_counts_all.items() if k.lower() != "neutral"}
        total_wo = sum(emotion_counts_wo.values())
        df_summary_wo = pd.DataFrame([
            {"Emotion": k, "Count": v, "Percentage": round((v/total_wo)*100, 2)}
            for k, v in emotion_counts_wo.items()
        ])
        df_summary_wo.loc[len(df_summary_wo)] = ["Total", total_wo, 100.0]

        # Save in session state
        st.session_state["emotion_results"] = {
            "df_results": df_results,
            "df_summary_all": df_summary_all,
            "df_summary_wo": df_summary_wo
        }
        st.success("✅ Emotion analysis complete!")

    # Render Emotion Results if available
    if "emotion_results" in st.session_state:
        res = st.session_state["emotion_results"]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            res["df_results"].to_excel(writer, sheet_name="Per-Comment Emotion", index=False)
            res["df_summary_all"].to_excel(writer, sheet_name="Breakdown All Emotions", index=False)
            res["df_summary_wo"].to_excel(writer, sheet_name="Breakdown Excl Neutral", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Emotion Results",
            data=output,
            file_name="reddit_emotion_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        tab1, tab2, tab3 = st.tabs([
            "📄 Per-Comment Emotion",
            "📊 Emotion Breakdown (All Emotions)",
            "📊 Emotion Breakdown (Excluding Neutral, Renormalized)"
        ])
        with tab1: st.dataframe(res["df_results"], use_container_width=True)
        with tab2: st.table(res["df_summary_all"])
        with tab3: st.table(res["df_summary_wo"])
