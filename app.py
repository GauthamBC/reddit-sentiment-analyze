import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import io

# ==============================
# 1) Load Sentiment Model
# ==============================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_model = load_model()

# ==============================
# 2) Streamlit UI
# ==============================
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")
st.title("📊 Reddit Comment Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload your Reddit CSV", type=["csv"])

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file, header=0, on_bad_lines="skip")
    st.success(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Column selector
    col_to_analyze = st.selectbox("Select column to analyze:", df.columns)

    # Row range
    start = st.number_input("Start row (0-indexed)", min_value=0, max_value=len(df), value=0)
    end = st.number_input("End row (exclusive)", min_value=1, max_value=len(df), value=len(df))

    if st.button("🚀 Run Sentiment Analysis"):
        texts = df[col_to_analyze].iloc[start:end].astype(str).tolist()
        st.info(f"Running analysis on {len(texts)} comments... Please wait ⏳")

        # Sentiment predictions
        results = sentiment_model(texts, batch_size=32, truncation=True)

        # Map raw model labels to human-readable ones
        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        
        # Add results back to DataFrame
        df_results = df.iloc[start:end].copy()
        df_results["sentiment_label"] = [label_map[r["label"]] for r in results]
        df_results["sentiment_score"] = [r["score"] for r in results]

        # Breakdown summary
        sentiment_counts = Counter(df_results["sentiment_label"])
        total = sum(sentiment_counts.values())
        df_summary = pd.DataFrame([
            {"Category": k, "Count": v, "Percentage": round((v/total)*100, 2)}
            for k, v in sentiment_counts.items()
        ])

        # Tabs
        tab1, tab2 = st.tabs(["📄 Per-Comment Results", "📊 Breakdown Summary"])

        with tab1:
            st.dataframe(df_results, use_container_width=True)

        with tab2:
            st.table(df_summary)

        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Per-Comment Results", index=False)
            df_summary.to_excel(writer, sheet_name="Breakdown Summary", index=False)
        output.seek(0)

        st.download_button(
            label="⬇️ Download Results as Excel",
            data=output,
            file_name="reddit_sentiment_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
