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
st.set_page_config(page_title="Reddit Comment Sentiment Analyzer", layout="wide")
st.title("📊 Reddit Comment Sentiment Analyzer")

# ------------------------------
# Create Tabs
# ------------------------------
tabs = st.tabs(["URLs Fetcher", "Comment scraper", "Sentiment / Emotion Analyzer"])

# ==============================
# Tab 1: URLs Fetcher
# ==============================
with tabs[0]:
    st.subheader("🔗 URLs Fetcher")
    col1, col2 = st.columns([3, 1])  # adjust ratio as needed
    with col1:
        serpapi_key = st.text_input("🔑 Enter your SerpAPI Key", type="password")
    with col2:
        st.markdown(
            "[👉 Get key](https://serpapi.com/dashboard)",
            unsafe_allow_html=True
        )
    google_url = st.text_input("🌐 Google Search URL", placeholder="Paste your Google Search URL here")

    num_pages = st.slider("Number of Pages to Fetch", min_value=1, max_value=20, value=10)

    if st.button("Fetch URLs", use_container_width=True):
        if not serpapi_key:
            st.error("❌ Please enter your SerpAPI Key.")
        elif not google_url:
            st.error("❌ Please enter a Google Search URL.")
        else:
            try:
                # Extract query from Google URL
                parsed = urllib.parse.urlparse(google_url)
                qs = urllib.parse.parse_qs(parsed.query)
                query = qs.get("q")
                if not query:
                    st.error("❌ Could not extract query from the given URL")
                else:
                    query = query[0]
                    urls = set()
                    progress = st.progress(0)

                    for page in range(num_pages):
                        params = {
                            "engine": "google",
                            "q": query,
                            "start": page * 10,
                            "num": 10,
                            "api_key": serpapi_key,
                        }
                        res = requests.get("https://serpapi.com/search.json", params=params).json()
                        for item in res.get("organic_results", []):
                            link = item.get("link")
                            if link:
                                urls.add(link)
                        progress.progress(int((page + 1) / num_pages * 100))

                    urls = sorted(urls)
                    if urls:
                        st.success(f"✅ Found {len(urls)} unique URLs")
                        st.dataframe(pd.DataFrame(urls, columns=["URL"]), use_container_width=True)

                        # Download button
                        url_output = io.BytesIO()
                        pd.DataFrame(urls, columns=["URL"]).to_csv(url_output, index=False)
                        url_output.seek(0)
                        st.download_button(
                            label="⬇️ Download URLs CSV",
                            data=url_output,
                            file_name="fetched_urls.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No URLs found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
# ==============================
# Tab 2: Comment Scraper
# ==============================
with tabs[1]:
    st.subheader("💬 Comment Scraper")
    urls = st.text_area("URLs:", placeholder="Paste Reddit URLs, one per line")
    if st.button("Scrape Comments", use_container_width=True):
        st.info(f"Scraping comments from {len(urls.splitlines())} URLs...")
        # (logic placeholder for scraping)

# ==============================
# Tab 3: Sentiment / Emotion Analyzer
# ==============================
with tabs[2]:
    st.subheader("📊 Sentiment / Emotion Analyzer")

    # Track which analysis is active
    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None

    # Step 1: Upload file
    uploaded_file = st.file_uploader("Upload your Reddit CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=0, on_bad_lines="skip")
        st.success(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Step 2: Show extra controls
        col_to_analyze = st.selectbox("Select column to analyze:", df.columns)
        start = st.number_input("Start row (0-indexed)", min_value=0, max_value=len(df), value=0)
        end = st.number_input("End row (exclusive)", min_value=1, max_value=len(df), value=len(df))
        texts = df[col_to_analyze].iloc[start:end].astype(str).tolist()

        col1, col2 = st.columns([1, 1])
        with col1:
            run_sentiment = st.button("🚀 Run Sentiment Analysis", use_container_width=True, key="btn_sent")
        with col2:
            run_emotion = st.button("🎭 Run Emotion Analysis", use_container_width=True, key="btn_emo")

        # Prevent conflicts
        if run_sentiment and st.session_state.active_analysis == "emotion":
            st.warning("⚠️ Please clear the Emotion Analysis table before running Sentiment Analysis.")
            run_sentiment = False
        if run_emotion and st.session_state.active_analysis == "sentiment":
            st.warning("⚠️ Please clear the Sentiment Analysis table before running Emotion Analysis.")
            run_emotion = False

        # ==============================
        # Sentiment Analysis
        # ==============================
        if run_sentiment:
            st.session_state.active_analysis = "sentiment"

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

            label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            df_results = df.iloc[start:end].copy()
            df_results["sentiment_label"] = [label_map[r["label"]] for r in results]
            df_results["sentiment_score"] = [r["score"] for r in results]

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

            st.session_state.sentiment_results = (df_results, df_summary_all, df_summary_wo)

        if st.session_state.active_analysis == "sentiment" and "sentiment_results" in st.session_state:
            df_results, df_summary_all, df_summary_wo = st.session_state.sentiment_results

            st.success("✅ Sentiment analysis complete!")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, sheet_name="Per-Comment Sentiment", index=False)
                    df_summary_all.to_excel(writer, sheet_name="Breakdown All Sentiments", index=False)
                    df_summary_wo.to_excel(writer, sheet_name="Breakdown Excl Neutral", index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Download Sentiment Results",
                    data=output,
                    file_name="reddit_sentiment_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sent",
                    use_container_width=True
                )
            with col_b:
                if st.button("🧹 Clear Table", use_container_width=True, key="clear_sent"):
                    st.session_state.active_analysis = None
                    del st.session_state["sentiment_results"]
                    st.rerun()

            tab1, tab2, tab3 = st.tabs([
                "📄 Per-Comment Sentiment",
                "📊 Sentiment Breakdown (All Sentiments)",
                "📊 Sentiment Breakdown (Excluding Neutral, Renormalized)"
            ])
            with tab1: st.dataframe(df_results, use_container_width=True)
            with tab2: st.table(df_summary_all)
            with tab3: st.table(df_summary_wo)

        # ==============================
        # Emotion Analysis
        # ==============================
        if run_emotion:
            st.session_state.active_analysis = "emotion"

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

            st.session_state.emotion_results = (df_results, df_summary_all, df_summary_wo)

        if st.session_state.active_analysis == "emotion" and "emotion_results" in st.session_state:
            df_results, df_summary_all, df_summary_wo = st.session_state.emotion_results

            st.success("✅ Emotion analysis complete!")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, sheet_name="Per-Comment Emotion", index=False)
                    df_summary_all.to_excel(writer, sheet_name="Breakdown All Emotions", index=False)
                    df_summary_wo.to_excel(writer, sheet_name="Breakdown Excl Neutral", index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Download Emotion Results",
                    data=output,
                    file_name="reddit_emotion_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_emo"
                )
            with col_b:
                if st.button("🧹 Clear Table", use_container_width=True, key="clear_emo"):
                    st.session_state.active_analysis = None
                    del st.session_state["emotion_results"]
                    st.rerun()

            tab1, tab2, tab3 = st.tabs([
                "📄 Per-Comment Emotion",
                "📊 Emotion Breakdown (All Emotions)",
                "📊 Emotion Breakdown (Excluding Neutral, Renormalized)"
            ])
            with tab1: st.dataframe(df_results, use_container_width=True)
            with tab2: st.table(df_summary_all)
            with tab3: st.table(df_summary_wo)

