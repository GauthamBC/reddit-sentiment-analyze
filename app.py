import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import io
import time
import praw, re
from datetime import datetime, date, timedelta, timezone
from dateutil import parser as dtparse

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
# Tab 1: Reddit URL Collector
# ==============================
CLIENT_ID     = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
USER_AGENT    = st.secrets["USER_AGENT"]

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
    ratelimit_seconds=5
)
reddit.read_only = True

with tabs[0]:
    st.subheader("🔗 Reddit URL Collector — Boolean + Per-Query Subreddit Targeting")

    # --- Boolean queries
    queries_text = st.text_area(
        "Enter Boolean queries (one per line)",
        placeholder='("Rory McIlroy" AND "Ryder Cup") OR "Bethpage"',
        height=120
    )

    # --- Per-query subreddit pairing
    pairing_enabled = st.checkbox("Enable per-query subreddit targeting", value=False)
    if pairing_enabled:
        st.markdown(
            "<small>✅ When enabled, each query line targets the subreddit on the same line below.<br>"
            "Leave blank = search all of Reddit.</small>",
            unsafe_allow_html=True
        )
        pairing_subs_text = st.text_area(
            "Enter subreddits (one per line matching each query)",
            placeholder="azcardinals\nfalcons",
            height=100
        )
    else:
        subs_mode = st.radio("Subreddits", ["All", "Specific"], horizontal=True)
        if subs_mode == "Specific":
            subs_text = st.text_area("Enter subreddits (one per line)", placeholder="SquaredCircle\ngolf")
            sub_list = [s.strip().lstrip("r/") for s in subs_text.splitlines() if s.strip()]
        else:
            sub_list = ["all"]

    # --- Time range
    mode = st.radio("Time range", ["Last N hours", "Last N days", "Custom dates"], horizontal=True)
    now = datetime.now(timezone.utc)
    if mode == "Last N hours":
        hours = st.slider("Hours:", 1, 48, 24)
        after_ts = int((now - timedelta(hours=hours)).timestamp())
        before_ts = int(now.timestamp())
    elif mode == "Last N days":
        days = st.slider("Days:", 1, 30, 7)
        after_ts = int((now - timedelta(days=days)).timestamp())
        before_ts = int(now.timestamp())
    else:
        from_date = st.date_input("From date", value=date.today() - timedelta(days=1))
        to_date = st.date_input("To date", value=date.today())
        start_local = dtparse.parse(f"{from_date} 00:00").astimezone()
        end_local   = dtparse.parse(f"{to_date} 23:59").astimezone()
        after_ts = int(start_local.astimezone(timezone.utc).timestamp())
        before_ts = int(end_local.astimezone(timezone.utc).timestamp())

    # --- Other options
    match_in = st.selectbox("Match in", ["Title + Selftext", "Title only", "Selftext only"])
    per_query_limit = st.number_input("Max posts per seed (0 = unlimited)", min_value=0, value=50)
    min_comments = st.number_input("Minimum comments per post", min_value=0, value=0)
    max_comments_per_sub = st.number_input("Stop after N total comments per subreddit", min_value=100, value=2000)
    global_comment_cap = st.number_input("Global comment cap (0 = unlimited)", min_value=0, value=0)

    # --- Boolean parser / evaluator (same as Colab)
    TOKEN_RE = re.compile(r'''
        ("[^"\\]*(?:\\.[^"\\]*)*")|(\()|(\))|
        (?:\bAND\b|&&|&)|(?:\bOR\b|\|\||\|)|
        (?:\bNOT\b|!|-)|
        ([^\s()]+)
    ''', re.IGNORECASE | re.VERBOSE)

    class Node: pass
    class Term(Node):  def __init__(self, s): self.s = s
    class Not(Node):   def __init__(self, a): self.a = a
    class And(Node):   def __init__(self, a,b): self.a,self.b = a,b
    class Or(Node):    def __init__(self, a,b): self.a,self.b = a,b

    def tokenize(expr:str):
        tokens=[]
        for m in TOKEN_RE.finditer(expr):
            if m.group(1): tokens.append(("TERM", m.group(1)[1:-1]))
            elif m.group(2): tokens.append(("LPAREN","("))
            elif m.group(3): tokens.append(("RPAREN",")"))
            else:
                t=m.group(0).strip(); u=t.upper()
                if u in ("AND","&&","&"): tokens.append(("AND","AND"))
                elif u in ("OR","||","|"): tokens.append(("OR","OR"))
                elif u in ("NOT","!","-"): tokens.append(("NOT","NOT"))
                else: tokens.append(("TERM",t))
        return tokens

    def parse(tokens):
        i=0
        def parse_or():
            nonlocal i; node=parse_and()
            while i<len(tokens) and tokens[i][0]=="OR": i+=1; node=Or(node,parse_and())
            return node
        def parse_and():
            nonlocal i; node=parse_not()
            while i<len(tokens) and tokens[i][0]=="AND": i+=1; node=And(node,parse_not())
            return node
        def parse_not():
            nonlocal i
            if i<len(tokens) and tokens[i][0]=="NOT": i+=1; return Not(parse_not())
            return parse_atom()
        def parse_atom():
            nonlocal i
            if i<len(tokens) and tokens[i][0]=="LPAREN":
                i+=1; node=parse_or()
                if i>=len(tokens) or tokens[i][0]!="RPAREN": raise ValueError("Unclosed parenthesis")
                i+=1; return node
            if i<len(tokens) and tokens[i][0]=="TERM": s=tokens[i][1]; i+=1; return Term(s)
            raise ValueError("Unexpected token")
        node=parse_or()
        if i!=len(tokens): raise ValueError("Extra tokens")
        return node

    def eval_node(node,title,body,where):
        def present(needle):
            n=needle.lower()
            if where=="Title only": return n in title
            if where=="Selftext only": return n in body
            return n in title or n in body
        if isinstance(node,Term): return present(node.s)
        if isinstance(node,Not): return not eval_node(node.a,title,body,where)
        if isinstance(node,And): return eval_node(node.a,title,body,where) and eval_node(node.b,title,body,where)
        if isinstance(node,Or):  return eval_node(node.a,title,body,where) or  eval_node(node.b,title,body,where)
        return False

    def collect_terms(node,under_not=False):
        out=set()
        if isinstance(node,Term):
            if not under_not and node.s.strip(): out.add(node.s)
        if isinstance(node,Not): out |= collect_terms(node.a,True)
        if isinstance(node,And) or isinstance(node,Or):
            out |= collect_terms(node.a,under_not)
            out |= collect_terms(node.b,under_not)
        return out

    # --- Runner
    if st.button("🚀 Run Reddit Collector", use_container_width=True):
        if not queries_text.strip():
            st.error("❌ Please enter at least one Boolean query.")
        else:
            query_lines=[q.strip() for q in queries_text.splitlines() if q.strip()]
            pairs=[]
            if pairing_enabled:
                sub_lines=[s.strip().lstrip("r/") for s in pairing_subs_text.splitlines()]
                for i,expr in enumerate(query_lines):
                    sub_in=sub_lines[i] if i<len(sub_lines) and sub_lines[i] else "all"
                    pairs.append((expr,sub_in))
            else:
                for expr in query_lines:
                    for s in sub_list:
                        pairs.append((expr,s))

            all_rows=[]; total_comments=0
            progress=st.progress(0); status=st.empty()
            start_time=time.time()

            for idx,(expr,subname) in enumerate(pairs):
                try:
                    sub=reddit.subreddit(subname)
                    ast=parse(tokenize(expr))
                except Exception as e:
                    st.warning(f"⚠️ Invalid pair ({expr}, r/{subname}) — {e}")
                    continue

                seed_terms=sorted(collect_terms(ast), key=len, reverse=True)[:3]
                comment_sum=0
                for seed in seed_terms:
                    try:
                        posts=sub.search(seed,sort="new",time_filter="week",
                                         limit=per_query_limit or None)
                        for p in posts:
                            ts=int(getattr(p,"created_utc",0))
                            if ts<after_ts or ts>before_ts: continue
                            title=(p.title or "").lower()
                            body=(getattr(p,"selftext","") or "").lower()
                            if not eval_node(ast,title,body,match_in): continue
                            n_comments=int(getattr(p,"num_comments",0))
                            if n_comments<min_comments: continue
                            all_rows.append({
                                "query":expr,"seed_term":seed,"subreddit":str(p.subreddit),
                                "title":p.title,
                                "author":str(p.author) if p.author else "[deleted]",
                                "created_utc":ts,
                                "created_utc_iso":datetime.fromtimestamp(ts,tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                                "num_comments":n_comments,"score":int(getattr(p,"score",0)),
                                "url":f"https://www.reddit.com{p.permalink}"
                            })
                            comment_sum+=n_comments; total_comments+=n_comments
                            if max_comments_per_sub and comment_sum>=max_comments_per_sub: break
                            if global_comment_cap and total_comments>=global_comment_cap: break
                        if global_comment_cap and total_comments>=global_comment_cap: break
                    except Exception as e:
                        st.warning(f"⚠️ Retry failed r/{subname} | {seed}: {e}")
                    time.sleep(0.5)

                percent=int(((idx+1)/len(pairs))*100)
                progress.progress(percent)
                status.text(f"Processed {idx+1}/{len(pairs)} | r/{subname} ({comment_sum} comments)")
                if global_comment_cap and total_comments>=global_comment_cap: break

            if not all_rows:
                st.warning("⚠️ No matching posts found.")
            else:
                df=(pd.DataFrame(all_rows)
                    .drop_duplicates(subset=["url"])
                    .query(f"num_comments >= {min_comments}")
                    .sort_values(["created_utc","subreddit"],ascending=[False,True])
                    .reset_index(drop=True))
                sub_summary=(df.groupby("subreddit",as_index=False)
                             .agg(threads=("url","count"),
                                  comments=("num_comments","sum"),
                                  avg_comments=("num_comments","mean"))
                             .sort_values(["comments","threads"],ascending=[False,False]))
                sub_summary["avg_comments"]=sub_summary["avg_comments"].round(1)

                st.session_state.last_df=df
                st.session_state.last_summary=sub_summary

                st.markdown(
                    f"✅ **Done in {time.time()-start_time:.1f}s** — "
                    f"{len(df)} posts, {df['num_comments'].sum()} comments across "
                    f"{df['subreddit'].nunique()} subreddits."
                )

                tab1,tab2=st.tabs(["📄 Full Results","📊 Frequency Table"])
                with tab1: st.dataframe(df,use_container_width=True)
                with tab2: st.dataframe(sub_summary,use_container_width=True)

    # --- Download cached results
    if "last_df" in st.session_state:
        df=st.session_state.last_df
        sub_summary=st.session_state.last_summary
        output=io.BytesIO()
        with pd.ExcelWriter(output,engine="openpyxl") as writer:
            df.to_excel(writer,sheet_name="Full Results",index=False)
            sub_summary.to_excel(writer,sheet_name="Frequency Table",index=False)
        output.seek(0)
        st.download_button("⬇️ Download Last Results",
                           data=output,
                           file_name="reddit_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
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

    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None

    uploaded_file = st.file_uploader("Upload your Reddit CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=0, on_bad_lines="skip")
        st.success(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

        col_to_analyze = st.selectbox("Select column to analyze:", df.columns)
        start = st.number_input("Start row (0-indexed)", min_value=0, max_value=len(df), value=0)
        end = st.number_input("End row (exclusive)", min_value=1, max_value=len(df), value=len(df))
        texts = df[col_to_analyze].iloc[start:end].astype(str).tolist()

        col1, col2 = st.columns([1, 1])
        with col1:
            run_sentiment = st.button("🚀 Run Sentiment Analysis", use_container_width=True, key="btn_sent")
        with col2:
            run_emotion = st.button("🎭 Run Emotion Analysis", use_container_width=True, key="btn_emo")

        if run_sentiment and st.session_state.active_analysis == "emotion":
            st.warning("⚠️ Please clear the Emotion Analysis table before running Sentiment Analysis.")
            run_sentiment = False
        if run_emotion and st.session_state.active_analysis == "sentiment":
            st.warning("⚠️ Please clear the Sentiment Analysis table before running Emotion Analysis.")
            run_emotion = False

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
