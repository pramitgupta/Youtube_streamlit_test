
# YouTube Comment Sentiment — v2 (robust UI)
# ------------------------------------------
# - Always shows inputs (in both sidebar and main "Fetch" tab), no early st.stop
# - Safer sorting & column handling
# - Optional Transformers; defaults to VADER
# - CSV export
#
# Colab:
# !pip -q install -r requirements_youtube.txt
# from pyngrok import ngrok; import os
# os.environ["STREAMLIT_SERVER_PORT"]="8501"
# public_url = ngrok.connect(8501); public_url
# !streamlit run streamlit_youtube_sentiment_v2.py --server.headless true --server.port 8501
import streamlit as st
import pandas as pd
import numpy as np
import requests, re
import altair as alt
from datetime import datetime
from typing import List, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Optional transformers
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="YouTube Sentiment v2", layout="wide")

# Ensure NLTK resources
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
try:
    nltk.data.find("corpora/stopwords.zip")
except LookupError:
    nltk.download("stopwords")

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

def extract_video_id(url_or_id: str) -> Optional[str]:
    s = (url_or_id or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", s):
        return s
    pats = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for p in pats:
        m = re.search(p, s)
        if m: return m.group(1)
    return None

@st.cache_data(show_spinner=False)
def fetch_comments(api_key: str, video_id: str, max_count: int = 1000, order: str = "relevance", include_replies: bool = True) -> pd.DataFrame:
    rows = []
    params = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,
        "key": api_key,
        "order": order,
        "textFormat": "plainText",
    }
    fetched = 0
    next_token = None
    while True:
        if next_token:
            params["pageToken"] = next_token
        r = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=30)
        if r.status_code != 200:
            try:
                j = r.json()
                msg = j.get("error", {}).get("message", str(j))
            except Exception:
                msg = r.text
            raise RuntimeError(f"YT API error {r.status_code}: {msg}")
        data = r.json()
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": top.get("authorDisplayName"),
                "text": top.get("textOriginal", ""),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt"),
                "is_reply": 0,
            })
            fetched += 1
            if fetched >= max_count: break
            if include_replies:
                for rep in item.get("replies", {}).get("comments", []):
                    rs = rep["snippet"]
                    rows.append({
                        "comment_id": rep["id"],
                        "author": rs.get("authorDisplayName"),
                        "text": rs.get("textOriginal", ""),
                        "like_count": rs.get("likeCount", 0),
                        "published_at": rs.get("publishedAt"),
                        "is_reply": 1,
                    })
                    fetched += 1
                    if fetched >= max_count: break
        if fetched >= max_count: break
        next_token = data.get("nextPageToken")
        if not next_token: break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
        df["text"] = df["text"].fillna("")
        df["author"] = df["author"].fillna("")
        df["is_reply"] = df["is_reply"].fillna(0).astype(int)
    return df

def run_vader(texts: List[str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t) for t in texts]
    out = pd.DataFrame(scores)
    def lab(c):
        if c >= 0.05: return "positive"
        if c <= -0.05: return "negative"
        return "neutral"
    out["label"] = out["compound"].apply(lab)
    return out

@st.cache_resource(show_spinner=False)
def load_transformer(model_name: str):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed")
    return pipeline("sentiment-analysis", model=model_name)

def run_transformer(texts: List[str], model_name: str) -> pd.DataFrame:
    clf = load_transformer(model_name)
    preds = []
    bs = 64
    for i in range(0, len(texts), bs):
        preds.extend(clf(texts[i:i+bs]))
    dfp = pd.DataFrame(preds)
    # normalize labels
    dfp["label"] = dfp["label"].astype(str).str.upper().replace({
        "NEGATIVE":"negative","POSITIVE":"positive",
        "LABEL_0":"negative","LABEL_1":"positive",
        "1 STAR":"negative","2 STARS":"negative","3 STARS":"neutral","4 STARS":"positive","5 STARS":"positive"
    }).str.lower()
    dfp["compound"] = dfp.apply(lambda r: r["score"] if r["label"]=="positive" else (-r["score"] if r["label"]=="negative" else 0.0), axis=1)
    for k in ["neg","neu","pos"]:
        if k not in dfp.columns: dfp[k] = np.nan
    # pad to length if needed (rare)
    if len(dfp) < len(texts):
        pad = pd.DataFrame({"neg":[np.nan], "neu":[np.nan], "pos":[np.nan], "compound":[0.0], "label":["neutral"]})
        dfp = pd.concat([dfp, pad.loc[[]]]).reset_index(drop=True)
    return dfp[["neg","neu","pos","compound","label"]]

def safe_top_examples(df: pd.DataFrame, k=10, positive=True) -> pd.DataFrame:
    tmp = df.copy()
    if "compound" not in tmp.columns:
        tmp["compound"] = 0.0
    tmp["_sort"] = tmp["compound"] if positive else -tmp["compound"]
    cols = [c for c in ["author","text","like_count","compound","label","published_at"] if c in tmp.columns]
    return tmp.sort_values("_sort", ascending=False).head(k)[cols]

def ngram_frequencies(texts: List[str], ngram_range=(1,2), top_k=20, language="english"):
    stop = set(stopwords.words(language)) if language in stopwords.fileids() else set()
    cv = CountVectorizer(stop_words=stop, ngram_range=ngram_range, min_df=2)
    if not texts:
        return pd.DataFrame(columns=["term","count"])
    try:
        mat = cv.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["term","count"])
    sums = np.asarray(mat.sum(axis=0)).ravel()
    terms = np.array(cv.get_feature_names_out())
    idx = np.argsort(-sums)[:top_k]
    return pd.DataFrame({"term": terms[idx], "count": sums[idx].astype(int)})

# ===== UI =====
st.title("🎬 YouTube Comment Sentiment — v2")
st.caption("Enter your API key + video, fetch comments, then analyze sentiment.")

# Sidebar always visible
with st.sidebar:
    st.header("Settings")
    api_key_sb = st.text_input("YouTube API key", type="password")
    url_or_id_sb = st.text_input("Video URL or ID")
    max_comments_sb = st.number_input("Max comments", 50, 20000, 1000, 50)
    order_sb = st.selectbox("Order", ["relevance","time"], 0)
    include_replies_sb = st.checkbox("Include replies", True)
    st.divider()
    engine_sb = st.radio("Sentiment Engine", ["VADER (fast)", "Transformers (advanced)"], 0)
    model_name_sb = st.text_input("Transformers model", "distilbert-base-uncased-finetuned-sst-2-english")
    st.caption("Tip: For multilingual videos, try 'nlptown/bert-base-multilingual-uncased-sentiment'.")

tab_fetch, tab_analysis, tab_keywords = st.tabs(["🛰️ Fetch", "🧠 Analysis", "🔎 Keywords"])

with tab_fetch:
    st.subheader("Fetch YouTube Comments")
    api_key = st.text_input("API key (mirror of sidebar)", type="password", value=api_key_sb)
    url_or_id = st.text_input("Video URL/ID (mirror of sidebar)", value=url_or_id_sb, placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_comments = st.number_input("Max comments", 50, 20000, int(max_comments_sb), 50)
    with col2:
        order = st.selectbox("Order", ["relevance","time"], ["relevance","time"].index(order_sb))
    with col3:
        include_replies = st.checkbox("Include replies", value=include_replies_sb)

    fetch_btn = st.button("🚀 Fetch", type="primary")
    if fetch_btn:
        vid = extract_video_id(url_or_id)
        if not api_key or not vid:
            st.error("Please enter a valid API key and video URL/ID.")
        else:
            with st.spinner("Calling YouTube API..."):
                try:
                    df = fetch_comments(api_key, vid, max_count=int(max_comments), order=order, include_replies=include_replies)
                except Exception as e:
                    st.error(f"Failed to fetch comments: {e}")
                    df = pd.DataFrame()
            if df.empty:
                st.warning("No comments found.")
            else:
                st.success(f"Fetched {len(df):,} comments.")
                st.session_state["comments_df"] = df

    # Show current DF if present
    if "comments_df" in st.session_state and not st.session_state["comments_df"].empty:
        df_cur = st.session_state["comments_df"]
        st.dataframe(df_cur[["author","text","like_count","published_at","is_reply"]], use_container_width=True, height=350)
        st.download_button("💾 Download comments CSV", data=df_cur.to_csv(index=False).encode("utf-8"),
                           file_name="youtube_comments.csv", mime="text/csv")

with tab_analysis:
    st.subheader("Sentiment Analysis")
    if "comments_df" not in st.session_state or st.session_state["comments_df"].empty:
        st.info("Fetch comments first on the **Fetch** tab.")
    else:
        df = st.session_state["comments_df"].copy()
        df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)
        engine = st.radio("Choose engine", ["VADER (fast)", "Transformers (advanced)"], 0 if engine_sb.startswith("VADER") else 1, horizontal=True)
        model_name = st.text_input("Transformers model", value=model_name_sb)

        run_btn = st.button("Run Sentiment")
        if run_btn:
            if engine.startswith("VADER"):
                with st.spinner("Scoring with VADER..."):
                    sent = run_vader(df["text"].tolist())
            else:
                if not TRANSFORMERS_AVAILABLE:
                    st.error("Transformers not installed. Add 'transformers' to requirements or pip install it.")
                    sent = pd.DataFrame({"neg":[], "neu":[], "pos":[], "compound":[], "label":[]})
                else:
                    with st.spinner("Scoring with Transformers..."):
                        sent = run_transformer(df["text"].tolist(), model_name)
            sdf = pd.concat([df.reset_index(drop=True), sent.reset_index(drop=True)], axis=1)
            st.session_state["scored_df"] = sdf

        if "scored_df" in st.session_state and not st.session_state["scored_df"].empty:
            sdf = st.session_state["scored_df"]
            # metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Positive", f"{(sdf['label'].str.lower()=='positive').mean()*100:.1f}%")
            c2.metric("Neutral", f"{(sdf['label'].str.lower()=='neutral').mean()*100:.1f}%")
            c3.metric("Negative", f"{(sdf['label'].str.lower()=='negative').mean()*100:.1f}%")
            c4.metric("Avg compound", f"{sdf['compound'].fillna(0).mean():.3f}")
            # distribution
            dist = sdf["label"].str.lower().value_counts(normalize=True).rename_axis("label").reset_index(name="share")
            st.altair_chart(alt.Chart(dist).mark_bar().encode(x=alt.X("label:N", sort="-y"), y=alt.Y("share:Q", axis=alt.Axis(format="%"))).properties(height=240), use_container_width=True)
            # time series
            ts = sdf.groupby(pd.Grouper(key="date", freq="D"))["compound"].mean().rolling(7, min_periods=1).mean().reset_index()
            st.altair_chart(alt.Chart(ts).mark_line().encode(x="date:T", y="compound:Q").properties(height=230), use_container_width=True)
            # examples
            st.markdown("**Examples**")
            lft, rgt = st.columns(2)
            with lft:
                st.caption("Top positive")
                st.dataframe(safe_top_examples(sdf, 10, True), use_container_width=True, height=280)
            with rgt:
                st.caption("Top negative")
                st.dataframe(safe_top_examples(sdf, 10, False), use_container_width=True, height=280)
            # export
            st.download_button("💾 Download scored CSV", data=sdf.to_csv(index=False).encode("utf-8"), file_name="youtube_comments_scored.csv", mime="text/csv")

with tab_keywords:
    st.subheader("Keyword Explorer")
    if "comments_df" not in st.session_state or st.session_state["comments_df"].empty:
        st.info("Fetch comments first on the **Fetch** tab.")
    else:
        df = st.session_state["comments_df"].copy()
        topk = st.slider("Top terms", 5, 40, 20, 1)
        ngram_choice = st.selectbox("N-grams", ["unigram","bigram","unigram+bigram"], 2)
        ngr = (1,2) if ngram_choice=="unigram+bigram" else ((1,1) if ngram_choice=="unigram" else (2,2))
        freq_df = ngram_frequencies(df["text"].tolist(), ngram_range=ngr, top_k=topk, language="english")
        st.dataframe(freq_df, use_container_width=True, height=350)
        st.altair_chart(alt.Chart(freq_df).mark_bar().encode(x=alt.X("term:N", sort="-y"), y="count:Q").properties(height=240), use_container_width=True)

st.markdown("---")
st.caption("Note: VADER is fast and English-focused. For multilingual, try a multilingual Transformers model (heavier).")
