
# YouTube Comment Sentiment App (Streamlit, Colab-friendly)
# --------------------------------------------------------
# Features
# - Input: YouTube API key + video URL/ID
# - Fetch all (up to user-specified limit) top-level comments and (optionally) replies
# - Save/export raw comments to CSV
# - Sentiment analysis (VADER fast; optional Transformers)
# - Visualizations: distribution, time series, top +/- comments, n-gram frequency
#
# Colab quick start:
# !pip -q install -r requirements_youtube.txt
# from pyngrok import ngrok, conf; import os
# os.environ["STREAMLIT_SERVER_PORT"] = "8501"
# public_url = ngrok.connect(8501)
# public_url
# !streamlit run streamlit_youtube_sentiment.py --server.headless true --server.port 8501
#
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import altair as alt

# NLP bits
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Optional advanced model
try:
    from transformers import pipeline   # heavy; used only if user toggles
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="YouTube Comment Sentiment", layout="wide")

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
try:
    nltk.data.find("corpora/stopwords.zip")
except LookupError:
    nltk.download("stopwords")

# ------------- Helpers -------------

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_COMMENT_URL = "https://www.googleapis.com/youtube/v3/comments"

def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract a YouTube video ID from a URL or return as-is if already an ID."""
    s = url_or_id.strip()
    # Direct ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", s):
        return s
    # Common URL patterns
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)
    return None

@st.cache_data(show_spinner=False)
def fetch_comments(api_key: str, video_id: str, max_count: int = 1000, order: str = "relevance", include_replies: bool = True) -> pd.DataFrame:
    """Fetch comments using YouTube Data API v3 (API key). Returns a DataFrame."""
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
        resp = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=30)
        if resp.status_code != 200:
            try:
                msg = resp.json()
            except Exception:
                msg = {"error": {"message": resp.text}}
            raise RuntimeError(f"API error {resp.status_code}: {msg}")
        data = resp.json()

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": top.get("authorDisplayName"),
                "text": top.get("textOriginal", ""),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt"),
                "updated_at": top.get("updatedAt"),
                "is_reply": 0,
            })
            fetched += 1
            if fetched >= max_count:
                break

            if include_replies:
                for r in item.get("replies", {}).get("comments", []):
                    rs = r["snippet"]
                    rows.append({
                        "comment_id": r["id"],
                        "author": rs.get("authorDisplayName"),
                        "text": rs.get("textOriginal", ""),
                        "like_count": rs.get("likeCount", 0),
                        "published_at": rs.get("publishedAt"),
                        "updated_at": rs.get("updatedAt"),
                        "is_reply": 1,
                    })
                    fetched += 1
                    if fetched >= max_count:
                        break
        if fetched >= max_count:
            break

        next_token = data.get("nextPageToken")
        if not next_token:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
        df["text"] = df["text"].fillna("")
    return df

def run_vader(texts: List[str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t) for t in texts]
    out = pd.DataFrame(scores)
    # Map to label using standard VADER thresholds
    def label(c):
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"
    out["label"] = out["compound"].apply(label)
    return out

@st.cache_resource(show_spinner=False)
def load_transformer(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. Install 'transformers' to enable advanced analysis.")
    return pipeline("sentiment-analysis", model=model_name)

def run_transformer(texts: List[str], model_name: str) -> pd.DataFrame:
    clf = load_transformer(model_name)
    # batch predictions to speed up
    preds = []
    batch = 64
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        preds.extend(clf(chunk))
    # Normalize to 'positive'/'negative' only (no neutral in this model)
    dfp = pd.DataFrame(preds)
    # Some models return 'LABEL_0/1', map common cases
    dfp["label"] = dfp["label"].replace({
        "NEGATIVE": "negative",
        "POSITIVE": "positive",
        "LABEL_0": "negative",
        "LABEL_1": "positive",
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive",
    })
    # Convert score to signed 'compound-like' proxy
    dfp["compound"] = dfp.apply(lambda r: r["score"] if r["label"]=="positive" else (-r["score"] if r["label"]=="negative" else 0.0), axis=1)
    # Add vader-like components as NaN to keep schema similar
    for k in ["neg","neu","pos"]:
        dfp[k] = np.nan
    return dfp[["neg","neu","pos","compound","label"]]

def top_k_examples(df: pd.DataFrame, k: int = 10, positive=True) -> pd.DataFrame:
    sign = 1 if positive else -1
    return df.sort_values(sign*df["compound"], ascending=False).head(k)[["author","text","like_count","compound","label","published_at"]]

def ngram_frequencies(texts: List[str], ngram_range=(1,2), top_k: int = 20, stop_lang: str = "english"):
    stop = set(stopwords.words(stop_lang)) if stop_lang in stopwords.fileids() else set()
    cv = CountVectorizer(stop_words=stop, ngram_range=ngram_range, min_df=2)
    try:
        mat = cv.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["term","count"])
    sums = np.asarray(mat.sum(axis=0)).ravel()
    terms = np.array(cv.get_feature_names_out())
    idx = np.argsort(-sums)[:top_k]
    return pd.DataFrame({"term": terms[idx], "count": sums[idx].astype(int)})

# ------------- UI -------------

st.title("🎬 YouTube Comment Sentiment")
st.caption("Paste a video URL or ID, fetch comments, then analyze sentiment.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("YouTube API key", type="password", help="API key only (no OAuth). Get it from Google Cloud Console > APIs & Services.")
    url_or_id = st.text_input("Video URL or ID", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    max_comments = st.number_input("Max comments to fetch", min_value=50, max_value=20000, value=1500, step=50)
    order = st.selectbox("Order", options=["relevance", "time"], index=0)
    include_replies = st.checkbox("Include replies", value=True)
    st.divider()
    st.markdown("**Sentiment Engine**")
    engine = st.radio("Choose analyzer", ["VADER (fast)", "Transformers (advanced)"], index=0)
    model_name = st.text_input("Transformers model (optional)", value="distilbert-base-uncased-finetuned-sst-2-english", help="Change for multilingual (e.g., 'nlptown/bert-base-multilingual-uncased-sentiment').")
    st.caption("Tip: For multilingual videos, try the nlptown model and map stars to pos/neg/neutral via label mapping.")

fetch_col, save_col = st.columns([1,1])
with fetch_col:
    clicked = st.button("🚀 Fetch Comments", type="primary")
with save_col:
    st.write("")

if clicked:
    if not api_key or not url_or_id:
        st.error("Please provide both API key and video URL/ID.")
        st.stop()
    vid = extract_video_id(url_or_id)
    if not vid:
        st.error("Could not parse a valid video ID from the input.")
        st.stop()
    with st.spinner("Fetching comments from YouTube..."):
        try:
            comments_df = fetch_comments(api_key, vid, int(max_comments), order=order, include_replies=include_replies)
        except Exception as e:
            st.error(f"Failed to fetch comments: {e}")
            st.stop()

    if comments_df.empty:
        st.warning("No comments found for this video (or filtered out).")
        st.stop()

    # Store in session
    st.session_state["comments_df"] = comments_df

# If we have data, show tabs
if "comments_df" in st.session_state:
    df = st.session_state["comments_df"].copy()
    df["date"] = df["published_at"].dt.tz_localize(None)
    tabs = st.tabs(["🗂️ Data", "🧠 Sentiment", "🔎 Keywords"])

    with tabs[0]:
        st.subheader("Raw Comments")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total comments", f"{len(df):,}")
        c2.metric("Top-level", f"{(df['is_reply']==0).sum():,}")
        c3.metric("Replies", f"{(df['is_reply']==1).sum():,}")
        c4.metric("Median likes", f"{int(df['like_count'].median()):,}")

        st.dataframe(df[["author","text","like_count","published_at","is_reply"]], use_container_width=True, height=350)

        # Time distribution
        st.markdown("**Comment volume over time**")
        vol = df.groupby(pd.Grouper(key="date", freq="D")).size().reset_index(name="count")
        st.altair_chart(
            alt.Chart(vol).mark_bar().encode(x="date:T", y="count:Q").properties(height=220),
            use_container_width=True
        )

        # Top commenters
        st.markdown("**Top commenters (by count)**")
        topc = df["author"].value_counts().head(15).reset_index()
        topc.columns = ["author","count"]
        st.altair_chart(
            alt.Chart(topc).mark_bar().encode(x=alt.X("author:N", sort="-y"), y="count:Q", tooltip=["author","count"]).properties(height=220),
            use_container_width=True
        )

        # Export
        st.download_button("💾 Download comments CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="youtube_comments.csv", mime="text/csv")

    with tabs[1]:
        st.subheader("Sentiment Analysis")
        if engine.startswith("VADER"):
            with st.spinner("Scoring with VADER..."):
                sent = run_vader(df["text"].tolist())
        else:
            if not TRANSFORMERS_AVAILABLE:
                st.error("Transformers not installed. Add 'transformers' to requirements and retry, or use VADER.")
                st.stop()
            with st.spinner("Scoring with Transformers (this can take a bit)..."):
                sent = run_transformer(df["text"].tolist(), model_name=model_name)

        sdf = pd.concat([df.reset_index(drop=True), sent.reset_index(drop=True)], axis=1)

        # Overview
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Positive", f"{(sdf['label']=='positive').mean()*100:.1f}%")
        colB.metric("Neutral", f"{(sdf['label']=='neutral').mean()*100:.1f}%")
        colC.metric("Negative", f"{(sdf['label']=='negative').mean()*100:.1f}%")
        colD.metric("Avg compound", f"{sdf['compound'].mean():.3f}")

        # Distribution
        st.markdown("**Distribution of sentiment**")
        dist = sdf["label"].value_counts(normalize=True).rename_axis("label").reset_index(name="share")
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(x=alt.X("label:N", sort="-y"), y=alt.Y("share:Q", axis=alt.Axis(format="%")), tooltip=["label","share"]).properties(height=240),
            use_container_width=True
        )

        # Time series of rolling compound
        st.markdown("**Sentiment over time (7-day MA of compound)**")
        ts = sdf.groupby(pd.Grouper(key="date", freq="D"))["compound"].mean().rolling(7, min_periods=1).mean().reset_index()
        st.altair_chart(
            alt.Chart(ts).mark_line().encode(x="date:T", y="compound:Q").properties(height=230),
            use_container_width=True
        )

        # Top positive/negative examples
        st.markdown("**Examples**")
        left, right = st.columns(2)
        with left:
            st.caption("Top positive")
            st.dataframe(top_k_examples(sdf, 10, positive=True), use_container_width=True, height=280)
        with right:
            st.caption("Top negative")
            st.dataframe(top_k_examples(sdf, 10, positive=False), use_container_width=True, height=280)

        # Export scored data
        st.download_button("💾 Download scored CSV", data=sdf.to_csv(index=False).encode("utf-8"), file_name="youtube_comments_scored.csv", mime="text/csv")

    with tabs[2]:
        st.subheader("Keyword Explorer")
        st.caption("Simple n-gram frequencies (unigrams & bigrams) with stopword removal (English).")
        topk = st.slider("Top terms", 5, 40, 20, 1)
        ngram = st.selectbox("N-grams", ["unigram", "unigram+bigram", "bigram"], index=1)
        if ngram == "unigram":
            ngr = (1,1)
        elif ngram == "bigram":
            ngr = (2,2)
        else:
            ngr = (1,2)
        freq_df = ngram_frequencies(df["text"].tolist(), ngram_range=ngr, top_k=topk, stop_lang="english")
        st.dataframe(freq_df, use_container_width=True, height=350)
        st.altair_chart(
            alt.Chart(freq_df).mark_bar().encode(x=alt.X("term:N", sort="-y"), y="count:Q", tooltip=["term","count"]).properties(height=240),
            use_container_width=True
        )

st.markdown("---")
st.caption("Note: VADER works best for English. For multilingual videos, try a multilingual Transformers model (heavier).')
