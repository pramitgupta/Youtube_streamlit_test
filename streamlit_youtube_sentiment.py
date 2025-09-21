
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
    # Ensure required columns exist
    needed = {"author","text","like_count","published_at","compound","label"}
    missing = [c for c in ["compound","label"] if c not in df.columns]
    if missing:
        # Try to fall back to a neutral compound if missing
        df = df.copy()
        if "score" in df.columns and "compound" not in df.columns:
            # sign score for label
            df["compound"] = df.apply(lambda r: r["score"] if str(r.get("label","")).lower()=="positive"
                                      else (-r["score"] if str(r.get("label","")).lower()=="negative" else 0.0), axis=1)
        elif "compound" not in df.columns:
            df["compound"] = 0.0
        if "label" not in df.columns:
            df["label"] = "neutral"
    tmp = df.copy()
    tmp["_sort"] = tmp["compound"] if positive else -tmp["compound"]
    cols = [c for c in ["author","text","like_count","compound","label","published_at"] if c in tmp.columns]
    return tmp.sort_values(by="_sort", ascending=False).head(k)[cols]

st.markdown("---")
st.caption("Note: VADER works best for English. For multilingual videos, try a multilingual Transformers model (heavier).")
