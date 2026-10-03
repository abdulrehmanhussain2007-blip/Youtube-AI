import os
import re

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

MAX_COMMENTS = 50
MAX_RECOMMENDATIONS = 5
SUMMARY_CHUNK_SIZE = 2000
NEUTRAL_THRESHOLD = 0.55

API_KEY = os.getenv("YOUTUBE_API_KEY")

transcript_cache = {}
summary_cache = {}
comments_cache = {}
recommendation_cache = {}


# Load sentiment model
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ML_MODEL_LOADED = True
except Exception:
    model = None
    vectorizer = None
    ML_MODEL_LOADED = False


# Load accuracy
try:
    with open("model_accuracy.txt", "r") as f:
        MODEL_ACCURACY = float(f.read())
except Exception:
    MODEL_ACCURACY = None


# Load T5
print("Loading T5-small...")

try:
    summarizer = pipeline(
        "summarization",
        model="t5-small"
    )
    SUMMARIZER_LOADED = True
except Exception as e:
    print("T5-small could not be loaded:", e)
    summarizer = None
    SUMMARIZER_LOADED = False


def get_video_id(url):
    if not url:
        return None

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_youtube_service():
    if not API_KEY:
        raise Exception("YouTube API key is not configured.")

    return build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )


def youtube_error_message(error):
    text = str(error).lower()

    if "quotaexceeded" in text or "quota exceeded" in text:
        return "⚠️ YouTube API daily quota has been reached."

    if "commentsdisabled" in text:
        return "⚠️ Comments are disabled for this video."

    if "video not found" in text:
        return "⚠️ Video not found."

    return "❌ YouTube API Error:\n\n" + str(error)


# ---------------- TRANSCRIPT ----------------

def get_transcript(video_id):
    if video_id in transcript_cache:
        return transcript_cache[video_id]

    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(
            video_id,
            languages=["en", "en-US", "en-GB", "ur", "hi"]
        )
    except Exception:
        try:
            available = [
                t.language_code
                for t in api.list(video_id)
            ]

            if not available:
                raise Exception("No transcript available.")

            transcript = api.fetch(
                video_id,
                languages=available
            )

        except Exception:
            raise Exception(
                "Could not get transcript. "
                "The video may not have captions."
            )

    text = " ".join(
        item.text for item in transcript
    )

    transcript_cache[video_id] = text

    return text


def split_text(text, chunk_size=SUMMARY_CHUNK_SIZE):
    chunks = []
    current = []
    length = 0

    for word in text.split():
        current.append(word)
        length += len(word) + 1

        if length >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            length = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def summarize_chunk(text):
    result = summarizer(
        "summarize: " + text,
        max_length=150,
        min_length=40,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]


def summarize_video(url):
    if not url:
        return "❌ Please enter a YouTube URL."

    video_id = get_video_id(url)

    if not video_id:
        return "❌ Invalid YouTube URL."

    if not SUMMARIZER_LOADED:
        return "❌ T5-small could not be loaded."

    if video_id in summary_cache:
        return summary_cache[video_id]

    try:
        text = get_transcript(video_id)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 100:
            return "❌ Transcript is too short."

        chunks = split_text(text)

        summaries = []

        for chunk in chunks:
            if len(chunk) >= 100:
                summaries.append(
                    summarize_chunk(chunk)
                )

        if not summaries:
            return "❌ Could not generate summary."

        final_summary = " ".join(summaries)

        if len(final_summary) > 2500:
            final_summary = summarize_chunk(
                final_summary[:2500]
            )

        result = (
            "## 🎬 Video Summary\n\n"
            + final_summary
            + f"\n\n**Chunks processed:** {len(chunks)}"
        )

        summary_cache[video_id] = result

        return result

    except Exception as e:
        return "❌ Summarization Error:\n\n" + str(e)


# ---------------- RECOMMENDATIONS ----------------

def recommend_videos(topic):
    if not topic:
        return "❌ Please enter a topic."

    if not API_KEY:
        return "❌ YouTube API key is not configured."

    cache_key = topic.strip().lower()

    if cache_key in recommendation_cache:
        return recommendation_cache[cache_key]

    try:
        youtube = get_youtube_service()

        response = youtube.search().list(
            part="snippet",
            q=topic,
            type="video",
            maxResults=10,
            order="relevance",
            relevanceLanguage="en",
            videoCaption="closedCaption"
        ).execute()

        items = response.get("items", [])

        if not items:
            return "❌ No videos found."

        documents = []

        for item in items:
            title = item["snippet"].get("title", "")
            description = item["snippet"].get(
                "description", ""
            )

            documents.append(
                title + " " + description
            )

        all_documents = [topic] + documents

        tfidf = TfidfVectorizer(
            stop_words="english"
        )

        vectors = tfidf.fit_transform(
            all_documents
        )

        similarities = cosine_similarity(
            vectors[0:1],
            vectors[1:]
        )[0]

        ranked = sorted(
            zip(items, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        output = "## 🔎 AI Recommended Videos\n\n"

        for item, score in ranked[:MAX_RECOMMENDATIONS]:

            video_id = item["id"]["videoId"]

            title = item["snippet"]["title"]

            channel = item["snippet"]["channelTitle"]

            description = item["snippet"].get(
                "description", ""
            )

            if len(description) > 200:
                description = description[:200] + "..."

            link = (
                "https://www.youtube.com/watch?v="
                + video_id
            )

            output += (
                f"### 🎥 {title}\n\n"
                f"**Channel:** {channel}\n\n"
                f"**AI Relevance:** {score * 100:.1f}%\n\n"
                f"{description}\n\n"
                f"[▶ Watch Video]({link})\n\n"
                "---\n\n"
            )

        recommendation_cache[cache_key] = output

        return output

    except HttpError as e:
        return youtube_error_message(e)

    except Exception as e:
        return "❌ Recommendation Error:\n\n" + str(e)


# ---------------- COMMENTS ----------------

def get_comments(video_id):
    if video_id in comments_cache:
        return comments_cache[video_id]

    youtube = get_youtube_service()

    comments = []
    next_page_token = None

    while len(comments) < MAX_COMMENTS:

        remaining = MAX_COMMENTS - len(comments)

        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, remaining),
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            try:
                comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                    ["textDisplay"]
                )

                comments.append(comment)

            except KeyError:
                continue

        next_page_token = response.get(
            "nextPageToken"
        )

        if not next_page_token:
            break

    comments_cache[video_id] = comments

    return comments


# ---------------- SENTIMENT ----------------

def create_sentiment_chart(
    positive,
    negative,
    neutral
):
    fig = plt.figure(figsize=(7, 5))

    plt.bar(
        ["Positive", "Negative", "Neutral"],
        [positive, negative, neutral]
    )

    plt.title("YouTube Comment Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")

    plt.tight_layout()
    plt.close(fig)

    return fig


def analyze_sentiment(url):
    if not url:
        return "❌ Please enter a YouTube URL.", None

    video_id = get_video_id(url)

    if not video_id:
        return "❌ Invalid YouTube URL.", None

    if not ML_MODEL_LOADED:
        return (
            "❌ Trained ML model not found.\n\n"
            "Run Training.py first.",
            None
        )

    if not API_KEY:
        return "❌ YouTube API key is not configured.", None

    try:
        comments = get_comments(video_id)

        if not comments:
            return "❌ No comments were found.", None

        vectors = vectorizer.transform(comments)

        probabilities = model.predict_proba(
            vectors
        )

        classes = [
            str(c).lower().strip()
            for c in model.classes_
        ]

        positive = 0
        negative = 0
        neutral = 0

        for probability in probabilities:

            if probability.max() < NEUTRAL_THRESHOLD:
                neutral += 1
                continue

            label = classes[
                probability.argmax()
            ]

            if label == "positive":
                positive += 1

            elif label == "negative":
                negative += 1

            elif label == "neutral":
                neutral += 1

            else:
                neutral += 1

        total = positive + negative + neutral

        positive_percent = positive / total * 100
        negative_percent = negative / total * 100
        neutral_percent = neutral / total * 100

        accuracy = (
            f"{MODEL_ACCURACY * 100:.2f}%"
            if MODEL_ACCURACY is not None
            else "Not available"
        )

        output = f"""
# 😊 YouTube Sentiment Analysis

**Comments analyzed:** {total}

🟢 **Positive:** {positive} ({positive_percent:.1f}%)

🔴 **Negative:** {negative} ({negative_percent:.1f}%)

⚪ **Neutral:** {neutral} ({neutral_percent:.1f}%)

---

### Machine Learning Model

**TF-IDF + Logistic Regression**

**Test Accuracy:** {accuracy}
"""

        chart = create_sentiment_chart(
            positive,
            negative,
            neutral
        )

        return output, chart

    except HttpError as e:
        return youtube_error_message(e), None

    except Exception as e:
        return "❌ Sentiment Error:\n\n" + str(e), None


# ---------------- GRADIO ----------------

accuracy_display = (
    f"{MODEL_ACCURACY * 100:.2f}%"
    if MODEL_ACCURACY is not None
    else "Run Training.py first"
)


with gr.Blocks(
    title="YouTube AI Content Analyzer"
) as app:

    gr.Markdown(
        """
# 🎥 YouTube AI Content Analyzer

AI-powered YouTube analysis using
Machine Learning, Deep Learning and the YouTube API.
"""
    )

    # Summarization
    with gr.Tab("🎬 Summarization"):

        video_url = gr.Textbox(
            label="YouTube URL",
            placeholder="Paste YouTube video URL"
        )

        summarize_button = gr.Button(
            "🧠 Generate Summary"
        )

        summary_output = gr.Markdown()

        summarize_button.click(
            summarize_video,
            inputs=video_url,
            outputs=summary_output
        )

    # Recommendations
    with gr.Tab("🔎 Recommendations"):

        topic = gr.Textbox(
            label="Topic",
            placeholder="Example: Machine Learning"
        )

        search_button = gr.Button(
            "🔎 Find Videos"
        )

        recommendations = gr.Markdown()

        search_button.click(
            recommend_videos,
            inputs=topic,
            outputs=recommendations
        )

    # Sentiment
    with gr.Tab("😊 Sentiment Analysis"):

        sentiment_url = gr.Textbox(
            label="YouTube URL",
            placeholder="Paste YouTube video URL"
        )

        sentiment_button = gr.Button(
            "🤖 Analyze Comments"
        )

        sentiment_output = gr.Markdown()

        sentiment_chart = gr.Plot(
            label="Sentiment Chart"
        )

        sentiment_button.click(
            analyze_sentiment,
            inputs=sentiment_url,
            outputs=[
                sentiment_output,
                sentiment_chart
            ]
        )

    # Model information
    with gr.Tab("📊 Model Information"):

        gr.Markdown(
            f"""
## 🤖 Machine Learning

**Algorithm:** TF-IDF + Logistic Regression

**Train/Test Split:** 80% / 20%

**Test Accuracy:** {accuracy_display}

---

## 🧠 Deep Learning

**Model:** T5-small

**Task:** YouTube Video Summarization

---

## 🔎 Recommendation System

**Method:** Content-Based Recommendation

**TF-IDF + Cosine Similarity**

---

## 📺 YouTube API

Used for:

- Video search
- YouTube comments
"""
        )


if __name__ == "__main__":
    app.launch()