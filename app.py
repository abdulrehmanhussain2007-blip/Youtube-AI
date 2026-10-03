import os
import re

import matplotlib
matplotlib.use("Agg")  # no GUI needed, avoids backend errors
import matplotlib.pyplot as plt

import gradio as gr
import joblib

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from  google import genai
from google.genai import errors,types

load_dotenv()

# ==========================================================
# CONFIGURATION
# ==========================================================

MAX_COMMENTS = 1000
SUMMARY_CHUNK_SIZE = 2000       # small chunks 
NEUTRAL_THRESHOLD = 0.55        # low-confidence predictions count as neutral
MAX_RECOMMENDATIONS = 5         # only 5 recomendations


# ==========================================================
# YOUTUBE API KEY (set as an environment variable)
# ==========================================================
# Windows CMD:   set YOUTUBE_API_KEY=YOUR_KEY
# PowerShell:    $env:YOUTUBE_API_KEY="YOUR_KEY"
# Mac/Linux:     export YOUTUBE_API_KEY="YOUR_KEY"

API_KEY = os.getenv("YOUTUBE_API_KEY")


def check_api_key():
    return bool(API_KEY) and API_KEY != "YOUR_YOUTUBE_API_KEY"


# ==========================================================
# LOAD MACHINE LEARNING MODEL
# ==========================================================

try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ML_MODEL_LOADED = True
except Exception:
    model = None
    vectorizer = None
    ML_MODEL_LOADED = False

try:
    with open("model_accuracy.txt", "r") as f:
        MODEL_ACCURACY = float(f.read())
except Exception:
    MODEL_ACCURACY = None



# ==========================================================
# HELPERS
# ==========================================================

def get_video_id(url):
    if not url:
        return None

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_youtube_service():
    if not check_api_key():
        raise Exception("YouTube API key is not configured.")
    return build("youtube", "v3", developerKey=API_KEY)

#==========================================================
# QUOTA ERROR HANDLER
# ==========================================================

def youtube_error_message(error):

    error_text = str(error).lower()

    if (
        "quotaexceeded" in error_text
        or "quota exceeded" in error_text
        or "daily limit" in error_text
    ):
        return (
            "⚠️ **YouTube API quota has been reached.**\n\n"
            "The application has reached its daily YouTube API "
            "quota. Please try again after the quota resets."
        )

    if "commentsdisabled" in error_text:
        return (
            "⚠️ **Comments are disabled for this video.**"
        )

    if "video not found" in error_text:
        return (
            "⚠️ **Video not found.**"
        )

    return "❌ YouTube API Error:\n\n" + str(error)


#

# ==========================================================
# TRANSCRIPT
# ==========================================================

from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # Try to get English captions first
    try:
        transcript = transcript_list.find_transcript(["en"])
        fetched = transcript.fetch()

        print("Using English captions")

    except Exception:
        # English captions not found.
        # Try Hindi or another available language.
        transcript = None

        for language in transcript_list:
            print("Available caption language:", language.language_code)

            try:
                if language.language_code == "hi":
                    transcript = language
                    break
            except Exception:
                continue

        if transcript is None:
            raise Exception("No usable captions were found.")

        # Translate Hindi captions to English if possible
        try:
            if transcript.is_translatable:
                transcript = transcript.translate("en")
                print("Translated Hindi captions to English")
            else:
                print("Hindi captions are not translatable; using Hindi")

        except Exception as e:
            print("Translation unavailable:", e)

        fetched = transcript.fetch()

    return " ".join(segment.text for segment in fetched)

def split_text(text, chunk_size=SUMMARY_CHUNK_SIZE):
    chunks, current, length = [], [], 0

    for word in text.split():
        current.append(word)
        length += len(word) + 1

        if length >= chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks

GEMINI_MODEL = "gemini-3.8-flash"

FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

PROMPT = """
Summarize this YouTube transcript in English.

Give:
1. A 2-line overview
2. Five key points as bullets
3. A one-line takeaway

Use Markdown headings: Overview, Key points, and Takeaway.

Transcript:
{text}
"""

_client = None


def llm_summary(text):
    global _client

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from your .env file.")

    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=45000,
                retry_options=types.HttpRetryOptions(attempts=1)
            )
        )

    last_error = None

    for model in [GEMINI_MODEL, *FALLBACK_MODELS]:
        try:
            response = _client.models.generate_content(
                model=model,
                contents=PROMPT.format(text=text)
            )

            if response.text:
                return response.text.strip()

            raise ValueError("Gemini returned an empty summary.")

        except errors.ClientError as e:
            if e.code != 429:
                raise

            print(f"{model} rate-limited. Trying the next model.")
            last_error = e

        except Exception as e:
            print(f"{model} failed: {e}")
            last_error = e

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}")
# ==========================================================
# VIDEO SUMMARIZATION
# ==========================================================

def summarize_video(url):
    if not url:
        return "❌ Please enter a YouTube URL."

    video_id = get_video_id(url)

    if video_id is None:
        return "❌ Invalid YouTube URL."

    try:
        print("Getting transcript...")

        text = get_transcript(video_id)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 100:
            return "❌ Transcript is too short to summarize."

        chunks = split_text(text)
        summaries = []

        print("Total chunks:", len(chunks))

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 100:
                continue

            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

            summary = llm_summary(chunk)

            if summary:
                summaries.append(summary)

        if not summaries:
            return "❌ Could not generate a summary."

        # Combine chunk summaries into one final summary
        combined_summary = "\n\n".join(summaries)

        print("Creating final summary...")

        final_summary = llm_summary(combined_summary)

        return (
            "## 🎬 Video Summary\n\n"
            + final_summary
            + f"\n\n**Transcript chunks processed:** {len(summaries)}"
        )

    except Exception as e:
        print("Summarization error:", e)

        return (
            "❌ Summarization Error:\n\n"
            + str(e)
        )
# ==========================================================
# VIDEO RECOMMENDATIONS
# ==========================================================

def recommend_videos(topic):
    if not topic:
        return "❌ Please enter a topic."

    if not check_api_key():
        return "❌ YouTube API key is not configured."

    try:
        youtube = get_youtube_service()

        response = youtube.search().list(
            part="snippet",
            q=topic,
            type="video",
            maxResults=10,
            order="relevance",
            relevanceLanguage="en",
            videoCaption="closedCaption",  # only videos that can be summarized
        ).execute()

        items = response.get("items", [])

        if not items:
            return "❌ No videos found."

        output = "## 🔎 Recommended Videos\n\n"

        for item in items:
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            channel = item["snippet"]["channelTitle"]
            description = item["snippet"].get("description", "")

            if len(description) > 200:
                description = description[:200] + "..."

            link = "https://www.youtube.com/watch?v=" + video_id

            output += (
                f"### 🎥 {title}\n\n"
                f"**Channel:** {channel}\n\n"
                f"{description}\n\n"
                f"[▶ Watch Video]({link})\n\n"
                "---\n\n"
            )

        return output

    except Exception as e:
        return "❌ YouTube API Error:\n\n" + str(e)


# ==========================================================
# COMMENTS
# ==========================================================

def get_comments(video_id, max_comments=MAX_COMMENTS):
    youtube = get_youtube_service()

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        remaining = max_comments - len(comments)

        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, remaining),
            textFormat="plainText",
            pageToken=next_page_token,
        ).execute()

        for item in response.get("items", []):
            try:
                comments.append(
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                )
            except KeyError:
                continue

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


# ==========================================================
# SENTIMENT CHART
# ==========================================================

def create_sentiment_chart(positive, negative, neutral):
    fig = plt.figure(figsize=(7, 5))

    plt.bar(
        ["Positive", "Negative", "Neutral"],
        [positive, negative, neutral],
        color=["#2e7d32", "#c62828", "#757575"],
    )

    plt.title("YouTube Comment Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.tight_layout()

    plt.close(fig)  # prevents figures piling up in memory
    return fig


# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================

def analyze_sentiment(url):
    if not url:
        return "❌ Please enter a YouTube URL.", None

    video_id = get_video_id(url)
    if video_id is None:
        return "❌ Invalid YouTube URL.", None

    if not ML_MODEL_LOADED:
        return "❌ Trained ML model not found.\n\nRun train_model.py first.", None

    if not check_api_key():
        return "❌ YouTube API key is not configured.", None

    try:
        comments = get_comments(video_id)

        if not comments:
            return "❌ No comments were found.", None

        positive = negative = neutral = 0

        # Classify all comments at once (much faster than a loop)
        vectors = vectorizer.transform(comments)
        probabilities = model.predict_proba(vectors)
        classes = [str(c).lower().strip() for c in model.classes_]

        for p in probabilities:
            if p.max() < NEUTRAL_THRESHOLD:
                neutral += 1
                continue

            label = classes[p.argmax()]

            if label == "positive":
                positive += 1
            elif label == "negative":
                negative += 1
            else:
                neutral += 1

        total = positive + negative + neutral

        positive_percent = positive / total * 100
        negative_percent = negative / total * 100
        neutral_percent = neutral / total * 100

        accuracy_text = (
            f"{MODEL_ACCURACY * 100:.2f}%"
            if MODEL_ACCURACY is not None
            else "Not available"
        )

        output = f"""
# 😊 YouTube Sentiment Analysis

### Comments analyzed

**{total}**

---

### Results

🟢 **Positive:** {positive}  
**{positive_percent:.1f}%**

🔴 **Negative:** {negative}  
**{negative_percent:.1f}%**

⚪ **Neutral:** {neutral}  
**{neutral_percent:.1f}%**

---

### Machine Learning Model

**TF-IDF + Logistic Regression**

**Test Accuracy:** {accuracy_text}

Comments the model is unsure about (confidence below {NEUTRAL_THRESHOLD:.0%}) are counted as neutral.
"""

        chart = create_sentiment_chart(positive, negative, neutral)
        return output, chart

    except Exception as e:
        return "❌ Sentiment Analysis Error:\n\n" + str(e), None


# ==========================================================
# GRADIO APPLICATION
# ==========================================================

accuracy_display = (
    f"{MODEL_ACCURACY * 100:.2f}%"
    if MODEL_ACCURACY is not None
    else "Run train_model.py first"
)

with gr.Blocks(title="YouTube AI Content Analyzer") as app:

    gr.Markdown(
        """
# 🎥 YouTube AI Content Analyzer

### AI-powered YouTube analysis using Machine Learning, Deep Learning and the YouTube API.

**Features**

🎬 Video Summarization  
🔎 Video Recommendations  
😊 Comment Sentiment Analysis  
🤖 Machine Learning  
API's
"""
    )

    # ---------------- SUMMARIZATION ----------------
    with gr.Tab("🎬 Video Summarization"):
        gr.Markdown(
            """
### Enter a YouTube video URL

The application gets the transcript and uses a
**Gemini API** to generate a summary.
"""
        )

        video_url = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        summarize_button = gr.Button("🧠 Generate Summary")
        summary_output = gr.Markdown()

        summarize_button.click(
            summarize_video, inputs=video_url, outputs=summary_output
        )

    # ---------------- RECOMMENDATIONS ----------------
    with gr.Tab("🔎 Recommendations"):
        gr.Markdown(
            """
### Search for YouTube videos

Enter a topic and the YouTube Data API
will return relevant videos.
"""
        )

        topic = gr.Textbox(
            label="Topic",
            placeholder="Example: Machine Learning",
        )
        search_button = gr.Button("🔎 Find Videos")
        recommendations = gr.Markdown()

        search_button.click(
            recommend_videos, inputs=topic, outputs=recommendations
        )

    # ---------------- SENTIMENT ----------------
    with gr.Tab("😊 Sentiment Analysis"):
        gr.Markdown(
            """
### Analyze YouTube comments

The application collects comments and uses a
**trained TF-IDF + Logistic Regression model**
to classify them as Positive, Negative or Neutral.
"""
        )

        sentiment_url = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        sentiment_button = gr.Button("🤖 Analyze Comments")
        sentiment_output = gr.Markdown()
        sentiment_chart = gr.Plot(label="Sentiment Chart")

        sentiment_button.click(
            analyze_sentiment,
            inputs=sentiment_url,
            outputs=[sentiment_output, sentiment_chart],
        )

    # ---------------- MODEL INFORMATION ----------------
    with gr.Tab("📊 Model Information"):
        gr.Markdown(
            f"""
# 🤖 Machine Learning

**Algorithm:** Logistic Regression

**Text Processing:** TF-IDF

**Train/Test Split:** 80% / 20%

**Test Accuracy:** {accuracy_display}

---

# 🧠 Gemini API

---

# 🔎 YouTube API

Used for:

- Video search
- YouTube comments

---

# 🎯 Project Pipeline

```text
YouTube Video
      ↓
Transcript
      ↓
Gemini API`
      ↓
Summary
```
"""
        )


if __name__ == "__main__":
    app.launch()