# 🎬 YouTube AI Content Analyzer

An AI-powered YouTube analysis application built with **Python, Gradio, Machine Learning, NLP, and AI APIs**.

The application analyzes YouTube videos by retrieving transcripts, generating summaries, recommending related videos, and performing sentiment analysis on YouTube comments.

---

## 🚀 Features

### 🎥 Video Summarization
- Accepts a YouTube video URL
- Retrieves the video's available transcript
- Supports captions in different languages
- Generates an AI-powered summary
- Provides:
  - Overview
  - Key points
  - Takeaway

### 🔎 Video Recommendations
- Analyzes the content/topic of the selected video
- Uses text similarity to find related YouTube videos
- Provides recommended videos based on content

### 😊 Comment Sentiment Analysis
- Retrieves YouTube comments
- Uses a trained Machine Learning model to classify sentiment
- Supports:
  - Positive
  - Negative
  - Neutral
- Displays sentiment analysis results and visualization

### 🤖 Machine Learning
The project includes a sentiment classification model using:

- TF-IDF Vectorization
- Logistic Regression
- Train/Test Split
- Scikit-learn

### 📊 Model Information
The application provides information about the trained Machine Learning model and its performance.

### 🖥️ Gradio Interface
The entire project is integrated into an easy-to-use Gradio web interface.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### AI & Machine Learning
- Scikit-learn
- Logistic Regression
- TF-IDF
- NLP
- Gemini API

### Data Processing
- Pandas
- NumPy
- Regular Expressions

### Visualization
- Matplotlib

### Web Interface
- Gradio

### APIs & Libraries
- YouTube Data API
- YouTube Transcript API
- Google Gemini API
- python-dotenv

---

## 📂 Project Structure

```text
YouTube-project/
│
├── app.py
├── Training.py
├── requirements.txt
├── .env
├── .gitignore
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
│
└── README.md
