import pandas as pd
from datasets import load_dataset

# 1. Your labeled YouTube comments
yt = pd.read_csv("YoutubeCommentsDataSet.csv")
yt = yt.rename(columns={"Comment": "text", "Sentiment": "sentiment"})
yt = yt.dropna(subset=["text", "sentiment"])
yt["text"] = yt["text"].astype(str)
yt["sentiment"] = yt["sentiment"].str.lower().str.strip()

# 2. Tweets with sentiment labels
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
tw = pd.concat([ds[s].to_pandas() for s in ds])
tw["sentiment"] = tw["label"].map(
    {0: "negative", 1: "neutral", 2: "positive"}
)
tw["text"] = tw["text"].str.replace("@user", "", regex=False).str.strip()

# 3. Combine and save
df = pd.concat([yt[["text", "sentiment"]], tw[["text", "sentiment"]]])
df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=42)

df.to_csv("sentiment.csv", index=False)
print(len(df))
print(df["sentiment"].value_counts())

import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
df = pd.read_csv("sentiment.csv")

# 2. Prepare data
X = df["text"].astype(str)
y = df["sentiment"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save trained model
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")