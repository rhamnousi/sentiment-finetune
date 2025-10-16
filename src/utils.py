import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =========================
# Sentiment Mapping
# =========================
def map_sentiment(label):
    positive = ["Positive", "Joy", "Excitement", "Gratitude", "Serenity", "Happy",
                "Awe", "Hopeful", "Acceptance", "Euphoria", "Elation", "Enthusiasm",
                "Pride", "Determination", "Playful", "Inspiration", "Admiration",
                "Satisfaction", "Kind", "Adoration", "Nostalgia"]
    negative = ["Negative", "Despair", "Grief", "Sad", "Frustration", "Regret",
                "Melancholy", "Numbness", "Bad", "Hate", "Embarrassed", "Betrayal",
                "Anger", "Disgust", "Boredom", "Disappointment"]
    neutral = ["Neutral", "Contentment", "Curiosity", "Confusion", "Ambivalence",
               "Indifference", "Surprise"]
    if label in positive:
        return "Positive"
    elif label in negative:
        return "Negative"
    else:
        return "Neutral"

# =========================
# Dataset preprocessing
# =========================
def load_and_preprocess(csv_path, tokenizer, max_length=128):
    df = pd.read_csv(csv_path).rename(columns=lambda x: x.strip())
    df = df[["Text", "Sentiment"]].dropna()
    df["Sentiment"] = df["Sentiment"].str.strip().str.capitalize()
    df["Sentiment"] = df["Sentiment"].apply(map_sentiment)

    # Oversample minority classes
    df_positive = resample(df[df['Sentiment'] == 'Positive'], replace=True, n_samples=df.shape[0], random_state=42)
    df_neutral = resample(df[df['Sentiment'] == 'Neutral'], replace=True, n_samples=df.shape[0], random_state=42)
    df_negative = resample(df[df['Sentiment'] == 'Negative'], replace=True, n_samples=df.shape[0], random_state=42)
    df = pd.concat([df_positive, df_neutral, df_negative])

    # Encode labels
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
    df["label"] = df["Sentiment"].map(label_map).astype(int)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    def preprocess(examples):
        return tokenizer(examples["Text"], truncation=True, max_length=max_length)

    tokenized = ds.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    return tokenized, data_collator

# =========================
# Metrics
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }
