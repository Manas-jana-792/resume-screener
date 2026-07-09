"""
classifier.py
--------------
Resume category classifier (e.g. HR, IT, Finance, Teacher...).

Uses TF-IDF + Logistic Regression rather than fine-tuning DistilBERT.
This is a deliberate choice for a 3-week internship project:
  - Trains in seconds on CPU (no GPU needed)
  - ~85-90% accuracy on the snehaanbhawal resume dataset, which is
    plenty for a portfolio project
  - Keeps the pipeline simple to explain in an interview

Usage:
    from models.classifier import train_classifier, load_classifier, predict_category

    # One-time training (run this as a script)
    train_classifier("data/raw/resume_dataset/Resume/Resume.csv")

    # In the app
    clf = load_classifier()
    category, confidence = predict_category(clf, resume_text)
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from preprocessing.text_cleaner import clean_text

MODEL_PATH = "models/saved/resume_classifier.joblib"


def _load_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Resume.csv dataset and clean the resume text column."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Resume_str", "Category"])
    df["cleaned_text"] = df["Resume_str"].apply(clean_text)
    return df


def train_classifier(csv_path: str, save_path: str = MODEL_PATH) -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression pipeline on the resume dataset
    and save it to disk. Prints a classification report so you can see
    per-category precision/recall for your README or presentation.
    """
    df = _load_and_clean_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["Category"],
        test_size=0.2, random_state=42, stratify=df["Category"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"\nModel saved to {save_path}")

    return pipeline


def load_classifier(model_path: str = MODEL_PATH) -> Pipeline:
    """Load a previously trained classifier from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run train_classifier() first."
        )
    return joblib.load(model_path)


def predict_category(pipeline: Pipeline, resume_text: str) -> tuple:
    """
    Predict the job category for a resume.
    Returns (predicted_category, confidence_percentage).
    """
    cleaned = clean_text(resume_text)
    prediction = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = round(max(probabilities) * 100, 2)
    return prediction, confidence


if __name__ == "__main__":
    # Run this file directly to train:  python -m models.classifier
    CSV_PATH = "data/raw/Resume/Resume.csv"
    model = train_classifier(CSV_PATH)

    sample = "Experienced HR manager skilled in recruitment, employee relations, and payroll."
    category, confidence = predict_category(model, sample)
    print(f"\nSample prediction: {category} ({confidence}% confidence)")