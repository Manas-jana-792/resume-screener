"""
text_cleaner.py
----------------
Text preprocessing utilities for cleaning resume and job description text
before embedding or classification.

Usage:
    from preprocessing.text_cleaner import clean_text, preprocess_pipeline

    cleaned = clean_text(raw_resume_text)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once, then cached)
_REQUIRED_NLTK_DATA = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
for _resource in _REQUIRED_NLTK_DATA:
    try:
        nltk.data.find(
            f"tokenizers/{_resource}"
            if "punkt" in _resource
            else f"corpora/{_resource}"
        )
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def remove_special_characters(text: str) -> str:
    """Remove URLs, emails, phone numbers, and non-alphanumeric characters."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\+?\d[\d\-\s()]{7,}\d", " ", text)       # phone numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def tokenize_and_clean(text: str, remove_stopwords: bool = True,
                        lemmatize: bool = True) -> list:
    """Tokenize text and optionally remove stopwords / lemmatize."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def clean_text(text: str, remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    """
    Full cleaning pipeline: lowercase -> remove special chars -> tokenize
    -> remove stopwords -> lemmatize -> join back to string.

    Returns a cleaned string (useful for embeddings, which work better
    on natural sentences rather than token lists).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_special_characters(text)
    tokens = tokenize_and_clean(text, remove_stopwords, lemmatize)
    return " ".join(tokens)


def preprocess_pipeline(text: str) -> dict:
    """
    Convenience wrapper that returns both the raw-cleaned sentence
    (for embeddings) and the token list (for keyword-based features).
    """
    cleaned_sentence = clean_text(text, remove_stopwords=False, lemmatize=False)
    keyword_tokens = tokenize_and_clean(text, remove_stopwords=True, lemmatize=True)
    return {
        "cleaned_sentence": cleaned_sentence,
        "tokens": keyword_tokens,
    }


if __name__ == "__main__":
    sample = """
    Contact: john.doe@email.com | +1 (555) 123-4567 | www.linkedin.com/in/johndoe
    Experienced Data Scientist with 5+ years in Python, Machine Learning,
    and Deep Learning. Skilled in TensorFlow, Scikit-learn, and NLP.
    """
    print("Original:\n", sample)
    print("\nCleaned:\n", clean_text(sample))