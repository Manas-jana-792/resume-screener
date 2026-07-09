"""
matcher.py
-----------
Computes a match score between a resume and a job description using
cosine similarity of their embeddings, plus a simple keyword-overlap
signal that makes the score more explainable to end users.

Usage:
    from models.matcher import compute_match

    result = compute_match(resume_text, jd_text)
    # {'match_score': 78.4, 'matched_keywords': [...], 'missing_keywords': [...]}
"""

from sentence_transformers import util
from models.embedder import get_embedder
from preprocessing.text_cleaner import clean_text, tokenize_and_clean


def cosine_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Returns a 0-100 similarity score between resume and JD embeddings.
    """
    embedder = get_embedder()
    resume_vec = embedder.encode(clean_text(resume_text, remove_stopwords=False, lemmatize=False))
    jd_vec = embedder.encode(clean_text(jd_text, remove_stopwords=False, lemmatize=False))

    similarity = util.cos_sim(resume_vec, jd_vec).item()  # range -1 to 1
    score = max(0.0, similarity) * 100                     # clip negatives, scale to %
    return round(score, 2)


def keyword_overlap(resume_text: str, jd_text: str, top_n: int = 15) -> dict:
    """
    Explainability helper: shows which important JD keywords are present
    or missing in the resume. This is what gets displayed in the UI so
    the match score isn't a black box.
    """
    resume_tokens = set(tokenize_and_clean(resume_text))
    jd_tokens = tokenize_and_clean(jd_text)

    # Preserve order of first appearance, dedupe, cap at top_n
    seen = []
    for tok in jd_tokens:
        if tok not in seen:
            seen.append(tok)
    jd_keywords = seen[:top_n]

    matched = [kw for kw in jd_keywords if kw in resume_tokens]
    missing = [kw for kw in jd_keywords if kw not in resume_tokens]

    return {"matched_keywords": matched, "missing_keywords": missing}


def compute_match(resume_text: str, jd_text: str) -> dict:
    """
    Full match report combining semantic similarity + keyword overlap.
    """
    score = cosine_similarity_score(resume_text, jd_text)
    keywords = keyword_overlap(resume_text, jd_text)

    return {
        "match_score": score,
        "matched_keywords": keywords["matched_keywords"],
        "missing_keywords": keywords["missing_keywords"],
    }


if __name__ == "__main__":
    resume = "Experienced Python developer skilled in machine learning, NLP, and Flask APIs."
    jd = "Looking for a Python developer with NLP and REST API experience using Flask or FastAPI."

    result = compute_match(resume, jd)
    print("Match Score:", result["match_score"])
    print("Matched Keywords:", result["matched_keywords"])
    print("Missing Keywords:", result["missing_keywords"])