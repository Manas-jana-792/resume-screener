"""
main.py — FastAPI backend for the Resume Screener.

Run with:
    uvicorn api.main:app --reload

Endpoints:
    POST /match-score     -> resume + JD text  -> similarity score
    POST /upload-resume   -> resume file        -> extracted text
    POST /classify        -> resume text        -> predicted category
    POST /full-analysis   -> resume file + JD    -> everything combined
    GET  /health          -> simple health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing.pdf_extractor import extract_text_from_bytes
from models.matcher import compute_match
from models.classifier import load_classifier, predict_category

app = FastAPI(
    title="Resume Screener API",
    description="AI-powered resume-to-JD matching and category classification",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the classifier once at startup rather than per-request
try:
    _classifier = load_classifier()
except FileNotFoundError:
    _classifier = None  # /classify will return a clear error until trained


class MatchRequest(BaseModel):
    resume_text: str
    jd_text: str


class ClassifyRequest(BaseModel):
    resume_text: str


@app.get("/health")
def health_check():
    return {"status": "ok", "classifier_loaded": _classifier is not None}


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Extract raw text from an uploaded PDF/DOCX/TXT resume."""
    file_bytes = await file.read()
    try:
        text = extract_text_from_bytes(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    return {"filename": file.filename, "extracted_text": text}


@app.post("/match-score")
def match_score(payload: MatchRequest):
    """Compute a semantic match score between resume text and JD text."""
    if not payload.resume_text.strip() or not payload.jd_text.strip():
        raise HTTPException(status_code=400, detail="Both resume_text and jd_text are required.")

    return compute_match(payload.resume_text, payload.jd_text)


@app.post("/classify")
def classify(payload: ClassifyRequest):
    """Predict the job category of a resume."""
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not found. Run `python -m models.classifier` to train it first."
        )

    category, confidence = predict_category(_classifier, payload.resume_text)
    return {"predicted_category": category, "confidence": confidence}


@app.post("/full-analysis")
async def full_analysis(jd_text: str, file: UploadFile = File(...)):
    """
    One-stop endpoint: upload a resume file + paste a JD, get back
    extracted text, match score, keyword breakdown, and predicted category.
    """
    file_bytes = await file.read()
    try:
        resume_text = extract_text_from_bytes(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    match_result = compute_match(resume_text, jd_text)

    classification_result = None
    if _classifier is not None:
        category, confidence = predict_category(_classifier, resume_text)
        classification_result = {"predicted_category": category, "confidence": confidence}

    return {
        "filename": file.filename,
        "extracted_text_preview": resume_text[:300] + ("..." if len(resume_text) > 300 else ""),
        "match_result": match_result,
        "classification": classification_result,
    }