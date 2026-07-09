"""
streamlit_app.py — Frontend dashboard for the Resume Screener.

Run with:
    streamlit run app/streamlit_app.py

This talks directly to the models/ modules (not through the FastAPI
layer) so it can run standalone.
"""

import sys
import os

# Allow running `streamlit run app/streamlit_app.py` from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from preprocessing.pdf_extractor import extract_text_from_bytes
from models.matcher import compute_match
from models.classifier import load_classifier, predict_category

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screener")
st.caption("Upload a resume and paste a job description to get a match score, "
           "keyword breakdown, and predicted job category.")

# --- Load classifier once (cached across reruns) -----------------------
@st.cache_resource
def get_classifier():
    try:
        return load_classifier()
    except FileNotFoundError:
        return None

classifier = get_classifier()

# --- Layout: two columns for inputs -------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

with col2:
    st.subheader("2. Paste Job Description")
    jd_text = st.text_area("Job description", height=220,
                            placeholder="Paste the full job description here...")

analyze_clicked = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

if analyze_clicked:
    if uploaded_file is None:
        st.warning("Please upload a resume file first.")
    elif not jd_text.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Extracting resume text..."):
            try:
                resume_text = extract_text_from_bytes(uploaded_file.read(), uploaded_file.name)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        if not resume_text.strip():
            st.error("Could not extract any text from this file. Try a different file.")
            st.stop()

        with st.spinner("Computing match score..."):
            result = compute_match(resume_text, jd_text)

        st.divider()
        st.subheader("Results")

        # --- Match score gauge-style display -----------------------------
        score = result["match_score"]
        score_col, category_col = st.columns(2)

        with score_col:
            st.metric("Match Score", f"{score}%")
            st.progress(min(int(score), 100) / 100)
            if score >= 70:
                st.success("Strong match!")
            elif score >= 45:
                st.info("Moderate match.")
            else:
                st.warning("Low match — resume may not align well with this JD.")

        with category_col:
            if classifier is not None:
                category, confidence = predict_category(classifier, resume_text)
                st.metric("Predicted Category", category)
                st.caption(f"Confidence: {confidence}%")
            else:
                st.info("Category classifier not trained yet. "
                        "Run `python -m models.classifier` to enable this.")

        # --- Keyword breakdown ---------------------------------------------
        st.divider()
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown("**✅ Matched Keywords**")
            if result["matched_keywords"]:
                st.write(", ".join(result["matched_keywords"]))
            else:
                st.write("None found.")

        with kw_col2:
            st.markdown("**❌ Missing Keywords**")
            if result["missing_keywords"]:
                st.write(", ".join(result["missing_keywords"]))
            else:
                st.write("None — great coverage!")

        # --- Extracted text preview ----------------------------------------
        with st.expander("View extracted resume text"):
            st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

st.divider()
st.caption("Built with Sentence-Transformers, scikit-learn, FastAPI, and Streamlit.")