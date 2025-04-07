import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from skillmap.parser import parse_text_with_gemini
from skillmap.embedder import get_embedding
from skillmap.matcher import calculate_similarity, find_skill_gap
import fitz  # PyMuPDF
import re

def extract_text(file):
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
    else:
        return ""

    # ✅ Clean and normalize text
    text = re.sub(r"•", " ", text)                 # remove bullet chars
    text = re.sub(r"[\r\n]+", " ", text)           # replace newlines with space
    text = re.sub(r"\s{2,}", " ", text)            # collapse multiple spaces
    return text.strip()

# === Streamlit App ===
st.set_page_config(page_title="SkillMap Matcher", layout="centered")

st.title("🧠 SkillMap: Resume-to-Job Matcher")
st.markdown("Upload a resume and a job description (PDF/TXT) to analyze match and skill gap.")

# Upload files
resume_file = st.file_uploader("📄 Upload Resume", type=["txt", "pdf"])
job_file = st.file_uploader("📌 Upload Job Description", type=["txt", "pdf"])

# Read files and run pipeline
if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    with st.spinner("🔍 Parsing and analyzing..."):
        resume_data = parse_text_with_gemini(resume_text, "resume")
        job_data = parse_text_with_gemini(job_text, "job")

        resume_embedding = get_embedding(resume_text)
        job_embedding = get_embedding(job_text)

        score = calculate_similarity(resume_embedding, job_embedding)
        matched, missing = find_skill_gap(resume_data.get("skills", []), job_data.get("required_skills", []))

    # Display results
    st.subheader("🔗 Match Score")
    st.metric(label="Cosine Similarity", value=f"{score:.2f}")

    st.subheader("✅ Matched Skills")
    st.write(matched if matched else "No skills matched.")

    st.subheader("❌ Missing Skills")
    st.write(missing if missing else "No missing skills — great fit!")

    st.subheader("📋 Resume (parsed)")
    st.json(resume_data)

    st.subheader("📝 Job Description (parsed)")
    st.json(job_data)