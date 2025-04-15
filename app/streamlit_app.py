import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from skillmap.parser import parse_text_with_gemini
from skillmap.embedder import get_embedding
from skillmap.matcher import calculate_similarity, find_skill_gap
from skillmap.enhancer import generate_resume_summary
from skillmap.enhancer_v2 import enhance_resume

import fitz  # PyMuPDF
import re
import pandas as pd
import json

def extract_text(file):
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
    else:
        return ""
    text = re.sub(r"•", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# === Streamlit App ===
st.set_page_config(page_title="SkillMap", layout="centered")
st.title("🧠 SkillMap")

mode = st.radio("Who are you?", ["🎯 Job Seeker", "🧠 Recruiter"])

# 🎯 Job Seeker Mode
if mode == "🎯 Job Seeker":
    st.markdown("Upload a resume and a job description (PDF/TXT), or paste the job description to analyze match and skill gap.")

    resume_file = st.file_uploader("📄 Upload Resume", type=["txt", "pdf"])

    job_input_mode = st.radio("📌 Job Description Input", ["Upload File", "Paste Text"])
    job_text = ""

    if job_input_mode == "Upload File":
        job_file = st.file_uploader("Upload Job Description (PDF/TXT)", type=["txt", "pdf"])
        if job_file:
            job_text = extract_text(job_file)
    else:
        job_text = st.text_area("Paste the job description below:", height=300)

    if resume_file and job_text.strip():
        resume_text = extract_text(resume_file)

        with st.spinner("🔍 Parsing and analyzing..."):
            resume_data = parse_text_with_gemini(resume_text, "resume")
            job_data = parse_text_with_gemini(job_text, "job")

            resume_embedding = get_embedding(resume_text)
            job_embedding = get_embedding(job_text)

            score = calculate_similarity(resume_embedding, job_embedding)
            matched, missing = find_skill_gap(resume_data.get("skills", []), job_data.get("required_skills", []))
            print("🔍 Using semantic skill matcher...")


        st.subheader("🔗 Match Score")
        st.metric(label="Cosine Similarity", value=f"{score:.2f}")

        st.subheader("✅ Matched Skills")
        st.write(matched if matched else "No skills matched.")

        st.subheader("❌ Missing Skills")
        st.write(missing if missing else "No missing skills — great fit!")

        # 🔧 Basic Summary Enhancer
        with st.expander("✨ Enhance Resume Summary"):
            if st.button("Generate AI Summary"):
                with st.spinner("Generating improved summary..."):
                    enhanced_summary = generate_resume_summary(resume_text, job_text)
                st.success("Here’s your improved summary:")
                st.write(enhanced_summary)

        # 🔧 Full Resume Enhancer
        with st.expander("🧠 Enhance Full Resume"):
            if st.button("Enhance My Resume Based on This Job"):
                with st.spinner("Rewriting key sections..."):
                    rewritten = enhance_resume(resume_data, job_data)

                st.subheader("🧠 Improved Summary")
                st.write(rewritten.get("summary", "No summary generated."))

                st.subheader("💼 Enhanced Experience")
                st.write(rewritten.get("experience", "No experience generated."))

                st.subheader("🛠 Refined Skills")
                st.write(rewritten.get("skills", "No skill update generated."))

                # 📥 Export
                st.download_button(
                    label="📥 Download Enhanced Sections (JSON)",
                    data=json.dumps(rewritten, indent=2),
                    file_name="enhanced_resume.json",
                    mime="application/json"
                )

        st.subheader("📋 Resume (parsed)")
        st.json(resume_data)

        st.subheader("📝 Job Description (parsed)")
        st.json(job_data)

# 🧠 Recruiter Mode
elif mode == "🧠 Recruiter":
    st.markdown("Upload one job description and multiple resumes (PDF/TXT) to rank candidates based on relevance.")

    job_file = st.file_uploader("📌 Upload Job Description", type=["txt", "pdf"])
    resume_files = st.file_uploader("📄 Upload Multiple Resumes", type=["txt", "pdf"], accept_multiple_files=True)

    if job_file and resume_files:
        job_text = extract_text(job_file)
        job_data = parse_text_with_gemini(job_text, "job")
        job_embedding = get_embedding(job_text)

        results = []
        with st.spinner("🔍 Analyzing all resumes..."):
            for resume_file in resume_files:
                resume_text = extract_text(resume_file)
                resume_data = parse_text_with_gemini(resume_text, "resume")
                resume_embedding = get_embedding(resume_text)
                score = calculate_similarity(resume_embedding, job_embedding)
                name = resume_data.get("name", resume_file.name)
                results.append({
                    "Candidate": name,
                    "Score": round(score, 2),
                    "Top Skills": ", ".join(resume_data.get("skills", [])[:5])
                })

        st.subheader("📊 Resume Ranking")
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.dataframe(df.reset_index(drop=True))
