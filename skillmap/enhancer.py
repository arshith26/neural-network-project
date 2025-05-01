# skillmap/enhancer.py

import google.generativeai as genai

# generate improved resume summary based on job description
def generate_resume_summary(resume_text, job_text):
    prompt = f"""
    Improve this candidate's resume summary to better match the following job description.

    Resume:
    {resume_text}

    Job Description:
    {job_text}

    Output a professional 3–4 sentence summary tailored to the job.
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Enhancer Error] {e}"
