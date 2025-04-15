import google.generativeai as genai
import re
import json

# Configure API Key (use .env later for security)
genai.configure(api_key="AIzaSyBBMyvutCXPqa2O-SeHdaiGw___6KyMgyE")

def parse_text_with_gemini(text: str, doc_type: str = "resume") -> dict:
    """
    Sends text to Gemini and extracts structured fields from a resume or job description.
    """

    model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""
    You are an expert parser. Extract structured data in JSON format from this {doc_type} text.

    If the document is a resume, extract:
    - name (string)
    - skills (list of technical and soft skills)
    - experience (list of projects or roles, each with title and description)
    - education (list with degree, university, location, and graduation date)

    If the document is a job description, extract:
    - title (string)
    - required_skills (list of skills or technologies)
    - responsibilities (list of tasks or responsibilities)

    Return only JSON — no explanation, no markdown, no bullet points.
    Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Extract JSON from response text
        match = re.search(r'{.*}', content, re.DOTALL)
        if match:
            return json.loads(match.group())

        print("⚠️ No valid JSON found in Gemini response.")
        return {}

    except Exception as e:
        print("[Gemini Parse Error]", e)
        return {}

