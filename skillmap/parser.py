# skillmap/parser.py
import google.generativeai as genai
import re
import json

# Configure API Key (use .env later for security)
genai.configure(api_key="AIzaSyBBMyvutCXPqa2O-SeHdaiGw___6KyMgyE")

def parse_text_with_gemini(text: str, doc_type: str = "resume") -> dict:
    """
    Sends text to Gemini and attempts to extract structured data.
    """
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""
    Extract structured fields from the following {doc_type}:
    - For resume: name, skills, experience, education
    - For job: title, required_skills, responsibilities
    Return valid JSON format without markdown or bullet points.
    Text: {text}
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
