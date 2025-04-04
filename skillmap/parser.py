# skillmap/parser.py

import google.generativeai as genai

# 👉 Replace this with environment variable or .env later
genai.configure(api_key="AIzaSyBBMyvutCXPqa2O-SeHdaiGw___6KyMgyE")

def parse_text_with_gemini(text: str, doc_type: str = "resume") -> dict:
    """
    Sends text to Gemini and returns structured data as a dictionary.
    """
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""
    Extract structured fields from the following {doc_type}:
    - For resume: name, skills, experience, education
    - For job: title, required_skills, responsibilities
    Text: {text}
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        if content.startswith("```json"):
            content = content[7:-3].strip()

        return eval(content)  # replace with json.loads if needed
    except Exception as e:
        print("[Gemini Parse Error]", e)
        return {}

