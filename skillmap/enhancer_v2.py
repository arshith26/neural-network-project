# skillmap/enhancer_v2.py

import google.generativeai as genai

# setup Gemini model
genai.configure(api_key="AIzaSyBBMyvutCXPqa2O-SeHdaiGw___6KyMgyE")
model = genai.GenerativeModel("models/gemini-1.5-flash")

# improve a resume section using job context
def enhance_section(resume_section: str, job_context: str, section_type: str = "summary") -> str:
    prompt = f"""
    Here is a candidate's {section_type} section:
    {resume_section}

    Here is the job description context it should match more closely:
    {job_context}

    Improve the candidate's {section_type} section to align better with the job. Keep it concise and focused.
    Return only the improved {section_type}.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Enhance Error] {e}")
        return resume_section

# remap keys to standard field names
def normalize_keys(data: dict, target_keys: dict) -> dict:
    normalized = {}
    for standard_key, possible_keys in target_keys.items():
        for key in possible_keys:
            if key in data:
                normalized[standard_key] = data[key]
                break
    return normalized

# enhance summary, experience, and skills using Gemini
def enhance_resume(resume_data: dict, job_data: dict) -> dict:
    resume_fields = {
        "summary": ["summary", "Summary", "Professional Summary"],
        "experience": ["experience", "Experience", "Work Experience"],
        "skills": ["skills", "Skills", "Technical Skills"]
    }
    job_fields = {
        "responsibilities": ["responsibilities", "Responsibilities", "Job Responsibilities"],
        "required_skills": ["required_skills", "Skills", "Required Skills"]
    }

    resume_data = normalize_keys(resume_data, resume_fields)
    job_data = normalize_keys(job_data, job_fields)

    enhanced = {}

    if "summary" in resume_data and "responsibilities" in job_data:
        enhanced["summary"] = enhance_section(
            resume_data["summary"],
            job_data["responsibilities"],
            section_type="summary"
        )

    if "experience" in resume_data and "responsibilities" in job_data:
        if isinstance(resume_data["experience"], list):
            joined_exp = "\n".join(
                f"{item.get('title', '')}: {item.get('description', '')}"
                for item in resume_data["experience"]
                if isinstance(item, dict)
            )
        else:
            joined_exp = resume_data["experience"]
        enhanced["experience"] = enhance_section(
            joined_exp,
            job_data["responsibilities"],
            section_type="experience"
        )

    if "skills" in resume_data and "required_skills" in job_data:
        resume_skills = ", ".join(resume_data["skills"]) if isinstance(resume_data["skills"], list) else resume_data["skills"]
        job_skills = ", ".join(job_data["required_skills"]) if isinstance(job_data["required_skills"], list) else job_data["required_skills"]
        enhanced["skills"] = enhance_section(
            resume_skills,
            job_skills,
            section_type="skills"
        )

    return enhanced
