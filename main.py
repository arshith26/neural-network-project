# main.py
from skillmap.parser import parse_text_with_gemini
from skillmap.embedder import get_embedding
from skillmap.matcher import calculate_similarity, find_skill_gap

# Sample resume and job description text
resume_text = """
John Doe
Experienced Data Scientist with 3+ years in Python, Machine Learning, and TensorFlow.
Worked at Google on NLP models and deployed deep learning systems.
"""

job_text = """
Hiring Data Scientist with experience in Python, Deep Learning, AWS, and NLP.
Looking for someone familiar with cloud deployment and transformer-based models.
"""

# Step 1: Parse resume and job description
resume_data = parse_text_with_gemini(resume_text, doc_type="resume")
job_data = parse_text_with_gemini(job_text, doc_type="job")

# Step 2: Get embeddings for full text
resume_embedding = get_embedding(resume_text)
job_embedding = get_embedding(job_text)

# Step 3: Compute similarity
match_score = calculate_similarity(resume_embedding, job_embedding)

# Step 4: Skill gap analysis
resume_skills = resume_data.get("skills", [])
job_skills = job_data.get("required_skills", [])
matched, missing = find_skill_gap(resume_skills, job_skills)

# Output the result
print("\n================ SkillMap Match Result ================")
print(f"✅ Match Score: {match_score:.2f}")
print(f"🧠 Matched Skills: {matched}")
print(f"❌ Missing Skills: {missing}")
print("=======================================================\n")

