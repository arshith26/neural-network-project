# skillmap/matcher.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(resume_embedding, job_embedding):
    """
    Computes cosine similarity between resume and job description embeddings.
    Returns a float between 0 and 1.
    """
    score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    return round(score, 4)

def find_skill_gap(resume_skills, job_skills):
    """
    Compares resume skills and job required skills.
    Returns a tuple: (matched_skills, missing_skills)
    """
    resume_set = set([skill.lower() for skill in resume_skills])
    job_set = set([skill.lower() for skill in job_skills])

    matched = list(job_set & resume_set)
    missing = list(job_set - resume_set)

    return matched, missing

# Test the functions
if __name__ == "__main__":
    emb1 = np.random.rand(384)
    emb2 = np.random.rand(384)
    score = calculate_similarity(emb1, emb2)
    print("🧠 Cosine Similarity Score:", score)

    r_skills = ["Python", "Machine Learning", "NLP"]
    j_skills = ["Python", "Deep Learning", "NLP", "Cloud"]
    matched, missing = find_skill_gap(r_skills, j_skills)
    print("✅ Matched Skills:", matched)
    print("❌ Missing Skills:", missing)
