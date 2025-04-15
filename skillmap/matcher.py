# skillmap/matcher.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once here
model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_similarity(resume_embedding, job_embedding):
    score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    return round(score, 4)

def find_skill_gap(resume_skills, job_skills, threshold=0.7):
    if not resume_skills or not job_skills:
        return [], job_skills  # return all as missing if any side is empty

    matched = []
    missing = []

    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    job_embeddings = model.encode(job_skills, convert_to_tensor=True)

    for i, job_emb in enumerate(job_embeddings):
        sims = cosine_similarity([job_emb.cpu().numpy()], resume_embeddings.cpu().numpy())[0]
        max_sim = max(sims)
        if max_sim >= threshold:
            matched.append(job_skills[i])
        else:
            missing.append(job_skills[i])

    return matched, missing



