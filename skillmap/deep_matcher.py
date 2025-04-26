# skillmap/deep_matcher_setup.py

import json
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

# Paths
JOBS_JSON    = "skillmap/assests/data/jobs_data_normalized.json"
RESUMES_JSON = "skillmap/assests/data/resume_data.json"

# 1) Load JSON
with open(JOBS_JSON, "r", encoding="utf-8") as f:
    jobs = json.load(f)
with open(RESUMES_JSON, "r", encoding="utf-8") as f:
    resumes = json.load(f)

# 2) Build job_texts
job_texts = []
for job in jobs:
    skills = job.get("skills_required", [])        # list of strings
    exp = job.get("experience_required", {})        # dict with min/max
    resp = job.get("responsibilities", "")          # string or list
    if isinstance(resp, list):
        resp = " ".join(resp)
    text = " ".join(skills) + " " \
         + str(exp.get("min_years", "")) + " " \
         + str(exp.get("max_years", "")) + " " \
         + resp
    job_texts.append(text)

# 3) Build resume_texts
resume_texts = []
for r in resumes:
    skills = r.get("skills", [])                   # list
    # experience: list of dicts with "details" or "description"
    exp_list = []
    for exp in r.get("work_experience", []):
        # some keys: "details" or "description"
        if "details" in exp:
            exp_list += exp["details"]
        elif "description" in exp:
            exp_list.append(exp["description"])

    # education: list of dicts — concatenate degree+institution+year
    edu_list = []
    for ed in r.get("education", []):
        inst = ed.get("institution", "")
        deg  = ed.get("degree", "")
        year = ed.get("year", "")
        edu_list.append(f"{inst} {deg} {year}".strip())

    # responsibilities & summary fields
    resp = r.get("responsibilities", "")
    summary = r.get("summary", "")

    text = " ".join(skills) + " " \
         + " ".join(exp_list) + " " \
         + " ".join(edu_list) + " " \
         + (resp if isinstance(resp, str) else " ".join(resp)) + " " \
         + summary
    resume_texts.append(text)

# 4) Fit Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(job_texts + resume_texts)
with open("skillmap/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 5) Fit LabelEncoder on titles
titles = [job["title"] for job in jobs]
le = LabelEncoder()
le.fit(titles)
with open("skillmap/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ tokenizer.pkl & label_encoder.pkl generated successfully!")
