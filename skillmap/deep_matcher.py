# skillmap/deep_matcher_setup.py

import json
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

# Paths to your JSONs
JOBS_JSON    = "skillmap/assests/data/jobs_data_normalized.json"
RESUMES_JSON = "skillmap/assests/data/resume_data.json"

# 1) Load JSON data
with open(JOBS_JSON, "r", encoding="utf-8") as f:
    jobs = json.load(f)
with open(RESUMES_JSON, "r", encoding="utf-8") as f:
    resumes = json.load(f)

# 2) Build combined texts
job_texts = []
for job in jobs:
    skills = job.get("skills_required", [])
    exp = job.get("experience_required", {})
    resp = job.get("responsibilities", "")
    text = " ".join(skills) + " " \
         + str(exp.get("min_years","")) + " " \
         + str(exp.get("max_years","")) + " " \
         + resp
    job_texts.append(text)

resume_texts = []
for r in resumes:
    skills = r.get("skills", [])
    exp_details = r.get("experience", {}).get("details", [])
    edu_details = r.get("education", {}).get("details", [])
    resp = r.get("responsibilities", "")
    summary = r.get("summary", "")
    text = " ".join(skills) + " " \
         + " ".join(exp_details) + " " \
         + " ".join(edu_details) + " " \
         + resp + " " + summary
    resume_texts.append(text)

# 3) Fit a Keras Tokenizer on all texts
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(job_texts + resume_texts)

# 4) Fit a LabelEncoder on job titles
titles = [job["title"] for job in jobs]
le = LabelEncoder()
le.fit(titles)

# 5) Save them
with open("skillmap/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("skillmap/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ tokenizer.pkl & label_encoder.pkl generated!")
