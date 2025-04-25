# skillmap/assests/data/normalize_datsets.py

import json
import os

# 1) Compute this script’s folder, once:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         └→ e.g.  C:/…/neural-network-project/skillmap/assests/data

# 2) Build absolute paths to the input JSONs:
jobs_path    = os.path.join(BASE_DIR, "jobs_data.json")
resumes_path = os.path.join(BASE_DIR, "resume_data.json")

# 3) Build output filenames:
out_jobs    = os.path.join(BASE_DIR, "jobs_data_normalized.json")
out_resumes = os.path.join(BASE_DIR, "resume_data_normalized.json")

# 4) Load raw data:
with open(jobs_path, "r", encoding="utf-8") as f:
    jobs = json.load(f)

with open(resumes_path, "r", encoding="utf-8") as f:
    resumes = json.load(f)

# 5) Perform your normalization steps here.
#    (For brevity, I’ll just deep-copy them; replace with real logic.)
normalized_jobs    = jobs   # ← apply whatever transforms you need
normalized_resumes = resumes

# 6) Write out the normalized files:
with open(out_jobs, "w", encoding="utf-8") as f:
    json.dump(normalized_jobs, f, indent=2, ensure_ascii=False)

with open(out_resumes, "w", encoding="utf-8") as f:
    json.dump(normalized_resumes, f, indent=2, ensure_ascii=False)

print("✅ Normalized data written to:")
print("  •", out_jobs)
print("  •", out_resumes)
