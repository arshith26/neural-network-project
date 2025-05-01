import json
import os

# get the path to the current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# set paths to input files
jobs_path = os.path.join(BASE_DIR, "jobs_data.json")
resumes_path = os.path.join(BASE_DIR, "resume_data.json")

# set paths to output files
out_jobs = os.path.join(BASE_DIR, "jobs_data_normalized.json")
out_resumes = os.path.join(BASE_DIR, "resume_data_normalized.json")

# load input data
with open(jobs_path, "r", encoding="utf-8") as f:
    jobs = json.load(f)

with open(resumes_path, "r", encoding="utf-8") as f:
    resumes = json.load(f)

# normalize the data (placeholder for now)
normalized_jobs = jobs
normalized_resumes = resumes

# save normalized data
with open(out_jobs, "w", encoding="utf-8") as f:
    json.dump(normalized_jobs, f, indent=2, ensure_ascii=False)

with open(out_resumes, "w", encoding="utf-8") as f:
    json.dump(normalized_resumes, f, indent=2, ensure_ascii=False)

print("✅ Normalized data written to:")
print("  •", out_jobs)
print("  •", out_resumes)
