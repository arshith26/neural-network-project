# skillmap/evaluate_models.py

import os
import sys
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add root path
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from matcher import calculate_similarity
from deep_matcher import deep_match_score

# Load text from .docx or .txt

def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"Unsupported file: {ext}")

# File locations
JD_DIR = "skillmap/jd"
RES_DIR = "skillmap/resumes"
PAIRS_FILE = "skillmap/pairs.csv"
THRESHOLD = 0.4

# Load pairs.csv

df = pd.read_csv(PAIRS_FILE, dtype=str)
print("Original columns:", df.columns.tolist())

# Label column check
label_col = "label"
if label_col not in df.columns:
    raise RuntimeError("Missing 'label' column")

print("Label counts:")
print(df[label_col].value_counts(dropna=False))

if df.columns.tolist().count("label") > 1:
    df = df.loc[:, ~df.columns.duplicated()]

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(str)
df = df[df["label"].isin({"0", "1"})].copy()
df["label"] = df["label"].astype(int)
print(f"Valid pairs: {len(df)}\n")

assert "jd_file" in df.columns and "resume_file" in df.columns

# Track metrics
counts = {"bert_tp": 0, "bert_fp": 0, "bert_fn": 0, "lstm_tp": 0, "lstm_fp": 0, "lstm_fn": 0}

# Evaluate each pair
for idx, row in df.iterrows():
    jd_file = os.path.join(JD_DIR, os.path.basename(row["jd_file"].replace("\\", "/")))
    res_file = os.path.join(RES_DIR, os.path.basename(row["resume_file"].replace("\\", "/")))

    if not os.path.exists(jd_file) or not os.path.exists(res_file):
        print(f"Missing file: {jd_file} or {res_file}, skipping\n")
        continue

    print(f"[{idx+1}/{len(df)}] JD: {jd_file}\nResume: {res_file}\nTrue Label = {row['label']}")

    jd_text = load_text(jd_file)
    res_text = load_text(res_file)

    resume_emb = model.encode(res_text)
    jd_emb = model.encode(jd_text)
    cos_sim = calculate_similarity(resume_emb, jd_emb)
    bert_pred = int(cos_sim >= THRESHOLD)

    lstm_sim = deep_match_score(res_text, jd_text)
    lstm_pred = int(lstm_sim >= THRESHOLD)

    print(f"BERT sim = {cos_sim:.2f}, pred = {bert_pred}")
    print(f"LSTM sim = {lstm_sim:.2f}, pred = {lstm_pred}")
    print("-" * 60)

    lbl = row["label"]
    if bert_pred == 1 and lbl == 1: counts["bert_tp"] += 1
    if bert_pred == 1 and lbl == 0: counts["bert_fp"] += 1
    if bert_pred == 0 and lbl == 1: counts["bert_fn"] += 1

    if lstm_pred == 1 and lbl == 1: counts["lstm_tp"] += 1
    if lstm_pred == 1 and lbl == 0: counts["lstm_fp"] += 1
    if lstm_pred == 0 and lbl == 1: counts["lstm_fn"] += 1

# Metric calculations

def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

bert_p, bert_r, bert_f1 = prf(counts["bert_tp"], counts["bert_fp"], counts["bert_fn"])
lstm_p, lstm_r, lstm_f1 = prf(counts["lstm_tp"], counts["lstm_fp"], counts["lstm_fn"])

# Output results
result_df = pd.DataFrame([
    ["BERT Cosine", bert_p, bert_r, bert_f1],
    ["LSTM DeepMatch", lstm_p, lstm_r, lstm_f1],
], columns=["Model", "Precision", "Recall", "F1"])

print("\n>>>> Quantitative Performance <<<<")
print(result_df.to_markdown(index=False, floatfmt=".2f"))
