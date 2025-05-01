# skillmap/evaluate_models.py

import os
import sys
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # only once at top

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from matcher import calculate_similarity
from deep_matcher import deep_match_score

# ────────────────────────────────────────────────────────────────────────────────
# 1) Helper to load .docx or .txt
# ────────────────────────────────────────────────────────────────────────────────
def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        doc = Document(path)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paras)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# ────────────────────────────────────────────────────────────────────────────────
# 2) Configuration
# ────────────────────────────────────────────────────────────────────────────────
JD_DIR     = "skillmap/jd"
RES_DIR    = "skillmap/resumes"
PAIRS_FILE = "skillmap/pairs.csv"
THRESHOLD  = 0.4

# ────────────────────────────────────────────────────────────────────────────────
# 3) Read pairs.csv and force correct label column
# ────────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(PAIRS_FILE, dtype=str)
print("Original columns:", df.columns.tolist())

# Force the correct label column
label_col = "label"
if label_col not in df.columns:
    raise RuntimeError("❌ 'label' column not found in pairs.csv")

print("Value counts in 'label' column BEFORE cleaning:")
print(df[label_col].value_counts(dropna=False))

# Drop extra label column if renamed before
if df.columns.tolist().count("label") > 1:
    df = df.loc[:, ~df.columns.duplicated()]

# Standardize label column
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(str)
df = df[df["label"].isin({"0", "1"})].copy()
df["label"] = df["label"].astype(int)
print(f">>> After cleaning: {len(df)} valid pairs\n")

assert "jd_file" in df.columns and "resume_file" in df.columns, \
    "❌ pairs.csv must have jd_file and resume_file columns"

# ────────────────────────────────────────────────────────────────────────────────
# 4) Counters for BERT and LSTM
# ────────────────────────────────────────────────────────────────────────────────
counts = {
    "bert_tp": 0, "bert_fp": 0, "bert_fn": 0,
    "lstm_tp": 0, "lstm_fp": 0, "lstm_fn": 0
}

# ────────────────────────────────────────────────────────────────────────────────
# 5) Iterate through each pair
# ────────────────────────────────────────────────────────────────────────────────
for idx, row in df.iterrows():
    jd_fname  = os.path.basename(row["jd_file"].replace("\\", "/").strip())
    res_fname = os.path.basename(row["resume_file"].replace("\\", "/").strip())

    jd_path  = os.path.join(JD_DIR, jd_fname)
    res_path = os.path.join(RES_DIR, res_fname)

    if not os.path.exists(jd_path) or not os.path.exists(res_path):
        print(f"⚠️ File not found: {jd_path} or {res_path}, skipping...\n")
        continue

    print(f"> [{idx+1}/{len(df)}] JD → {jd_path}")
    print(f"           Resume → {res_path}")
    print(f"           True Label = {row['label']}")

    jd_text  = load_text(jd_path)
    res_text = load_text(res_path)
    print(f"             JD chars = {len(jd_text)} | Resume chars = {len(res_text)}")

    resume_embedding = model.encode(res_text)
    jd_embedding = model.encode(jd_text)
    cos_sim = calculate_similarity(resume_embedding, jd_embedding)
    bert_pred = int(cos_sim >= THRESHOLD)

    lstm_sim  = deep_match_score(res_text, jd_text)
    lstm_pred = int(lstm_sim >= THRESHOLD)

    print(f"           BERT sim = {cos_sim:.2f}, pred = {bert_pred}")
    print(f"          LSTM sim = {lstm_sim:.2f}, pred = {lstm_pred}")
    print("-" * 60)

    true_lbl = row["label"]
    # BERT tallies
    if bert_pred == 1 and true_lbl == 1: counts["bert_tp"] += 1
    if bert_pred == 1 and true_lbl == 0: counts["bert_fp"] += 1
    if bert_pred == 0 and true_lbl == 1: counts["bert_fn"] += 1
    # LSTM tallies
    if lstm_pred == 1 and true_lbl == 1: counts["lstm_tp"] += 1
    if lstm_pred == 1 and true_lbl == 0: counts["lstm_fp"] += 1
    if lstm_pred == 0 and true_lbl == 1: counts["lstm_fn"] += 1

# ────────────────────────────────────────────────────────────────────────────────
# 6) Compute Precision / Recall / F1
# ────────────────────────────────────────────────────────────────────────────────
def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

bert_p, bert_r, bert_f1 = prf(counts["bert_tp"], counts["bert_fp"], counts["bert_fn"])
lstm_p, lstm_r, lstm_f1 = prf(counts["lstm_tp"], counts["lstm_fp"], counts["lstm_fn"])

# ────────────────────────────────────────────────────────────────────────────────
# 7) Print Results as Markdown Table
# ────────────────────────────────────────────────────────────────────────────────
table = pd.DataFrame([
    ["BERT Cosine",    bert_p, bert_r, bert_f1],
    ["LSTM DeepMatch", lstm_p, lstm_r, lstm_f1],
], columns=["Model", "Precision", "Recall", "F1"])

print("\n>>>> Quantitative Performance <<<<")
print(table.to_markdown(index=False, floatfmt=".2f"))
