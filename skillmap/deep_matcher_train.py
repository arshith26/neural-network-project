# skillmap/deep_matcher_train.py

import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from deep_matcher_model import JobDescriptionModel

# paths to files
DATA_DIR        = "skillmap/assests/data"
JOBS_JSON       = f"{DATA_DIR}/jobs_data_normalized.json"
RESUMES_JSON    = f"{DATA_DIR}/resume_data.json"
TOKENIZER_PKL   = f"{DATA_DIR}/tokenizer.pkl"
LABELENC_PKL    = f"{DATA_DIR}/label_encoder.pkl"
OUTPUT_MODEL    = f"{DATA_DIR}/job_description_model.pth"

# load tokenizer and label encoder
with open(TOKENIZER_PKL, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABELENC_PKL, "rb") as f:
    label_encoder = pickle.load(f)

# load job and resume data
with open(JOBS_JSON, "r", encoding="utf-8") as f:
    jobs = json.load(f)
with open(RESUMES_JSON, "r", encoding="utf-8") as f:
    resumes = json.load(f)

# prepare job texts and labels
job_texts = []
job_labels = []
for job in jobs:
    parts = []
    exp = job.get("experience_required", {})
    parts += job.get("responsibilities", "").split()
    parts += job.get("description", "").split()
    parts += [str(exp.get("min_years", "")), str(exp.get("max_years", ""))]
    parts += job.get("responsibilities", "").split()
    text = " ".join(parts).strip()
    if text:
        job_texts.append(text)
        job_labels.append(job["title"])

# tokenize and pad
sequences = tokenizer.texts_to_sequences(job_texts)
filtered = [(seq, lbl) for seq, lbl in zip(sequences, job_labels) if len(seq) > 0]
sequences, job_labels = zip(*filtered)

max_len = max(len(s) for s in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding="post")
y = label_encoder.transform(job_labels)

# dataset class
class JobDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(JobDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader  = DataLoader(JobDataset(X_test, y_test), batch_size=16, shuffle=False)

# set up model and training
vocab_size    = len(tokenizer.word_index) + 1
embedding_dim = 64
hidden_dim    = 128
num_classes   = len(label_encoder.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JobDescriptionModel(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# training loop
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

# save model weights
torch.save(model.state_dict(), OUTPUT_MODEL)
print(f"✅ Saved LSTM model to {OUTPUT_MODEL}")
