# skillmap/deep_matcher.py

import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_matcher_model import JobDescriptionModel

# Load tokenizer
with open("skillmap/assests/data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("skillmap/assests/data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
hidden_dim = 128
num_classes = len(label_encoder.classes_)

deep_model = JobDescriptionModel(vocab_size, embedding_dim, hidden_dim, num_classes)
deep_model.load_state_dict(torch.load("skillmap/assests/data/job_description_model.pth", map_location="cpu"))
deep_model.eval()

# Deep match score function
def deep_match_score(resume_text, job_text):
    resume_seq = tokenizer.texts_to_sequences([resume_text])
    job_seq = tokenizer.texts_to_sequences([job_text])

    max_len = max(len(resume_seq[0]), len(job_seq[0]))
    resume_pad = pad_sequences(resume_seq, maxlen=max_len, padding="post")
    job_pad = pad_sequences(job_seq, maxlen=max_len, padding="post")

    resume_tensor = torch.tensor(resume_pad, dtype=torch.long)
    job_tensor = torch.tensor(job_pad, dtype=torch.long)

    with torch.no_grad():
        resume_vec = deep_model(resume_tensor).numpy()
        job_vec = deep_model(job_tensor).numpy()

    numerator = (resume_vec * job_vec).sum()
    denominator = ( (resume_vec**2).sum()**0.5 ) * ( (job_vec**2).sum()**0.5 )
    similarity = numerator / denominator
    return similarity
