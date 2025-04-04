# skillmap/embedder.py
from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence-BERT model (MiniLM)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    """
    Returns a 384-d embedding vector for the input text.
    """
    return model.encode(text, convert_to_numpy=True)
