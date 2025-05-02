# SkillMap: Deep Resume-Job Matching Using Semantic Similarity and LSTM Models

SkillMap is an intelligent job-resume matching system that combines transformer-based embeddings, LSTM similarity models, and generative AI to evaluate and enhance candidate fit. Designed with recruiters and job seekers in mind, SkillMap identifies matched and missing skills, scores resume-job alignment, and provides resume enhancement suggestions using Gemini.

## 🚀 Features

- **Dual Model Scoring**: Combines BERT-based cosine similarity and an LSTM-based deep matcher to evaluate resume-job fit.
- **Skill Gap Analysis**: Detects skills present in the resume vs. required in the job description.
- **Resume Enhancement**: Suggests improved summaries, experiences, and skills using Gemini API.
- **Dual Modes**:
  - *Job Seeker*: Match and improve your resume for a specific job.
  - *Recruiter*: Rank multiple resumes against a job description.
- **Streamlit UI**: Easy-to-use interface for uploading, analyzing, and downloading enhanced resume sections.

## 🧠 Models Used

- Sentence-BERT (`all-MiniLM-L6-v2`) for semantic embeddings
- Custom LSTM model trained on normalized resume-job data
- Google Gemini (Generative AI) for resume enhancement and structured parsing

## 📁 Project Structure
skillmap/
│
├── assets/data/ # Raw and normalized data, tokenizer, model weights
├── deep_matcher_model.py # LSTM model class
├── deep_matcher_train.py # Model training script
├── deep_matcher.py # Deep match inference logic
├── embedder.py # Sentence-BERT embedding utility
├── enhancer.py # Summary enhancer (Gemini)
├── enhancer_v2.py # Full resume enhancer using Gemini
├── matcher.py # Cosine similarity + skill gap logic
├── parser.py # Resume/Job JSON parser via Gemini
├── evaluate_models.py # Evaluation against labeled JD-resume pairs
├── normalize_datasets.py # Preprocessing for jobs/resumes
├── app.py # Streamlit interface


## 🧪 Evaluation Results

| Model           | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| BERT Cosine     | 1.00      | 0.75   | 0.86     |
| LSTM DeepMatch  | 0.67      | 1.00   | 0.80     |

## 🔧 Installation

```bash
pip install -r requirements.txt

Usage:
To run the web app:
streamlit run skillmap/app.py
To run evaluation:
python skillmap/evaluate_models.py

📄 Requirements
Python ≥ 3.8

sentence-transformers, scikit-learn, pandas, torch, tensorflow, google-generativeai, streamlit, PyMuPDF

📚 References
See references.bib for all citations used in the accompanying paper.



