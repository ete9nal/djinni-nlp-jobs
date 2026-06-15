# 💼 Djinni NLP Jobs Classifier

A multi-class NLP classification project that predicts job categories from Ukrainian job postings scraped from Djinni.co. Compares classical ML (TF-IDF + LinearSVC) against transformer-based approaches (zero-shot mDeBERTa, fine-tuned DistilBERT).

---

## 📌 Problem

The Ukrainian IT job market has dozens of categories — from Data Science to DevOps to Marketing. Given a raw job description, can a model automatically assign the correct category? This enables smarter job search, automatic tagging, and market analysis.

---

## 📁 Project Structure

```
djinni-nlp-jobs/
│
├── data/
│   └── .gitkeep                 # Empty — see Data section below
│
├── models/
│   ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│   ├── linear_svc.pkl           # Best model (LinearSVC)
│   ├── label_encoder.pkl        # Label encoder
│   └── .gitkeep                 # distilbert/ not included — see Models section
│
├── notebooks/
│   ├── 01_eda.ipynb             # EDA, preprocessing, lemmatization, word clouds
│   └── 02_modeling.ipynb        # TF-IDF, LinearSVC, Zero-shot, DistilBERT
│
├── app.py                       # Streamlit demo app
├── README.md
└── pyproject.toml
```

---

## 📊 Dataset

> **Note:** `data/djinni_processed.csv` is not included due to size (114MB).
> Run `notebooks/01_eda.ipynb` to generate it automatically from HuggingFace.

**Source:** [lang-uk/recruitment-dataset-job-descriptions-ukrainian](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-ukrainian)

| Property | Value |
|----------|-------|
| Total rows | 27,461 → 24,942 after filtering |
| Language | Ukrainian |
| Source | Djinni.co |
| Target variable | `Primary Keyword` — job category |
| Number of classes | 35 |

The dataset was filtered to remove categories with fewer than 100 examples (`C#`, `Rust`, `Scrum Master`, `Salesforce`, `Scala`, `Other`).

---

## 🔍 EDA Highlights

- Strong class imbalance: Marketing (3,562) vs Flutter (106)
- Word clouds reveal clear semantic differences between categories (Data Science: `дані`, `модель`, `python` vs DevOps: `інфраструктура`, `docker`, `kubernetes`)
- Ukrainian spaCy model (`uk_core_news_sm`) used for lemmatization with custom stopwords

---

## ⚙️ Preprocessing Pipeline

```
Raw text → clean (remove \r\n, punctuation, lowercase) → lemmatize (spaCy uk) → remove stopwords
```

Two text columns produced:
- `Clean_description` → used for TF-IDF
- `Lemmatized` → used for LDA topic modeling and word clouds

---

## 🤖 Models & Results

| Model | Accuracy | Macro F1 | Train Time |
|-------|----------|----------|------------|
| TF-IDF + Logistic Regression | 0.77 | 0.69 | seconds |
| **TF-IDF + LinearSVC ✅** | **0.81** | **0.76** | seconds |
| Zero-shot mDeBERTa | 0.38 | 0.42 | ~15 min |
| Fine-tuned DistilBERT | 0.80 | 0.73 | ~24 min |

**TF-IDF parameters:** `max_features=50000`, `ngram_range=(1,2)`, `max_df=0.8`, `min_df=2`

**DistilBERT:** `distilbert-base-multilingual-cased`, 3 epochs, `lr=2e-5`, `batch_size=16`, RTX 4090

> **Note:** Fine-tuned DistilBERT checkpoints are not included due to size (~500MB).
> To reproduce: run `notebooks/02_modeling.ipynb` section 5. GPU recommended (trained on RTX 4090 in ~24 min).

---

## 💡 Key Insights

**LinearSVC beats fine-tuned DistilBERT** on this dataset — a strong reminder that classical ML is not dead. With 25k labeled examples and domain-specific Ukrainian text, TF-IDF captures the right signals efficiently.

**Zero-shot fails hard** (0.38 accuracy) — without any training examples, even a powerful multilingual model struggles with specific IT job categories like "QA Automation" or "Lead Generation".

**Hardest categories:** `Lead`, `Technical Writing`, `Data Engineer` — too few examples and overlapping job descriptions with other categories.

**Production recommendation:** TF-IDF + LinearSVC — fast, interpretable, accurate, no GPU required.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ete9nal/djinni-nlp-jobs.git
cd djinni-nlp-jobs

# Install dependencies
poetry install

# Download spaCy Ukrainian model
poetry run python -m spacy download uk_core_news_sm

# Generate processed dataset (downloads from HuggingFace automatically)
poetry run jupyter notebook notebooks/01_eda.ipynb

# Train models
poetry run jupyter notebook notebooks/02_modeling.ipynb

# Run Streamlit app
poetry run streamlit run app.py
```

---

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arhny4azjzne7eramhwvh6.streamlit.app/)

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas / NumPy** — data manipulation
- **spaCy** (`uk_core_news_sm`) — Ukrainian lemmatization
- **scikit-learn** — TF-IDF, Logistic Regression, LinearSVC, metrics
- **HuggingFace Transformers** — DistilBERT fine-tuning, zero-shot pipeline
- **PyTorch** — model training (CUDA / RTX 4090)
- **Matplotlib / Seaborn / WordCloud** — visualization
- **Streamlit** — demo web app
- **Joblib** — model serialization
- **Poetry** — dependency management
