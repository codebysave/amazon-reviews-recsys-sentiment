# Amazon Reviews — Recommender System & Sentiment Analysis

This is an end-to-end machine learning project I built on the **Amazon Reviews 2023** dataset (*Movies & TV* category), designed as three incremental stages: **collaborative filtering**, **content-based filtering**, and **sentiment analysis** — plus a bonus LLM-based classifier.

I developed it during the third year of my Computer Science degree, for the *Machine Learning & Intelligent Agents* course. My goal was to go beyond the standard coursework: take a real, messy dataset and squeeze out honest, reproducible results at each stage, comparing techniques instead of just running one model.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![Platform](https://img.shields.io/badge/runs%20on-Google%20Colab-yellow)

---

## Overview

I answer three questions on the same real-world dataset (~463k ratings after filtering):

| Stage | Question | Approach |
| --- | --- | --- |
| **1. Base** | Which items should we recommend to a user, given only past ratings? | Collaborative Filtering — K-NN and SVD (Surprise), K-Means user segmentation, top-N lists |
| **2. Intermediate** | Can we recommend well by looking at *what* the items are (text), not just who rated them? | Content-Based — TF-IDF and Transformer embeddings, per-user K-NN regression |
| **3. Advanced** | Can we automatically infer sentiment (positive / neutral / negative) from review text? | Sentiment Analysis — 4 classifiers × 2 embeddings, class-imbalance handling, bonus LLM (Groq) |

I made every stage fully reproducible: fixed seed (`42`), stratified splits, cross-validation, and metric-based model selection.

---

## Dataset

- **Source:** [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), streamed via `hf://` — no manual download.
- **Category:** `Movies_and_TV`.
- **After k-core filtering (users ≥ 5, items ≥ 5):** 33,698 users · 30,923 products · 463,509 ratings.
- **Text features:** review title + body + product metadata (title, description, features, categories).

---

## Key Results

### Recommender systems — RMSE on held-out ratings (lower is better)

| Model | Family | RMSE |
| --- | --- | --- |
| **SVD** (Matrix Factorization) | Collaborative Filtering | **0.9498** |
| K-NN (item-based, best k) | Collaborative Filtering | 1.0710 |
| K-NN + Transformer embedding | Content-Based | 0.9835 |
| K-NN + TF-IDF embedding | Content-Based | 0.9836 |
| K-NN + Sentence-Transformer (all-mpnet) | Content-Based | 0.9837 |

**My reading:** on this dataset SVD wins by exploiting the co-rating signal; the content-based approaches close the gap and stay the only viable option for cold-start products, so I keep both in the pipeline.

### Sentiment classification — 3 classes (positive / neutral / negative)

Class imbalance is severe (neutral is the rare class), so I picked **F1-macro** as the primary metric instead of accuracy.

| Classifier | Embedding | Accuracy | F1-macro |
| --- | --- | --- | --- |
| **LinearSVC** | Transformer (DistilBERT) | 0.863 | **0.708** |
| LinearSVC | TF-IDF | 0.859 | 0.689 |
| Logistic Regression | TF-IDF | 0.826 | 0.681 |
| Random Forest | TF-IDF | 0.861 | 0.621 |
| Decision Tree | TF-IDF | 0.785 | 0.587 |
| KNN (k=5) | TF-IDF | 0.501 | 0.397 |

**Bonus:** I also tried an LLM classifier (Groq + LangChain, zero-shot). It is competitive without any training, but slower and more expensive per prediction than the classical pipeline — a useful comparison, not a replacement.

---

## Notebook structure

```
Part 1 — Collaborative Filtering (BASE)
  ├─ EDA, k-core filtering, Surprise conversion
  ├─ K-NN grid search + elbow method
  ├─ SVD grid search
  ├─ Model comparison, matrix filling, top-N lists
  └─ K-Means user segmentation

Part 2 — Content-Based Filtering (INTERMEDIATE)
  ├─ NLP preprocessing (lemmatization, stopword removal)
  ├─ TF-IDF and Transformer (DistilBERT) embeddings
  ├─ Per-user K-NN regression
  └─ Fair comparison vs. CF on the same split

Part 3 — Sentiment Analysis (ADVANCED)
  ├─ Sentiment label construction from ratings
  ├─ Stratified sampling + train/test split
  ├─ TF-IDF and Transformer embeddings
  ├─ 4 classifiers × class-weight handling
  ├─ Confusion matrices, per-class metrics, F1-macro focus
  └─ Bonus: LLM zero-shot via Groq (with HF fallback)

Chapter 8 — Extra experiments (out of scope)
  ├─ KNNBaseline (bias correction)
  ├─ SVD on the full k-core
  ├─ Sentence-Transformer (all-mpnet) embedding
  └─ Linear classifiers (LinearSVC, LogReg) — best F1-macro of the project
```

---

## How to run

I designed the notebook for **Google Colab** with a GPU runtime (T4 is enough).

1. Open [`notebook.ipynb`](notebook.ipynb) in Google Colab.
2. `Runtime → Change runtime type → GPU (T4)`.
3. `Runtime → Run all`. No kernel restart is required; `scikit-surprise` builds against the runtime's NumPy.
4. At cell **3.13** (bonus LLM), paste your `GROQ_API_KEY` when prompted. Pressing enter falls back to a zero-shot HuggingFace pipeline.
5. *Optional* — set `MOUNT_DRIVE = True` in cell **0.2** to persist figures and tables to Google Drive across sessions.

If the runtime crashes, checkpoints on disk (`parquet`, `.npy`, JSON) allow resuming from any Part without recomputing everything.

### Local execution

Requires Python 3.11+ and a CUDA-capable GPU (recommended for the transformer embeddings). Install the dependencies with:

```bash
pip install -r requirements.txt
```

The full dependency list with the exact versions I used is in [`requirements.txt`](requirements.txt).

---

## Repository layout

```
amazon-reviews-recsys-sentiment/
├── README.md          # this file
├── requirements.txt   # Python dependencies with versions
├── notebook.ipynb     # full 3-stage pipeline (96 cells)
└── report.pdf         # written report (methodology + results)
```

---

## Tech stack

- **Recommender:** `scikit-surprise` (K-NN, SVD, KNNBaseline), `scikit-learn` (K-Means)
- **NLP embeddings:** `scikit-learn` TF-IDF, `transformers` (DistilBERT), `sentence-transformers` (all-mpnet-base-v2)
- **Classifiers:** `scikit-learn` (Decision Tree, Random Forest, Naive Bayes, KNN, LinearSVC, Logistic Regression)
- **LLM bonus:** `langchain-groq` with HuggingFace zero-shot fallback
- **Reproducibility:** fixed seed 42, stratified splits, cross-validation, saved checkpoints

