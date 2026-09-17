# Amazon Reviews — Recommender System & Sentiment Analysis

End-to-end machine learning project on the **Amazon Reviews 2023** dataset (*Movies & TV* category), built as three incremental stages: **collaborative filtering**, **content-based filtering**, and **sentiment analysis** — with a bonus LLM-based classifier.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/runs%20on-Google%20Colab-yellow)

---

## Overview

The project answers three questions on the same real-world dataset (~463k ratings after filtering):

| Stage | Question | Approach |
| --- | --- | --- |
| **1. Base** | Which items should we recommend to a user, given only past ratings? | Collaborative Filtering — K-NN and SVD (Surprise), K-Means user segmentation, top-N lists |
| **2. Intermediate** | Can we recommend well by looking at *what* the items are (text), not just who rated them? | Content-Based — TF-IDF and Transformer embeddings, per-user K-NN regression |
| **3. Advanced** | Can we automatically infer sentiment (positive / neutral / negative) from review text? | Sentiment Analysis — 4 classifiers × 2 embeddings, class-imbalance handling, bonus LLM (Groq) |

Every stage is fully reproducible: fixed seed (`42`), stratified splits, cross-validation, and metric-based model selection.

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

**Takeaway:** on this dataset SVD wins by exploiting co-rating signal; content-based approaches close the gap and remain the only viable option for cold-start products.

### Sentiment classification — 3 classes (positive / neutral / negative)

Class-imbalance is severe (neutral is rare), so the primary metric is **F1-macro**.

| Classifier | Embedding | Accuracy | F1-macro |
| --- | --- | --- | --- |
| **LinearSVC** | Transformer (DistilBERT) | 0.863 | **0.708** |
| LinearSVC | TF-IDF | 0.859 | 0.689 |
| Logistic Regression | TF-IDF | 0.826 | 0.681 |
| Random Forest | TF-IDF | 0.861 | 0.621 |
| Decision Tree | TF-IDF | 0.785 | 0.587 |
| KNN (k=5) | TF-IDF | 0.501 | 0.397 |

**Bonus:** the LLM classifier (Groq + LangChain, zero-shot) is competitive without any training, but is slower and more expensive per prediction than the classical pipeline.

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

The notebook is designed for **Google Colab** with a GPU runtime (T4 is enough).

1. Open [`notebook.ipynb`](notebook.ipynb) in Google Colab.
2. `Runtime → Change runtime type → GPU (T4)`.
3. `Runtime → Run all`. No kernel restart is required; `scikit-surprise` builds against the runtime's NumPy.
4. At cell **3.13** (bonus LLM), paste your `GROQ_API_KEY` when prompted. Pressing enter falls back to a zero-shot HuggingFace pipeline.
5. *Optional* — set `MOUNT_DRIVE = True` in cell **0.2** to persist figures and tables to Google Drive across sessions.

If the runtime crashes, checkpoints on disk (`parquet`, `.npy`, JSON) allow resuming from any Part without recomputing everything.

### Local execution

Requires Python 3.11+ and CUDA-capable GPU (recommended). Main dependencies:

```
numpy pandas scikit-learn scikit-surprise
torch transformers sentence-transformers
matplotlib seaborn tqdm
datasets huggingface_hub
langchain langchain-groq  # optional, for bonus
```

---

## Repository layout

```
amazon-reviews-recsys-sentiment/
├── README.md                    # this file
├── LICENSE                      # MIT
├── notebook.ipynb               # full 3-stage pipeline (96 cells)
├── report/
│   ├── report.pdf               # written report (methodology + results)
│   └── report.docx
└── presentation/
    └── presentation.pptx        # final slide deck
```

---

## Tech stack

- **Recommender:** `scikit-surprise` (K-NN, SVD, KNNBaseline), `scikit-learn` (K-Means)
- **NLP embeddings:** `scikit-learn` TF-IDF, `transformers` (DistilBERT), `sentence-transformers` (all-mpnet-base-v2)
- **Classifiers:** `scikit-learn` (Decision Tree, Random Forest, Naive Bayes, KNN, LinearSVC, Logistic Regression)
- **LLM bonus:** `langchain-groq` with HuggingFace zero-shot fallback
- **Reproducibility:** fixed seed 42, stratified splits, cross-validation, saved checkpoints

---

## Author

**Andrea Saverino** — third-year Computer Science student, working on machine learning, NLP, and recommender systems.

- GitHub: [@codebysave](https://github.com/codebysave)
- Project developed for the *Machine Learning & Intelligent Agents* course.

---

## License

Distributed under the MIT License — see [LICENSE](LICENSE) for details.
