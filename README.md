# Hybrid BM25 + TF-IDF Information Retrieval System

This repository contains a complete local information retrieval (IR) system built for an academic assignment.  
The system is implemented from scratch and combines **BM25**, **TF-IDF**, and lightweight boosting strategies to produce accurate document rankings.  
Everything runs fully offline and is designed to be minimal, and reproducible. The goal is to demonstrate a complete, transparent IR pipeline — from raw documents to ranked results.

---
## 1. Project Overview
This IR system includes:
- A full preprocessing pipeline  
- Dual indexing: BM25 and TF-IDF  
- A custom hybrid ranking model  
- A flexible query parser (Boolean, phrase, category, date filters)  
- Clean and modular Python scripts  
- Evaluation using standard IR metrics
---
## 2. Installation
### 2.1 Create a virtual environment
```bash
py 3.10 -m venv .venv
```
### 2.2 Activate it

Windows:
```
.venv\scripts\activate
```
Mac/Linux:
```
source .venv/bin/activate
```
### 2.3 Install dependencies
```
pip install -r requirements.txt
```
```
python -m spacy download en_core_web_sm
```

### 2.4 Run Scripts

#### 2.4.1. Preprocessing

Run manually:
```
python scripts/preprocessing.py
```

This script performs:

- Lowercasing

- HTML removal

- URL removal

- Unicode cleanup

- spaCy tokenization & lemmatization

- Stopword removal

- Duplicate removal

Output:
```
datasets/Articles_Cleaned.csv
```

#### 2.4.2. Index Building

Run:
```
python scripts/indexes_building.py
```

Creates:
```
indexes/bm25_index.pkl
```
```
indexes/tfidf_vectorizer.pkl
```
```
indexes/tfidf_matrix.npz
```
#### 2.4.3. Search Engine

Run:
```
python -m scripts.search_engine
```

Supported Query Features:

- Boolean logic
```
pakistan AND cricket
oil NOT gas
```
- Phrase queries
```
west texas oil"
```
- Category filtering
```
category:sports
category:business
```
- Date filtering
```
date:2016
```
- Boosting
    * Heading keyword overlap
    * Phrase match bonus

#### 2.4.4. Hybrid Ranking Model

Documents are scored using:
```
final_score = (0.45 * BM25_norm) + (0.35 * TFIDF_norm) + (0.15 * heading_overlap) + (0.05 * phrase_match)
```

This balances lexical relevance, semantic matching, and simple structural cues.

#### 2.4.5. Evaluation

Run:
```
python -m scripts.evaluating
```

The system computes:
- Recall@10

- Precision@10
- nDCG@10
- MRR

##### Evaluation Results (50 Queries)

| Metric        | TF-IDF | BM25  | Hybrid |
|---------------|--------|-------|--------|
| Precision@10  | 0.920  | 0.920 | **0.940** |
| Recall@10     | 0.920  | 0.920 | **0.940** |
| MRR           | 0.769  | 0.835 | **0.892** |
| nDCG@10       | 0.805  | 0.856 | **0.904** |

The hybrid model performs the best across all evaluation metrics.


## 3. License

This project is for academic use.
You may reuse the code for learning purposes.

## Thank You

If you need help understanding or reproducing any part of the pipeline, feel free to open an issue or contact. mubashar.itu@gmai.com
