# Assignment 1: Sparse Retrieval Arena Submission

## Overview
This directory contains the complete implementation of the Inverted Index and retrieval models for Assignment 1:
- `indexer.py`: High-performance inverted indexer with alphanumeric regex tokenization, memoized Porter stemming (`_STEM_CACHE`), and Gzip-compressed binary serialization (`index.pkl.gz`).
- `boolean_vsm.py`: Boolean Retrieval (AND/OR) and Vector Space Model (TF-IDF Cosine Similarity) with lazy vector norm evaluation.
- `bm25.py`: Robertson-Sparck Jones BM25 ranking function with tunable parameters $k_1$ and $b$.
- `custom_scorer.py`: Tuned BM25 scorer with optimal parameters ($k_1=1.5, b=0.50$) and low-IDF stopword pruning.
- `retrieve.py`: Entrypoint exposing `build_index()`, `load_index()`, and `retrieve()`.

---

## Reproduction Instructions

### 1. Requirements & Dependencies
Ensure Python 3.9+ is installed along with standard dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `nltk>=3.8`, `numpy>=1.23`, `pytest>=7.0`)*

### 2. Running Unit Tests
Verify that all interface conformance and metric tests pass:
```bash
pytest tests/ -v
```

### 3. Building the Inverted Index
To index the toy corpus (or full dataset):
```bash
python -c "from submission.retrieve import build_index; build_index('data/toy/corpus.jsonl', 'data/toy/index_toy')"
```

### 4. Running Retrieval Queries
To load the index and retrieve top-k documents:
```bash
python -c "from submission.retrieve import load_index, retrieve; load_index('data/toy/index_toy'); print(retrieve('coronavirus transmission', k=5))"
```

### 5. Running the Complete Evaluation Harness
To run the full end-to-end evaluation harness against reference run and compute nDCG@10, MAP@10, MRR, and P@10:
```bash
python -m harness.run_harness   --corpus data/toy/corpus.jsonl   --queries data/toy/queries_dev.tsv   --qrels data/toy/qrels_dev.txt   --baseline-run data/toy/reference_bm25_run_dev.trec   --run-out runs/dev_run.trec   --report-out runs/dev_report.json
```

### 6. Running Smoke Test
```bash
bash scripts/smoke_test.sh
```
