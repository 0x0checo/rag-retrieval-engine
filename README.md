# RAG Retrieval Engine: Sparse/Dense/Hybrid Baselines + Evaluation

A lightweight retrieval benchmark + demo app for RAG, built on **BEIR (SciFact)**.  
This project compares **sparse retrieval (BM25)**, **dense retrieval (BGE / E5)**, and a **hybrid fusion (BM25 + BGE via RRF)** with standard IR metrics and an interactive **Streamlit** UI.

**Here's the website:**

https://rag-retrieval-engine.streamlit.app/
---

## What’s Included

- **Retrievers**
  - **BM25** (sparse / keyword)
  - **Dense (BGE)** vector retrieval
  - **Dense (E5)** vector retrieval (`query:` prefix)
  - **Hybrid (BM25 + BGE)** via **RRF (Reciprocal Rank Fusion)**

- **Evaluation**
  - Metrics: **NDCG@10**, **MAP@10**, **P@10**, **Recall@100**
  - Results exported to CSV for reporting

- **Streamlit Demo**
  - Switch retrievers from a dropdown
  - Pick a SciFact query ID or type your own query
  - Inspect top-k docs (snippet + expandable full text)
  - Show **qrels ground truth** and **hit@k**

---

## Repo Structure

```

rag-retrieval-lab/
├─ app.py
├─ src/
│  ├─ eval_bm25.py
│  ├─ eval_dense_bge.py
│  ├─ eval_dense_e5.py
│  ├─ eval_hybrid_rrf.py
│  ├─ summarize_metrics.py
│  └─ build_dense_index.py
├─ indexes/          # generated (FAISS + docid maps)
├─ results/          # generated (metrics CSVs)
└─ data_raw/         # downloaded locally (BEIR datasets)

````

> Recommended `.gitignore`: `.venv/`, `data_raw/`, `indexes/`, `storage/` (and other generated artifacts).

---

## Setup

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
````

### 2) Install dependencies

If you have `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise (minimal set):

```bash
pip install beir bm25s sentence-transformers faiss-cpu streamlit pandas numpy tqdm
```

---

## Data: BEIR SciFact

Place SciFact under:

```
data_raw/scifact/scifact/
  ├─ corpus.jsonl
  ├─ queries.jsonl
  └─ qrels/
```

(Keep the BEIR folder structure consistent.)

---

## Build Dense Indices (BGE / E5)

Dense retrieval uses FAISS indices stored under `indexes/`.

```bash
python -m src.build_dense_index
```

Expected files:

* `indexes/faiss_scifact_bge.index`
* `indexes/faiss_scifact_bge_docids.json`
* `indexes/faiss_scifact_e5.index`
* `indexes/faiss_scifact_e5_docids.json`

---

## Evaluate

### BM25

```bash
python -m src.eval_bm25
```

### Dense (BGE)

```bash
python -m src.eval_dense_bge
```

### Dense (E5)

```bash
python -m src.eval_dense_e5
```

### Hybrid (BM25 + BGE, RRF)

```bash
python -m src.eval_hybrid_rrf
```

### Merge into one CSV

```bash
python -m src.summarize_metrics
```

Output:

* `results/metrics_scifact_all.csv`

---

## Current Results (SciFact)

| Method                   | NDCG@10 | Recall@100 |  MAP@10 |    P@10 |
| ------------------------ | ------: | ---------: | ------: | ------: |
| BM25                     | 0.66169 |    0.87589 | 0.61988 | 0.08600 |
| Dense (BGE)              | 0.72000 |    0.95333 | 0.67636 | 0.09533 |
| Dense (E5)               | 0.71943 |    0.96267 | 0.67307 | 0.09533 |
| Hybrid (BM25 + BGE, RRF) | 0.71060 |    0.96500 | 0.66770 | 0.09367 |

**Takeaways**

* **BGE** slightly leads on top-10 ranking quality (**NDCG@10 / MAP@10**).
* **E5** leads on **Recall@100**, improving evidence coverage.
* **Hybrid (RRF)** further improves recall (good for RAG evidence coverage) with a small trade-off on top-10 metrics.

---

## Run the Demo (Streamlit)

```bash
streamlit run app.py
```

In the UI you can:

* select **BM25 / BGE / E5 / Hybrid**
* choose a SciFact query ID (or type your own)
* view retrieved docs with snippets/full text
* view **qrels** and **hit@k**

---

## Notes

* **E5 formatting**: queries should be encoded as `query: <text>` (as implemented in the demo).
* If FAISS is unstable on your machine, consider running single-threaded FAISS or adding a numpy fallback for the demo.

---

## License

MIT
