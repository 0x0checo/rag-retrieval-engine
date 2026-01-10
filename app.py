import streamlit as st
import bm25s
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip

BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
}

def ensure_dataset(dataset: str, data_root: Path) -> Path:
    """
    Ensure BEIR dataset exists locally. If not, download and unzip.
    Returns the BEIR data_folder path that contains corpus.jsonl.
    """
    data_root.mkdir(parents=True, exist_ok=True)

    data_folder = data_root / dataset / dataset  # data_raw/scifact/scifact
    corpus_file = data_folder / "corpus.jsonl"

    if corpus_file.exists():
        return data_folder

    url = BEIR_URLS[dataset]
    download_and_unzip(url, str(data_root))  # downloads to data_raw/<dataset>/
    if not corpus_file.exists():
        raise FileNotFoundError(f"Download finished but {corpus_file} not found. Check dataset structure.")
    return data_folder


@st.cache_resource
def load_scifact():
    data_root = Path("data_raw")
    data_folder = ensure_dataset("scifact", data_root)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(split="test")
    return corpus, queries, qrels


def build_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip()

@st.cache_resource
def build_bm25(corpus):
    doc_ids = list(corpus.keys())
    texts = [build_text(corpus[did]) for did in doc_ids]

    bm25 = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    bm25.index(corpus_tokens)
    return bm25, doc_ids, texts

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for docs in rank_lists:
        for r, did in enumerate(docs, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    return scores

@st.cache_data
def load_metrics_table():
    path = Path("results") / "metrics_scifact_all.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_bge_resources():
    dataset = "scifact"
    tag = "bge"
    index = faiss.read_index(f"indexes/faiss_{dataset}_{tag}.index")
    with open(f"indexes/faiss_{dataset}_{tag}_docids.json") as f:
        dense_doc_ids = json.load(f)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return index, dense_doc_ids, model

@st.cache_resource
def load_e5_resources():
    dataset = "scifact"
    tag = "e5"
    index = faiss.read_index(f"indexes/faiss_{dataset}_{tag}.index")
    with open(f"indexes/faiss_{dataset}_{tag}_docids.json") as f:
        dense_doc_ids = json.load(f)

    model = SentenceTransformer("intfloat/e5-base-v2")
    return index, dense_doc_ids, model

st.set_page_config(page_title="RAG Retrieval Engine", layout="wide")
st.title("RAG Retrieval Engine")
st.sidebar.header("Benchmark (SciFact)")
metrics_df = load_metrics_table()
if metrics_df is None:
    st.sidebar.warning("metrics_scifact_all.csv not found. Run summarize_metrics first.")
else:
    st.sidebar.dataframe(metrics_df, use_container_width=True)

corpus, queries, qrels = load_scifact()
bm25, doc_ids, texts = build_bm25(corpus)
bge_index, bge_doc_ids, bge_model = load_bge_resources()
e5_index, e5_doc_ids, e5_model = load_e5_resources()

st.success(f"Loaded SciFact ✅ Docs={len(corpus)} Queries={len(queries)}")
st.divider()
st.subheader("Search")
col_left, col_right = st.columns([2, 1], gap="large")


method = st.sidebar.selectbox(
    "Retrieval method",
    ["BM25", "Dense (BGE)", "Dense (E5)", "Hybrid (BM25 + BGE, RRF)"],
    index=0
)

if "query_text" not in st.session_state:
    st.session_state["query_text"] = "Antiretroviral therapy reduces rates of tuberculosis."

st.caption("Tip: pick a SciFact query to test quickly")
qid_options = list(queries.keys())
pick_qid = st.selectbox("Pick a SciFact query id (optional)", ["(manual input)"] + qid_options)

if pick_qid != "(manual input)":
    if st.button("Use this SciFact query"):
        st.session_state["query_text"] = queries[pick_qid]
        st.rerun()

query = st.text_input("Enter a query", key="query_text")
topk = st.sidebar.slider("Top K", min_value=5, max_value=50, value=10, step=5)

if st.button("Search"):
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.write(f"Results for: **{query}**")
        st.caption(f"Method: {method}")
        if pick_qid != "(manual input)":
            rel_set = set(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else set()


        if method == "BM25":
            q_tokens = bm25s.tokenize(query, stopwords="en")
            if len(q_tokens) > 0 and isinstance(q_tokens[0], int):
                q_tokens = [q_tokens]

            idxs, scores = bm25.retrieve(q_tokens, k=topk)
            idxs_list = idxs.tolist()[0] if hasattr(idxs, "tolist") else list(idxs)[0]
            retrieved_doc_ids = [doc_ids[int(i)] for i in idxs_list]
            if pick_qid != "(manual input)":
                rel_set = set(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else set()
                hit = any(d in rel_set for d in retrieved_doc_ids[:topk])
                st.success(f"hit@{topk}: {hit}")

            scores_list = scores.tolist()[0] if hasattr(scores, "tolist") else list(scores)[0]

            for rank, (i, s) in enumerate(zip(idxs_list, scores_list), start=1):
                i = int(i)
                did = doc_ids[i]
                snippet = texts[i][:240].replace("\n", " ")
                st.markdown(f"**{rank}. doc_id:** `{did}`  |  **score:** {float(s):.4f}")
                st.write(snippet)
                with st.expander("Show full text"):
                    st.text(texts[i])
                st.divider()

        elif method == "Dense (BGE)":
            qvec = bge_model.encode([query], convert_to_numpy=True)
            qvec = l2_normalize(qvec.astype("float32"))
            qvec = np.ascontiguousarray(qvec, dtype="float32")

            scores, idxs = bge_index.search(qvec, topk)
            scores = scores[0]
            idxs = idxs[0]
            retrieved_doc_ids = [bge_doc_ids[int(i)] for i in idxs]
            if pick_qid != "(manual input)":
                rel_set = set(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else set()
                hit = any(d in rel_set for d in retrieved_doc_ids[:topk])
                st.success(f"hit@{topk}: {hit}")


            for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
                did = bge_doc_ids[int(i)]
                doc = corpus[did]
                snippet = build_text(doc)[:240].replace("\n", " ")

                st.markdown(f"**{rank}. doc_id:** `{did}`  |  **score:** {float(s):.4f}")
                st.write(snippet)
                with st.expander("Show full text"):
                    st.text(build_text(doc))
                st.divider()

        elif method == "Dense (E5)":
            # add prefix
            q = "query: " + query

            qvec = e5_model.encode([q], convert_to_numpy=True)
            qvec = l2_normalize(qvec.astype("float32"))
            qvec = np.ascontiguousarray(qvec, dtype="float32")

            scores, idxs = e5_index.search(qvec, topk)
            scores = scores[0]
            idxs = idxs[0]
            retrieved_doc_ids = [e5_doc_ids[int(i)] for i in idxs]
            if pick_qid != "(manual input)":
                rel_set = set(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else set()
                hit = any(d in rel_set for d in retrieved_doc_ids[:topk])
                st.success(f"hit@{topk}: {hit}")


            for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
                did = e5_doc_ids[int(i)]
                doc = corpus[did]
                snippet = build_text(doc)[:240].replace("\n", " ")

                st.markdown(f"**{rank}. doc_id:** `{did}`  |  **score:** {float(s):.4f}")
                st.write(snippet)
                with st.expander("Show full text"):
                    st.text(build_text(doc))
                st.divider()

        else:  # Hybrid (BM25 + BGE, RRF)
            fuse_k = st.sidebar.slider("RRF k (fusion strength)", 10, 200, 60, 10)
            # --- BM25 rank list ---
            q_tokens = bm25s.tokenize(query, stopwords="en")
            if len(q_tokens) > 0 and isinstance(q_tokens[0], int):
                q_tokens = [q_tokens]
            bm25_idxs, bm25_scores = bm25.retrieve(q_tokens, k=topk)
            bm25_idxs = bm25_idxs.tolist()[0] if hasattr(bm25_idxs, "tolist") else list(bm25_idxs)[0]
            bm25_rank = [doc_ids[int(i)] for i in bm25_idxs]

            # --- BGE rank list ---
            qvec = bge_model.encode([query], convert_to_numpy=True)
            qvec = l2_normalize(qvec.astype("float32"))
            qvec = np.ascontiguousarray(qvec, dtype="float32")
            dense_scores, dense_idxs = bge_index.search(qvec, topk)
            dense_idxs = dense_idxs[0]
            bge_rank = [bge_doc_ids[int(i)] for i in dense_idxs]

            # --- RRF fuse ---
            fused = rrf_fuse([bm25_rank, bge_rank], k=fuse_k)
            fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
            retrieved_doc_ids = [did for did, _ in fused_sorted]
            if pick_qid != "(manual input)":
                rel_set = set(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else set()
                hit = any(d in rel_set for d in retrieved_doc_ids[:topk])
                st.success(f"hit@{topk}: {hit}")

            bm25_pos = {did: r for r, did in enumerate(bm25_rank, start=1)}
            bge_pos = {did: r for r, did in enumerate(bge_rank, start=1)}

            for rank, (did, s) in enumerate(fused_sorted, start=1):
                doc = corpus[did]
                snippet = build_text(doc)[:240].replace("\n", " ")

                st.markdown(
                    f"**{rank}. doc_id:** `{did}`  |  **rrf:** {float(s):.6f}  "
                    f"|  bm25_rank: {bm25_pos.get(did, '-')}, bge_rank: {bge_pos.get(did, '-')}"
                )
                st.write(snippet)
                with st.expander("Show full text"):
                    st.text(build_text(doc))
                st.divider()
    with col_right:
        st.subheader("Ground Truth (qrels)")
        if pick_qid == "(manual input)":
            st.info("Pick a SciFact query id to see qrels ground truth.")
        else:
            rel_docs = list(qrels[str(pick_qid)].keys()) if str(pick_qid) in qrels else []
            st.write(f"Relevant docs: **{len(rel_docs)}**")
            if rel_docs:
                st.code("\n".join(rel_docs[:20]))
                if len(rel_docs) > 20:
                    st.caption(f"... +{len(rel_docs)-20} more")



# any_doc_id = next(iter(corpus))
# st.subheader("Sample doc")
# st.write("doc_id:", any_doc_id)
# st.write(corpus[any_doc_id])

