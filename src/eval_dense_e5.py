import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader 
from beir.retrieval.evaluation import EvaluateRetrieval 

def load_data_folder(dataset="scifact"):
    return Path("data_raw") / dataset / dataset

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def main():
    dataset = "scifact"
    model_name = "intfloat/e5-base-v2"
    tag = "e5"
    topk = 100

    data_folder = load_data_folder(dataset=dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(split="test")

    # load index and doc ids
    index = faiss.read_index(f"indexes/faiss_{dataset}_{tag}.index")
    with open(f"indexes/faiss_{dataset}_{tag}_docids.json") as f:
        doc_ids = json.load(f)

    model = SentenceTransformer(model_name)
    
    qids = list(queries.keys())
    qtexts = ["query: " + queries[qid] for qid in qids]
    qvecs = model.encode(qtexts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    qvecs = l2_normalize(qvecs.astype("float32"))

    scores, idxs = index.search(qvecs, topk)

    results = {}
    # convert to beir required format
    for qi, qid in enumerate(qids):
        pairs = [(doc_ids[int(i)], float(scores[qi, r])) for r, i in enumerate(idxs[qi])]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        results[qid] = dict(pairs)
    
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [10, 100])

    if isinstance(metrics, tuple):
        merged = {}
        for part in metrics:
            if isinstance(part, dict):
                merged.update(part)
        metrics_dict = merged
    else:
        metrics_dict = metrics

    out = {
        "dataset": dataset,
        "method": "dense_e5",
        "ndcg@10": metrics_dict["NDCG@10"],
        "recall@100": metrics_dict["Recall@100"],
        "map@10": metrics_dict["MAP@10"],
        "p@10": metrics_dict["P@10"],
    }

    print("Dense-E5 metrics:", out)

if __name__ == "__main__":
    main()