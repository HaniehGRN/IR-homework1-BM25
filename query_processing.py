# query_processing.py
"""Loading WANDS queries and relevance labels, and the tune / test split."""
import random

import pandas as pd

from document_processing import DATA_DIR, preprocess

# Graded gains of the WANDS paper (used for nDCG); only Exact counts as relevant for the other metrics
GAIN = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}


def load_queries(data_dir=DATA_DIR):
    """{query_id: query text}"""
    query_df = pd.read_csv(f"{data_dir}/query.csv", sep="\t", dtype=str)
    return dict(zip(query_df["query_id"], query_df["query"]))


def preprocess_queries(queries):
    """{query_id: list of preprocessed tokens} (same preprocessing as the products)."""
    return {q: preprocess(text) for q, text in queries.items()}


def load_judgments(doc_ids, data_dir=DATA_DIR):
    """Return (label_df, gains, relevant).

    gains[q]    : {product row: graded gain} for every judged product of query q
    relevant[q] : set of product rows labelled Exact; queries without any Exact product are dropped
    """
    label_df = pd.read_csv(f"{data_dir}/label.csv", sep="\t", dtype=str)
    row_of = {d: i for i, d in enumerate(doc_ids)}
    gains = {}
    for q, d, label in zip(label_df["query_id"], label_df["product_id"], label_df["label"]):
        gains.setdefault(q, {})[row_of[d]] = GAIN[label]
    relevant = {q: {r for r, g in gd.items() if g == 1.0} for q, gd in gains.items()}
    relevant = {q: rel for q, rel in relevant.items() if rel}
    return label_df, gains, relevant


def split_queries(qids, seed):
    """Shuffle once with a fixed seed and split into a tune half and a test half."""
    qids = sorted(qids, key=int)
    shuffled = qids[:]
    random.Random(seed).shuffle(shuffled)
    tune_qids = sorted(shuffled[: len(qids) // 2], key=int)
    test_qids = sorted(shuffled[len(qids) // 2:], key=int)
    return tune_qids, test_qids
