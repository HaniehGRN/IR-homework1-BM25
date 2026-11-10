# inverted_index.py
"""Fielded sparse index: one product x term matrix per field."""
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from document_processing import COLUMNS, preprocess


def build_index(products, fields):
    """Sparse fielded index. Returns a dict of plain arrays / matrices.

    tf[f]      : CSC matrix (products x terms) of term frequencies in field f; tf["all"] is their sum
    length[f]  : length of field f of every product
    avgdl[f]   : average length of field f
    df, idf    : document frequency (on the whole product) and idf of every term.
                 idf is the non-negative Okapi / Lucene form log(1 + (N - df + 0.5) / (df + 0.5)):
                 the RSJ idf log((N - df + 0.5) / (df + 0.5)) is negative for df > N / 2.
    idf_rsj    : the original RSJ idf (used by BM25F, as in Pérez-Iglesias et al.)
    tokens[f]  : the preprocessed tokens of field f of every product (used for sanity checks)
    """
    field_tokens = {f: [preprocess(text) for text in products[COLUMNS[f]]] for f in fields}
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens)        # input is already tokenised
    vectorizer.fit([toks for f in fields for toks in field_tokens[f]])  # one shared vocabulary
    tf = {f: vectorizer.transform(field_tokens[f]).astype(float).tocsc() for f in fields}
    tf["all"] = sum(tf[f] for f in fields).tocsc()
    length = {f: np.asarray(tf[f].sum(axis=1)).ravel() for f in tf}
    N = tf["all"].shape[0]
    df = np.diff(tf["all"].indptr)                                      # products containing each term
    return {"N": N, "vocab": vectorizer.vocabulary_, "tf": tf, "length": length,
            "avgdl": {f: length[f].mean() for f in tf}, "df": df,
            "idf": np.log(1 + (N - df + 0.5) / (df + 0.5)),
            "idf_rsj": np.log((N - df + 0.5) / (df + 0.5)), "tokens": field_tokens}


def query_terms(tokens, index):
    """[(term id, query term frequency)] for the query terms that occur in the collection."""
    return [(index["vocab"][t], c) for t, c in Counter(tokens).items() if t in index["vocab"]]


def postings(index, field, term_id):
    """Posting list of a term in a field = one column of the field matrix: (product rows, term frequencies)."""
    m = index["tf"][field]
    start, end = m.indptr[term_id], m.indptr[term_id + 1]
    return m.indices[start:end], m.data[start:end]


def length_norm(index, field, b):
    """1 - b + b * |d_field| / avgdl_field   for every product."""
    return 1 - b + b * index["length"][field] / index["avgdl"][field]
