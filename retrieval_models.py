# retrieval_models.py
import math

import numpy as np

from document_processing import FIELDS
from inverted_index import query_terms, postings, length_norm


#  plain BM25
def bm25(q_tokens, index, field="all", k1=1.2, b=0.75):
    """field="all" scores the concatenated document, any other name one field."""
    scores = np.zeros(index["N"])
    matched = np.zeros(index["N"], dtype=bool)
    norm = length_norm(index, field, b)
    for t, qtf in query_terms(q_tokens, index):
        rows, tf = postings(index, field, t)       # all products containing t, and their tf
        scores[rows] += qtf * index["idf"][t] * (k1 + 1) * tf / (tf + k1 * norm[rows])
        matched[rows] = True
    return np.where(matched, scores, -np.inf)



#  linear combination of fields
def linear_combination(q_tokens, index, weights=None, k1=1.2, b=0.75):
    """sum_c w_c * BM25_c(q, d), with k1 and b shared by all fields (saturation per field)."""
    ### YOUR CODE HERE
    pass



#  BM25 variants
def bm25l(q_tokens, index, k1=1.2, b=0.75, delta=0.5, field="all"):
    """BM25L (Lv & Zhai, 2011): the length-normalised tf is shifted up by delta."""
    ### YOUR CODE HERE
    pass



def raw_entropy(index, t, field="all"):
    """Ẽ(t) = -Σ p log p over the products containing t, with p = sigmoid(tf)."""
    ### YOUR CODE HERE
    pass


def bmx_defaults(index, field="all"):
    """Paper defaults: alpha = max(min(1.5, avgdl / 100), 0.5), beta = 1 / log(1 + N)."""
    ### YOUR CODE HERE
    pass


def bmx(q_tokens, index, alpha=None, beta=None, field="all"):
    """BMX (Li et al., 2024): entropy-weighted BM25 plus a query-document overlap term."""
    ### YOUR CODE HERE
    pass


# BM25F
def bm25f(q_tokens, index, k1=1.2, boost=None, b=None, fields=FIELDS):
    """BM25F exactly as in Pérez-Iglesias et al. (arXiv:0911.5046): combine the boosted,
    length-normalised field frequencies first, then saturate once:
        weight(t,d) = sum_c boost_c * tf_c / ((1 - b_c) + b_c * l_c / avl_c)
        R(q,d)      = sum_{t in q} idf(t) * weight(t,d) / (k1 + weight(t,d))
    with the RSJ idf(t) = log((N - df + 0.5) / (df + 0.5)) (index["idf_rsj"]),
    no query term weights and no (k1 + 1)."""
    ### YOUR CODE HERE
    pass
