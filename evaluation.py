# evaluation.py
"""Ranking, evaluation metrics, and significance testing.
(The parameter-search helpers are in parameter_search.py.)"""
import math

import numpy as np
import pandas as pd

METRICS = ["MAP", "MRR", "P@3", "P@10", "R@3", "R@10", "nDCG@10"]


def rank(scores, depth=1000):
    """Rows of the top `depth` products, best first (only products with a finite score)."""
    candidates = np.flatnonzero(np.isfinite(scores))
    order = np.argsort(-scores[candidates], kind="stable")[:depth]
    return candidates[order]


def query_metrics(ranking, rel, graded):
    """Metrics of one ranking. rel: set of Exact rows, graded: {row: gain} of all judged rows."""
    hits, precisions, first_hit = 0, [], None
    for i, r in enumerate(ranking, start=1):
        if r in rel:
            hits += 1
            precisions.append(hits / i)
            first_hit = first_hit or i
    top3 = sum(r in rel for r in ranking[:3])
    top10 = sum(r in rel for r in ranking[:10])
    dcg = sum(graded.get(r, 0.0) / math.log2(i + 2) for i, r in enumerate(ranking[:10]))
    ideal = sorted(graded.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return {"AP": sum(precisions) / len(rel), "RR": 1 / first_hit if first_hit else 0.0,
            "P@3": top3 / 3, "P@10": top10 / 10, "R@3": top3 / len(rel), "R@10": top10 / len(rel),
            "nDCG@10": dcg / idcg}


def summary(per_query):
    """Macro average of the per-query metrics, with the usual metric names."""
    s = per_query.mean().rename({"AP": "MAP", "RR": "MRR"})
    return s[METRICS]


class Evaluator:
    """Holds the index, the preprocessed queries and the judgments, so that a scoring function
    can be evaluated with one call."""

    def __init__(self, index, query_tokens, gains, relevant):
        self.index = index
        self.query_tokens = query_tokens
        self.gains = gains
        self.relevant = relevant

    def evaluate(self, score_fn, qids, **params):
        """Per-query metrics (DataFrame) of score_fn(query_tokens, index, **params)."""
        rows = {q: query_metrics(rank(score_fn(self.query_tokens[q], self.index, **params)),
                                 self.relevant[q], self.gains[q])
                for q in qids}
        return pd.DataFrame.from_dict(rows, orient="index")

    def mean_ap(self, score_fn, qids, **params):
        return self.evaluate(score_fn, qids, **params)["AP"].mean()


def significance_test(runs, baseline, metrics=("AP", "nDCG@10"), alpha=0.05):
    """Paired t-test of every run against `baseline` on the per-query values of each metric."""
    ### YOUR CODE HERE
    pass
