# parameter_search.py
import itertools

import numpy as np
import pandas as pd


def grid_search(evaluator, score_fn, grid, qids, **fixed):
    """Evaluate MAP for every combination of the parameter values in `grid`.

    evaluator : ev.Evaluator (use evaluator.mean_ap(score_fn, qids, **params))
    score_fn  : scoring function, e.g. rm.bm25
    grid      : {parameter name: list of values}, e.g. {"k1": [0.5, 1.0], "b": [0.25, 0.75]}
    qids      : query ids to evaluate on
    **fixed   : parameters that stay fixed, e.g. field="all"

    Returns a DataFrame with one column per parameter and a "MAP" column,
    one row per combination, sorted by MAP (best first) and with index 0, 1, 2, ...
    """
    ### YOUR CODE HERE
    comb_rows = []
    h_params = list(grid.keys())
    h_params_range = [grid[h_param] for h_param in h_params]

    for combination in itertools.product(*h_params_range):

        name_comb = dict(zip(h_params, combination))
        params = {**fixed, **name_comb}
        MAP_per_comb = evaluator.mean_ap(score_fn, qids, **params)
        comb_rows.append({**name_comb, "MAP": MAP_per_comb})

    return (pd.DataFrame(comb_rows, columns=[*h_params, "MAP"])).sort_values("MAP", ascending=False).reset_index(drop=True)




def coordinate_ascent(objective, start, candidates, max_rounds=4, verbose=True):
    """Maximise `objective` by optimising one parameter at a time.

    objective  : function {parameter name: value} -> score (e.g. MAP on the tuning queries)
    start      : starting parameters {name: value}
    candidates : {name: list of values to try}; parameters not listed here stay fixed
    In each round, for every parameter in `candidates`, try all its values while the others
    stay fixed and keep a value only if it improves the score by more than 1e-6.
    Stop after a round with no improvement, or after `max_rounds` rounds.
    If verbose, print the best score after every round:  "  round r: MAP=0.1234"

    Returns (params, best, history): the best parameters, their score, and the list of the
    best score at the start and after every parameter sweep.
    """
    ### YOUR CODE HERE

    h_params = start.copy()
    best = objective(h_params)
    history = [best]

    for round in range(max_rounds):
        improved_score_flag = False
        for name, values in candidates.items():
            sweep_h_params = h_params
            sweep_best_score = best
            for value in values:
                h_param_pair = h_params.copy()
                h_param_pair[name] = value
                score = objective(h_param_pair)
                if score > sweep_best_score:
                    sweep_h_params = h_param_pair
                    sweep_best_score = score

            if (sweep_best_score - best) > 1e-6:
                h_params = sweep_h_params
                best = sweep_best_score
                improved_score_flag = True

            history.append(best)
        if verbose: print(f"  round {round + 1}: MAP={best:.4f}")
        if not improved_score_flag: break

    return h_params, best, history


