"""Ranking metrics used by the held-out recommendation evaluation.

Used by eval/recommend_eval.py (path ranker vs popularity / item-CF).
Not used by GraphRAG QA — that is eval/qa_eval.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def precision_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(predicted[:k]) & relevant) / k


def recall_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(predicted[:k]) & relevant) / len(relevant)


def hit_rate_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    return float(bool(set(predicted[:k]) & relevant))


def ndcg_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    """Binary nDCG@K — relevant items have gain 1."""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _mean_se(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, se


def summarize_user_rows(per_user: pd.DataFrame, method: str) -> dict:
    if per_user.empty:
        return {
            "method": method,
            "n_users": 0,
            "precision@k": np.nan,
            "recall@k": np.nan,
            "hit_rate@k": np.nan,
            "ndcg@k": np.nan,
            "ndcg@k_se": np.nan,
        }
    p_mean, _ = _mean_se(per_user["precision_at_k"])
    r_mean, _ = _mean_se(per_user["recall_at_k"])
    h_mean, _ = _mean_se(per_user["hit_rate_at_k"])
    n_mean, n_se = _mean_se(per_user["ndcg_at_k"])
    return {
        "method": method,
        "n_users": int(len(per_user)),
        "precision@k": p_mean,
        "recall@k": r_mean,
        "hit_rate@k": h_mean,
        "ndcg@k": n_mean,
        "ndcg@k_se": n_se,
    }


def paired_win_rates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    metric: str = "ndcg_at_k",
) -> dict:
    """Share of shared users where left beats / ties / loses to right."""
    a = left.set_index("userId")[metric]
    b = right.set_index("userId")[metric]
    both = a.index.intersection(b.index)
    delta = a.loc[both] - b.loc[both]
    n = int(len(delta))
    return {
        "left": left_name,
        "right": right_name,
        "metric": metric,
        "n_users": n,
        "mean_delta": float(delta.mean()) if n else float("nan"),
        "pct_left_better": float((delta > 0).mean()) if n else float("nan"),
        "pct_tie": float((delta == 0).mean()) if n else float("nan"),
        "pct_right_better": float((delta < 0).mean()) if n else float("nan"),
    }
