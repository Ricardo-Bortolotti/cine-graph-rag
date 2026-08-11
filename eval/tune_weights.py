"""Cache GraphRecommender signals once, then search weights offline.

Re-querying Neo4j for every weight vector is too slow. Instead:

1. Pull candidates with unit weights and a wide per-seed limit.
2. Store raw signal counts per (user, movie).
3. Rescore in NumPy / pandas for each trial.
4. Pick weights on a *tune* split; report nDCG on a held-out user split.

The candidate pool is frozen at collect time, so a weight that would have
surfaced a movie outside the top-`per_seed_limit` cannot be discovered.

This tunes the path ranker only. It does not touch GraphRAG or vector RAG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "recommender"))

from recommend_eval import liked_seed_ids  # noqa: E402
from ranking import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from split import (  # noqa: E402
    MIN_RATING_RELEVANT,
    RANDOM_STATE,
    TOP_K,
    load_movielens,
    user_holdout_split,
)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from graph_recommender import (  # noqa: E402
    PRIOR_COEF,
    SIGNAL_KEYS,
    UNIT_WEIGHTS,
    WEIGHTS,
    GraphRecommender,
    evidence_raw_counts,
    score_raw_features,
)
from neo4j_config import connect_neo4j  # noqa: E402

FEATURE_COLS = list(SIGNAL_KEYS)


def sample_eval_users(
    test_ratings: pd.DataFrame,
    train_user_ids: set[int],
    *,
    max_users: int,
    min_rating: float = MIN_RATING_RELEVANT,
    random_state: int = RANDOM_STATE,
) -> list[int]:
    relevant = test_ratings.loc[test_ratings["rating"] >= min_rating]
    users = [int(u) for u in relevant["userId"].unique() if int(u) in train_user_ids]
    if max_users > 0 and len(users) > max_users:
        rng = np.random.default_rng(random_state)
        users = [int(u) for u in rng.choice(users, size=max_users, replace=False)]
    return users


def relevant_by_user(
    test_ratings: pd.DataFrame,
    user_ids: list[int],
    min_rating: float = MIN_RATING_RELEVANT,
) -> dict[int, set[int]]:
    relevant = test_ratings.loc[
        test_ratings["userId"].isin(user_ids) & (test_ratings["rating"] >= min_rating)
    ]
    return {
        int(uid): set(int(mid) for mid in group["movieId"])
        for uid, group in relevant.groupby("userId")
    }


def collect_feature_table(
    *,
    max_users: int = 200,
    n_seeds: int = 5,
    per_seed_limit: int = 40,
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    _, ratings = load_movielens()
    train, test = user_holdout_split(ratings)
    train_users = set(int(u) for u in train["userId"].unique())
    users = sample_eval_users(test, train_users, max_users=max_users)
    truth = relevant_by_user(test, users)
    seen = {
        int(uid): set(int(mid) for mid in group["movieId"])
        for uid, group in train.groupby("userId")
    }

    driver, database = connect_neo4j()
    graph = GraphRecommender(driver, database, weights=dict(UNIT_WEIGHTS))
    rows: list[dict] = []
    try:
        for index, user_id in enumerate(users, start=1):
            if index == 1 or index % 25 == 0 or index == len(users):
                print(f"  collect features: user {index}/{len(users)}", flush=True)
            seeds = liked_seed_ids(train, user_id, n_seeds=n_seeds)
            blocked = set(seen.get(user_id, set()))
            blocked.update(seeds)
            acc: dict[int, dict] = {}
            for seed_id in seeds:
                for rec in graph.recommend(int(seed_id), limit=per_seed_limit):
                    if rec.movie_id in blocked:
                        continue
                    raw = evidence_raw_counts(rec.evidence, UNIT_WEIGHTS)
                    slot = acc.get(rec.movie_id)
                    if slot is None:
                        acc[rec.movie_id] = {
                            **raw,
                            "n_seed_hits": 1,
                            "n_ratings": rec.n_ratings,
                            "avg_rating": rec.avg_rating,
                        }
                    else:
                        for key in FEATURE_COLS:
                            slot[key] += raw[key]
                        slot["n_seed_hits"] += 1
            for movie_id, slot in acc.items():
                rows.append({"userId": user_id, "movieId": movie_id, **slot})
    finally:
        driver.close()

    return pd.DataFrame(rows), truth


def rank_from_features(
    user_rows: pd.DataFrame,
    weights: dict[str, float],
    *,
    prior_coef: float,
    k: int,
) -> list[int]:
    if user_rows.empty:
        return []
    scores = [
        score_raw_features(
            {key: float(row[key]) for key in FEATURE_COLS},
            weights,
            n_ratings=int(row["n_ratings"]),
            avg_rating=row["avg_rating"],
            n_seed_hits=int(row["n_seed_hits"]),
            prior_coef=prior_coef,
        )
        for row in user_rows.to_dict("records")
    ]
    order = np.argsort(np.asarray(scores))[::-1]
    return [int(user_rows.iloc[int(i)]["movieId"]) for i in order[:k]]


def evaluate_weights(
    features: pd.DataFrame,
    truth: dict[int, set[int]],
    user_ids: list[int],
    weights: dict[str, float],
    *,
    prior_coef: float = PRIOR_COEF,
    k: int = TOP_K,
) -> dict[str, float]:
    rows = []
    for user_id in user_ids:
        relevant = truth.get(user_id) or set()
        if not relevant:
            continue
        predicted = rank_from_features(
            features.loc[features["userId"] == user_id],
            weights,
            prior_coef=prior_coef,
            k=k,
        )
        rows.append(
            {
                "precision": precision_at_k(predicted, relevant, k),
                "recall": recall_at_k(predicted, relevant, k),
                "ndcg": ndcg_at_k(predicted, relevant, k),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_users": 0, "precision@k": float("nan"), "recall@k": float("nan"), "ndcg@k": float("nan")}
    return {
        "n_users": int(len(frame)),
        "precision@k": float(frame["precision"].mean()),
        "recall@k": float(frame["recall"].mean()),
        "ndcg@k": float(frame["ndcg"].mean()),
    }


def sample_weights(rng: np.random.Generator) -> dict[str, float]:
    """Positive weights; director stays in a range that can still win explanations."""
    return {
        "director": float(rng.choice([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])),
        "actor": float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])),
        "keyword": float(rng.choice([0.5, 1.0, 2.0, 3.0, 4.0])),
        "genre": float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0])),
        "co_rating": float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0])),
    }


def split_users(
    user_ids: list[int],
    *,
    tune_frac: float = 0.7,
    random_state: int = RANDOM_STATE,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(random_state)
    shuffled = [int(u) for u in rng.permutation(user_ids)]
    cut = max(1, int(len(shuffled) * tune_frac))
    if cut >= len(shuffled):
        cut = len(shuffled) - 1
    return shuffled[:cut], shuffled[cut:]


def run_weight_search(
    *,
    max_users: int = 200,
    n_seeds: int = 5,
    trials: int = 60,
    per_seed_limit: int = 40,
    prior_coef: float = PRIOR_COEF,
    k: int = TOP_K,
    random_state: int = RANDOM_STATE,
) -> dict:
    print("Collecting graph features from Neo4j (once)...", flush=True)
    features, truth = collect_feature_table(
        max_users=max_users,
        n_seeds=n_seeds,
        per_seed_limit=per_seed_limit,
    )
    user_ids = sorted(int(u) for u in features["userId"].unique())
    tune_users, hold_users = split_users(user_ids, random_state=random_state)
    print(
        f"Users with candidates: {len(user_ids)} "
        f"(tune={len(tune_users)}, holdout={len(hold_users)})",
        flush=True,
    )

    rng = np.random.default_rng(random_state)
    trials_payload = [{"weights": dict(WEIGHTS), "prior_coef": prior_coef, "tag": "default"}]
    for _ in range(max(0, trials - 1)):
        trials_payload.append(
            {"weights": sample_weights(rng), "prior_coef": prior_coef, "tag": "random"}
        )

    records = []
    best = None
    for index, trial in enumerate(trials_payload, start=1):
        tune_metrics = evaluate_weights(
            features, truth, tune_users, trial["weights"], prior_coef=trial["prior_coef"], k=k
        )
        hold_metrics = evaluate_weights(
            features, truth, hold_users, trial["weights"], prior_coef=trial["prior_coef"], k=k
        )
        row = {
            "trial": index,
            "tag": trial["tag"],
            **{f"w_{key}": trial["weights"][key] for key in SIGNAL_KEYS},
            "prior_coef": trial["prior_coef"],
            "tune_ndcg": tune_metrics["ndcg@k"],
            "hold_ndcg": hold_metrics["ndcg@k"],
            "tune_precision": tune_metrics["precision@k"],
            "hold_precision": hold_metrics["precision@k"],
        }
        records.append(row)
        if best is None or row["tune_ndcg"] > best["tune_ndcg"]:
            best = row
        if index == 1 or index % 10 == 0:
            print(
                f"  trial {index}/{len(trials_payload)} "
                f"tune_ndcg={row['tune_ndcg']:.4f} hold_ndcg={row['hold_ndcg']:.4f}",
                flush=True,
            )

    table = pd.DataFrame(records).sort_values("tune_ndcg", ascending=False)
    default_row = table.loc[table["tag"] == "default"].iloc[0]
    assert best is not None
    return {
        "n_feature_rows": int(len(features)),
        "n_tune_users": len(tune_users),
        "n_hold_users": len(hold_users),
        "k": k,
        "n_seeds": n_seeds,
        "per_seed_limit": per_seed_limit,
        "default": {
            "weights": dict(WEIGHTS),
            "tune_ndcg": float(default_row["tune_ndcg"]),
            "hold_ndcg": float(default_row["hold_ndcg"]),
            "hold_precision": float(default_row["hold_precision"]),
        },
        "best_on_tune": {
            "weights": {key: float(best[f"w_{key}"]) for key in SIGNAL_KEYS},
            "tune_ndcg": float(best["tune_ndcg"]),
            "hold_ndcg": float(best["hold_ndcg"]),
            "hold_precision": float(best["hold_precision"]),
            "tag": best["tag"],
        },
        "leaderboard": table.head(8).to_dict(orient="records"),
    }
