"""Held-out ranking: popularity, item-CF, and the graph recommender.

Uses the same per-user 80/20 split as notebooks/03_baseline_recommenders.ipynb.
Published protocol: 200 users, K=10, 5 liked seeds — see docs/graph_recommender.md.
This is the path ranker, not GraphRAG (docs/graph_rag.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "recommender"))

from ranking import (  # noqa: E402
    hit_rate_at_k,
    ndcg_at_k,
    paired_win_rates,
    precision_at_k,
    recall_at_k,
    summarize_user_rows,
)
from split import (  # noqa: E402
    MIN_RATING_RELEVANT,
    N_SIMILAR,
    RANDOM_STATE,
    TOP_K,
    load_movielens,
    user_holdout_split,
)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from graph_recommender import GraphRecommender  # noqa: E402
from neo4j_config import connect_neo4j  # noqa: E402


def build_popularity_table(ratings_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = ratings_df["rating"].mean()
    stats = ratings_df.groupby("movieId", as_index=False).agg(
        n_ratings=("rating", "size"),
        avg_rating=("rating", "mean"),
    )
    prior_m = float(stats["n_ratings"].median())
    stats["score"] = (stats["n_ratings"] / (stats["n_ratings"] + prior_m)) * stats[
        "avg_rating"
    ] + (prior_m / (stats["n_ratings"] + prior_m)) * global_mean
    return stats.sort_values("score", ascending=False).reset_index(drop=True)


class BaselineModels:
    """Popularity + item-CF on the notebook-03 training split."""

    def __init__(self, train_ratings: pd.DataFrame) -> None:
        self.train = train_ratings
        self.popularity = build_popularity_table(train_ratings)
        self.user_item = train_ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
        ).fillna(0.0)
        self.user_ids = self.user_item.index.to_numpy()
        self.movie_ids = self.user_item.columns.to_numpy()
        self.user_pos = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.values = self.user_item.values.astype(np.float64)
        self.item_sim = cosine_similarity(self.values.T)
        self.seen = {
            int(uid): set(group["movieId"].astype(int))
            for uid, group in train_ratings.groupby("userId")
        }

    def recommend_popularity(self, user_id: int, k: int = TOP_K) -> list[int]:
        seen = self.seen.get(int(user_id), set())
        ranked = self.popularity.loc[~self.popularity["movieId"].isin(seen), "movieId"]
        return [int(mid) for mid in ranked.head(k).tolist()]

    def recommend_item_cf(
        self,
        user_id: int,
        k: int = TOP_K,
        n_similar: int = N_SIMILAR,
    ) -> list[int]:
        if int(user_id) not in self.user_pos:
            return []
        u_pos = self.user_pos[int(user_id)]
        user_vec = self.values[u_pos]
        rated_pos = np.where(user_vec > 0)[0]
        if len(rated_pos) == 0:
            return []
        scores = np.zeros(len(self.movie_ids), dtype=np.float64)
        weights = np.zeros(len(self.movie_ids), dtype=np.float64)
        for j in rated_pos:
            sims = self.item_sim[j]
            top_idx = np.argpartition(sims, -n_similar)[-n_similar:]
            top_idx = top_idx[sims[top_idx] > 0]
            if len(top_idx) == 0:
                continue
            scores[top_idx] += sims[top_idx] * user_vec[j]
            weights[top_idx] += np.abs(sims[top_idx])
        preds = np.divide(
            scores, weights, out=np.zeros_like(scores), where=weights > 0
        )
        preds[rated_pos] = -np.inf
        order = np.argsort(preds)[::-1]
        out: list[int] = []
        for pos in order:
            if not np.isfinite(preds[pos]) or preds[pos] <= 0:
                continue
            out.append(int(self.movie_ids[pos]))
            if len(out) >= k:
                break
        return out


def liked_seed_ids(
    train_ratings: pd.DataFrame,
    user_id: int,
    min_rating: float = MIN_RATING_RELEVANT,
    n_seeds: int = 3,
) -> list[int]:
    liked = train_ratings.loc[
        (train_ratings["userId"] == user_id)
        & (train_ratings["rating"] >= min_rating)
    ].sort_values(["rating", "timestamp"], ascending=[False, False])
    if liked.empty:
        liked = train_ratings.loc[train_ratings["userId"] == user_id].sort_values(
            "rating", ascending=False
        )
    return [int(mid) for mid in liked["movieId"].head(n_seeds).tolist()]


def evaluate_method(
    recommend_fn,
    test_ratings: pd.DataFrame,
    train_user_ids: set[int],
    *,
    k: int = TOP_K,
    min_rating: float = MIN_RATING_RELEVANT,
    max_users: int = 40,
    random_state: int = RANDOM_STATE,
    method: str,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(random_state)
    relevant = test_ratings.loc[test_ratings["rating"] >= min_rating]
    users = [int(u) for u in relevant["userId"].unique() if int(u) in train_user_ids]
    if max_users > 0 and len(users) > max_users:
        users = [int(u) for u in rng.choice(users, size=max_users, replace=False)]

    rows = []
    for index, user_id in enumerate(users, start=1):
        if index == 1 or index % 25 == 0 or index == len(users):
            print(f"  {method}: user {index}/{len(users)}", flush=True)
        truth = set(
            int(mid)
            for mid in relevant.loc[relevant["userId"] == user_id, "movieId"].tolist()
        )
        if not truth:
            continue
        predicted = recommend_fn(user_id, k)
        rows.append(
            {
                "userId": user_id,
                "n_relevant": len(truth),
                "n_hits": len(set(predicted[:k]) & truth),
                "precision_at_k": precision_at_k(predicted, truth, k),
                "recall_at_k": recall_at_k(predicted, truth, k),
                "hit_rate_at_k": hit_rate_at_k(predicted, truth, k),
                "ndcg_at_k": ndcg_at_k(predicted, truth, k),
            }
        )
    per_user = pd.DataFrame(rows)
    return per_user, summarize_user_rows(per_user, method)


def run_recommendation_eval(
    *,
    max_users: int = 200,
    k: int = TOP_K,
    n_seeds: int = 5,
    include_graph: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict]]:
    """Return a summary table, per-user frames, and paired win-rate rows.

    ``max_users <= 0`` evaluates every user with a held-out relevant item.
    Graph seeds are the user's top liked train titles (not the full history).
    """
    _, ratings = load_movielens()
    train, test = user_holdout_split(ratings)
    models = BaselineModels(train)
    train_users = set(int(u) for u in train["userId"].unique())
    detail: dict[str, pd.DataFrame] = {}
    summaries = []

    for method, fn in (
        ("popularity", models.recommend_popularity),
        ("item_based_cf", models.recommend_item_cf),
    ):
        cap = "all" if max_users <= 0 else str(max_users)
        print(f"Evaluating {method} on up to {cap} users (K={k})...", flush=True)
        per_user, summary = evaluate_method(
            fn,
            test,
            train_users,
            k=k,
            max_users=max_users,
            method=method,
        )
        detail[method] = per_user
        summaries.append(summary)

    if include_graph:
        cap = "all" if max_users <= 0 else str(max_users)
        print(
            f"Evaluating graph_recommender on up to {cap} users "
            f"(K={k}, n_seeds={n_seeds})...",
            flush=True,
        )
        driver, database = connect_neo4j()
        graph = GraphRecommender(driver, database)

        def recommend_graph(user_id: int, limit: int = k) -> list[int]:
            seeds = liked_seed_ids(train, user_id, n_seeds=n_seeds)
            seen = models.seen.get(int(user_id), set())
            recs = graph.recommend_for_user(seeds, seen_ids=seen, limit=limit)
            return [rec.movie_id for rec in recs]

        per_user, summary = evaluate_method(
            recommend_graph,
            test,
            train_users,
            k=k,
            max_users=max_users,
            method="graph_recommender",
        )
        detail["graph_recommender"] = per_user
        summaries.append(summary)
        driver.close()

    paired: list[dict] = []
    if "graph_recommender" in detail:
        for other in ("popularity", "item_based_cf"):
            if other not in detail:
                continue
            for metric in ("ndcg_at_k", "hit_rate_at_k", "precision_at_k"):
                paired.append(
                    paired_win_rates(
                        detail["graph_recommender"],
                        detail[other],
                        left_name="graph_recommender",
                        right_name=other,
                        metric=metric,
                    )
                )

    return pd.DataFrame(summaries), detail, paired
