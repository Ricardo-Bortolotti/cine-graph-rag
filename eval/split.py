"""Train/test split matching notebooks/03_baseline_recommenders.ipynb.

Used by the path-ranker holdout (eval/recommend_eval.py), not by GraphRAG QA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_RATINGS = 5
TOP_K = 10
MIN_RATING_RELEVANT = 4.0
N_SIMILAR = 40


def default_data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "raw" / "ml-latest-small" / "ml-latest-small"


def load_movielens(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = data_dir or default_data_dir()
    movies = pd.read_csv(folder / "movies.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    return movies, ratings


def user_holdout_split(
    ratings_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_ratings: int = MIN_RATINGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings per user so every test user also appears in train."""
    train_parts, test_parts = [], []
    for _, user_ratings in ratings_df.groupby("userId"):
        if len(user_ratings) < min_ratings:
            train_parts.append(user_ratings)
            continue
        train_u, test_u = train_test_split(
            user_ratings,
            test_size=test_size,
            random_state=random_state,
        )
        train_parts.append(train_u)
        test_parts.append(test_u)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )
