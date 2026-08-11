"""Load MovieLens into Neo4j: (User)-[:RATED {rating, timestamp}]->(Movie).

Idempotent: uses MERGE + uniqueness constraints. Safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from neo4j_config import connect_neo4j

LOGGER = logging.getLogger("build_graph")

CONSTRAINTS = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.title)",
    "CREATE INDEX rated_rating IF NOT EXISTS FOR ()-[r:RATED]-() ON (r.rating)",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def batched(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def create_schema(driver, database: str) -> None:
    with driver.session(database=database) as session:
        for stmt in CONSTRAINTS + INDEXES:
            LOGGER.info("Schema: %s", stmt)
            session.run(stmt)


def load_movies(driver, database: str, movies: pd.DataFrame, batch_size: int) -> None:
    rows = movies[["movieId", "title", "genres"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (m:Movie {movieId: row.movieId})
    SET m.title = row.title,
        m.genres = row.genres
    """
    chunks = list(batched(rows, batch_size))
    with driver.session(database=database) as session:
        for chunk in tqdm(chunks, desc="Movies"):
            session.run(query, rows=chunk)


def load_users(driver, database: str, user_ids: list[int], batch_size: int) -> None:
    rows = [{"userId": int(uid)} for uid in user_ids]
    query = """
    UNWIND $rows AS row
    MERGE (u:User {userId: row.userId})
    """
    chunks = list(batched(rows, batch_size))
    with driver.session(database=database) as session:
        for chunk in tqdm(chunks, desc="Users"):
            session.run(query, rows=chunk)


def load_ratings(driver, database: str, ratings: pd.DataFrame, batch_size: int) -> None:
    # Keep native Python types for the driver
    frame = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    frame["userId"] = frame["userId"].astype(int)
    frame["movieId"] = frame["movieId"].astype(int)
    frame["rating"] = frame["rating"].astype(float)
    frame["timestamp"] = frame["timestamp"].astype(int)
    rows = frame.to_dict("records")

    query = """
    UNWIND $rows AS row
    MATCH (u:User {userId: row.userId})
    MATCH (m:Movie {movieId: row.movieId})
    MERGE (u)-[r:RATED]->(m)
    SET r.rating = row.rating,
        r.timestamp = row.timestamp
    """
    chunks = list(batched(rows, batch_size))
    with driver.session(database=database) as session:
        for chunk in tqdm(chunks, desc="Ratings"):
            session.run(query, rows=chunk)


def verify(driver, database: str) -> None:
    with driver.session(database=database) as session:
        users = session.run("MATCH (u:User) RETURN count(u) AS n").single()["n"]
        movies = session.run("MATCH (m:Movie) RETURN count(m) AS n").single()["n"]
        rated = session.run("MATCH ()-[r:RATED]->() RETURN count(r) AS n").single()["n"]
    LOGGER.info("Graph counts — User=%s Movie=%s RATED=%s", users, movies, rated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load MovieLens into Neo4j")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/ml-latest-small/ml-latest-small"),
        help="Directory containing movies.csv and ratings.csv",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    setup_logging(args.verbose)

    movies_path = args.data_dir / "movies.csv"
    ratings_path = args.data_dir / "ratings.csv"
    if not movies_path.exists() or not ratings_path.exists():
        LOGGER.error("Missing CSV files under %s", args.data_dir.resolve())
        return 1

    LOGGER.info("Reading %s", movies_path)
    movies = pd.read_csv(movies_path)
    LOGGER.info("Reading %s", ratings_path)
    ratings = pd.read_csv(ratings_path)

    user_ids = sorted(int(x) for x in ratings["userId"].unique().tolist())
    LOGGER.info(
        "Loaded movies=%s ratings=%s users=%s",
        len(movies),
        len(ratings),
        len(user_ids),
    )

    driver, database = connect_neo4j()
    try:
        create_schema(driver, database)
        load_movies(driver, database, movies, args.batch_size)
        load_users(driver, database, user_ids, args.batch_size)
        load_ratings(driver, database, ratings, args.batch_size)
        verify(driver, database)
    finally:
        driver.close()

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
