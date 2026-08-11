"""Graph-based movie recommender over Neo4j (MovieLens + TMDB enrichment).

Recommendation signals (seed movie → candidates):
  1. Same director   (DIRECTED_BY)
  2. Same actors     (ACTED_BY)
  3. Same genres     (HAS_GENRE)
  4. Graph traversal — shared keywords + co-rating users (HAS_KEYWORD, RATED)

Each candidate is ranked by a weighted score and returned with explanation paths.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver

# Allow `uv run python recommender/graph_recommender.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
from neo4j_config import connect_neo4j  # noqa: E402

LOGGER = logging.getLogger("graph_recommender")

# Relative signal weights (tuned for interpretability, not ML optimality)
WEIGHTS = {
    "director": 5.0,
    "actor": 3.0,
    "keyword": 2.0,
    "genre": 1.0,
    "co_rating": 1.5,
}

RECOMMEND_CYPHER = """
MATCH (seed:Movie)
WHERE seed.movieId = $movieId

CALL (seed) {
  MATCH (seed)-[:DIRECTED_BY]->(bridge:Director)<-[:DIRECTED_BY]-(rec:Movie)
  WHERE rec <> seed
  RETURN rec, 'director' AS signal, bridge.name AS via, $wDirector AS points

  UNION ALL

  MATCH (seed)-[:ACTED_BY]->(bridge:Actor)<-[:ACTED_BY]-(rec:Movie)
  WHERE rec <> seed
  RETURN rec, 'actor' AS signal, bridge.name AS via, $wActor AS points

  UNION ALL

  MATCH (seed)-[:HAS_GENRE]->(bridge:Genre)<-[:HAS_GENRE]-(rec:Movie)
  WHERE rec <> seed
  RETURN rec, 'genre' AS signal, bridge.name AS via, $wGenre AS points

  UNION ALL

  MATCH (seed)-[:HAS_KEYWORD]->(bridge:Keyword)<-[:HAS_KEYWORD]-(rec:Movie)
  WHERE rec <> seed
  RETURN rec, 'keyword' AS signal, bridge.name AS via, $wKeyword AS points

  UNION ALL

  MATCH (seed)<-[r1:RATED]-(u:User)-[r2:RATED]->(rec:Movie)
  WHERE rec <> seed
    AND r1.rating >= $minRating
    AND r2.rating >= $minRating
  WITH rec, count(DISTINCT u) AS sharedUsers
  WHERE sharedUsers >= $minSharedUsers
  RETURN rec,
         'co_rating' AS signal,
         toString(sharedUsers) + ' shared fans' AS via,
         $wCoRating * log10(sharedUsers + 1) AS points
}

WITH rec,
     sum(points) AS rawScore,
     collect({
       signal: signal,
       via: via,
       points: points,
       path: CASE signal
         WHEN 'director' THEN '(Movie)-[:DIRECTED_BY]->(Director)<-[:DIRECTED_BY]-(Movie)'
         WHEN 'actor' THEN '(Movie)-[:ACTED_BY]->(Actor)<-[:ACTED_BY]-(Movie)'
         WHEN 'genre' THEN '(Movie)-[:HAS_GENRE]->(Genre)<-[:HAS_GENRE]-(Movie)'
         WHEN 'keyword' THEN '(Movie)-[:HAS_KEYWORD]->(Keyword)<-[:HAS_KEYWORD]-(Movie)'
         WHEN 'co_rating' THEN '(Movie)<-[:RATED]-(User)-[:RATED]->(Movie)'
         ELSE signal
       END
     }) AS evidence

OPTIONAL MATCH (rec)<-[rr:RATED]-()
WITH rec, rawScore, evidence,
     count(rr) AS nRatings,
     avg(rr.rating) AS avgRating

WITH rec, evidence, nRatings, avgRating,
     rawScore
       + CASE
           WHEN nRatings > 0 AND avgRating IS NOT NULL
           THEN 0.25 * log10(nRatings + 1) * (avgRating / 5.0)
           ELSE 0.0
         END AS score

RETURN
  rec.movieId AS movieId,
  rec.title AS title,
  rec.genres AS movielensGenres,
  rec.tmdbEnriched AS tmdbEnriched,
  score,
  nRatings,
  avgRating,
  evidence
ORDER BY score DESC
LIMIT $limit
"""

RESOLVE_MOVIE_CYPHER = """
MATCH (m:Movie)
WHERE ($movieId IS NOT NULL AND m.movieId = $movieId)
   OR ($title IS NOT NULL AND toLower(m.title) CONTAINS toLower($title))
RETURN m.movieId AS movieId, m.title AS title, m.tmdbEnriched AS tmdbEnriched
ORDER BY
  CASE WHEN $movieId IS NOT NULL AND m.movieId = $movieId THEN 0 ELSE 1 END,
  size(m.title) ASC
LIMIT 10
"""


@dataclass
class Evidence:
    signal: str
    via: str
    points: float
    path: str

    def explanation(self) -> str:
        labels = {
            "director": "Same director",
            "actor": "Shared actor",
            "genre": "Shared genre",
            "keyword": "Shared keyword",
            "co_rating": "Similar audience",
        }
        return f"{labels.get(self.signal, self.signal)}: {self.via} (+{self.points:.2f}) via {self.path}"


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    n_ratings: int
    avg_rating: float | None
    movielens_genres: str | None
    tmdb_enriched: bool | None
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def explanation_paths(self) -> list[str]:
        # Collapse duplicate via-values per signal, keep strongest points
        best: dict[tuple[str, str], Evidence] = {}
        for item in self.evidence:
            key = (item.signal, item.via)
            if key not in best or item.points > best[key].points:
                best[key] = item
        ranked = sorted(best.values(), key=lambda e: e.points, reverse=True)
        return [e.explanation() for e in ranked[:8]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["explanation_paths"] = self.explanation_paths
        return payload


@dataclass
class SeedMovie:
    movie_id: int
    title: str
    tmdb_enriched: bool | None


class GraphRecommender:
    """Produce ranked movie recommendations from Neo4j graph paths."""

    def __init__(
        self,
        driver: Driver,
        database: str,
        weights: dict[str, float] | None = None,
        min_rating: float = 4.0,
        min_shared_users: int = 3,
    ) -> None:
        self.driver = driver
        self.database = database
        self.weights = {**WEIGHTS, **(weights or {})}
        self.min_rating = min_rating
        self.min_shared_users = min_shared_users

    def resolve_movie(
        self,
        movie_id: int | None = None,
        title: str | None = None,
    ) -> list[SeedMovie]:
        if movie_id is None and not title:
            raise ValueError("Provide movie_id and/or title")

        with self.driver.session(database=self.database) as session:
            rows = session.run(
                RESOLVE_MOVIE_CYPHER,
                movieId=movie_id,
                title=title,
            )
            return [
                SeedMovie(
                    movie_id=int(r["movieId"]),
                    title=r["title"],
                    tmdb_enriched=r["tmdbEnriched"],
                )
                for r in rows
            ]

    def recommend(
        self,
        movie_id: int,
        limit: int = 10,
    ) -> list[Recommendation]:
        params = {
            "movieId": int(movie_id),
            "limit": int(limit),
            "minRating": float(self.min_rating),
            "minSharedUsers": int(self.min_shared_users),
            "wDirector": self.weights["director"],
            "wActor": self.weights["actor"],
            "wGenre": self.weights["genre"],
            "wKeyword": self.weights["keyword"],
            "wCoRating": self.weights["co_rating"],
        }

        with self.driver.session(database=self.database) as session:
            rows = list(session.run(RECOMMEND_CYPHER, **params))

        recommendations: list[Recommendation] = []
        for row in rows:
            evidence = [
                Evidence(
                    signal=e["signal"],
                    via=str(e["via"]),
                    points=float(e["points"]),
                    path=e["path"],
                )
                for e in (row["evidence"] or [])
                if e and e.get("signal")
            ]
            recommendations.append(
                Recommendation(
                    movie_id=int(row["movieId"]),
                    title=row["title"],
                    score=float(row["score"]),
                    n_ratings=int(row["nRatings"] or 0),
                    avg_rating=(
                        float(row["avgRating"]) if row["avgRating"] is not None else None
                    ),
                    movielens_genres=row["movielensGenres"],
                    tmdb_enriched=row["tmdbEnriched"],
                    evidence=evidence,
                )
            )
        return recommendations

    def recommend_for_query(
        self,
        movie_id: int | None = None,
        title: str | None = None,
        limit: int = 10,
    ) -> tuple[SeedMovie, list[Recommendation]]:
        matches = self.resolve_movie(movie_id=movie_id, title=title)
        if not matches:
            raise LookupError(
                f"No movie found for movie_id={movie_id!r} title={title!r}"
            )
        seed = matches[0]
        if movie_id is None and len(matches) > 1:
            LOGGER.info(
                "Multiple title matches; using movieId=%s (%s). Alternatives: %s",
                seed.movie_id,
                seed.title,
                ", ".join(f"{m.movie_id}:{m.title}" for m in matches[1:5]),
            )
        recs = self.recommend(seed.movie_id, limit=limit)
        return seed, recs


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_recommendations(seed: SeedMovie, recs: list[Recommendation]) -> None:
    print("=" * 72)
    print(f"Seed: {seed.title} (movieId={seed.movie_id})")
    print(f"TMDB enriched: {seed.tmdb_enriched}")
    print("=" * 72)
    if not recs:
        print("No recommendations found. Enrich more movies with TMDB metadata.")
        return

    for rank, rec in enumerate(recs, start=1):
        avg = f"{rec.avg_rating:.2f}" if rec.avg_rating is not None else "n/a"
        print(f"\n#{rank}  {rec.title}  (movieId={rec.movie_id})")
        print(
            f"    score={rec.score:.3f}  |  ratings={rec.n_ratings}  |  avg={avg}"
        )
        print("    explanations:")
        for line in rec.explanation_paths:
            print(f"      - {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-based movie recommendations from Neo4j"
    )
    parser.add_argument("--movie-id", type=int, default=None, help="MovieLens movieId")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Case-insensitive title substring (e.g. 'Matrix')",
    )
    parser.add_argument("--limit", type=int, default=10, help="Top-N recommendations")
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating for co-rating traversal",
    )
    parser.add_argument(
        "--min-shared-users",
        type=int,
        default=3,
        help="Minimum shared fans for co-rating signal",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    setup_logging(args.verbose)

    if args.movie_id is None and not args.title:
        LOGGER.error("Provide --movie-id and/or --title")
        return 2

    driver, database = connect_neo4j()
    try:
        recommender = GraphRecommender(
            driver,
            database,
            min_rating=args.min_rating,
            min_shared_users=args.min_shared_users,
        )
        seed, recs = recommender.recommend_for_query(
            movie_id=args.movie_id,
            title=args.title,
            limit=args.limit,
        )
        print_recommendations(seed, recs)
    except LookupError as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
