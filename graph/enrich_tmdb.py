"""Enrich Movie nodes with TMDB metadata and write graph edges.

(Movie)-[:HAS_GENRE]->(Genre)
(Movie)-[:DIRECTED_BY]->(Director)
(Movie)-[:ACTED_BY]->(Actor)
(Movie)-[:HAS_KEYWORD]->(Keyword)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from neo4j_config import connect_neo4j, env_or_fail

LOGGER = logging.getLogger("enrich_tmdb")

CONSTRAINTS = [
    "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT director_id IF NOT EXISTS FOR (d:Director) REQUIRE d.tmdbId IS UNIQUE",
    "CREATE CONSTRAINT actor_id IF NOT EXISTS FOR (a:Actor) REQUIRE a.tmdbId IS UNIQUE",
    "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.tmdbId IS UNIQUE",
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
]


class RateLimiter:
    """Simple client-side rate limiter."""

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / max(requests_per_second, 0.1)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class TMDBClient:
    def __init__(self, api_key: str, cfg: dict[str, Any]) -> None:
        self.api_key = api_key
        self.base_url = cfg["base_url"].rstrip("/")
        self.language = cfg.get("language", "en-US")
        self.timeout = float(cfg.get("timeout_seconds", 30))
        self.top_actors = int(cfg.get("top_actors", 5))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.limiter = RateLimiter(float(cfg.get("requests_per_second", 3)))
        self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        @retry(
            retry=retry_if_exception_type((requests.RequestException,)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _do_get() -> dict[str, Any]:
            self.limiter.wait()
            url = f"{self.base_url}{path}"
            query = {"api_key": self.api_key, **(params or {})}
            resp = self.session.get(url, params=query, timeout=self.timeout)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "5"))
                LOGGER.warning("TMDB rate limited; sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                raise requests.RequestException("429 Too Many Requests")

            if resp.status_code == 404:
                return {}

            resp.raise_for_status()
            return resp.json()

        return _do_get()

    def fetch_movie_bundle(self, tmdb_id: int) -> dict[str, Any] | None:
        details = self._get(
            f"/movie/{tmdb_id}",
            {"language": self.language, "append_to_response": "credits,keywords"},
        )
        if not details:
            return None

        genres = [
            {"name": g["name"]}
            for g in details.get("genres", [])
            if g.get("name")
        ]

        directors = [
            {"tmdbId": int(p["id"]), "name": p["name"]}
            for p in details.get("credits", {}).get("crew", [])
            if p.get("job") == "Director" and p.get("id") and p.get("name")
        ]

        cast = details.get("credits", {}).get("cast", [])
        actors = [
            {
                "tmdbId": int(p["id"]),
                "name": p["name"],
                "character": p.get("character"),
                "order": int(p.get("order", 999)),
            }
            for p in cast[: self.top_actors]
            if p.get("id") and p.get("name")
        ]

        kw_block = details.get("keywords", {})
        keywords_raw = kw_block.get("keywords") or kw_block.get("results") or []
        keywords = [
            {"tmdbId": int(k["id"]), "name": k["name"]}
            for k in keywords_raw
            if k.get("id") and k.get("name")
        ]

        return {
            "tmdbId": int(tmdb_id),
            "title": details.get("title"),
            "overview": details.get("overview"),
            "releaseDate": details.get("release_date"),
            "genres": genres,
            "directors": directors,
            "actors": actors,
            "keywords": keywords,
        }


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_schema(driver, database: str) -> None:
    with driver.session(database=database) as session:
        for stmt in CONSTRAINTS:
            LOGGER.info("Schema: %s", stmt)
            session.run(stmt)


def already_enriched_ids(driver, database: str) -> set[int]:
    query = """
    MATCH (m:Movie)
    WHERE m.tmdbEnriched = true
    RETURN m.movieId AS movieId
    """
    with driver.session(database=database) as session:
        return {int(r["movieId"]) for r in session.run(query)}


def write_enrichment(
    driver, database: str, movie_id: int, payload: dict[str, Any]
) -> None:
    query = """
    MATCH (m:Movie {movieId: $movieId})
    SET m.tmdbId = $tmdbId,
        m.tmdbTitle = $title,
        m.overview = $overview,
        m.releaseDate = $releaseDate,
        m.tmdbEnriched = true,
        m.tmdbEnrichedAt = datetime()

    WITH m
    FOREACH (g IN $genres |
      MERGE (genre:Genre {name: g.name})
      MERGE (m)-[:HAS_GENRE]->(genre)
    )

    WITH m
    FOREACH (d IN $directors |
      MERGE (dir:Director {tmdbId: d.tmdbId})
      SET dir.name = d.name
      MERGE (m)-[:DIRECTED_BY]->(dir)
    )

    WITH m
    FOREACH (a IN $actors |
      MERGE (actor:Actor {tmdbId: a.tmdbId})
      SET actor.name = a.name
      MERGE (m)-[r:ACTED_BY]->(actor)
      SET r.character = a.character, r.order = a.order
    )

    WITH m
    FOREACH (k IN $keywords |
      MERGE (kw:Keyword {tmdbId: k.tmdbId})
      SET kw.name = k.name
      MERGE (m)-[:HAS_KEYWORD]->(kw)
    )
    """
    with driver.session(database=database) as session:
        session.run(
            query,
            movieId=int(movie_id),
            tmdbId=payload["tmdbId"],
            title=payload.get("title"),
            overview=payload.get("overview"),
            releaseDate=payload.get("releaseDate"),
            genres=payload.get("genres", []),
            directors=payload.get("directors", []),
            actors=payload.get("actors", []),
            keywords=payload.get("keywords", []),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j movies with TMDB metadata"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("graph/tmdb_config.yaml"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max movies to enrich in this run",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    setup_logging(args.verbose)

    if not args.config.exists():
        LOGGER.error("Config not found: %s", args.config.resolve())
        return 1

    cfg = load_config(args.config)
    api_key = env_or_fail("TMDB_API_KEY")

    links_path = Path(cfg["data"]["links_csv"])
    if not links_path.exists():
        LOGGER.error("links.csv not found: %s", links_path.resolve())
        return 1

    links = pd.read_csv(links_path)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)

    client = TMDBClient(api_key, cfg["tmdb"])
    driver, database = connect_neo4j()

    try:
        create_schema(driver, database)

        skip = bool(cfg["data"].get("skip_enriched", True))
        done = already_enriched_ids(driver, database) if skip else set()
        work = links.loc[~links["movieId"].isin(done)].copy()
        if args.limit is not None:
            work = work.head(args.limit)

        LOGGER.info(
            "Movies to enrich: %s (already enriched skipped=%s)",
            len(work),
            len(done),
        )

        ok = 0
        fail = 0
        for _, row in tqdm(work.iterrows(), total=len(work), desc="TMDB enrich"):
            movie_id = int(row["movieId"])
            tmdb_id = int(row["tmdbId"])
            try:
                payload = client.fetch_movie_bundle(tmdb_id)
                if not payload:
                    LOGGER.warning(
                        "TMDB 404 / empty response movieId=%s tmdbId=%s",
                        movie_id,
                        tmdb_id,
                    )
                    fail += 1
                    continue
                write_enrichment(driver, database, movie_id, payload)
                ok += 1
            except Exception:
                LOGGER.exception("Failed movieId=%s tmdbId=%s", movie_id, tmdb_id)
                fail += 1

        LOGGER.info("Enrichment finished — ok=%s fail=%s", ok, fail)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
