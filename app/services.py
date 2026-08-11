"""Shared services for the CineGraphRAG Streamlit product."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "recommender"))
sys.path.insert(0, str(ROOT / "rag"))

load_dotenv(ROOT / ".env")

from neo4j_config import connect_neo4j  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_driver():
    return connect_neo4j()


def run_cypher(query: str, **params: Any) -> list[dict[str, Any]]:
    driver, database = get_driver()
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, **params)]


def search_movies(query: str, limit: int = 25) -> list[dict[str, Any]]:
    if not query or len(query.strip()) < 2:
        return []
    return run_cypher(
        """
        MATCH (m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($q)
        RETURN m.movieId AS movieId,
               m.title AS title,
               coalesce(m.tmdbEnriched, false) AS enriched
        ORDER BY size(m.title) ASC
        LIMIT $limit
        """,
        q=query.strip(),
        limit=limit,
    )


def list_users(limit: int = 200) -> list[int]:
    rows = run_cypher(
        """
        MATCH (u:User)-[r:RATED]->()
        WITH u, count(r) AS n
        RETURN u.userId AS userId
        ORDER BY n DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    return [int(r["userId"]) for r in rows]


def user_profile(user_id: int, limit: int = 12) -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
        RETURN m.movieId AS movieId, m.title AS title, r.rating AS rating
        ORDER BY r.rating DESC, m.title ASC
        LIMIT $limit
        """,
        userId=int(user_id),
        limit=limit,
    )


def graph_kpis() -> dict[str, Any]:
    movies = run_cypher(
        """
        MATCH (m:Movie)
        RETURN count(m) AS movies,
               sum(CASE WHEN m.tmdbEnriched THEN 1 ELSE 0 END) AS enriched
        """
    )[0]
    users = run_cypher("MATCH (u:User) RETURN count(u) AS users")[0]
    ratings = run_cypher(
        "MATCH ()-[r:RATED]->() RETURN count(r) AS ratings, avg(r.rating) AS avgRating"
    )[0]
    people = run_cypher(
        """
        OPTIONAL MATCH (d:Director)
        WITH count(d) AS directors
        OPTIONAL MATCH (a:Actor)
        WITH directors, count(a) AS actors
        OPTIONAL MATCH (g:Genre)
        WITH directors, actors, count(g) AS genres
        OPTIONAL MATCH (k:Keyword)
        RETURN directors, actors, genres, count(k) AS keywords
        """
    )[0]
    return {**movies, **users, **ratings, **people}


def analytics_rating_distribution() -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH ()-[r:RATED]->()
        RETURN r.rating AS rating, count(*) AS n
        ORDER BY rating
        """
    )


def analytics_top_genres(limit: int = 15) -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH (:User)-[r:RATED]->(m:Movie)-[:HAS_GENRE]->(g:Genre)
        RETURN g.name AS genre, count(r) AS nRatings, avg(r.rating) AS avgRating
        ORDER BY nRatings DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def analytics_top_movies(limit: int = 15) -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH ()-[r:RATED]->(m:Movie)
        WITH m, count(r) AS nRatings, avg(r.rating) AS avgRating
        WHERE nRatings >= 30
        RETURN m.title AS title, nRatings, avgRating
        ORDER BY avgRating DESC, nRatings DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def analytics_degree_hubs(limit: int = 12) -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH (p)<-[:DIRECTED_BY|ACTED_BY]-(:Movie)
        WHERE p:Director OR p:Actor
        RETURN labels(p)[0] AS type, p.name AS name, count(*) AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def analytics_relationship_counts() -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS rel, count(*) AS n
        ORDER BY n DESC
        """
    )


@st.cache_resource(show_spinner="Loading GraphRAG (Ollama + Neo4j schema)...")
def get_graph_rag(verbose: bool = False):
    from graph_rag import GraphRAGConfig, GraphRAGSystem

    config = GraphRAGConfig.from_env(verbose=verbose)
    return GraphRAGSystem(config)
