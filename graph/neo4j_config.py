"""Shared Neo4j credential helpers (Aura or local Docker)."""

from __future__ import annotations

import os

from neo4j import Driver, GraphDatabase


def env_or_fail(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise SystemExit(f"Missing required environment variable: {key}")
    return value


def neo4j_username() -> str:
    """Aura credentials files use NEO4J_USERNAME; scripts also accept NEO4J_USER."""
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    if not user:
        raise SystemExit(
            "Missing Neo4j username. Set NEO4J_USERNAME (Aura) or NEO4J_USER."
        )
    return user


def neo4j_database() -> str:
    """Default database name; Aura Free usually uses 'neo4j'."""
    return os.getenv("NEO4J_DATABASE") or "neo4j"


def connect_neo4j() -> tuple[Driver, str]:
    """Return a connected driver and the target database name."""
    uri = env_or_fail("NEO4J_URI")
    password = env_or_fail("NEO4J_PASSWORD")
    user = neo4j_username()
    database = neo4j_database()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver, database
