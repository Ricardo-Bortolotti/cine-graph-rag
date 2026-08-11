"""Explanation engine for graph recommendations.

Given Movie A and Movie B, find the shortest connecting path in Neo4j and return:
  1) The Cypher query used
  2) A human-readable explanation
  3) A JSON payload suitable for path visualization

Example:
  Interstellar -> Christopher Nolan -> Inception
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver
from neo4j.graph import Node
from neo4j.graph import Path as Neo4jPath
from neo4j.graph import Relationship

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
from neo4j_config import connect_neo4j  # noqa: E402

LOGGER = logging.getLogger("explanation_engine")


def _json_safe(value: Any) -> Any:
    """Convert Neo4j temporal / nested values into JSON-serializable forms."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    # neo4j.time.DateTime / Date / Time, etc.
    if hasattr(value, "iso_format"):
        try:
            return value.iso_format()
        except Exception:
            return str(value)
    return str(value)

# Prefer semantic / content edges for explanations; fall back to RATED.
CONTENT_RELS = ("DIRECTED_BY", "ACTED_BY", "HAS_GENRE", "HAS_KEYWORD")
FALLBACK_RELS = CONTENT_RELS + ("RATED",)

RESOLVE_MOVIE_CYPHER = """
MATCH (m:Movie)
WHERE ($movieId IS NOT NULL AND m.movieId = $movieId)
   OR ($title IS NOT NULL AND toLower(m.title) CONTAINS toLower($title))
RETURN m.movieId AS movieId,
       m.title AS title,
       m.tmdbEnriched AS tmdbEnriched
ORDER BY
  CASE WHEN $movieId IS NOT NULL AND m.movieId = $movieId THEN 0 ELSE 1 END,
  size(m.title) ASC
LIMIT 10
"""

# Built dynamically so relationship types stay explicit and auditable.
SHORTEST_PATH_TEMPLATE = """
MATCH (a:Movie {{movieId: $movieIdA}})
MATCH (b:Movie {{movieId: $movieIdB}})
MATCH path = allShortestPaths(
  (a)-[:{rels}*..{max_depth}]-(b)
)
WITH path,
     length(path) AS hops,
     reduce(
       score = 0,
       r IN relationships(path) |
       score + CASE type(r)
         WHEN 'DIRECTED_BY' THEN 100
         WHEN 'ACTED_BY' THEN 80
         WHEN 'HAS_KEYWORD' THEN 40
         WHEN 'HAS_GENRE' THEN 10
         WHEN 'RATED' THEN 5
         ELSE 1
       END
     ) AS quality
RETURN path,
       hops,
       quality
ORDER BY hops ASC, quality DESC
LIMIT $limit
"""


@dataclass
class MovieRef:
    movie_id: int
    title: str
    tmdb_enriched: bool | None = None


@dataclass
class PathNode:
    element_id: str
    labels: list[str]
    key: str
    display: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathEdge:
    element_id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    movie_a: MovieRef
    movie_b: MovieRef
    cypher: str
    human_explanation: str
    path_text: str
    hops: int
    relationship_mode: str
    nodes: list[PathNode]
    edges: list[PathEdge]
    found: bool = True

    def to_visualization_json(self) -> dict[str, Any]:
        """JSON representation for front-end / notebook visualization."""
        return {
            "found": self.found,
            "relationship_mode": self.relationship_mode,
            "hops": self.hops,
            "path_text": self.path_text,
            "human_explanation": self.human_explanation,
            "movie_a": asdict(self.movie_a),
            "movie_b": asdict(self.movie_b),
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "cypher": self.cypher,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_visualization_json()


def _node_display(node: Node) -> tuple[str, str]:
    labels = list(node.labels)
    props = dict(node)
    if "Movie" in labels:
        return "Movie", str(props.get("title") or props.get("movieId") or "Movie")
    if "Director" in labels:
        return "Director", str(props.get("name") or "Director")
    if "Actor" in labels:
        return "Actor", str(props.get("name") or "Actor")
    if "Genre" in labels:
        return "Genre", str(props.get("name") or "Genre")
    if "Keyword" in labels:
        return "Keyword", str(props.get("name") or "Keyword")
    if "User" in labels:
        return "User", f"User {props.get('userId', '?')}"
    label = labels[0] if labels else "Node"
    return label, str(props.get("name") or props.get("title") or label)


def _edge_phrase(rel_type: str, left: str, right: str, mid_label: str | None = None) -> str:
    mapping = {
        "DIRECTED_BY": "share director",
        "ACTED_BY": "share actor",
        "HAS_GENRE": "share genre",
        "HAS_KEYWORD": "share keyword",
        "RATED": "are connected through shared fans",
    }
    if rel_type in mapping and mid_label:
        if rel_type == "RATED":
            return f"{left} and {right} {mapping[rel_type]} ({mid_label})"
        return f"{left} and {right} {mapping[rel_type]} '{mid_label}'"
    return f"{left} -[{rel_type}]- {right}"


def path_to_text(nodes: list[PathNode]) -> str:
    return " -> ".join(n.display for n in nodes)


def path_to_human_explanation(nodes: list[PathNode], edges: list[PathEdge]) -> str:
    if len(nodes) < 2:
        return "No connecting path found."

    if len(nodes) == 3 and len(edges) == 2:
        # Classic A → bridge → B pattern
        left, bridge, right = nodes
        rel = edges[0].type
        phrases = {
            "DIRECTED_BY": f"{left.display} and {right.display} were both directed by {bridge.display}.",
            "ACTED_BY": f"{left.display} and {right.display} both feature actor {bridge.display}.",
            "HAS_GENRE": f"{left.display} and {right.display} share the genre {bridge.display}.",
            "HAS_KEYWORD": f"{left.display} and {right.display} share the keyword '{bridge.display}'.",
            "RATED": f"{left.display} and {right.display} are linked through {bridge.display} (shared audience).",
        }
        return phrases.get(
            rel,
            f"{left.display} connects to {right.display} via {bridge.display} ({rel}).",
        )

    parts: list[str] = [f"Shortest path ({len(edges)} hops): {path_to_text(nodes)}."]
    details: list[str] = []
    for i, edge in enumerate(edges):
        left = nodes[i]
        right = nodes[i + 1]
        details.append(f"{left.display} -[{edge.type}]- {right.display}")
    parts.append("Steps: " + "; ".join(details) + ".")
    return " ".join(parts)


def neo4j_path_to_structures(path: Neo4jPath) -> tuple[list[PathNode], list[PathEdge]]:
    nodes: list[PathNode] = []
    for node in path.nodes:
        key, display = _node_display(node)
        nodes.append(
            PathNode(
                element_id=node.element_id,
                labels=list(node.labels),
                key=key,
                display=display,
                properties=_json_safe(
                    {k: v for k, v in dict(node).items() if v is not None}
                ),
            )
        )

    edges: list[PathEdge] = []
    for rel in path.relationships:
        assert isinstance(rel, Relationship)
        edges.append(
            PathEdge(
                element_id=rel.element_id,
                type=rel.type,
                source=rel.start_node.element_id if rel.start_node else "",
                target=rel.end_node.element_id if rel.end_node else "",
                properties=_json_safe(
                    {k: v for k, v in dict(rel).items() if v is not None}
                ),
            )
        )
    return nodes, edges


class ExplanationEngine:
    """Find and explain shortest graph paths between two movies."""

    def __init__(
        self,
        driver: Driver,
        database: str,
        max_depth: int = 4,
        prefer_content: bool = True,
    ) -> None:
        self.driver = driver
        self.database = database
        self.max_depth = max_depth
        self.prefer_content = prefer_content

    def resolve_movie(
        self,
        movie_id: int | None = None,
        title: str | None = None,
    ) -> list[MovieRef]:
        if movie_id is None and not title:
            raise ValueError("Provide movie_id and/or title")

        with self.driver.session(database=self.database) as session:
            rows = session.run(RESOLVE_MOVIE_CYPHER, movieId=movie_id, title=title)
            return [
                MovieRef(
                    movie_id=int(r["movieId"]),
                    title=r["title"],
                    tmdb_enriched=r["tmdbEnriched"],
                )
                for r in rows
            ]

    def _pick_movie(
        self,
        movie_id: int | None,
        title: str | None,
        label: str,
    ) -> MovieRef:
        matches = self.resolve_movie(movie_id=movie_id, title=title)
        if not matches:
            raise LookupError(f"No movie found for {label}: id={movie_id!r} title={title!r}")
        if movie_id is None and len(matches) > 1:
            LOGGER.info(
                "%s matched multiple titles; using movieId=%s (%s). Alternatives: %s",
                label,
                matches[0].movie_id,
                matches[0].title,
                ", ".join(f"{m.movie_id}:{m.title}" for m in matches[1:5]),
            )
        return matches[0]

    def _build_cypher(self, rels: tuple[str, ...]) -> str:
        rel_clause = "|".join(rels)
        return SHORTEST_PATH_TEMPLATE.format(
            rels=rel_clause,
            max_depth=self.max_depth,
        ).strip()

    def _run_shortest_path(
        self,
        movie_id_a: int,
        movie_id_b: int,
        rels: tuple[str, ...],
        limit: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        cypher = self._build_cypher(rels)
        with self.driver.session(database=self.database) as session:
            rows = list(
                session.run(
                    cypher,
                    movieIdA=int(movie_id_a),
                    movieIdB=int(movie_id_b),
                    limit=int(limit),
                )
            )
        return cypher, rows

    def explain(
        self,
        movie_id_a: int | None = None,
        title_a: str | None = None,
        movie_id_b: int | None = None,
        title_b: str | None = None,
        limit_paths: int = 3,
    ) -> ExplanationResult:
        movie_a = self._pick_movie(movie_id_a, title_a, "Movie A")
        movie_b = self._pick_movie(movie_id_b, title_b, "Movie B")

        if movie_a.movie_id == movie_b.movie_id:
            raise ValueError("Movie A and Movie B must be different titles")

        modes: list[tuple[str, tuple[str, ...]]] = []
        if self.prefer_content:
            modes.append(("content", CONTENT_RELS))
        modes.append(("content_and_ratings", FALLBACK_RELS))

        last_cypher = ""
        for mode_name, rels in modes:
            cypher, rows = self._run_shortest_path(
                movie_a.movie_id,
                movie_b.movie_id,
                rels,
                limit=limit_paths,
            )
            last_cypher = cypher
            if not rows:
                LOGGER.info(
                    "No path in mode=%s between %s and %s",
                    mode_name,
                    movie_a.title,
                    movie_b.title,
                )
                continue

            # Take the first (shortest) path
            record = rows[0]
            path: Neo4jPath = record["path"]
            hops = int(record["hops"])
            nodes, edges = neo4j_path_to_structures(path)
            path_text = path_to_text(nodes)
            human = path_to_human_explanation(nodes, edges)

            return ExplanationResult(
                movie_a=movie_a,
                movie_b=movie_b,
                cypher=cypher,
                human_explanation=human,
                path_text=path_text,
                hops=hops,
                relationship_mode=mode_name,
                nodes=nodes,
                edges=edges,
                found=True,
            )

        # Nothing found
        empty = ExplanationResult(
            movie_a=movie_a,
            movie_b=movie_b,
            cypher=last_cypher,
            human_explanation=(
                f"No path found between '{movie_a.title}' and '{movie_b.title}' "
                f"within depth {self.max_depth}."
            ),
            path_text="",
            hops=-1,
            relationship_mode="none",
            nodes=[],
            edges=[],
            found=False,
        )
        return empty


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_explanation(result: ExplanationResult, *, show_json: bool = True) -> None:
    print("=" * 72)
    print(f"Movie A: {result.movie_a.title} (movieId={result.movie_a.movie_id})")
    print(f"Movie B: {result.movie_b.title} (movieId={result.movie_b.movie_id})")
    print(f"Mode: {result.relationship_mode} | hops={result.hops} | found={result.found}")
    print("-" * 72)
    print("1) Cypher query:")
    print(result.cypher)
    print("-" * 72)
    print("2) Human-readable explanation:")
    print(result.human_explanation)
    if result.path_text:
        print(f"   Path: {result.path_text}")
    print("-" * 72)
    if show_json:
        print("3) JSON for visualization:")
        print(json.dumps(result.to_visualization_json(), indent=2, ensure_ascii=False))
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain the shortest Neo4j path between two movies"
    )
    parser.add_argument("--movie-id-a", type=int, default=None)
    parser.add_argument("--title-a", type=str, default=None)
    parser.add_argument("--movie-id-b", type=int, default=None)
    parser.add_argument("--title-b", type=str, default=None)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum path length for shortestPath",
    )
    parser.add_argument(
        "--include-ratings-first",
        action="store_true",
        help="Do not prefer content-only paths before falling back to RATED",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the visualization JSON",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    setup_logging(args.verbose)

    if (args.movie_id_a is None and not args.title_a) or (
        args.movie_id_b is None and not args.title_b
    ):
        LOGGER.error("Provide Movie A and Movie B via --movie-id-* and/or --title-*")
        return 2

    driver, database = connect_neo4j()
    result: ExplanationResult | None = None
    try:
        engine = ExplanationEngine(
            driver,
            database,
            max_depth=args.max_depth,
            prefer_content=not args.include_ratings_first,
        )
        result = engine.explain(
            movie_id_a=args.movie_id_a,
            title_a=args.title_a,
            movie_id_b=args.movie_id_b,
            title_b=args.title_b,
        )
        if args.json_only:
            print(json.dumps(result.to_visualization_json(), indent=2, ensure_ascii=False))
        else:
            print_explanation(result)
    except (LookupError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        driver.close()

    if result is None:
        return 1
    return 0 if result.found else 3


if __name__ == "__main__":
    sys.exit(main())
