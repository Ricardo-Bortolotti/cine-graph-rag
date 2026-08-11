"""Factual QA: TF-IDF vector RAG vs GraphRAG over the same Neo4j gold set.

Published snapshot (qwen3:8b, 30 questions): GraphRAG 29/30 factual vs
vector RAG 22/30, both ~93% evidence-in-context. GraphRAG wins on structured
lookup, shared-cast multi-hop questions, and Cypher-level explainability —
not on retrieving more gold strings. Write-up: docs/graph_rag.md.
Numbers: eval/latest_results.json (written only by scripts/run_evaluation.py).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "rag"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from graph_rag import GraphRAGConfig, GraphRAGSystem, OllamaLLMFactory  # noqa: E402
from neo4j_config import connect_neo4j  # noqa: E402

DOC_CYPHER = """
MATCH (m:Movie)
WHERE m.tmdbEnriched = true AND m.overview IS NOT NULL
OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Director)
OPTIONAL MATCH (m)-[:ACTED_BY]->(a:Actor)
OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
WITH m,
     collect(DISTINCT d.name)[0..4] AS directors,
     collect(DISTINCT a.name)[0..8] AS actors,
     collect(DISTINCT g.name) AS genres
RETURN m.movieId AS movieId,
       m.title AS title,
       m.overview AS overview,
       directors,
       actors,
       genres
"""

# ~30 relational questions. Gold labels are filled from Neo4j at eval time.
# Published comparison: qwen3:8b on the full list (docs/graph_rag.md).
FACT_QUERIES = [
    # --- who directed X? ---
    {
        "id": "director_interstellar",
        "kind": "director",
        "title_contains": "Interstellar",
        "question": "Who directed Interstellar?",
    },
    {
        "id": "director_inception",
        "kind": "director",
        "title_contains": "Inception",
        "question": "Who directed Inception?",
    },
    {
        "id": "director_matrix",
        "kind": "director",
        "title_contains": "Matrix, The (1999)",
        "question": "Who directed The Matrix (1999)?",
    },
    {
        "id": "director_pulp",
        "kind": "director",
        "title_contains": "Pulp Fiction (1994)",
        "question": "Who directed Pulp Fiction?",
    },
    {
        "id": "director_shawshank",
        "kind": "director",
        "title_contains": "Shawshank Redemption",
        "question": "Who directed The Shawshank Redemption?",
    },
    {
        "id": "director_godfather",
        "kind": "director",
        "title_contains": "Godfather, The (1972)",
        "question": "Who directed The Godfather (1972)?",
    },
    {
        "id": "director_dark_knight",
        "kind": "director",
        "title_contains": "Dark Knight, The (2008)",
        "question": "Who directed The Dark Knight (2008)?",
    },
    {
        "id": "director_spirited",
        "kind": "director",
        "title_contains": "Spirited Away",
        "question": "Who directed Spirited Away?",
    },
    # --- genres ---
    {
        "id": "genres_toy_story",
        "kind": "genres",
        "title_contains": "Toy Story (1995)",
        "question": "Which genres does Toy Story belong to?",
    },
    {
        "id": "genres_forrest",
        "kind": "genres",
        "title_contains": "Forrest Gump",
        "question": "Which genres does Forrest Gump belong to?",
    },
    {
        "id": "genres_alien",
        "kind": "genres",
        "title_contains": "Alien (1979)",
        "question": "Which genres does Alien (1979) belong to?",
    },
    {
        "id": "genres_get_out",
        "kind": "genres",
        "title_contains": "Get Out (2017)",
        "question": "Which genres does Get Out belong to?",
    },
    {
        "id": "genres_fury_road",
        "kind": "genres",
        "title_contains": "Mad Max: Fury Road",
        "question": "Which genres does Mad Max: Fury Road belong to?",
    },
    {
        "id": "genres_whiplash",
        "kind": "genres",
        "title_contains": "Whiplash (2014)",
        "question": "Which genres does Whiplash belong to?",
    },
    # --- actors in a title ---
    {
        "id": "actors_inception",
        "kind": "actors",
        "title_contains": "Inception",
        "question": "Which actors appear in Inception?",
    },
    {
        "id": "actors_godfather",
        "kind": "actors",
        "title_contains": "Godfather, The (1972)",
        "question": "Which actors appear in The Godfather (1972)?",
    },
    {
        "id": "actors_pulp",
        "kind": "actors",
        "title_contains": "Pulp Fiction (1994)",
        "question": "Which actors appear in Pulp Fiction?",
    },
    {
        "id": "actors_dark_knight",
        "kind": "actors",
        "title_contains": "Dark Knight, The (2008)",
        "question": "Which actors appear in The Dark Knight (2008)?",
    },
    {
        "id": "actors_matrix",
        "kind": "actors",
        "title_contains": "Matrix, The (1999)",
        "question": "Which actors appear in The Matrix (1999)?",
    },
    # --- shared director (multi-hop) ---
    {
        "id": "shared_director_nolan",
        "kind": "shared_director",
        "title_a": "Interstellar",
        "title_b": "Inception",
        "question": "What do Interstellar and Inception have in common through their director?",
    },
    {
        "id": "shared_director_nolan_batman",
        "kind": "shared_director",
        "title_a": "Dark Knight, The (2008)",
        "title_b": "Batman Begins",
        "question": "What do The Dark Knight (2008) and Batman Begins have in common through their director?",
    },
    {
        "id": "shared_director_coppola",
        "kind": "shared_director",
        "title_a": "Godfather, The (1972)",
        "title_b": "Godfather: Part II",
        "question": "What do The Godfather (1972) and The Godfather Part II have in common through their director?",
    },
    {
        "id": "shared_director_tarantino",
        "kind": "shared_director",
        "title_a": "Pulp Fiction (1994)",
        "title_b": "Kill Bill: Vol. 1",
        "question": "What do Pulp Fiction and Kill Bill: Vol. 1 have in common through their director?",
    },
    {
        "id": "shared_director_lasseter",
        "kind": "shared_director",
        "title_a": "Toy Story (1995)",
        "title_b": "Toy Story 2",
        "question": "What do Toy Story (1995) and Toy Story 2 have in common through their director?",
    },
    # --- shared actors (multi-hop) ---
    {
        "id": "shared_actors_matrix",
        "kind": "shared_actors",
        "title_a": "Matrix, The (1999)",
        "title_b": "Matrix Reloaded",
        "question": "Which actors appear in both The Matrix (1999) and The Matrix Reloaded?",
    },
    {
        "id": "shared_actors_lotr",
        "kind": "shared_actors",
        "title_a": "Fellowship of the Ring",
        "title_b": "Two Towers, The",
        "question": "Which actors appear in both The Lord of the Rings: The Fellowship of the Ring and The Two Towers?",
    },
    {
        "id": "shared_actors_godfather",
        "kind": "shared_actors",
        "title_a": "Godfather, The (1972)",
        "title_b": "Godfather: Part II",
        "question": "Which actors appear in both The Godfather (1972) and The Godfather Part II?",
    },
    {
        "id": "shared_actors_toy_story",
        "kind": "shared_actors",
        "title_a": "Toy Story (1995)",
        "title_b": "Toy Story 2",
        "question": "Which actors appear in both Toy Story (1995) and Toy Story 2?",
    },
    {
        "id": "shared_actors_batman",
        "kind": "shared_actors",
        "title_a": "Dark Knight, The (2008)",
        "title_b": "Batman Begins",
        "question": "Which actors appear in both The Dark Knight (2008) and Batman Begins?",
    },
    {
        "id": "shared_actors_oceans",
        "kind": "shared_actors",
        "title_a": "Ocean's Eleven (2001)",
        "title_b": "Ocean's Twelve",
        "question": "Which actors appear in both Ocean's Eleven (2001) and Ocean's Twelve?",
    },
]


@dataclass
class GoldQuestion:
    id: str
    question: str
    expected: list[str]
    notes: str = ""
    kind: str = ""


@dataclass
class QAResult:
    id: str
    method: str
    question: str
    expected: list[str]
    answer: str
    factual_correct: bool
    evidence_in_context: bool
    retrieved_or_cypher: str = ""
    error: str = ""
    kind: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def contains_expected(text: str, expected: list[str], *, min_hits: int | None = None) -> bool:
    blob = _norm(text)
    if not blob or not expected:
        return False
    hits = sum(1 for item in expected if _norm(item) and _norm(item) in blob)
    need = min_hits if min_hits is not None else max(1, min(2, len(expected)))
    if len(expected) == 1:
        need = 1
    return hits >= min(need, len(expected))


def serialize_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _first_movie(session, fragment: str) -> dict[str, Any] | None:
    row = session.run(
        """
        MATCH (m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($q)
        RETURN m.movieId AS movieId, m.title AS title
        ORDER BY size(m.title) ASC
        LIMIT 1
        """,
        q=fragment,
    ).single()
    return dict(row) if row else None


def build_gold_questions() -> list[GoldQuestion]:
    driver, database = connect_neo4j()
    questions: list[GoldQuestion] = []
    with driver.session(database=database) as session:
        for spec in FACT_QUERIES:
            if spec["kind"] == "director":
                movie = _first_movie(session, spec["title_contains"])
                if not movie:
                    continue
                names = [
                    r["name"]
                    for r in session.run(
                        """
                        MATCH (m:Movie {movieId: $id})-[:DIRECTED_BY]->(d:Director)
                        RETURN d.name AS name
                        """,
                        id=movie["movieId"],
                    )
                    if r["name"]
                ]
                if names:
                    questions.append(
                        GoldQuestion(
                            spec["id"],
                            spec["question"],
                            names,
                            notes=f"seed={movie['title']}",
                            kind=spec["kind"],
                        )
                    )
            elif spec["kind"] == "genres":
                movie = _first_movie(session, spec["title_contains"])
                if not movie:
                    continue
                names = [
                    r["name"]
                    for r in session.run(
                        """
                        MATCH (m:Movie {movieId: $id})-[:HAS_GENRE]->(g:Genre)
                        RETURN g.name AS name
                        """,
                        id=movie["movieId"],
                    )
                    if r["name"]
                ]
                if names:
                    questions.append(
                        GoldQuestion(
                            spec["id"],
                            spec["question"],
                            names,
                            notes=f"seed={movie['title']}",
                            kind=spec["kind"],
                        )
                    )
            elif spec["kind"] == "actors":
                movie = _first_movie(session, spec["title_contains"])
                if not movie:
                    continue
                names = [
                    r["name"]
                    for r in session.run(
                        """
                        MATCH (m:Movie {movieId: $id})-[:ACTED_BY]->(a:Actor)
                        RETURN a.name AS name
                        LIMIT 8
                        """,
                        id=movie["movieId"],
                    )
                    if r["name"]
                ]
                if names:
                    questions.append(
                        GoldQuestion(
                            spec["id"],
                            spec["question"],
                            names,
                            notes=f"seed={movie['title']}",
                            kind=spec["kind"],
                        )
                    )
            elif spec["kind"] == "shared_director":
                a = _first_movie(session, spec["title_a"])
                b = _first_movie(session, spec["title_b"])
                if not a or not b:
                    continue
                names = [
                    r["name"]
                    for r in session.run(
                        """
                        MATCH (a:Movie {movieId: $a})-[:DIRECTED_BY]->(d:Director)
                              <-[:DIRECTED_BY]-(b:Movie {movieId: $b})
                        RETURN DISTINCT d.name AS name
                        """,
                        a=a["movieId"],
                        b=b["movieId"],
                    )
                    if r["name"]
                ]
                if names:
                    questions.append(
                        GoldQuestion(
                            spec["id"],
                            spec["question"],
                            names,
                            notes=f"{a['title']} AND {b['title']}",
                            kind=spec["kind"],
                        )
                    )
            elif spec["kind"] == "shared_actors":
                a = _first_movie(session, spec["title_a"])
                b = _first_movie(session, spec["title_b"])
                if not a or not b:
                    continue
                names = [
                    r["name"]
                    for r in session.run(
                        """
                        MATCH (a:Movie {movieId: $a})-[:ACTED_BY]->(p:Actor)
                              <-[:ACTED_BY]-(b:Movie {movieId: $b})
                        RETURN DISTINCT p.name AS name
                        LIMIT 8
                        """,
                        a=a["movieId"],
                        b=b["movieId"],
                    )
                    if r["name"]
                ]
                if names:
                    questions.append(
                        GoldQuestion(
                            spec["id"],
                            spec["question"],
                            names,
                            notes=f"{a['title']} AND {b['title']}",
                            kind=spec["kind"],
                        )
                    )
    driver.close()
    return questions


def load_movie_documents() -> list[dict[str, Any]]:
    driver, database = connect_neo4j()
    docs: list[dict[str, Any]] = []
    with driver.session(database=database) as session:
        for row in session.run(DOC_CYPHER):
            directors = [n for n in (row["directors"] or []) if n]
            actors = [n for n in (row["actors"] or []) if n]
            genres = [n for n in (row["genres"] or []) if n]
            text = (
                f"Title: {row['title']}\n"
                f"Directors: {', '.join(directors) or 'unknown'}\n"
                f"Actors: {', '.join(actors) or 'unknown'}\n"
                f"Genres: {', '.join(genres) or 'unknown'}\n"
                f"Overview: {row['overview']}"
            )
            docs.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "text": text,
                }
            )
    driver.close()
    return docs


class VectorRAG:
    """TF-IDF retrieval over TMDB-enriched movie documents + local Ollama.

    Honest chunk-RAG baseline for docs/graph_rag.md — not dense embeddings.
    Same Ollama model as GraphRAG; only the evidence channel differs.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        top_k: int = 4,
        config: GraphRAGConfig | None = None,
    ) -> None:
        self.documents = documents
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(doc["text"] for doc in documents)
        self.llm = OllamaLLMFactory.create(
            config or GraphRAGConfig.from_env(), purpose="vector-rag"
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        top = scores.argsort()[::-1][: self.top_k]
        hits = []
        for idx in top:
            item = dict(self.documents[idx])
            item["score"] = float(scores[idx])
            hits.append(item)
        return hits

    def ask(self, question: str) -> dict[str, Any]:
        hits = self.retrieve(question)
        context = "\n\n---\n\n".join(hit["text"] for hit in hits)
        prompt = (
            "Answer using only the movie documents below. "
            "If the documents do not contain the fact, say you do not know.\n\n"
            f"Documents:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        message = self.llm.invoke(prompt)
        answer = getattr(message, "content", None) or str(message)
        return {"answer": answer, "context": context, "hits": hits}


def score_output(gold: GoldQuestion, method: str, answer: str, context: Any) -> QAResult:
    context_text = serialize_context(context)
    min_hits = 1 if len(gold.expected) <= 2 else 2
    return QAResult(
        id=gold.id,
        method=method,
        question=gold.question,
        expected=gold.expected,
        answer=answer or "",
        factual_correct=contains_expected(answer or "", gold.expected, min_hits=min_hits),
        evidence_in_context=contains_expected(
            context_text, gold.expected, min_hits=min_hits
        ),
        kind=gold.kind,
    )


def run_qa_eval(
    *,
    top_k: int = 4,
    model: str | None = None,
) -> tuple[list[GoldQuestion], list[QAResult]]:
    """Run vector RAG and GraphRAG on every gold question. Does not write JSON."""
    print("Building gold questions from Neo4j...", flush=True)
    gold = build_gold_questions()
    if not gold:
        raise SystemExit("No gold questions could be built — is the TMDB graph loaded?")

    config = GraphRAGConfig.from_env(verbose=False, model=model)
    print(f"QA model: {config.ollama_model}", flush=True)
    print(f"Loading movie documents for vector RAG ({len(gold)} gold questions)...", flush=True)
    vector = VectorRAG(load_movie_documents(), top_k=top_k, config=config)
    print("Starting GraphRAG (schema refresh)...", flush=True)
    graph = GraphRAGSystem(config)
    results: list[QAResult] = []

    for item in gold:
        print(f"QA {item.id}: vector_rag...", flush=True)
        try:
            vector_out = vector.ask(item.question)
            results.append(
                score_output(item, "vector_rag", vector_out["answer"], vector_out["context"])
            )
        except Exception as exc:
            results.append(
                QAResult(
                    item.id,
                    "vector_rag",
                    item.question,
                    item.expected,
                    "",
                    False,
                    False,
                    error=str(exc),
                    kind=item.kind,
                )
            )
        print(f"QA {item.id}: graph_rag...", flush=True)
        try:
            graph_out = graph.ask(item.question)
            scored = score_output(
                item, "graph_rag", graph_out.get("answer") or "", graph_out.get("context")
            )
            scored.retrieved_or_cypher = graph_out.get("cypher") or ""
            results.append(scored)
        except Exception as exc:
            results.append(
                QAResult(
                    item.id,
                    "graph_rag",
                    item.question,
                    item.expected,
                    "",
                    False,
                    False,
                    error=str(exc),
                    kind=item.kind,
                )
            )
    return gold, results


def results_frame(results: list[QAResult]):
    import pandas as pd

    rows = []
    for item in results:
        payload = asdict(item)
        payload["expected"] = "; ".join(item.expected)
        payload.pop("extras", None)
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_qa(frame) -> Any:
    import pandas as pd

    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("method", as_index=False)
        .agg(
            n=("id", "count"),
            factual_accuracy=("factual_correct", "mean"),
            evidence_in_context=("evidence_in_context", "mean"),
            errors=("error", lambda s: int((s.fillna("") != "").sum())),
        )
        .sort_values("method")
    )


def summarize_qa_by_kind(frame) -> Any:
    import pandas as pd

    if frame.empty or "kind" not in frame.columns:
        return pd.DataFrame()
    return (
        frame.groupby(["kind", "method"], as_index=False)
        .agg(
            n=("id", "count"),
            factual_accuracy=("factual_correct", "mean"),
            evidence_in_context=("evidence_in_context", "mean"),
        )
        .sort_values(["kind", "method"])
    )
