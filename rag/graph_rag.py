"""
GraphRAG conversational interface over the MovieLens + TMDB Neo4j graph.

Stack:
  - LangChain
  - langchain_neo4j.Neo4jGraph
  - langchain_neo4j.GraphCypherQAChain
  - Ollama (free local LLM) via langchain_ollama.ChatOllama

Pipeline:
  1) LLM generates Cypher from the natural-language question + graph schema
  2) Cypher is executed against Neo4j
  3) LLM turns query results into a natural-language answer

This is not the path ranker (`recommender/graph_recommender.py`). Offline, on a
30-question gold set with qwen3:8b, this pipeline beat TF-IDF vector RAG on
factual accuracy (29/30 vs 22/30) at the same retrieval coverage — the win is
structured lookup, multi-hop shared-cast questions, and Cypher-level
explainability. See docs/graph_rag.md and eval/qa_eval.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import ChatOllama

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
from neo4j_config import neo4j_database, neo4j_username  # noqa: E402

LOGGER = logging.getLogger("graph_rag")

# ---------------------------------------------------------------------------
# Prompt engineering — movie-graph Cypher generation
# ---------------------------------------------------------------------------

CYPHER_EXAMPLES = [
    {
        "question": "Recommend movies like Interstellar",
        "query": """
MATCH (seed:Movie)
WHERE toLower(seed.title) CONTAINS toLower('Interstellar')
MATCH (seed)-[:HAS_GENRE|DIRECTED_BY|ACTED_BY|HAS_KEYWORD]->(bridge)
      <-[:HAS_GENRE|DIRECTED_BY|ACTED_BY|HAS_KEYWORD]-(rec:Movie)
WHERE rec <> seed
OPTIONAL MATCH (rec)<-[r:RATED]-()
WITH rec, count(DISTINCT bridge) AS shared, count(r) AS nRatings, avg(r.rating) AS avgRating
RETURN rec.title AS title, shared, nRatings, avgRating
ORDER BY shared DESC, avgRating DESC
LIMIT 10
""".strip(),
    },
    {
        "question": "Why was Oppenheimer recommended after watching Interstellar?",
        "query": """
MATCH (a:Movie), (b:Movie)
WHERE toLower(a.title) CONTAINS toLower('Interstellar')
  AND toLower(b.title) CONTAINS toLower('Oppenheimer')
OPTIONAL MATCH (a)-[:DIRECTED_BY]->(d:Director)<-[:DIRECTED_BY]-(b)
OPTIONAL MATCH (a)-[:ACTED_BY]->(actor:Actor)<-[:ACTED_BY]-(b)
OPTIONAL MATCH (a)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(b)
OPTIONAL MATCH (a)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(b)
RETURN a.title AS seed,
       b.title AS recommended,
       collect(DISTINCT d.name) AS sharedDirectors,
       collect(DISTINCT actor.name) AS sharedActors,
       collect(DISTINCT g.name) AS sharedGenres,
       collect(DISTINCT k.name) AS sharedKeywords
LIMIT 1
""".strip(),
    },
    {
        "question": "Show connections between The Matrix and Cloud Atlas",
        "query": """
MATCH (a:Movie), (b:Movie)
WHERE toLower(a.title) CONTAINS toLower('Matrix')
  AND NOT toLower(a.title) CONTAINS 'reloaded'
  AND NOT toLower(a.title) CONTAINS 'revolutions'
  AND toLower(b.title) CONTAINS toLower('Cloud Atlas')
OPTIONAL MATCH (a)-[:DIRECTED_BY]->(d:Director)<-[:DIRECTED_BY]-(b)
OPTIONAL MATCH (a)-[:ACTED_BY]->(actor:Actor)<-[:ACTED_BY]-(b)
OPTIONAL MATCH (a)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(b)
OPTIONAL MATCH (a)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(b)
RETURN a.title AS movie1,
       b.title AS movie2,
       collect(DISTINCT d.name) AS directors,
       collect(DISTINCT actor.name) AS actors,
       collect(DISTINCT g.name) AS genres,
       collect(DISTINCT k.name) AS keywords
LIMIT 1
""".strip(),
    },
    {
        "question": "Find movies connected through actors or directors to Inception",
        "query": """
MATCH (seed:Movie)
WHERE toLower(seed.title) CONTAINS toLower('Inception')
MATCH (seed)-[:ACTED_BY|DIRECTED_BY]->(person)
MATCH (person)<-[:ACTED_BY|DIRECTED_BY]-(rec:Movie)
WHERE rec <> seed
RETURN labels(person)[0] AS personType,
       person.name AS person,
       collect(DISTINCT rec.title)[0..8] AS relatedMovies
ORDER BY size(relatedMovies) DESC
LIMIT 15
""".strip(),
    },
    {
        "question": "Which genres does Toy Story belong to?",
        "query": """
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
WHERE toLower(m.title) CONTAINS toLower('Toy Story')
  AND m.title CONTAINS '1995'
RETURN m.title AS title, collect(g.name) AS genres
LIMIT 1
""".strip(),
    },
    {
        "question": "Who directed Interstellar?",
        "query": """
MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
WHERE toLower(m.title) CONTAINS toLower('Interstellar')
RETURN m.title AS title, collect(d.name) AS directors
LIMIT 1
""".strip(),
    },
    {
        "question": "Which actors appear in Pulp Fiction?",
        "query": """
MATCH (m:Movie)-[:ACTED_BY]->(a:Actor)
WHERE toLower(m.title) CONTAINS toLower('Pulp Fiction')
RETURN m.title AS title, collect(a.name)[0..10] AS actors
LIMIT 1
""".strip(),
    },
    {
        "question": "Which actors appear in both The Matrix (1999) and The Matrix Reloaded?",
        "query": """
MATCH (a:Movie)-[:ACTED_BY]->(p:Actor)<-[:ACTED_BY]-(b:Movie)
WHERE toLower(a.title) CONTAINS 'matrix'
  AND a.title CONTAINS '1999'
  AND toLower(b.title) CONTAINS 'reloaded'
RETURN p.name AS actor
LIMIT 15
""".strip(),
    },
]

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "query"],
    template="User question: {question}\nCypher:\n{query}",
)

CYPHER_PREFIX = """Task: Generate a Neo4j Cypher read query for a movie knowledge graph.

Graph schema (relevant types):
{schema}

Rules:
- Use ONLY the relationship types and properties present in the schema.
- Prefer case-insensitive title matching with: toLower(m.title) CONTAINS toLower('...')
- Always LIMIT results (default LIMIT 25 unless the user asks otherwise).
- Do NOT use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, or CALL dbms.*.
- Prefer OPTIONAL MATCH when explaining overlaps (shared directors/actors/genres/keywords).
- For recommendations, traverse HAS_GENRE, DIRECTED_BY, ACTED_BY, HAS_KEYWORD and/or co-RATED users.
- Return human-readable columns (title, name, rating aggregates) when possible.
- Do not wrap the Cypher in markdown fences.
- Reply with ONLY the Cypher statement. No preamble such as "Here is the query".

Examples:
"""

CYPHER_SUFFIX = """
User question: {question}
Cypher:
"""

QA_TEMPLATE = """You are a helpful movie graph assistant.
Answer the user using ONLY the Neo4j query results below.
If the results are empty, say you could not find evidence in the graph.
Be concise, cite concrete titles / people / genres from the context, and explain connections when relevant.

Question: {question}

Query results:
{context}

Answer:
"""


def build_cypher_prompt() -> FewShotPromptTemplate:
    return FewShotPromptTemplate(
        examples=CYPHER_EXAMPLES,
        example_prompt=EXAMPLE_PROMPT,
        prefix=CYPHER_PREFIX,
        suffix=CYPHER_SUFFIX,
        input_variables=["schema", "question"],
    )


def build_qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "context"],
        template=QA_TEMPLATE,
    )


_CYPHER_START = re.compile(
    r"(?im)^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|CALL|USE|CYPHER)\b"
)
_CYPHER_CONTINUE = re.compile(
    r"(?i)^\s*(LIMIT|ORDER BY|SKIP|UNION|AND|OR|WHERE|WITH|RETURN|OPTIONAL|"
    r"MATCH|UNWIND|CALL|USE|CYPHER|,|//|/\*|\*|\)|\}|\]|`)"
)


def strip_cypher_preamble(text: str) -> str:
    """Drop 'Here is the Cypher…' prose that small local models prepend."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", text, re.I)
    if fenced:
        text = fenced.group(1).strip()
    match = _CYPHER_START.search(text)
    if match:
        text = text[match.start() :]
    kept: list[str] = []
    seen_return = False
    for line in text.splitlines():
        if re.match(r"(?i)^\s*RETURN\b", line):
            seen_return = True
            kept.append(line)
            continue
        if (
            seen_return
            and line.strip()
            and not _CYPHER_CONTINUE.match(line)
            and not line.startswith(" ")
        ):
            break
        kept.append(line)
    return "\n".join(kept).strip().strip("`")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphRAGConfig:
    """Runtime configuration for GraphRAG (env + CLI overrides)."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    top_k: int = 25
    verbose: bool = False
    enhanced_schema: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        top_k: int | None = None,
        verbose: bool = False,
    ) -> GraphRAGConfig:
        uri = os.getenv("NEO4J_URI")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            raise SystemExit("Missing NEO4J_URI / NEO4J_PASSWORD in environment")

        return cls(
            neo4j_uri=uri,
            neo4j_username=neo4j_username(),
            neo4j_password=password,
            neo4j_database=neo4j_database(),
            ollama_model=model or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            ollama_base_url=base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            top_k=top_k or int(os.getenv("GRAPH_RAG_TOP_K", "25")),
            verbose=verbose,
            enhanced_schema=os.getenv("GRAPH_RAG_ENHANCED_SCHEMA", "true").lower()
            in {"1", "true", "yes"},
        )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class Neo4jGraphFactory:
    """Build a LangChain Neo4jGraph from project credentials."""

    @staticmethod
    def create(config: GraphRAGConfig) -> Neo4jGraph:
        LOGGER.info(
            "Connecting Neo4jGraph database=%s model_host=%s",
            config.neo4j_database,
            config.ollama_base_url,
        )
        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database,
            refresh_schema=True,
            enhanced_schema=config.enhanced_schema,
            sanitize=True,
        )
        # Force schema refresh for Aura after enrichment updates
        graph.refresh_schema()
        LOGGER.info("Graph schema loaded (%s chars)", len(graph.schema or ""))
        return graph


class OllamaLLMFactory:
    """Free local LLM via Ollama."""

    @staticmethod
    def create(config: GraphRAGConfig, *, purpose: str) -> ChatOllama:
        LOGGER.info("Loading Ollama model=%s (%s)", config.ollama_model, purpose)
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            reasoning=False,
        )


class GraphRAGSystem:
    """
    Modular GraphRAG pipeline:
      question → Cypher generation → Neo4j execution → LLM answer

    Better than chunk RAG when the answer is a relationship (shared director /
    cast), not a prose snippet. Needs a model that can write Cypher (qwen3:8b
    in the published eval; llama3.2:3b often fails that step).
    """

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.graph = Neo4jGraphFactory.create(config)
        self.cypher_llm = OllamaLLMFactory.create(config, purpose="cypher")
        self.qa_llm = OllamaLLMFactory.create(config, purpose="qa")
        self.chain = self._build_chain()
        # Small local models often wrap Cypher in prose; LangChain only strips fences.
        self.chain.cypher_generation_chain = (
            self.chain.cypher_generation_chain | RunnableLambda(strip_cypher_preamble)
        )

    def _build_chain(self) -> GraphCypherQAChain:
        return GraphCypherQAChain.from_llm(
            cypher_llm=self.cypher_llm,
            qa_llm=self.qa_llm,
            graph=self.graph,
            verbose=self.config.verbose,
            top_k=self.config.top_k,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            cypher_prompt=build_cypher_prompt(),
            qa_prompt=build_qa_prompt(),
            validate_cypher=True,
        )

    def ask(self, question: str) -> dict[str, Any]:
        """Run full GraphRAG and return answer + intermediate Cypher/context."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        LOGGER.info("Question: %s", question)
        raw = self.chain.invoke({"query": question})

        intermediate = raw.get("intermediate_steps") or []
        cypher = None
        context = None
        if intermediate:
            # LangChain typically returns [{'query': cypher}, {'context': rows}]
            for step in intermediate:
                if isinstance(step, dict):
                    if "query" in step and cypher is None:
                        cypher = step["query"]
                    if "context" in step and context is None:
                        context = step["context"]

        result = {
            "question": question,
            "answer": raw.get("result"),
            "cypher": cypher,
            "context": context,
            "raw": raw,
        }
        return result

    def print_result(self, payload: dict[str, Any]) -> None:
        print("=" * 72)
        print(f"Q: {payload['question']}")
        print("-" * 72)
        if payload.get("cypher"):
            print("Generated Cypher:")
            print(payload["cypher"])
            print("-" * 72)
        if self.config.verbose and payload.get("context") is not None:
            print("Context (top rows):")
            print(payload["context"])
            print("-" * 72)
        print("Answer:")
        print(payload.get("answer") or "(empty)")
        print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SAMPLE_QUESTIONS = [
    "Recommend movies like Interstellar",
    "Why was Oppenheimer recommended after watching Interstellar?",
    "Show connections between The Matrix and Cloud Atlas",
    "Find movies connected through actors or directors to Inception",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphRAG over Neo4j with LangChain + Ollama (free local LLM)"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Single question to ask",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive chat loop",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run sample GraphRAG questions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: OLLAMA_MODEL or qwen3:8b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Max Cypher rows for QA")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_interactive(system: GraphRAGSystem) -> None:
    print("GraphRAG interactive mode. Type 'exit' or 'quit' to stop.")
    print("Examples:")
    for q in SAMPLE_QUESTIONS:
        print(f"  - {q}")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        try:
            payload = system.ask(question)
            system.print_result(payload)
        except Exception:
            LOGGER.exception("Failed to answer question")


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    setup_logging(args.verbose)

    if not args.question and not args.interactive and not args.demo:
        LOGGER.error("Provide --question, --interactive, or --demo")
        return 2

    config = GraphRAGConfig.from_env(
        model=args.model,
        base_url=args.base_url,
        top_k=args.top_k,
        verbose=args.verbose,
    )
    system = GraphRAGSystem(config)

    if args.demo:
        for question in SAMPLE_QUESTIONS:
            try:
                payload = system.ask(question)
                system.print_result(payload)
            except Exception:
                LOGGER.exception("Demo question failed: %s", question)

    if args.question:
        payload = system.ask(args.question)
        system.print_result(payload)

    if args.interactive:
        run_interactive(system)

    return 0


if __name__ == "__main__":
    sys.exit(main())
