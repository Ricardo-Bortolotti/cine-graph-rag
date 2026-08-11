# GraphRAG vs vector RAG

Natural-language QA over the MovieLens + TMDB Neo4j graph.

This **is** GraphRAG (`rag/graph_rag.py`). It is not the path ranker — that lives in [`graph_recommender.md`](graph_recommender.md) and has no LLM.

Published numbers: [`eval/latest_results.json`](../eval/latest_results.json), produced by [`eval/qa_eval.py`](../eval/qa_eval.py). Walkthrough: [`notebooks/05_evaluation.ipynb`](../notebooks/05_evaluation.ipynb).

The first QA snapshot was only 7 questions on `llama3.2:3b` and looked like a toss-up (vector even won that tiny run). After the ranking eval moved to **200 users / 5 seeds**, the same notebook's gold set was expanded to **30 questions** on `qwen3:8b`. That larger QA sample is what this page reports.

---

## What each method does

| | Vector RAG | GraphRAG |
|---|---|---|
| Retrieval | TF-IDF over title + overview + up to 4 directors, 8 actors, genres | LLM writes Cypher from the live schema + few-shots |
| Evidence | Top-4 nearest documents | Neo4j rows |
| Answer | Ollama, documents only | Ollama verbalizes those rows |
| Model in the published run | `qwen3:8b` | `qwen3:8b` |

Same model, same gold set. The difference is **how the fact reaches the prompt**.

```text
Vector RAG
  question → TF-IDF → 4 documents → LLM

GraphRAG
  question → schema-aware Cypher → Neo4j (read-only) → LLM
```

---

## Headline (`qwen3:8b`, 30 questions)

| Method | n | Factual accuracy | Evidence in context | Errors |
|---|---:|---:|---:|---:|
| **graph_rag** | 30 | **0.967 (29/30)** | 0.933 | 0 |
| vector_rag | 30 | 0.733 (22/30) | 0.933 | 0 |

Retrieval coverage is **the same** (~93%). GraphRAG wins because it **uses** the evidence, and because that evidence is an explicit path, not a nearest-document snippet.

Metrics:

- `factual_correct` — gold name(s) appear in the answer
- `evidence_in_context` — the same names appear in the TF-IDF documents or the Cypher result

Gold labels are built from the live graph (directors, genres, cast, shared director, shared actors) so they stay consistent with the loaded catalog.

---

## Where GraphRAG is better

### 1. Factual accuracy (+23 points)

29/30 vs 22/30. Vector RAG missed eight questions. GraphRAG missed one (`shared_actors_godfather` — bad Cypher, empty evidence).

### 2. Explainability (the main product gap)

Same retrieval coverage, different kind of evidence:

| Question | Vector RAG | GraphRAG |
|---|---|---|
| Who directed X? | A string in the movie document, if TF-IDF retrieved it | `(Movie)-[:DIRECTED_BY]->(Director)` |
| Genres of X? | A `Genres:` field in a chunk | `(Movie)-[:HAS_GENRE]->(Genre)` |
| Shared cast? | Implicit overlap across two overviews | `(A)-[:ACTED_BY]->(Actor)<-[:ACTED_BY]-(B)` |

The LLM does not invent the catalog. It writes Cypher, Neo4j returns grounded rows, and the model turns those rows into a sentence. You can inspect the query. A TF-IDF snippet does not show *why* two titles connect.

### 3. Less “found it, then dropped it”

Several vector failures are `evidence_in_context=true` and `factual_correct=false`: the fact was in the prompt and the model still omitted it.

| id | Kind | Fact in docs? | Vector answer |
|---|---|---|---|
| `director_dark_knight` | director | yes | miss |
| `genres_forrest` | genres | yes | miss |
| `actors_dark_knight` | cast | yes | miss |
| `shared_actors_toy_story` | multi-hop | yes | miss |
| `shared_actors_batman` | multi-hop | yes | miss |
| `shared_actors_oceans` | multi-hop | yes | miss |

Cypher rows are short and aligned with the question. Four concatenated documents mix overview, cast, and genres — a local model often skips the fact.

### 4. Structured lookup (director / genre / cast)

| Kind | n | GraphRAG | Vector RAG |
|---|---:|---:|---:|
| director | 8 | **1.00** | 0.75 |
| genres | 6 | **1.00** | 0.67 |
| actors | 5 | **1.00** | 0.80 |

The graph `MATCH`es the title. TF-IDF has to retrieve the right document first.

Two vector misses were retrieval, not generation:

- `director_shawshank` — wrong document (`evidence_in_context=false`)
- `genres_get_out` — same

GraphRAG answered both with a direct traversal.

### 5. Multi-hop questions (the graph-shaped case)

| Kind | n | GraphRAG | Vector RAG |
|---|---:|---:|---:|
| shared_director | 5 | 1.00 | 1.00 |
| shared_actors | 6 | **0.83** | 0.50 |

Shared director ties (Nolan / Coppola / Tarantino / Lasseter are easy even in prose). **Shared cast** is where vector RAG breaks: the answer is an intersection, not a nearest neighbor.

Example: “Which actors appear in both Ocean's Eleven and Ocean's Twelve?”

- Vector RAG: both docs can sit in the top-4; the model does not cross the casts → miss
- GraphRAG: `MATCH (a)-[:ACTED_BY]->(p)<-[:ACTED_BY]-(b)` → hit

GraphRAG's only miss on this set (`shared_actors_godfather`) is invalid Cypher, not the traversal pattern.

---

## Where it did not win

**Tie on `shared_director`.** 5/5 on both sides. The director name is usually in both movie documents.

**Same evidence coverage (0.93).** GraphRAG does not retrieve more gold strings. It retrieves them in a form the model can use. On `shared_director_coppola` the answer was correct even with `evidence_in_context=false` (parametric knowledge or context serialization). The protocol records that; it is not the usual path.

**A small model can invert the ranking.** Snapshot `llama3.2:3b`, 7 questions only:

| Method | n | Factual | Evidence |
|---|---:|---:|---:|
| vector_rag | 7 | **0.86** | 1.00 |
| graph_rag | 7 | 0.71 | 0.71 |

Small models write bad Cypher (`I couldn't find evidence in the graph`). The published win assumes a model that can generate Cypher (`qwen3:8b` here). Docker Compose still defaults to `llama3.2:3b` for RAM; use `qwen3:8b` if you want this QA behavior.

**Vector RAG is still better at prose.** Plot summaries and “same vibe” are not what the gold set measures. The gap is **relational facts**, not style.

**The vector baseline is TF-IDF, not dense embeddings.** That is the honest baseline in `qa_eval.py`, not a production RAG stack. The claim is about the *kind* of evidence, not “we beat every retriever.”

---

## Protocol

```bash
uv run python scripts/run_evaluation.py --qa --model qwen3:8b
```

1. Build ~30 gold questions from Neo4j (needs `tmdbEnriched=true` movies).
2. Run vector RAG (TF-IDF, `top_k=4`) and GraphRAG on the same question.
3. Score normalized gold-name substrings in the answer and in the context.
4. Write `eval/latest_results.json` (this CLI is the only writer).

For list-valued gold (`genres`, `actors`, `shared_*`) the scorer needs at least 2 hits when there are more than 2 expected names; otherwise 1.

---

## How to read a row

| factual | evidence | Meaning |
|---|---|---|
| true / true | Retrieval and generation both worked |
| false / true | Retrieval worked; the LLM dropped the fact (common vector-RAG failure here) |
| false / false on GraphRAG | Cypher was probably invalid — inspect `retrieved_or_cypher` |
| false / false on vector RAG | TF-IDF did not retrieve the document |

---

## Re-running notebooks

Running notebooks **01–03** is read-only on the CSVs. It does not touch `latest_results.json`.

Notebook **04** writes `louvainCommunity` onto Movie nodes in Neo4j. That is additive GDS state, not a revert of the QA/ranking snapshot.

Notebook **05** *reads* `eval/latest_results.json` in the first results cell. Live re-runs (`RUN_GRAPH_RANKER` / `RUN_QA`) stay in the notebook kernel. They do **not** overwrite the JSON — only `scripts/run_evaluation.py` does. Defaults in the notebook are `False` so Run All keeps the published tables. If you set `RUN_QA = True` with `OLLAMA_MODEL=llama3.2:3b` and then **save** the notebook, the live QA cell can look like the old 7-question story again. The markdown tables and `latest_results.json` still hold the 30-question / `qwen3:8b` numbers until you re-run the CLI.

---

## Relation to the rest of the repo

| Piece | Job |
|---|---|
| `GraphRAG` | Question → Cypher → verbalized answer |
| `GraphRecommender` | Path ranking; **no** LLM ([doc](graph_recommender.md)) |
| `explanation_engine` | Shortest path between two titles already chosen |

The ranking win (Hit@10 **0.34 vs ~0.10** on 200 users) is the path ranker, not this comparison. Do not mix the two numbers.

---

## Code map

| Symbol | Role |
|---|---|
| `rag/graph_rag.py` | `GraphCypherQAChain` + few-shots + Cypher preamble strip |
| `eval/qa_eval.py` | Gold set, `VectorRAG`, scoring |
| `scripts/run_evaluation.py --qa` | CLI; writes `eval/latest_results.json` |
| Streamlit **Home** | Same `GraphRAGSystem` |
