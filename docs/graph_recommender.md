# GraphRecommender

Path-based movie ranking over the MovieLens + TMDB Neo4j graph.

This is **not** GraphRAG. There is no LLM in the loop. A fixed Cypher query walks five kinds of bridges from a seed title, scores candidates with hand-set weights, and returns the evidence paths so a person can read *why* a title was ranked.

Source of truth: [`recommender/graph_recommender.py`](../recommender/graph_recommender.py).

Related, but different:

| Piece | Job |
|---|---|
| `GraphRecommender` | Choose *what* to recommend from a seed (or a user's liked seeds) |
| [`explanation_engine.py`](../recommender/explanation_engine.py) | Given two titles already chosen, return the shortest path between them |
| [`graph_rag.py`](../rag/graph_rag.py) | Natural-language question → generated Cypher → verbalized answer ([why it beats vector RAG](graph_rag.md)) |

---

## What it needs

1. `graph/build_graph.py` — `User`, `Movie`, `RATED`
2. `graph/enrich_tmdb.py` — `Director`, `Actor`, `Genre`, `Keyword` and their edges

Without TMDB enrichment, director / actor / keyword signals are empty. Genre still comes from MovieLens on the movie node in some views, but the ranker traverses `HAS_GENRE` from the graph. Co-rating still works from MovieLens ratings alone.

---

## Signals

Every candidate is another `Movie` that shares a bridge node with the seed:

```text
(seed:Movie)-[:REL]->(bridge)<-[:REL]-(rec:Movie)
```

| Signal | Bridge | Relationship | Weight | `via` example |
|---|---|---|---:|---|
| `director` | `Director` | `DIRECTED_BY` | 5.0 | Christopher Nolan |
| `actor` | `Actor` | `ACTED_BY` | 3.0 | Leonardo DiCaprio |
| `keyword` | `Keyword` | `HAS_KEYWORD` | 2.0 | space travel |
| `co_rating` | `User` | `RATED` both ≥ 4.0, ≥ 3 shared fans | 1.5 × log₁₀(n+1) | `12 shared fans` |
| `genre` | `Genre` | `HAS_GENRE` | 1.0 | Sci-Fi |

Weights live in `WEIGHTS` and are documented in code as tuned for **interpretability**, not offline NDCG. Director outranks genre on purpose: “same Nolan film” is a sentence people accept; “both are Drama” is not.

Co-rating is the only collaborative signal. It requires:

- both ratings ≥ `min_rating` (default `4.0`)
- at least `min_shared_users` distinct fans (default `3`)
- score `wCoRating * log10(sharedUsers + 1)` so a blockbuster pair does not drown everything else

---

## Score

For one seed movie, Cypher does the following:

1. `UNION ALL` the five traversals. Each hit is `(rec, signal, via, points)`.
2. Group by candidate and **sum** `points`. Sharing three actors adds three actor hits, not one.
3. Collect those hits as `evidence` (signal, via, points, and a path template).
4. Add a small catalog prior so obscure zero-rating titles do not crowd the top:

```text
score = rawScore
      + 0.25 * log10(nRatings + 1) * (avgRating / 5)   if the movie has ratings
```

5. `ORDER BY score DESC LIMIT $limit`.

The Python layer turns each evidence row into a readable line, for example:

```text
Same director: Christopher Nolan (+5.00) via (Movie)-[:DIRECTED_BY]->(Director)<-[:DIRECTED_BY]-(Movie)
Shared actor: Leonardo DiCaprio (+3.00) via (Movie)-[:ACTED_BY]->(Actor)<-[:ACTED_BY]-(Movie)
```

Duplicate `(signal, via)` pairs keep the strongest points. At most eight lines are shown.

---

## Two entry points

### Seed title — `recommend(movie_id)`

Used by the CLI and the Streamlit “start from a movie” flow.

```bash
uv run python recommender/graph_recommender.py --title "Matrix" --limit 5
```

Resolves the title with a case-insensitive `CONTAINS` (shortest title wins if several match), then runs `RECOMMEND_CYPHER`.

### User profile — `recommend_for_user(seed_ids, seen_ids)`

Used by Personalized Recommendations and by [`eval/recommend_eval.py`](../eval/recommend_eval.py).

1. Take the user's top liked training titles (eval default: 5 movies with rating ≥ 4).
2. Call `recommend` on each seed (`per_seed_limit=25`).
3. Drop anything in `seen_ids` and the seeds themselves.
4. **Sum** candidate scores across seeds. A title that is a Nolan neighbor *and* a DiCaprio neighbor of two different likes rises.
5. Keep the evidence blob from the seed where that candidate scored highest.

This is how the ranker is compared to popularity and item-CF. Item-CF sees the full rating vector; the graph only sees those liked seeds. Using one or two seeds understates it.

Offline protocol and numbers: [`notebooks/05_evaluation.ipynb`](../notebooks/05_evaluation.ipynb), snapshot [`eval/latest_results.json`](../eval/latest_results.json). GraphRAG QA (a different comparison) is in [`graph_rag.md`](graph_rag.md).

---

## What it does not do

- No matrix factorization, no learned embeddings, no gradient step.
- No LLM. GraphRAG can *talk* about a recommendation; it does not produce this ranking.
- It does not find the shortest path between two arbitrary titles — that is `explanation_engine`.
- Weights are not cross-validated. Changing them changes both the list and the story you can tell.

---

## How to read a result

A high score usually means several cheap signals stacked (many shared genres) **or** one expensive signal (same director) plus a bit of audience overlap.

If TMDB enrichment is missing on the seed, expect genre + co-rating only, and thinner explanations.

If everything recommended is a blockbuster, the `0.25 * log10(nRatings)` term and the co-rating fan count are doing more work than the metadata paths.

---

## Tuning the weights

Yes — they are hyperparameters. Do **not** re-run Neo4j for every combination (200 users × 5 seeds is ~90s per vector). Cache raw signal counts once, then rescore offline.

```bash
uv run python scripts/tune_recommender_weights.py --max-users 200 --trials 60
```

What it does:

1. Pulls candidates with unit weights and `per_seed_limit=40`.
2. Stores counts for director / actor / keyword / genre / co-rating per `(user, movie)`.
3. Random-searches the five weights.
4. Chooses the best **tune** split (70% of users) and reports nDCG on the **holdout** 30%.

It writes `eval/tuned_weights.json` and does **not** overwrite `WEIGHTS` in code. A holdout gain can mean “more co-rating / less director”, which may help nDCG and hurt the story you want to tell. The default weights stay interpretability-first until you decide to promote a vector.

Limits: the candidate pool is frozen at collect time. A weight that would promote a movie outside those 40-per-seed slots cannot appear. `min_rating` and `min_shared_users` still require a new Neo4j pull.

---

## Code map

| Symbol | Role |
|---|---|
| `WEIGHTS` | Default signal weights (interpretability, not a fitted optimum) |
| `RECOMMEND_CYPHER` | All five traversals + scoring |
| `GraphRecommender.recommend` | One seed → ranked `Recommendation` list |
| `GraphRecommender.recommend_for_user` | Many seeds → user-level top-N |
| `Evidence` / `Recommendation.explanation_paths` | Human-readable why |
| Streamlit **Personalized Recommendations** | Calls the same class |
| Streamlit **Explain Recommendation** | `explanation_engine`, not this ranker |
