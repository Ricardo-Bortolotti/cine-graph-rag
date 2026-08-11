# Graph EDA — Neo4j Browser queries

Exploratory Cypher for the MovieLens + TMDB knowledge graph (Aura or local Docker).

## Prerequisites

1. Run `uv run python graph/build_graph.py` (Users, Movies, `RATED`).
2. Run `uv run python graph/enrich_tmdb.py` (genres, directors, actors, keywords).
3. Open Neo4j Browser (`http://localhost:7474` or Aura) and select the target database.
4. Paste each query below and run.

### Schema reminder

```text
(User)-[:RATED {rating, timestamp}]->(Movie)
(Movie)-[:HAS_GENRE]->(Genre)
(Movie)-[:DIRECTED_BY]->(Director)
(Movie)-[:ACTED_BY {character, order}]->(Actor)
(Movie)-[:HAS_KEYWORD]->(Keyword)
```

---

## 1. Dataset overview

### Node counts by label

```cypher
MATCH (n)
RETURN labels(n)[0] AS label, count(*) AS n
ORDER BY n DESC
```

### Relationship counts by type

```cypher
MATCH ()-[r]->()
RETURN type(r) AS rel, count(*) AS n
ORDER BY n DESC
```

### TMDB enrichment coverage

```cypher
MATCH (m:Movie)
RETURN
  count(m) AS totalMovies,
  sum(CASE WHEN m.tmdbEnriched THEN 1 ELSE 0 END) AS enriched,
  sum(CASE WHEN m.tmdbEnriched THEN 0 ELSE 1 END) AS pending,
  round(
    100.0 * sum(CASE WHEN m.tmdbEnriched THEN 1 ELSE 0 END) / count(m),
    1
  ) AS enrichedPct
```

**What to look for:** enrichment should cover most movies with a valid `tmdbId`. Pending rows are expected for missing/invalid TMDB links.

---

## 2. Ratings analysis

### Rating distribution

```cypher
MATCH ()-[r:RATED]->()
RETURN r.rating AS rating, count(*) AS n
ORDER BY rating
```

### Top movies by average rating (minimum support)

```cypher
MATCH (u:User)-[r:RATED]->(m:Movie)
WITH m, avg(r.rating) AS avgRating, count(*) AS n
WHERE n >= 50
RETURN m.title AS title, round(avgRating, 3) AS avgRating, n
ORDER BY avgRating DESC
LIMIT 15
```

### Most rated movies

```cypher
MATCH ()-[r:RATED]->(m:Movie)
RETURN m.title AS title, count(r) AS nRatings, round(avg(r.rating), 3) AS avgRating
ORDER BY nRatings DESC
LIMIT 15
```

**What to look for:** positive skew (many 4.0/5.0), and a popularity head vs long tail.

---

## 3. User analysis

### Ratings per user (summary)

```cypher
MATCH (u:User)-[r:RATED]->()
WITH u, count(r) AS n
RETURN
  min(n) AS minRatings,
  percentileCont(n, 0.5) AS medianRatings,
  percentileCont(n, 0.95) AS p95Ratings,
  max(n) AS maxRatings,
  avg(n) AS avgRatings
```

### Most active users

```cypher
MATCH (u:User)-[r:RATED]->()
RETURN u.userId AS userId, count(r) AS nRatings, round(avg(r.rating), 3) AS avgRating
ORDER BY nRatings DESC
LIMIT 15
```

### One user's favorite genres (multi-hop)

```cypher
MATCH (u:User {userId: 1})-[r:RATED]->(m:Movie)-[:HAS_GENRE]->(g:Genre)
WHERE r.rating >= 4.0
RETURN g.name AS genre, count(*) AS likes
ORDER BY likes DESC
```

---

## 4. Genre / cast / crew (TMDB)

### Genre frequency (catalog)

```cypher
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
RETURN g.name AS genre, count(m) AS nMovies
ORDER BY nMovies DESC
```

### Average rating by genre

```cypher
MATCH (:User)-[r:RATED]->(m:Movie)-[:HAS_GENRE]->(g:Genre)
RETURN
  g.name AS genre,
  count(r) AS nRatings,
  round(avg(r.rating), 3) AS avgRating
ORDER BY avgRating DESC
```

### Enriched movie profile (sample)

```cypher
MATCH (m:Movie)
WHERE m.tmdbEnriched = true
OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Director)
OPTIONAL MATCH (m)-[:ACTED_BY]->(a:Actor)
OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
RETURN
  m.title AS title,
  m.overview AS overview,
  collect(DISTINCT g.name) AS genres,
  collect(DISTINCT d.name) AS directors,
  collect(DISTINCT a.name) AS actors,
  collect(DISTINCT k.name) AS keywords
LIMIT 5
```

---

## 5. Graph structure / visual EDA

### Visual neighborhood (good for screenshots)

```cypher
MATCH path = (m:Movie)-[:HAS_GENRE|DIRECTED_BY|ACTED_BY|HAS_KEYWORD]->(x)
WHERE m.tmdbEnriched = true
RETURN path
LIMIT 50
```

### Same-director related movies

```cypher
MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)<-[:DIRECTED_BY]-(other:Movie)
WHERE m.title CONTAINS 'Toy Story'
RETURN d.name AS director, other.title AS related
LIMIT 20
```

### Shared actors between movies

```cypher
MATCH (m1:Movie)-[:ACTED_BY]->(a:Actor)<-[:ACTED_BY]-(m2:Movie)
WHERE m1.tmdbEnriched AND m2.tmdbEnriched AND elementId(m1) < elementId(m2)
RETURN
  m1.title AS movie1,
  m2.title AS movie2,
  collect(a.name) AS sharedActors,
  size(collect(a.name)) AS nShared
ORDER BY nShared DESC
LIMIT 10
```

### Keyword bridge (content similarity signal)

```cypher
MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(other:Movie)
WHERE m.title CONTAINS 'Matrix' AND m <> other
RETURN other.title AS related, collect(k.name) AS sharedKeywords
ORDER BY size(sharedKeywords) DESC
LIMIT 15
```

---

## 6. Long-tail / sparsity checks

### Movies with few ratings

```cypher
MATCH (m:Movie)
OPTIONAL MATCH (m)<-[r:RATED]-()
WITH m, count(r) AS n
RETURN
  sum(CASE WHEN n = 0 THEN 1 ELSE 0 END) AS unrated,
  sum(CASE WHEN n = 1 THEN 1 ELSE 0 END) AS oneRating,
  sum(CASE WHEN n <= 5 THEN 1 ELSE 0 END) AS upTo5,
  sum(CASE WHEN n >= 100 THEN 1 ELSE 0 END) AS popular100Plus
```

### Degree of Actor / Director hubs

```cypher
MATCH (a:Actor)<-[:ACTED_BY]-(m:Movie)
RETURN a.name AS actor, count(m) AS nMovies
ORDER BY nMovies DESC
LIMIT 15
```

---

## 7. Suggested EDA narrative

Use the queries above to write short insights:

1. **Scale** — how many nodes/relationships per label/type.
2. **Quality** — enrichment coverage; TMDB 404s leave some movies without metadata.
3. **Behavior** — rating skew; heavy-tailed user activity.
4. **Content graph** — genres/actors/directors/keywords create multi-hop paths. That is why GraphRAG beats chunk RAG on shared-cast / shared-director questions ([`graph_rag.md`](graph_rag.md)).
5. **Long tail** — many movies have few ratings → collaborative filtering alone is weak; graph/content edges help.

---

## Re-run enrichment (idempotent)

Already enriched movies are skipped (`tmdbEnriched = true`):

```powershell
uv run python graph/enrich_tmdb.py
```

If the PC slept mid-run and some writes failed, re-running resumes pending movies.
