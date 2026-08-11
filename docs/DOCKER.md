# Docker setup

Services:

| Service | Role | Ports |
|---|---|---|
| `app` | Streamlit UI | `8501` |
| `neo4j` | Neo4j Community + GDS + APOC | HTTP `7474`, Bolt `7687` |
| `ollama` | Local LLM for GraphRAG | `11434` |
| `ollama-init` | One-shot model pull | — |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine + Compose v2
- MovieLens **latest-small** extracted under `data/raw/ml-latest-small/ml-latest-small/`
- Optional: TMDB API key for enrichment

On Windows, Docker Desktop must be **Running** before `docker compose up`.  
`open //./pipe/dockerDesktopLinuxEngine` means the engine is offline.

## 1. Environment

If you already have a working Aura `.env`, do **not** overwrite it with `.env.example`.

Keep cloud credentials under `NEO4J_*` and add Docker-only keys:

```env
DOCKER_NEO4J_USER=neo4j
DOCKER_NEO4J_PASSWORD=password
DOCKER_NEO4J_DATABASE=neo4j
DOCKER_OLLAMA_MODEL=llama3.2:3b
```

| Variable | Used by | Notes |
|---|---|---|
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | `uv run` | Aura **or** `bolt://localhost:7687` |
| `DOCKER_NEO4J_*` | Compose Neo4j | Separate from Aura on purpose |
| `DOCKER_OLLAMA_MODEL` | `ollama-init` and the `app` container | Default `llama3.2:3b` |
| `OLLAMA_MODEL` | Local `uv run streamlit` | Must match a model already pulled |
| `TMDB_API_KEY` | Enrichment | Shared |
| `STREAMLIT_PORT` | Host publish | Default `8501` |

First-time setup only:

```bash
cp .env.example .env
```

## 2. Start the stack

```bash
docker compose up --build -d
```

Neo4j installs **Graph Data Science** and **APOC** on first boot (`NEO4J_PLUGINS`). That download can take a few minutes. GDS is required for `notebooks/04_graph_analytics.ipynb`. APOC is required for GraphRAG schema introspection (`apoc.meta.data()`).

Boot order:

1. Neo4j becomes healthy
2. Ollama becomes healthy
3. `ollama-init` pulls `DOCKER_OLLAMA_MODEL`
4. Streamlit starts

```bash
docker compose ps
docker compose logs -f neo4j
```

Open:

- Streamlit: http://localhost:8501
- Neo4j Browser: http://localhost:7474

Do not run Cypher against Bolt until `cine-neo4j` is **healthy**. `Started` is not enough while plugins are installing.

## 3. Load the graph (once)

```bash
docker compose exec app python graph/build_graph.py \
  --data-dir data/raw/ml-latest-small/ml-latest-small
```

Optional TMDB enrichment (`TMDB_API_KEY` required):

```bash
docker compose exec app python graph/enrich_tmdb.py --limit 50
docker compose exec app python graph/enrich_tmdb.py
```

From the host, against published Bolt:

```bash
uv run python graph/build_graph.py
uv run python graph/enrich_tmdb.py
```

## 4. Use the app

1. Open http://localhost:8501
2. **Personalized Recommendations** / **Explain** / **Explorer** / **Analytics**
3. **Home → GraphRAG** — Ollama must have finished pulling the model. Compose defaults to `llama3.2:3b` (small RAM). The published QA win in [`graph_rag.md`](graph_rag.md) used `qwen3:8b`; the 3B model often emits invalid Cypher.

Example questions:

- Recommend movies like Interstellar
- Show connections between The Matrix and Cloud Atlas
- Which genres does Toy Story belong to?

Offline eval (from the host, against Bolt + Ollama). This **writes** `eval/latest_results.json`:

```bash
uv run python scripts/run_evaluation.py --recs --max-users 200
uv run python scripts/run_evaluation.py --qa --model qwen3:8b
```

## 5. Common commands

```bash
docker compose down
docker compose down -v          # also deletes Neo4j + Ollama volumes

docker compose build app
docker compose up -d app

docker compose run --rm ollama-init
docker compose exec ollama ollama list
docker compose exec ollama ollama pull llama3.2:3b

docker compose exec neo4j cypher-shell -u neo4j -p password "RETURN gds.version(), apoc.version();"
```

## 6. Windows PowerShell

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose exec app python graph/build_graph.py --data-dir data/raw/ml-latest-small/ml-latest-small
```

Confirm `data\raw\ml-latest-small\ml-latest-small\movies.csv` exists on the host (mounted read-only into `app`).

## 7. Resources

| Service | RAM (approx.) |
|---|---|
| Neo4j | 1–2 GB |
| Ollama `llama3.2:3b` | ~3–4 GB |
| Ollama `qwen3:8b` | ~6–8 GB+ |
| Streamlit | ~0.5–1 GB |

CPU-only Ollama works; answers are slower. GPU passthrough is optional in `docker-compose.yml`.

## 8. Aura vs Docker Neo4j

| | Docker Neo4j | AuraDB Free |
|---|---|---|
| Setup | `docker compose up` | Cloud console |
| GDS / APOC | Installed via `NEO4J_PLUGINS` | GDS not included on Aura Free |
| App container | URI forced to `bolt://neo4j:7687` | Use `NEO4J_*` in `.env` for `uv run` |

You can keep Aura credentials in `.env` for cloud demos and `DOCKER_NEO4J_*` for the local stack.

## Troubleshooting

**`open //./pipe/dockerDesktopLinuxEngine`**  
Start Docker Desktop and wait until the engine is Running.

**`Unrecognized setting ... PASSWORD`**  
Do not set `NEO4J_PASSWORD` as a container env var. Auth is `NEO4J_AUTH=user/password` only.

**`Connection refused` / incomplete Bolt handshake**  
Wait until `docker compose ps` shows `healthy`. First boot downloads GDS and APOC.

**`Could not use APOC procedures`**  
Recreate Neo4j after enabling `apoc` in `NEO4J_PLUGINS`, then `RETURN apoc.version();`.

**`model '…' not found (404)`**  
`OLLAMA_MODEL` in `.env` must match `docker compose exec ollama ollama list`. Align the name or `ollama pull` it.

**No movies in the UI**  
Run `build_graph.py`. The data path must exist under `./data` on the host.

**Port already in use**  
Change `STREAMLIT_PORT`, `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT`, or `OLLAMA_PORT` in `.env`.

**Aura login broken after copying `.env.example`**  
Restore Aura `NEO4J_*` values. Keep Docker settings under `DOCKER_NEO4J_*` only.
