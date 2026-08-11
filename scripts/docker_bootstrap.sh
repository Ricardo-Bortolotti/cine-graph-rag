#!/usr/bin/env bash
# Bootstrap CineGraphRAG Docker stack (Linux/macOS)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit secrets if needed."
else
  echo "Keeping existing .env (Aura credentials preserved)."
fi

DATA_MOVIES="data/raw/ml-latest-small/ml-latest-small/movies.csv"
if [[ ! -f "$DATA_MOVIES" ]]; then
  echo "ERROR: missing $DATA_MOVIES"
  echo "Extract MovieLens latest-small under data/raw/ first."
  exit 1
fi

echo "Starting stack..."
docker compose up --build -d

echo "Waiting for app health..."
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:${STREAMLIT_PORT:-8501}/_stcore/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

echo "Loading MovieLens into Neo4j..."
docker compose exec -T app python graph/build_graph.py \
  --data-dir data/raw/ml-latest-small/ml-latest-small

echo
echo "Done."
echo "  Streamlit: http://localhost:${STREAMLIT_PORT:-8501}"
echo "  Neo4j:     http://localhost:${NEO4J_HTTP_PORT:-7474}"
echo "Optional enrichment:"
echo "  docker compose exec app python graph/enrich_tmdb.py --limit 50"
