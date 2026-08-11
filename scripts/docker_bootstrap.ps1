# Bootstrap CineGraphRAG Docker stack (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example — edit secrets if needed."
} else {
  Write-Host "Keeping existing .env (Aura credentials preserved)."
}

$movies = "data\raw\ml-latest-small\ml-latest-small\movies.csv"
if (-not (Test-Path $movies)) {
  Write-Error "Missing $movies — extract MovieLens latest-small under data/raw/ first."
}

Write-Host "Starting stack..."
docker compose up --build -d

Write-Host "Loading MovieLens into Neo4j..."
docker compose exec app python graph/build_graph.py --data-dir data/raw/ml-latest-small/ml-latest-small

Write-Host ""
Write-Host "Done."
Write-Host "  Streamlit: http://localhost:8501"
Write-Host "  Neo4j:     http://localhost:7474"
Write-Host "Optional enrichment:"
Write-Host "  docker compose exec app python graph/enrich_tmdb.py --limit 50"
