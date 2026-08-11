"""Offline evaluation helpers for ranking and GraphRAG QA.

Published snapshot: eval/latest_results.json
  - ranking: 200 users, K=10, 5 liked seeds (path ranker, not GraphRAG)
  - QA: qwen3:8b, 30 gold questions (GraphRAG vs TF-IDF vector RAG)

Only scripts/run_evaluation.py writes that JSON. Notebooks read it.
"""
