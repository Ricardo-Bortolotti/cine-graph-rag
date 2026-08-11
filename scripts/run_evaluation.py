"""Run offline recommendation and/or QA evaluations.

  uv run python scripts/run_evaluation.py --recs --max-users 200
  uv run python scripts/run_evaluation.py --qa --model qwen3:8b
  uv run python scripts/run_evaluation.py --all

Writes eval/latest_results.json (the published snapshot). Notebooks read that
file; they do not write it. Re-running --qa with llama3.2:3b will overwrite
the 30-question qwen3:8b numbers — use --model qwen3:8b to keep them comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="CineGraphRAG offline evaluation")
    parser.add_argument("--recs", action="store_true", help="Ranking eval vs baselines")
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Vector RAG vs GraphRAG (30 gold questions; use --model qwen3:8b)",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--max-users",
        type=int,
        default=200,
        help="Held-out users to sample (0 = all users with relevant test items)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Liked train titles used as GraphRecommender seeds",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip GraphRecommender (baselines only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model for QA (overrides OLLAMA_MODEL)",
    )
    args = parser.parse_args()
    run_recs = args.recs or args.all
    run_qa = args.qa or args.all
    if not run_recs and not run_qa:
        parser.print_help()
        return 2

    payload: dict = {}
    if run_recs:
        from recommend_eval import run_recommendation_eval

        summary, _, paired = run_recommendation_eval(
            max_users=args.max_users,
            k=args.k,
            n_seeds=args.n_seeds,
            include_graph=not args.no_graph,
        )
        print("\nRecommendation ranking")
        print(summary.to_string(index=False))
        if paired:
            import pandas as pd

            print("\nPaired wins (graph vs baseline, same users)")
            print(pd.DataFrame(paired).to_string(index=False))
        payload["ranking"] = summary.to_dict(orient="records")
        payload["ranking_protocol"] = {
            "split": "per-user holdout test_size=0.2 random_state=42",
            "k": args.k,
            "max_users": args.max_users,
            "n_seeds": args.n_seeds,
            "relevant": "held-out rating >= 4.0",
        }
        payload["ranking_paired"] = paired

    if run_qa:
        from qa_eval import results_frame, run_qa_eval, summarize_qa, summarize_qa_by_kind

        _, results = run_qa_eval(model=args.model)
        frame = results_frame(results)
        print("\nQA factual eval")
        print(summarize_qa(frame).to_string(index=False))
        by_kind = summarize_qa_by_kind(frame)
        if not by_kind.empty:
            print("\nBy question type")
            print(by_kind.to_string(index=False))
        print("\nPer question")
        cols = [
            "method",
            "kind",
            "id",
            "factual_correct",
            "evidence_in_context",
            "error",
        ]
        print(frame[cols].to_string(index=False))
        model_name = args.model or "default"
        qa_block = {
            "model": model_name,
            "n_questions": int(frame["id"].nunique()) if not frame.empty else 0,
            "qa_summary": summarize_qa(frame).to_dict(orient="records"),
            "qa_by_kind": by_kind.to_dict(orient="records") if not by_kind.empty else [],
            "qa": frame[cols].to_dict(orient="records"),
        }
        payload["qa_summary"] = qa_block["qa_summary"]
        payload["qa"] = qa_block["qa"]
        payload.setdefault("qa_by_model", {})
        payload["qa_by_model"][model_name] = qa_block

    if payload:
        out = ROOT / "eval" / "latest_results.json"
        existing: dict = {}
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        by_model = existing.get("qa_by_model", {})
        by_model.update(payload.get("qa_by_model", {}))
        existing.update(payload)
        existing["qa_by_model"] = by_model
        out.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
