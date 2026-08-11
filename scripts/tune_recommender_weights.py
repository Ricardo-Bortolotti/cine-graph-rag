"""Search GraphRecommender signal weights on a cached feature table.

  uv run python scripts/tune_recommender_weights.py --max-users 200 --trials 60

Does not overwrite WEIGHTS in graph_recommender.py. Writes eval/tuned_weights.json.
Path ranker only — not GraphRAG (see docs/graph_recommender.md).
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
    parser = argparse.ArgumentParser(description="Tune GraphRecommender weights")
    parser.add_argument("--max-users", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--per-seed-limit", type=int, default=40)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    from tune_weights import run_weight_search

    result = run_weight_search(
        max_users=args.max_users,
        n_seeds=args.n_seeds,
        trials=args.trials,
        per_seed_limit=args.per_seed_limit,
        k=args.k,
    )
    default = result["default"]
    best = result["best_on_tune"]
    print("\nDefault weights")
    print(json.dumps(default, indent=2))
    print("\nBest on tune split")
    print(json.dumps(best, indent=2))
    improved = best["hold_ndcg"] > default["hold_ndcg"]
    print(
        "\nHoldout nDCG: "
        f"default={default['hold_ndcg']:.4f}  tuned={best['hold_ndcg']:.4f}  "
        f"{'improved' if improved else 'no holdout gain — keep the hand weights'}"
    )
    out = ROOT / "eval" / "tuned_weights.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
