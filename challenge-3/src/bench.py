"""Measure per-row inference latency against the competition budget.

The scorer feeds 37,460,000 validation rows (and a test set of the same shape)
through PredictionModel.predict one row at a time, with a 60 minute wall clock
budget on a single vCPU. That is ~96 us per call. This script times a model on
synthetic rows so the budget can be checked before any real training.

Usage:
    python src/bench.py                        # provided GRU baseline
    python src/bench.py --solution src/solution.py
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTERPACK = ROOT / "wnn_connectome_starterpack"

# Competition constants.
TEST_ROWS = 37_460_000          # sequences x 20,000, same shape as valid
BUDGET_SECONDS = 60 * 60
SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112


def load_model(solution_path: Path):
    """Import solution_path as a module and instantiate its PredictionModel."""
    sys.path.insert(0, str(STARTERPACK))          # for `from utils import DataPoint`
    sys.path.insert(0, str(solution_path.parent))  # for the model's own artifacts
    spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def synthetic_rows(rows: int, seed: int = 0) -> np.ndarray:
    """Plausible-magnitude features. Only the shape and dtype affect timing."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, N_FEATURES)).astype(np.float32)


def bench(model, rows: int, repeats: int) -> dict:
    from utils import DataPoint

    features = synthetic_rows(rows)
    per_run = []
    for run in range(repeats):
        # A fresh seq_ix each run so the model takes its sequence-reset path,
        # exactly as it will at every 20,000 row boundary.
        seq_ix = 1000 + run
        start = time.perf_counter()
        for step in range(rows):
            model.predict(DataPoint(seq_ix, step, step >= WARMUP, features[step]))
        per_run.append(time.perf_counter() - start)
    return {
        "rows": rows,
        "repeats": repeats,
        "best_seconds": min(per_run),
        "median_seconds": statistics.median(per_run),
        "per_row_us": min(per_run) / rows * 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path,
                        default=STARTERPACK / "baseline" / "solution.py")
    parser.add_argument("--rows", type=int, default=SEQUENCE_LENGTH,
                        help="rows per timed run (default: one full sequence)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not args.solution.exists():
        print(f"no solution at {args.solution}", file=sys.stderr)
        return 1

    model = load_model(args.solution)
    result = bench(model, args.rows, args.repeats)

    per_row = result["per_row_us"]
    projected = per_row * 1e-6 * TEST_ROWS
    budget_per_row = BUDGET_SECONDS / TEST_ROWS * 1e6

    print(f"solution      {args.solution}")
    print(f"rows timed    {result['rows']:,} x {result['repeats']}")
    print(f"best run      {result['best_seconds']:.3f} s")
    print(f"per row       {per_row:.1f} us   (budget {budget_per_row:.1f} us)")
    print(f"full test set {projected / 60:.1f} min  (budget {BUDGET_SECONDS / 60:.0f} min)")
    print(f"headroom      {BUDGET_SECONDS / projected:.2f}x")
    print()
    print("Note: timings here are this machine's cores and ISA. The scorer runs one")
    print("x86 vCPU in a container -- confirm with `mise run docker-test` before trusting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
