"""Runs inside the scorer container: drives /app/src/solution.py through the
organisers' row-by-row scorer on the first N validation sequences and reports
the projected time for the full test set.

    SEQS=20 python run_scorer.py
"""
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
from utils import BlockAccumulator, DataPoint, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, validate_sequence  # noqa: E402
from solution import PredictionModel  # noqa: E402

DATASET = "/app/datasets/valid.parquet"
TEST_ROWS = 37_460_000
BUDGET = 3600.0
SEQS = int(os.environ.get("SEQS", "20"))

print(f"python {sys.version.split()[0]}  numpy {np.__version__}  cpus {os.cpu_count()}")
model = PredictionModel()
parquet = pq.ParquetFile(DATASET)
n = min(SEQS, parquet.num_row_groups) if SEQS > 0 else parquet.num_row_groups
acc = BlockAccumulator()
t0 = time.time()
for group in range(n):
    table = parquet.read_row_group(group, use_threads=False)
    seq, need = validate_sequence(table)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    t_seq = time.time()
    for step in range(SEQUENCE_LENGTH):
        value = model.predict(DataPoint(seq, step, bool(need[step]), features[step]))
        if need[step]:
            preds[step] = value
        elif value is not None:
            raise ValueError("returned a value during warm-up")
    acc.add(targets, preds, table["is_scored"].to_numpy().astype(bool) & need)
    print(f"seq {group + 1}/{n}  {(time.time() - t_seq) / SEQUENCE_LENGTH * 1e6:.1f} us/row  "
          f"running WP {acc.result()['weighted_pearson']:.4f}", flush=True)
elapsed = time.time() - t0
per_row = elapsed / (n * SEQUENCE_LENGTH)
print()
print(f"WP {acc.result()['weighted_pearson']:.6f} on {n} sequences")
print(f"{per_row * 1e6:.1f} us/row  ->  full test set {per_row * TEST_ROWS / 60:.1f} min  "
      f"(budget 60, headroom {BUDGET / (per_row * TEST_ROWS):.2f}x)")
