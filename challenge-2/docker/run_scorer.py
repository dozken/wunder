"""
Scorer script that runs inside the Docker container.
Mimics what the competition scorer does.
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/app")

from utils import ScorerStepByStep
from solution import PredictionModel

DATASET = "/app/datasets/valid.parquet"

print("=" * 50)
print("WUNDER SCORER — Docker Environment")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

model = PredictionModel()
scorer = ScorerStepByStep(DATASET)

print(f"\nScoring on {DATASET}...")
t0 = time.time()
results = scorer.score(model)
elapsed = time.time() - t0

print(f"\n{'='*50}")
print(f"Weighted Pearson: {results['weighted_pearson']:.6f}")
for key, val in results.items():
    if key != "weighted_pearson":
        print(f"  {key}: {val:.6f}")
print(f"\nTime: {elapsed:.1f}s")
print(f"Extrapolated to ~1500 seqs: {elapsed / 1444 * 1500:.1f}s")
print(f"Under 60 min limit: {'✅ YES' if elapsed / 1444 * 1500 < 3600 else '❌ NO'}")
