"""Quick scorer script for src/solution.py."""
import sys
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "competition_package"))

from utils import ScorerStepByStep
from solution import PredictionModel

model = PredictionModel()
scorer = ScorerStepByStep(os.path.join(CURRENT_DIR, "..", "competition_package", "datasets", "valid.parquet"))

print("Scoring solution on valid.parquet...")
t0 = time.time()
results = scorer.score(model)
elapsed = time.time() - t0

print(f"\nWeighted Pearson: {results['weighted_pearson']:.6f}")
for key, val in results.items():
    if key != "weighted_pearson":
        print(f"  {key}: {val:.6f}")
print(f"\nTime: {elapsed:.1f}s ({elapsed/1444*1500:.1f}s extrapolated to test set)")
