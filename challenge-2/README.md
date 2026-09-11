# Wunder Predictorium — LSTM V5 Solution

[![Validation WPC](https://img.shields.io/badge/Validation%20WPC-0.2574-blue)](https://wunder.dog)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

My solution for [Wunder Challenge 2](https://wunder.dog), a high-frequency trading sequence prediction competition. Validation weighted Pearson correlation: **0.2574**.

## Task

Predict two future price movement indicators (`t0`, `t1`) from a sequence of Limit Order Book states. Each sequence is 1000 steps; predictions are scored on steps 99–999 using weighted Pearson correlation (weighted by target amplitude).

## Architecture

**LSTM V5** — trained with 5-fold cross-validation, ensemble inference over all fold checkpoints.

```
Input (32) → FeatureEngineer (→ 101) → LayerNorm
          → LSTM (hidden=256, 3 layers)
          → Skip connection (engineered features → 64)
          → Concat (256 + 64 = 320) → LayerNorm
          → FC(320→256) → GELU → Dropout
          → FC(256→128) → GELU → Dropout
          → FC(128→2)
```

### Feature engineering (32 raw → 101)

Hand-crafted LOB features computed inside the ONNX graph:
- Bid/ask spreads, volume imbalance, price distances from mid
- Cumulative volumes, inter-level price steps
- Volume-weighted mid price, pressure ratios
- Trade imbalance (raw + normalized), spread as % of mid
- Depth imbalance, micro-price deviation from mid
- Relative spread depth across levels, trade intensity, net order flow
- Herfindahl concentration index for bid/ask volume distribution

### Training

- **Loss**: 85% weighted Pearson + 15% weighted MSE
- **Optimizer**: AdamW + OneCycleLR (lr=1e-3, 50 epochs)
- **Regularization**: gradient clipping (1.0), weight decay (5e-4), dropout (0.2)
- **Augmentation**: sequence-level mixup (α=0.2)
- **Gradient accumulation**: 4 steps (effective batch size = 1024)
- **SWA**: stochastic weight averaging from epoch 35

## Setup

```bash
# Python 3.11+
python -m venv env
source env/bin/activate
pip install torch numpy pandas pyarrow onnx onnxruntime scikit-learn tqdm
```

Place competition data in:
```
competition_package/datasets/train.parquet
competition_package/datasets/valid.parquet
```

## Usage

**Train all folds:**
```bash
python src/train.py
```

**Train single fold:**
```bash
python src/train.py --fold 0
```

Checkpoints are saved as `src/model_fold_{i}.pt` and exported to `src/model_fold_{i}.onnx`.

**Export existing checkpoints to ONNX:**
```bash
python export_folds.py
```

**Score locally:**
```bash
python src/score.py
```

**Run contract tests:**
```bash
cd src && python -m pytest test_contract.py -v
```

**Test in Docker (matches competition environment):**
```bash
# Requires competition datasets in competition_package/datasets/
mise run docker-test
# or manually:
docker build -t wunder-scorer -f docker/Dockerfile .
docker run --rm --cpus="1" --memory="16g" --network=none \
  -v "$(pwd)/src/solution.py:/app/solution.py:ro" \
  -v "$(pwd)/docker/run_scorer.py:/app/run_scorer.py:ro" \
  wunder-scorer
```

**Package for submission:**
```bash
mise run submit
```

## Project structure

```
src/
  model.py          # LOBPredictor: FeatureEngineer + LSTM + skip connection
  train.py          # 5-fold CV training pipeline
  solution.py       # Competition submission entry point (ONNX ensemble)
  loss.py           # WeightedPearsonLoss + WeightedMSELoss
  score.py          # Local validation scorer
  test_contract.py  # Competition contract tests
export_folds.py     # Export .pt checkpoints to .onnx
competition_package/
  utils.py          # DataPoint, ScorerStepByStep, weighted_pearson_correlation
  example_solution/ # Baseline GRU provided by organizers
docker/
  Dockerfile        # Mirrors competition scoring environment
  run_scorer.py     # Scorer script inside Docker
```

## License

MIT — see [LICENSE](LICENSE).
