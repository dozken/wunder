# Wunder Alpha Connectome (WNN33)

[Challenge site](https://wundernn.io/connectome/docs/quick_start) ·
Sep 11 2026 → Nov 15 2026 (submissions close) → Dec 1 2026 (winners) ·
$13,600 prize pool, top 8 · 5 submissions/day

## Task

Predict `t0` and `t1`, two undisclosed indicators of the future price movement of
instrument `i0`, from a streaming sequence of market states covering **two**
instruments (`i0` with targets, `i1` input-only).

Inference is a row-by-row callback: `PredictionModel.predict(data_point)` sees one row
at a time, must update internal state on every row, reset state when `seq_ix` changes,
and may never look ahead.

## Key numbers

| | |
|---|---|
| Features | 112 float32 (L11 book × 2 instruments + trades + 8 extra) |
| Sequence length | 20,000 rows (steps 0–98 warm-up, 99–19,999 predicted) |
| Train | 10,607 sequences / 212,140,000 rows |
| Valid | 1,873 sequences / 37,460,000 rows |
| Scored rows | subset of predicted rows, marked `is_scored` |
| Metric | Weighted Pearson, per target, per sequence, equally averaged; values clipped to [-2, 2] |
| Baseline | stateful vanilla GRU — **0.589595** WP on full valid |
| Runtime limit | 60 min for the whole test set, 1 vCPU, 16 GB RAM, no GPU, no network |
| Submission | `.zip` ≤ 20 MB with `solution.py` at the root |
| Starter pack | 33.7 GB |

## Differences from [Challenge 2 (Predictorium)](../challenge-2/)

The submission contract is nearly identical, but the modelling problem is not:

*   112 features instead of 32, spanning two instruments instead of one.
*   Sequences are 20× longer (20,000 vs 1,000 rows) and there are 20× more rows.
*   A hidden `is_scored` mask selects which predicted rows count.
*   **Feature index is not a depth level.** The docs state explicitly that `p0` is not
    necessarily the best bid. Most of the Challenge 2 hand-crafted LOB features
    (spreads, inter-level steps, micro-price) assume a price-ordered book and do not
    transfer without first recovering the level ordering.
*   Inference throughput matters: 37.5M rows in 60 min on 1 vCPU is ~10.4k rows/s,
    roughly **96 µs per `predict` call**.

## Layout

```
challenge-3/
├── README.md
├── mise.toml
├── docker/                      # local replica of the scoring container
├── src/                         # model, training, inference
├── wunder_docs/                 # challenge docs (copied from the starter pack)
└── wnn_connectome_starterpack/  # gitignored, 33.7 GB
    ├── datasets/{train,valid,valid_mask}.parquet
    ├── baseline/                # provided GRU baseline + ready-made submission zip
    ├── utils.py                 # DataPoint, weighted_pearson, ScorerStepByStep
    └── METRIC.md
```

## Get the data

```bash
mise run fetch-data       # streams the 33.7 GB starter pack into ./
mise run score-baseline   # reproduce the provided 0.589595 WP baseline
```

Needs ~35 GB free disk.

## Notes from the starter pack

*   `utils.py` enforces the contract strictly: rows must arrive in order, a new
    sequence must start at step 0, and `predict` must return `None` on warm-up rows
    **after** updating state.
*   A sequence is excluded from scoring unless each target has at least one strictly
    positive and one strictly negative value among its scored rows.
*   WP weights are `abs(clipped target)`, so large-magnitude moves dominate the score.
    Constant predictions score 0 for that sequence but do not exclude it.
*   The site says the container is Python 3.11; the starter pack README says 3.10.
    Pinned deps are `numpy==2.2.6`, `onnxruntime==1.23.2`, `pyarrow==19.0.1`.

## Inference cost model

`predict` is called 37.46M times per test set with a 60 minute budget on one
vCPU: **96 µs per call**. The cost is dominated by reading the recurrent
weights once per row, so it scales with parameter bytes rather than FLOPs.
Measured on an M-series core via `src/bench.py` (the scorer's x86 vCPU is
expected to be slower; calibrate with a real submission before trusting a
margin below ~2x):

| model | params | µs/row fp32 | µs/row int8 |
|---|---:|---:|---:|
| provided baseline GRU 128×2 | 190k | 25 | – |
| GRU 128×2 + proj 64 | 206k | 23 | 23 |
| GRU 192×2 + proj 64 | 429k | 36 | 35 |
| GRU 256×1 + proj 64 | 338k | 30 | 26 |
| GRU 256×2 + proj 64 | 733k | 54 | 52 |
| GRU 384×2 + proj 64 | 1.59M | 100 | 90 |
| LSTM 128×2 + proj 64 | 264k | 29 | 24 |
| LSTM 256×1 + proj 64 | 420k | 35 | 22 |
| LSTM 256×2 + proj 64 | 947k | 69 | 37 |

ONNX Runtime has a dynamic-int8 kernel for LSTM but not for GRU, so a
quantised LSTM buys roughly twice the capacity per microsecond. Ensembles
multiply cost linearly. Per-call fixed overhead is ~20 µs.

## Workflow

```bash
mise run bench                                   # latency of src/solution.py
python src/train.py --tag dev --seqs 512 --epochs 2
python src/train.py --tag gru192 --hidden 192 --epochs 8
python src/score.py --checkpoint runs/gru192/best.pt      # full valid, batched
python src/export.py runs/gru192/best.pt src/model_a.onnx  # + parity check
python src/score.py --strict --seqs 20                    # organisers' scorer path
cd src && python -m pytest test_contract.py -v
mise run docker-test                              # 1 CPU container, SEQS=20
mise run submit                                   # -> submission.zip
```

## Data notes (from `src/eda.py`, 48+48 sampled sequences)

*   Features are **already rank-transformed and clipped**: every group lives in
    roughly [-2.4, 2.4] with mass piled at ±2.32 (the clip), `a0..a7` reach ±5.2.
    "Price-like" columns are not prices — no monotone ordering within a row,
    ask−bid signs are arbitrary per index — so book-geometry feature engineering
    from Challenge 2 does not apply. Standardisation is a no-op in practice.
*   Targets: std ≈ 0.95, ~22% exact zeros (zero weight in the metric), 3% beyond
    the ±2 clip. **`t0` and `t1` are strongly anti-correlated (−0.74)**.
*   Targets are smooth: within-sequence autocorrelation 0.96 at lag 1, 0.78 at
    lag 20, 0.41 at lag 100, ~0 by lag 500. They behave like overlapping
    forward-looking windows of a few hundred rows.
*   Scoring mask covers ~11% of required rows in ~540 short runs per sequence
    (median run length 2); first scored row is typically ~170. Scored rows have
    slightly larger |t| than unscored ones.
*   Regime structure is real: per-sequence feature means spread by ~0.5 global
    std across sequences and drift by 0.2–0.45 std within a sequence.
*   Matching `i0`/`i1` columns are only weakly correlated (|r| < 0.3).
*   A 256-sequence, 1-epoch smoke run already reaches val WP 0.525, so the
    problem saturates quickly; differences between good models will be small.
