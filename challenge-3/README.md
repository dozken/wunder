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
├── datasets/       # starter pack data (gitignored)
├── docker/         # local replica of the scoring container
├── src/            # model, training, inference
└── wunder_docs/    # mirrored challenge docs
```

## Get the data

```bash
mise run fetch-data     # streams the 33.7 GB starter pack into ./
```

Needs ~35 GB free disk.
