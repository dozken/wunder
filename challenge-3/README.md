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

**Calibration from submission `25KKGXOR` (GRU 192×2 proj64, 35 µs/row idle on
the Mac): the platform took 45m40s for the test set, i.e. ~73 µs/row — the
scorer's vCPU is ~2× slower than an M-series core.** With a 60 min budget the
ceiling is ~45 µs/row measured idle on the Mac; GRU 256×2 (57 µs) would time
out. Capacity gains have to come from cheaper graphs (the ~20 µs fixed
overhead is more than the GRU-192 compute itself), int8 LSTM if x86 VNNI makes
it fast there, or better training at equal size.

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

## Experiment log (dev proxy: 2048 train seqs, 1 epoch, 128 val seqs unless noted)

| date | run | val WP | note |
|---|---|---|---|
| 09-12 | GRU 256×2 proj128, lr 2e-3 | 0.433 | peak LR held too long wrecks it |
| 09-12 | same, lr 1e-3 | 0.505 | |
| 09-12 | same, **lr 5e-4** | **0.589** | reference recipe |
| 09-12 | same, lr 5e-4, seq-level Pearson loss | 0.570 | all-rows sequence objective is the wrong target |
| 09-12 | same, lr 5e-4, chunk 2000 | 0.559 | half the optimiser steps |
| 09-12 | LSTM 256×2 proj64, lr 5e-4 | 0.522 | trains 2.3× faster on MPS (native kernel) |
| 09-12 | oracle: train on 1500 valid seqs, loss on all rows, 2 ep | 0.569 | held-out valid; overfits in epoch 2 |
| 09-12 | oracle: same, **loss on is_scored rows only** | **0.605** | +0.04 from matching the metric's row selection |
| 09-12 | LSTM 256×2 proj128, lr 5e-4 / 1e-3 | 0.525 / 0.545 | GRU beats LSTM at equal size |
| 09-12 | LSTM 320×2 proj128, lr 5e-4 | 0.528 | |
| 09-12 | GRU 192×2 proj64, lr 5e-4 | 0.543 | |
| 09-12 | GRU 256×2 proj128, lr 3e-4 | 0.543 | the 0.589 reference is probably a lucky draw; proxy noise ≈ ±0.03 |
| 09-12 | GRU 256×2 proj128, **predicted soft mask** | 0.583 | mask model: held-out AUC 0.93, AP 0.66 |
| 09-12 | GRU 192×2 proj64, **predicted soft mask** | 0.575 | +0.03 over the same model on all rows |
| 09-12 | **full data**, GRU 192×2 proj64, soft mask, epoch 1 of 2 | **0.5955** (EMA, 192 held-out seqs) | packaged as `submissions/2026-09-12_gru192_e1`; epoch 2 lost to a session restart |
| 09-12 | ↳ submitted as `25KKGXOR` | **public 0.5617** (#76) | baseline's public score is 0.5719; 45m40s runtime |
| 09-12 | provided baseline on the same 192 held-out seqs | 0.6062 | so gru192_e1 was −0.011 locally too — the held-out subset is easier than the full valid (0.5896); no generalisation gap |
| 09-12 | train + valid mix, GRU 192×2, soft mask, chunk loss, epochs 1 / 2 | 0.5747 / 0.5527 (EMA) | gets *worse* with training under the chunk loss; killed before epoch 3 |
| 09-12 | GRU 192×2, soft mask, **seq-loss** (proxy) | 0.5887 proxy / **0.6083 held-out 192** | vs 0.5753 / 0.6011 for the same run with the chunk loss; beats the baseline (0.6062) on identical rows with 20% of the data, 1 epoch |
| 09-12 | seq-loss proxy variants (held-out 192): chunk 2000 / diff / no MSE / lr 7e-4 / dropout 0.2 | 0.5992 / 0.6151 / 0.5954 / 0.6158 / 0.6081 | diff and lr 7e-4 help slightly; keep chunk 1000 and the 0.1 MSE term |
| 09-12 | **full train+valid, GRU 192×2, soft mask, seq-loss, epoch 1 of 3** | **0.6496** (EMA; raw 0.6481) | quarters 0.66/0.65/0.65/0.63; epochs 2–3 pending; packaged as `submissions/2026-09-12_seq_mix_e1` |
| 09-12 | ↳ submitted as `0ATOIBJ1` | **public 0.6128** (#24) | offset −0.037; 38m36s runtime; prize line (#8) was 0.6272 |
| 09-13 | seq-loss proxy, 2 epochs | 0.6288 held-out | +0.02 over 1 epoch (0.6083): epochs matter |
| 09-13 | **full train+valid, seq-loss, epochs 2 / 3** | 0.6585 / **0.6603** (raw; EMA 0.6600) | quarters 0.67/0.67/0.66/0.64; packaged as `submissions/2026-09-13_seq_mix_e3` |
| 09-13 | ↳ submitted as `052UVJK5` | **public 0.6221** (#11) | offset −0.038; 38m19s; prize line (#8) 0.6272 |
| 09-13 | seq-loss proxy + aux is_scored head (0.3) | 0.6102 held-out | vs 0.6083 without: noise, dropped |
| 09-13 | full train+valid, seq-loss + **diff inputs, lr 7e-4**, 3 epochs | 0.6495 / 0.6591 / **0.6626** (raw) | +0.002 over the reference recipe; quarters 0.67/0.67/0.66/0.64 |
| 09-13 | mask model v2 (GRU 192, 4 epochs) | held-out AUC 0.9416 / AP 0.6925 | v1: 0.9315 / 0.6589; train relabelled to `runs/mask2/` |
| 09-13 | ↳ diff run submitted as `211TK6NX` | **public 0.6221** (= #3's score) | +0.002 held-out did not transfer; policy now: upload only if projected top-5 |
| 09-13 | seq-loss proxies: mask v2 / GRU 320×1 / 160×3 / 224×2 | 0.6106 / 0.6108 / 0.6083 / 0.6102 | all within ±0.005 of the 0.6083 reference — 1-epoch proxies can't resolve differences this small any more |
| 09-13 | proxies on diff + mask v2 + seq-loss: lr 7e-4 / **lr 1e-3** / **proj 128** | 0.6227 / 0.6273 / 0.6284 | the combined base is +0.01 over the older cluster; both knobs add ~+0.005 |
| 09-13 | teacher GRU 384×2 proj128, epoch 1 of 3 | 0.6546 | only +0.004 over the 192 at the same point — capacity is not the limit |

**Drift finding.** On the held-out set gru192_e1 beats the baseline in every
quarter of the sequence (0.64/0.64/0.62/0.60 vs 0.61/0.61/0.58/0.58) but
loses on whole sequences: its prediction level drifts within a sequence.
Per-chunk Pearson is blind to that; the metric's whole-sequence centring is
not. `--seq-loss` (running-statistics sequence Pearson) fixes most of it: the
per-target quarter WPs stay the same but the pooled score rises ~+0.01, and a
2048-sequence proxy trained with it beats the baseline on identical rows.
Annealing matters as much as data: both fully-annealed proxies beat the
mid-schedule epoch-1 checkpoint of the full-data run. Always compare against
the baseline on the *same* sequences and only judge full runs at the end of
their LR schedule.
The scored rows are a distinct regime: the reference model scores 0.58 on
them and 0.30 on all required rows (or any random 11%). `is_scored` is
partly predictable from the row itself (GBM AUC 0.83; `a3`, `a2`, `a4` carry
most of it) and is sticky (P(scored | previous scored) = 0.77), so a
sequence model should do better — see `src/maskmodel.py`.

## Held-out error analysis (seq_diff_e3, 192 sequences)

*   Per-sequence WP: median 0.715, mean 0.663, std 0.20, min −0.08. The mean is
    dragged by a tail of hard sequences (36 of 192 score < 0.5).
*   Difficulty is regime-driven: per-sequence score correlates −0.68 with the
    sequence mean of `a5`, −0.58 with `a7` (a discrete, 86-valued feature), −0.64
    with the zero-target fraction, +0.55 with the share of |t| > 2 rows. The
    baseline suffers the same way; our gain over it is concentrated in the easy
    sequences (bottom target-std quartile 0.549 vs 0.498, top 0.736 vs 0.681).
*   t0 and t1 per-sequence scores are 0.84 correlated — no target-specific
    failure.
*   Scores fall with position for both models (rows ≥ 10 000: 0.655 vs 0.663
    overall); target statistics near the end are normal, so it is not horizon
    truncation. A wider teacher (GRU 384×2) is only +0.004 at epoch 1, so
    capacity is not the limit either; remaining levers are optimisation length,
    distillation, and the mask weighting.
