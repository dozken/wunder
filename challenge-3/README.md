# Wunder Alpha Connectome (WNN33)

[Challenge site](https://wundernn.io/connectome/docs/quick_start) ·
Sep 11 → Nov 15 2026 (submissions close) → Dec 1 (winners) · $13,600 pool, top 8 · 5 uploads/day

**Status (2026-09-13):** public **0.6221**, rank #11–12 of ~90 scoring entries.
Provided baseline 0.5719 public; top 3 ≥ 0.651, #5 0.643, prize line (#8) 0.627.
Local held-out best **0.6671** (`full_seq_5ep`, epoch 5; `checkpoints/seq_5ep.pt`), projected public ≈ 0.629 — below the top-5 upload gate. See `RESUME.md` for the next runs.

## Task

Predict `t0`, `t1` — two undisclosed future-price-movement indicators of instrument
`i0` — from a streaming sequence of 112 anonymised market-state features covering
two instruments. Inference is a row-by-row callback (`PredictionModel.predict`):
update state on every row, reset on a new `seq_ix`, never look ahead. Score:
weighted Pearson (weights |t|, clipped to ±2) per sequence on the hidden
`is_scored` rows, averaged over sequences and the two targets.

| | |
|---|---|
| Sequences | 20,000 rows; steps 0–98 warm-up, 99–19,999 predicted, ~10% scored |
| Train / valid | 10,607 / 1,873 sequences (212M / 37M rows, ZSTD parquet, one row group per sequence) |
| Runtime limit | 60 min for the whole test set (~37.5M rows) on 1 vCPU, 16 GB, no GPU/network |
| Submission | zip ≤ 20 MB with `solution.py` at the root; python 3.11, `onnxruntime==1.23.2`, `numpy==2.2.6`, `pyarrow==19.0.1` |

## What works (the current recipe)

GRU 192×2 with a 64–128-wide input projection, first-difference inputs, trained
with truncated BPTT (chunk 1000, batch 64 whole sequences, state carried,
gradients cut), AdamW + cosine, 3–5 epochs, EMA/SWA weights, on train + the
non-held-out validation sequences. The loss is what made the difference:

1. **Sequence-level weighted Pearson** (`--seq-loss`): running detached
   sufficient statistics of the sequence prefix, gradient through the current
   chunk. Per-chunk Pearson subtracts the slow component the metric scores, and
   models trained on it get *worse* with more training (0.575 → 0.553).
2. **Loss only on rows likely to be scored** (`--loss-mask soft`): the scored rows
   are a distinct regime (same model: 0.58 on them, 0.30 on all rows). A GRU mask
   predictor trained on the labelled validation file (held-out AUC 0.94) labels
   train.parquet; validation sequences in training use their real mask.

Local → public offset is a stable −0.037 (both for us and the baseline), so the
192-sequence held-out score predicts the leaderboard well. Compare candidates
against the baseline **on identical rows** (`src/diagnose.py`); the held-out
subset is easier than the full validation set (baseline 0.606 vs 0.590).

## What did not

LSTM (worse than GRU at equal cost, even though int8 quantisation would make it
cheaper); chunk-level Pearson; longer chunks; dropping the MSE term; an
auxiliary `is_scored` head; width/depth swaps at equal parameters; peak LR ≥ 1e-3
under the chunk loss; graph micro-optimisation (the batch-1 GRU weight reads are
the floor, not glue ops). A GRU 384×2 teacher is only +0.004 at epoch 1: capacity
is not the bottleneck.

## Inference budget

37.46M `predict` calls in 60 min = 96 µs per call, and the platform's vCPU is
~2× slower than an M-series core (a 35 µs/row-on-Mac model ran in 38–46 min).
Cost is weight-bandwidth bound: ≈ 20 µs + 45 µs per MB of weights on the Mac.
Ceiling ≈ 45 µs/row measured idle on the Mac → GRU 192×2 / proj 128 is about the
limit; GRU 256×2 (57 µs) would time out; ensembles are out. `src/bench.py`
measures any `solution.py`; `mise run docker-test` replicates the container.

## Workflow

```bash
scripts/setup_remote.sh                       # new machine: venv, data, mask labels, sanity checks
python3 src/train.py --tag NAME --rnn gru --hidden 192 --proj 128 --lr 1e-3 --diff \
  --batch 64 --chunk 1000 --epochs 5 --val-seqs 192 --loss-mask soft \
  --soft-mask runs/mask2/train_scored_p.f16 --add-valid --seq-loss --swa
python3 src/diagnose.py runs/NAME/best.pt     # held-out WP, per-quarter, vs baseline on identical rows
scripts/make_submission.sh runs/NAME/best.pt NAME   # slim ONNX export + parity, tests, organisers' scorer, Docker, zip
```

Upload policy: only when the projected public score (held-out − 0.038) would
place top 5. Long runs: `nohup … &` plus `caffeinate -s -i` on the Mac (lid
open — lid-close sleep pauses training).

Other tools: `src/eda.py` (data summary), `src/maskmodel.py train|label` (mask
predictor), `src/teacher_label.py` + `--distill` (distillation), `src/export.py`
(torch exporter, `--int8`), `src/score.py --strict` (organisers' row-by-row path),
`src/test_contract.py` (metric replica vs reference, solution contract).

## Layout

```
challenge-3/
├── README.md, RESUME.md, requirements-train.txt, mise.toml
├── src/            data, metric, loss, model, train, export(_slim), solution, score, diagnose, maskmodel, teacher_label, bench, eda, tests
├── scripts/        setup_remote.sh, fetch_starterpack.sh, make_submission.sh
├── docker/         scorer container replica (datasets bind-mounted)
├── checkpoints/    shipped small checkpoints: mask2.pt, submitted models, teacher
├── submissions/    each upload: solution.py + onnx + zip + NOTES.md with local/public numbers
├── wunder_docs/    challenge docs from the starter pack (+ METRIC.md)
├── runs/           (gitignored) checkpoints, logs, mask labels, teacher labels
└── wnn_connectome_starterpack/   (gitignored, 33.7 GB) datasets, baseline, utils.py
```

## Data notes

*   Features are already rank-transformed and clipped (mass at ±2.32; `a0..a7`
    reach ±5.2); "price-like" columns carry no book geometry, so LOB feature
    engineering does not apply. Some columns have tiny std, so the ±10 clip on
    standardised inputs is load-bearing (values reach 58).
*   Targets: std ≈ 0.95, ~22% exact zeros (zero weight), 3% beyond ±2; `t0`/`t1`
    anti-correlated (−0.74); smooth (autocorr 0.96 at lag 1, ~0 by lag 500).
*   Scoring mask: ~10% of required rows in short sticky runs (P(scored | previous
    scored) = 0.77); `a3`, `a2`, `a4` carry most of its row-level signal.
*   Held-out error analysis (seq_diff_e3): per-sequence WP median 0.715, mean
    0.663, std 0.20; difficulty is regime-driven (score correlates −0.68 with the
    sequence mean of `a5`, −0.58 with `a7`, −0.64 with the zero-target fraction);
    both models degrade with position but the target itself does not change near
    the end.

## Experiment log (held-out 192 unless noted; proxy = 2048 train seqs, 1 epoch)

| run | held-out WP | public | note |
|---|---|---|---|
| provided baseline | 0.6062 | 0.5719 | full-valid 0.5896 |
| GRU 192×2, soft mask v1, chunk loss, full data, epoch 1 (`gru192_e1`) | 0.5955 | 0.5617 | `25KKGXOR`, 45m40s |
| same + train+valid, epochs 1 / 2 | 0.5747 / 0.5527 | — | worsens under the chunk loss; killed |
| oracle: valid-only training, loss on all rows vs is_scored rows (2 ep) | 0.569 vs **0.605** | — | mask matters +0.04 |
| proxy: chunk loss vs **seq-loss** (soft mask) | 0.6011 vs **0.6083** | — | first to beat the baseline on identical rows |
| proxy variants: chunk 2000 / diff / no MSE / lr 7e-4 / dropout 0.2 / 2 epochs / aux head | 0.599 / 0.615 / 0.595 / 0.616 / 0.608 / 0.629 / 0.610 | — | |
| **seq-loss, train+valid, 3 epochs** (`seq_mix`) | 0.6496 / 0.6585 / **0.6603** | 0.6128 (e1) / **0.6221** (e3) | `0ATOIBJ1`, `052UVJK5` |
| + diff, lr 7e-4 (`seq_diff`) | 0.6495 / 0.6591 / **0.6626** | 0.6221 | `211TK6NX`; +0.002 did not transfer |
| proxies on diff + mask v2: lr 7e-4 / lr 1e-3 / proj 128 | 0.623 / 0.627 / 0.628 | — | small positives |
| proxies: mask v2 / 320×1 / 160×3 / 224×2 | 0.611 / 0.611 / 0.608 / 0.610 | — | noise |
| seq-loss + diff, mask v2, 5 epochs (`full_seq_5ep`) | 0.6487 / 0.6597 / 0.6659 / 0.6654 / **0.6671** | — | first-quarter WP rises to 0.70 with training, last quarter stays 0.638 |
| teacher GRU 384×2 proj128 | 0.6546 / 0.6630 / … | — | for distillation |
| mask model v1 → v2 | AUC 0.93 → 0.94, AP 0.66 → 0.69 | — | `checkpoints/mask2.pt` |

## Running on another machine

```bash
git clone git@github.com:dozken/wunder.git && cd wunder/challenge-3
scripts/setup_remote.sh
```

Device auto-detects CUDA → MPS → CPU; on CUDA use `--batch 128+` (cuDNN GRU is
several times faster than MPS). `checkpoints/` lets a new machine relabel,
diagnose, distil or fine-tune without retraining; data and `runs/` never enter git.
