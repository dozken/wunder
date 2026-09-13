# seq_5ep — fifth submission

Checkpoint: `checkpoints/seq_5ep.pt` = `runs/full_seq_5ep/best.pt` (raw weights, epoch 5 of 5).

Recipe: GRU 192×2, proj 64, diff inputs, lr 7e-4 cosine over 5 epochs, batch 64,
chunk 1000, sequence-level Pearson 0.9 + weighted MSE 0.1, soft mask **v2**
(held-out AUC 0.94) on train rows + real mask on the 1,681 non-held-out
validation rows, EMA tracked (raw won). Slim hand-built ONNX graph (18 nodes).

Local numbers:
- held-out 192 validation sequences: **0.6671** (epochs 0.6487 → 0.6597 → 0.6659 → 0.6654 → 0.6671);
  previous upload 0.6626; baseline 0.6062 on the same rows
- quarters 0.703 / 0.672 / 0.654 / 0.638
- organisers' row-by-row scorer, first 10 validation sequences: 0.742 (previous: 0.720)
- Docker 1-CPU replica, machine idle: 40.5 µs/row → same runtime class as the previous uploads (~39 min)
- export parity 1.6e-6; contract tests 5/5

Uploaded knowing it projects to ~0.629 public (below the top-5 gate) — banking the
gain while the day's slots were unused.

Submitted 2026-09-13 20:12 as `00NKMQVT` (submission 3/5 that day).
