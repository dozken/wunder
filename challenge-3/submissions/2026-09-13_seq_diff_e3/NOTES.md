# seq_diff_e3 — fourth submission

Checkpoint: `runs/full_seq_diff/epoch3.pt` (raw weights, end of a 3-epoch cosine schedule).

Recipe: as `2026-09-13_seq_mix_e3` plus **first-difference inputs** (x_t − x_{t−1},
previous standardised row carried in the state) and lr 7e-4. Exported with the
slim hand-built graph (18 nodes, diff handled by one Sub + Concat).

Local numbers:
- held-out 192 validation sequences: **0.6626** (epochs 0.6495 → 0.6591 → 0.6626); previous best 0.6603
- quarters 0.674 / 0.668 / 0.656 / 0.644
- organisers' row-by-row scorer, first 10 validation sequences: 0.720 (e3: 0.719)
- export parity 9e-7; contract tests 5/5; graph cost same class as e3

Submitted 2026-09-13 09:38 as `211TK6NX` (submission 2/5 that day).

Public score **0.6221** (same as seq_mix_e3), runtime 39m29s.
