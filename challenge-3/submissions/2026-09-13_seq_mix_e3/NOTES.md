# seq_mix_e3 — third submission

Checkpoint: `runs/full_seq_mix/epoch3.pt` (raw weights at the end of the
3-epoch cosine schedule; EMA was 0.6600, raw 0.6603).

Recipe: identical to `2026-09-12_seq_mix_e1` (GRU 192×2, proj 64, lr 5e-4
cosine, batch 64, chunk 1000, sequence-level Pearson 0.9 + weighted MSE 0.1,
predicted soft mask on train rows + real mask on the 1,681 non-held-out
validation rows), simply trained to the end of the schedule.

Local numbers:
- held-out 192 validation sequences: **0.6603** (epochs: 0.6496 → 0.6585 → 0.6603);
  baseline 0.6062 on the same rows
- quarters 0.673 / 0.666 / 0.657 / 0.640
- organisers' row-by-row scorer, first 10 validation sequences: 0.719 (e1: 0.698)
- export parity 7e-7; contract tests 5/5; same graph size as e1 (38m36s on the platform)

Submitted 2026-09-13 07:15 as `052UVJK5` (submission 1/5 that day).

Public score **0.6221** (#11 at the time), runtime 38m19s.
