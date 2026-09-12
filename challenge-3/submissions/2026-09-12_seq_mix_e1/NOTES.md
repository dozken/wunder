# seq_mix_e1 — second submission

Checkpoint: `runs/full_seq_mix/epoch1.pt` (EMA after epoch 1 of 3; epochs 2–3
were still training when this was packaged).

Recipe: GRU 192×2, proj 64, lr 5e-4 cosine, batch 64, TBPTT chunk 1000,
**sequence-level running-statistics Pearson loss** (0.9) + weighted MSE (0.1),
loss rows weighted by the predicted P(is_scored) soft mask, trained on
train.parquet + the 1,681 validation sequences not held out (real mask),
EMA 0.999. Exported with `src/export_slim.py` (hand-built 15-node graph);
`solution.py` uses io_binding.

Local numbers:
- held-out 192 validation sequences (exact metric): **0.6496** (raw weights 0.6481);
  provided baseline on the same rows: 0.6062; previous submission: 0.5955
- organisers' row-by-row scorer, first 10 validation sequences: 0.698 (previous: 0.626)
- Docker 1-CPU replica under GPU load: 46 µs/row → ~29 min projected here; the
  platform ran the previous, same-size model in 45m40s
- export parity torch vs onnxruntime: 7e-7; contract tests 5/5

Submitted 2026-09-12 22:47 as `0ATOIBJ1` (submission 2/5 that day).
