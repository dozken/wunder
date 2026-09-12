# gru192_e1 — first submission candidate

Checkpoint: `runs/full_gru192/epoch1.pt` (EMA weights after epoch 1 of the
first full-data run; the process was killed at step 5900/6600 of epoch 2 when
the session restarted, so epoch 2 never validated).

Recipe: GRU 192×2, proj 64, lr 5e-4 cosine, batch 64, TBPTT chunk 1000,
loss = 0.9·per-chunk weighted Pearson + 0.1·weighted MSE, loss rows weighted
by the predicted P(is_scored) soft mask (`runs/mask/`), EMA 0.999.

Local numbers:
- held-out 192 validation sequences (exact metric): **0.5955** (raw weights 0.552)
- organisers' row-by-row scorer, first 10 validation sequences: 0.626
- Docker 1-CPU replica: 46.7 µs/row while the GPU was busy → 29 min projected, 2.06× headroom
- export parity torch vs onnxruntime: 6e-7; contract tests 5/5

Provided baseline for reference: 0.5896 on the full validation set.
