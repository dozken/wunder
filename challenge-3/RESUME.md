# Resume notes (paused 2026-09-13 ~19:10)

State when paused: `full_seq_5ep` finished (see below); `teacher384` finished at ~20:00 and is
shipped as `checkpoints/teacher384.pt`. Their results land in `runs/full_seq_5ep.out`
and `runs/teacher384.out` (`grep "^=="`). Best held-out so far: **0.6671** (`full_seq_5ep`, epoch 5, shipped as
`checkpoints/seq_5ep.pt`; quarters 0.70/0.67/0.65/0.64 — late-sequence rows do not improve
with more epochs, an open lead). Upload gate: projected public top-5 → held-out ≥ ~0.681
(#5 was 0.6431; local→public offset ≈ −0.038).

Teacher result: 0.6645 raw / 0.6652 SWA (3 epochs) — barely above the 192, so
distillation (step 2) is a small bet; the open lead is the late-sequence gap
(quarters 0.70 → 0.64 on every model). Ideas not yet tried: state re-centering /
running feature normalisation as explicit inputs, longer effective context
(chunk 2000–4000 now that the loss is sequence-level and the LR is settled),
per-sequence adaptive loss weighting for the hard regimes (`a5`, `a7`).

Next steps, in priority order (each ~4–7 h on the Mac; run one at a time with
`caffeinate -s -i` and the lid open):

1. Combined recipe, 5 epochs + SWA:
   python3 src/train.py --tag full_v3 --rnn gru --hidden 192 --layers 2 --proj 128 --lr 1e-3 --diff \
     --batch 64 --chunk 1000 --epochs 5 --val-seqs 192 --log-every 100 \
     --loss-mask soft --soft-mask runs/mask2/train_scored_p.f16 --add-valid --seq-loss --swa > runs/full_v3.out 2>&1

2. Distillation from the GRU-384 teacher (needs runs/teacher384/best.pt):
   python3 src/teacher_label.py runs/teacher384/best.pt runs/teacher384
   python3 src/train.py --tag student_distill --rnn gru --hidden 192 --layers 2 --proj 128 --lr 1e-3 --diff \
     --batch 64 --chunk 1000 --epochs 5 --val-seqs 192 --log-every 100 \
     --loss-mask soft --soft-mask runs/mask2/train_scored_p.f16 --add-valid --seq-loss --swa \
     --distill 0.5 --teacher-train runs/teacher384/teacher_train.f16 --teacher-valid runs/teacher384/teacher_valid.f16 \
     > runs/student_distill.out 2>&1

3. For any candidate: `python3 src/diagnose.py runs/<tag>/best.pt` (held-out vs baseline on identical rows),
   then `scripts/make_submission.sh runs/<tag>/best.pt <name>` and upload only if held-out ≥ 0.681.

Submissions so far: 0.5617 → 0.6128 → 0.6221 → 0.6221 (public), rank #11–12.
