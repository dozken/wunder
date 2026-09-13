# Wunder challenges — working notes for Claude

Repo of solutions for Wunder Fund ML competitions. One directory per challenge,
each self-contained (own `mise.toml`, README, Docker scorer). Run commands from
inside the challenge directory.

- `challenge-2/` — Predictorium (finished, LSTM, val WPC 0.2574).
- `challenge-3/` — **Alpha Connectome, active until 2026-11-15.** Read
  `challenge-3/README.md` (recipe, budget, experiment log) and
  `challenge-3/RESUME.md` (next runs) before touching anything.

## Rules that matter in challenge-3

- **Budget first.** 96 µs per `predict` call on a vCPU ~2× slower than this Mac.
  Check any architecture with `src/bench.py` (target ≤ 45 µs/row here) and
  `mise run docker-test` before training it. GRU 192×2 / proj 128 is the ceiling.
- **Compare on identical rows.** Judge candidates with `src/diagnose.py` (held-out
  192 vs the provided baseline on the same sequences), never against the
  baseline's full-validation number. Judge full runs only at the end of their LR
  schedule. 1-epoch proxies cannot resolve differences below ~0.01.
- **Upload policy:** submit only when the projected public score
  (held-out − 0.038) would place top 5. Confirm with the user before an upload;
  5 per day. Archive every upload under `submissions/<date>_<name>/` with NOTES.md.
- **Loss = sequence-level weighted Pearson on soft-masked rows** (`--seq-loss
  --loss-mask soft`). Per-chunk Pearson diverges from the metric.
- Long runs: `nohup python3 src/train.py … &` plus `caffeinate -s -i`; the Mac
  sleeps on lid-close and pauses training. Session restarts kill foreground jobs.
- Never commit data, `runs/`, or large binaries. Small checkpoints go in
  `checkpoints/` (gitignore exception). Keep `README.md` experiment log and
  `RESUME.md` current after each result.
- Memory: 18 GB. One training job plus one proxy is the practical limit on MPS;
  three concurrent jobs slow everything ~2×.

## Conventions

- Python 3.12 here, 3.11 in the scorer; deps pinned in `requirements-train.txt`.
- ONNX export via `src/export_slim.py` (hand-built graph; falls back to
  `src/export.py`); `solution.py` averages every `*.onnx` beside it and uses
  io_binding. Always run `src/test_contract.py` and `src/score.py --strict` before
  packaging.
- Commit messages: conventional prefix (`feat:`, `fix:`, `docs:`), no trailers.
