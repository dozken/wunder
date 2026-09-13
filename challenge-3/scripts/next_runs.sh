#!/bin/bash
# Run queue for the Mac, in priority order. Public-leaderboard context (09-13):
# #1 is 0.671, the prize line (#8) ~0.627; our seq_mix_e1 is 0.6128 public from
# 0.6496 held-out, so held-out -> public is about -0.037. Public 0.69 means
# held-out ~0.725.
#
# Sizing (README "Sizing for the platform"): unrolled + dynamic-int8 graphs
# project to ~52 us/row for GRU 192x2, ~56 for 256x2, ~67 for 320x2 on the
# platform (budget 96); a 2-model 192x2 ensemble is ~92 and too tight. Always
# package with --unroll dynamic and confirm the Docker timing before uploading.
#
#   scripts/next_runs.sh            # everything in sequence
#   scripts/next_runs.sh r2         # one stage
set -euo pipefail
cd "$(dirname "$0")/.."
ONLY=${1:-all}
run() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

MASK="runs/mask/train_scored_p.f16"     # written by: python src/maskmodel.py label --tag mask
# the seq_mix_e1 recipe (held-out 0.6496 mid-schedule): seq-loss, soft mask,
# train + non-held-out valid, chunk 1000, 0.1 MSE term. lr 7e-4 and --diff were
# each worth ~+0.007 on the proxy.
COMMON="--seq-loss --loss-mask soft --soft-mask $MASK --add-valid --holdout 192 --lr 7e-4 --batch 64 --chunk 1000 --ema 0.999"

# r1: let the running full seq-mix run finish its schedule (epochs 2-3) and
# package the final checkpoint; the 2-epoch proxy gained +0.02 over 1 epoch.
if run r1; then
  scripts/make_submission.sh runs/full_seq_mix/best.pt seq_mix_e3 --unroll dynamic
fi

# r2: capacity. GRU 256x2 was +0.02-0.04 over 192x2 on the proxy and now fits
# the budget as an unrolled dynamic-int8 graph (it did not as fused fp32).
# 320x2 (~67 us projected) is the next step if 256x2 lands with margin.
if run r2; then
  python3 src/train.py --tag seq_gru256 --rnn gru --hidden 256 --proj 64 --epochs 4 --diff $COMMON
  scripts/make_submission.sh runs/seq_gru256/best.pt seq_gru256 --unroll dynamic
fi

# r3: a second seed of the best 192x2 recipe for a 2-model ensemble. Projected
# ~92 us/row on the platform: only upload if the Docker replica shows < 45 us
# (the replica is ~2x faster than the platform).
if run r3; then
  python3 src/train.py --tag seq_gru192_s1 --rnn gru --hidden 192 --proj 64 --epochs 4 --diff --seed 1 $COMMON
  rm -f src/*.onnx
  python3 src/export_slim.py runs/full_seq_mix/best.pt src/a.onnx
  python3 src/export_slim.py runs/seq_gru192_s1/best.pt src/b.onnx
  python3 src/unroll.py src/a.onnx src/a_dyn.onnx --quant dynamic
  python3 src/unroll.py src/b.onnx src/b_dyn.onnx --quant dynamic
  rm -f src/a.onnx src/b.onnx src/*.fp32.onnx
  (cd src && python3 -m pytest test_contract.py -q | tail -1)
  python3 src/score.py --strict --seqs 20
  mise run docker-test
  mise run submit
fi

# r4: longer schedule on the winner of r1/r2 (8 epochs, lr 5e-4): the proxies
# say annealing and epochs matter as much as data.
if run r4; then
  python3 src/train.py --tag seq_gru256_long --rnn gru --hidden 256 --proj 64 --epochs 8 --diff \
    --seq-loss --loss-mask soft --soft-mask $MASK --add-valid --holdout 192 --lr 5e-4 --batch 64 --chunk 1000 --ema 0.999
  scripts/make_submission.sh runs/seq_gru256_long/best.pt seq_gru256_long --unroll dynamic
fi
