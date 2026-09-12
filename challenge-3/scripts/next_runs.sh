#!/bin/bash
# Run queue for the Mac, in priority order, sized by the x86 timing matrix in
# README.md (GRU 256x2 8-bit ~65 us/row end to end is the largest single model
# that fits; two 192x2 models fit as an ensemble).
#
#   scripts/next_runs.sh            # runs everything below in sequence
#   scripts/next_runs.sh r1         # just one run
#
# Each run ends with a packaged, timed, contract-tested submission zip in
# submission.zip; move it to submissions/<date>_<name>/ with a NOTES.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ONLY=${1:-all}
run() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

MASK="runs/mask/soft_mask.f16"          # from maskmodel.py label (adjust if named differently)
COMMON="--loss-mask soft --soft-mask $MASK --lr 5e-4 --batch 64 --chunk 1000 --holdout 192 --ema 0.999"

# r1: the same recipe that gave 0.5955 after one epoch, run to completion.
# Epoch 2 was never validated; the dev proxy says 3-4 epochs is where it
# flattens. This is the cheapest expected gain.
if run r1; then
  python3 src/train.py --tag gru192_full --rnn gru --hidden 192 --proj 64 --epochs 4 $COMMON
  scripts/make_submission.sh runs/gru192_full/best.pt gru192_full --unroll nbits8
fi

# r2: the largest single model that fits on x86. On the dev proxy 256x2 beat
# 192x2 by ~0.02-0.04 at equal LR.
if run r2; then
  python3 src/train.py --tag gru256_full --rnn gru --hidden 256 --proj 64 --epochs 4 $COMMON
  scripts/make_submission.sh runs/gru256_full/best.pt gru256_full --unroll nbits8
fi

# r3: second seed of r1 for a 2-model ensemble (2 x 192x2 8-bit ~ 85 us/row
# end to end on x86 -- check the Docker timing before shipping). Ensembling
# two independently seeded runs is usually worth +0.005-0.01 WP here.
if run r3; then
  python3 src/train.py --tag gru192_s1 --rnn gru --hidden 192 --proj 64 --epochs 4 --seed 1 $COMMON
  # package both graphs into one zip: export each, unroll each, put both .onnx in src/
  rm -f src/*.onnx
  python3 src/export.py runs/gru192_full/best.pt src/a.onnx
  python3 src/export.py runs/gru192_s1/best.pt src/b.onnx
  python3 src/unroll.py src/a.onnx src/a_nbits8.onnx --quant nbits8
  python3 src/unroll.py src/b.onnx src/b_nbits8.onnx --quant nbits8
  rm -f src/a.onnx src/b.onnx src/*.fp32.onnx
  python3 src/score.py --strict --seqs 20
  mise run docker-test
  mise run submit
fi

# r4: train + validation in one stream for the final candidate (no held-out
# check possible, so only after r1/r2 have fixed the epoch count).
if run r4; then
  python3 src/train.py --tag gru256_trainvalid --rnn gru --hidden 256 --proj 64 --epochs 4 --add-valid \
    --loss-mask soft --soft-mask $MASK --lr 5e-4 --batch 64 --chunk 1000 --ema 0.999
  scripts/make_submission.sh runs/gru256_trainvalid/best.pt gru256_trainvalid --unroll nbits8 --strict-seqs 5
fi
