#!/bin/bash
# One-shot submission prep:
#   scripts/make_submission.sh runs/<tag>/best.pt <name> [--int8] [--strict-seqs N] [--docker-seqs N]
# Exports the checkpoint to src/<name>.onnx (removing any other .onnx in src/),
# runs the contract tests, the organisers' row-by-row scorer on a few
# sequences, the 1-CPU Docker timing, and zips submission.zip.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=$1; NAME=$2; shift 2
INT8=""; STRICT=20; DOCKER=10
while [ $# -gt 0 ]; do
  case $1 in
    --int8) INT8="--int8";;
    --strict-seqs) STRICT=$2; shift;;
    --docker-seqs) DOCKER=$2; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac; shift
done

echo "== export"
rm -f src/*.onnx
if [ -z "$INT8" ] && python3 src/export_slim.py "$CKPT" "src/$NAME.onnx" 2>&1 | grep -v -i warning; then
  echo "(hand-built slim graph)"
else
  python3 src/export.py "$CKPT" "src/$NAME.onnx" $INT8 2>&1 | grep -v -i -E "warning|torch.onnx|_generic_rnn"
  if [ -n "$INT8" ]; then rm -f "src/$NAME.onnx"; fi     # keep only the int8 graph
fi
ls -la src/*.onnx

echo "== contract tests"
(cd src && python3 -m pytest test_contract.py -q 2>&1 | tail -1)

echo "== organisers' scorer path, $STRICT validation sequences"
python3 src/score.py --strict --seqs "$STRICT" 2>&1 | grep -E "^(weighted_pearson|t0|t1|us_per_row|projected|headroom|eligible)"

echo "== docker timing (1 CPU, no network), $DOCKER sequences"
docker build -q -t wnn-connectome-scorer -f docker/Dockerfile . > /dev/null
docker run --rm --cpus="1" --memory="16g" --network=none -e SEQS="$DOCKER" \
  -v "$(pwd)/src:/app/src:ro" -v "$(pwd)/wnn_connectome_starterpack/datasets:/app/datasets:ro" \
  wnn-connectome-scorer 2>&1 | tail -3

echo "== zip"
rm -f submission.zip
(cd src && zip -q ../submission.zip solution.py *.onnx)
unzip -l submission.zip
du -h submission.zip
