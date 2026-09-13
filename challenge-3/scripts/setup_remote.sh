#!/bin/bash
# Bootstrap Challenge 3 on a fresh Linux/macOS machine (GPU optional; CUDA or MPS auto-detected).
#   git clone git@github.com:dozken/wunder.git && cd wunder/challenge-3 && scripts/setup_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv env 2>/dev/null || true
. env/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-train.txt
echo "== python $(python3 --version) torch $(python3 -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))')"
if [ ! -f wnn_connectome_starterpack/datasets/train.parquet ]; then
  echo "== fetching the 33.7 GB starter pack (resumable; re-run if interrupted)"
  scripts/fetch_starterpack.sh
fi
mkdir -p runs/mask2
cp checkpoints/mask2.pt runs/mask2/mask.pt
if [ ! -f runs/mask2/train_scored_p.f16 ]; then
  echo "== labelling train.parquet with the mask-v2 predictor (~20 min on CPU, faster on GPU)"
  python3 src/maskmodel.py label --tag mask2 --batch 64
fi
echo "== sanity: contract tests and the held-out diagnostic on the shipped checkpoint"
(cd src && python3 -m pytest test_contract.py -q -k "metric or loss")
python3 src/diagnose.py checkpoints/seq_diff_e3.pt
echo "== ready. See RESUME.md for the next runs."
