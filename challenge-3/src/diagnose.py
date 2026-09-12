"""Drift diagnostic on the held-out validation subset.

Scores a checkpoint on the same 192 sequences train.py holds out (seed 0) and
prints whole-sequence WP next to WP per quarter of the sequence, alongside the
provided baseline on the identical rows. A model that wins every quarter but
loses on whole sequences has a drifting prediction level.

    python src/diagnose.py runs/<tag>/best.pt [more checkpoints...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import SEQUENCE_LENGTH, STARTERPACK, VALID_PATH, SequenceReader, load_subset  # noqa: E402
from export import load_checkpoint  # noqa: E402
from metric import score_batch  # noqa: E402
from train import pick_device  # noqa: E402

CACHE = HERE.parent / "runs" / "heldout192"


def heldout():
    reader = SequenceReader(VALID_PATH)
    idx = np.random.default_rng(0).choice(len(reader), size=192, replace=False)   # train.py's draw with --seqs 0
    return load_subset(reader, idx, workers=4)


def baseline_preds(v) -> np.ndarray:
    """Provided GRU baseline, row by row through onnxruntime; cached on disk."""
    path = CACHE / "baseline_preds.npy"
    if path.exists():
        return np.load(path)
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    s = ort.InferenceSession(str(STARTERPACK / "baseline" / "baseline.onnx"), opts, providers=["CPUExecutionProvider"])
    preds = np.zeros(v.targets.shape, np.float32)
    t0 = time.time()
    for b in range(len(v.features)):
        h0 = np.zeros((1, 1, 128), np.float32); h1 = np.zeros((1, 1, 128), np.float32)
        x = v.features[b]
        for t in range(SEQUENCE_LENGTH):
            p, h0, h1 = s.run(None, {"features": x[t].reshape(1, 1, 112), "hidden_0": h0, "hidden_1": h1})
            preds[b, t] = p[0, 0]
        if b % 32 == 31:
            print(f"  baseline {b + 1}/192 {time.time() - t0:.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, preds)
    return preds


@torch.no_grad()
def model_preds(ckpt: Path, v, device) -> np.ndarray:
    model, _ = load_checkpoint(ckpt)
    model = model.to(device).eval()
    preds = np.zeros(v.targets.shape, np.float32)
    for i in range(0, len(v.features), 32):
        x = torch.from_numpy(v.features[i:i + 32]).to(device)
        state = model.initial_state(x.shape[0], device)
        for t in range(0, SEQUENCE_LENGTH, 4000):
            out, state = model(x[:, t:t + 4000], state)
            preds[i:i + 32, t:t + 4000] = out.float().cpu().numpy()
    return preds


def report(name: str, preds: np.ndarray, v) -> None:
    full = score_batch(v.targets, preds, v.scored)
    q = SEQUENCE_LENGTH // 4
    quarters = [score_batch(v.targets[:, i * q:(i + 1) * q], preds[:, i * q:(i + 1) * q],
                            v.scored[:, i * q:(i + 1) * q])["weighted_pearson"] for i in range(4)]
    # level drift: per-sequence weighted mean of predictions in the last vs first quarter, in units of pred std
    w = np.abs(np.clip(v.targets, -2, 2)) * v.scored[..., None]
    def wmean(a, sl):
        ww = w[:, sl]; return (ww * a[:, sl]).sum(1) / np.maximum(ww.sum(1), 1e-8)
    drift = np.abs(wmean(preds, slice(3 * q, None)) - wmean(preds, slice(0, q))).mean() / (preds[:, 99:].std() + 1e-8)
    print(f"{name:32s} full {full['weighted_pearson']:.4f} (t0 {full['t0']:.4f} t1 {full['t1']:.4f})  "
          f"quarters {' '.join(f'{x:.4f}' for x in quarters)}  level-drift {drift:.3f}")


def main() -> int:
    v = heldout()
    device = pick_device("auto")
    report("provided baseline", baseline_preds(v), v)
    for arg in sys.argv[1:]:
        report(arg, model_preds(Path(arg), v, device), v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
