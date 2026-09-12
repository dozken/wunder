"""Export a checkpoint to ONNX and verify it against the PyTorch model.

    python src/export.py runs/gru256/best.pt src/model_a.onnx

The graph takes one row and the recurrent state and returns the prediction and
the next state, which is what solution.py drives row by row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build  # noqa: E402

INPUT_NAMES = ["features", "state"]
OUTPUT_NAMES = ["prediction", "next_state"]


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ckpt["config"], ckpt["mean"], ckpt["std"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic int8 weights. ONNX Runtime has a quantised LSTM kernel (about 2x
    faster than fp32 at batch 1) but no quantised GRU, so this only pays for
    LSTM models. Check the WP impact with score.py --strict before shipping."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def export(model, out_path: Path) -> None:
    x = torch.zeros(1, 1, 112)
    state = model.initial_state(1)
    torch.onnx.export(
        model, (x, state), str(out_path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    import onnx
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    onnx.save(m, str(out_path), save_as_external_data=False)
    leftover = out_path.with_suffix(out_path.suffix + ".data")
    if leftover.exists():
        leftover.unlink()


@torch.no_grad()
def verify(model, onnx_path: Path, rows: int = 2000, seed: int = 0) -> float:
    """Max abs difference between torch (whole sequence) and ORT (row by row)."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, rows, 112)).astype(np.float32) * 3
    ref, _ = model(torch.from_numpy(x), model.initial_state(1))
    ref = ref.numpy()[0]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = np.zeros(model.initial_state(1).shape, dtype=np.float32)
    out = np.zeros((rows, 2), dtype=np.float32)
    for t in range(rows):
        pred, state = sess.run(OUTPUT_NAMES, {"features": x[:, t:t + 1], "state": state})
        out[t] = pred[0, 0]
    return float(np.abs(out - ref).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--int8", action="store_true", help="also write <out>.int8.onnx with dynamic int8 weights")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    export(model, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.2f} MB) config={ckpt['config']} val_wp={ckpt.get('val_wp')}")
    if not args.no_verify:
        diff = verify(model, args.out)
        print(f"torch vs onnxruntime max abs diff over 2000 rows: {diff:.2e}")
        if diff > 1e-3:
            print("MISMATCH: exported graph does not reproduce the torch model", file=sys.stderr)
            return 1
    if args.int8:
        q_path = args.out.with_suffix(".int8.onnx")
        quantize_int8(args.out, q_path)
        print(f"quantised {q_path} ({q_path.stat().st_size / 1e6:.2f} MB)")
        if not args.no_verify:
            diff = verify(model, q_path)
            print(f"torch vs int8 onnxruntime max abs diff over 2000 rows: {diff:.2e} "
                  f"(expected ~1e-2; confirm WP with score.py --strict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
