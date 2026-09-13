"""Score a model on the validation set.

Fast mode runs a checkpoint through PyTorch in whole-sequence batches and
applies the vectorised metric -- minutes for the full 1,873 sequences:

    python src/score.py --checkpoint runs/gru256/best.pt

Strict mode drives src/solution.py (the ONNX ensemble) through the organisers'
row-by-row ScorerStepByStep on the first N sequences and reports the projected
wall time for the full test set:

    python src/score.py --strict --seqs 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import SEQUENCE_LENGTH, VALID_PATH, BatchStream, SequenceReader  # noqa: E402
from metric import ScoreAccumulator  # noqa: E402

TEST_ROWS = 37_460_000
BUDGET = 3600.0


def fast(checkpoint: Path, device_name: str, batch: int, limit: int) -> dict:
    import torch
    from export import load_checkpoint
    from train import pick_device

    device = pick_device(device_name)
    model, ckpt = load_checkpoint(checkpoint)
    model = model.to(device).eval()
    reader = SequenceReader(VALID_PATH)
    indices = np.arange(len(reader) if not limit else min(limit, len(reader)))
    stream = BatchStream(reader, batch, indices, prefetch=2, drop_last=False)
    stream.rng = np.random.default_rng(0)
    acc = ScoreAccumulator()
    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(stream):
            x = torch.from_numpy(b.features).to(device)
            state = model.initial_state(x.shape[0], device)
            preds = np.zeros(b.targets.shape, dtype=np.float32)
            for t in range(0, SEQUENCE_LENGTH, 4000):
                out, state = model(x[:, t:t + 4000], state)
                preds[:, t:t + 4000] = out[..., :2].float().cpu().numpy()
            acc.add(b.targets, preds, b.scored)
            print(f"  {acc.blocks}/{len(indices)} sequences  running WP {acc.result()['weighted_pearson']:.4f}  {time.time() - t0:.0f}s", flush=True)
    result = acc.result()
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_val_wp"] = ckpt.get("val_wp")
    return result


def strict(limit: int) -> dict:
    sys.path.insert(0, str(HERE.parent / "wnn_connectome_starterpack"))
    from utils import DataPoint, BlockAccumulator, validate_sequence, FEATURE_COLUMNS, TARGET_COLUMNS
    import pyarrow.parquet as pq
    from solution import PredictionModel

    model = PredictionModel()
    parquet = pq.ParquetFile(VALID_PATH)
    acc = BlockAccumulator()
    n = parquet.num_row_groups if not limit else min(limit, parquet.num_row_groups)
    rows = 0
    t0 = time.time()
    for group in range(n):
        table = parquet.read_row_group(group, use_threads=False)
        seq, need = validate_sequence(table)
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        t_seq = time.time()
        for step in range(SEQUENCE_LENGTH):
            value = model.predict(DataPoint(seq, step, bool(need[step]), features[step]))
            if need[step]:
                preds[step] = value
            elif value is not None:
                raise ValueError("returned a value during warm-up")
        rows += SEQUENCE_LENGTH
        acc.add(targets, preds, table["is_scored"].to_numpy().astype(bool) & need)
        print(f"  seq {group + 1}/{n}  {(time.time() - t_seq) / SEQUENCE_LENGTH * 1e6:.1f} us/row  "
              f"running WP {acc.result()['weighted_pearson']:.4f}", flush=True)
    elapsed = time.time() - t0
    result = acc.result()
    per_row = elapsed / rows
    result["us_per_row"] = per_row * 1e6
    result["projected_test_minutes"] = per_row * TEST_ROWS / 60
    result["headroom"] = BUDGET / (per_row * TEST_ROWS)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, help="fast mode: score this .pt")
    ap.add_argument("--strict", action="store_true", help="drive src/solution.py through the organisers' scorer")
    ap.add_argument("--seqs", type=int, default=0, help="limit sequences (0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.strict:
        result = strict(args.seqs)
    elif args.checkpoint:
        result = fast(args.checkpoint, args.device, args.batch, args.seqs)
    else:
        ap.error("pass --checkpoint or --strict")
    for k, v in result.items():
        print(f"{k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
