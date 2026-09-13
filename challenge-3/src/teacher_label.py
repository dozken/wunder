"""Write a teacher's predictions for every row of train.parquet and valid.parquet
as float16 memmaps, for distillation into a smaller student (train.py --distill).

    python src/teacher_label.py runs/<teacher>/best.pt runs/<teacher>
    -> runs/<teacher>/teacher_train.f16 (N_train, 20000, 2), teacher_valid.f16 (N_valid, 20000, 2)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SEQUENCE_LENGTH, TRAIN_PATH, VALID_PATH, SequenceReader, load_subset  # noqa: E402
from export import load_checkpoint  # noqa: E402
from train import pick_device  # noqa: E402


@torch.no_grad()
def label(model, reader, out_path: Path, device, batch: int = 64) -> None:
    out = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(len(reader), SEQUENCE_LENGTH, 2))
    t0 = time.time()
    for start in range(0, len(reader), batch):
        idx = list(range(start, min(start + batch, len(reader))))
        b = load_subset(reader, idx, workers=8)
        x = torch.from_numpy(b.features).to(device)
        state = model.initial_state(x.shape[0], device)
        preds = np.zeros((len(idx), SEQUENCE_LENGTH, 2), np.float32)
        for t in range(0, SEQUENCE_LENGTH, 4000):
            o, state = model(x[:, t:t + 4000], state)
            preds[:, t:t + 4000] = o[..., :2].float().cpu().numpy()
        out[start:start + len(idx)] = preds.astype(np.float16)
        if (start // batch) % 20 == 0:
            print(f"  {out_path.name}: {start + len(idx)}/{len(reader)} {time.time() - t0:.0f}s", flush=True)
    out.flush()


def main() -> int:
    ckpt, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    device = pick_device("auto")
    model, meta = load_checkpoint(ckpt)
    model = model.to(device).eval()
    print(f"teacher {meta['config']} val_wp={meta.get('val_wp')}")
    label(model, SequenceReader(VALID_PATH), out_dir / "teacher_valid.f16", device)
    label(model, SequenceReader(TRAIN_PATH), out_dir / "teacher_train.f16", device)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
