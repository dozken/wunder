"""Predict P(is_scored) for every row so the training loss can be focused on
the rows the metric will count.

Only valid.parquet carries the real mask, so the predictor is trained there
(with a held-out split for AUC) and then used to label train.parquet.

    python src/maskmodel.py train --tag mask --epochs 3
    python src/maskmodel.py label --tag mask            # -> runs/mask/train_scored_p.f16
    python src/train.py ... --loss-mask soft --soft-mask runs/mask/train_scored_p.f16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (SEQUENCE_LENGTH, STEP_MASK, TRAIN_PATH, VALID_PATH, BatchStream,  # noqa: E402
                  SequenceReader, feature_stats, load_subset)
from model import ModelConfig, Predictor  # noqa: E402
from train import RUNS, pick_device  # noqa: E402


@torch.no_grad()
def predict_probs(model, features: np.ndarray, device, batch=64, chunk=4000) -> np.ndarray:
    model.eval()
    out = np.zeros(features.shape[:2], dtype=np.float32)
    for i in range(0, len(features), batch):
        x = torch.from_numpy(features[i:i + batch]).to(device)
        state = model.initial_state(x.shape[0], device)
        for t in range(0, SEQUENCE_LENGTH, chunk):
            logit, state = model(x[:, t:t + chunk], state)
            out[i:i + batch, t:t + chunk] = torch.sigmoid(logit[..., 0]).float().cpu().numpy()
    model.train()
    return out


def auc_ap(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def train(args):
    device = pick_device(args.device)
    run_dir = RUNS / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    reader = SequenceReader(VALID_PATH)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(reader))
    held, train_idx = perm[:args.holdout], perm[args.holdout:]
    held_batch = load_subset(reader, held[:args.val_seqs])
    mean, std = feature_stats(reader, 64, args.seed)
    cfg = ModelConfig(rnn="gru", hidden=args.hidden, layers=args.layers, proj=args.proj, dropout=0.1, out=1)
    model = Predictor(cfg, mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    stream = BatchStream(reader, args.batch, train_idx, seed=args.seed)
    step_mask = torch.from_numpy(STEP_MASK).to(device)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    total = stream.batches_per_epoch() * (SEQUENCE_LENGTH // args.chunk) * args.epochs
    step, best = 0, -1.0
    print(f"mask model {cfg.to_dict()} train {len(train_idx)} held {len(held)} steps {total}", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for batch in stream:
            x_all = torch.from_numpy(batch.features).to(device)
            y_all = torch.from_numpy(batch.scored.astype(np.float32)).to(device)
            state = model.initial_state(x_all.shape[0], device)
            for t in range(0, SEQUENCE_LENGTH, args.chunk):
                lr = args.lr * 0.5 * (1 + np.cos(np.pi * step / total))
                for g in opt.param_groups:
                    g["lr"] = lr
                logit, state = model(x_all[:, t:t + args.chunk], state)
                m = step_mask[t:t + args.chunk].expand(x_all.shape[0], -1).float()
                loss = (F.binary_cross_entropy_with_logits(logit[..., 0], y_all[:, t:t + args.chunk],
                                                           pos_weight=pos_weight, reduction="none") * m).sum() / m.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                state = state.detach()
                step += 1
                if step % 50 == 0:
                    print(f"ep {epoch} step {step}/{total} bce {loss.item():.4f} lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
        p = predict_probs(model, held_batch.features, device)
        y = held_batch.scored
        auc, ap = auc_ap(y[:, 99:].ravel(), p[:, 99:].ravel())
        print(f"== epoch {epoch}: held-out AUC {auc:.4f} AP {ap:.4f} (base rate {y[:, 99:].mean():.3f})", flush=True)
        if auc > best:
            best = auc
            torch.save({"config": cfg.to_dict(), "mean": mean, "std": std,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "auc": auc, "ap": ap}, run_dir / "mask.pt")
    print(f"best held-out AUC {best:.4f} -> {run_dir / 'mask.pt'}")


def label(args):
    device = pick_device(args.device)
    run_dir = RUNS / args.tag
    ckpt = torch.load(run_dir / "mask.pt", map_location="cpu", weights_only=False)
    model = Predictor(ModelConfig(**ckpt["config"]), ckpt["mean"], ckpt["std"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    reader = SequenceReader(TRAIN_PATH)
    out_path = run_dir / "train_scored_p.f16"
    out = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(len(reader), SEQUENCE_LENGTH))
    t0 = time.time()
    for start in range(0, len(reader), args.batch):
        idx = list(range(start, min(start + args.batch, len(reader))))
        batch = load_subset(reader, idx)
        out[start:start + len(idx)] = predict_probs(model, batch.features, device).astype(np.float16)
        if (start // args.batch) % 10 == 0:
            print(f"labelled {start + len(idx)}/{len(reader)} {time.time() - t0:.0f}s", flush=True)
    out.flush()
    print(f"wrote {out_path} mean p {float(np.asarray(out[:, 99:], dtype=np.float32).mean()):.4f}")
    json.dump({"source": str(ckpt.get("auc")), "rows": len(reader)}, open(run_dir / "label.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "label"])
    ap.add_argument("--tag", default="mask")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--proj", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--holdout", type=int, default=373)
    ap.add_argument("--val-seqs", type=int, default=192)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"train": train, "label": label}[args.cmd](args)


if __name__ == "__main__":
    main()
