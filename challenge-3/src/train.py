"""Truncated-BPTT training over streamed sequences.

Each batch is B whole sequences. The recurrent state is carried across
consecutive chunks of T rows with gradients cut at chunk boundaries, exactly
mirroring the row-by-row stateful inference the scorer performs. One optimiser
step per chunk.

    python src/train.py --tag dev --seqs 512 --epochs 2          # smoke run
    python src/train.py --tag gru256 --hidden 256 --epochs 8      # full data

Outputs go to runs/<tag>/: best.pt (weights + config + feature stats),
log.jsonl, and stdout progress.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (SEQUENCE_LENGTH, STEP_MASK, TRAIN_PATH, VALID_PATH, BatchStream,  # noqa: E402
                  SequenceReader, feature_stats, load_subset)
from loss import CombinedLoss, SequencePearsonLoss, WeightedMSELoss  # noqa: E402
from metric import score_batch  # noqa: E402
from model import ModelConfig, Predictor, count_params  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of weights, evaluated instead of the raw weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


@torch.no_grad()
def evaluate(model: Predictor, features: np.ndarray, targets: np.ndarray, scored: np.ndarray,
             device: torch.device, batch: int = 64, chunk: int = 4000) -> dict:
    """Score whole sequences with the exact per-sequence metric."""
    model.eval()
    preds = np.zeros(targets.shape, dtype=np.float32)
    for i in range(0, len(features), batch):
        x = torch.from_numpy(features[i:i + batch]).to(device)
        state = model.initial_state(x.shape[0], device)
        for t in range(0, SEQUENCE_LENGTH, chunk):
            out, state = model(x[:, t:t + chunk], state)
            preds[i:i + batch, t:t + chunk] = out[..., :2].float().cpu().numpy()
    model.train()
    result = score_batch(targets, preds, scored)
    q = SEQUENCE_LENGTH // 4
    result["quarters"] = [round(score_batch(targets[:, i * q:(i + 1) * q], preds[:, i * q:(i + 1) * q],
                                            scored[:, i * q:(i + 1) * q])["weighted_pearson"], 4) for i in range(4)]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rnn", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--diff", action="store_true", help="feed first differences alongside the raw row")
    ap.add_argument("--batch", type=int, default=128, help="sequences per batch")
    ap.add_argument("--chunk", type=int, default=1000, help="TBPTT length in rows")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pearson-weight", type=float, default=0.9)
    ap.add_argument("--aux-mask", type=float, default=0.0,
                    help="weight of an auxiliary BCE head predicting is_scored (third output channel, dropped at export)")
    ap.add_argument("--seq-loss", action="store_true",
                    help="sequence-level Pearson with running statistics instead of per-chunk Pearson")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seqs", type=int, default=0, help="limit training sequences (0 = all)")
    ap.add_argument("--train-path", default=str(TRAIN_PATH),
                    help="parquet to train on; pass valid.parquet for mask experiments")
    ap.add_argument("--holdout", type=int, default=0,
                    help="when training on valid.parquet: number of its sequences held out for evaluation")
    ap.add_argument("--loss-mask", default="all", choices=["all", "scored", "soft"],
                    help="rows the loss sees: all required rows, the is_scored rows, or the predicted "
                         "P(is_scored) weights from --soft-mask")
    ap.add_argument("--soft-mask", default=None, help="float16 memmap from maskmodel.py label")
    ap.add_argument("--add-valid", action="store_true",
                    help="also train on the validation sequences not held out (their real is_scored mask)")
    ap.add_argument("--val-seqs", type=int, default=192, help="validation subset for the per-epoch check")
    ap.add_argument("--stats-seqs", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=20, help="chunks")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    run_dir = RUNS / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log(**kv):
        kv["time"] = time.time()
        with open(log_path, "a") as f:
            f.write(json.dumps(kv) + "\n")

    train_reader = SequenceReader(args.train_path, soft_mask=args.soft_mask if args.loss_mask == "soft" else None)
    valid_reader = SequenceReader(VALID_PATH)
    rng = np.random.default_rng(args.seed)
    if args.holdout:
        # training on the masked validation file: split it into train / held-out
        perm = rng.permutation(len(train_reader))
        val_idx, train_idx = perm[:args.holdout], perm[args.holdout:]
        val_idx = val_idx[:args.val_seqs]
        valid_reader = train_reader
    else:
        train_idx = np.arange(len(train_reader))
        val_idx = rng.choice(len(valid_reader), size=min(args.val_seqs, len(valid_reader)), replace=False)
    if args.seqs:
        train_idx = rng.choice(train_idx, size=min(args.seqs, len(train_idx)), replace=False)
    if args.loss_mask != "all" and not train_reader.has_mask:
        ap.error(f"--loss-mask {args.loss_mask} needs a masked training file or --soft-mask")
    extra = []
    if args.add_valid:
        if args.holdout:
            ap.error("--add-valid is for training on train.parquet; --holdout already trains on valid")
        rest = np.setdiff1d(np.arange(len(valid_reader)), val_idx)
        extra = [(valid_reader, rest)]
    # a fixed slice of training sequences scored with the exact metric: the
    # train/val gap is the overfitting diagnostic
    fit_idx = train_idx[:min(64, len(train_idx))]

    print(f"device={device} train_seqs={len(train_idx)}{' + valid ' + str(len(extra[0][1])) if extra else ''} "
          f"val_seqs={len(val_idx)}", flush=True)
    t0 = time.time()
    val = load_subset(valid_reader, val_idx, args.workers)
    fit = load_subset(train_reader, fit_idx, args.workers)
    fit_scored = np.broadcast_to(STEP_MASK, fit.targets.shape[:2])
    mean, std = feature_stats(train_reader, args.stats_seqs, args.seed)
    print(f"loaded validation subset + feature stats in {time.time() - t0:.0f}s", flush=True)

    cfg = ModelConfig(rnn=args.rnn, hidden=args.hidden, layers=args.layers, proj=args.proj,
                      dropout=args.dropout, diff=args.diff, out=3 if args.aux_mask > 0 else 2)
    model = Predictor(cfg, mean, std).to(device)
    ema = EMA(model, args.ema)
    print(f"model {cfg.to_dict()} params={count_params(model):,}", flush=True)

    stream = BatchStream(train_reader, args.batch, train_idx, seed=args.seed, workers=args.workers, extra=extra)
    chunks_per_seq = math.ceil(SEQUENCE_LENGTH / args.chunk)
    steps_per_epoch = stream.batches_per_epoch() * chunks_per_seq
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup))

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name.lower() or name.endswith("bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress)))

    criterion = CombinedLoss(args.pearson_weight)
    seq_pearson, mse = SequencePearsonLoss(), WeightedMSELoss()
    step_mask = torch.from_numpy(STEP_MASK).to(device)
    best = -1.0
    step = 0
    print(f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        run_loss, run_n = 0.0, 0
        for b, batch in enumerate(stream):
            x_all = torch.from_numpy(batch.features).to(device, non_blocking=True)
            y_all = torch.from_numpy(batch.targets).to(device, non_blocking=True)
            m_all = (torch.from_numpy(batch.scored).to(device) if args.loss_mask != "all"
                     else step_mask.expand(x_all.shape[0], -1))
            state = model.initial_state(x_all.shape[0], device)
            seq_pearson.reset()
            for t in range(0, SEQUENCE_LENGTH, args.chunk):
                x = x_all[:, t:t + args.chunk]
                y = y_all[:, t:t + args.chunk]
                m = m_all[:, t:t + args.chunk]
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                out, state = model(x, state)
                pred = out[..., :2]
                if args.seq_loss:
                    loss = (args.pearson_weight * seq_pearson.step(pred, y, m)
                            + (1 - args.pearson_weight) * mse(pred, y, m))
                else:
                    loss = criterion(pred, y, m)
                if args.aux_mask > 0:
                    # target: real mask on valid rows, predicted probability on train rows;
                    # only required (post warm-up) rows count
                    req = step_mask[t:t + args.chunk].to(out.dtype).expand(x.shape[0], -1)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[..., 2], m.to(out.dtype), reduction="none")
                    loss = loss + args.aux_mask * (bce * req).sum() / req.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                ema.update(model)
                state = state.detach()
                step += 1
                run_loss += loss.item()
                run_n += 1
                if step % args.log_every == 0:
                    elapsed = time.time() - t_epoch
                    print(f"ep {epoch} batch {b + 1}/{stream.batches_per_epoch()} step {step} "
                          f"loss {run_loss / run_n:.4f} lr {lr_at(step):.2e} {elapsed:.0f}s", flush=True)
                    log(epoch=epoch, step=step, loss=run_loss / run_n, lr=lr_at(step))
                    run_loss, run_n = 0.0, 0

        scores = evaluate(ema.shadow, val.features, val.targets, val.scored, device)
        raw = evaluate(model, val.features, val.targets, val.scored, device)
        fit_raw = evaluate(model, fit.features, fit.targets, fit_scored, device)
        print(f"== epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min: "
              f"val WP ema {scores['weighted_pearson']:.4f} (t0 {scores['t0']:.4f} t1 {scores['t1']:.4f}) "
              f"raw {raw['weighted_pearson']:.4f} quarters {scores['quarters']} | train-subset WP (all rows) {fit_raw['weighted_pearson']:.4f}", flush=True)
        log(epoch=epoch, step=step, val_ema=scores, val_raw=raw, fit_raw=fit_raw)

        use_ema = scores["weighted_pearson"] >= raw["weighted_pearson"]
        current = max(scores["weighted_pearson"], raw["weighted_pearson"])
        source = ema.shadow if use_ema else model
        ckpt = {"config": cfg.to_dict(), "mean": mean, "std": std,
                "state_dict": {k: v.detach().cpu() for k, v in source.state_dict().items()},
                "val_wp": current, "epoch": epoch, "ema": use_ema, "args": vars(args)}
        torch.save(ckpt, run_dir / "last.pt")
        if current > best:
            best = current
            torch.save(ckpt, run_dir / "best.pt")
            print(f"   saved best.pt ({'ema' if use_ema else 'raw'}) WP {best:.4f}", flush=True)

    print(f"best val WP {best:.4f} -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
