"""First look at the data: feature scales, book ordering, target behaviour,
scoring mask structure. Reads a random sample of sequences from train and valid.

    python src/eda.py --seqs 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (FEATURE_COLUMNS, SEQUENCE_LENGTH, TRAIN_PATH, VALID_PATH, WARMUP,  # noqa: E402
                  SequenceReader, load_subset)

np.set_printoptions(precision=3, suppress=True, linewidth=160)


def group_slices():
    """Feature index ranges per group, from the data overview table."""
    out = {}
    base = 0
    for inst in ("i0", "i1"):
        for name, n in (("bid_p", 11), ("ask_p", 11), ("bid_v", 11), ("ask_v", 11), ("dp", 4), ("dv", 4)):
            out[f"{inst}_{name}"] = slice(base, base + n)
            base += n
    out["a"] = slice(base, base + 8)
    return out


def describe_features(x: np.ndarray):
    """x: (N, 112)."""
    G = group_slices()
    print("\n== feature scale per group (over all sampled rows) ==")
    print(f"{'group':10s} {'mean':>10s} {'std':>10s} {'min':>10s} {'p1':>10s} {'p50':>10s} {'p99':>10s} {'max':>10s} {'zero%':>6s} {'>5sd%':>6s}")
    for g, sl in G.items():
        v = x[:, sl].ravel()
        mu, sd = v.mean(), v.std()
        q = np.quantile(v, [0.01, 0.5, 0.99])
        print(f"{g:10s} {mu:10.3f} {sd:10.3f} {v.min():10.3f} {q[0]:10.3f} {q[1]:10.3f} {q[2]:10.3f} {v.max():10.3f} "
              f"{(v == 0).mean() * 100:6.2f} {(np.abs(v - mu) > 5 * sd).mean() * 100:6.2f}")

    print("\n== per-column std for i0 price-like and a0..a7 ==")
    for name in ("i0_bid_p", "i0_ask_p", "a"):
        sl = G[name]
        print(f"{name:10s}", x[:, sl].std(axis=0))

    print("\n== book ordering: is p[j] monotone across j within a row? ==")
    for name in ("i0_bid_p", "i0_ask_p", "i1_bid_p", "i1_ask_p", "i0_bid_v", "i0_ask_v"):
        v = x[:, G[name]]
        d = np.diff(v, axis=1)
        print(f"{name:10s} rows with all diffs >=0: {(d >= 0).all(axis=1).mean() * 100:5.1f}%   "
              f"all diffs <=0: {(d <= 0).all(axis=1).mean() * 100:5.1f}%   "
              f"mean |diff| {np.abs(d).mean():.4f}")

    bid, ask = x[:, G["i0_bid_p"]], x[:, G["i0_ask_p"]]
    print("\n== i0 ask - bid by index (mean, % positive) ==")
    d = ask - bid
    print("  mean", d.mean(axis=0))
    print("  pos%", (d > 0).mean(axis=0) * 100)

    print("\n== correlations of matching i0 / i1 columns (same group index) ==")
    for a, b in (("i0_bid_p", "i1_bid_p"), ("i0_ask_p", "i1_ask_p"), ("i0_bid_v", "i1_bid_v"), ("i0_dp", "i1_dp")):
        va, vb = x[:, G[a]], x[:, G[b]]
        cors = [np.corrcoef(va[:, j], vb[:, j])[0, 1] for j in range(va.shape[1])]
        print(f"  {a} vs {b}:", np.array(cors))


def describe_targets(t: np.ndarray, f: np.ndarray, scored: np.ndarray | None):
    """t: (B, T, 2); f: (B, T, 112)."""
    print("\n== targets ==")
    flat = t.reshape(-1, 2)
    for k in range(2):
        v = flat[:, k]
        q = np.quantile(v, [0.001, 0.01, 0.5, 0.99, 0.999])
        print(f"t{k}: mean {v.mean():+.4f} std {v.std():.4f} zero% {(v == 0).mean() * 100:.2f} "
              f"|t|>2: {(np.abs(v) > 2).mean() * 100:.3f}%  q[.001,.01,.5,.99,.999] {q}")
    print(f"corr(t0, t1) = {np.corrcoef(flat[:, 0], flat[:, 1])[0, 1]:.4f}")

    print("\n== target autocorrelation within sequence (mean over sequences) ==")
    for lag in (1, 5, 20, 100, 500):
        ac = []
        for b in range(t.shape[0]):
            for k in range(2):
                a, c = t[b, WARMUP:-lag, k], t[b, WARMUP + lag:, k]
                ac.append(np.corrcoef(a, c)[0, 1])
        print(f"  lag {lag:4d}: t0 {np.mean(ac[0::2]):+.3f}  t1 {np.mean(ac[1::2]):+.3f}")

    print("\n== per-sequence target scale (std of t0 per sequence) ==")
    s = t[:, WARMUP:, 0].std(axis=1)
    print(f"  min {s.min():.4f} median {np.median(s):.4f} max {s.max():.4f}")

    G = group_slices()
    mid = (f[..., G["i0_bid_p"]].mean(-1) + f[..., G["i0_ask_p"]].mean(-1)) / 2   # crude proxy
    print("\n== corr(target, future change of crude i0 mid proxy over horizon h) ==")
    for h in (1, 5, 20, 100, 500):
        cs = []
        for b in range(t.shape[0]):
            dm = mid[b, WARMUP + h:] - mid[b, WARMUP:-h]
            for k in range(2):
                cs.append(np.corrcoef(t[b, WARMUP:-h, k], dm)[0, 1])
        print(f"  h {h:4d}: t0 {np.nanmean(cs[0::2]):+.3f}  t1 {np.nanmean(cs[1::2]):+.3f}")
    print("== corr(target, PAST change of mid proxy) — is the target partly lagged? ==")
    for h in (1, 20, 100):
        cs = []
        for b in range(t.shape[0]):
            dm = mid[b, WARMUP:-h] - mid[b, WARMUP - h:-2 * h] if WARMUP - h >= 0 else None
            for k in range(2):
                cs.append(np.corrcoef(t[b, WARMUP + h:, k], mid[b, WARMUP + h:] - mid[b, WARMUP:-h])[0, 1])
        print(f"  h {h:4d}: t0 {np.nanmean(cs[0::2]):+.3f}  t1 {np.nanmean(cs[1::2]):+.3f}")

    if scored is not None:
        print("\n== scoring mask ==")
        frac = scored[:, WARMUP:].mean()
        runs = []
        for b in range(scored.shape[0]):
            m = scored[b].astype(np.int8)
            edges = np.diff(np.concatenate([[0], m, [0]]))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            runs.extend((ends - starts).tolist())
        runs = np.array(runs)
        print(f"  scored fraction of required rows: {frac * 100:.2f}%")
        print(f"  runs per sequence: {len(runs) / scored.shape[0]:.1f}, run length min/median/max: {runs.min()}/{np.median(runs):.0f}/{runs.max()}")
        first = [np.argmax(scored[b]) for b in range(scored.shape[0])]
        print(f"  first scored step: min {min(first)} median {np.median(first):.0f} max {max(first)}")
        # what do scored rows look like vs unscored: target magnitude
        ts, tu = np.abs(t[scored]), np.abs(t[~scored & (np.arange(SEQUENCE_LENGTH) >= WARMUP)[None, :]])
        print(f"  mean |t| scored {ts.mean(axis=0)}  unscored {tu.mean(axis=0)}")


def describe_drift(f: np.ndarray):
    G = group_slices()
    print("\n== within-sequence drift: mean of first 1000 vs last 1000 rows, in units of global std ==")
    gstd = f.reshape(-1, 112).std(axis=0) + 1e-9
    shift = (f[:, -1000:].mean(axis=1) - f[:, :1000].mean(axis=1)) / gstd   # (B, 112)
    for g, sl in G.items():
        print(f"  {g:10s} mean|shift| {np.abs(shift[:, sl]).mean():.3f}")
    print("== across-sequence spread of per-sequence feature means (units of global std) ==")
    seq_mean = f.mean(axis=1)                                          # (B, 112)
    for g, sl in G.items():
        print(f"  {g:10s} {(seq_mean[:, sl].std(axis=0) / gstd[sl]).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tr = SequenceReader(TRAIN_PATH)
    va = SequenceReader(VALID_PATH)
    print(f"train: {len(tr)} sequences, valid: {len(va)} sequences, {len(FEATURE_COLUMNS)} features")
    trb = load_subset(tr, rng.choice(len(tr), args.seqs, replace=False))
    vab = load_subset(va, rng.choice(len(va), args.seqs, replace=False))
    print(f"loaded {args.seqs} + {args.seqs} sequences")
    print("finite:", np.isfinite(trb.features).all(), np.isfinite(trb.targets).all())

    describe_features(trb.features.reshape(-1, 112))
    describe_targets(trb.targets, trb.features, None)
    print("\n\n######## VALID ########")
    describe_targets(vab.targets, vab.features, vab.scored)
    describe_drift(trb.features)


if __name__ == "__main__":
    main()
