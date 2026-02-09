"""
Data Exploration for Wunder Predictorium
Analyzes feature distributions, target behavior, correlations, and sequence structure.
"""
import numpy as np
import pandas as pd
import time

print("=" * 60)
print("WUNDER PREDICTORIUM — DATA EXPLORATION")
print("=" * 60)

# Load data
print("\n📂 Loading datasets...")
t0 = time.time()
train = pd.read_parquet("datasets/train.parquet")
valid = pd.read_parquet("datasets/valid.parquet")
print(f"  Train: {train.shape} loaded in {time.time()-t0:.1f}s")
print(f"  Valid: {valid.shape}")
print(f"  Train memory: {train.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"  Valid memory: {valid.memory_usage(deep=True).sum()/1e6:.1f} MB")

# Column info
print(f"\n📋 Columns ({len(train.columns)}):")
print(f"  {list(train.columns)}")
print(f"  Dtypes:\n{train.dtypes.value_counts().to_string()}")

# Basic stats
print("\n" + "=" * 60)
print("1. BASIC STATISTICS")
print("=" * 60)

features = [c for c in train.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1']]
targets = ['t0', 't1']
print(f"\n  Features ({len(features)}): {features}")
print(f"  Targets: {targets}")

# NaN check
print(f"\n🔍 NaN check:")
nan_counts = train[features + targets].isna().sum()
if nan_counts.sum() == 0:
    print("  ✅ No NaNs in features or targets")
else:
    print(f"  ⚠️ NaN counts:\n{nan_counts[nan_counts > 0]}")

# Feature statistics
print(f"\n📊 Feature summary (train):")
feat_stats = train[features].describe().T[['mean', 'std', 'min', 'max']]
print(feat_stats.to_string())

# Target statistics
print(f"\n🎯 Target summary (train):")
tgt_stats = train[targets].describe()
print(tgt_stats.to_string())

# Target correlation with each other
print(f"\n  t0-t1 correlation: {train['t0'].corr(train['t1']):.4f}")

print("\n" + "=" * 60)
print("2. FEATURE-TARGET CORRELATIONS")
print("=" * 60)

# Pearson correlation of each feature with each target
print("\n  Feature correlations with targets:")
corr_t0 = train[features].corrwith(train['t0'])
corr_t1 = train[features].corrwith(train['t1'])
corr_df = pd.DataFrame({'corr_t0': corr_t0, 'corr_t1': corr_t1, 'abs_mean': (corr_t0.abs() + corr_t1.abs()) / 2})
corr_df = corr_df.sort_values('abs_mean', ascending=False)
print(corr_df.to_string())

print("\n" + "=" * 60)
print("3. SEQUENCE STRUCTURE")
print("=" * 60)

# Sequence info
n_seqs_train = train['seq_ix'].nunique()
n_seqs_valid = valid['seq_ix'].nunique()
seq_lengths = train.groupby('seq_ix').size()
print(f"\n  Train sequences: {n_seqs_train}")
print(f"  Valid sequences: {n_seqs_valid}")
print(f"  Steps per sequence: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.0f}")

# need_prediction distribution
np_counts = train.groupby('seq_ix')['need_prediction'].sum()
print(f"  Predictions per sequence: min={np_counts.min()}, max={np_counts.max()}, mean={np_counts.mean():.0f}")

# When does need_prediction start?
first_pred = train[train['need_prediction'] == 1].groupby('seq_ix')['step_in_seq'].min()
print(f"  First prediction step: min={first_pred.min()}, max={first_pred.max()}, mean={first_pred.mean():.0f}")

print("\n" + "=" * 60)
print("4. TEMPORAL PATTERNS")
print("=" * 60)

# Pick a random sequence and analyze temporal patterns
sample_seq = train[train['seq_ix'] == train['seq_ix'].unique()[0]]
print(f"\n  Sample sequence (seq_ix={sample_seq['seq_ix'].iloc[0]}), length={len(sample_seq)}")

# Autocorrelation of targets within sequence
for target in targets:
    vals = sample_seq[target].values
    autocorr_1 = np.corrcoef(vals[:-1], vals[1:])[0, 1]
    autocorr_5 = np.corrcoef(vals[:-5], vals[5:])[0, 1]
    autocorr_10 = np.corrcoef(vals[:-10], vals[10:])[0, 1]
    print(f"  {target} autocorrelation: lag1={autocorr_1:.4f}, lag5={autocorr_5:.4f}, lag10={autocorr_10:.4f}")

# Average autocorrelation across multiple sequences
print(f"\n  Average autocorrelation (across 100 seqs):")
seq_ids = train['seq_ix'].unique()[:100]
for target in targets:
    lags = {1: [], 5: [], 10: []}
    for sid in seq_ids:
        seq = train[train['seq_ix'] == sid][target].values
        for lag in lags:
            if len(seq) > lag:
                lags[lag].append(np.corrcoef(seq[:-lag], seq[lag:])[0, 1])
    print(f"  {target}: " + ", ".join([f"lag{l}={np.mean(v):.4f}" for l, v in lags.items()]))

print("\n" + "=" * 60)
print("5. FEATURE DYNAMICS")
print("=" * 60)

# How much do features change step-to-step?
sample_feats = sample_seq[features].values
diffs = np.diff(sample_feats, axis=0)
print(f"\n  Step-to-step feature changes (sample sequence):")
print(f"  Mean abs change per feature:")
mean_abs_diffs = np.mean(np.abs(diffs), axis=0)
for f, d in sorted(zip(features, mean_abs_diffs), key=lambda x: -x[1]):
    print(f"    {f}: {d:.6f}")

# Feature value ranges
print(f"\n  Feature value ranges:")
for f in features:
    print(f"    {f}: [{train[f].min():.4f}, {train[f].max():.4f}], std={train[f].std():.4f}")

print("\n" + "=" * 60)
print("6. TARGET DISTRIBUTION ANALYSIS")
print("=" * 60)

for target in targets:
    vals = train[target].values
    print(f"\n  {target}:")
    print(f"    mean={vals.mean():.6f}, std={vals.std():.6f}")
    print(f"    skew={pd.Series(vals).skew():.4f}, kurtosis={pd.Series(vals).kurtosis():.4f}")
    print(f"    percentiles: 1%={np.percentile(vals, 1):.4f}, 25%={np.percentile(vals, 25):.4f}, "
          f"50%={np.percentile(vals, 50):.4f}, 75%={np.percentile(vals, 75):.4f}, 99%={np.percentile(vals, 99):.4f}")
    print(f"    |val| > 3: {(np.abs(vals) > 3).sum()} ({(np.abs(vals) > 3).mean()*100:.2f}%)")
    print(f"    |val| > 6: {(np.abs(vals) > 6).sum()} ({(np.abs(vals) > 6).mean()*100:.2f}%)")

print("\n✅ Exploration complete!")
