"""Sequence-level access to the competition parquet files.

Every row group in the files is exactly one 20,000-row sequence, so the unit of
IO here is the row group. The training set is ~95 GB decoded, far more than
fits in memory, so training streams sequences from disk in a background thread
while the previous batch is on the GPU. A small fixed subset can be pinned in
memory for fast experiments and for the per-epoch validation check.
"""
from __future__ import annotations

import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STARTERPACK = ROOT / "wnn_connectome_starterpack"
DATASETS = STARTERPACK / "datasets"
TRAIN_PATH = DATASETS / "train.parquet"
VALID_PATH = DATASETS / "valid.parquet"

sys.path.insert(0, str(STARTERPACK))
from utils import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP  # noqa: E402

STEP_MASK = np.arange(SEQUENCE_LENGTH) >= WARMUP   # need_prediction, by contract


@dataclass
class Sequence:
    seq_ix: int
    features: np.ndarray        # (20000, 112) float32
    targets: np.ndarray         # (20000, 2) float32
    scored: np.ndarray | None   # (20000,) bool, only in valid


class SequenceReader:
    """Random access to the sequences of one parquet file."""

    def __init__(self, path: Path | str, soft_mask: Path | str | None = None):
        self.path = Path(path)
        self.parquet = pq.ParquetFile(self.path)
        names = self.parquet.schema_arrow.names
        self.has_mask = "is_scored" in names
        # predicted P(is_scored) per row for files without the real mask, as
        # written by maskmodel.py: float16 memmap of shape (row_groups, 20000)
        self.soft_mask = None
        if soft_mask is not None:
            self.soft_mask = np.memmap(soft_mask, dtype=np.float16, mode="r",
                                       shape=(self.parquet.num_row_groups, SEQUENCE_LENGTH))
            self.has_mask = True
        self.columns = ["seq_ix", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        if "is_scored" in names:
            self.columns.append("is_scored")

    def __len__(self) -> int:
        return self.parquet.num_row_groups

    def read(self, index: int) -> Sequence:
        table = self.parquet.read_row_group(index, columns=self.columns, use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {index} has {table.num_rows} rows")
        features = np.empty((SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
        for j, name in enumerate(FEATURE_COLUMNS):
            features[:, j] = table.column(name).to_numpy(zero_copy_only=False)
        targets = np.empty((SEQUENCE_LENGTH, 2), dtype=np.float32)
        for j, name in enumerate(TARGET_COLUMNS):
            targets[:, j] = table.column(name).to_numpy(zero_copy_only=False)
        scored = None
        if self.soft_mask is not None:
            scored = np.asarray(self.soft_mask[index], dtype=np.float32) * STEP_MASK
        elif "is_scored" in self.columns:
            scored = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool) & STEP_MASK
        seq_ix = int(table.column("seq_ix")[0].as_py())
        return Sequence(seq_ix, features, targets, scored)

    def read_many(self, indices, workers: int = 8) -> list[Sequence]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self.read, indices))


@dataclass
class Batch:
    features: np.ndarray        # (B, 20000, 112)
    targets: np.ndarray         # (B, 20000, 2)
    scored: np.ndarray | None   # (B, 20000) bool


def collate(sequences: list[Sequence]) -> Batch:
    features = np.stack([s.features for s in sequences])
    targets = np.stack([s.targets for s in sequences])
    scored = None
    if all(s.scored is not None for s in sequences):
        # real masks are bool, predicted ones float; a mixed batch becomes float
        scored = np.stack([s.scored.astype(np.float32) for s in sequences])
    return Batch(features, targets, scored)


class BatchStream:
    """Yields batches of whole sequences in a random order, prefetched in a thread.

    `indices` restricts an epoch to a subset of row groups (for dev runs and for
    holding out sequences). Each epoch reshuffles. Pass `extra=[(reader,
    indices), ...]` to draw from more files in the same epoch, e.g. the masked
    validation file alongside the training file; batches then mix sources.
    """

    def __init__(self, reader: SequenceReader, batch_size: int, indices=None,
                 seed: int = 0, prefetch: int = 1, workers: int = 8, drop_last: bool = True,
                 extra=()):
        self.sources = [(reader, np.arange(len(reader)) if indices is None else np.asarray(indices))]
        self.sources += [(r, np.asarray(ix)) for r, ix in extra]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.prefetch = prefetch
        self.workers = workers
        self.drop_last = drop_last

    def __len__(self) -> int:
        return sum(len(ix) for _, ix in self.sources)

    def batches_per_epoch(self) -> int:
        n = len(self)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def _read(self, keys) -> list[Sequence]:
        """keys: (source, row_group) pairs; reads each source's part in parallel."""
        out = [None] * len(keys)
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.sources[src][0].read, int(rg)): pos
                       for pos, (src, rg) in enumerate(keys)}
            for fut, pos in futures.items():
                out[pos] = fut.result()
        return out

    def __iter__(self):
        keys = np.concatenate([np.stack([np.full(len(ix), s), ix], axis=1)
                               for s, (_, ix) in enumerate(self.sources)])
        order = keys[self.rng.permutation(len(keys))]
        chunks = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.drop_last and chunks and len(chunks[-1]) < self.batch_size:
            chunks.pop()
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = object()

        def producer():
            try:
                for idx in chunks:
                    q.put(collate(self._read(idx)))
            except Exception as exc:  # surface reader errors in the consumer
                q.put(exc)
            finally:
                q.put(stop)

        threading.Thread(target=producer, daemon=True).start()
        while True:
            item = q.get()
            if item is stop:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def load_subset(reader: SequenceReader, indices, workers: int = 8) -> Batch:
    """Read a fixed set of sequences into memory once (validation subset, dev runs)."""
    return collate(reader.read_many(list(indices), workers))


def feature_stats(reader: SequenceReader, n_sequences: int = 128, seed: int = 0):
    """Per-feature mean/std over a random sample of training sequences.

    Used to standardise the raw anonymised features before the network. The
    statistics are baked into the exported model, so they are computed once
    per run and saved alongside the checkpoint.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(reader), size=min(n_sequences, len(reader)), replace=False)
    total = np.zeros(N_FEATURES, dtype=np.float64)
    total_sq = np.zeros(N_FEATURES, dtype=np.float64)
    count = 0
    for seq in reader.read_many(idx):                 # one sequence at a time, no big copy
        x = seq.features.astype(np.float64)
        total += x.sum(axis=0)
        total_sq += (x * x).sum(axis=0)
        count += len(x)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
