"""Contract tests: metric replica vs the organisers' reference, and the
solution's behaviour under the row-by-row callback.

    cd src && python -m pytest test_contract.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "wnn_connectome_starterpack"))

from metric import score_batch, weighted_pearson_batch  # noqa: E402
from utils import BlockAccumulator, DataPoint, SEQUENCE_LENGTH, WARMUP  # noqa: E402


def _random_blocks(rng, n, sparse_mask=True):
    targets = rng.standard_normal((n, SEQUENCE_LENGTH, 2)).astype(np.float32) * 1.5
    preds = (0.3 * targets + rng.standard_normal(targets.shape)).astype(np.float32)
    mask = np.zeros((n, SEQUENCE_LENGTH), dtype=bool)
    if sparse_mask:
        mask[:, WARMUP:] = rng.random((n, SEQUENCE_LENGTH - WARMUP)) < 0.2
    else:
        mask[:, WARMUP:] = True
    return targets, preds, mask


def test_metric_matches_reference():
    rng = np.random.default_rng(1)
    targets, preds, mask = _random_blocks(rng, 6)
    # make one block ineligible: all-positive t1 on scored rows
    targets[2, :, 1] = np.abs(targets[2, :, 1]) + 0.1
    # and push some values past the clip
    preds[0] *= 4
    ref = BlockAccumulator()
    for i in range(len(targets)):
        ref.add(targets[i], preds[i], mask[i])
    expected = ref.result()
    got = score_batch(targets, preds, mask)
    assert got["eligible_blocks"] == expected["eligible_blocks"] == 5
    assert abs(got["t0"] - expected["t0"]) < 1e-9
    assert abs(got["t1"] - expected["t1"]) < 1e-9
    assert abs(got["weighted_pearson"] - expected["weighted_pearson"]) < 1e-9


def test_metric_degenerate_cases():
    rng = np.random.default_rng(2)
    targets, preds, mask = _random_blocks(rng, 2, sparse_mask=False)
    preds[0] = 0.0                       # constant prediction -> 0, still eligible
    corr, eligible = weighted_pearson_batch(targets, preds, mask)
    assert eligible.all()
    assert corr[0].tolist() == [0.0, 0.0]
    assert (np.abs(corr[1]) > 0).all()


def test_torch_loss_matches_metric():
    torch = pytest.importorskip("torch")
    from loss import weighted_pearson
    rng = np.random.default_rng(3)
    targets, preds, mask = _random_blocks(rng, 4, sparse_mask=False)
    # the loss leaves predictions unclipped so gradients flow; the metric clips
    # them, so compare on already-clipped predictions
    preds = np.clip(preds, -2, 2)
    corr, _ = weighted_pearson_batch(targets, preds, mask)
    got = weighted_pearson(torch.from_numpy(preds), torch.from_numpy(targets), torch.from_numpy(mask)).numpy()
    assert np.abs(got - corr).max() < 1e-5


@pytest.fixture(scope="module")
def solution_model():
    if not list(HERE.glob("*.onnx")):
        pytest.skip("no exported .onnx model in src/")
    from solution import PredictionModel
    return PredictionModel()


def test_solution_contract(solution_model):
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((300, 112)).astype(np.float32)
    outs = []
    for seq in (7, 3):
        for step in range(300):
            out = solution_model.predict(DataPoint(seq, step, step >= WARMUP, feats[step]))
            if step < WARMUP:
                assert out is None
            else:
                assert out.shape == (2,) and out.dtype == np.float32 and np.isfinite(out).all()
                outs.append(out.copy())
    a = np.stack(outs[:201])
    b = np.stack(outs[201:])
    assert np.allclose(a, b), "state was not reset between sequences / not deterministic"
    # prediction must depend on history, not just the current row
    solution_model.predict(DataPoint(9, 0, False, feats[0]))
    for step in range(1, WARMUP):
        solution_model.predict(DataPoint(9, step, False, feats[step]))
    fresh = solution_model.predict(DataPoint(9, WARMUP, True, feats[299]))
    assert not np.allclose(fresh, a[-1])


def test_solution_rejects_out_of_order(solution_model):
    feats = np.zeros(112, dtype=np.float32)
    solution_model.predict(DataPoint(11, 0, False, feats))
    with pytest.raises(ValueError):
        solution_model.predict(DataPoint(11, 2, False, feats))
    with pytest.raises(ValueError):
        solution_model.predict(DataPoint(12, 5, False, feats))
