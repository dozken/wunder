"""Competition entry point: stateful ONNX inference, one row per call.

Loads every *.onnx next to this file and averages their predictions. Each
graph takes (features[1,1,112], state) and returns (prediction[1,1,2], next
state); the state shape is read from the graph so models of different sizes
can be mixed.

Inputs and outputs are bound once to preallocated buffers (ORT IO binding)
and the recurrent state ping-pongs between two buffers, so a call does no
allocation and no numpy<->OrtValue conversion. That is worth ~5 us of the
~96 us per-row budget on the scorer's single vCPU.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
N_FEATURES = 112
N_TARGETS = 2


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class _BoundModel:
    """One ONNX graph with its state buffers and two prebuilt IO bindings."""

    def __init__(self, path: Path, features: np.ndarray):
        self.session = _session(path)
        shape = tuple(self.session.get_inputs()[1].shape)
        self.states = (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))
        self.prediction = np.zeros((1, 1, N_TARGETS), dtype=np.float32)
        self.bindings = []
        for src, dst in ((0, 1), (1, 0)):
            b = self.session.io_binding()
            b.bind_input("features", "cpu", 0, np.float32, features.shape, features.ctypes.data)
            b.bind_input("state", "cpu", 0, np.float32, shape, self.states[src].ctypes.data)
            b.bind_output("prediction", "cpu", 0, np.float32, self.prediction.shape, self.prediction.ctypes.data)
            b.bind_output("next_state", "cpu", 0, np.float32, shape, self.states[dst].ctypes.data)
            self.bindings.append(b)

    def reset(self) -> None:
        self.states[0].fill(0.0)
        self.states[1].fill(0.0)

    def step(self, flip: int, run_options) -> np.ndarray:
        self.session.run_with_iobinding(self.bindings[flip], run_options)
        return self.prediction


class PredictionModel:
    def __init__(self):
        paths = sorted(HERE.glob("*.onnx"))
        if not paths:
            raise FileNotFoundError(f"no .onnx models next to {__file__}")
        self.features = np.zeros((1, 1, N_FEATURES), dtype=np.float32)
        self.models = [_BoundModel(p, self.features) for p in paths]
        self.run_options = ort.RunOptions()
        self.scale = 1.0 / len(self.models)
        self.flip = 0
        self.current_seq_ix = None
        self.previous_step = None

    def _reset(self) -> None:
        for m in self.models:
            m.reset()
        self.flip = 0

    def predict(self, data_point):
        seq_ix = data_point.seq_ix
        step = data_point.step_in_seq
        if seq_ix != self.current_seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self._reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        self.features[0, 0, :] = data_point.state
        flip = self.flip
        self.flip = flip ^ 1
        total = None
        for m in self.models:
            pred = m.step(flip, self.run_options)
            total = pred[0, 0].copy() if total is None else total + pred[0, 0]
        if not data_point.need_prediction:
            return None
        out = (total * self.scale).astype(np.float32)
        if not np.isfinite(out).all():
            out = np.zeros(N_TARGETS, dtype=np.float32)
        return out


if __name__ == "__main__":
    import argparse
    import sys
    import time

    sys.path.insert(0, str(HERE.parent / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    t0 = time.time()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
    print(f"{time.time() - t0:.0f}s")
