"""Competition entry point: stateful ONNX inference, one row per call.

Loads every *.onnx next to this file and averages their predictions. Each
graph takes (features[1,1,112], state) and returns (prediction[1,1,2], next
state); the state shape is read from the graph so models of different sizes
can be mixed.
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


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class PredictionModel:
    def __init__(self):
        paths = sorted(HERE.glob("*.onnx"))
        if not paths:
            raise FileNotFoundError(f"no .onnx models next to {__file__}")
        self.sessions = [_session(p) for p in paths]
        self.state_shapes = [tuple(s.get_inputs()[1].shape) for s in self.sessions]
        self.states = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]
        self.output_names = ["prediction", "next_state"]
        self.current_seq_ix = None
        self.previous_step = None
        self.features = np.zeros((1, 1, N_FEATURES), dtype=np.float32)
        self.scale = 1.0 / len(self.sessions)

    def _reset(self) -> None:
        for i, shape in enumerate(self.state_shapes):
            self.states[i] = np.zeros(shape, dtype=np.float32)

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
        total = None
        for i, sess in enumerate(self.sessions):
            pred, self.states[i] = sess.run(self.output_names,
                                            {"features": self.features, "state": self.states[i]})
            total = pred if total is None else total + pred
        if not data_point.need_prediction:
            return None
        out = (total[0, 0] * self.scale).astype(np.float32)
        if not np.isfinite(out).all():
            out = np.zeros(2, dtype=np.float32)
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
