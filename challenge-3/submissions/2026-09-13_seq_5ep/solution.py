"""Competition entry point: stateful ONNX inference, one row per call.

Loads every *.onnx next to this file and averages their predictions. Each
graph takes `features` plus one or more recurrent state tensors (every other
input) and returns `prediction` followed by the next states in the same
order. Inputs and outputs are bound once to preallocated buffers; each call
only copies the new row in and the new state across.
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


class _Graph:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        feat = next(i for i in inputs if i.name == "features")
        self.features = np.zeros([d if isinstance(d, int) else 1 for d in feat.shape], dtype=np.float32)
        self.state_names = [i.name for i in inputs if i.name != "features"]
        self.state_shapes = [tuple(d if isinstance(d, int) else 1 for d in i.shape)
                             for i in inputs if i.name != "features"]
        self.next_names = [o.name for o in outputs if o.name != "prediction"]
        pred = next(o for o in outputs if o.name == "prediction")
        self.pred_shape = tuple(d if isinstance(d, int) else 1 for d in pred.shape)
        if len(self.next_names) != len(self.state_names):
            raise ValueError(f"{path.name}: state inputs and outputs do not match")
        self.states = [np.zeros(s, dtype=np.float32) for s in self.state_shapes]
        self.next_states = [np.zeros(s, dtype=np.float32) for s in self.state_shapes]
        self.prediction = np.zeros(self.pred_shape, dtype=np.float32)
        self._bind()

    def _bind(self) -> None:
        self.x_val = ort.OrtValue.ortvalue_from_numpy(self.features)
        self.s_vals = [ort.OrtValue.ortvalue_from_numpy(s) for s in self.states]
        self.n_vals = [ort.OrtValue.ortvalue_from_numpy(s) for s in self.next_states]
        self.p_val = ort.OrtValue.ortvalue_from_numpy(self.prediction)
        b = self.session.io_binding()
        b.bind_ortvalue_input("features", self.x_val)
        for name, val in zip(self.state_names, self.s_vals):
            b.bind_ortvalue_input(name, val)
        b.bind_ortvalue_output("prediction", self.p_val)
        for name, val in zip(self.next_names, self.n_vals):
            b.bind_ortvalue_output(name, val)
        self.binding = b
        self.run_options = ort.RunOptions()

    def reset(self) -> None:
        for v, s in zip(self.s_vals, self.state_shapes):
            v.update_inplace(np.zeros(s, dtype=np.float32))

    def step(self, row: np.ndarray) -> np.ndarray:
        self.x_val.update_inplace(row.reshape(self.features.shape))
        self.session.run_with_iobinding(self.binding, self.run_options)
        for s_val, n_val in zip(self.s_vals, self.n_vals):
            s_val.update_inplace(n_val.numpy())
        return self.p_val.numpy().reshape(-1)


class PredictionModel:
    def __init__(self):
        paths = sorted(HERE.glob("*.onnx"))
        if not paths:
            raise FileNotFoundError(f"no .onnx models next to {__file__}")
        self.graphs = [_Graph(p) for p in paths]
        self.scale = 1.0 / len(self.graphs)
        self.current_seq_ix = None
        self.previous_step = None

    def predict(self, data_point):
        seq_ix = data_point.seq_ix
        step = data_point.step_in_seq
        if seq_ix != self.current_seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            for g in self.graphs:
                g.reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        row = np.asarray(data_point.state, dtype=np.float32)
        total = None
        for g in self.graphs:
            p = g.step(row)
            total = p.copy() if total is None else total + p
        if not data_point.need_prediction:
            return None
        out = (total * self.scale).astype(np.float32)
        if out.shape != (2,) or not np.isfinite(out).all():
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
