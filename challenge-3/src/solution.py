"""Competition entry point: stateful ONNX inference, one row per call.

Loads every *.onnx next to this file and averages their predictions. Each
graph takes `features` plus one or more recurrent state tensors (every other
input) and returns `prediction` followed by the next states in the same
order, so graphs from export.py (one packed state), export_slim.py (one state
per layer) and unroll.py can be mixed.

Inputs and outputs are bound once to preallocated buffers (ORT IO binding)
and the recurrent state ping-pongs between two sets of buffers via two
prebuilt bindings, so a call does no allocation and no state copy. On the
scorer's single x86 vCPU that is worth ~5 us of the ~96 us per-row budget.
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


def _static(shape):
    return tuple(d if isinstance(d, int) else 1 for d in shape)


class _Graph:
    """One ONNX graph, its state buffers and two IO bindings (A->B, B->A)."""

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
        self.features = np.zeros(_static(feat.shape), dtype=np.float32)
        self.state_names = [i.name for i in inputs if i.name != "features"]
        state_shapes = [_static(i.shape) for i in inputs if i.name != "features"]
        self.next_names = [o.name for o in outputs if o.name != "prediction"]
        pred = next(o for o in outputs if o.name == "prediction")
        self.prediction = np.zeros(_static(pred.shape), dtype=np.float32)
        if len(self.next_names) != len(self.state_names):
            raise ValueError(f"{path.name}: state inputs and outputs do not match")
        # two buffer sets; binding k reads set k and writes set 1-k
        self.buffers = ([np.zeros(s, dtype=np.float32) for s in state_shapes],
                        [np.zeros(s, dtype=np.float32) for s in state_shapes])
        self.bindings = []
        for src in (0, 1):
            b = self.session.io_binding()
            b.bind_input("features", "cpu", 0, np.float32, self.features.shape, self.features.ctypes.data)
            for name, buf in zip(self.state_names, self.buffers[src]):
                b.bind_input(name, "cpu", 0, np.float32, buf.shape, buf.ctypes.data)
            b.bind_output("prediction", "cpu", 0, np.float32, self.prediction.shape, self.prediction.ctypes.data)
            for name, buf in zip(self.next_names, self.buffers[1 - src]):
                b.bind_output(name, "cpu", 0, np.float32, buf.shape, buf.ctypes.data)
            self.bindings.append(b)
        self.run_options = ort.RunOptions()
        self.flip = 0

    def reset(self) -> None:
        for bufs in self.buffers:
            for buf in bufs:
                buf.fill(0.0)
        self.flip = 0

    def step(self, row: np.ndarray) -> np.ndarray:
        self.features.reshape(-1)[:] = row
        self.session.run_with_iobinding(self.bindings[self.flip], self.run_options)
        self.flip ^= 1
        return self.prediction.reshape(-1)


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
        if out.shape != (N_TARGETS,) or not np.isfinite(out).all():
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
