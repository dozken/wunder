"""
Wunder Predictorium — Solution V5
Entry point for the competition submission.

LSTM-based model with temporal attention + skip connection, ensemble of fold checkpoints.
V5: LSTM (hidden=256, 3 layers), 102 engineered features, temporal attention, skip connection.
"""
import numpy as np
import os
import sys

# Ensure utils is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint

# ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None


class PredictionModel:
    """
    Competition-compliant prediction model.

    Contract:
    - predict() receives DataPoint with .seq_ix, .step_in_seq, .need_prediction, .state (32,)
    - Returns None when need_prediction is False
    - Returns np.ndarray of shape (2,) when need_prediction is True
    - Must reset internal state when seq_ix changes
    - Must be deterministic
    """

    def __init__(self):
        self._current_seq_ix: int | None = None
        self._sessions: list[ort.InferenceSession] = []
        self._hiddens: list[np.ndarray | None] = []
        self._cells: list[np.ndarray | None] = []  # LSTM cell state

        # Load ALL ONNX models in the current directory
        onnx_files = [f for f in os.listdir(CURRENT_DIR) if f.endswith(".onnx")]
        onnx_files.sort()

        if not onnx_files:
            print("WARNING: No .onnx models found!")

        if ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            for f in onnx_files:
                path = os.path.join(CURRENT_DIR, f)
                try:
                    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
                    self._sessions.append(sess)
                    self._hiddens.append(None)
                    self._cells.append(None)
                    print(f"Loaded ONNX model: {f}")
                except Exception as e:
                    print(f"Failed to load {f}: {e}")

        # Initialize states
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state for a new sequence."""
        for i, sess in enumerate(self._sessions):
            if sess is not None:
                try:
                    # hidden_in shape: (num_layers, batch=1, hidden_dim)
                    hidden_shape = sess.get_inputs()[1].shape
                    h_shape = [s if isinstance(s, int) else 3 for s in hidden_shape]
                    self._hiddens[i] = np.zeros(h_shape, dtype=np.float32)

                    # cell_in shape: same as hidden
                    cell_shape = sess.get_inputs()[2].shape
                    c_shape = [s if isinstance(s, int) else 3 for s in cell_shape]
                    self._cells[i] = np.zeros(c_shape, dtype=np.float32)
                except Exception:
                    # Fallback
                    self._hiddens[i] = np.zeros((3, 1, 256), dtype=np.float32)
                    self._cells[i] = np.zeros((3, 1, 256), dtype=np.float32)
            else:
                self._hiddens[i] = None
                self._cells[i] = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset state on new sequence
        if self._current_seq_ix != data_point.seq_ix:
            self._current_seq_ix = data_point.seq_ix
            self._reset_state()

        # Always run inference to update hidden/cell state
        preds = []
        for i, sess in enumerate(self._sessions):
            x = data_point.state.astype(np.float32).reshape(1, 1, -1)

            outputs = sess.run(
                ["prediction", "hidden_out", "cell_out"],
                {
                    "input": x,
                    "hidden_in": self._hiddens[i],
                    "cell_in": self._cells[i],
                },
            )

            p = outputs[0][0, 0, :].astype(np.float64)
            self._hiddens[i] = outputs[1]
            self._cells[i] = outputs[2]
            preds.append(p)

        # No prediction needed during warm-up
        if not data_point.need_prediction:
            return None

        # Average predictions from all models
        if not preds:
            return np.zeros(2)

        avg_pred = np.mean(preds, axis=0)
        return avg_pred
