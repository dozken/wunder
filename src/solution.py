"""
Wunder Predictorium — Solution
Entry point for the competition submission.

This file defines PredictionModel which complies with the competition contract.
The actual model is a GRU trained on LOB sequences, exported to ONNX.
"""
import numpy as np
import os
import sys

# Ensure utils is importable (for DataPoint)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint

# ONNX Runtime (lazy import for flexibility)
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

        # Load ONNX model
        self._session: ort.InferenceSession | None = None
        self._hidden: np.ndarray | None = None

        onnx_path = os.path.join(CURRENT_DIR, "model.onnx")
        if ort is not None and os.path.exists(onnx_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                onnx_path, opts, providers=["CPUExecutionProvider"]
            )
            print(f"Loaded ONNX model from {onnx_path}")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state for a new sequence."""
        # Reset GRU hidden state: (num_layers, batch=1, hidden_dim)
        if self._session is not None:
            # Infer hidden dim from model
            hidden_shape = self._session.get_inputs()[1].shape
            self._hidden = np.zeros(
                [s if isinstance(s, int) else 2 for s in hidden_shape],
                dtype=np.float32,
            )
        else:
            self._hidden = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset state on new sequence
        if self._current_seq_ix != data_point.seq_ix:
            self._current_seq_ix = data_point.seq_ix
            self._reset_state()

        # Always run inference to update hidden state (even during warm-up)
        prediction = self._infer(data_point.state)

        # No prediction needed during warm-up
        if not data_point.need_prediction:
            return None

        return prediction

    def _infer(self, state: np.ndarray) -> np.ndarray:
        """
        Run single-step GRU inference via ONNX.
        Updates hidden state and returns prediction.
        """
        if self._session is None:
            return np.zeros(2, dtype=np.float64)

        # Prepare input: (batch=1, seq_len=1, features=32)
        x = state.astype(np.float32).reshape(1, 1, -1)

        # Run ONNX inference
        pred, self._hidden = self._session.run(
            ["prediction", "hidden_out"],
            {"input": x, "hidden_in": self._hidden},
        )

        # pred shape: (1, 1, 2) → flatten to (2,)
        return pred[0, 0, :].astype(np.float64)
