"""
Wunder Predictorium — Solution
Entry point for the competition submission.

This file defines PredictionModel which complies with the competition contract.
The actual model logic is delegated to separate modules for easy swapping.
"""
import numpy as np
import os
import sys

# Ensure utils is importable (for DataPoint)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint


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
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state for a new sequence."""
        self._history: list[np.ndarray] = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset state on new sequence
        if self._current_seq_ix != data_point.seq_ix:
            self._current_seq_ix = data_point.seq_ix
            self._reset_state()

        # Accumulate history (used by models that need temporal context)
        self._history.append(data_point.state.copy())

        # No prediction needed during warm-up
        if not data_point.need_prediction:
            return None

        # --- Model inference goes here ---
        prediction = self._infer()

        return prediction

    def _infer(self) -> np.ndarray:
        """
        Run model inference. Override or extend this for different models.
        Currently returns zeros as a minimal baseline.

        Returns:
            np.ndarray of shape (2,) — predictions for [t0, t1]
        """
        return np.zeros(2, dtype=np.float64)
