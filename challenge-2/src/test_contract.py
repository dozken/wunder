"""
Contract tests for PredictionModel.
Ensures solution.py complies with all competition requirements.
"""
import sys
import os
import numpy as np
import pytest

# Add paths
SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_DIR = os.path.join(SOLUTION_DIR, "..", "competition_package")
sys.path.insert(0, SOLUTION_DIR)
sys.path.insert(0, COMPETITION_DIR)

from utils import DataPoint, ScorerStepByStep
from solution import PredictionModel


def make_data_point(
    seq_ix: int = 0,
    step_in_seq: int = 99,
    need_prediction: bool = True,
    state: np.ndarray | None = None,
) -> DataPoint:
    """Helper to create a DataPoint with sensible defaults."""
    if state is None:
        state = np.random.randn(32).astype(np.float64)
    return DataPoint(
        seq_ix=seq_ix,
        step_in_seq=step_in_seq,
        need_prediction=need_prediction,
        state=state,
    )


class TestPredictionModelContract:
    """Tests that PredictionModel meets all competition requirements."""

    def test_class_exists(self):
        """PredictionModel must be importable from solution.py."""
        model = PredictionModel()
        assert model is not None

    def test_has_predict_method(self):
        """Must have a predict method."""
        model = PredictionModel()
        assert hasattr(model, "predict")
        assert callable(model.predict)

    def test_returns_none_when_no_prediction_needed(self):
        """Must return None when need_prediction is False."""
        model = PredictionModel()
        dp = make_data_point(need_prediction=False, step_in_seq=50)
        result = model.predict(dp)
        assert result is None, f"Expected None for warm-up step, got {result}"

    def test_returns_ndarray_when_prediction_needed(self):
        """Must return np.ndarray when need_prediction is True."""
        model = PredictionModel()
        dp = make_data_point(need_prediction=True, step_in_seq=99)
        result = model.predict(dp)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"

    def test_returns_correct_shape(self):
        """Must return shape (2,) — one value for t0 and t1."""
        model = PredictionModel()
        dp = make_data_point(need_prediction=True)
        result = model.predict(dp)
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"

    def test_returns_finite_values(self):
        """Predictions must be finite (no NaN/Inf)."""
        model = PredictionModel()
        dp = make_data_point(need_prediction=True)
        result = model.predict(dp)
        assert np.all(np.isfinite(result)), f"Non-finite predictions: {result}"

    def test_handles_sequence_reset(self):
        """Must handle seq_ix changes (new sequence = fresh state)."""
        model = PredictionModel()

        # First sequence
        for step in range(100):
            dp = make_data_point(seq_ix=0, step_in_seq=step, need_prediction=(step >= 99))
            result = model.predict(dp)
            if step < 99:
                assert result is None
            else:
                assert isinstance(result, np.ndarray)

        # Second sequence — should reset state, not crash
        for step in range(100):
            dp = make_data_point(seq_ix=1, step_in_seq=step, need_prediction=(step >= 99))
            result = model.predict(dp)
            if step < 99:
                assert result is None
            else:
                assert isinstance(result, np.ndarray)
                assert result.shape == (2,)

    def test_deterministic(self):
        """Must produce same output when run twice on same data."""
        np.random.seed(42)
        states = [np.random.randn(32) for _ in range(100)]

        def run_model():
            model = PredictionModel()
            results = []
            for step, state in enumerate(states):
                dp = make_data_point(seq_ix=0, step_in_seq=step, need_prediction=(step >= 99), state=state)
                result = model.predict(dp)
                if result is not None:
                    results.append(result.copy())
            return results

        run1 = run_model()
        run2 = run_model()

        assert len(run1) == len(run2), "Different number of predictions"
        for i, (r1, r2) in enumerate(zip(run1, run2)):
            np.testing.assert_array_equal(r1, r2, err_msg=f"Prediction {i} differs between runs")

    def test_handles_full_sequence(self):
        """Must handle a complete 1000-step sequence without errors."""
        model = PredictionModel()
        predictions_count = 0

        for step in range(1000):
            dp = make_data_point(
                seq_ix=0,
                step_in_seq=step,
                need_prediction=(step >= 99),
            )
            result = model.predict(dp)
            if step >= 99:
                assert result is not None, f"Missing prediction at step {step}"
                assert result.shape == (2,), f"Wrong shape at step {step}"
                predictions_count += 1
            else:
                assert result is None, f"Unexpected prediction at warm-up step {step}"

        assert predictions_count == 901, f"Expected 901 predictions, got {predictions_count}"

    def test_passes_scorer_check(self):
        """Must pass the official ScorerStepByStep validation checks."""
        model = PredictionModel()

        # Simulate what the scorer does — warm-up then prediction
        for step in range(100):
            need_pred = step >= 99
            dp = make_data_point(seq_ix=0, step_in_seq=step, need_prediction=need_pred)
            prediction = model.predict(dp)

            # Replicate scorer's check_prediction logic
            if not need_pred:
                assert prediction is None, "Scorer would raise: prediction not needed"
            else:
                assert prediction is not None, "Scorer would raise: prediction required"
                assert prediction.shape[0] == 2, f"Scorer would raise: wrong shape {prediction.shape}"
