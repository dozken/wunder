"""
Training pipeline for Wunder Predictorium.
Trains a GRU model on train.parquet and exports to ONNX.

Usage:
    cd src && python train.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from model import LOBPredictor

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(CURRENT_DIR, "..", "competition_package", "datasets")
TRAIN_PATH = os.path.join(DATASETS_DIR, "train.parquet")
VALID_PATH = os.path.join(DATASETS_DIR, "valid.parquet")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "model.onnx")

# Hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
SEQ_LEN = 1000  # Fixed sequence length
INPUT_DIM = 32
OUTPUT_DIM = 2
WARMUP_STEPS = 99  # Steps 0-98 are warm-up


class LOBDataset(Dataset):
    """Dataset that loads LOB sequences grouped by seq_ix."""

    def __init__(self, parquet_path: str):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        self.seq_ids = df["seq_ix"].unique()
        self.n_seqs = len(self.seq_ids)

        # Feature and target column names
        feature_cols = [c for c in df.columns if c not in ["seq_ix", "step_in_seq", "need_prediction", "t0", "t1"]]
        target_cols = ["t0", "t1"]

        # Pre-extract all sequences as numpy arrays for fast access
        print(f"Preparing {self.n_seqs} sequences...")
        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []  # 1 where need_prediction is True

        for seq_id in tqdm(self.seq_ids, desc="Building dataset"):
            seq_data = df[df["seq_ix"] == seq_id].sort_values("step_in_seq")
            self.features.append(seq_data[feature_cols].values.astype(np.float32))
            self.targets.append(seq_data[target_cols].values.astype(np.float32))
            self.masks.append(seq_data["need_prediction"].values.astype(np.float32))

        print(f"Dataset ready: {self.n_seqs} sequences × {SEQ_LEN} steps")

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.masks[idx]),
        )


def weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss weighted by abs(target), only on scored steps.
    Mirrors the competition's weighted Pearson metric.
    """
    # Expand mask to match prediction dims: (batch, seq, 1)
    mask = mask.unsqueeze(-1)

    # Weight by target amplitude (mirrors evaluation metric)
    weights = torch.abs(targets) * mask
    weights = torch.clamp(weights, min=1e-8)

    # Squared error on scored steps only
    se = (predictions - targets) ** 2
    weighted_se = se * weights

    # Mean over all scored predictions
    return weighted_se.sum() / weights.sum()


def export_to_onnx(model: LOBPredictor, save_path: str) -> None:
    """Export trained model to ONNX for fast CPU inference."""
    model.eval()
    model.cpu()

    # Dummy inputs for tracing — single step inference
    dummy_x = torch.randn(1, 1, INPUT_DIM)  # (batch=1, seq=1, features=32)
    dummy_h = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)  # GRU hidden state

    torch.onnx.export(
        model,
        (dummy_x, dummy_h),
        save_path,
        input_names=["input", "hidden_in"],
        output_names=["prediction", "hidden_out"],
        dynamic_axes={
            "input": {1: "seq_len"},
            "prediction": {1: "seq_len"},
        },
        opset_version=17,
    )
    print(f"✅ Model exported to {save_path}")


def validate(
    model: LOBPredictor,
    valid_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and compute weighted Pearson correlation."""
    # Import the official scorer metric
    sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "competition_package"))
    from utils import weighted_pearson_correlation

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, mask in valid_loader:
            features, targets = features.to(device), targets.to(device)
            predictions, _ = model(features)

            # Extract only scored steps
            for i in range(features.shape[0]):
                scored = mask[i] > 0
                all_preds.append(predictions[i][scored].cpu().numpy())
                all_targets.append(targets[i][scored].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_targets, axis=0)

    scores = {}
    target_names = ["t0", "t1"]
    for ix, name in enumerate(target_names):
        scores[name] = weighted_pearson_correlation(tgts[:, ix], preds[:, ix])
    scores["weighted_pearson"] = np.mean([scores["t0"], scores["t1"]])

    return scores


def train() -> None:
    """Main training loop."""
    device = torch.device("cpu")  # Match competition environment
    print(f"Device: {device}")

    # Load datasets
    train_dataset = LOBDataset(TRAIN_PATH)
    valid_dataset = LOBDataset(VALID_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = LOBPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_score = -float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for features, targets, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)

            # Forward pass — full sequence at once
            predictions, _ = model(features)

            # Loss only on scored steps
            loss = weighted_mse_loss(predictions, targets, mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t_start

        # Validate
        val_scores = validate(model, valid_loader, device)

        print(
            f"  Epoch {epoch}: loss={avg_loss:.6f}, "
            f"wpc={val_scores['weighted_pearson']:.6f} "
            f"(t0={val_scores['t0']:.4f}, t1={val_scores['t1']:.4f}), "
            f"lr={scheduler.get_last_lr()[0]:.2e}, "
            f"time={elapsed:.0f}s"
        )

        # Save best
        if val_scores["weighted_pearson"] > best_score:
            best_score = val_scores["weighted_pearson"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ⭐ New best: {best_score:.6f}")

    # Load best model and export
    print(f"\n{'='*50}")
    print(f"Best validation score: {best_score:.6f}")
    model.load_state_dict(best_state)

    # Save PyTorch checkpoint first (in case ONNX export fails)
    checkpoint_path = os.path.join(CURRENT_DIR, "model_checkpoint.pt")
    torch.save(best_state, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")

    # Export to ONNX
    export_to_onnx(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
