"""
Training pipeline for Wunder Predictorium — V5.
Key changes from V4:
  - LSTM with temporal attention + skip connection
  - Combined loss (alpha=0.85 Pearson + 0.15 MSE)
  - 50 epochs with SWA
  - Gradient accumulation (effective batch size = 1024)
  - lr=1e-3 with OneCycleLR
  - 102 engineered features
  - Sequence-level mixup augmentation

Usage:
    cd /Users/dozken/projects/wunder && env/bin/python3 src/train.py
    cd /Users/dozken/projects/wunder && env/bin/python3 src/train.py --fold 0
"""
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from model import LOBPredictor, ENGINEERED_DIM
from loss import CombinedLoss

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(CURRENT_DIR, "..", "competition_package", "datasets")
TRAIN_PATH = os.path.join(DATASETS_DIR, "train.parquet")
VALID_PATH = os.path.join(DATASETS_DIR, "valid.parquet")

# Hyperparameters — V5
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 12
SEQ_LEN = 1000
INPUT_DIM = 32
OUTPUT_DIM = 2
GRAD_CLIP = 1.0
WEIGHT_DECAY = 5e-4
N_FOLDS = 5
ACCUM_STEPS = 4  # Gradient accumulation → effective batch = 1024
SWA_START_EPOCH = 35  # Start SWA averaging from epoch 35
MIXUP_ALPHA = 0.2  # Mixup interpolation parameter


class LOBDataset(Dataset):
    """Dataset that loads LOB sequences grouped by seq_ix."""

    def __init__(self, parquet_path: str):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        self.seq_ids = df["seq_ix"].unique()
        self.n_seqs = len(self.seq_ids)

        feature_cols = [
            c for c in df.columns
            if c not in ["seq_ix", "step_in_seq", "need_prediction", "t0", "t1"]
        ]
        target_cols = ["t0", "t1"]

        print(f"Preparing {self.n_seqs} sequences...")
        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

        grouped = df.groupby("seq_ix", sort=False)

        for seq_id, seq_data in tqdm(grouped, desc="Building dataset", total=self.n_seqs):
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


def mixup_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply sequence-level mixup augmentation.

    Randomly blends pairs of sequences to create augmented training data.
    Only applied to features and targets; mask uses the original.
    """
    if alpha <= 0:
        return features, targets, masks

    batch_size = features.shape[0]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 to keep original dominant

    # Shuffle indices for mixing partners
    perm = torch.randperm(batch_size, device=features.device)

    mixed_features = lam * features + (1 - lam) * features[perm]
    mixed_targets = lam * targets + (1 - lam) * targets[perm]
    # Keep original mask (intersection would lose too many scored steps)

    return mixed_features, mixed_targets, masks


def export_to_onnx(model: LOBPredictor, save_path: str) -> None:
    """Export trained model to ONNX for fast CPU inference.

    LSTM requires both hidden state and cell state as inputs/outputs.
    """
    import onnx

    model.eval()
    model.cpu()

    dummy_x = torch.randn(1, 1, INPUT_DIM)
    dummy_h = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)
    dummy_c = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)

    torch.onnx.export(
        model,
        (dummy_x, dummy_h, dummy_c),
        save_path,
        input_names=["input", "hidden_in", "cell_in"],
        output_names=["prediction", "hidden_out", "cell_out"],
        dynamic_axes={
            "input": {1: "seq_len"},
            "prediction": {1: "seq_len"},
        },
        opset_version=17,
    )

    # Re-save with all weights embedded
    onnx_model = onnx.load(save_path)
    onnx.save(onnx_model, save_path, save_as_external_data=False)

    # Clean up any leftover external data files
    ext_data_path = save_path + ".data"
    if os.path.exists(ext_data_path):
        os.remove(ext_data_path)
        print(f"  Removed external data file: {ext_data_path}")

    print(f"✅ Model exported to {save_path} (all weights embedded)")


def validate(
    model: LOBPredictor,
    valid_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and compute weighted Pearson correlation."""
    sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "competition_package"))
    from utils import weighted_pearson_correlation

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, mask in valid_loader:
            features, targets = features.to(device), targets.to(device)
            predictions, _, _ = model(features)

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


def train_fold(
    fold_idx: int,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    device: torch.device,
) -> tuple[float, dict]:
    """Train a single fold and return best score and state dict."""
    print(f"\n{'='*20} Fold {fold_idx+1}/{N_FOLDS} {'='*20}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + fold_idx),
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Model
    model = LOBPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(device)

    # Optimizer + OneCycleLR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,   # initial_lr = max_lr / 25
        final_div_factor=1000.0,  # final_lr = initial_lr / 1000
    )

    criterion = CombinedLoss(alpha=0.85, clip_pred=6.0)

    # SWA state collection
    swa_states: list[dict] = []

    # Training loop
    best_score = -float("inf")
    best_state = None
    patience_counter = 0
    ema_loss = None  # Exponential moving average of loss for better tracking

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        optimizer.zero_grad()
        for batch_idx, (features, targets, mask) in enumerate(tqdm(
            train_loader, desc=f"Fold {fold_idx+1} Ep {epoch}", leave=False
        )):
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # Mixup augmentation
            if MIXUP_ALPHA > 0:
                features, targets, mask = mixup_batch(features, targets, mask, MIXUP_ALPHA)

            predictions, _, _ = model(features)
            loss = criterion(predictions, targets, mask)
            loss = loss / ACCUM_STEPS  # Scale for gradient accumulation

            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                scheduler.step()  # OneCycleLR steps per optimizer step
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS  # Unscale for logging
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t_start

        # EMA loss tracking
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * avg_loss

        # Validate
        val_scores = validate(model, valid_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Ep {epoch}: loss={avg_loss:.4f} (ema={ema_loss:.4f}), "
            f"wpc={val_scores['weighted_pearson']:.4f}, "
            f"t0={val_scores['t0']:.4f}, t1={val_scores['t1']:.4f}, "
            f"lr={current_lr:.2e}, "
            f"time={elapsed:.0f}s"
        )

        # SWA: collect model states from late epochs
        if epoch >= SWA_START_EPOCH:
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        # Save best + early stopping
        if val_scores["weighted_pearson"] > best_score:
            best_score = val_scores["weighted_pearson"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    # Apply SWA if we collected enough states
    if len(swa_states) >= 3:
        print(f"  🔄 Applying SWA over {len(swa_states)} states...")
        swa_state = {}
        for key in swa_states[0]:
            swa_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(dim=0)

        # Check if SWA improves over best
        model.cpu()
        model.load_state_dict(swa_state)
        model.to(device)
        swa_scores = validate(model, valid_loader, device)
        print(f"  SWA score: {swa_scores['weighted_pearson']:.4f} (best single: {best_score:.4f})")

        if swa_scores["weighted_pearson"] > best_score:
            best_score = swa_scores["weighted_pearson"]
            best_state = swa_state
            print(f"  ✅ SWA improved! Using SWA weights.")
        else:
            print(f"  ℹ️ SWA did not improve. Keeping best single checkpoint.")

    print(f"  ✅ Fold {fold_idx+1} Best: {best_score:.6f}")
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_score, best_state


import argparse

def train() -> None:
    """Main training loop — V5 with LSTM + attention + skip + SWA + K-Fold CV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run only specific fold (0-4)")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU (slow)")

    print(f"Device: {device}")

    # Load datasets — combine train + valid for CV
    ds1 = LOBDataset(TRAIN_PATH)
    ds2 = LOBDataset(VALID_PATH)
    full_dataset = ConcatDataset([ds1, ds2])

    total_len = len(full_dataset)
    indices = np.arange(total_len)

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(indices)):
        if args.fold is not None and fold_idx != args.fold:
            continue

        train_ds = Subset(full_dataset, train_idx)
        valid_ds = Subset(full_dataset, valid_idx)

        score, state = train_fold(fold_idx, train_ds, valid_ds, device)
        fold_scores.append(score)

        # Save fold checkpoint
        ckpt_path = os.path.join(CURRENT_DIR, f"model_fold_{fold_idx}.pt")
        torch.save(state, ckpt_path)

    # Summary
    if args.fold is None:
        print(f"\n{'='*50}")
        print(f"CV Scores: {fold_scores}")
        if fold_scores:
            print(f"Mean Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")

    # Export ALL fold models to ONNX
    model = LOBPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    for fold_idx in range(N_FOLDS):
        fold_ckpt = os.path.join(CURRENT_DIR, f"model_fold_{fold_idx}.pt")
        if os.path.exists(fold_ckpt):
            state = torch.load(fold_ckpt, map_location="cpu")
            model.load_state_dict(state)
            save_path = os.path.join(CURRENT_DIR, f"model_fold_{fold_idx}.onnx")
            export_to_onnx(model, save_path)


if __name__ == "__main__":
    train()
