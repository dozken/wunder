"""Export fold checkpoint .pt files to ONNX for LSTM V5 architecture."""
import os
import torch
from src.model import LOBPredictor

# Configuration — must match train.py
INPUT_DIM = 32
HIDDEN_DIM = 256
NUM_LAYERS = 3
OUTPUT_DIM = 2
DROPOUT = 0.0  # No dropout for export


def export_fold_models():
    print("Exporting available fold checkpoints to ONNX (LSTM V5)...")
    device = torch.device("cpu")

    # Find all fold checkpoints
    fold_files = [f for f in os.listdir("src") if f.startswith("model_fold_") and f.endswith(".pt")]
    fold_files.sort()

    if not fold_files:
        print("No fold checkpoints found!")
        return

    dummy_x = torch.randn(1, 1, INPUT_DIM)
    dummy_h = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)
    dummy_c = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)

    for f in fold_files:
        pt_path = os.path.join("src", f)
        onnx_name = f.replace(".pt", ".onnx")
        onnx_path = os.path.join("src", onnx_name)

        print(f"Processing {pt_path} -> {onnx_path}")

        # Load model
        model = LOBPredictor(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT
        )
        state_dict = torch.load(pt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Export with LSTM states (hidden + cell)
        torch.onnx.export(
            model,
            (dummy_x, dummy_h, dummy_c),
            onnx_path,
            input_names=["input", "hidden_in", "cell_in"],
            output_names=["prediction", "hidden_out", "cell_out"],
            dynamic_axes={
                "input": {1: "seq_len"},
                "prediction": {1: "seq_len"},
            },
            opset_version=17,
        )

        # Clean up external data if any
        if os.path.exists(onnx_path + ".data"):
            os.remove(onnx_path + ".data")

        print(f"✅ Exported {onnx_name}")


if __name__ == "__main__":
    export_fold_models()
