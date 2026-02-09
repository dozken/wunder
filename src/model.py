"""
GRU model definition for Wunder Predictorium.
Architecture: Input(32) → GRU(hidden=128, layers=2) → FC(128→64) → FC(64→2)
"""
import torch
import torch.nn as nn


class LOBPredictor(nn.Module):
    """
    GRU-based model for LOB price movement prediction.

    Processes market state sequences step-by-step and outputs
    predictions for t0 and t1 targets.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional hidden state (num_layers, batch, hidden_dim)

        Returns:
            predictions: (batch, seq_len, 2) — t0/t1 predictions for each step
            hidden: Updated hidden state
        """
        # Normalize input
        x = self.input_norm(x)

        # GRU forward
        gru_out, hidden = self.gru(x, hidden)

        # Predict at every step
        predictions = self.head(gru_out)

        return predictions, hidden
