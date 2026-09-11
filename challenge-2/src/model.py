"""
LSTM model definition for Wunder Predictorium — V5.
Architecture:
    Input(32) → FeatureEngineer(→101) → LSTM(hidden=256, layers=3)
    → SkipConnection(input→64) → LayerNorm
    → FC(320→256) → GELU → Dropout → FC(256→128) → GELU → Dropout → FC(128→2)

V5 changes from V4:
  - Skip connection from engineered features to prediction head
  - Enhanced feature engineering (92 → 101 features)
  - Micro-price, trade intensity, relative spread depth, net order flow
  - Herfindahl concentration index for volume distribution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEngineer(nn.Module):
    """
    Computes derived LOB features from raw 32-dim state.
    Runs inside the ONNX graph.

    Input features layout (32):
      p0..p5   (bid prices)     [0:6]
      p6..p11  (ask prices)     [6:12]
      v0..v5   (bid volumes)    [12:18]
      v6..v11  (ask volumes)    [18:24]
      dp0..dp3 (trade prices)   [24:28]
      dv0..dv3 (trade volumes)  [28:32]

    Derived features (69):
      spread_0..5: p_ask - p_bid                              [6]
      vimb_0..5: (v_bid - v_ask) / (v_bid + v_ask + eps)      [6]
      ask_dist_mid_0..5: ask - mid                             [6]
      bid_dist_mid_0..5: mid - bid                             [6]
      bid_cum_v_0..5: cumulative bid volume                    [6]
      ask_cum_v_0..5: cumulative ask volume                    [6]
      bid_step_0..4: bid price differences between levels      [5]
      ask_step_0..4: ask price differences between levels      [5]
      weighted_mid: volume-weighted mid price                   [1]
      pressure_0..5: bid_v / (bid_v + ask_v + eps)             [6]
      trade_imbalance_raw: buy - sell flow                      [1]
      trade_imbalance_norm: normalized version                  [1]
      spread_pct: spread as fraction of mid                    [1]
      total_bid_v: sum of bid volumes                          [1]
      total_ask_v: sum of ask volumes                          [1]
      vol_ratio: total_bid_v / (total_bid_v + total_ask_v)     [1]
      depth_imbalance: (cum_bid - cum_ask) / (cum_bid + cum_ask) [1]
      mid_vs_micro: mid - micro_price (price direction signal)  [1]
      rel_spread_1..4: spread[i]/spread[0] (depth shape)        [4]
      trade_intensity: sum of all trade volumes                  [1]
      net_order_flow: signed trade volume difference             [1]
      bid_v_hhi: concentration measure of bid volumes (Herfindahl)  [1]
      ask_v_hhi: concentration measure of ask volumes (Herfindahl)  [1]
    Total = 32 + 69 = 101
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bid_p = x[..., 0:6]
        ask_p = x[..., 6:12]
        bid_v = x[..., 12:18]
        ask_v = x[..., 18:24]
        dp = x[..., 24:28]
        dv = x[..., 28:32]

        # 1. Spread (Ask - Bid) — 6 features
        spread = ask_p - bid_p

        # 2. Volume Imbalance — 6 features
        v_sum = bid_v + ask_v + self.eps
        vimb = (bid_v - ask_v) / v_sum

        # 3. Mid-Price
        mid_price = (ask_p[..., 0:1] + bid_p[..., 0:1]) / 2

        # 4. Price Distances from Mid — 12 features
        ask_dist_mid = ask_p - mid_price
        bid_dist_mid = mid_price - bid_p

        # 5. Cumulative Volume — 12 features
        bid_cum_v = torch.cumsum(bid_v, dim=-1)
        ask_cum_v = torch.cumsum(ask_v, dim=-1)

        # 6. Inter-level Price Steps — 10 features
        bid_step = bid_p[..., :-1] - bid_p[..., 1:]
        ask_step = ask_p[..., 1:] - ask_p[..., :-1]

        # 7. Volume-weighted mid price — 1 feature
        best_bid_v = bid_v[..., 0:1]
        best_ask_v = ask_v[..., 0:1]
        weighted_mid = (bid_p[..., 0:1] * best_ask_v + ask_p[..., 0:1] * best_bid_v) / (
            best_bid_v + best_ask_v + self.eps
        )

        # 8. Pressure ratio per level — 6 features
        pressure = bid_v / (bid_v + ask_v + self.eps)

        # 9. Trade imbalance — 2 features
        buy_flow = (dv[..., 0:1] * dp[..., 0:1] + dv[..., 1:2] * dp[..., 1:2])
        sell_flow = (dv[..., 2:3] * dp[..., 2:3] + dv[..., 3:4] * dp[..., 3:4])
        trade_imb_raw = buy_flow - sell_flow
        trade_imb_norm = (buy_flow - sell_flow) / (buy_flow + sell_flow + self.eps)

        # 10. Spread as fraction of mid — 1 feature
        spread_pct = spread[..., 0:1] / (mid_price + self.eps)

        # 11. Total volume sums — 2 features
        total_bid_v = bid_v.sum(dim=-1, keepdim=True)
        total_ask_v = ask_v.sum(dim=-1, keepdim=True)

        # 12. Volume ratio — 1 feature
        vol_ratio = total_bid_v / (total_bid_v + total_ask_v + self.eps)

        # 13. Depth imbalance — 1 feature
        cum_bid_total = bid_cum_v[..., -1:]
        cum_ask_total = ask_cum_v[..., -1:]
        depth_imbalance = (cum_bid_total - cum_ask_total) / (cum_bid_total + cum_ask_total + self.eps)

        # === V5 NEW features ===

        # 14. Micro-price — 1 feature
        micro_price = (bid_p[..., 0:1] * best_ask_v + ask_p[..., 0:1] * best_bid_v) / (
            best_bid_v + best_ask_v + self.eps
        )
        mid_vs_micro = mid_price - micro_price

        # 15. Relative spread depth — 4 features
        best_spread = spread[..., 0:1] + self.eps
        rel_spread = spread[..., 1:5] / best_spread

        # 16. Trade intensity — 1 feature
        trade_intensity = dv.sum(dim=-1, keepdim=True)

        # 17. Net order flow — 1 feature
        net_flow = (dv[..., 0:1] + dv[..., 1:2]) - (dv[..., 2:3] + dv[..., 3:4])

        # 18. Volume concentration (Herfindahl index) — 2 features
        bid_v_norm = bid_v / (total_bid_v + self.eps)
        bid_v_hhi = (bid_v_norm ** 2).sum(dim=-1, keepdim=True)
        ask_v_norm = ask_v / (total_ask_v + self.eps)
        ask_v_hhi = (ask_v_norm ** 2).sum(dim=-1, keepdim=True)

        return torch.cat([
            x,                  # 32
            spread,             # 6
            vimb,               # 6
            ask_dist_mid,       # 6
            bid_dist_mid,       # 6
            bid_cum_v,          # 6
            ask_cum_v,          # 6
            bid_step,           # 5
            ask_step,           # 5
            weighted_mid,       # 1
            pressure,           # 6
            trade_imb_raw,      # 1
            trade_imb_norm,     # 1
            spread_pct,         # 1
            total_bid_v,        # 1
            total_ask_v,        # 1
            vol_ratio,          # 1
            depth_imbalance,    # 1
            # V5 new
            mid_vs_micro,       # 1
            rel_spread,         # 4
            trade_intensity,    # 1
            net_flow,           # 1
            bid_v_hhi,          # 1
            ask_v_hhi,          # 1
        ], dim=-1)


ENGINEERED_DIM = 101  # Total features after FeatureEngineer


class LOBPredictor(nn.Module):
    """
    LSTM-based model with skip connection and enhanced features.
    V5: 101 features, skip connection, improved training pipeline.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Feature engineering (32 → 101)
        self.feature_eng = FeatureEngineer()

        # Input normalization
        self.input_norm = nn.LayerNorm(ENGINEERED_DIM)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=ENGINEERED_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Skip connection: project raw features down to manageable dim
        skip_dim = 64
        self.skip_proj = nn.Linear(ENGINEERED_DIM, skip_dim)

        # Pre-head normalization
        head_input_dim = hidden_dim + skip_dim  # lstm_out + skip
        self.head_norm = nn.LayerNorm(head_input_dim)

        # Deeper prediction head
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, 32)
            hidden: Optional hidden state (num_layers, batch, hidden_dim)
            cell: Optional cell state (num_layers, batch, hidden_dim) [LSTM only]

        Returns:
            predictions: (batch, seq_len, 2)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        # Feature engineering (32 → 101)
        feat = self.feature_eng(x)

        # Normalize input
        feat_normed = self.input_norm(feat)

        # LSTM forward
        if hidden is not None and cell is not None:
            lstm_out, (hidden, cell) = self.lstm(feat_normed, (hidden, cell))
        else:
            lstm_out, (hidden, cell) = self.lstm(feat_normed)

        # Skip connection from raw features
        skip = self.skip_proj(feat_normed)

        # Concatenate: lstm_out + skip
        combined = torch.cat([lstm_out, skip], dim=-1)

        # Normalize before head
        combined = self.head_norm(combined)

        # Predict at every step
        predictions = self.head(combined)

        return predictions, hidden, cell
