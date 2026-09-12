"""Stateful recurrent predictor.

    state(112) -> standardise -> Linear -> GELU -> LayerNorm      (input projection)
              -> GRU / LSTM stack                                  (recurrent core)
              -> concat(rnn_out, projected input) -> LayerNorm
              -> Linear -> GELU -> Linear(2)                       (head)

The whole thing runs one row at a time at inference, so all sequence state
lives in a single tensor that is passed in and out of forward(). That tensor
is the ONNX interface: GRU state is (layers, 1, hidden); LSTM packs h and c as
(2 * layers, 1, hidden).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

N_FEATURES = 112
N_TARGETS = 2


@dataclass
class ModelConfig:
    rnn: str = "gru"            # "gru" | "lstm"
    hidden: int = 256
    layers: int = 2
    proj: int = 128             # input projection width, 0 disables
    dropout: float = 0.1
    input_clip: float = 10.0    # clamp standardised inputs to +-clip
    diff: bool = False          # also feed x_t - x_{t-1}; previous row rides in the state
    out: int = N_TARGETS        # output channels (1 for the scored-row predictor)

    def to_dict(self):
        return asdict(self)


class Predictor(nn.Module):
    def __init__(self, cfg: ModelConfig, mean=None, std=None):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.zeros(N_FEATURES) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(N_FEATURES) if std is None else torch.as_tensor(std, dtype=torch.float32))

        in_dim = N_FEATURES * (2 if cfg.diff else 1)
        if cfg.diff and cfg.hidden < N_FEATURES:
            raise ValueError("diff needs hidden >= 112 so the previous row fits in one state slot")
        if cfg.proj > 0:
            self.inproj = nn.Sequential(nn.Linear(in_dim, cfg.proj), nn.GELU(), nn.LayerNorm(cfg.proj))
            rnn_in = cfg.proj
        else:
            self.inproj = nn.Identity()
            rnn_in = in_dim

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cfg.rnn]
        self.rnn = rnn_cls(rnn_in, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                           dropout=cfg.dropout if cfg.layers > 1 else 0.0)

        head_in = cfg.hidden + rnn_in
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(nn.Linear(head_in, cfg.hidden), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, cfg.out))

    @property
    def rnn_state_layers(self) -> int:
        return self.cfg.layers * (2 if self.cfg.rnn == "lstm" else 1)

    @property
    def state_layers(self) -> int:
        # one extra slot holds the previous (standardised) row when diff is on
        return self.rnn_state_layers + (1 if self.cfg.diff else 0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(self.state_layers, batch, self.cfg.hidden, device=device)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """x: (B, T, 112) raw features; state: (state_layers, B, hidden)."""
        x = (x - self.mean) / self.std
        x = x.clamp(-self.cfg.input_clip, self.cfg.input_clip)
        if self.cfg.diff:
            rnn_state, carry = state[:self.rnn_state_layers], state[self.rnn_state_layers]
            prev = carry[:, :N_FEATURES].unsqueeze(1)                       # (B, 1, 112)
            shifted = torch.cat([prev, x[:, :-1]], dim=1)
            x = torch.cat([x, x - shifted], dim=-1)
            new_carry = torch.cat([x[:, -1, :N_FEATURES],
                                   carry[:, N_FEATURES:] * 0], dim=-1).unsqueeze(0)
        else:
            rnn_state = state
        z = self.inproj(x)
        if self.cfg.rnn == "lstm":
            h, c = rnn_state.chunk(2, dim=0)
            out, (h, c) = self.rnn(z, (h.contiguous(), c.contiguous()))
            rnn_state = torch.cat([h, c], dim=0)
        else:
            out, rnn_state = self.rnn(z, rnn_state)
        state = torch.cat([rnn_state, new_carry], dim=0) if self.cfg.diff else rnn_state
        pred = self.head(self.head_norm(torch.cat([out, z], dim=-1)))
        return pred, state


def build(cfg_dict: dict, mean=None, std=None) -> Predictor:
    return Predictor(ModelConfig(**cfg_dict), mean, std)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
