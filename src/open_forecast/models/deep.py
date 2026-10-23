# set up the deep learning models
"""PyTorch adapters for LSTM, Cross-Attention Transformer, and PatchTST."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..contracts import SequencePartition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MultiZoneLSTM(nn.Module):
    def __init__(
        self,
        features: int,
        zones: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.zones = zones
        self.lstm = nn.LSTM(
            input_size=features * zones,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, zones),
        )

    def forward(self, x: Tensor) -> Tensor:
        # [batch, zones, lookback, features] -> shared multivariate sequence
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        attended, _ = self.attention(
            query,
            context,
            context,
            need_weights=False,
        )
        x = self.norm1(query + attended)
        return self.norm2(x + self.feedforward(x))


class MultiZoneCrossAttentionTransformer(nn.Module):
    """
    Proposed cross-zone architecture.

    Each zone keeps its own temporal stream. At every block it attends to the
    mean representation of the *other* zones; this generalizes the original
    bidirectional two-zone design without discarding Zone 3.
    """

    def __init__(
        self,
        features: int,
        zones: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.zones = zones
        self.input_projection = nn.Linear(features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_encoders = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    copy.deepcopy(encoder_layer),
                    num_layers=1,
                )
                for _ in range(zones)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, nhead, dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * zones),
            nn.Linear(d_model * zones, zones),
        )

    def forward(self, x: Tensor) -> Tensor:
        streams = [
            self.self_encoders[z](
                self.input_projection(x[:, z])
            )
            for z in range(self.zones)
        ]

        for block in self.cross_blocks:
            updated = []
            for zone, stream in enumerate(streams):
                others = [
                    candidate
                    for i, candidate in enumerate(streams)
                    if i != zone
                ]
                other = torch.stack(others).mean(dim=0)
                updated.append(block(stream, other))
            streams = updated

        pooled = torch.cat(
            [stream.mean(dim=1) for stream in streams],
            dim=1,
        )
        return self.head(pooled)


class PatchTSTStyle(nn.Module):
    """A compact PatchTST-style baseline with shared weights across zones."""

    def __init__(
        self,
        features: int,
        zones: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        patch_length: int,
        stride: int,
        dropout: float,
    ):
        super().__init__()
        self.zones = zones
        self.patch_length = patch_length
        self.stride = stride

        self.patch_projection = nn.Linear(
            patch_length * features,
            d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Build local temporal patches separately per zone, then share encoder weights.
        batch, zones, lookback, features = x.shape
        if lookback < self.patch_length:
            raise ValueError("Lookback must be at least patch_length.")

        patches = x.unfold(
            dimension=2,
            size=self.patch_length,
            step=self.stride,
        )
        # [B, Z, patches, F, patch] -> [B*Z, patches, patch*F]
        patches = patches.permute(0, 1, 2, 4, 3).reshape(
            batch * zones,
            patches.shape[2],
            self.patch_length * features,
        )
        encoded = self.encoder(
            self.patch_projection(patches)
        )
        output = self.head(
            encoded.mean(dim=1)
        ).reshape(batch, zones)
        return output


class DeepForecaster:
    """Training adapter with identical early-stopping behavior for all deep models."""

    def __init__(
        self,
        name: str,
        parameters: dict,
        training: dict,
        seed: int,
    ):
        self.name = name
        self.parameters = parameters
        self.training = training
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: nn.Module | None = None

    def _build(self, partition: SequencePartition) -> nn.Module:
        features = partition.x.shape[-1]
        zones = partition.y.shape[-1]

        if self.name == "lstm":
            return MultiZoneLSTM(
                features,
                zones,
                **self.parameters,
            )
        if self.name == "cross_attention_transformer":
            return MultiZoneCrossAttentionTransformer(
                features,
                zones,
                **self.parameters,
            )
        if self.name == "patchtst":
            return PatchTSTStyle(
                features,
                zones,
                **self.parameters,
            )

        raise ValueError(f"Unknown deep model: {self.name}")

    def fit(
        self,
        train: SequencePartition,
        validation: SequencePartition,
    ) -> "DeepForecaster":
        set_seed(self.seed)
        self.model = self._build(train).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training["learning_rate"],
            weight_decay=self.training["weight_decay"],
        )
        loss_function = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train.x),
                torch.from_numpy(train.y),
            ),
            batch_size=self.training["batch_size"],
            shuffle=True,
        )

        val_x = torch.from_numpy(validation.x).to(self.device)
        val_y = torch.from_numpy(validation.y).to(self.device)

        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        waiting = 0

        for _ in range(self.training["epochs"]):
            self.model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                prediction = self.model(
                    batch_x.to(self.device)
                )
                loss = loss_function(
                    prediction,
                    batch_y.to(self.device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                )
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_function(
                    self.model(val_x),
                    val_y,
                ).item()

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = copy.deepcopy(
                    self.model.state_dict()
                )
                waiting = 0
            else:
                waiting += 1

            if waiting >= self.training["early_stopping_patience"]:
                break

        self.model.load_state_dict(best_state)
        return self

    def predict(
        self,
        partition: SequencePartition,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before predict.")

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(partition.x).to(self.device)
            )
        return output.cpu().numpy()
