# -*- coding: utf-8 -*-
"""
Lightweight Temporal Pose-Transformer (Format B: 16F x 92D)
16-frame multi-head self-attention capturing global phase relationships.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional

from src.models.second_models.losses import AsymmetricLoss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TemporalPoseTransformerNet(nn.Module):
    def __init__(self, in_features: int = 92, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=16, C=92)
        B = x.size(0)
        h = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) # (B, 17, d_model)
        h = self.pos_encoder(h)
        out = self.transformer(h)
        cls_out = out[:, 0, :] # (B, d_model)
        logits = self.head(cls_out).squeeze(-1)
        return logits

class TemporalPoseTransformerBinaryClassifier:
    def __init__(
        self,
        in_features: int = 92,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        batch_size: int = 64,
        random_seed: int = 42
    ):
        torch.manual_seed(random_seed)
        self.in_features = in_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cpu')

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Optional[list] = None):
        self.in_features = X_train.shape[2]
        self.model = TemporalPoseTransformerNet(
            in_features=self.in_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)

        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t_x = torch.tensor(X_train, dtype=torch.float32)
        t_y = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(t_x, t_y), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        t_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(t_x)
            pos_probs = torch.sigmoid(logits).cpu().numpy()
        pos_probs = np.clip(pos_probs, 1e-6, 1.0 - 1e-6)
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(np.int64)
