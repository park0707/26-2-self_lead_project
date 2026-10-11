# -*- coding: utf-8 -*-
"""
Dilated TCN with Asymmetric Loss (Format B: 16F x 92D)
High-speed causal temporal convolutions with dilation 1, 2, 4.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional

from src.models.second_models.losses import AsymmetricLoss

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal trim
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.drop1(self.act1(out))

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.drop2(self.act2(out))
        return self.act2(out + res)

class DilatedTCNNet(nn.Module):
    def __init__(self, in_channels: int = 92, num_channels: list = [64, 64, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_c = in_channels if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            layers.append(TemporalBlock(in_c, out_c, kernel_size=kernel_size, dilation=dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        feats = self.network(x)
        pooled = self.pool(feats).squeeze(-1)
        logits = self.head(pooled).squeeze(-1)
        return logits

class DilatedTCNBinaryClassifier:
    def __init__(
        self,
        in_channels: int = 92,
        num_channels: list = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        batch_size: int = 64,
        random_seed: int = 42
    ):
        torch.manual_seed(random_seed)
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cpu')

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Optional[list] = None):
        # X_train: (N, 16, 92)
        self.in_channels = X_train.shape[2]
        self.model = DilatedTCNNet(
            in_channels=self.in_channels,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
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
