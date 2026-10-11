# -*- coding: utf-8 -*-
"""
Tabular SE-ResNet Binary Classifier (Format A: 161D)
Modern MLP with LayerNorm, Mish, Squeeze-and-Excitation, and Asymmetric Loss.
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

class SqueezeExcitation1D(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        red_dim = max(8, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim, red_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red_dim, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w

class TabularResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.Mish()
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.se = SqueezeExcitation1D(out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.drop(self.act(self.fc1(self.ln1(x))))
        h = self.fc2(self.ln2(h))
        h = self.se(h)
        return self.act(h + res)

class TabularSEResNet(nn.Module):
    def __init__(self, in_features: int = 161, hidden_dim: int = 128, num_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.blocks = nn.ModuleList([
            TabularResBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        logits = self.head(h).squeeze(-1)
        return logits

class TabularSEResNetBinaryClassifier:
    def __init__(
        self,
        in_features: int = 161,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64,
        random_seed: int = 42
    ):
        torch.manual_seed(random_seed)
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.mean = None
        self.std = None
        self.device = torch.device('cpu')

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.std = np.std(X, axis=0, keepdims=True) + 1e-6
        return (X - self.mean) / self.std

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Optional[list] = None):
        X_norm = self._normalize(X_train, fit=True)
        self.in_features = X_train.shape[1]
        self.model = TabularSEResNet(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout
        ).to(self.device)

        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t_x = torch.tensor(X_norm, dtype=torch.float32)
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
        X_norm = self._normalize(X, fit=False)
        t_x = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(t_x)
            pos_probs = torch.sigmoid(logits).cpu().numpy()
        pos_probs = np.clip(pos_probs, 1e-6, 1.0 - 1e-6)
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(np.int64)
