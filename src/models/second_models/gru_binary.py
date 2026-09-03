import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

class BinaryFocalLoss(nn.Module):
    """클래스 불균형 해결을 위한 Focal Loss (gamma=2.0, alpha=0.75)"""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss
        return loss.mean()

class ConvBiGRUBinaryNet(nn.Module):
    def __init__(self, input_dim: int = 69, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=64, C=69)
        x_conv = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(x_conv)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        pooled = torch.sum(gru_out * attn_w, dim=1)
        logits = self.fc(pooled).squeeze(-1) # (B,)
        return logits

class TemporalGRUBinaryClassifier:
    """2단계: PyTorch 1D-CNN + Bi-GRU + Attention + Focal Loss 정상/오류 판별기"""

    def __init__(self, epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvBiGRUBinaryNet(input_dim=69).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TemporalGRUBinaryClassifier":
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        probs_pos = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                p = torch.sigmoid(logits).cpu().numpy()
                probs_pos.append(p)
        probs_pos = np.concatenate(probs_pos, axis=0)
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(np.int64)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        inference_time_ms = ((time.time() - t0) / len(X_test)) * 1000.0

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        pos_f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        pos_prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        pos_rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
        except Exception:
            roc_auc, pr_auc = 0.5, 0.0

        return {
            'model_name': '1D-CNN + Bi-GRU (Focal)',
            'split_name': split_name,
            'num_samples': len(y_test),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'pos_f1': pos_f1,
            'pos_precision': pos_prec,
            'pos_recall': pos_rec,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds,
            'probabilities': probs
        }
