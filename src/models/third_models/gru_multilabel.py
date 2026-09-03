import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

class ConvBiGRUMultiHeadNet(nn.Module):
    def __init__(self, input_dim: int = 69, num_labels: int = 4, hidden_dim: int = 128):
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
            nn.Linear(64, num_labels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=64, C=69)
        x_conv = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(x_conv)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        pooled = torch.sum(gru_out * attn_w, dim=1)
        logits = self.fc(pooled) # (B, num_labels)
        return logits

class TemporalGRUMultiLabelClassifier:
    """3단계: PyTorch 1D-CNN + Bi-GRU + Attention 다중 라벨 세부 결함 진단기"""

    def __init__(self, num_labels: int = 4, epochs: int = 15, batch_size: int = 32, lr: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.model = ConvBiGRUMultiHeadNet(input_dim=69, num_labels=num_labels).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "TemporalGRUMultiLabelClassifier":
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
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
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int64)

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0

        subset_acc = accuracy_score(Y_test, preds)
        h_loss = hamming_loss(Y_test, preds)
        micro_f1 = f1_score(Y_test, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(Y_test, preds, average='macro', zero_division=0)

        return {
            'model_name': '1D-CNN + Bi-GRU (Multi-Head)',
            'split_name': split_name,
            'num_samples': len(Y_test),
            'subset_accuracy': subset_acc,
            'hamming_loss': h_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds
        }
