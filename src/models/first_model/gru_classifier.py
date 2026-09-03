# [실행 명령어]: python -m src.models.first_model.gru_classifier

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper

class ConvBiGRUActionNet(nn.Module):
    """1단계: 1D-CNN + Bi-GRU 시계열 심층 신경망 (Format B (64, 69) 입력)"""
    def __init__(self, input_dim: int = 69, seq_len: int = 64, num_classes: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
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
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1(x_conv)
        x_conv = self.conv2(x_conv)
        x_gru_in = x_conv.transpose(1, 2)
        gru_out, _ = self.gru(x_gru_in)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        pooled = torch.sum(gru_out * attn_weights, dim=1)
        logits = self.fc(pooled)
        return logits

class TemporalGRUActionClassifier:
    """PyTorch 1D-CNN + Bi-GRU 시계열 분류기 Wrapper"""
    def __init__(self, num_classes: int = 9, epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = ConvBiGRUActionNet(input_dim=69, seq_len=64, num_classes=num_classes).to(self.device)
        self.hyperparams = {
            'architecture': '1D-CNN (2 Layers) + Bi-GRU (2 Layers, H=128) + Self-Attention',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': 'AdamW + CosineAnnealingLR',
            'loss': 'CrossEntropyLoss',
            'device': str(self.device)
        }
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TemporalGRUActionClassifier":
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                p = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds, axis=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0
        metrics = Stage1MetricsHelper.evaluate_predictions(y_test, preds, inference_time_ms=inference_time_ms)
        metrics['model_name'] = '1D-CNN + Bi-GRU'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y_test)
        metrics['predictions'] = preds
        return metrics

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    data = Stage1DatasetLoader.load_all_dual_splits()
    X_train, y_train = data['train']['X_Format_B'], data['train']['y_action']

    clf = TemporalGRUActionClassifier(epochs=15, batch_size=64)
    Stage1MetricsHelper.print_model_header(
        model_name="1D-CNN + Bi-GRU + Self-Attention Temporal Neural Network",
        hyperparams=clf.hyperparams,
        train_samples=X_train.shape[0],
        feat_dim="Format B (64 프레임 x 69차원 시계열 텐서)"
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[*] 모델 학습 완료 (소요 시간: {time.time()-t0:.2f}초)")

    split_results = {}
    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    strict_res = None
    for display_name, key in split_map:
        res = clf.evaluate(data[key]['X_Format_B'], data[key]['y_action'], split_name=key)
        split_results[display_name] = res
        if key == 'test_strict':
            strict_res = res

    Stage1MetricsHelper.print_multi_split_summary(split_results)
    if strict_res is not None:
        Stage1MetricsHelper.print_class_breakdown_report(strict_res, data['test_strict']['y_action'])
        Stage1MetricsHelper.print_top_confusion_analysis(strict_res, top_k=3)
