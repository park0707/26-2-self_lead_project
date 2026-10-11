# =========================================================================================================
# [모듈 설명]: 2단계 동작 정상/오류 판별 - 1D-CNN + Bi-GRU + Attention (Focal Loss) 이진 분류기
# [입력 데이터]: 64 프레임 x 69차원 시계열 텐서 (Format B)
# [출력 목표]: 0=오류(비정상 동작), 1=정상 동작 이진 분류 및 최적 임계값(θ*) 적용
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~15초):
#      python -m src.models.second_models.gru_binary
#      또는 python -m src.models.second_models.gru_binary --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~60초):
#      python -m src.models.second_models.gru_binary --mode compare
# =========================================================================================================

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BEST_PARAMS = {
    'conv_filters': 128,
    'epochs': 15,
    'hidden_dim': 64,
    'lr': 0.001,
    'num_layers': 2,
    'opt_threshold': 0.536506175994873
}

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
    def __init__(self, input_dim: int = 69, conv_filters: int = 128, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gru = nn.GRU(
            input_size=conv_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
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

    def __init__(self, input_dim: int = 92, conv_filters: int = 128, hidden_dim: int = 128, num_layers: int = 2, epochs: int = 2, batch_size: int = 256, lr: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = ConvBiGRUBinaryNet(input_dim=input_dim, conv_filters=conv_filters, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TemporalGRUBinaryClassifier":
        if X_train.shape[2] != self.input_dim:
            self.input_dim = X_train.shape[2]
            self.model = ConvBiGRUBinaryNet(input_dim=self.input_dim, conv_filters=self.conv_filters, hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)
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

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from src.models.second_models.dataset import Stage2DatasetLoader
    from src.models.second_models.metrics_helper import Stage2MetricsHelper

    parser = argparse.ArgumentParser(description="Stage 2: 1D-CNN + Bi-GRU (Focal) Binary Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    args = parser.parse_args()

    splits = Stage2DatasetLoader.load_all_dual_splits()
    X_train, y_train = splits['train']['X_Format_B'], splits['train']['y']

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print("    [2단계: 1D-CNN + Bi-GRU (Focal) 정상/오류 판별 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        fit_params = {k: v for k, v in BEST_PARAMS.items() if k != 'opt_threshold'}
        opt_thresh = BEST_PARAMS.get('opt_threshold', 0.5)
        clf = TemporalGRUBinaryClassifier(**fit_params)
        clf.fit(X_train, y_train)
        best_clf = clf
    else:
        print("\n" + "=" * 105)
        print("    [2단계: 1D-CNN + Bi-GRU (Focal) 정상/오류 판별 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        from src.models.second_models.tune_all import run_stage2_grid_search_gru
        def_trial, other_5, best_trial, best_clf = run_stage2_grid_search_gru(splits)
        opt_thresh = best_trial['params'].get('opt_threshold', 0.5)

    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    print("\n" + "=" * 115)
    print(f"      [2단계 1D-CNN + Bi-GRU 다중 분할 평가 결과 (최적 임계값 θ* = {opt_thresh:.3f})]")
    print("=" * 115)
    header = f"{'평가 데이터셋 분할':<38} | {'Macro F1':^10} | {'Pos-F1(정상)':^12} | {'Safety F2':^11} | {'PR-AUC':^9} | {'MCC':^8} | {'G-Mean':^8}"
    print(header)
    print("-" * 115)

    for disp_name, key in split_map:
        X_ev = splits[key]['X_Format_B']
        y_ev = splits[key]['y']
        probs = best_clf.predict_proba(X_ev)[:, 1]
        res = Stage2MetricsHelper.evaluate_predictions(y_ev, probs, threshold=opt_thresh)
        print(f"{disp_name:<38} | {res['macro_f1']*100:>8.2f}% | {res['pos_f1']*100:>10.2f}% | {res['pos_f2_safety']*100:>9.2f}% | {res['pr_auc']:>8.3f} | {res['mcc']:>7.3f} | {res['g_mean']:>7.3f}")
    print("=" * 115)

    if args.mode == 'compare':
        print("\n" + "=" * 115)
        print("                  [하이퍼파라미터 튜닝 전후 3중 비교표 (Default vs Random 5 vs Best)]")
        print("=" * 115)
        cmp_header = f"{'구분 (Trial Type)':<20} | {'Strict Macro F1':^16} | {'Strict PR-AUC':^14} | {'파라미터 구성 요약':<45}"
        print(cmp_header)
        print("-" * 115)
        print(f"{'기본 모델 (Default)':<20} | {def_trial['strict_f1']*100:>14.2f}% | {def_trial['strict_pr_auc']:>13.3f} | {str(def_trial['params'])[:45]}")
        for i, t in enumerate(other_5, 1):
            p_str = f"filters={t['params'].get('conv_filters')}, hidden={t['params'].get('hidden_dim')}, layers={t['params'].get('num_layers')}, lr={t['params'].get('lr')}"
            print(f"{f'랜덤 후보 #{i}':<20} | {t['strict_f1']*100:>14.2f}% | {t['strict_pr_auc']:>13.3f} | {p_str:<45}")
        p_best = f"filters={best_trial['params'].get('conv_filters')}, hidden={best_trial['params'].get('hidden_dim')}, layers={best_trial['params'].get('num_layers')}, lr={best_trial['params'].get('lr')}"
        print("-" * 115)
        imp = (best_trial['strict_f1'] - def_trial['strict_f1']) * 100
        print(f"{'★ 최적 모델 (Best)':<20} | {best_trial['strict_f1']*100:>14.2f}% | {best_trial['strict_pr_auc']:>13.3f} | {p_best:<45} (성능 향상: +{imp:.2f}%p)")
        print("=" * 115)

