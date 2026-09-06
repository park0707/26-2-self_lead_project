# =========================================================================================================
# [모듈 설명]: 3단계 세부 결함(Conditions) 다중 라벨 진단 - 1D-CNN + Bi-GRU + Attention 다중 헤드 분류기
# [입력 데이터]: 64 프레임 x 69차원 시계열 텐서 (Format B)
# [출력 목표]: 운동별 세부 결함 항목 (Conditions) 동시 다중 라벨 (Multi-label) 예측
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~15초):
#      python -m src.models.third_models.gru_multilabel
#      또는 python -m src.models.third_models.gru_multilabel --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~60초):
#      python -m src.models.third_models.gru_multilabel --mode compare
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
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

BEST_PARAMS = {   'conv_filters': 64,
    'epochs': 15,
    'hidden_dim': 128,
    'lr': 0.0005,
    'num_layers': 2}

class ConvBiGRUMultiHeadNet(nn.Module):
    def __init__(self, input_dim: int = 69, num_labels: int = 4, conv_filters: int = 128, hidden_dim: int = 128, num_layers: int = 2):
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

    def __init__(self, num_labels: int = 4, conv_filters: int = 128, hidden_dim: int = 128, num_layers: int = 2, epochs: int = 15, batch_size: int = 32, lr: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.model = ConvBiGRUMultiHeadNet(input_dim=69, num_labels=num_labels, conv_filters=conv_filters, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
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

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from src.models.third_models.dataset import Stage3DatasetLoader
    from src.models.third_models.metrics_helper import Stage3MetricsHelper

    parser = argparse.ArgumentParser(description="Stage 3: 1D-CNN + Bi-GRU Multi-label Defect Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    parser.add_argument('--exercise', type=str, default='01_바벨_스쿼트',
                        help="Target exercise folder (default: '01_바벨_스쿼트')")
    args = parser.parse_args()

    splits = Stage3DatasetLoader.load_all_splits_for_exercise(args.exercise)
    cond_names = splits['train']['cond_names']
    num_labels = splits['train']['num_conditions']
    X_train, Y_train = splits['train']['X_Format_B'], splits['train']['Y_multi']

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print(f"    [3단계: 1D-CNN + Bi-GRU {args.exercise} 결함 진단 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        clf = TemporalGRUMultiLabelClassifier(num_labels=num_labels, **BEST_PARAMS)
        clf.fit(X_train, Y_train)
        best_clf = clf
    else:
        print("\n" + "=" * 105)
        print(f"    [3단계: 1D-CNN + Bi-GRU {args.exercise} 결함 진단 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        from src.models.third_models.tune_all import run_stage3_grid_search_gru
        def_trial, other_5, best_trial, best_clf = run_stage3_grid_search_gru(splits)

    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    print("\n" + "=" * 115)
    print(f"      [3단계 1D-CNN + Bi-GRU ({args.exercise}) 세부 결함 진단 다중 분할 평가 결과]")
    print("=" * 115)
    header = f"{'평가 데이터셋 분할':<38} | {'Exact Match':^12} | {'Micro F1':^10} | {'Macro F1':^10} | {'Hamming Loss':^13} | {'Jaccard IoU':^12}"
    print(header)
    print("-" * 115)

    for disp_name, key in split_map:
        X_ev = splits[key]['X_Format_B']
        Y_ev = splits[key]['Y_multi']
        preds = best_clf.predict(X_ev)
        res = Stage3MetricsHelper.evaluate_predictions(Y_ev, preds, cond_names=cond_names)
        print(f"{disp_name:<38} | {res.get('subset_accuracy', 0)*100:>10.2f}% | {res.get('micro_f1', 0)*100:>8.2f}% | {res.get('macro_f1', 0)*100:>8.2f}% | {res.get('hamming_loss', 0):>12.4f} | {res.get('jaccard_samples_iou', 0)*100:>10.2f}%")
    print("=" * 115)

    if args.mode == 'compare':
        print("\n" + "=" * 115)
        print("                  [하이퍼파라미터 튜닝 전후 3중 비교표 (Default vs Random 5 vs Best)]")
        print("=" * 115)
        cmp_header = f"{'구분 (Trial Type)':<20} | {'Strict Micro F1':^16} | {'Strict Exact Match':^18} | {'파라미터 구성 요약':<45}"
        print(cmp_header)
        print("-" * 115)
        print(f"{'기본 모델 (Default)':<20} | {def_trial['strict_f1']*100:>14.2f}% | {def_trial['strict_exact']*100:>16.2f}% | {str(def_trial['params'])[:45]}")
        for i, t in enumerate(other_5, 1):
            p_str = f"filters={t['params'].get('conv_filters')}, hidden={t['params'].get('hidden_dim')}, layers={t['params'].get('num_layers')}, lr={t['params'].get('lr')}"
            print(f"{f'랜덤 후보 #{i}':<20} | {t['strict_f1']*100:>14.2f}% | {t['strict_exact']*100:>16.2f}% | {p_str:<45}")
        p_best = f"filters={best_trial['params'].get('conv_filters')}, hidden={best_trial['params'].get('hidden_dim')}, layers={best_trial['params'].get('num_layers')}, lr={best_trial['params'].get('lr')}"
        print("-" * 115)
        imp = (best_trial['strict_f1'] - def_trial['strict_f1']) * 100
        print(f"{'★ 최적 모델 (Best)':<20} | {best_trial['strict_f1']*100:>14.2f}% | {best_trial['strict_exact']*100:>16.2f}% | {p_best:<45} (성능 향상: +{imp:.2f}%p)")
        print("=" * 115)
