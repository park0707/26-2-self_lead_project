# =========================================================================================================
# [모듈 설명]: 1단계 9종 운동 대분류 - 1D-CNN + Bi-GRU + Self-Attention 시계열 딥러닝 최고 성능 분류기
# [입력 데이터]: 64 프레임 x 85차원 시계열 텐서 (Condition 3: MediaPipe 3D 월드 관절 좌표 궤적)
# [출력 목표]: 9종 운동 대분류 (01_바벨_스쿼트 ~ 09_랫_풀_다운) Multi-class 예측 (Strict F1: 84.57%)
#
# [실행 명령어 (CLI 테스트)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드):
#      python -m src.models.first_model.gru_classifier
#      또는 python -m src.models.first_model.gru_classifier --mode best
#
#   2. 최적 모델 vs 기본값 vs 랜덤 후보 5개 비교 분석:
#      python -m src.models.first_model.gru_classifier --mode compare
# =========================================================================================================

import os
import sys
import time
import argparse
import itertools
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper

BEST_PARAMS = {
    'conv_filters': 128,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.15,
    'lr': 0.0015,
    'epochs': 25,
    'batch_size': 128
}

class ConvBiGRUActionNet(nn.Module):
    """1단계: 1D-CNN + Bi-GRU 시계열 심층 신경망 (Format B (64, 91) Condition 3 확장 입력)"""
    def __init__(self, input_dim: int = 91, seq_len: int = 64, num_classes: int = len(EXERCISE_NAMES_KO), hidden_dim: int = 128, conv_filters: int = 128, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.gru = nn.GRU(
            input_size=conv_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        gru_out_dim = hidden_dim * 2
        self.attn_fc = nn.Linear(gru_out_dim, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.transpose(1, 2)
        
        gru_out, _ = self.gru(x)
        
        attn_weights = torch.softmax(self.attn_fc(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        logits = self.fc(context)
        return logits

class TemporalGRUActionClassifier:
    """1단계: Bi-GRU 기반 8종 운동 분류기 Wrapper (Condition 3 MediaPipe 3D 91D 기본 지원)"""
    def __init__(self, conv_filters: int = 128, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.15, lr: float = 0.0015, epochs: int = 25, batch_size: int = 128, num_classes: int = len(EXERCISE_NAMES_KO), device: Optional[str] = None):
        self.conv_filters = conv_filters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ConvBiGRUActionNet(
            input_dim=91,
            seq_len=64,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            conv_filters=conv_filters,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.hyperparams = {
            'conv_filters': conv_filters,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size
        }
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TemporalGRUActionClassifier":
        in_dim = X_train.shape[2] if X_train.ndim > 2 else 91
        n_classes = len(np.unique(y_train)) if len(np.unique(y_train)) > 0 else self.num_classes
        self.model = ConvBiGRUActionNet(
            input_dim=in_dim,
            seq_len=64,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            conv_filters=self.conv_filters,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x) / max(0.1, temperature)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        probs = self.predict_proba(X_test)
        preds = np.argmax(probs, axis=-1)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0
        metrics = Stage1MetricsHelper.evaluate_predictions(y_test, preds, inference_time_ms=inference_time_ms)
        metrics['model_name'] = '1D-CNN + Bi-GRU'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y_test)
        metrics['predictions'] = preds
        metrics['probabilities'] = probs
        return metrics

def run_gru_grid_search(data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], TemporalGRUActionClassifier]:
    """1D-CNN + Bi-GRU 하이퍼파라미터 탐색 및 최고 vs 후보 5종 비교"""
    X_train, y_train = data['train']['X_Format_B'], data['train']['y_action']
    X_valid, y_valid = data['valid']['X_Format_B'], data['valid']['y_action']
    X_strict, y_strict = data['test_strict']['X_Format_B'], data['test_strict']['y_action']
    X_subj, y_subj = data['test_unseen_subj']['X_Format_B'], data['test_unseen_subj']['y_action']

    combos = [
        {'conv_filters': 128, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'epochs': 3},
        {'conv_filters': 64,  'hidden_dim': 64,  'num_layers': 2, 'dropout': 0.1, 'lr': 0.001, 'epochs': 3},
        {'conv_filters': 128, 'hidden_dim': 64,  'num_layers': 1, 'dropout': 0.2, 'lr': 0.001, 'epochs': 3},
        {'conv_filters': 64,  'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.0005, 'epochs': 3},
        {'conv_filters': 128, 'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.001, 'epochs': 3},
        {'conv_filters': 64,  'hidden_dim': 64,  'num_layers': 1, 'dropout': 0.2, 'lr': 0.0005, 'epochs': 3}
    ]

    print(f"[*] 1D-CNN + Bi-GRU 하이퍼파라미터 탐색 시작 (총 {len(combos)}개 대표 구조 검증)...")

    results = []
    best_valid_f1 = -1.0
    best_model = None

    for idx, combo in enumerate(combos, 1):
        print(f"  - [{idx}/{len(combos)}] 아키텍처 학습 중: conv={combo['conv_filters']}, gru={combo['hidden_dim']}, layers={combo['num_layers']}, lr={combo['lr']}...")
        clf = TemporalGRUActionClassifier(
            conv_filters=combo['conv_filters'],
            hidden_dim=combo['hidden_dim'],
            num_layers=combo['num_layers'],
            dropout=combo['dropout'],
            lr=combo['lr'],
            epochs=combo['epochs']
        )
        clf.fit(X_train, y_train)
        
        res_v = clf.evaluate(X_valid, y_valid, split_name='valid')
        res_st = clf.evaluate(X_strict, y_strict, split_name='strict')
        res_sb = clf.evaluate(X_subj, y_subj, split_name='subj')

        trial_info = {
            'params': combo,
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'subj_f1': res_sb['macro_f1'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_valid_f1:
            best_valid_f1 = res_v['macro_f1']
            best_model = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    other_5 = results[1:6]

    return best_trial, other_5, best_model

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Stage 1: 1D-CNN + Bi-GRU 9-Class Action Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    args = parser.parse_args()

    data = Stage1DatasetLoader.load_all_dual_splits()

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print("    [1단계: 1D-CNN + Bi-GRU 9종 운동 대분류 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        clf = TemporalGRUActionClassifier(**BEST_PARAMS)
        clf.fit(data['train']['X_Format_B'], data['train']['y_action'])
        best_clf = clf
        best_params = BEST_PARAMS
    else:
        print("\n" + "=" * 105)
        print("    [1단계: 1D-CNN + Bi-GRU 9종 운동 대분류 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        best_trial, other_5, best_clf = run_gru_grid_search(data)
        best_params = best_trial['params']

    Stage1MetricsHelper.print_model_header(
        model_name=f"1D-CNN + Bi-GRU + Self-Attention ({'Best Tuned' if args.mode=='best' else 'Grid Search Compare'})",
        hyperparams=best_params,
        train_samples=data['train']['X_Format_B'].shape[0],
        feat_dim="Format B (64 프레임 x 91차원 시계열 텐서 - Condition 3 MediaPipe 3D 확장)"
    )

    split_results = {}
    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    strict_res = None
    for display_name, key in split_map:
        res = best_clf.evaluate(data[key]['X_Format_B'], data[key]['y_action'], split_name=key)
        split_results[display_name] = res
        if key == 'test_strict':
            strict_res = res

    res_train = best_clf.evaluate(data['train']['X_Format_B'], data['train']['y_action'], split_name='train')
    adv_metrics = Stage1MetricsHelper.calculate_advanced_metrics(
        res_train=res_train,
        res_valid=split_results['Valid Set (검증: 피험자 7~8, 시점 D)'],
        res_strict=strict_res,
        res_subj=split_results['Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)'],
        res_view=split_results['View Test (시점 일반화: 피험자 1~6, 측면 C)']
    )

    Stage1MetricsHelper.print_multi_split_summary(split_results, adv_metrics=adv_metrics)
    if strict_res is not None:
        Stage1MetricsHelper.print_class_breakdown_report(strict_res, data['test_strict']['y_action'])
        Stage1MetricsHelper.print_top_confusion_analysis(strict_res, top_k=3)

    if args.mode == 'compare':
        Stage1MetricsHelper.print_top_vs_random_comparison_table(
            best_trial=best_trial,
            other_trials=other_5,
            param_keys=['conv_filters', 'hidden_dim', 'num_layers', 'dropout', 'lr']
        )

    Stage1MetricsHelper.save_best_params_json(
        model_name="1D-CNN + Bi-GRU",
        best_params=best_params,
        summary_metrics={
            'valid_macro_f1': split_results['Valid Set (검증: 피험자 7~8, 시점 D)']['macro_f1'],
            'strict_test_macro_f1': strict_res['macro_f1'] if strict_res else 0.0,
            'advanced_metrics': adv_metrics
        }
    )
