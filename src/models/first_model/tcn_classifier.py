# =========================================================================================================
# [모듈 설명]: 1단계 9종 운동 대분류 - Temporal Convolutional Network (TCN) 시계열 딥러닝 분류기
# [입력 데이터]: 64 프레임 x 69차원 시계열 텐서 (Format B: 전신 23개 관절 3차원 좌표 궤적)
# [출력 목표]: 9종 운동 대분류 (01_바벨_스쿼트 ~ 09_랫_풀_다운) Multi-class 예측
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~15초):
#      python -m src.models.first_model.tcn_classifier
#      또는 python -m src.models.first_model.tcn_classifier --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~90초):
#      python -m src.models.first_model.tcn_classifier --mode compare
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
    'base_channels': 64,
    'dropout': 0.1,
    'epochs': 10,
    'kernel_size': 7,
    'lr': 0.001
}

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop1,
                                 self.conv2, self.bn2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # padding trim if lengths differ due to dilation
        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int = 69, num_classes: int = 9, num_channels: list = [64, 64, 128, 128], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 69) -> (B, 69, 64)
        x_in = x.transpose(1, 2)
        feat = self.network(x_in)
        pooled = self.pool(feat).squeeze(-1)
        return self.fc(pooled)

class TemporalTCNActionClassifier:
    """1단계: Temporal Convolutional Network (TCN) 9종 운동 분류기 Wrapper"""
    def __init__(self, num_classes: int = 9, kernel_size: int = 3, base_channels: int = 64, dropout: float = 0.2, lr: float = 0.001, epochs: int = 2, batch_size: int = 256, device: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.dropout = dropout

        num_channels = [base_channels, base_channels, base_channels * 2, base_channels * 2]
        self.model = TemporalConvNet(input_dim=69, num_classes=num_classes, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(self.device)
        self.hyperparams = {
            'kernel_size': kernel_size,
            'base_channels': base_channels,
            'num_channels': num_channels,
            'dropout': dropout,
            'epochs': epochs,
            'lr': lr,
            'device': str(self.device)
        }
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TemporalTCNActionClassifier":
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
        metrics['model_name'] = 'Temporal_TCN'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y_test)
        metrics['predictions'] = preds
        return metrics

def run_tcn_grid_search(data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], TemporalTCNActionClassifier]:
    """TCN 하이퍼파라미터 탐색 및 최고 vs 후보 5종 비교"""
    X_train, y_train = data['train']['X_Format_B'], data['train']['y_action']
    X_valid, y_valid = data['valid']['X_Format_B'], data['valid']['y_action']
    X_strict, y_strict = data['test_strict']['X_Format_B'], data['test_strict']['y_action']
    X_subj, y_subj = data['test_unseen_subj']['X_Format_B'], data['test_unseen_subj']['y_action']

    combos = [
        {'kernel_size': 3, 'base_channels': 64,  'dropout': 0.2, 'lr': 0.001, 'epochs': 2},
        {'kernel_size': 5, 'base_channels': 64,  'dropout': 0.2, 'lr': 0.001, 'epochs': 2},
        {'kernel_size': 3, 'base_channels': 128, 'dropout': 0.3, 'lr': 0.001, 'epochs': 2},
        {'kernel_size': 5, 'base_channels': 128, 'dropout': 0.2, 'lr': 0.0005, 'epochs': 2},
        {'kernel_size': 7, 'base_channels': 64,  'dropout': 0.1, 'lr': 0.001, 'epochs': 2},
        {'kernel_size': 3, 'base_channels': 64,  'dropout': 0.1, 'lr': 0.0005, 'epochs': 2}
    ]

    print(f"[*] Temporal TCN 하이퍼파라미터 탐색 시작 (총 {len(combos)}개 대표 구조 검증)...")

    results = []
    best_valid_f1 = -1.0
    best_model = None
    best_combo = None

    for idx, combo in enumerate(combos, 1):
        print(f"  - [{idx}/{len(combos)}] TCN 아키텍처 학습 중: k={combo['kernel_size']}, ch={combo['base_channels']}, lr={combo['lr']}...")
        clf = TemporalTCNActionClassifier(
            kernel_size=combo['kernel_size'],
            base_channels=combo['base_channels'],
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
            best_combo = combo
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

    parser = argparse.ArgumentParser(description="Stage 1: Temporal Convolutional Network (TCN) 9-Class Action Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    args = parser.parse_args()

    data = Stage1DatasetLoader.load_all_dual_splits()

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print("    [1단계: TCN 9종 운동 대분류 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        clf = TemporalTCNActionClassifier(**BEST_PARAMS)
        clf.fit(data['train']['X_Format_B'], data['train']['y_action'])
        best_clf = clf
        best_params = BEST_PARAMS
    else:
        print("\n" + "=" * 105)
        print("    [1단계: TCN 9종 운동 대분류 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        best_trial, other_5, best_clf = run_tcn_grid_search(data)
        best_params = best_trial['params']

    Stage1MetricsHelper.print_model_header(
        model_name=f"Temporal Convolutional Network (TCN) ({'Best Tuned' if args.mode=='best' else 'Grid Search Compare'})",
        hyperparams=best_params,
        train_samples=data['train']['X_Format_B'].shape[0],
        feat_dim="Format B (64 프레임 x 69차원 시계열 텐서)"
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
            param_keys=['kernel_size', 'base_channels', 'dropout', 'lr']
        )

    Stage1MetricsHelper.save_best_params_json(
        model_name="Temporal_TCN",
        best_params=best_params,
        summary_metrics={
            'valid_macro_f1': split_results['Valid Set (검증: 피험자 7~8, 시점 D)']['macro_f1'],
            'strict_test_macro_f1': strict_res['macro_f1'] if strict_res else 0.0,
            'advanced_metrics': adv_metrics
        }
    )
