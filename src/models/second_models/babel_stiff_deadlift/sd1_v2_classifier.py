# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 1: sd1 (무릎 구부린 채 고정 위반 / Knee Fixed-Angle Defect)
Version: v2.0 (sd1_v2 - Curated 6D Feature Diet)
Module: sd1_v2_classifier.py
Purpose: 스티프 데드리프트 1번 오류(sd1_v2) 전용 독립 2계층 & 3-State 추론 및 진단 인터페이스
"""

import os
import sys
import pickle
import numpy as np

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stiff_deadlift_knee_features_v2 import (
    extract_knee_invariance_features_6d, extract_knee_temporal_8ch_v2
)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v2_bundle.pkl")

# ==============================================================================
# TCN 네트워크 정의 (8채널 시계열 입력)
# ==============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class KneeTCNClassifierV2(nn.Module):
    def __init__(self, in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.network(x)
        pooled = torch.mean(feat, dim=-1)
        return self.fc(pooled)

# ==============================================================================
# sd1_v2 통합 Expert 인터페이스
# ==============================================================================
class SD1V2Classifier:
    """바벨 스티프 데드리프트 1번 오류(sd1_v2: 무릎 고정 위반 - 6D Curated Diet) 전용 2계층 & 3-State Expert"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            alt_path = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "sd1_v2_bundle.pkl")
            if os.path.exists(alt_path):
                bundle_path = alt_path
            else:
                raise FileNotFoundError(f"sd1_v2 번들을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.fold_models_lgb = bundle['fold_models_lgb']
        self.feature_names = bundle['feature_names']
        self.tau_lower = bundle['thresholds']['tau_lower']
        self.tau_upper = bundle['thresholds']['tau_upper']
        self.w_lgb = bundle['stream_weights']['lgb']
        self.w_tcn = bundle['stream_weights']['tcn']

        # TCN 복원 (8ch)
        self.tcn_models = []
        for s_dict in bundle['tcn_state_dicts']:
            net = KneeTCNClassifierV2(in_channels=8)
            net.load_state_dict(s_dict)
            net.eval()
            self.tcn_models.append(net)

    def predict_proba(self, x_seq: np.ndarray) -> float:
        """
        단일 시퀀스(64, 98)에 대해 무릎 고정 실패 결함 확률 P(Defect) 반환
        """
        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, ...] # (1, 64, 98)

        # 1. 생체역학 6D 정예 피처
        feat_6 = extract_knee_invariance_features_6d(x_seq[0])[np.newaxis, :]
        p_lgb = float(np.mean([m.predict_proba(feat_6)[0, 1] for m in self.fold_models_lgb]))

        # 2. 8ch 시계열
        series_8 = extract_knee_temporal_8ch_v2(x_seq[0])[np.newaxis, :]
        with torch.no_grad():
            bx = torch.tensor(series_8, dtype=torch.float32)
            p_tcns = [F.softmax(m(bx), dim=-1)[0, 1].item() for m in self.tcn_models]
            p_tcn = float(np.mean(p_tcns))

        # 3. 앙상블 융합
        p_ens = float(self.w_lgb * p_lgb + self.w_tcn * p_tcn)
        return p_ens

    def diagnose(self, x_seq: np.ndarray, view_name: str = "view3") -> dict:
        """
        단일 시퀀스에 대한 3-State 최종 진단 결과 반환
        """
        p = self.predict_proba(x_seq)
        state_2s = 1 if p >= 0.50 else 0

        if p < self.tau_lower:
            state_3s = 0
            label = "정상 확정 (CLEAN)"
            feedback = "무릎 약 15도 각도를 견고하게 잠근 상태로 완벽하게 고관절 힌지를 수행했습니다."
        elif p >= self.tau_upper:
            state_3s = 1
            label = "결함 확정 (DEFECT)"
            feedback = "무릎 고정이 풀렸습니다. 무릎이 스쿼트처럼 과도하게 굽혀지거나 주저앉지 않도록 최저점에서도 각도를 일정하게 고정하세요."
        else:
            state_3s = 2
            label = "판단 불가 / 재촬영 (UNCERTAIN)"
            feedback = "카메라 각도 또는 복합 동작으로 무릎 관절 궤적이 모호합니다. 시점 B(우측 전방 45도) 또는 시점 D(좌측 전방 45도)에서 1회 다시 수행해 주세요."

        return {
            'defect_code': 'sd1_v2',
            'defect_name': '무릎 구부린 채 고정 위반 (Curated Diet)',
            'probability': p,
            'margin': abs(p - 0.50),
            'state_2state': state_2s,
            'state_3state': state_3s,
            'state_label': label,
            'feedback': feedback
        }

if __name__ == '__main__':
    clf = SD1V2Classifier()
    print("[+] sd1_v2 Classifier 로딩 및 초기화 성공!")
    print(f"  - 3-State Thresholds: tau_lower={clf.tau_lower:.4f}, tau_upper={clf.tau_upper:.4f}")
    print(f"  - Stream Weights: LightGBM={clf.w_lgb}, TCN={clf.w_tcn}")
