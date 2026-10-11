# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v2.0 Advanced Research Edition
Module: bl4_v2.py
Purpose: 선행 연구 4대 핵심 기술(골드 템플릿 DTW, 체간 로컬 불변 투영, Multi-Scale SE-TCN, Focal Gating) 탑재
         - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 4) 또는 (64, 33, 3) 시퀀스 직접 추론
         - 36D 생체역학 정예 피처 + Multi-Scale SE-TCN + GBDT 하이브리드 Tri-Ensemble 융합
         - 전체 데이터셋 대상 3-State Selective Classification (Clean / Defect / Retake) 지원
         - 척추-경추 다각도 상세 진단 및 교정 피드백 산출
"""

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "D:/"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c4_features_v2 import (
    extract_lunge_c4_features_v2,
    LUNGE_C4_FEATURE_NAMES_V2
)
from src.features.lunge_c4_features import extract_lunge_c4_temporal_series

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "bl4_v2_weights.pkl")


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _ = x.shape
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)
        return x * w


class MultiScaleSETCN(nn.Module):
    def __init__(self, in_channels: int = 14, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        self.conv_k3 = nn.Conv1d(in_channels, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_channels, hidden_dim // 3, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(in_channels, hidden_dim - 2 * (hidden_dim // 3), kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.dilated1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dilated2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)

        self.se = SEBlock1D(hidden_dim * 2, reduction=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h3 = self.conv_k3(x)
        h5 = self.conv_k5(x)
        h7 = self.conv_k7(x)
        h = torch.cat([h3, h5, h7], dim=1)
        h = F.relu(self.bn1(h))

        h = F.relu(self.bn2(self.dilated1(h)))
        h = F.relu(self.bn3(self.dilated2(h)))
        h = self.se(h)

        pooled = self.pool(h).squeeze(-1)
        out = F.relu(self.fc1(self.dropout(pooled)))
        logits = self.fc2(out)
        return logits


class BL4V2Classifier:
    """바벨 런지 c4 (척추의 중립 위반) 선행연구 고도화 추론기 (bl4_v2)"""

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"bl4_v2 번들 파일을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'bl4_v2')
        self.version = bundle.get('version', 'v2.0 Advanced Research Edition')
        self.exercise = bundle.get('exercise', '바벨 런지 (04_바벨_런지)')
        self.defect = bundle.get('defect', 'c4 (척추의 중립 위반)')
        self.feature_names = bundle.get('feature_names', LUNGE_C4_FEATURE_NAMES_V2)
        self.weights = bundle.get('weights', {'w_tcn': 0.25, 'w_lgb': 0.25, 'w_xgb': 0.25, 'w_cat': 0.25})
        self.thresholds = bundle.get('thresholds', {'tau_clean': 0.35, 'tau_def': 0.60, 'tau_binary': 0.50})
        self.overall_3state = bundle.get('metrics_3state_overall', {})
        self.overall_2state = bundle.get('metrics_2state_overall', {})
        self.view_metrics = bundle.get('view_metrics', {})

        self.fold_models = bundle.get('fold_models', {})
        self.num_folds = len(self.fold_models.get('scalers', []))

        # MultiScale SE-TCN 모델 복원
        self.tcn_models = []
        for state in self.fold_models.get('tcn', []):
            m = MultiScaleSETCN(in_channels=14, hidden_dim=48, dropout=0.0)
            m.load_state_dict(state)
            m.eval()
            self.tcn_models.append(m)

    def extract_features(self, seq_33: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스로부터 36D 도메인 피처 및 14D 시계열 추출"""
        feats_36d = extract_lunge_c4_features_v2(seq_33)
        temp_14d = extract_lunge_c4_temporal_series(seq_33)
        return feats_36d, temp_14d

    def predict_proba(self, seq_33: np.ndarray) -> float:
        """단일 64프레임 시퀀스의 결함 확률(0.0~1.0) 산출 (5-Fold 앙상블)"""
        feats_36d, temp_14d = self.extract_features(seq_33)

        tcn_x = torch.tensor(temp_14d.T, dtype=torch.float32).unsqueeze(0)  # (1, 14, 64)
        prob_tcn_list = []
        with torch.no_grad():
            for m in self.tcn_models:
                p = torch.sigmoid(m(tcn_x)).item()
                prob_tcn_list.append(p)
        p_tcn = float(np.mean(prob_tcn_list))

        prob_lgb_list = []
        prob_xgb_list = []
        prob_cat_list = []

        scalers = self.fold_models.get('scalers', [])
        lgb_models = self.fold_models.get('lgb', [])
        xgb_models = self.fold_models.get('xgb', [])
        cat_models = self.fold_models.get('cat', [])

        for fold in range(self.num_folds):
            scaler = scalers[fold]
            feats_scaled = scaler.transform(feats_36d.reshape(1, -1))

            if fold < len(lgb_models):
                p_l = lgb_models[fold].predict_proba(feats_scaled)[0, 1]
                prob_lgb_list.append(p_l)

            if fold < len(xgb_models):
                p_x = xgb_models[fold].predict_proba(feats_scaled)[0, 1]
                prob_xgb_list.append(p_x)

            if fold < len(cat_models):
                p_c = cat_models[fold].predict_proba(feats_scaled)[0, 1]
                prob_cat_list.append(p_c)

        p_lgb = float(np.mean(prob_lgb_list)) if prob_lgb_list else 0.5
        p_xgb = float(np.mean(prob_xgb_list)) if prob_xgb_list else 0.5
        p_cat = float(np.mean(prob_cat_list)) if prob_cat_list else 0.5

        w = self.weights
        p_ens = w['w_tcn'] * p_tcn + w['w_lgb'] * p_lgb + w['w_xgb'] * p_xgb + w['w_cat'] * p_cat
        return float(np.clip(p_ens, 0.0, 1.0))

    def predict(self, seq_33: np.ndarray) -> int:
        p = self.predict_proba(seq_33)
        return 1 if p >= self.thresholds['tau_binary'] else 0

    def predict_3state(self, seq_33: np.ndarray) -> Dict[str, Any]:
        feats_36d, _ = self.extract_features(seq_33)
        p = self.predict_proba(seq_33)

        tau_clean = self.thresholds['tau_clean']
        tau_def = self.thresholds['tau_def']

        if p <= tau_clean:
            state = "Clean"
            state_code = 0
            feedback = "척추 및 경추의 중립성이 정상 골드 템플릿과 완벽히 일치합니다."
        elif p >= tau_def:
            state = "Defect"
            state_code = 1
            yaw_max = feats_36d[0]
            pitch_max = feats_36d[2]
            pitch_min = feats_36d[3]
            roll_max = feats_36d[8]
            loc_torsion = feats_36d[34]

            reasons = []
            if roll_max >= 7.5:
                reasons.append(f"체간 좌우 기울임(측굴 {roll_max:.1f}°)")
            if yaw_max >= 14.0:
                reasons.append(f"시선 좌/우 회전(경추 편차 {yaw_max:.1f}°)")
            if pitch_max >= 18.0:
                reasons.append(f"고개 과신전(하늘보기 {pitch_max:.1f}°)")
            if pitch_min <= -14.0:
                reasons.append(f"고개 과굴곡(바닥보기 {pitch_min:.1f}°)")
            if loc_torsion >= 12.0:
                reasons.append(f"상체 비틀림(흉곽 비틀림 {loc_torsion:.1f}°)")

            if not reasons:
                reasons.append("척추 및 경추 자세 정렬 불량")

            feedback = f"척추 중립 위반 감지: {', '.join(reasons)}. 척추를 수직으로 세우고 정면을 응시하세요."
        else:
            state = "Retake"
            state_code = 2
            feedback = "판정 신뢰도 경계 구간입니다. 정확한 분석을 위해 정면(시점 C) 또는 우측 전방(시점 B)에 거치 후 재촬영을 권장합니다."

        certainty = float(abs(p - 0.5) * 2.0)

        return {
            'state': state,
            'state_code': state_code,
            'defect_prob': p,
            'certainty': certainty,
            'feedback': feedback,
            'features': {
                'cervical_yaw_max': float(feats_36d[0]),
                'cervical_pitch_max': float(feats_36d[2]),
                'cervical_pitch_min': float(feats_36d[3]),
                'trunk_roll_max': float(feats_36d[8]),
                'golden_spine_dtw': float(feats_36d[28]),
                'local_trunk_roll': float(feats_36d[32])
            }
        }


if __name__ == '__main__':
    print("=" * 60)
    print("  bl4_v2.py 공식 추론 모듈 인터페이스 검증")
    print("=" * 60)
    print(f"[*] 기본 번들 경로: {DEFAULT_BUNDLE_PATH}")
