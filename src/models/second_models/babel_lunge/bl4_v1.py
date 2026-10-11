# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v1.0 Production Official
Module: bl4_v1.py
Purpose: 바벨 런지 4번 오류(척추의 중립 위반) 전용 초정밀 판정 및 3-State 추론 인터페이스
         - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 4) 또는 (64, 33, 3) 시퀀스 직접 추론
         - 28D 생체역학 도메인 피처 + 14D 시계열 TCN 하이브리드 Tri-Ensemble 융합
         - 전체 데이터셋 기반 3-State Selective Classification (Clean / Defect / Retake) 지원
         - 다각도 척추-경추 상세 기하학 진단 피드백 산출
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

from src.features.lunge_c4_features import (
    extract_lunge_c4_features,
    extract_lunge_c4_temporal_series,
    LUNGE_C4_FEATURE_NAMES
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "bl4_v1_weights.pkl")


# 14D 시계열 전용 Dilated TCN 아키텍처
class LungeC4DilatedTCN(nn.Module):
    def __init__(self, in_channels: int = 14, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        pooled = self.pool(h).squeeze(-1)
        out = F.relu(self.fc1(self.dropout(pooled)))
        logits = self.fc2(out)
        return logits


class BL4V1Classifier:
    """바벨 런지 c4 (척추의 중립 위반) 공식 배포 추론기 (bl4_v1)"""

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"bl4_v1 번들 파일을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'bl4_v1')
        self.version = bundle.get('version', 'v1.0')
        self.exercise = bundle.get('exercise', '바벨 런지 (04_바벨_런지)')
        self.defect = bundle.get('defect', 'c4 (척추의 중립 위반)')
        self.feature_names = bundle.get('feature_names', LUNGE_C4_FEATURE_NAMES)
        self.weights = bundle.get('weights', {'w_tcn': 0.25, 'w_lgb': 0.25, 'w_xgb': 0.25, 'w_cat': 0.25})
        self.thresholds = bundle.get('thresholds', {'tau_clean': 0.30, 'tau_def': 0.65, 'tau_binary': 0.50})
        self.overall_3state = bundle.get('metrics_3state_overall', {})
        self.overall_2state = bundle.get('metrics_2state_overall', {})
        self.view_metrics = bundle.get('view_metrics', {})

        self.fold_models = bundle.get('fold_models', {})
        self.num_folds = len(self.fold_models.get('scalers', []))

        # TCN 모델 인스턴스 복원 (CPU 모드)
        self.tcn_models = []
        for state in self.fold_models.get('tcn', []):
            m = LungeC4DilatedTCN(in_channels=14, hidden_dim=48, dropout=0.0)
            m.load_state_dict(state)
            m.eval()
            self.tcn_models.append(m)

    def extract_features(self, seq_33: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 텐서로부터 28D 도메인 피처 및 14D 시계열 추출"""
        feats_28d = extract_lunge_c4_features(seq_33)
        temp_14d = extract_lunge_c4_temporal_series(seq_33)
        return feats_28d, temp_14d

    def predict_proba(self, seq_33: np.ndarray) -> float:
        """단일 64프레임 시퀀스의 결함 확률(0.0~1.0) 산출 (5-Fold 앙상블)"""
        feats_28d, temp_14d = self.extract_features(seq_33)

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
            feats_scaled = scaler.transform(feats_28d.reshape(1, -1))

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
        """2-State 이진 판정 (0: 정상, 1: 결함)"""
        p = self.predict_proba(seq_33)
        return 1 if p >= self.thresholds['tau_binary'] else 0

    def predict_3state(self, seq_33: np.ndarray) -> Dict[str, Any]:
        """
        ★ 전체 데이터셋 기준 3-State Conformal Selective Classification
        Returns:
            - state: 'Clean' (정상 확정), 'Defect' (결함 확정), 'Retake' (신뢰 보류/재촬영 권고)
            - state_code: 0 (정상), 1 (결함), 2 (보류)
            - defect_prob: 결함 확률 (float)
            - certainty: 확신도 (|prob - 0.5| * 2)
            - feedback: 한국어 코칭 피드백
        """
        feats_28d, _ = self.extract_features(seq_33)
        p = self.predict_proba(seq_33)

        tau_clean = self.thresholds['tau_clean']
        tau_def = self.thresholds['tau_def']

        if p <= tau_clean:
            state = "Clean"
            state_code = 0
            feedback = "척추와 경추의 중립성이 완벽하게 유지되었습니다."
        elif p >= tau_def:
            state = "Defect"
            state_code = 1
            # 결함 상세 분석
            yaw_max = feats_28d[0]
            pitch_max = feats_28d[2]
            pitch_min = feats_28d[3]
            roll_max = feats_28d[8]

            reasons = []
            if roll_max >= 8.0:
                reasons.append(f"몸통 좌우 기울임(체간 롤 {roll_max:.1f}°)")
            if yaw_max >= 15.0:
                reasons.append(f"시선 좌/우 회전(경추 편차 {yaw_max:.1f}°)")
            if pitch_max >= 18.0:
                reasons.append(f"고개 상방 젖힘(하늘보기 {pitch_max:.1f}°)")
            if pitch_min <= -15.0:
                reasons.append(f"고개 하방 숙임(바닥보기 {pitch_min:.1f}°)")

            if not reasons:
                reasons.append("척추 및 경추 밸런스 불안정")

            feedback = f"척추 중립 위반 감지: {', '.join(reasons)}. 체간을 수직으로 고정하고 정면을 응시하세요."
        else:
            state = "Retake"
            state_code = 2
            feedback = "판정 신뢰도 경계 구간입니다. 정확한 진단을 위해 스마트폰을 정면(시점 C) 또는 전방 45도(시점 B, D)에 거치하고 재촬영해주세요."

        certainty = float(abs(p - 0.5) * 2.0)

        return {
            'state': state,
            'state_code': state_code,
            'defect_prob': p,
            'certainty': certainty,
            'feedback': feedback,
            'features': {
                'cervical_yaw_max': float(feats_28d[0]),
                'cervical_pitch_max': float(feats_28d[2]),
                'cervical_pitch_min': float(feats_28d[3]),
                'trunk_roll_max': float(feats_28d[8]),
                'shoulder_roll_max': float(feats_28d[10]),
                'pelvic_obliquity_roll': float(feats_28d[11])
            }
        }


# 모듈 테스트용
if __name__ == '__main__':
    print("=" * 60)
    print("  bl4_v1.py 모듈 인터페이스 검증")
    print("=" * 60)
    print(f"[*] 기본 번들 경로: {DEFAULT_BUNDLE_PATH}")
