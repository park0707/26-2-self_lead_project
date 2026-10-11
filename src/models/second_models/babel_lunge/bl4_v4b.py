# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v4b.0 Noise-Robust Release (v3 24D 베이스라인 계승 + Confident-Learning 소프트 가중 + SCE)
Module: bl4_v4b.py
Purpose: 피처 전수 감사(다중공선성/노이즈 제거) 기반 24차원 정예 피처 및 모델 구조 최적화 공식 배포 모듈
         - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 4) 또는 (64, 33, 3) 시퀀스 직접 추론
         - 24D 정예 피처 + Multi-Scale SE-TCN + GBDT 4-Way 앙상블 블렌딩
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

from src.features.lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    extract_lunge_c4_temporal_features,
    LUNGE_C4_FEATURE_NAMES_V3
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "bl4_v4b_weights.pkl")


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _ = x.shape
        w = torch.mean(x, dim=2)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1)
        return x * w


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        branch_channels = out_channels // 3
        rem = out_channels - branch_channels * 2

        self.conv_k3 = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_k5 = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.conv_k7 = nn.Conv1d(in_channels, rem, kernel_size=7, padding=3 * dilation, dilation=dilation)

        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=4)
        self.dropout = nn.Dropout(0.20)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        o3 = self.conv_k3(x)
        o5 = self.conv_k5(x)
        o7 = self.conv_k7(x)
        out = torch.cat([o3, o5, o7], dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.se(out)
        out = self.dropout(out)
        return F.gelu(out + res)


class MultiScaleSETCN(nn.Module):
    def __init__(self, in_features: int = 14, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU()
        )
        self.b1 = MultiScaleConvBlock(48, 48, dilation=1)
        self.b2 = MultiScaleConvBlock(48, 48, dilation=2)
        self.b3 = MultiScaleConvBlock(48, 48, dilation=4)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        feat = self.pool(h).squeeze(-1)
        logits = self.classifier(feat)
        return logits


class BL4V4bClassifier:
    """바벨 런지 c4 척추의 중립 판정 공식 배포 추론기 (v4b)"""

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        self.bundle_path = bundle_path
        self.is_loaded = False
        self.feature_names = LUNGE_C4_FEATURE_NAMES_V3
        self.weights = {'w_tcn': 0.20, 'w_lgb': 0.20, 'w_xgb': 0.20, 'w_cat': 0.40}
        self.thresholds = {'tau_clean': 0.40, 'tau_def': 0.53, 'tau_binary': 0.50}
        self.fold_models = None
        self.num_folds = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(bundle_path):
            self.load_bundle(bundle_path)

    def load_bundle(self, bundle_path: str):
        main_mod = sys.modules['__main__']
        if not hasattr(main_mod, 'MultiScaleSETCN'):
            setattr(main_mod, 'MultiScaleSETCN', MultiScaleSETCN)
        if not hasattr(main_mod, 'MultiScaleConvBlock'):
            setattr(main_mod, 'MultiScaleConvBlock', MultiScaleConvBlock)
        if not hasattr(main_mod, 'SEBlock1D'):
            setattr(main_mod, 'SEBlock1D', SEBlock1D)

        with open(bundle_path, 'rb') as f:
            data = pickle.load(f)

        self.feature_names = data.get('feature_names', self.feature_names)
        self.weights = data.get('weights', self.weights)
        self.thresholds = data.get('thresholds', self.thresholds)
        self.fold_models = data.get('fold_models', {})
        self.num_folds = len(self.fold_models.get('scaler', []))
        self.is_loaded = True

    def extract_features(self, seq_33: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """64프레임 시퀀스로부터 24차원 정예 피처 및 14차원 시계열 텐서 추출"""
        feats_24d = extract_lunge_c4_features_v3(seq_33)
        temp_14d = extract_lunge_c4_temporal_features(seq_33)
        return feats_24d, temp_14d

    def predict_proba(self, seq_33: np.ndarray) -> float:
        if not self.is_loaded:
            raise RuntimeError(f"가중치 번들이 로드되지 않았습니다: {self.bundle_path}")

        feats_24d, temp_14d = self.extract_features(seq_33)

        # 1. MultiScale-SE-TCN 5-Fold 앙상블
        tcn_models = self.fold_models.get('tcn', [])
        prob_tcn_list = []
        inp_tensor = torch.from_numpy(temp_14d).unsqueeze(0).float().to(self.device)

        for model in tcn_models:
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                out = model(inp_tensor)
                p = F.softmax(out, dim=-1)[0, 1].item()
                prob_tcn_list.append(p)
        p_tcn = float(np.mean(prob_tcn_list)) if prob_tcn_list else 0.5

        # 2. GBDT 3총사 5-Fold 앙상블
        scalers = self.fold_models.get('scaler', [])
        lgb_models = self.fold_models.get('lgb', [])
        xgb_models = self.fold_models.get('xgb', [])
        cat_models = self.fold_models.get('cat', [])

        prob_lgb_list = []
        prob_xgb_list = []
        prob_cat_list = []

        for fold in range(self.num_folds):
            scaler = scalers[fold]
            feats_scaled = scaler.transform(feats_24d.reshape(1, -1))

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
        feats_24d, _ = self.extract_features(seq_33)
        p = self.predict_proba(seq_33)

        tau_clean = self.thresholds['tau_clean']
        tau_def = self.thresholds['tau_def']

        if p <= tau_clean:
            state = "Clean"
            state_code = 0
            feedback = "척추 및 경추의 중립성이 정상 기준과 완벽히 일치합니다."
        elif p >= tau_def:
            state = "Defect"
            state_code = 1
            yaw_max = feats_24d[0]
            pitch_max = feats_24d[2]
            pitch_min = feats_24d[3]
            roll_max = feats_24d[8]
            loc_torsion = feats_24d[22]

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
                'cervical_yaw_max': float(feats_24d[0]),
                'cervical_pitch_max': float(feats_24d[2]),
                'cervical_pitch_min': float(feats_24d[3]),
                'trunk_roll_max': float(feats_24d[8]),
                'local_trunk_roll_inv': float(feats_24d[20]),
                'local_torso_torsion_max': float(feats_24d[22])
            }
        }


if __name__ == '__main__':
    print("=" * 60)
    print("  bl4_v4b.py 공식 추론 모듈 인터페이스 검증")
    print("=" * 60)
    print(f"[*] 기본 번들 경로: {DEFAULT_BUNDLE_PATH}")
