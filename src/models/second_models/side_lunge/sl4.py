# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v1.0 Production Release (바벨 런지 bl4 정예 아키텍처 계승)
Module: sl4.py
Purpose: 사이드 런지 c4 결함 판정 모델
         - 24차원 정예 피처 및 14D 시계열 MultiScale-SE-TCN + GBDT 4-Way 앙상블 공식 배포 추론기
         - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 시퀀스 직접 추론
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

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "sl4_weights.pkl")


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
            nn.Dropout(0.20),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.b1(feat)
        feat = self.b2(feat)
        feat = self.b3(feat)
        pooled = self.pool(feat).squeeze(-1)
        return self.classifier(pooled)


class SL4Classifier:
    """
    사이드 런지 c4 (척추의 중립 위반) 공식 4-Way 앙상블 추론 엔진:
    - MultiScale-SE-TCN (14ch Temporal)
    - LightGBM (24D Feature)
    - XGBoost (24D Feature)
    - CatBoost (24D Feature)
    - 3-State Selective Classification (State 0: Clean, State 1: Defect, State 2: Retake)
    """
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.models_bundle = None
        self.is_loaded = False
        self._load_bundle()

    def _load_bundle(self):
        if not os.path.exists(self.bundle_path):
            return
        with open(self.bundle_path, 'rb') as f:
            self.models_bundle = pickle.load(f)
        self.is_loaded = True

        self.tau_low = self.models_bundle.get('tau_low', 0.40)
        self.tau_high = self.models_bundle.get('tau_high', 0.55)
        self.weights = self.models_bundle.get('ensemble_weights', {
            'tcn': 0.35, 'lgb': 0.25, 'xgb': 0.20, 'cat': 0.20
        })

    def predict_sequence(self, seq_33: np.ndarray) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스 추론:
        Input: (64, 33, 3) or (64, 33, 4) MediaPipe 관절 텐서
        Output: 종합 판정 딕셔너리
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model bundle not loaded from {self.bundle_path}")

        coords = seq_33[:, :, :3].astype(np.float32)
        f24 = extract_lunge_c4_features_v3(coords)
        t14 = extract_lunge_c4_temporal_features(coords)

        f24_norm = (f24 - self.models_bundle['mean']) / (self.models_bundle['std'] + 1e-6)
        f24_2d = f24_norm.reshape(1, -1)

        # 1. TCN 추론
        tcn_model = self.models_bundle['tcn_model']
        tcn_model.eval()
        with torch.no_grad():
            t14_tensor = torch.tensor(t14.T, dtype=torch.float32).unsqueeze(0)
            logits = tcn_model(t14_tensor)
            p_tcn = float(F.softmax(logits, dim=-1)[0, 1].item())

        # 2. GBDT 추론
        p_lgb = float(self.models_bundle['lgb_model'].predict_proba(f24_2d)[0, 1])
        p_xgb = float(self.models_bundle['xgb_model'].predict_proba(f24_2d)[0, 1])
        p_cat = float(self.models_bundle['cat_model'].predict_proba(f24_2d)[0, 1])

        # 3. 앙상블 소프트 보팅
        p_ensemble = (
            self.weights['tcn'] * p_tcn +
            self.weights['lgb'] * p_lgb +
            self.weights['xgb'] * p_xgb +
            self.weights['cat'] * p_cat
        )

        # 4. 3-State 판정
        if p_ensemble >= self.tau_high:
            state = 1
            state_label = "DEFECT"
            status_desc = "척추의 중립 위반 (체간 측굴 또는 과도한 경추 회전/신전 결함)"
        elif p_ensemble < self.tau_low:
            state = 0
            state_label = "CLEAN"
            status_desc = "정상 (척추 및 체간 중립 양호)"
        else:
            state = 2
            state_label = "RETAKE"
            status_desc = "판정보류 (모호 구간, 1회 재촬영 권고)"

        # 5. 세부 피처 기반 피드백 생성
        feedback = self._generate_feedback(f24)

        return {
            'state': state,
            'state_label': state_label,
            'status_description': status_desc,
            'ensemble_prob': float(p_ensemble),
            'stream_probs': {
                'tcn': p_tcn,
                'lgb': p_lgb,
                'xgb': p_xgb,
                'cat': p_cat
            },
            'thresholds': {
                'tau_low': self.tau_low,
                'tau_high': self.tau_high
            },
            'feedback': feedback
        }

    def _generate_feedback(self, f24: np.ndarray) -> List[str]:
        feedback = []
        c_yaw = f24[0]
        c_pitch = f24[2]
        t_roll = f24[8]
        s_roll = f24[9]

        if t_roll >= 8.0 or s_roll >= 7.0:
            feedback.append(f"체간이 측면으로 {t_roll:.1f}° 기울어졌습니다. 상체를 똑바로 세우고 전두면 중립을 유지하세요.")
        if c_yaw >= 18.0:
            feedback.append(f"고개가 측면으로 {c_yaw:.1f}° 과도하게 돌아갔습니다. 시선을 정면에 고정하세요.")
        if c_pitch >= 15.0:
            feedback.append(f"턱이 들려 목이 뒤로 {c_pitch:.1f}° 젖혀졌습니다. 턱을 살짝 당겨 경추 중립을 유지하세요.")

        if not feedback:
            feedback.append("척추 및 경추의 중립 상태가 훌륭합니다.")
        return feedback
