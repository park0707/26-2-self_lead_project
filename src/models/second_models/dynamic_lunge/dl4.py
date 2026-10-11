# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 통합 (05_스텝_포워드_다이나믹_런지 + 06_스텝_백워드_다이나믹_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v1.0 Production Release (바벨 런지 bl4 정예 아키텍처 계승)
Module: dl4.py
Purpose: 다이나믹 런지 c4 결함 판정 모델
         - 24차원 정예 피처 및 14D 시계열 MultiScale-SE-TCN + GBDT 4-Way 앙상블 공식 배포 추론기
         - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 시퀀스 직접 추론
         - 전체 데이터셋 대상 3-State Selective Classification (Clean / Defect / Retake) 지원
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

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl4_weights.pkl")


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


class DL4Classifier:
    """다이나믹 런지 c4 척추의 중립 판정 공식 배포 추론기"""

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        self.bundle_path = bundle_path
        self.is_loaded = False
        self.feature_names = []
        self.weights = {'w_tcn': 0.25, 'w_lgb': 0.25, 'w_xgb': 0.25, 'w_cat': 0.25}
        self.thresholds = {'tau_clean': 0.40, 'tau_def': 0.50, 'tau_binary': 0.50}
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
            sc = scalers[fold]
            x_sc = sc.transform(feats_24d.reshape(1, -1))

            if fold < len(lgb_models):
                p_l = lgb_models[fold].predict_proba(x_sc)[0, 1]
                prob_lgb_list.append(p_l)

            if fold < len(xgb_models):
                p_x = xgb_models[fold].predict_proba(x_sc)[0, 1]
                prob_xgb_list.append(p_x)

            if fold < len(cat_models):
                p_c = cat_models[fold].predict_proba(x_sc)[0, 1]
                prob_cat_list.append(p_c)

        p_lgb = float(np.mean(prob_lgb_list)) if prob_lgb_list else 0.5
        p_xgb = float(np.mean(prob_xgb_list)) if prob_xgb_list else 0.5
        p_cat = float(np.mean(prob_cat_list)) if prob_cat_list else 0.5

        # 3. 최적 가중 앙상블
        w_t = self.weights.get('w_tcn', 0.25)
        w_l = self.weights.get('w_lgb', 0.25)
        w_x = self.weights.get('w_xgb', 0.25)
        w_c = self.weights.get('w_cat', 0.25)

        raw_prob = w_t * p_tcn + w_l * p_lgb + w_x * p_xgb + w_c * p_cat
        return float(np.clip(raw_prob, 0.0, 1.0))

    def predict_3state(self, seq_33: np.ndarray) -> Dict[str, Any]:
        prob = self.predict_proba(seq_33)
        tc = self.thresholds.get('tau_clean', 0.40)
        td = self.thresholds.get('tau_def', 0.50)

        if prob <= tc:
            state = "Clean (정상)"
            label = 0
            is_determined = True
        elif prob >= td:
            state = "Defect (결함)"
            label = 1
            is_determined = True
        else:
            state = "Retake (판정보류)"
            label = -1
            is_determined = False

        return {
            'defect_prob': prob,
            'state': state,
            'label': label,
            'is_determined': is_determined,
            'thresholds': {'tau_clean': tc, 'tau_def': td}
        }
