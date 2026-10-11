# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.0_b (c1_model_v11_b.py)
Status: ★ [실험 코드 E1 탑재] 경량 Dilated TCN (Temporal Convolutional Network) + GBDT 융합 공식 모델
        - 1) 관절 64F 시계열 텐서(64, 24) 직접 입력
        - 2) 3-Block Dilated Residual TCN (Dilation 1, 2, 4, 파라미터 ~4만 개, 과적합 완벽 방지)
        - 3) TCN 시계열 확률과 GBDT(v11_a_diet 27D) 확률의 최적 Late Fusion 앙상블
        - 4) 3-State 선택적 거부(Selective Abstention) 아키텍처 100% 호환
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_a_diet import (
    extract_c1_v11_a_diet_features_from_seq,
    FEATURE_NAMES_27_C1_DIET,
    BabelSquatC1ModelV11ADiet
)
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d


class TemporalBlock(nn.Module):
    """Dilated Residual Convolutional Block"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.bn2(self.conv2(out)))
        return self.relu2(out + res)


class SquatC1DilatedTCN(nn.Module):
    """
    바벨 스쿼트 c1 전용 경량 Dilated TCN 신경망 (~42k 파라미터)
    입력: (B, 24, 64) -> 출력: (B, 1) Logits
    """
    def __init__(self, in_channels: int = 24, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block1 = TemporalBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout)

        # Attention Pooling
        self.att_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 64)
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        # Attention weights along time (B, 1, 64)
        att = F.softmax(self.att_conv(h), dim=-1)
        pooled = torch.sum(h * att, dim=-1) # (B, hidden_dim)

        logits = self.head(pooled) # (B, 1)
        return logits


class BabelSquatC1ModelV11B:
    """
    c1 v11_b (실험 코드 E1: 경량 TCN + GBDT Late Fusion 앙상블)
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60,
        tcn_weight: float = 0.35,
        gbdt_weight: float = 0.65
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.tcn_weight = tcn_weight
        self.gbdt_weight = gbdt_weight

        self.gbdt_model = BabelSquatC1ModelV11ADiet()
        self.tcn_models: List[SquatC1DilatedTCN] = []
        self.is_fitted = False

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(self, X_27d: np.ndarray, X_temporal: np.ndarray) -> np.ndarray:
        """
        GBDT(27D) 및 TCN(64, 24) 입력으로부터 융합 확률 반환
        """
        if not self.is_fitted:
            raise RuntimeError("C1 v11_b 모델 가중치가 로드되지 않았습니다.")

        # GBDT 확률 (N, 2)
        p_gbdt = self.gbdt_model.predict_proba(X_27d)[:, 1]

        # TCN 확률 (N,)
        # X_temporal: (N, 64, 24) -> PyTorch (N, 24, 64)
        x_t = torch.tensor(X_temporal.transpose(0, 2, 1), dtype=torch.float32)
        p_tcn = np.zeros(len(X_temporal), dtype=np.float32)

        with torch.no_grad():
            for tcn in self.tcn_models:
                tcn.eval()
                logits = tcn(x_t).squeeze(-1)
                p_tcn += torch.sigmoid(logits).cpu().numpy() / len(self.tcn_models)

        # Late Fusion
        p_fused = self.gbdt_weight * p_gbdt + self.tcn_weight * p_tcn
        p_fused = np.clip(p_fused, 0.0, 1.0)
        p0 = 1.0 - p_fused
        return np.column_stack([p0, p_fused])

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tcn_state_dicts = [tcn.state_dict() for tcn in self.tcn_models]
        payload = {
            'gbdt_cb_models': self.gbdt_model.cb_models,
            'gbdt_lgb_models': self.gbdt_model.lgb_models,
            'tcn_state_dicts': tcn_state_dicts,
            'tcn_weight': self.tcn_weight,
            'gbdt_weight': self.gbdt_weight,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v11_b 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.gbdt_model.cb_models = payload['gbdt_cb_models']
        self.gbdt_model.lgb_models = payload['gbdt_lgb_models']
        self.gbdt_model.is_fitted = True

        self.tcn_models = []
        for sd in payload['tcn_state_dicts']:
            tcn = SquatC1DilatedTCN()
            tcn.load_state_dict(sd)
            tcn.eval()
            self.tcn_models.append(tcn)

        self.tcn_weight = payload.get('tcn_weight', 0.35)
        self.gbdt_weight = payload.get('gbdt_weight', 0.65)
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.is_fitted = True
        return self
