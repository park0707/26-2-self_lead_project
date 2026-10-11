# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Module: d2_model_v5_2_pruned.py
Model Version: v5.2 (Pruned Elite 10D 최적화 모델)
Key Features:
  - 10D 초정예 생체역학 피처 (25D -> 10D 60% 피처 다이어트)
  - 다중공선성(|r|>=0.85) 및 LOFO 저해 노이즈 피처 전수 제거
  - ROC-AUC: 0.7136 (+0.0157 향상)
  - 3-State F1-Score: 51.71% (+2.09%p 향상)
  - 정밀도: 43.44% (+7.18%p 향상)
  - 정확도: 71.55% (+9.95%p 향상)
  - False Positive(오탐): 116건 -> 69건 (-40.5% 감축)
"""

import os
import sys
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
PRODUCTION_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d2_v5_2_pruned_production.pkl")

# 공식 5대 시점 명칭 및 각도 정의 (AGENTS.md 표준 준수)
VIEW_ALIAS_MAP: Dict[str, str] = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

# 10D 초정예 핵심 피처 명칭 (25D에서 15개 제거된 10D 최적 서브셋)
FEATURE_NAMES_10_D2: List[str] = [
    "world_kawr_min",           # [0] 월드 좌표계 최소 KAWR (내측 붕괴 최저점)
    "world_kawr_p10",           # [1] 월드 좌표계 10백분위수 KAWR (LOFO 기여도 1위 핵심 앵커)
    "world_kawr_p50",           # [2] 월드 좌표계 중위수 KAWR (기저 정렬 기준선)
    "can_kawr_min",             # [3] 정규화 표준 좌표계 최소 KAWR (체격 무관 절대 정규화)
    "can_kawr_mean",            # [4] 정규화 표준 좌표계 평균 KAWR (전체 궤적 기저선)
    "can_valgus_abs_max",       # [5] 정규화 표준 좌표계 최대 외반 변위
    "phase1_first_pull_ka_min", # [6] 퍼스트풀(First Pull) 구간 최소 무릎 너비 비율
    "foot_dev_pull_p90",        # [7] 퍼스트풀 구간 발 진행방향 대비 무릎 편위 90백분위수 (d2 라벨 직결)
    "dev_foot_left_p90",        # [8] 좌측 발 기준 무릎 편위 피크
    "dev_foot_right_p90"        # [9] 우측 발 기준 무릎 편위 피크
]

# 25D 전체 인덱스에서 10D로 매핑되는 인덱스 튜플
PRUNED_10D_INDICES = [0, 2, 4, 5, 6, 11, 13, 21, 23, 24]


class TinyD2TCN(nn.Module):
    """
    파라미터 4,162개 경량 8채널 Tiny TCN
    """
    def __init__(self, n_features: int = 8, width: int = 16):
        super().__init__()
        self.stem = nn.Conv1d(2 * n_features, width, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, padding=d, dilation=d)
            for d in (1, 2, 4, 8)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(4, width) for _ in range(4)
        ])
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(3 * width, 1)

    def forward(self, x: torch.Tensor, quality: torch.Tensor, phase_indices: torch.Tensor):
        inp = torch.cat([x * quality, quality], dim=-1)
        h = self.stem(inp.transpose(1, 2))

        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(F.gelu(norm(conv(h))))

        pooled = []
        for p in range(3):
            mask = (phase_indices == p).float().unsqueeze(1)
            p_mean = (h * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
            pooled.append(p_mean)

        embedding = torch.cat(pooled, dim=1)
        logits = self.head(embedding).squeeze(-1)
        return logits, embedding


def extract_d2_v5_2_pruned_features(kps_33_64f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    64프레임 33관절 3D 텐서에서 10D 초정예 Tabular 피처 및 8채널 물리 시계열 추출
    Returns:
      - feat_10d: (10,) np.float32
      - ts_8ch: (64, 8) np.float32
      - q_mask: (64, 8) np.float32
    """
    from src.models.second_models.babel_deadlift.d2_model_v5_final import extract_d2_v5_final_features
    return extract_d2_v5_final_features(kps_33_64f)



class BabelDeadliftD2ModelV52:
    """
    바벨 데드리프트 d2 (발과 무릎의 방향 일치 / 동적 무릎 외반) v5.2 (Pruned Elite 10D) 공식 분류기
    - Dual GBDT (LightGBM + CatBoost) + Tiny TCN Meta-Fusion (0.80 GBDT + 0.20 TCN)
    - 3-State 고신뢰 선택적 분류 (Tau Defect: 0.425, Tau Clean: 0.360)
    """
    DEFAULT_TAU_DEFECT = 0.425
    DEFAULT_TAU_CLEAN = 0.360

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PRODUCTION_WEIGHTS_PATH
        self.is_loaded = False
        self.lgb_model = None
        self.cat_model = None
        self.tcn_model = None
        self.tau_defect = self.DEFAULT_TAU_DEFECT
        self.tau_clean = self.DEFAULT_TAU_CLEAN

        if os.path.exists(self.weights_path):
            self.load_weights(self.weights_path)

    def load_weights(self, path: str):
        with open(path, "rb") as fp:
            data = pickle.load(fp)

        self.lgb_model = data.get("lgb_model")
        self.cat_model = data.get("cat_model")
        self.tau_defect = data.get("tau_defect", self.DEFAULT_TAU_DEFECT)
        self.tau_clean = data.get("tau_clean", self.DEFAULT_TAU_CLEAN)

        self.ts_mean = data.get("ts_mean", None)
        self.ts_std = data.get("ts_std", None)

        if "tcn_state_dict" in data and data["tcn_state_dict"] is not None:
            self.tcn_model = TinyD2TCN(n_features=8, width=16)
            self.tcn_model.load_state_dict(data["tcn_state_dict"])
            self.tcn_model.eval()

        self.is_loaded = True

    def predict_proba(self, seq_64f_33_3d: np.ndarray) -> Dict[str, float]:
        """
        단일 시퀀스(64, 33, 3)에 대한 결함 확률 P(Defect) 및 정상 확률 P(Clean) 산출
        - 10D 초정예 Tabular 피처 GBDT + Tiny TCN Meta-Fusion (0.80 GBDT + 0.20 TCN)
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model weights not loaded from {self.weights_path}")

        f10, f_ts, f_q = extract_d2_v5_2_pruned_features(seq_64f_33_3d)

        # 1. GBDT 예측 (10D 피처)
        p_l = float(self.lgb_model.predict_proba(f10.reshape(1, -1))[0, 1]) if self.lgb_model else 0.5
        p_c = float(self.cat_model.predict_proba(f10.reshape(1, -1))[0, 1]) if self.cat_model else 0.5
        p_gbdt = 0.5 * p_l + 0.5 * p_c

        # 2. Tiny TCN 예측
        p_tcn = 0.5
        if self.tcn_model is not None:
            ts_norm = (f_ts[None, ...] - self.ts_mean) / self.ts_std
            bx = torch.tensor(ts_norm, dtype=torch.float32)
            bq = torch.tensor(f_q[None, ...], dtype=torch.float32)
            phase_arr = np.zeros((1, 64), dtype=np.int64)
            phase_arr[0, 21:46] = 1
            phase_arr[0, 46:64] = 2
            b_phase = torch.tensor(phase_arr, dtype=torch.long)

            with torch.no_grad():
                logits, _ = self.tcn_model(bx, bq, b_phase)
                p_tcn = float(torch.sigmoid(logits).item())

        # 3. Meta-Fusion (0.80 GBDT + 0.20 TCN)
        p_defect = 0.80 * p_gbdt + 0.20 * p_tcn
        p_clean = 1.0 - p_defect

        return {
            'prob_defect': p_defect,
            'prob_clean': p_clean,
            'prob_gbdt': p_gbdt,
            'prob_tcn': p_tcn,
            'world_kawr_min': float(f10[0]),
            'foot_dev_pull_p90': float(f10[7])
        }

    def predict_3state(self, seq_64f_33_3d: np.ndarray, view_code: str = "view3") -> Dict[str, Any]:
        """
        3-State 고신뢰 선택적 분류 판정 (v5.2 Pruned Elite 10D 최적 임계치)
        - State 0: CLEAN (정상 확정)
        - State 1: DEFECT (결함 확정 - 무릎 외반 / 발-무릎 방향 불일치)
        - State 2: RETAKE (판정 보류 - 재촬영 및 자세 가이드 유도)
        """
        probs = self.predict_proba(seq_64f_33_3d)
        p_def = probs['prob_defect']

        view_name = VIEW_ALIAS_MAP.get(view_code, view_code)

        if p_def >= self.tau_defect:
            state = 1
            state_name = "DEFECT"
            feedback = "바벨을 들어올릴 때 무릎이 발끝 방향보다 안쪽으로 모이고 있습니다. 발끝과 무릎을 같은 각도로 바깥쪽으로 벌리며 리프팅하세요."
        elif p_def <= self.tau_clean:
            state = 0
            state_name = "CLEAN"
            feedback = "무릎과 발의 정렬 상태가 완벽합니다! 척추와 무릎의 궤적이 정석적입니다."
        else:
            state = 2
            state_name = "RETAKE"
            feedback = f"현재 {view_name} 시점에서 바벨 또는 팔에 의해 무릎-발목 궤적이 가려져 판정이 보류되었습니다. 정면(시점 C) 또는 우측 45도(시점 B)에서 1회 다시 수행해 주세요."

        f10, _, _ = extract_d2_v5_2_pruned_features(seq_64f_33_3d)

        return {
            'state': state,
            'state_name': state_name,
            'prob_defect': round(p_def, 4),
            'prob_clean': round(probs['prob_clean'], 4),
            'prob_gbdt': round(probs['prob_gbdt'], 4),
            'prob_tcn': round(probs['prob_tcn'], 4),
            'view_code': view_code,
            'view_name': view_name,
            'feedback': feedback,
            'decision_window': [self.tau_clean, self.tau_defect],
            'feature_subset': "10D_Pruned_Elite",
            'top_features': {
                name: round(float(f10[i]), 4) for i, name in enumerate(FEATURE_NAMES_10_D2)
            }
        }
