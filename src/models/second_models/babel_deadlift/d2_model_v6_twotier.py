# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Module: d2_model_v6_twotier.py
Model Version: v6.0 Two-Tier (2계층 생체역학 모호성 정제 모델)

Architecture:
  - 10D 초정예 핵심 생체역학 피처셋 (다중공선성 및 저해 노이즈 배제)
  - Dual GBDT (LightGBM + CatBoost) + Tiny TCN Meta-Fusion (0.80 GBDT + 0.20 TCN)
  - 4단계 생체역학 모호성 스펙트럼 적용 및 Level 4 오염군(27건, 7.01%) 전수 폐기
  - 2계층 평가 구조:
    * 1계층 (Tier 1: Canonical Standard): Level 1(완전 명확군) 기준 평가 -> ROC-AUC 0.9392, F1 77.42%, Acc 90.41%
    * 2계층 (Tier 2: Real-World Robustness): Level 1~3(실무 수용군) 기준 평가 -> ROC-AUC 0.8153, F1 57.65%, Acc 77.29%
    * 오탐(FP) 93건 -> 54건 (-41.9% 격감), 미탐(FN) 30건 -> 18건 (-40.0% 격감)
"""

import os
import sys
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d2_model_v5_2_pruned import (
    extract_d2_v5_2_pruned_features, FEATURE_NAMES_10_D2, TinyD2TCN, VIEW_ALIAS_MAP
)

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
PRODUCTION_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d2_v6_twotier_production.pkl")


class BabelDeadliftD2ModelV6:
    """
    바벨 데드리프트 d2 (발과 무릎의 방향 일치 / 동적 무릎 외반) v6.0 Two-Tier 공식 분류기
    - Dual GBDT (LightGBM + CatBoost) + Tiny TCN Meta-Fusion (0.80 GBDT + 0.20 TCN)
    - 3-State 고신뢰 선택적 분류 (Tau Defect: 0.425, Tau Clean: 0.360)
    - 2계층(Two-Tier) 모호성 정제 파이프라인 탑재
    """
    DEFAULT_TAU_DEFECT = 0.425
    DEFAULT_TAU_CLEAN = 0.360

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PRODUCTION_WEIGHTS_PATH
        self.is_loaded = False
        self.lgb_model = None
        self.cat_model = None
        self.tcn_model = None
        self.ts_mean = None
        self.ts_std = None
        self.tau_defect = self.DEFAULT_TAU_DEFECT
        self.tau_clean = self.DEFAULT_TAU_CLEAN
        self.tier1_metrics = {}
        self.tier2_metrics = {}
        self.baseline_metrics = {}

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
        self.tier1_metrics = data.get("tier1_metrics", {})
        self.tier2_metrics = data.get("tier2_metrics", {})
        self.baseline_metrics = data.get("baseline_metrics", {})

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
        if self.tcn_model is not None and self.ts_mean is not None:
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
            'phase1_first_pull_ka_min': float(f10[6]),
            'foot_dev_pull_p90': float(f10[7])
        }

    def predict_3state(self, seq_64f_33_3d: np.ndarray, view_code: str = "view3") -> Dict[str, Any]:
        """
        3-State 고신뢰 선택적 분류 판정 (v6.0 Two-Tier 최적 임계치)
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
            'feature_subset': "10D_Pruned_Elite_TwoTier",
            'top_features': {
                name: round(float(f10[i]), 4) for i, name in enumerate(FEATURE_NAMES_10_D2)
            }
        }
