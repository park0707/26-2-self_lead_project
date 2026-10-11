# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v3.0 (sd2_v3) - Two-Tier Phase 2 & 3 통합 고도화 모델
Module: sd2_v3_classifier.py
Purpose: 스티프 데드리프트 2번 오류(sd2_v3: 척추 중립 위반) 전용 Two-Tier Tri-Ensemble & 3-State 추론 및 진단 인터페이스
  - 바벨 기울임(Key 211~214), 허리 과신전(Key 215), 등 굽힘(Key 216) 3대 결함 통합 감지
  - Tier 1(명확군) 및 Tier 2(전체군) 계층별 신뢰도 진단 지원
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "d:/"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.stiff_spine_features_v3 import (
    extract_stiff_spine_features_v3,
    FEATURE_NAMES_V3
)
from src.features.stiff_spine_feature_extractor import extract_stiff_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

BUNDLE_PATH = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")


class SD2V3Classifier:
    """바벨 스티프 데드리프트 2번 오류(sd2_v3: 척추 중립 위반) 전용 Two-Tier SOTA Expert"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd2_v3 번들 파일을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'sd2_v3')
        self.version = bundle.get('version', 'v3.0 (Two-Tier Phase 2 & 3)')
        self.defect_name = bundle.get('defect', 'sd2 (척추의 중립 위반)')
        self.feature_names = bundle.get('feature_names', FEATURE_NAMES_V3)
        
        self.tcn_weights = bundle['tcn_weights']
        self.lgb_models = bundle['lgb_models']
        self.cat_models = bundle['cat_models']
        self.scalers = bundle.get('scalers', None)
        self.weights = bundle.get('ensemble_weights', {'tcn': 0.25, 'lgb': 0.25, 'cat': 0.50})
        
        self.tau_clean = bundle['thresholds']['tau_clean']
        self.tau_defect = bundle['thresholds']['tau_defect']
        self.binary_threshold = bundle['thresholds'].get('binary_threshold', 0.50)

        # PyTorch TCN 모델 초기화 (5-Fold)
        self.device = torch.device("cpu")
        self.tcn_models = []
        for state in self.tcn_weights:
            m = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(self.device)
            m.load_state_dict(state)
            m.eval()
            self.tcn_models.append(m)

    def extract_features(self, kps_64f: np.ndarray, spine_64f: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """(64, 33, 3) 시퀀스로부터 v3 30D 생체역학 피처 및 24D 시계열 텐서 추출"""
        feats_30d = extract_stiff_spine_features_v3(kps_64f, spine_64f)
        temp_24d = extract_stiff_temporal_24d(kps_64f, spine_64f)
        return {
            'features_30d': feats_30d,
            'temporal_24d': temp_24d
        }

    def predict_proba(self, kps_64f: np.ndarray, spine_64f: Optional[np.ndarray] = None) -> float:
        """단일 시퀀스(64, 33, 3)에 대한 척추 중립 결함 확률 P(Defect) 산출 (0.0 ~ 1.0)"""
        extracted = self.extract_features(kps_64f, spine_64f)
        f30 = extracted['features_30d'].reshape(1, -1)
        t24 = extracted['temporal_24d']  # (64, 24)

        # 1. 5-Fold TCN 추론
        tcn_x = torch.tensor(t24.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
        tcn_probs = []
        with torch.no_grad():
            for m in self.tcn_models:
                out = m(tcn_x)
                p = torch.sigmoid(out).item()
                tcn_probs.append(p)
        mean_tcn_p = float(np.mean(tcn_probs))

        # 2. 5-Fold LightGBM & CatBoost 추론 (각 Fold별 스케일러 적용)
        lgb_probs = []
        cat_probs = []
        for i in range(len(self.lgb_models)):
            if self.scalers is not None and i < len(self.scalers):
                f30_scaled = self.scalers[i].transform(f30)
            else:
                f30_scaled = f30
            lgb_probs.append(float(self.lgb_models[i].predict_proba(f30_scaled)[0, 1]))
            cat_probs.append(float(self.cat_models[i].predict_proba(f30_scaled)[0, 1]))
        
        mean_lgb_p = float(np.mean(lgb_probs))
        mean_cat_p = float(np.mean(cat_probs))

        # 4. Tri-Ensemble Soft-Voting
        p_defect = (
            self.weights['tcn'] * mean_tcn_p +
            self.weights['lgb'] * mean_lgb_p +
            self.weights['cat'] * mean_cat_p
        )
        return float(np.clip(p_defect, 0.0, 1.0))

    def diagnose(self, kps_64f: np.ndarray, spine_64f: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        단일 시퀀스 종합 진단 및 세부 결함(기울임, 과신전, 등 굽힘) 피드백 제공
        """
        extracted = self.extract_features(kps_64f, spine_64f)
        f30 = extracted['features_30d']
        p_def = self.predict_proba(kps_64f, spine_64f)
        binary_pred = 1 if p_def >= self.binary_threshold else 0

        # 세부 결함 유형별 분석 (피처 기반 룰 감지)
        # f30[18]: barbell_roll_max, f30[23]: lockout_posterior_lean, f30[27]: thoracic_kyphosis_peak
        sub_defects = []
        if f30[18] >= 4.5:
            sub_defects.append("바벨 좌우 기울임(Barbell Tilt Asymmetry)")
        if f30[23] >= 3.5:
            sub_defects.append("기립 락아웃 시 허리 과신전(Lumbar Hyperextension)")
        if f30[27] >= 0.05:
            sub_defects.append("상체 하강 시 등/어깨 굽힘(Round Back / Thoracic Kyphosis)")

        # 3-State 판정
        if p_def < self.tau_clean:
            state = 0
            state_label = "CLEAN"
            feedback = "척추의 중립이 전 구간에서 훌륭하게 유지되고 있습니다. 완벽한 자세입니다!"
        elif p_def >= self.tau_defect:
            state = 1
            state_label = "DEFECT"
            if sub_defects:
                detail_str = ", ".join(sub_defects)
                feedback = f"척추 중립 위반이 감지되었습니다 [{detail_str}]. 바벨을 수평으로 유지하고 허리가 과도하게 젖혀지거나 굽지 않도록 코어에 힘을 주세요."
            else:
                feedback = "척추의 중립이 무너졌습니다. 체간을 곧게 유지하고 고관절 힌지에 집중하세요."
        else:
            state = 2
            state_label = "RETAKE"
            feedback = "동작의 척추 정렬이 모호하게 측정되었습니다. 카메라 각도를 측면 45도로 재조정하여 1회 다시 수행해 주세요."

        return {
            'model': self.model_name,
            'version': self.version,
            'defect_name': self.defect_name,
            'defect_id': 'sd2',
            'probability_defect': float(p_def),
            'binary_decision': binary_pred,
            'binary_label': "DEFECT" if binary_pred == 1 else "CLEAN",
            'three_state_decision': state,
            'three_state_label': state_label,
            'tau_clean': self.tau_clean,
            'tau_defect': self.tau_defect,
            'detected_sub_defects': sub_defects,
            'feedback_korean': feedback
        }
