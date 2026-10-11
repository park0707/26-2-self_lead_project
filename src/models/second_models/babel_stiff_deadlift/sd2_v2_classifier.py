# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v2.0 (sd2_v2) - 20D 정예 피처 최적화 모델
Module: sd2_v2_classifier.py
Purpose: 스티프 데드리프트 2번 오류(sd2_v2: 척추 중립 위반) 전용 정예 Tri-Ensemble & 3-State 추론 및 진단 인터페이스
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np

# Windows OpenMP 충돌 방지
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

from src.features.stiff_spine_feature_extractor import (
    extract_stiff_spine_features_30d,
    extract_stiff_temporal_24d
)
from stiff_spine_features_v2 import (
    select_v2_features,
    V2_FEATURE_NAMES,
    V2_SELECTED_INDICES,
    PRUNED_FEATURES_INFO
)
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

BUNDLE_PATH = os.path.join(CURRENT_DIR, "sd2_model_v2_weights.pkl")


class SD2V2Classifier:
    """바벨 스티프 데드리프트 2번 오류(sd2_v2: 척추 중립 위반) 전용 20D 정예 Tri-Ensemble & 3-State Expert"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            alt_path = os.path.join(CURRENT_DIR, "weights", "sd2_v2_bundle.pkl")
            if os.path.exists(alt_path):
                bundle_path = alt_path
            else:
                raise FileNotFoundError(f"sd2_v2 번들 파일을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'sd2_v2')
        self.version = bundle.get('version', 'v2.0 (Diet 20D)')
        self.defect_name = bundle.get('defect', 'sd2 (척추의 중립 위반)')
        
        self.feature_indices = bundle.get('feature_indices', V2_SELECTED_INDICES)
        self.feature_names = bundle.get('feature_names', V2_FEATURE_NAMES)
        
        self.tcn_weights = bundle['tcn_weights']
        self.lgb_models = bundle['lgb_models']
        self.cat_models = bundle['cat_models']
        self.weights = bundle.get('ensemble_weights', {'tcn': 0.20, 'lgb': 0.20, 'cat': 0.60})
        
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
        """(64, 33, 3) 시퀀스로부터 20D 정예 생체역학 피처 및 24D 시계열 텐서 추출"""
        feats_30d = extract_stiff_spine_features_30d(kps_64f, spine_64f)
        feats_20d = select_v2_features(feats_30d)
        temp_24d = extract_stiff_temporal_24d(kps_64f, spine_64f)
        return {
            'features_20d': feats_20d,
            'features_30d_raw': feats_30d,
            'temporal_24d': temp_24d
        }

    def predict_proba(self, kps_64f: np.ndarray, spine_64f: Optional[np.ndarray] = None) -> float:
        """단일 시퀀스(64, 33, 3)에 대한 척추 중립 결함 확률 P(Defect) 산출 (0.0 ~ 1.0)"""
        extracted = self.extract_features(kps_64f, spine_64f)
        f20 = extracted['features_20d'].reshape(1, -1)
        t24 = extracted['temporal_24d']  # (64, 24)

        # 1. 5-Fold TCN 추론
        tcn_x = torch.tensor(t24.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 24, 64)
        tcn_probs = []
        with torch.no_grad():
            for m in self.tcn_models:
                out = m(tcn_x)
                p = torch.sigmoid(out).item()
                tcn_probs.append(p)
        mean_tcn_p = float(np.mean(tcn_probs))

        # 2. 5-Fold LightGBM 추론 (20D 정예 피처)
        lgb_probs = [float(m.predict_proba(f20)[0, 1]) for m in self.lgb_models]
        mean_lgb_p = float(np.mean(lgb_probs))

        # 3. 5-Fold CatBoost 추론 (20D 정예 피처)
        cat_probs = [float(m.predict_proba(f20)[0, 1]) for m in self.cat_models]
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
        단일 시퀀스 진단 종합 리포트
        반환:
          - p_defect: 결함 확률 (0.0 ~ 1.0)
          - binary_decision: 이진 판정 (0: 정상, 1: 결함)
          - three_state: 3상태 판정 (0: CLEAN, 1: DEFECT, 2: RETAKE)
          - feedback: 사용자 친화적 교정 피드백 메시지
        """
        p_def = self.predict_proba(kps_64f, spine_64f)
        binary_pred = 1 if p_def >= self.binary_threshold else 0

        # 3-State 판정
        if p_def < self.tau_clean:
            state = 0
            state_label = "CLEAN"
            feedback = "척추의 중립이 올바르게 유지되고 있습니다. 훌륭한 자세입니다!"
        elif p_def >= self.tau_defect:
            state = 1
            state_label = "DEFECT"
            feedback = "척추의 중립이 무너졌습니다. 좌우로 몸이 기울어지거나 어깨가 말리지 않도록 체간을 곧게 유지하세요."
        else:
            state = 2
            state_label = "RETAKE"
            feedback = "척추 정렬이 모호하게 측정되었습니다. 카메라 각도를 정면 또는 45도 방향으로 재조정하여 1회 다시 수행해 주세요."

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
            'feedback_korean': feedback
        }
