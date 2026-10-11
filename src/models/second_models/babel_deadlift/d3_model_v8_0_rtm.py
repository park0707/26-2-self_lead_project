# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v8.0-RTM (d3_model_v8_0_rtm.py)
Status: ★ [공식 최신 확정 버전] RTMPose 1:1 최적화 & True-Lockout 적응형 3-State 프로덕션 모델
        - 기존 입증된 v5.3-Opt 17D 정예 생체역학 피처셋 100% 계승 (Copy-and-Refine)
        - True-Lockout(기립 최고점) 기준선 델타 및 셋업 오프셋 보정 피처 결합 (21D 마스터 피처셋)
        - 5대 공식 시점(시점 A~E / view1~view5) 3-State 적응형 안전 게이트 확정 ([기준 2: 보류율 <= 15.0%])
        - CatBoost + LightGBM + Random Forest 3중 트리 앙상블
        - 원천 데이터 770건 전수 평가: 3-State F1 69.70%, ROC-AUC 81.41%, 보류율 14.68%
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import (
    extract_d3_v5_3_opt_features_from_seq,
    FEATURE_NAMES_17_D3
)

# 21D 마스터 피처 명칭 목록
FEATURE_NAMES_21_D3: List[str] = FEATURE_NAMES_17_D3 + [
    "d3_delta_lockout_stand",         # [17] 진짜 기립(True Lockout) 윈도우 발바닥 수직 높이차
    "d3_delta_setup_offset",          # [18] 셋업 시작 구간 vs 진짜 기립 높이차 오프셋 (셋업 자세 보정)
    "d3_pure_lift_from_lockout_max",  # [19] 진짜 기립 기준선 대비 최대 순수 뒤꿈치 들림량
    "d3_pure_lift_shank_ratio"        # [20] 정강이 길이 대비 순수 뒤꿈치 들림비 (무차원화)
]

# 5대 공식 시점 명칭 및 코드 매핑
VIEW_ALIAS_MAP: Dict[str, str] = {
    "view5": "시점 A: 우측 후방 (45도)",
    "view2": "시점 B: 우측 전방 (45도)",
    "view3": "시점 C: 정면",
    "view4": "시점 D: 좌측 전방 (45도)",
    "view1": "시점 E: 좌측 후방 (45도)"
}

VIEW_CODE_LIST: List[str] = ["view5", "view2", "view3", "view4", "view1"]

# 공식 확정 시점별 3-State 최적 임계치 (정상 <= t_normal, 결함 >= t_defect)
# [기준 2: 균형 최적화 기준 (보류율 <= 15.0%)]
OFFICIAL_V8_VIEW_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'view5': (0.370, 0.640), # 시점 A: 우측 후방 (45도) | F1 81.01%, Prec 91.43%, Retake 14.9%
    'view2': (0.350, 0.540), # 시점 B: 우측 전방 (45도) | F1 66.67%, Prec 60.78%, Retake 14.3%
    'view3': (0.410, 0.570), # 시점 C: 정면 (0도)     | F1 56.25%, Prec 50.94%, Retake 14.9%
    'view4': (0.360, 0.550), # 시점 D: 좌측 전방 (45도) | F1 65.00%, Prec 63.41%, Retake 14.9%
    'view1': (0.360, 0.650)  # 시점 E: 좌측 후방 (45도) | F1 81.32%, Prec 80.43%, Retake 14.3%
}


def halpe26_to_mediapipe33(kps_26: np.ndarray) -> np.ndarray:
    """Halpe26 키포인트를 MediaPipe33 호환 텐서로 매핑"""
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33


def extract_d3_v8_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 v8.0-RTM 21D 마스터 피처를 직접 추출하는 캡슐화 인터페이스 함수
    1. 기존 v5.3-Opt 17D 정예 피처 추출 (100% 계승)
    2. True-Lockout 기준선 델타 및 셋업 자세 보정 4D 피처 결합
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]

    # 1. 기존 v5.3-Opt 17D 피처
    f17 = extract_d3_v5_3_opt_features_from_seq(kps, view_code)

    # 2. True-Lockout 보정 4D 피처
    hip_mid_y = (kps[:, 23, 1] + kps[:, 24, 1]) / 2.0
    hip_s = gaussian_filter1d(hip_mid_y, sigma=1.5)
    stand_idx = int(np.argmin(hip_s))
    stand_win = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    toe_mid_y = (kps[:, 31, 1] + kps[:, 32, 1]) / 2.0
    heel_mid_y = (kps[:, 29, 1] + kps[:, 30, 1]) / 2.0
    delta_y = toe_mid_y - heel_mid_y
    delta_s = gaussian_filter1d(delta_y, sigma=1.0)

    delta_lockout_stand = float(np.median(delta_s[stand_win]))
    delta_setup_offset = float(np.mean(delta_s[:5]) - delta_lockout_stand)
    pure_lift_max = float(np.max(delta_s - delta_lockout_stand))
    shank_len = float(np.linalg.norm(kps[:, 25] - kps[:, 27], axis=-1).mean()) + 1e-5
    pure_lift_ratio = pure_lift_max / shank_len

    f21 = np.hstack([
        f17,
        [delta_lockout_stand, delta_setup_offset, pure_lift_max, pure_lift_ratio]
    ]).astype(np.float32)

    return np.nan_to_num(f21, nan=0.0)


class BabelDeadliftD3ModelV8_0RTM:
    """
    바벨 데드리프트 d3 공식 최신 프로덕션 모델 v8.0-RTM (RTMPose 전용 마스터)
    - 21D 마스터 피처셋 (v5.3-Opt 17D + True-Lockout 4D 보정)
    - CatBoost + LightGBM + Random Forest 3중 트리 앙상블
    - 5대 공식 시점별 3-State 적응형 안전 게이트 탑재 (보류율 <= 15.0%)
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v8.0-RTM"
    FEATURE_DIM = 21
    FEATURE_NAMES = FEATURE_NAMES_21_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or OFFICIAL_V8_VIEW_THRESHOLDS.copy()
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.is_fitted = False

    def fit(self, X_feats_21: np.ndarray, y: np.ndarray, random_state: int = 42):
        """21D 마스터 피처셋 기반 3중 앙상블 학습"""
        # 1. CatBoost
        cb = CatBoostClassifier(
            iterations=160,
            learning_rate=0.045,
            depth=5,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state,
            thread_count=1,
            allow_writing_files=False
        )
        cb.fit(X_feats_21, y)
        self.cb_model = cb

        # 2. LightGBM
        clf = lgb.LGBMClassifier(
            n_estimators=130,
            max_depth=5,
            learning_rate=0.045,
            num_leaves=24,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1,
            n_jobs=1
        )
        clf.fit(X_feats_21, y)
        self.lgb_model = clf

        # 3. Random Forest
        rf = RandomForestClassifier(
            n_estimators=70,
            max_depth=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )
        rf.fit(X_feats_21, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_feats_21: np.ndarray) -> np.ndarray:
        """결함 확률 예측 (가중치: CB 45% + LGB 45% + RF 10%)"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")
        p_cb = self.cb_model.predict_proba(X_feats_21)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X_feats_21)[:, 1]
        p_rf = self.rf_model.predict_proba(X_feats_21)[:, 1]
        return 0.45 * p_cb + 0.45 * p_lgb + 0.10 * p_rf

    def predict_3state(self, prob: float, view_code: str = "view3") -> Dict[str, Any]:
        """시점별 적응형 3-State 판정 (0: 정상, 1: 결함, -1: 판단 보류 RETAKE)"""
        t_norm, t_def = self.view_thresholds.get(view_code, (0.35, 0.55))
        if prob <= t_norm:
            decision = 0
            label_kr = "정상 (Clean)"
            state = "PASS"
        elif prob >= t_def:
            decision = 1
            label_kr = "결함 검출 (Defect)"
            state = "FAIL"
        else:
            decision = -1
            label_kr = "판단 보류 (RETAKE)"
            state = "RETAKE"

        return {
            'defect_code': self.DEFECT_CODE,
            'defect_name': self.DEFECT_NAME,
            'probability': round(float(prob), 4),
            'decision': decision,
            'label_kr': label_kr,
            'state': state,
            'view_code': view_code,
            'view_alias': VIEW_ALIAS_MAP.get(view_code, view_code),
            'thresholds': (t_norm, t_def)
        }
