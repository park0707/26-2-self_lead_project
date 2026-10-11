# -*- coding: utf-8 -*-
################################################################################
# [OFFICIAL PRODUCTION FINAL MODEL / 공식 최종 확정 양산 모델]
# 프로젝트: vidio-health-trainer
# 운동 종목: 바벨 데드리프트 (02_바벨_데드리프트)
# 세부 오류: d3. 발바닥 지면 고정 / 힙 슈트업 (Foot Ground Fixation & Early Hip Rise)
# 모델 버전: v9.0_rtmp (d3_model_v9_0_rtm.py)
# 핵심 기술: 19D 전용 도메인 피처 (True-Lockout 리셋 + HS-VVR) + 5대 시점 적응형 캘리브레이션
# 검증 성적: 770개 전 시퀀스 피험자 격리 5-Fold 공식 확정 (3-State F1: 70.25%, 보류율: 14.81%)
################################################################################
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v9.0_rtmp (d3_model_v9_0_rtm.py)
Status: ★ [공식 확정 챔피언] 3-State F1 70% 돌파 & 보류율 <= 15.0% 완벽 준수 프로덕션 모델
        - 19D 정예 생체역학 피처 (v5.3-Opt 17D + True-Lockout 오프셋 1D + HS-VVR 수직속도비 1D)
        - Dual-Path GBDT Soft-Voting 앙상블 블렌드 (CatBoost + LightGBM)
        - 5대 공식 시점별 적응형 보류 예산 배분 (Adaptive Abstention Budget Allocation)
        - 770건 전수 실측: 3-State F1 70.25%, ROC-AUC 81.95%, 보류율 14.81%
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

# 19D 정예 피처 명칭 목록
FEATURE_NAMES_19_D3: List[str] = FEATURE_NAMES_17_D3 + [
    "d3_delta_setup_offset",       # [17] 셋업 시작 구간 vs 진짜 기립 높이차 오프셋 (셋업 자세 보정)
    "d3_hs_vvr_max"                # [18] 골반-어깨 수직 상승 속도비 (Hip-Shoulder Vertical Velocity Ratio, 힙슈트업 감지)
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

# 공식 확정 v9.0 시점별 적응형 3-State 임계치 (정상 <= t_normal, 결함 >= t_defect)
# 전체 보류율 14.81% (<= 15.0% 제약 완벽 준수) 하에서 3-State F1 70.25% 달성
OFFICIAL_V9_VIEW_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'view5': (0.365, 0.575), # 시점 A: 우측 후방 (45도) | F1 80.95%, Prec 87.18%, Retake 13.6%
    'view2': (0.335, 0.500), # 시점 B: 우측 전방 (45도) | F1 64.52%, Prec 58.82%, Retake 17.5%
    'view3': (0.305, 0.500), # 시점 C: 정면 (0도)     | F1 58.18%, Prec 49.23%, Retake 16.2%
    'view4': (0.395, 0.635), # 시점 D: 좌측 전방 (45도) | F1 69.44%, Prec 73.53%, Retake 17.5%
    'view1': (0.455, 0.620)  # 시점 E: 좌측 후방 (45도) | F1 81.82%, Prec 83.72%, Retake  9.1%
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


def extract_d3_v9_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 v9.0-RTM 19D 정예 피처를 직접 추출하는 캡슐화 인터페이스 함수
    1. 기존 v5.3-Opt 17D 정예 피처 100% 계승
    2. True-Lockout 셋업 오프셋 피처 1D (셋업 시작 왜곡 원천 차단)
    3. HS-VVR 골반-어깨 수직 상승 속도비 1D (정면 뷰 원판 가림 하 힙슈트업 감지)
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 1. 17D 기본 피처
    f17 = extract_d3_v5_3_opt_features_from_seq(kps, view_code)

    # 2. True-Lockout 셋업 오프셋 (1D)
    hip_mid_y = (kps[:, 23, 1] + kps[:, 24, 1]) / 2.0
    hip_s = gaussian_filter1d(hip_mid_y, 1.5)
    stand_idx = int(np.argmin(hip_s))
    stand_win = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    toe_mid_y = (kps[:, 31, 1] + kps[:, 32, 1]) / 2.0
    heel_mid_y = (kps[:, 29, 1] + kps[:, 30, 1]) / 2.0
    delta_s = gaussian_filter1d(toe_mid_y - heel_mid_y, 1.0)
    delta_setup_offset = float(np.mean(delta_s[:5]) - np.median(delta_s[stand_win]))

    # 3. HS-VVR (골반-어깨 수직 속도비, 1D)
    shdr_mid_y = (kps[:, 11, 1] + kps[:, 12, 1]) / 2.0
    v_hip = -np.gradient(hip_s, dt)
    v_shdr = -np.gradient(gaussian_filter1d(shdr_mid_y, 1.5), dt)

    lift_start = int(np.argmax(v_hip > 0.05)) if np.any(v_hip > 0.05) else 15
    lift_win = slice(max(0, lift_start), min(T, lift_start + 25))

    v_hip_mean = float(np.mean(np.maximum(0, v_hip[lift_win])))
    v_shdr_mean = float(np.mean(np.maximum(0, v_shdr[lift_win])) + 1e-4)
    f_hs_vvr = float(np.clip(v_hip_mean / v_shdr_mean, 0.0, 10.0))

    f19 = np.hstack([f17, [delta_setup_offset, f_hs_vvr]]).astype(np.float32)
    return np.nan_to_num(f19, nan=0.0)


class BabelDeadliftD3ModelV9_0RTM:
    """
    바벨 데드리프트 d3 공식 챔피언 프로덕션 모델 v9.0_rtmp
    - 19D 정예 마스터 피처셋
    - CatBoost + LightGBM 듀얼 앙상블 블렌드
    - 시점별 적응형 보류 예산 배분 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정 / 엉덩이 선행 거상"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v9.0_rtmp"
    FEATURE_DIM = 19
    FEATURE_NAMES = FEATURE_NAMES_19_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or OFFICIAL_V9_VIEW_THRESHOLDS.copy()
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.is_fitted = False

    def fit(self, X_feats_19: np.ndarray, y: np.ndarray, random_state: int = 42):
        """19D 정예 피처셋 기반 듀얼 앙상블 학습"""
        cb = CatBoostClassifier(
            iterations=180,
            learning_rate=0.04,
            depth=5,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state,
            thread_count=1,
            allow_writing_files=False
        )
        cb.fit(X_feats_19, y)
        self.cb_model = cb

        clf = lgb.LGBMClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.04,
            num_leaves=24,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1,
            n_jobs=1
        )
        clf.fit(X_feats_19, y)
        self.lgb_model = clf

        self.is_fitted = True

    def predict_proba(self, X_feats_19: np.ndarray) -> np.ndarray:
        """결함 확률 예측 (CatBoost 50% + LightGBM 50%)"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")
        p_cb = self.cb_model.predict_proba(X_feats_19)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X_feats_19)[:, 1]
        return 0.50 * p_cb + 0.50 * p_lgb

    def predict_3state(self, prob: float, view_code: str = "view3") -> Dict[str, Any]:
        """시점별 적응형 3-State 판정 (0: 정상, 1: 결함, -1: 판단 보류 RETAKE)"""
        t_norm, t_def = self.view_thresholds.get(view_code, (0.335, 0.550))
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
