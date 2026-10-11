# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v3.0 (d1_model_v3.py)
Status: ★ [공식 최신 프로덕션 모델] 형태 조건부 잔차화(Morphology Residualization) 및 3-State 선택적 판정 탑재
        - 1) 28D 생체역학 기본 피처 (스쿼트 Nadir 노이즈 소거 + 데드리프트 5대 특화 피처)
        - 2) 사지 비율 4D (팔/체간, 대퇴/하퇴, 다리/체간, 팔/다리 비율)
        - 3) 형태 조건부 셋업 잔차 2D (체형에 따른 기대 셋업각 및 요추 곡률과의 잔차) -> 총 34D 피처셋
        - 4) 24D 시계열 경량 Dilated TCN + 34D GBDT (LightGBM + CatBoost) Tri-Ensemble
        - 5) 공식 3-State 선택적 판정 [tau_normal=0.28, tau_defect=0.37] (보류율 9.61%, F1 72.85%)
"""

import os
import sys
import glob
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d, FEATURE_NAMES_28_D1

warnings.filterwarnings("ignore")

MP = dict(
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

FEATURE_NAMES_34_D1: List[str] = FEATURE_NAMES_28_D1 + [
    "d1_res_torso_pitch_deepest",       # [29] 체형(사지비율) 대비 셋업 체간각 잔차
    "d1_res_setup_lumbar_poly_a",       # [30] 체형 대비 셋업 요추 곡률 잔차
    "d1_ratio_arm_to_torso",            # [31] 팔 길이 대 체간 길이 비
    "d1_ratio_femur_to_tibia",          # [32] 대퇴 대 하퇴 길이 비
    "d1_ratio_leg_to_torso",            # [33] 다리 전체 대 체간 길이 비
    "d1_ratio_arm_to_leg"               # [34] 팔 대 다리 길이 비
]


def extract_morphology_ratios(kps_64: np.ndarray) -> np.ndarray:
    """피험자 상체, 하체, 팔 뼈 길이 비율 추출 (6D)"""
    k_base = np.median(kps_64[:8], axis=0)  # (33, 3)

    sh_mid = 0.5 * (k_base[MP['LSH']] + k_base[MP['RSH']])
    hip_mid = 0.5 * (k_base[MP['LHIP']] + k_base[MP['RHIP']])
    torso_len = max(1e-4, float(np.linalg.norm(sh_mid - hip_mid)))

    l_arm = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['LELB']]) + np.linalg.norm(k_base[MP['LELB']] - k_base[MP['LWR']])
    r_arm = np.linalg.norm(k_base[MP['RSH']] - k_base[MP['RELB']]) + np.linalg.norm(k_base[MP['RELB']] - k_base[MP['RWR']])
    arm_len = 0.5 * (l_arm + r_arm)

    l_femur = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['LKNE']])
    r_femur = np.linalg.norm(k_base[MP['RHIP']] - k_base[MP['RKNE']])
    femur_len = 0.5 * (l_femur + r_femur)

    l_tibia = np.linalg.norm(k_base[MP['LKNE']] - k_base[MP['LANK']])
    r_tibia = np.linalg.norm(k_base[MP['RKNE']] - k_base[MP['RANK']])
    tibia_len = 0.5 * (l_tibia + r_tibia)

    leg_len = femur_len + tibia_len
    sh_w = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['RSH']])
    hip_w = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['RHIP']])

    return np.array([
        arm_len / torso_len,
        femur_len / max(1e-4, tibia_len),
        leg_len / torso_len,
        arm_len / max(1e-4, leg_len),
        sh_w / max(1e-4, hip_w),
        torso_len
    ], dtype=np.float32)


class MorphologyResidualizer:
    """체형 비율에 따른 정상 셋업 자세 기대치 Ridge 회귀 및 잔차 변환기"""
    def __init__(self, alpha: float = 15.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_pitch = Ridge(alpha=alpha)
        self.ridge_poly = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, ratios_train_normal: np.ndarray, pitch_train_normal: np.ndarray, poly_train_normal: np.ndarray):
        Z = self.scaler.fit_transform(ratios_train_normal)
        self.ridge_pitch.fit(Z, pitch_train_normal)
        self.ridge_poly.fit(Z, poly_train_normal)
        self.is_fitted = True
        return self

    def transform(self, ratios: np.ndarray, pitch_deepest: float, poly_a: float) -> Tuple[float, float]:
        if not self.is_fitted:
            return 0.0, 0.0
        Z = self.scaler.transform(ratios.reshape(1, -1))
        exp_pitch = float(self.ridge_pitch.predict(Z)[0])
        exp_poly = float(self.ridge_poly.predict(Z)[0])
        return pitch_deepest - exp_pitch, poly_a - exp_poly


class BabelDeadliftD1ModelV3:
    """
    바벨 데드리프트 d1. 척추의 중립 위반 v3.0 공식 프로덕션 모델
    """
    DEFECT_CODE = "d1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 데드리프트"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_normal: float = 0.28,
        tau_defect: float = 0.37,
        decision_threshold: float = 0.35
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.decision_threshold = decision_threshold

        # 5-Fold 실측 최적 가중치 (34D GBDT 45% + 24D TCN 55%)
        self.w_gbdt = 0.45
        self.w_tcn = 0.55

        self.clf_lgb: Optional[Any] = None
        self.clf_cat: Optional[Any] = None
        self.residualizer: Optional[MorphologyResidualizer] = None
        self.tcn_models: List[SquatC1DilatedTCN] = []
        self.is_fitted = False

        if weights_dir is None:
            weights_dir = os.path.join(CURRENT_DIR, "weights")
        self.weights_dir = weights_dir

        self._try_load_weights()

    def _try_load_weights(self):
        pkl_path = os.path.join(self.weights_dir, "d1_v3_gbdt_models.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
                self.clf_lgb = d["clf_lgb"]
                self.clf_cat = d["clf_cat"]
                self.residualizer = d.get("residualizer", None)
                self.decision_threshold = d.get("decision_threshold", 0.35)
                self.tau_normal = d.get("tau_normal", 0.28)
                self.tau_defect = d.get("tau_defect", 0.37)

            tcn_files = sorted(glob.glob(os.path.join(self.weights_dir, "d1_v3_tcn_fold*.pth")))
            self.tcn_models = []
            for tf in tcn_files:
                m = SquatC1DilatedTCN(in_channels=24)
                m.load_state_dict(torch.load(tf, map_location="cpu"))
                m.eval()
                self.tcn_models.append(m)

            self.is_fitted = (self.clf_lgb is not None and len(self.tcn_models) > 0)

    def extract_features_34d(self, kps_33_64f: np.ndarray) -> np.ndarray:
        """단일 64프레임 시퀀스로부터 34D 정제 피처 추출"""
        f28 = extract_d1_features_28d(kps_33_64f)
        ratios = extract_morphology_ratios(kps_33_64f)

        if self.residualizer is not None and self.residualizer.is_fitted:
            res_pitch, res_poly = self.residualizer.transform(ratios, float(f28[17]), float(f28[23]))
        else:
            res_pitch, res_poly = 0.0, 0.0

        f34 = np.concatenate([
            f28,
            np.array([res_pitch, res_poly], dtype=np.float32),
            ratios[:4]
        ])
        return f34

    def predict_proba_from_seq(self, kps_33_64f: np.ndarray) -> float:
        """단일 64프레임 시퀀스로부터 d1 결함 확률 p_defect 산출"""
        if not self.is_fitted:
            raise RuntimeError("D1 v3.0 모델 가중치가 로드되지 않았습니다.")

        f34 = self.extract_features_34d(kps_33_64f)[None, :]
        p_lgb = float(self.clf_lgb.predict_proba(f34)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(f34)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        # TCN
        t24 = extract_temporal_24d(kps_33_64f, None)[None, ...]  # (1, 64, 24)
        x_t = torch.tensor(t24.transpose(0, 2, 1), dtype=torch.float32)

        p_tcns = []
        with torch.no_grad():
            for m in self.tcn_models:
                m.eval()
                logits = m(x_t).squeeze(-1)
                p_tcns.append(float(torch.sigmoid(logits).item()))
        p_tcn = float(np.mean(p_tcns))

        p_final = float(self.w_gbdt * p_gbdt + self.w_tcn * p_tcn)
        return p_final

    def predict_selective(
        self,
        kps_33_64f: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        바벨 데드리프트 d1 3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED)
        """
        tau_norm = tau_normal if tau_normal is not None else self.tau_normal
        tau_def = tau_defect if tau_defect is not None else self.tau_defect

        f34 = self.extract_features_34d(kps_33_64f)
        active_rom = float(f34[6])  # d1_active_rom

        # 1. 가동범위 안전 게이트
        if active_rom < 0.05:
            return {
                "status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "바벨 리프팅 가동범위가 너무 작아 척추 중립 상태를 판정할 수 없습니다.",
                "action": "바벨을 바닥부터 골반 높이까지 충분한 가동범위로 들어 올려 주세요."
            }

        p_defect = self.predict_proba_from_seq(kps_33_64f)

        # 2. 3-State 상태 결정
        if p_defect <= tau_norm:
            status = "NORMAL"
            reason = "CLEAR_DECISION"
            msg = "데드리프트 중 척추의 자연스러운 중립 곡선이 바르게 유지되었습니다."
            act = None
        elif p_defect >= tau_def:
            status = "DEFECT"
            reason = "CLEAR_DECISION"
            msg = "바벨 리프팅 시 요추가 말리거나(라운드백) 척추 중립이 무너졌습니다."
            act = "코어 복압을 강하게 유지하고, 가슴을 펴며 엉덩이 힌지를 사용해 들어 올리세요."
        else:
            status = "UNDETERMINED"
            reason = "MODEL_UNCERTAINTY"
            msg = "척추 굴곡 패턴이 경계 구간에 있어 정확한 판정을 위해 재촬영을 권장합니다."
            act = "허리 각도가 잘 관측되도록 대각선 45도 측면(시점 B/D)에서 1회 다시 수행해 주세요."

        return {
            "status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(min(1.0, 2.0 * abs(p_defect - 0.5)), 4),
            "reason_code": reason,
            "message": msg,
            "action": act
        }
