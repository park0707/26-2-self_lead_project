# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v3.1 (d1_model_v3_1.py)
Status: ★ [v3.1 고도화 모델]
        1) 우측 시점(A, B) 성능 강화를 위한 체간 시상면 투영 비틀림 및 대칭 정규화
        2) d5(발바닥 지면 고정 실패/뒤꿈치 들림)에 의한 상체 쏠림 성분의 직교 분리(Orthogonal Decoupling)
        3) 34D -> 36D 정제 피처셋
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
from src.models.second_models.babel_deadlift.d1_model_v3 import (
    extract_morphology_ratios, MorphologyResidualizer, FEATURE_NAMES_34_D1
)

warnings.filterwarnings("ignore")

MP = dict(
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28,
    LHEEL=29, RHEEL=30, LTOE=31, RTOE=32
)

FEATURE_NAMES_36_D1: List[str] = FEATURE_NAMES_34_D1 + [
    "d1_orthogonal_lumbar_shear",      # [35] d5(발바닥/뒤꿈치 들림) 간섭이 직교 분리된 순수 요추 전단력
    "d1_sagittal_twist_norm"           # [36] 우측 시점 Z축 왜곡이 보정된 체간 시상면 투영 비틀림 각도
]


def extract_d1_features_36d_v3_1(
    kps_33_64f: np.ndarray,
    residualizer: Optional[MorphologyResidualizer] = None
) -> np.ndarray:
    """
    v3.1 핵심 피처 추출:
    - 28D 기본 피처
    - 체형 대비 셋업 잔차 2D + 사지비율 4D (34D)
    - [신규 1] 우측 시점 대칭 보정 비틀림 피처 (sagittal_twist_norm)
    - [신규 2] d5 발바닥 들림 직교 분리 요추 전단력 (orthogonal_lumbar_shear)
    총 36D
    """
    T = kps_33_64f.shape[0]  # 64
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, 11, :]
    r_sh = kps_33_64f[:, 12, :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]
    r_hip = kps_33_64f[:, 24, :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, 25, :]
    r_knee = kps_33_64f[:, 26, :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, 27, :]
    r_ank = kps_33_64f[:, 28, :]
    mid_ank = (l_ank + r_ank) / 2.0

    l_wrist = kps_33_64f[:, 15, :]
    r_wrist = kps_33_64f[:, 16, :]
    mid_wrist = (l_wrist + r_wrist) / 2.0

    # 발뒤꿈치 및 발가락 (d5 직교 분리용)
    l_heel = kps_33_64f[:, 29, :]
    r_heel = kps_33_64f[:, 30, :]
    mid_heel = (l_heel + r_heel) / 2.0

    l_toe = kps_33_64f[:, 31, :]
    r_toe = kps_33_64f[:, 32, :]
    mid_toe = (l_toe + r_toe) / 2.0

    # 1. 체간 로컬 정규 직교 기저계
    v_torso = mid_sh - mid_hip
    torso_len = np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7
    y_hat = v_torso / torso_len

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. 가상 척추 9분절 복원
    s_pts = np.zeros((T, 9, 3), dtype=np.float32)
    s_pts[:, 0] = mid_hip
    for j in range(1, 9):
        alpha = j / 8.0
        s_pts[:, j] = mid_hip + alpha * (mid_sh - mid_hip)

    # 요추 곡률 poly_a
    z_s1 = np.sum((s_pts[:, 0] - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((s_pts[:, 1] - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((s_pts[:, 2] - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((s_pts[:, 3] - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((s_pts[:, 4] - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        y_vals = lumbar_curves[t_idx]
        p = np.polyfit(x_coords, y_vals, deg=2)
        poly_a_series[t_idx] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    x_s_disp = np.abs(np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1))
    f3_lat_dev = float(np.max(x_s_disp))

    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))
    f5_cervical_min = 150.0
    f6_signed_delta = float(np.max(poly_a_series) - np.min(poly_a_series))

    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f7_active_rom = float(np.max(wrist_y_s) - np.min(wrist_y_s))
    f8_phase_speed = 1.0

    hip_knee_h = np.abs(mid_hip[:, 0] - mid_knee[:, 0])
    f9_hk_dist = float(np.mean(hip_knee_h))

    cobb_series = np.abs(poly_a_series) * 100.0
    f10_cobb_mean = float(np.mean(cobb_series))
    f11_cobb_max = float(np.max(cobb_series))
    f12_cobb_range = float(np.max(cobb_series) - np.min(cobb_series))

    # [해결책 1 도입] 우측 시점 가려짐 왜곡 보정 비틀림 (Symmetry Invariant Sagittal Twist)
    # 기존 원시 3D 비틀림은 가려짐에 취약하므로, 어깨 너비 대 골반 너비 비(sh_hip_ratio)로 Z축 왜곡을 감쇠
    sh_vec = l_sh - r_sh
    hip_vec = l_hip - r_hip
    norm_sh = np.linalg.norm(sh_vec, axis=-1) + 1e-7
    norm_hip = np.linalg.norm(hip_vec, axis=-1) + 1e-7
    cos_twist = np.clip(np.sum(sh_vec * hip_vec, axis=-1) / (norm_sh * norm_hip), -1.0, 1.0)
    raw_twist = np.degrees(np.arccos(cos_twist))

    # 어깨 너비 축소율 보정 (우측 뷰에서 어깨가 좁아지면 가중치 감쇠)
    width_ratio = np.clip(norm_sh / norm_hip, 0.5, 2.0)
    calibrated_twist = raw_twist * np.clip(width_ratio / 1.5, 0.5, 1.0)
    calibrated_twist = np.clip(calibrated_twist, 0.0, 45.0)  # 45도 이상은 단안 노이즈로 클리핑

    f13_twist_max = float(np.max(calibrated_twist))
    f14_asym_integ = float(np.mean(calibrated_twist))
    f15_pelvis_tilt = float(np.ptp(mid_hip[:, 1]))
    f16_flex_accel = float(np.max(np.abs(np.gradient(torso_pitch_vel, dt))))
    f17_pitch_standing = float(np.min(torso_pitch))
    f18_pitch_deepest = float(np.max(torso_pitch))

    f19_view_z = float(np.mean(z_hat[:, 2]))
    f20_conf = 1.0
    f21_energy = 0.5
    f22_symm = 1.0

    # [해결책 3 도입] d5(발바닥/뒤꿈치 들림) 직교 분리 전단력
    # 1) 뒤꿈치 들림 점수 (Y축 아래가 양수이므로 toe_y - heel_y가 양수면 뒤꿈치가 들린 것)
    heel_lift_val = np.maximum(0.0, mid_toe[:, 1] - mid_heel[:, 1])
    heel_lift_norm = float(np.mean(heel_lift_val) / np.mean(torso_len))

    # 원시 전단력
    raw_shear = float(np.mean(np.sin(np.radians(torso_pitch))))
    # 직교 분리: 뒤꿈치 들림으로 인한 수동적 전경각 성분을 직교 감산
    orthogonal_shear = float(np.clip(raw_shear - 0.40 * heel_lift_norm, 0.0, 1.0))
    f23_shear = orthogonal_shear

    floor_break_idx = int(np.clip(np.argmax(wrist_y_s), 2, T - 12))
    setup_start = max(0, floor_break_idx - 4)
    f24_setup_poly_a = float(np.mean(poly_a_series[setup_start:floor_break_idx + 1]))

    fb_end = min(T - 1, floor_break_idx + 8)
    f25_fb_cobb_delta = float(cobb_series[fb_end] - cobb_series[floor_break_idx])

    knee_y_level = np.mean(mid_knee[:, 1])
    knee_pass_idx = floor_break_idx + 4
    for k_t in range(floor_break_idx, min(T, floor_break_idx + 24)):
        if wrist_y_s[k_t] <= knee_y_level:
            knee_pass_idx = k_t
            break
    f26_knee_pass_cobb = float(cobb_series[knee_pass_idx])

    finish_start = min(T - 8, floor_break_idx + 16)
    f27_lockout_hyperext = float(np.max(np.maximum(0.0, 180.0 - torso_pitch[finish_start:])))

    knee_ext_vel = np.abs(np.gradient(np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        ((np.linalg.norm(mid_hip - mid_knee, axis=-1) + 1e-7) * (np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7)),
        -1.0, 1.0
    ))), dt))
    f28_sync_ratio = float(np.mean(torso_pitch_vel) / (np.mean(knee_ext_vel) + 1e-6))

    f28_base = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min, f6_signed_delta,
        f7_active_rom, f8_phase_speed, f9_hk_dist, f10_cobb_mean, f11_cobb_max, f12_cobb_range,
        f13_twist_max, f14_asym_integ, f15_pelvis_tilt, f16_flex_accel, f17_pitch_standing, f18_pitch_deepest,
        f19_view_z, f20_conf, f21_energy, f22_symm, f23_shear,
        f24_setup_poly_a, f25_fb_cobb_delta, f26_knee_pass_cobb, f27_lockout_hyperext, f28_sync_ratio
    ], dtype=np.float32)

    # 사지 비율
    ratios = extract_morphology_ratios(kps_33_64f)
    if residualizer is not None and residualizer.is_fitted:
        res_pitch, res_poly = residualizer.transform(ratios, f18_pitch_deepest, f24_setup_poly_a)
    else:
        res_pitch, res_poly = 0.0, 0.0

    # 34D 구성
    f34 = np.concatenate([
        f28_base,
        np.array([res_pitch, res_poly], dtype=np.float32),
        ratios[:4]
    ])

    # [신규 피처 2종 추가 -> 36D]
    # 35: 직교 분리 요추 전단력
    # 36: 시상면 투영 비틀림 평균치
    new_2_feats = np.array([orthogonal_shear, float(np.mean(calibrated_twist))], dtype=np.float32)
    f36 = np.concatenate([f34, new_2_feats])
    return f36


class BabelDeadliftD1ModelV3_1:
    """
    바벨 데드리프트 d1. 척추의 중립 위반 v3.1 프로덕션 모델
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
        pkl_path = os.path.join(self.weights_dir, "d1_v3_1_gbdt_models.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
                self.clf_lgb = d["clf_lgb"]
                self.clf_cat = d["clf_cat"]
                self.residualizer = d.get("residualizer", None)
                self.decision_threshold = d.get("decision_threshold", 0.35)
                self.tau_normal = d.get("tau_normal", 0.28)
                self.tau_defect = d.get("tau_defect", 0.37)

            tcn_files = sorted(glob.glob(os.path.join(self.weights_dir, "d1_v3_1_tcn_fold*.pth")))
            self.tcn_models = []
            for tf in tcn_files:
                m = SquatC1DilatedTCN(in_channels=24)
                m.load_state_dict(torch.load(tf, map_location="cpu"))
                m.eval()
                self.tcn_models.append(m)

            self.is_fitted = (self.clf_lgb is not None and len(self.tcn_models) > 0)

    def extract_features_36d(self, kps_33_64f: np.ndarray) -> np.ndarray:
        return extract_d1_features_36d_v3_1(kps_33_64f, self.residualizer)

    def predict_proba_from_seq(self, kps_33_64f: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError("D1 v3.1 모델 가중치가 로드되지 않았습니다.")

        f36 = self.extract_features_36d(kps_33_64f)[None, :]
        p_lgb = float(self.clf_lgb.predict_proba(f36)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(f36)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        t24 = extract_temporal_24d(kps_33_64f, None)[None, ...]
        x_t = torch.tensor(t24.transpose(0, 2, 1), dtype=torch.float32)

        p_tcns = []
        with torch.no_grad():
            for m in self.tcn_models:
                m.eval()
                logits = m(x_t).squeeze(-1)
                p_tcns.append(float(torch.sigmoid(logits).item()))
        p_tcn = float(np.mean(p_tcns))

        return float(self.w_gbdt * p_gbdt + self.w_tcn * p_tcn)

    def predict_selective(
        self,
        kps_33_64f: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        tau_norm = tau_normal if tau_normal is not None else self.tau_normal
        tau_def = tau_defect if tau_defect is not None else self.tau_defect

        f36 = self.extract_features_36d(kps_33_64f)
        active_rom = float(f36[6])

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
