# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Model Version: v1.0 (d4_model_v1.py)
Status: 24D 생체역학 무릎-골반 연동 동역학 피처셋 및 Tri-Ensemble 아키텍처 모델
        - 1) 24D 생체역학 피처:
             * Group 1: 슬관절-고관절 신전 동시성 및 각속도 비 (7D)
             * Group 2: 상체 경사각 안정도 & 힙 슈트업 (Hip Shoot Up) 지수 (6D)
             * Group 3: 락아웃(Lockout) 위상차(Phase Lag) 및 가동범위 (5D)
             * Group 4: 5대 시점 불변 강건성 및 대칭성 (6D)
        - 2) LightGBM + CatBoost + Random Forest 3-Way Tri-Ensemble 하이브리드 결합
        - 3) 3-State 선택적 판정(Selective Classification: ACCEPT / DECLINE / RETAKE) 메서드 탑재
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# MediaPipe 33 랜드마크 인덱스
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

FEATURE_NAMES_24_D4: List[str] = [
    # Group 1: 슬관절-고관절 신전 동시성 및 각속도 비 (7D)
    'd4_knee_hip_corr_full',              # [0] 전 구간 무릎-골반 각도 상관계수
    'd4_knee_hip_corr_ascent',            # [1] 상승기 무릎-골반 각도 상관계수
    'd4_vel_ratio_first_half',            # [2] 상승기 전반부(First Pull) 무릎/골반 각속도 비
    'd4_vel_ratio_second_half',           # [3] 상승기 후반부(Second Pull) 무릎/골반 각속도 비
    'd4_vel_ratio_full_ascent',           # [4] 상승기 전 구간 무릎/골반 평균 각속도 비
    'd4_extension_rate_diff_max',         # [5] 무릎 신전율과 골반 신전율의 최대 순간 편차
    'd4_angular_accel_mismatch',          # [6] 무릎 각가속도와 골반 각가속도의 평균 절대 편차

    # Group 2: 상체 경사각 안정도 & 힙 슈트업 (6D)
    'd4_trunk_angle_nadir_deg',           # [7] 최하사점 상체 경사각 (수직 대비, deg)
    'd4_trunk_drop_ascent_max_deg',       # [8] 상승기 상체 전경각 추가 급락(숙여짐) 각도 (deg)
    'd4_hip_shoot_up_speed_norm',         # [9] First Pull 골반 상승 속도 - 어깨 상승 속도 편차
    'd4_torso_horizontal_ratio',          # [10] 상체가 수평에 가깝게 무너진 시간 비율
    'd4_trunk_angle_std_ascent',          # [11] 상승기 상체 경사각의 변동성(std)
    'd4_hip_shoulder_vert_dist_min',      # [12] 상승기 골반-어깨 수직 간격 최소비

    # Group 3: 락아웃 위상차 및 가동범위 (5D)
    'd4_knee_lock_time_norm',             # [13] 무릎 최대 신전 도달 시점 (0~1 정규화)
    'd4_hip_lock_time_norm',              # [14] 골반 최대 신전 도달 시점 (0~1 정규화)
    'd4_lockout_phase_lag_norm',          # [15] 무릎 락아웃과 골반 락아웃 시점의 위상차
    'd4_knee_rom_deg',                    # [16] 무릎 관절 가동 범위 (deg)
    'd4_hip_rom_deg',                     # [17] 고관절 가동 범위 (deg)

    # Group 4: 5대 시점 불변 강건성 및 대칭성 (6D)
    'd4_knee_hip_rom_ratio',              # [18] 무릎 ROM 대비 골반 ROM 비
    'd4_bilateral_knee_asym_deg',         # [19] 좌우 무릎 각도 평균 비대칭도 (deg)
    'd4_bilateral_hip_asym_deg',          # [20] 좌우 골반 각도 평균 비대칭도 (deg)
    'd4_ascent_duration_ratio',           # [21] 전체 동작 중 상승기 구간 길이 비율
    'd4_pelvis_y_rise_rate',              # [22] 골반 수직 상승 총 변위비
    'd4_view_robust_coupling'             # [23] 시점 불변 체간-대퇴 3D 벡터 내적 변화율
]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """세 점 p1, p2(중심), p3 사이의 3D 사잇각(deg) 계산 (T, 3) -> (T,)"""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    denom = norm1 * norm2 + 1e-7
    cos_ang = np.sum(v1 * v2, axis=-1) / denom
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def extract_d4_v1_features_from_seq(kps: np.ndarray) -> np.ndarray:
    """
    데드리프트 64프레임 33관절 3D 포즈 시퀀스로부터 d4 24D 생체역학 피처 추출
    Args:
        kps: np.ndarray (64, 33, 3) 64프레임 MediaPipe 관절 좌표
    Returns:
        feats_24: np.ndarray (24,) float32 피처 벡터
    """
    T = kps.shape[0]
    assert T >= 16, f"Sequence length must be >= 16, got {T}"

    # 1. 주요 관절 위치 추출 (가우시안 평활화 적용)
    sh_l, sh_r = kps[:, MP_LEFT_SHOULDER], kps[:, MP_RIGHT_SHOULDER]
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]

    sh_mid = (sh_l + sh_r) / 2.0
    hip_mid = (hip_l + hip_r) / 2.0
    knee_mid = (knee_l + knee_r) / 2.0
    ank_mid = (ank_l + ank_r) / 2.0

    # 스케일 정규화 (체간 길이 기준)
    trunk_lens = np.linalg.norm(sh_mid - hip_mid, axis=-1)
    ref_len = float(np.median(trunk_lens))
    if ref_len < 1e-4:
        ref_len = 1.0

    # 2. 관절 각도 계산
    # 무릎 각도: Hip - Knee - Ankle
    knee_ang_l = compute_angle_3d(hip_l, knee_l, ank_l)
    knee_ang_r = compute_angle_3d(hip_r, knee_r, ank_r)
    knee_ang = (knee_ang_l + knee_ang_r) / 2.0
    knee_ang_smooth = gaussian_filter1d(knee_ang, sigma=1.5)

    # 골반(고관절) 각도: Shoulder - Hip - Knee
    hip_ang_l = compute_angle_3d(sh_l, hip_l, knee_l)
    hip_ang_r = compute_angle_3d(sh_r, hip_r, knee_r)
    hip_ang = (hip_ang_l + hip_ang_r) / 2.0
    hip_ang_smooth = gaussian_filter1d(hip_ang, sigma=1.5)

    # 상체 전경각 (수직축 대비 상체 기울기, 수직=0도, 수평=90도)
    trunk_vec = sh_mid - hip_mid
    trunk_vec_norm = np.linalg.norm(trunk_vec, axis=-1) + 1e-7
    # Y축은 아래 방향 (MediaPipe 기준 Y 증가가 아래)
    # 기립 시 shoulder.y < hip.y -> trunk_vec.y < 0
    cos_vert = np.clip(-trunk_vec[:, 1] / trunk_vec_norm, -1.0, 1.0)
    trunk_lean_deg = np.degrees(np.arccos(cos_vert))  # 0도(기립) ~ 90도(수평)
    trunk_lean_smooth = gaussian_filter1d(trunk_lean_deg, sigma=1.5)

    # 3. 주요 구간 분할: 최하사점(Nadir) 및 상승기(Ascent)
    # 데드리프트 최하사점은 고관절 각도가 가장 깊이 굽혀진(최소) 시점 또는 골반이 가장 낮은 시점
    nadir_idx = int(np.argmin(hip_ang_smooth))
    nadir_idx = np.clip(nadir_idx, 5, T - 10)

    # 상승기 구간: nadir_idx ~ 종료(또는 고관절/무릎 최대 신전 복귀 시점)
    # 락아웃 시점 탐색: nadir_idx 이후 고관절 각도가 최대가 되는 시점
    ascent_slice = slice(nadir_idx, T)
    post_hip_ang = hip_ang_smooth[ascent_slice]
    post_knee_ang = knee_ang_smooth[ascent_slice]

    lock_hip_rel = int(np.argmax(post_hip_ang))
    lock_knee_rel = int(np.argmax(post_knee_ang))
    lock_hip_idx = nadir_idx + lock_hip_rel
    lock_knee_idx = nadir_idx + lock_knee_rel

    end_ascent_idx = max(lock_hip_idx, lock_knee_idx)
    if end_ascent_idx <= nadir_idx + 3:
        end_ascent_idx = T - 1

    ascent_len = max(end_ascent_idx - nadir_idx, 4)
    mid_ascent_idx = nadir_idx + ascent_len // 2

    # 4. 각속도 및 각가속도 계산
    knee_vel = np.gradient(knee_ang_smooth)
    hip_vel = np.gradient(hip_ang_smooth)
    knee_acc = np.gradient(knee_vel)
    hip_acc = np.gradient(hip_vel)

    # Group 1 피처 산출
    # [0] 전 구간 상관계수
    corr_full = float(np.corrcoef(knee_ang_smooth, hip_ang_smooth)[0, 1])
    if np.isnan(corr_full):
        corr_full = 0.0

    # [1] 상승기 상관계수
    if ascent_len >= 4:
        corr_ascent = float(np.corrcoef(knee_ang_smooth[nadir_idx:end_ascent_idx],
                                       hip_ang_smooth[nadir_idx:end_ascent_idx])[0, 1])
        if np.isnan(corr_ascent):
            corr_ascent = 0.0
    else:
        corr_ascent = corr_full

    # [2] First Pull 구간 각속도 비 (무릎 각속도 / 고관절 각속도)
    first_pull_k_vel = np.mean(knee_vel[nadir_idx:mid_ascent_idx])
    first_pull_h_vel = np.mean(hip_vel[nadir_idx:mid_ascent_idx])
    vel_ratio_first = float(first_pull_k_vel / (abs(first_pull_h_vel) + 1e-4))

    # [3] Second Pull 구간 각속도 비
    second_pull_k_vel = np.mean(knee_vel[mid_ascent_idx:end_ascent_idx])
    second_pull_h_vel = np.mean(hip_vel[mid_ascent_idx:end_ascent_idx])
    vel_ratio_second = float(second_pull_k_vel / (abs(second_pull_h_vel) + 1e-4))

    # [4] 상승기 전 구간 평균 각속도 비
    mean_ascent_k_vel = np.mean(knee_vel[nadir_idx:end_ascent_idx])
    mean_ascent_h_vel = np.mean(hip_vel[nadir_idx:end_ascent_idx])
    vel_ratio_full = float(mean_ascent_k_vel / (abs(mean_ascent_h_vel) + 1e-4))

    # [5] 신전율 편차 최대치
    knee_rom = float(np.max(knee_ang_smooth) - np.min(knee_ang_smooth) + 1e-4)
    hip_rom = float(np.max(hip_ang_smooth) - np.min(hip_ang_smooth) + 1e-4)
    knee_ext_progress = (knee_ang_smooth[nadir_idx:end_ascent_idx] - knee_ang_smooth[nadir_idx]) / knee_rom
    hip_ext_progress = (hip_ang_smooth[nadir_idx:end_ascent_idx] - hip_ang_smooth[nadir_idx]) / hip_rom
    ext_rate_diff_max = float(np.max(np.abs(knee_ext_progress - hip_ext_progress)))

    # [6] 각가속도 편차 평균
    acc_mismatch = float(np.mean(np.abs(knee_acc[nadir_idx:end_ascent_idx] - hip_acc[nadir_idx:end_ascent_idx])))

    # Group 2 피처 산출: 상체 경사각 & 힙 슈트업
    # [7] 최하사점 상체 경사각 (deg)
    trunk_nadir = float(trunk_lean_smooth[nadir_idx])

    # [8] 상승기 상체 전경각 추가 급락(숙여짐) 각도 (deg)
    # 정상: 상승 시작 직후 상체 각도가 줄어들지 않고 서서히 세워짐
    # 힙 슈트업: 바벨 뽑을 때 엉덩이가 먼저 올라가면서 상체 전경각이 일시적으로 더 커짐(더 숙여짐)
    first_pull_trunk = trunk_lean_smooth[nadir_idx:mid_ascent_idx]
    trunk_drop_max = float(np.max(first_pull_trunk) - trunk_nadir)
    if trunk_drop_max < 0.0:
        trunk_drop_max = 0.0

    # [9] First Pull 골반 상승 속도 - 어깨 상승 속도
    # Y 좌표 감소가 상승임
    hip_y_vel = -np.gradient(hip_mid[:, 1]) / ref_len
    sh_y_vel = -np.gradient(sh_mid[:, 1]) / ref_len
    hip_shoot_speed = float(np.mean(hip_y_vel[nadir_idx:mid_ascent_idx] - sh_y_vel[nadir_idx:mid_ascent_idx]))

    # [10] 상체가 수평에 가깝게 무너진 시간 비율 (> 60도 전경)
    torso_horiz_ratio = float(np.mean(trunk_lean_smooth[nadir_idx:end_ascent_idx] > 60.0))

    # [11] 상승기 상체 경사각 변동성
    trunk_std = float(np.std(trunk_lean_smooth[nadir_idx:end_ascent_idx]))

    # [12] 상승기 골반-어깨 수직 간격 최소비
    vert_dist = np.abs(sh_mid[:, 1] - hip_mid[:, 1]) / ref_len
    min_vert_dist = float(np.min(vert_dist[nadir_idx:end_ascent_idx]))

    # Group 3 피처 산출: 락아웃 위상차 및 가동범위
    # [13, 14, 15] 락아웃 시점 및 위상차
    knee_lock_norm = float(lock_knee_idx / float(T))
    hip_lock_norm = float(lock_hip_idx / float(T))
    lockout_lag = float(abs(knee_lock_norm - hip_lock_norm))

    # [16, 17] 관절 가동 범위 (ROM)
    k_rom_deg = knee_rom
    h_rom_deg = hip_rom

    # Group 4 피처 산출: 5대 시점 불변 및 대칭성
    # [18] 무릎 ROM / 골반 ROM 비
    rom_ratio = float(k_rom_deg / (h_rom_deg + 1e-4))

    # [19, 20] 좌우 비대칭도 (deg)
    asym_knee = float(np.mean(np.abs(knee_ang_l - knee_ang_r)))
    asym_hip = float(np.mean(np.abs(hip_ang_l - hip_ang_r)))

    # [21] 전체 대비 상승기 길이 비
    ascent_ratio = float(ascent_len / float(T))

    # [22] 골반 수직 상승 변위비
    pelvis_y_rise = float(abs(hip_mid[nadir_idx, 1] - hip_mid[end_ascent_idx, 1]) / ref_len)

    # [23] 시점 불변 체간-대퇴 3D 벡터 내적 변화율
    # 체간 벡터(hip -> shoulder)와 대퇴 벡터(hip -> knee)의 3D 내적 시계열
    thigh_vec = knee_mid - hip_mid
    thigh_norm = np.linalg.norm(thigh_vec, axis=-1, keepdims=True) + 1e-7
    trunk_norm_3d = trunk_vec / (trunk_vec_norm[:, None] + 1e-7)
    thigh_norm_3d = thigh_vec / thigh_norm
    dot_coupling = np.sum(trunk_norm_3d * thigh_norm_3d, axis=-1)
    view_robust_coupling = float(np.std(dot_coupling[nadir_idx:end_ascent_idx]))

    feats = np.array([
        corr_full, corr_ascent, vel_ratio_first, vel_ratio_second, vel_ratio_full,
        ext_rate_diff_max, acc_mismatch,
        trunk_nadir, trunk_drop_max, hip_shoot_speed, torso_horiz_ratio, trunk_std, min_vert_dist,
        knee_lock_norm, hip_lock_norm, lockout_lag, k_rom_deg, h_rom_deg,
        rom_ratio, asym_knee, asym_hip, ascent_ratio, pelvis_y_rise, view_robust_coupling
    ], dtype=np.float32)

    # NaN / Inf 방어 클리핑
    feats = np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0)
    return feats


class BabelDeadliftD4ModelV1:
    """
    바벨 데드리프트 d4(무릎과 골반 동시 펴짐 위반) 3-Way Tri-Ensemble 모델
    - LightGBM (50%) + CatBoost (30%) + Random Forest (20%)
    - 3-State 선택적 판정(ACCEPT / DECLINE / RETAKE) 탑재
    """
    def __init__(self, tau_normal: float = 0.35, tau_defect: float = 0.65):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect

        self.clf_lgb = None
        self.clf_cat = None
        self.clf_rf = None

        self.weights = {'lgb': 0.50, 'cat': 0.30, 'rf': 0.20}
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Tri-Ensemble 하이브리드 학습"""
        # 1. LightGBM
        if lgb is not None:
            self.clf_lgb = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                verbose=-1
            )
            self.clf_lgb.fit(X, y)

        # 2. CatBoost
        if CatBoostClassifier is not None:
            self.clf_cat = CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=4,
                random_seed=42,
                verbose=False
            )
            self.clf_cat.fit(X, y)

        # 3. Random Forest
        self.clf_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.clf_rf.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """앙상블 결함 확률 P(defect=1) 산출"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        probs = np.zeros(len(X), dtype=np.float32)
        total_w = 0.0

        if self.clf_lgb is not None:
            p_lgb = self.clf_lgb.predict_proba(X)[:, 1]
            probs += self.weights['lgb'] * p_lgb
            total_w += self.weights['lgb']

        if self.clf_cat is not None:
            p_cat = self.clf_cat.predict_proba(X)[:, 1]
            probs += self.weights['cat'] * p_cat
            total_w += self.weights['cat']

        if self.clf_rf is not None:
            p_rf = self.clf_rf.predict_proba(X)[:, 1]
            probs += self.weights['rf'] * p_rf
            total_w += self.weights['rf']

        if total_w > 0:
            probs /= total_w

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """기본 이진 분류 예측 (0: 정상, 1: 결함)"""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(np.int32)

    def predict_3state(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        3-State 선택적 판정
        - ACCEPT_NORMAL: prob <= tau_normal (확실한 정상)
        - ACCEPT_DEFECT: prob >= tau_defect (확실한 결함)
        - RETAKE: tau_normal < prob < tau_defect (판정 보류 / 재촬영 권고)
        """
        probs = self.predict_proba(X)
        results = []
        for p in probs:
            if p <= self.tau_normal:
                state = "ACCEPT_NORMAL"
                decision = 0
            elif p >= self.tau_defect:
                state = "ACCEPT_DEFECT"
                decision = 1
            else:
                state = "RETAKE"
                decision = -1

            results.append({
                "probability": float(p),
                "state": state,
                "decision": decision,
                "confidence": float(abs(p - 0.5) * 2.0)
            })
        return results

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BabelDeadliftD4ModelV1":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
