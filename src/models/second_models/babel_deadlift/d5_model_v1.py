# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body / Barbell Drift)
Model Version: v1.0 (d5_model_v1.py)
Status: 28D 바벨 궤적-신체 밀착 동역학(Barbell-Body Proximity Dynamics) 피처셋 및 
        LightGBM + CatBoost + Random Forest 3-Way Tri-Ensemble 독립 결함 분류 엔진
        - 1) 양 손목(L:15, R:16) 랜드마크 중심 기반 가상 바벨 중심점(Virtual Barbell Center) 정밀 추적
        - 2) 정강이(Shank), 무릎(Knee), 대퇴부(Thigh), 미드풋(BOS) 대비 시공간 이격거리 정량화
        - 3) 어깨-손목 벡터 및 광배근(Latissimus Dorsi) 상완골 신전 모멘트암 결합 지표
        - 4) 리프팅 구간(First Pull, Knee Pass, Second Pull, Lockout) 위상 동역학 추출
        - 5) 공식 5대 시점(A~E, view1~view5) 투영 왜곡 보정 및 시점 불변 정규화
        - 6) 3-State 선택적 판정(Selective Classification: NORMAL, DEFECT, RETAKE)
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

# MediaPipe 33 관절 인덱스
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31  # Left Toe
MP_RIGHT_FOOT_INDEX = 32  # Right Toe

FEATURE_NAMES_28_D5: List[str] = [
    # Group 1: 바벨-하지 수평 이격 거리 동역학 (8D)
    'd5_bar_shank_dist_max',            # [0] 정강이 구간 통과 시 바벨-정강이 최대 수평 이격거리
    'd5_bar_shank_dist_mean',           # [1] 정강이 구간 바벨-정강이 평균 이격거리
    'd5_bar_knee_dist_pass',            # [2] 무릎 높이 통과 순간 바벨-무릎 3D 거리
    'd5_bar_thigh_dist_mean',           # [3] 대퇴부 구간 바벨-허벅지 평균 이격거리
    'd5_bar_thigh_dist_max',            # [4] 대퇴부 구간 바벨-허벅지 최대 이격거리
    'd5_bar_midfoot_init_offset',       # [5] 리프팅 시작점(셋업/이륙) 바벨-미드풋 수평 정렬 오차
    'd5_bar_midfoot_max_offset',        # [6] 전 구간 바벨-미드풋 최대 전후 이격거리
    'd5_bar_leg_contact_deficit_ratio', # [7] 정강이/허벅지 밀착 역치 초과 프레임 비율

    # Group 2: 바벨 수직 궤적 선형성 및 전후 표류 (6D)
    'd5_bar_drift_horizontal_range',    # [8] 바벨 X/Z 전후 표류 총 범위 (max - min)
    'd5_bar_drift_horizontal_std',      # [9] 바벨 전후 표류 표준편차 (흔들림)
    'd5_bar_path_curvature',            # [10] 바벨 총 3D 궤적 길이 / 수직 변위 비율 (선형성)
    'd5_bar_forward_overshoot',         # [11] 시작 셋업선 대비 전방 최대 오버슈트량
    'd5_bar_lateral_wobble_std',        # [12] 바벨 좌우(X축) 흔들림 표준편차
    'd5_bar_ascent_velocity_smoothness',# [13] 바벨 수직 상승 속도 저크/부드러움

    # Group 3: 상체 모멘트 암 & 광배 결합 (5D)
    'd5_shoulder_wrist_angle_max_deg',  # [14] 어깨-손목 벡터와 수직선의 최대 전방 경사각 (deg)
    'd5_shoulder_wrist_angle_mean_deg', # [15] 리프팅 구간 어깨-손목 평균 전방 각도 (deg)
    'd5_shoulder_bar_moment_arm_max',   # [16] 어깨 수직선 대비 바벨 최대 모멘트 암 길이
    'd5_torso_arm_coupling_deficit',    # [17] 상체 전경각 대비 팔 신전각 결손 (광배근 텐션 실패 지표)
    'd5_hip_bar_moment_arm_at_knee',    # [18] 무릎 통과 순간 고관절-바벨 모멘트 암 길이

    # Group 4: 구간별 위상 동역학 (5D)
    'd5_p1_shank_clearance_rate',       # [19] Phase 1 (바닥~무릎) 동안 바벨 전방 이탈률
    'd5_p2_thigh_drag_contact_rate',    # [20] Phase 2 (무릎~락아웃) 동안 허벅지 밀착 유지율
    'd5_eccentric_bar_drift_max',       # [21] 편심성 하강 구간 바벨 전방 최대 이탈량
    'd5_peak_drift_timing_norm',        # [22] 최대 전방 이탈 발생 프레임의 정규화 시간 (0.0~1.0)
    'd5_knee_pass_timing_norm',         # [23] 무릎 통과 발생 프레임의 정규화 시간 (0.0~1.0)

    # Group 5: 멀티뷰 투영 & 왜곡 보정 (4D)
    'd5_sagittal_proj_drift',           # [24] 시점별 회전 변환 보정 시상면 유효 바벨 표류량
    'd5_coronal_overlap_deficit',       # [25] 정면 뷰(view3) 정강이-바벨 수직 투영 겹침 결손도
    'd5_view_vis_confidence',           # [26] 손목 및 하지 관절 가시성 신뢰도
    'd5_bilateral_wrist_asym'           # [27] 좌우 손목 바벨 수평 비대칭도 (비틀림)
]


def extract_d5_v1_features_from_seq(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3 또는 4) 차원의 MediaPipe 랜드마크 배열에서 28차원 d5 바벨 밀착 생체역학 피처 추출
    """
    T = kps_arr.shape[0]
    pts = kps_arr[:, :, :3].astype(np.float32)

    # 관절별 좌표 추출
    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]
    heel_l = pts[:, MP_LEFT_HEEL]
    heel_r = pts[:, MP_RIGHT_HEEL]
    toe_l = pts[:, MP_LEFT_FOOT_INDEX]
    toe_r = pts[:, MP_RIGHT_FOOT_INDEX]

    # 중심점 계산
    shoulder = (shoulder_l + shoulder_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0  # 가상 바벨 중심점
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0
    midfoot = (heel_l + heel_r + toe_l + toe_r) / 4.0

    # 1. 체형 기준 길이 정규화 (Height / Leg Scale Normalization)
    thigh_len = np.mean(np.linalg.norm(hip - knee, axis=-1))
    shank_len = np.mean(np.linalg.norm(knee - ankle, axis=-1))
    torso_len = np.mean(np.linalg.norm(shoulder - hip, axis=-1))
    ref_len = float(max(thigh_len + shank_len, 0.2))

    # 2. 바벨 수직 궤적 및 주요 위상(Phases) 추출
    # Y 좌표: 클수록 바닥(최하사점 Nadir), 작을수록 상단(최상사점 Apex)
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))
    apex_idx = int(np.argmin(bar_y_smooth))

    # 리프팅(Concentric) 및 하강(Eccentric) 구간 분할
    if nadir_idx < apex_idx:
        concentric_slice = slice(nadir_idx, apex_idx + 1)
        eccentric_slice = slice(apex_idx, T)
    else:
        # 서서 시작하여 내려갔다 올라오는 경우
        concentric_slice = slice(nadir_idx, T)
        eccentric_slice = slice(0, nadir_idx + 1)

    # 무릎 높이 통과 시점 (Knee Pass Event)
    knee_y_smooth = gaussian_filter1d(knee[:, 1], sigma=2.0)
    bar_knee_y_diff = np.abs(bar_y_smooth - knee_y_smooth)
    knee_pass_idx = int(np.argmin(bar_knee_y_diff))

    # 3. 바벨과 하지 간의 수평/3D 이격 거리 계산
    # 수평 거리 (X축 및 Z축 복합 거리)
    horiz_bar_knee_dist = np.sqrt((bar[:, 0] - knee[:, 0])**2 + (bar[:, 2] - knee[:, 2])**2) / ref_len
    horiz_bar_midfoot_dist = np.sqrt((bar[:, 0] - midfoot[:, 0])**2 + (bar[:, 2] - midfoot[:, 2])**2) / ref_len
    
    # 정강이 선분(Ankle to Knee)과의 최근접 수평 거리
    shank_mid = (ankle + knee) / 2.0
    horiz_bar_shank_dist = np.sqrt((bar[:, 0] - shank_mid[:, 0])**2 + (bar[:, 2] - shank_mid[:, 2])**2) / ref_len
    
    # 대퇴 선분(Knee to Hip)과의 최근접 수평 거리
    thigh_mid = (knee + hip) / 2.0
    horiz_bar_thigh_dist = np.sqrt((bar[:, 0] - thigh_mid[:, 0])**2 + (bar[:, 2] - thigh_mid[:, 2])**2) / ref_len

    # --- Group 1: 바벨-하지 수평 이격 거리 동역학 (8D) ---
    # Phase 1: 바벨이 무릎보다 아래에 있는 구간
    shank_zone_mask = (bar_y_smooth >= knee_y_smooth - 0.02)
    if np.any(shank_zone_mask):
        f0_bar_shank_dist_max = float(np.max(horiz_bar_shank_dist[shank_zone_mask]))
        f1_bar_shank_dist_mean = float(np.mean(horiz_bar_shank_dist[shank_zone_mask]))
    else:
        f0_bar_shank_dist_max = float(np.max(horiz_bar_shank_dist))
        f1_bar_shank_dist_mean = float(np.mean(horiz_bar_shank_dist))

    f2_bar_knee_dist_pass = float(horiz_bar_knee_dist[knee_pass_idx])

    # Phase 2: 바벨이 무릎보다 위에 있는 대퇴 구간
    thigh_zone_mask = (bar_y_smooth < knee_y_smooth + 0.02)
    if np.any(thigh_zone_mask):
        f3_bar_thigh_dist_mean = float(np.mean(horiz_bar_thigh_dist[thigh_zone_mask]))
        f4_bar_thigh_dist_max = float(np.max(horiz_bar_thigh_dist[thigh_zone_mask]))
    else:
        f3_bar_thigh_dist_mean = float(np.mean(horiz_bar_thigh_dist))
        f4_bar_thigh_dist_max = float(np.max(horiz_bar_thigh_dist))

    # 초기 셋업(이륙 직전) 바벨-미드풋 정렬 오차
    init_slice = slice(max(0, nadir_idx - 3), min(T, nadir_idx + 4))
    f5_bar_midfoot_init_offset = float(np.mean(horiz_bar_midfoot_dist[init_slice]))
    f6_bar_midfoot_max_offset = float(np.max(horiz_bar_midfoot_dist))

    # 밀착 역치(threshold = 0.25 * ref_len) 초과 프레임 비율
    contact_threshold = 0.25
    f7_bar_leg_contact_deficit_ratio = float(np.mean(horiz_bar_shank_dist > contact_threshold))

    # --- Group 2: 바벨 수직 궤적 선형성 및 전후 표류 (6D) ---
    bar_x = bar[:, 0]
    bar_z = bar[:, 2]
    drift_horiz = np.sqrt((bar_x - np.mean(bar_x))**2 + (bar_z - np.mean(bar_z))**2) / ref_len

    f8_bar_drift_horizontal_range = float(np.max(drift_horiz) - np.min(drift_horiz))
    f9_bar_drift_horizontal_std = float(np.std(drift_horiz))

    # 3D 실제 궤적 길이 vs 수직 유효 변위 비율 (곡률 Curvature)
    bar_disp_diffs = np.linalg.norm(np.diff(bar, axis=0), axis=-1)
    total_path_length = float(np.sum(bar_disp_diffs))
    vertical_displacement = float(np.abs(bar_y_smooth[nadir_idx] - bar_y_smooth[apex_idx]) + 1e-6)
    f10_bar_path_curvature = float(total_path_length / vertical_displacement)

    # 셋업 시작선 대비 전방 최대 오버슈트
    start_midfoot_x = midfoot[nadir_idx, 0]
    start_midfoot_z = midfoot[nadir_idx, 2]
    overshoot_dist = np.sqrt((bar_x - start_midfoot_x)**2 + (bar_z - start_midfoot_z)**2) / ref_len
    f11_bar_forward_overshoot = float(np.max(overshoot_dist))

    # 좌우 흔들림 표준편차
    f12_bar_lateral_wobble_std = float(np.std(bar[:, 0]) / ref_len)

    # 수직 상승 가속도 저크(Jerk) 부드러움
    bar_vel_y = np.gradient(bar_y_smooth)
    bar_acc_y = np.gradient(bar_vel_y)
    bar_jerk_y = np.gradient(bar_acc_y)
    f13_bar_ascent_velocity_smoothness = float(np.std(bar_jerk_y) / ref_len)

    # --- Group 3: 상체 모멘트 암 & 광배 결합 (5D) ---
    # 어깨-손목 벡터가 수직축(-Y)과 이루는 사잇각 (deg)
    arm_vec = bar - shoulder  # (T, 3)
    arm_len = np.linalg.norm(arm_vec, axis=-1) + 1e-6
    # Y축 하향 벡터 (0, 1, 0)
    cos_arm_vert = arm_vec[:, 1] / arm_len
    cos_arm_vert = np.clip(cos_arm_vert, -1.0, 1.0)
    arm_angle_deg = np.arccos(cos_arm_vert) * 180.0 / np.pi

    f14_shoulder_wrist_angle_max_deg = float(np.max(arm_angle_deg))
    f15_shoulder_wrist_angle_mean_deg = float(np.mean(arm_angle_deg[concentric_slice]))

    # 어깨 수직선 대비 바벨 최대 모멘트 암 길이
    shoulder_bar_moment_arm = np.sqrt((bar[:, 0] - shoulder[:, 0])**2 + (bar[:, 2] - shoulder[:, 2])**2) / ref_len
    f16_shoulder_bar_moment_arm_max = float(np.max(shoulder_bar_moment_arm))

    # 상체 전경각(Torso inclination) 대비 팔 신전각 결손 (광배근 텐션 실패)
    torso_vec = shoulder - hip
    torso_len_arr = np.linalg.norm(torso_vec, axis=-1) + 1e-6
    cos_torso_vert = -torso_vec[:, 1] / torso_len_arr
    cos_torso_vert = np.clip(cos_torso_vert, -1.0, 1.0)
    torso_inclination_deg = np.arccos(cos_torso_vert) * 180.0 / np.pi
    # 상체가 숙여질수록 팔은 뒤로(광배근으로) 당겨져야 상쇄됨
    lat_coupling_deficit = arm_angle_deg - (torso_inclination_deg * 0.4)
    f17_torso_arm_coupling_deficit = float(np.max(lat_coupling_deficit))

    # 무릎 통과 순간 고관절-바벨 모멘트 암
    f18_hip_bar_moment_arm_at_knee = float(
        np.sqrt((bar[knee_pass_idx, 0] - hip[knee_pass_idx, 0])**2 + 
                (bar[knee_pass_idx, 2] - hip[knee_pass_idx, 2])**2) / ref_len
    )

    # --- Group 4: 구간별 위상 동역학 (5D) ---
    p1_slice = slice(nadir_idx, knee_pass_idx + 1) if nadir_idx < knee_pass_idx else slice(knee_pass_idx, nadir_idx + 1)
    p2_slice = slice(knee_pass_idx, apex_idx + 1) if knee_pass_idx < apex_idx else slice(apex_idx, knee_pass_idx + 1)

    p1_dists = horiz_bar_shank_dist[p1_slice]
    f19_p1_shank_clearance_rate = float(np.mean(p1_dists > contact_threshold)) if len(p1_dists) > 0 else 0.0

    p2_dists = horiz_bar_thigh_dist[p2_slice]
    f20_p2_thigh_drag_contact_rate = float(np.mean(p2_dists <= contact_threshold)) if len(p2_dists) > 0 else 0.0

    ecc_dists = horiz_bar_knee_dist[eccentric_slice]
    f21_eccentric_bar_drift_max = float(np.max(ecc_dists)) if len(ecc_dists) > 0 else 0.0

    peak_drift_frame = int(np.argmax(horiz_bar_midfoot_dist))
    f22_peak_drift_timing_norm = float(peak_drift_frame / float(T - 1))
    f23_knee_pass_timing_norm = float(knee_pass_idx / float(T - 1))

    # --- Group 5: 멀티뷰 투영 & 왜곡 보정 (4D) ---
    # 시점별 보정 회전각 반영
    view_rotations = {
        'view5': -np.pi / 4.0,  # 시점 A: 우측 후방 45°
        'view2': np.pi / 4.0,   # 시점 B: 우측 전방 45°
        'view3': 0.0,           # 시점 C: 정면 0°
        'view4': -np.pi / 4.0,  # 시점 D: 좌측 전방 45°
        'view1': np.pi / 4.0    # 시점 E: 좌측 후방 45°
    }
    theta = view_rotations.get(view_name, 0.0)
    # 회전된 시상면 전후 변위
    z_rot = (bar_x - midfoot[:, 0]) * np.sin(theta) + (bar_z - midfoot[:, 2]) * np.cos(theta)
    f24_sagittal_proj_drift = float(np.std(z_rot) / ref_len)

    # 정면 뷰 수직 투영 시 바벨과 정강이 좌우 겹침 결손도
    overlap_x_diff = np.abs(bar[:, 0] - shank_mid[:, 0]) / ref_len
    f25_coronal_overlap_deficit = float(np.mean(overlap_x_diff))

    # 가시성 신뢰도
    if kps_arr.shape[-1] >= 4:
        vis_scores = kps_arr[:, [MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_KNEE, MP_RIGHT_KNEE], 3]
        f26_view_vis_confidence = float(np.mean(vis_scores))
    else:
        f26_view_vis_confidence = 1.0

    # 좌우 손목 바벨 수평 비대칭도
    wrist_y_diff = np.abs(wrist_l[:, 1] - wrist_r[:, 1]) / ref_len
    f27_bilateral_wrist_asym = float(np.mean(wrist_y_diff))

    features = np.array([
        f0_bar_shank_dist_max,
        f1_bar_shank_dist_mean,
        f2_bar_knee_dist_pass,
        f3_bar_thigh_dist_mean,
        f4_bar_thigh_dist_max,
        f5_bar_midfoot_init_offset,
        f6_bar_midfoot_max_offset,
        f7_bar_leg_contact_deficit_ratio,
        f8_bar_drift_horizontal_range,
        f9_bar_drift_horizontal_std,
        f10_bar_path_curvature,
        f11_bar_forward_overshoot,
        f12_bar_lateral_wobble_std,
        f13_bar_ascent_velocity_smoothness,
        f14_shoulder_wrist_angle_max_deg,
        f15_shoulder_wrist_angle_mean_deg,
        f16_shoulder_bar_moment_arm_max,
        f17_torso_arm_coupling_deficit,
        f18_hip_bar_moment_arm_at_knee,
        f19_p1_shank_clearance_rate,
        f20_p2_thigh_drag_contact_rate,
        f21_eccentric_bar_drift_max,
        f22_peak_drift_timing_norm,
        f23_knee_pass_timing_norm,
        f24_sagittal_proj_drift,
        f25_coronal_overlap_deficit,
        f26_view_vis_confidence,
        f27_bilateral_wrist_asym
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


class BabelDeadliftD5ModelV1:
    """
    바벨 데드리프트 d5(바벨 궤적과 몸 밀착) 전용 3-Way Tri-Ensemble 분류기
    - LightGBM (50%) + CatBoost (30%) + Random Forest (20%)
    - 28차원 바벨 밀착 생체역학 동역학 피처셋 입력
    - 3-State 선택적 판정(Selective Gate: NORMAL, DEFECT, RETAKE)
    """

    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)):
        self.weights = weights
        self.is_fitted = False

        self.lgb_params = lgb_params or {
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        self.cb_params = cb_params or {
            'iterations': 150,
            'depth': 4,
            'learning_rate': 0.05,
            'random_seed': 42,
            'verbose': 0
        }
        self.rf_params = rf_params or {
            'n_estimators': 150,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }

        self.lgb_model = None
        self.cb_model = None
        self.rf_model = RandomForestClassifier(**self.rf_params)

        if lgb is not None:
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        if CatBoostClassifier is not None:
            self.cb_model = CatBoostClassifier(**self.cb_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """앙상블 모델 학습"""
        if self.lgb_model is not None:
            self.lgb_model.fit(X, y)
        if self.cb_model is not None:
            self.cb_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """결함 확률(P(defect)) 예측"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        preds = []
        w_eff = []

        if self.lgb_model is not None:
            preds.append(self.lgb_model.predict_proba(X)[:, 1])
            w_eff.append(self.weights[0])
        if self.cb_model is not None:
            preds.append(self.cb_model.predict_proba(X)[:, 1])
            w_eff.append(self.weights[1])
        preds.append(self.rf_model.predict_proba(X)[:, 1])
        w_eff.append(self.weights[2])

        w_sum = sum(w_eff)
        norm_weights = [w / w_sum for w in w_eff]

        ensemble_prob = np.zeros(X.shape[0], dtype=np.float32)
        for p, w in zip(preds, norm_weights):
            ensemble_prob += p * w

        return np.column_stack([1.0 - ensemble_prob, ensemble_prob])

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """이진 분류 예측 (1: 결함, 0: 정상)"""
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)

    def predict_3state(self, X: np.ndarray, tau_norm: float = 0.40, tau_def: float = 0.60) -> np.ndarray:
        """
        3-State 선택적 판정:
        - prob < tau_norm: 0 (NORMAL)
        - prob >= tau_def: 1 (DEFECT)
        - tau_norm <= prob < tau_def: -1 (RETAKE / 안전 보류)
        """
        prob = self.predict_proba(X)[:, 1]
        preds = np.full(X.shape[0], -1, dtype=int)
        preds[prob < tau_norm] = 0
        preds[prob >= tau_def] = 1
        return preds

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "BabelDeadliftD5ModelV1":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
