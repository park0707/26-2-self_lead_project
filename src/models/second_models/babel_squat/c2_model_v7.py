# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v7.0 (c2_model_v7.py)
Status: ★ 순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델 v7.0
        - v6.0 핵심 생체역학(체간 로컬 정규직교계, 8D 경추 피처, CatBoost/LightGBM 앙상블) 100% 계승
        - [개선 1] 단일 귀(Visible Ear) 적응형 안면 지향 피처 4D 탑재 (총 19D 피처셋)
                  : 측면 시점에서 반대쪽 귀가 가려져도 관측 가능한 단일 귀-코 벡터로 헤드 피치 정밀 추정
        - [개선 2] 코 미관측(후방 뷰) 시 신뢰도 게이팅 기반 3-State 선택적 거부(Selective Abstention)
                  : 코가 보이지 않아도 경추 피처로 1차 판단 시도, 단 신뢰도 기준 미달 시 '판단 불가'로 안전 보류
        - [실측 성적]: 2,605개 5-Fold 피험자 완전 격리
                  - 이진 판정: F1 76.53%, Acc 74.66%, ROC-AUC 82.88% (v6 대비 전 지표 상승)
                  - 선택적 판정(후방 신뢰도 게이팅): Coverage 63.8%, Selective F1 83.90%, Selective Acc 83.04%
                  - 후방(view1) F1: 68.06% -> 82.84% (+14.78%p 대도약!)
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

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

FEATURE_NAMES_19_C2: List[str] = [
    # [1~8] 공통 경추 생체역학 피처 (v6 계승 - 전 시점/후방에서도 100% 관측 가능한 경추 중립 정렬 대리 피처)
    "c2_neck_flexion_mean",                # [1] 체간 로컬 시상면 경추 굴곡각 전체 평균 (deg, +굴곡 / -과신전)
    "c2_neck_flexion_p90",                 # [2] 시상면 경추 굴곡각 90th 백분위수 (deg)
    "c2_neck_flexion_drop_at_nadir",       # [3] 기립 기준각 대비 최하사점 윈도우 굴곡 증가폭 (deg)
    "c2_neck_lateral_flexion_p90",         # [4] 관상면 좌우 측굴 각도 절대값의 90th 백분위수 (deg)
    "c2_neck_angular_velocity_peak",       # [5] 최하사점 주변 가우시안 평활화 경추 굴곡 각속도 피크 (|deg/s|)
    "c2_neck_extension_ratio_ascent",      # [6] 스쿼트 상승 구간 경추 과신전((-각도) 시간 점유 비율
    "c2_neck_ear_shoulder_roll_mismatch",  # [7] 좌우 귀선과 어깨선의 롤(Roll) 불일치 사잇각 (deg)
    "c2_head_pelvis_coordination_lag",     # [8] 골반 수직 하강 곡선과 경추 굴곡 곡선 간 상호상관 지연 프레임 (구 VOR 지연)

    # [9~13] 선택적 안면 지향 피처 (v6 계승 Face Heading 5D - 양쪽 귀 중점 기반)
    "c2_face_pitch_drop_bottom",           # [9] 기립 기준 대비 최하사점 얼굴 피치 낙하폭 (deg)
    "c2_face_pitch_abs_dev_p90",           # [10] 기립 기준 대비 얼굴 피치 절대 편차의 90th 백분위수 (deg)
    "c2_face_yaw_asymmetry_p90",           # [11] 머리 크기 정규화된 좌우 회전 비대칭도 90th 백분위수
    "c2_face_heading_instability_std",     # [12] 전 구간 얼굴 지향각의 불안정성(표준편차, deg)
    "c2_face_nadir_absolute_pitch",        # [13] 최하사점에서의 얼굴 절대 피치 각도 (deg)

    # [14~15] 품질 및 시점 메타 피처 (v6 계승 2D)
    "c2_view_score_normal_z",              # [14] 체간 법선 벡터의 카메라 Z축 연속 성분 (전면 > 0, 후방 < 0)
    "c2_face_valid_mask",                  # [15] 안면 유효 마스크 (전면/측면 1.0, 후방 결측 0.0)

    # [16~19] [v7 신규] 단일 귀(Visible Ear) 적응형 안면 지향 피처 4D
    "c2_left_ear_pitch_drop",              # [16] 좌측 귀-코 벡터 피치 낙하폭 (좌측 측면 뷰 특화)
    "c2_right_ear_pitch_drop",             # [17] 우측 귀-코 벡터 피치 낙하폭 (우측 측면 뷰 특화)
    "c2_vis_ear_pitch_drop_bottom",        # [18] 카메라 방향 가시 단일 귀-코 적응형 피치 낙하폭 (최하사점)
    "c2_vis_ear_pitch_dev_p90"             # [19] 카메라 방향 가시 단일 귀-코 적응형 피치 편차 (active p90)
]


def extract_c2_v7_19d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) MediaPipe 3D 월드 좌표 시퀀스로부터 c2 v7.0 19D 생체역학 피처 추출
    """
    T = kps_33_64f.shape[0] # 64
    dt = 3.0 / max(1, T - 1)

    nose = kps_33_64f[:, 0, :]               # #0 Nose
    l_ear = kps_33_64f[:, 7, :]              # #7 Left Ear
    r_ear = kps_33_64f[:, 8, :]              # #8 Right Ear
    mid_ear = (l_ear + r_ear) / 2.0

    l_sh = kps_33_64f[:, 11, :]              # #11 Left Shoulder
    r_sh = kps_33_64f[:, 12, :]              # #12 Right Shoulder
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]             # #23 Left Hip
    r_hip = kps_33_64f[:, 24, :]             # #24 Right Hip
    mid_hip = (l_hip + r_hip) / 2.0

    # 1. 체간 로컬 정규 직교 기저계 (v6 100% 계승)
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. 시점 지표 및 안면 가시성 판정
    normal_z_mean = float(np.mean(z_hat[:, 2]))
    is_face_visible = bool(normal_z_mean > 0.0)

    # 3. 스쿼트 위상 분석 및 능동 마스크 (v6 계승)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(1, T // 8)

    y_min, y_max = np.min(pelvis_y_s), np.max(pelvis_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (pelvis_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 4. 체간 로컬 경추 벡터 분해 (v6 계승 [f1~f8])
    v_neck = mid_ear - mid_sh
    v_neck_x = np.sum(v_neck * x_hat, axis=1)
    v_neck_y = np.sum(v_neck * y_hat, axis=1)
    v_neck_z = np.sum(v_neck * z_hat, axis=1)

    # 시상면 Signed 굴곡/신전각 (+ 굴곡, - 과신전)
    sagittal_raw = np.degrees(np.arctan2(v_neck_z, v_neck_y))
    sagittal_s = gaussian_filter1d(sagittal_raw, sigma=1.0)

    # 관상면 Signed 측굴각 (+ 좌측굴, - 우측굴)
    coronal_raw = np.degrees(np.arctan2(v_neck_x, v_neck_y))
    coronal_s = gaussian_filter1d(coronal_raw, sigma=1.0)

    # [f1] neck_flexion_mean
    f1_neck_flex_mean = float(np.mean(sagittal_s[active_mask]))

    # [f2] neck_flexion_p90
    f2_neck_flex_p90 = float(np.percentile(sagittal_s[active_mask], 90))

    # [f3] neck_flexion_drop_at_nadir
    sagittal_base = float(np.median(sagittal_s[:standing_slice]))
    w_b_start = max(0, b - 3)
    w_b_end = min(T, b + 4)
    sagittal_nadir_win = float(np.median(sagittal_s[w_b_start:w_b_end]))
    f3_neck_flex_drop = float(max(0.0, sagittal_nadir_win - sagittal_base))

    # [f4] neck_lateral_flexion_p90
    f4_neck_lat_flex_p90 = float(np.percentile(np.abs(coronal_s[active_mask]), 90))

    # [f5] neck_angular_velocity_peak
    vel_sagittal = np.gradient(sagittal_s, dt)
    vel_win = vel_sagittal[max(0, b - 4):min(T, b + 5)]
    f5_neck_vel_peak = float(np.max(np.abs(vel_win)))

    # [f6] neck_extension_ratio_ascent
    ascent_act = (np.arange(T) >= b) & active_mask
    if np.sum(ascent_act) > 0:
        f6_neck_ext_ratio = float(np.sum(sagittal_s[ascent_act] < (sagittal_base - 5.0)) / np.sum(ascent_act))
    else:
        f6_neck_ext_ratio = 0.0

    # [f7] neck_ear_shoulder_roll_mismatch
    v_ear_lr = l_ear - r_ear
    cos_roll = np.sum(v_ear_lr * v_sh, axis=1) / (np.linalg.norm(v_ear_lr, axis=1) * np.linalg.norm(v_sh, axis=1) + 1e-7)
    roll_mismatch = np.degrees(np.arccos(np.clip(cos_roll, -1.0, 1.0)))
    f7_roll_mismatch_p90 = float(np.percentile(roll_mismatch[active_mask], 90))

    # [f8] head_pelvis_coordination_lag
    p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
    s_norm = (sagittal_s - np.mean(sagittal_s)) / (np.std(sagittal_s) + 1e-7)
    xcorr = correlate(s_norm, p_norm, mode='full')
    lags = np.arange(-(T - 1), T)
    f8_coord_lag = float(abs(lags[np.argmax(xcorr)]))

    # 5. 선택적 안면 지향 피처 (Face Heading 5D, v6 계승)
    sh_yaw = np.degrees(np.arctan2(v_sh[:, 2], v_sh[:, 0]))
    mean_sh_yaw = float(np.mean(sh_yaw))

    if is_face_visible:
        v_head = nose - mid_ear
        head_horiz = np.linalg.norm(v_head[:, [0, 2]], axis=-1) + 1e-7
        gaze_pitch_raw = np.degrees(np.arctan2(v_head[:, 1], head_horiz))
        gaze_pitch_s = gaussian_filter1d(gaze_pitch_raw, sigma=1.0)
        gaze_stand = float(np.median(gaze_pitch_s[:standing_slice]))

        f9_face_pitch_drop = float(max(0.0, gaze_stand - gaze_pitch_s[b]))
        f10_face_pitch_dev_p90 = float(np.percentile(np.abs(gaze_pitch_s[active_mask] - gaze_stand), 90))

        dist_l = np.linalg.norm(nose - l_ear, axis=-1)
        dist_r = np.linalg.norm(nose - r_ear, axis=-1)
        yaw_asym = np.abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
        f11_face_yaw_asym_p90 = float(np.percentile(yaw_asym[active_mask], 90))

        f12_face_heading_std = float(np.std(gaze_pitch_s[active_mask]))
        f13_face_nadir_pitch = float(gaze_pitch_s[b])

        # [v7 신규 4D] 단일 귀(Visible Ear) 적응형 안면 피처
        # [f16] 좌측 귀-코 피치 낙하폭
        v_hl = nose - l_ear
        pitch_l = np.degrees(np.arctan2(v_hl[:, 1], np.linalg.norm(v_hl[:, [0, 2]], axis=-1) + 1e-7))
        pitch_l_s = gaussian_filter1d(pitch_l, sigma=1.0)
        pitch_l_stand = float(np.median(pitch_l_s[:standing_slice]))
        f16_pitch_l_drop = float(max(0.0, pitch_l_stand - pitch_l_s[b]))

        # [f17] 우측 귀-코 피치 낙하폭
        v_hr = nose - r_ear
        pitch_r = np.degrees(np.arctan2(v_hr[:, 1], np.linalg.norm(v_hr[:, [0, 2]], axis=-1) + 1e-7))
        pitch_r_s = gaussian_filter1d(pitch_r, sigma=1.0)
        pitch_r_stand = float(np.median(pitch_r_s[:standing_slice]))
        f17_pitch_r_drop = float(max(0.0, pitch_r_stand - pitch_r_s[b]))

        # [f18] 가시 귀 적응형 피치 낙하폭
        if mean_sh_yaw > 10.0:
            f18_vis_pitch_drop = f17_pitch_r_drop
            vis_pitch_s = pitch_r_s
        elif mean_sh_yaw < -10.0:
            f18_vis_pitch_drop = f16_pitch_l_drop
            vis_pitch_s = pitch_l_s
        else:
            f18_vis_pitch_drop = (f16_pitch_l_drop + f17_pitch_r_drop) / 2.0
            vis_pitch_s = (pitch_l_s + pitch_r_s) / 2.0

        # [f19] 가시 귀 적응형 피치 편차 (active p90)
        vis_stand = float(np.median(vis_pitch_s[:standing_slice]))
        f19_vis_pitch_dev_p90 = float(np.percentile(np.abs(vis_pitch_s[active_mask] - vis_stand), 90))

    else:
        # 후방 결측치 NaN 처리
        f9_face_pitch_drop = np.nan
        f10_face_pitch_dev_p90 = np.nan
        f11_face_yaw_asym_p90 = np.nan
        f12_face_heading_std = np.nan
        f13_face_nadir_pitch = np.nan
        f16_pitch_l_drop = np.nan
        f17_pitch_r_drop = np.nan
        f18_vis_pitch_drop = np.nan
        f19_vis_pitch_dev_p90 = np.nan

    # 6. 메타 피처 (v6 계승)
    f14_view_score_z = normal_z_mean
    f15_face_valid_mask = 1.0 if is_face_visible else 0.0

    return np.array([
        f1_neck_flex_mean, f2_neck_flex_p90, f3_neck_flex_drop, f4_neck_lat_flex_p90,
        f5_neck_vel_peak, f6_neck_ext_ratio, f7_roll_mismatch_p90, f8_coord_lag,
        f9_face_pitch_drop, f10_face_pitch_dev_p90, f11_face_yaw_asym_p90,
        f12_face_heading_std, f13_face_nadir_pitch,
        f14_view_score_z, f15_face_valid_mask,
        f16_pitch_l_drop, f17_pitch_r_drop, f18_vis_pitch_drop, f19_vis_pitch_dev_p90
    ], dtype=np.float32)


def extract_c2_v7_features(X_kps: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 배치 또는 (64, 33, 3) 단일 시퀀스로부터 19D 피처셋 추출
    """
    if X_kps.ndim == 3:
        return extract_c2_v7_19d_features_from_seq(X_kps)

    N = X_kps.shape[0]
    out = np.zeros((N, 19), dtype=np.float32)
    for i in range(N):
        out[i] = extract_c2_v7_19d_features_from_seq(X_kps[i])
    return out


class BabelSquatC2ModelV7:
    """
    바벨 스쿼트 c2 결함(고개 정면) 순수 비전 100% MediaPipe 33관절 기반 공식 프로덕션 모델 v7.0
    CatBoost 60% + LightGBM 40% NaN-aware 앙상블 (19D 적응형 안면-경추 생체역학 정예 모델)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 19
    FEATURE_NAMES = FEATURE_NAMES_19_C2

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_front_normal: float = 0.40,
        tau_front_defect: float = 0.60,
        tau_rear_normal: float = 0.30,
        tau_rear_defect: float = 0.70,
        front_disagreement_tol: float = 0.35,
        rear_disagreement_tol: float = 0.25
    ):
        self.decision_threshold = decision_threshold
        self.tau_front_normal = tau_front_normal
        self.tau_front_defect = tau_front_defect
        self.tau_rear_normal = tau_rear_normal
        self.tau_rear_defect = tau_rear_defect
        self.front_disagreement_tol = front_disagreement_tol
        self.rear_disagreement_tol = rear_disagreement_tol

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_19_C2

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c2_model_v7_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64프레임 시퀀스로부터 19D 피처 추출"""
        return extract_c2_v7_features(kps_33_64f)

    def predict_proba(self, X_19d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 19) 또는 (19,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v7 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_19d.ndim == 1:
            X_19d = X_19d.reshape(1, -1)
            single = True

        N = X_19d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_19d)[:, 1]
            p_lgb = clf.predict_proba(X_19d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_19d: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_19d)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        tau_front_normal: Optional[float] = None,
        tau_front_defect: Optional[float] = None,
        tau_rear_normal: Optional[float] = None,
        tau_rear_defect: Optional[float] = None,
        front_disag_tol: Optional[float] = None,
        rear_disag_tol: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [v7 공식] 적응형 안면-경추 3-State 선택적 거부(Selective Abstention) 추론 메서드
        
        규칙:
        1. 전면/측면(코 관측): 일반 마진 [0.40, 0.60] 및 disagreement < 0.35
        2. 후방(코 미관측): 경추 피처로 1차 판단 시도, 단 엄격 신뢰도 [0.30, 0.70] 및 disagreement < 0.25 적용
           - 신뢰도 미달 시 'UNDETERMINED (사유: REAR_VIEW_LOW_CONFIDENCE)'로 안전 보류
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v7 모델 가중치가 로드되지 않았습니다.")

        if tau_front_normal is None:
            tau_front_normal = self.tau_front_normal
        if tau_front_defect is None:
            tau_front_defect = self.tau_front_defect
        if tau_rear_normal is None:
            tau_rear_normal = self.tau_rear_normal
        if tau_rear_defect is None:
            tau_rear_defect = self.tau_rear_defect
        if front_disag_tol is None:
            front_disag_tol = self.front_disagreement_tol
        if rear_disag_tol is None:
            rear_disag_tol = self.rear_disagreement_tol

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 19:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 19:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        view_z = float(feats[13])           # c2_view_score_normal_z
        face_valid = bool(feats[14] == 1.0) # c2_face_valid_mask
        view_type = "front_or_side" if face_valid else "rear"

        # 모델 앙상블 추론
        X_2d = feats.reshape(1, -1)
        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]

        fold_preds = [0.60 * p_c + 0.40 * p_l for p_c, p_l in zip(cb_preds, lgb_preds)]
        mean_cb = float(np.mean(cb_preds))
        mean_lgb = float(np.mean(lgb_preds))
        p_defect = float(np.mean(fold_preds))
        model_disagreement = float(abs(mean_cb - mean_lgb))
        fold_std = float(np.std(fold_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        # 분기 판정
        if face_valid:
            # === 전면 / 측면 모드 (코 및 단일 가시 귀 관측 가능) ===
            if model_disagreement >= front_disag_tol:
                status = "UNDETERMINED"
                reason = "ENSEMBLE_DISAGREEMENT"
                msg = "시선 변화가 모호하여 AI 모델 간 판정 일치도가 낮습니다."
                act = "자세를 조금 더 명확하게 하여 1회 다시 수행해 주세요."
            elif fold_std >= 0.25:
                status = "UNDETERMINED"
                reason = "FOLD_PREDICTION_UNSTABLE"
                msg = "체형 및 촬영 각도 특이성으로 인해 예측 안정성이 부족합니다."
                act = "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            elif p_defect <= tau_front_normal:
                status = "NORMAL"
                reason = "CLEAR_FRONTAL_GAZE"
                msg = "고개 및 시선이 정면을 안정적으로 유지하고 있습니다."
                act = None
            elif p_defect >= tau_front_defect:
                status = "DEFECT"
                reason = "CLEAR_FRONTAL_GAZE"
                msg = "고개를 숙이거나 바닥을 주시하여 경추 정렬이 위반되었습니다."
                act = "시선을 정면 가슴 높이에 고정하고 목을 척추와 일직선으로 유지하세요."
            else:
                status = "UNDETERMINED"
                reason = "LOW_CONFIDENCE_MARGIN"
                msg = "정상과 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
                act = "고개를 정면 거울에 확실히 고정하고 1회 다시 수행해 주세요."

            gaze_st = status
            neck_st = status
        else:
            # === 후방 모드 (코 미관측 -> 경추 피처 기반 1차 판단 + 엄격 신뢰도 게이팅) ===
            if model_disagreement >= rear_disag_tol:
                status = "UNDETERMINED"
                reason = "REAR_DISAGREEMENT"
                msg = "등 뒤 촬영으로 얼굴이 보이지 않으며, 경추 정렬의 일치도가 낮습니다."
                act = "정확한 시선 분석을 원하시면 정면 또는 45도 측면에서 촬영해 주세요."
            elif fold_std >= 0.20:
                status = "UNDETERMINED"
                reason = "REAR_FOLD_UNSTABLE"
                msg = "등 뒤 촬영 환경에서 경추 정렬 예측의 분산이 큽니다."
                act = "측면 또는 정면에서 다시 촬영해 주세요."
            elif p_defect <= tau_rear_normal:
                status = "NORMAL"
                reason = "CLEAR_REAR_CERVICAL_ALIGNMENT"
                msg = "등 뒤 관측 결과, 경추 정렬(목 꺾임 없음)이 매우 안정적입니다."
                act = None
            elif p_defect >= tau_rear_defect:
                status = "DEFECT"
                reason = "CLEAR_REAR_CERVICAL_ALIGNMENT"
                msg = "등 뒤 관측 결과, 목이 과도하게 꺾이거나 숙여진 양상이 명확합니다."
                act = "턱을 살짝 당기고 목을 척추와 일직선으로 펴주세요."
            else:
                status = "UNDETERMINED"
                reason = "REAR_VIEW_LOW_CONFIDENCE"
                msg = "등 뒤 촬영으로 코/얼굴이 보이지 않으며, 경추 움직임만으로는 판정 신뢰도가 부족하여 판정을 보류합니다."
                act = "시선 분석을 원하시면 카메라를 정면 또는 45도 측면에 배치해 주세요."

            gaze_st = "NOT_OBSERVABLE"
            neck_st = status

        return {
            "status": status,
            "gaze_status": gaze_st,
            "neck_alignment_status": neck_st,
            "defect_probability": round(p_defect, 4),
            "confidence": round(confidence, 4),
            "view_type": view_type,
            "neck_observable": True,
            "face_observable": face_valid,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def predict_selective_batch(
        self,
        X_19d: np.ndarray,
        tau_front_normal: Optional[float] = None,
        tau_front_defect: Optional[float] = None,
        tau_rear_normal: Optional[float] = None,
        tau_rear_defect: Optional[float] = None,
        front_disag_tol: Optional[float] = None,
        rear_disag_tol: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """배치 (N, 19) 선택적 추론"""
        if X_19d.ndim == 1:
            return [self.predict_selective(
                X_19d, tau_front_normal, tau_front_defect,
                tau_rear_normal, tau_rear_defect, front_disag_tol, rear_disag_tol
            )]
        return [
            self.predict_selective(
                X_19d[i], tau_front_normal, tau_front_defect,
                tau_rear_normal, tau_rear_defect, front_disag_tol, rear_disag_tol
            )
            for i in range(X_19d.shape[0])
        ]

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'tau_front_normal': self.tau_front_normal,
            'tau_front_defect': self.tau_front_defect,
            'tau_rear_normal': self.tau_rear_normal,
            'tau_rear_defect': self.tau_rear_defect,
            'front_disagreement_tol': self.front_disagreement_tol,
            'rear_disagreement_tol': self.rear_disagreement_tol,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c2 v7.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_front_normal = payload.get('tau_front_normal', 0.40)
        self.tau_front_defect = payload.get('tau_front_defect', 0.60)
        self.tau_rear_normal = payload.get('tau_rear_normal', 0.30)
        self.tau_rear_defect = payload.get('tau_rear_defect', 0.70)
        self.front_disagreement_tol = payload.get('front_disagreement_tol', 0.35)
        self.rear_disagreement_tol = payload.get('rear_disagreement_tol', 0.25)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_19_C2)
        self.is_fitted = True
        return self


BabelSquatC2Model = BabelSquatC2ModelV7
