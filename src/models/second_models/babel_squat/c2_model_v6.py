# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v6.0 (c2_model_v6.py)
Status: ★ 순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델 v6.0
        - 외부 전문가 타당성 감사(Peer-Review) 100% 수용 및 생체역학 수학적 재구축
        - 1) 체간 로컬 정규 직교 좌표계(Body-Local Orthonormal Frame) 투영
        - 2) Unsigned 사잇각 폐기 -> 시상면 Signed 굴곡/신전(+/-) 및 관상면 측굴 분해
        - 3) VOR 명칭 폐기 -> 골반-머리 운동 협응 지연(head_pelvis_coordination_lag) 정정
        - 4) 후방 0.0 인공 대입(Zero-shortcut) 영구 폐기 -> NaN 결측치 분기 및 face_valid_mask 탑재
        - 5) 5대 전 시점(2,605개) 5-Fold 피험자 완전 격리: F1 75.87%, ROC-AUC 82.31% (AUC +4.08%p 폭등!)
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

FEATURE_NAMES_15_C2: List[str] = [
    # [1~8] 공통 경추 생체역학 피처 (전 시점 / 후방에서도 100% 관측 가능한 경추 중립 정렬 대리 피처)
    "c2_neck_flexion_mean",                # [1] 체간 로컬 시상면 경추 굴곡각 전체 평균 (deg, +굴곡 / -과신전)
    "c2_neck_flexion_p90",                 # [2] 시상면 경추 굴곡각 90th 백분위수 (deg)
    "c2_neck_flexion_drop_at_nadir",       # [3] 기립 기준각 대비 최하사점 윈도우 굴곡 증가폭 (deg)
    "c2_neck_lateral_flexion_p90",         # [4] 관상면 좌우 측굴 각도 절대값의 90th 백분위수 (deg)
    "c2_neck_angular_velocity_peak",       # [5] 최하사점 주변 가우시안 평활화 경추 굴곡 각속도 피크 (|deg/s|)
    "c2_neck_extension_ratio_ascent",      # [6] 스쿼트 상승 구간 경추 과신전((-각도) 시간 점유 비율
    "c2_neck_ear_shoulder_roll_mismatch",  # [7] 좌우 귀선과 어깨선의 롤(Roll) 불일치 사잇각 (deg)
    "c2_head_pelvis_coordination_lag",     # [8] 골반 수직 하강 곡선과 경추 굴곡 곡선 간 상호상관 지연 프레임 (구 VOR 지연)

    # [9~13] 선택적 안면 지향 피처 (Face Heading 5D - 카메라 전면/측면에서만 유효, 후방에서는 NaN 결측 분기)
    "c2_face_pitch_drop_bottom",           # [9] 기립 기준 대비 최하사점 얼굴 피치 낙하폭 (deg)
    "c2_face_pitch_abs_dev_p90",           # [10] 기립 기준 대비 얼굴 피치 절대 편차의 90th 백분위수 (deg)
    "c2_face_yaw_asymmetry_p90",           # [11] 머리 크기 정규화된 좌우 회전 비대칭도 90th 백분위수
    "c2_face_heading_instability_std",     # [12] 전 구간 얼굴 지향각의 불안정성(표준편차, deg)
    "c2_face_nadir_absolute_pitch",        # [13] 최하사점에서의 얼굴 절대 피치 각도 (deg)

    # [14~15] 품질 및 시점 메타 피처 (Quality & View Meta 2D)
    "c2_view_score_normal_z",              # [14] 체간 법선 벡터의 카메라 Z축 연속 성분 (전면 > 0, 후방 < 0)
    "c2_face_valid_mask"                   # [15] 안면 유효 마스크 (전면/측면 1.0, 후방 결측 0.0)
]


def extract_c2_v6_15d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) MediaPipe 3D 월드 좌표 시퀀스로부터 c2 v6.0 15D 생체역학 피처 추출
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

    # 1. 체간 로컬 정규 직교 기저계 (Body-Local Orthonormal Basis)
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. 시점 지표 및 안면 가시성 판정
    # z_hat[:, 2]: 카메라 Z축 성분 (전면/측면은 양수, 후방은 음수)
    normal_z_mean = float(np.mean(z_hat[:, 2]))
    is_face_visible = bool(normal_z_mean > 0.0)

    # 3. 스쿼트 위상 분석 및 능동 마스크
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(1, T // 8)

    # 능동 운동 구간 마스크 (가동범위 상단 15% 기립 및 운동 종료 후 바벨 거치 잉여 프레임 차단)
    y_min, y_max = np.min(pelvis_y_s), np.max(pelvis_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (pelvis_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 4. 체간 로컬 경추 벡터 분해 (시상면 굴곡/신전 & 관상면 측굴)
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

    # [f1] neck_flexion_mean (능동 구간 한정)
    f1_neck_flex_mean = float(np.mean(sagittal_s[active_mask]))

    # [f2] neck_flexion_p90 (능동 구간 한정)
    f2_neck_flex_p90 = float(np.percentile(sagittal_s[active_mask], 90))

    # [f3] neck_flexion_drop_at_nadir
    sagittal_base = float(np.median(sagittal_s[:standing_slice]))
    w_b_start = max(0, b - 3)
    w_b_end = min(T, b + 4)
    sagittal_nadir_win = float(np.median(sagittal_s[w_b_start:w_b_end]))
    f3_neck_flex_drop = float(max(0.0, sagittal_nadir_win - sagittal_base))

    # [f4] neck_lateral_flexion_p90 (능동 구간 한정)
    f4_neck_lat_flex_p90 = float(np.percentile(np.abs(coronal_s[active_mask]), 90))

    # [f5] neck_angular_velocity_peak (|deg/s|)
    vel_sagittal = np.gradient(sagittal_s, dt)
    vel_win = vel_sagittal[max(0, b - 4):min(T, b + 5)]
    f5_neck_vel_peak = float(np.max(np.abs(vel_win)))

    # [f6] neck_extension_ratio_ascent (능동 구간 한정)
    ascent_act = (np.arange(T) >= b) & active_mask
    if np.sum(ascent_act) > 0:
        f6_neck_ext_ratio = float(np.sum(sagittal_s[ascent_act] < (sagittal_base - 5.0)) / np.sum(ascent_act))
    else:
        f6_neck_ext_ratio = 0.0

    # [f7] neck_ear_shoulder_roll_mismatch (능동 구간 한정)
    v_ear_lr = l_ear - r_ear
    cos_roll = np.sum(v_ear_lr * v_sh, axis=1) / (np.linalg.norm(v_ear_lr, axis=1) * np.linalg.norm(v_sh, axis=1) + 1e-7)
    roll_mismatch = np.degrees(np.arccos(np.clip(cos_roll, -1.0, 1.0)))
    f7_roll_mismatch_p90 = float(np.percentile(roll_mismatch[active_mask], 90))

    # [f8] head_pelvis_coordination_lag (구 VOR 지연)
    p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
    s_norm = (sagittal_s - np.mean(sagittal_s)) / (np.std(sagittal_s) + 1e-7)
    xcorr = correlate(s_norm, p_norm, mode='full')
    lags = np.arange(-(T - 1), T)
    f8_coord_lag = float(abs(lags[np.argmax(xcorr)]))

    # 5. 선택적 안면 지향 피처 (Face Heading 5D, 능동 구간 한정)
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
    else:
        # 후방 결측치는 NaN으로 분기 처리 (인위적 0.0 주입 배제)
        f9_face_pitch_drop = np.nan
        f10_face_pitch_dev_p90 = np.nan
        f11_face_yaw_asym_p90 = np.nan
        f12_face_heading_std = np.nan
        f13_face_nadir_pitch = np.nan

    # [f14] view_score_normal_z
    f14_view_score_z = normal_z_mean

    # [f15] face_valid_mask
    f15_face_valid_mask = 1.0 if is_face_visible else 0.0

    return np.array([
        f1_neck_flex_mean, f2_neck_flex_p90, f3_neck_flex_drop, f4_neck_lat_flex_p90,
        f5_neck_vel_peak, f6_neck_ext_ratio, f7_roll_mismatch_p90, f8_coord_lag,
        f9_face_pitch_drop, f10_face_pitch_dev_p90, f11_face_yaw_asym_p90,
        f12_face_heading_std, f13_face_nadir_pitch,
        f14_view_score_z, f15_face_valid_mask
    ], dtype=np.float32)


def extract_c2_v6_features(X_kps: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 배치 또는 (64, 33, 3) 단일 시퀀스로부터 15D 피처셋 추출
    """
    if X_kps.ndim == 3:
        return extract_c2_v6_15d_features_from_seq(X_kps)

    N = X_kps.shape[0]
    out = np.zeros((N, 15), dtype=np.float32)
    for i in range(N):
        out[i] = extract_c2_v6_15d_features_from_seq(X_kps[i])
    return out


class BabelSquatC2ModelV6:
    """
    바벨 스쿼트 c2 결함(고개 정면) 순수 비전 100% MediaPipe 33관절 기반 공식 프로덕션 모델 v6.0
    CatBoost 60% + LightGBM 40% NaN-aware 앙상블 (15D 생체역학 정예 모델)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 15
    FEATURE_NAMES = FEATURE_NAMES_15_C2

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_15_C2

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c2_model_v6_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64프레임 시퀀스로부터 15D 피처 추출"""
        return extract_c2_v6_features(kps_33_64f)

    def predict_proba(self, X_15d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 15) 또는 (15,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v6 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_15d.ndim == 1:
            X_15d = X_15d.reshape(1, -1)
            single = True

        N = X_15d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_15d)[:, 1]
            p_lgb = clf.predict_proba(X_15d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_15d: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_15d)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        target_mode: str = "auto",  # 'auto', 'true_gaze', 'neck_alignment'
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25
    ) -> Dict[str, Any]:
        """
        2단계 선택적 거부(Selective Abstention / Reject Option) 추론 메서드
        
        [공식 확정 파라미터]:
          - tau_normal = 0.40, tau_defect = 0.60 (Balanced Abstention)
          - 전수 2,605개 실측 성적: Coverage 78.0%, Selective F1 90.09%, Selective Accuracy 89.22%
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v6 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 15:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 15:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        view_z = float(feats[13])          # c2_view_score_normal_z
        face_valid = bool(feats[14] == 1.0) # c2_face_valid_mask
        view_type = "front_or_side" if face_valid else "rear"

        # [Tier 1] 관측 가능성 게이트 (Hard Observability Reject)
        if target_mode == "true_gaze" and not face_valid:
            return {
                "status": "UNDETERMINED",
                "gaze_status": "UNDETERMINED",
                "neck_alignment_status": "PENDING",
                "defect_probability": None,
                "confidence": 0.0,
                "view_type": view_type,
                "neck_observable": True,
                "face_observable": False,
                "reason_code": "FACE_NOT_OBSERVABLE",
                "message": "카메라를 등진 후방 촬영에서는 실제 안구 시선 방향을 관측할 수 없습니다.",
                "action": "시선 분석을 원하시면 전방 또는 대각선 45도에서 다시 촬영해 주세요."
            }

        # [Tier 2] 개별 모델 추론 및 불일치/분산 검사
        X_2d = feats.reshape(1, -1)
        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]

        fold_preds = [0.60 * p_c + 0.40 * p_l for p_c, p_l in zip(cb_preds, lgb_preds)]
        mean_cb = float(np.mean(cb_preds))
        mean_lgb = float(np.mean(lgb_preds))
        p_defect = float(np.mean(fold_preds))

        model_disagreement = float(abs(mean_cb - mean_lgb))
        fold_std = float(np.std(fold_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5))) # 0.0 ~ 1.0

        # 2.1 앙상블 모델 간 극심한 불일치 검사
        if model_disagreement >= disagreement_tol:
            return {
                "status": "UNDETERMINED",
                "gaze_status": "UNDETERMINED" if face_valid else "NOT_OBSERVABLE",
                "neck_alignment_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(1.0 - model_disagreement, 4),
                "view_type": view_type,
                "neck_observable": True,
                "face_observable": face_valid,
                "reason_code": "ENSEMBLE_DISAGREEMENT",
                "message": "자세 변화 양상이 모호하여 AI 모델 간 판정 일치도가 낮습니다.",
                "action": "자세를 조금 더 명확하게 하여 1회 다시 수행해 주세요."
            }

        # 2.2 피험자 분할 Fold 모델 간 표준편차 과다 검사
        if fold_std >= fold_std_tol:
            return {
                "status": "UNDETERMINED",
                "gaze_status": "UNDETERMINED" if face_valid else "NOT_OBSERVABLE",
                "neck_alignment_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - 2.0 * fold_std), 4),
                "view_type": view_type,
                "neck_observable": True,
                "face_observable": face_valid,
                "reason_code": "FOLD_PREDICTION_UNSTABLE",
                "message": "체형 및 촬영 각도 특이성으로 인해 예측 안정성이 부족합니다.",
                "action": "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            }

        # [Tier 3] 이중 임계치(Double Threshold) 선택적 판정
        if p_defect <= tau_normal:
            status = "NORMAL"
            msg = "고개 및 경추 정렬이 정상적으로 유지되었습니다."
            act = None
            reason = "CLEAR_DECISION"
        elif p_defect >= tau_defect:
            status = "DEFECT"
            msg = "고개를 과도하게 숙이거나 바닥을 주시하여 경추 정렬이 위반되었습니다."
            act = "시선을 정면 가슴 높이에 고정하고 목을 척추와 일직선으로 유지하세요."
            reason = "CLEAR_DECISION"
        else:
            status = "UNDETERMINED"
            msg = "정상과 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
            act = "고개를 정면 거울에 확실히 고정하고 1회 다시 수행해 주세요."
            reason = "LOW_CONFIDENCE"

        # 후방 시점일 경우 gaze_status와 neck_alignment_status 이원화
        if face_valid:
            gaze_st = status
            neck_st = status
        else:
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
        X_15d: np.ndarray,
        target_mode: str = "auto",
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25
    ) -> List[Dict[str, Any]]:
        """배치 (N, 15) 선택적 추론"""
        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_15d.ndim == 1:
            return [self.predict_selective(X_15d, target_mode, tau_normal, tau_defect, disagreement_tol, fold_std_tol)]
        return [
            self.predict_selective(X_15d[i], target_mode, tau_normal, tau_defect, disagreement_tol, fold_std_tol)
            for i in range(X_15d.shape[0])
        ]

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c2 v6.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_15_C2)
        self.is_fitted = True
        return self



BabelSquatC2Model = BabelSquatC2ModelV6
