# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c4. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v4.0 (c4_model_v4.py)
Status: 순수 MediaPipe 33관절 64F 표준 인터페이스 & 22D 정예 지면 접지 피처 & 3-State 선택적 판정 공식 프로덕션 모델

[아키텍처 및 공학적 명세]:
1. 레거시 텐서 슬라이싱 100% 영구 제거:
   - 과거 Format B 92차원 시계열 텐서(seq[:, 58, 89 등]) 및 AI-Hub 3D JSON 종속성 완전 소거
   - 순수 MediaPipe 33관절 64프레임 (64, 33, 3) 월드 좌표 단일 표준 인터페이스 확립
2. 22D 정예 생체역학 지면 접지 피처셋:
   - 발뒤꿈치-발끝 상대 수직 변위 (Relative Y Delta Nadir/Drop/Max)
   - 발바닥 3D 피치각 (Foot Pitch Nadir/Drop/Max)
   - 발바닥 3D 롤각 (Foot Roll Inversion/Max)
   - 정강이 경사 대비 발목 가동성 커플링 비율 (Shank-Ankle Coupling Ratio)
   - 상승기 지면 흔들림 및 가속도 저크 (Ascent Wobble & Jerk)
   - 발앞꿈치 최대 들림량 (Toe Lift Max)
   - 좌우 비대칭도 (Bilateral Pitch & Delta Asymmetry)
   - 발목 관절 최소각 (Ankle Min Angle)
3. 3-State 선택적 판정 (Selective Classification with Rejection Option):
   - 정상: P(defect) <= tau_normal (0.40)
   - 결함: P(defect) >= tau_defect (0.60)
   - 판정 보류: 0.40 < P(defect) < 0.60 (UNDETERMINED_AMBIGUOUS)
4. 소프트 보팅 앙상블:
   - CatBoost(50%) + LightGBM(30%) + Random Forest(20%)
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
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# MediaPipe 33 랜드마크 인덱스
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31 # Left Toe
MP_RIGHT_FOOT_INDEX = 32 # Right Toe

# 22D 정예 피처 명칭
FEATURE_NAMES_22_C4 = [
    'c4_foot_pitch_max_deg',              # [0] 전 구간 최대 발바닥 경사각 (deg)
    'c4_toe_lift_max_norm',               # [1] 스케일 정규화 앞꿈치 최대 들림량
    'c4_foot_pitch_nadir_deg',            # [2] 최하사점 발바닥 경사각 (deg)
    'c4_delta_wobble_active',             # [3] 상승기 뒤꿈치-앞꿈치 수직차 흔들림 (std)
    'c4_roll_max_active',                 # [4] 능동 구간 최대 발바닥 롤각 (deg)
    'c4_delta_max_active',                # [5] 능동 구간 최대 수직 높이차
    'c4_heel_toe_ydelta_nadir_cm',        # [6] 최하사점 뒤꿈치-앞꿈치 수직차 (cm)
    'c4_coupling_ratio_shank_pitch',      # [7] 정강이 경사 대비 발목 가동성 커플링비
    'c4_bilateral_foot_pitch_asym_deg',   # [8] 좌우 발바닥 경사각 비대칭도 (deg)
    'c4_pitch_max_active_deg',            # [9] 능동 구간 최대 피치각 (deg)
    'c4_shank_ankle_coupling_ratio_legacy',# [10] 정강이-발목 시계열 상관계수
    'c4_foot_pitch_drop_deg',             # [11] 기립 대비 최하사점 경사각 변화폭 (deg)
    'c4_foot_roll_inversion_deg',         # [12] 최하사점 발목 롤 / 안쪽날 들림 (deg)
    'c4_asym_pitch_active',               # [13] 능동 구간 최대 피치 비대칭도
    'c4_roll_nadir_deg',                  # [14] 최하사점 발바닥 롤각 (deg)
    'c4_delta_jerk_active',               # [15] 상승기 지면 가속도 저크
    'c4_ank_min_angle',                   # [16] 능동 구간 최소 발목 관절각 (deg)
    'c4_ascent_ankle_destabilization',    # [17] 상승기 발목 불안정성 (cm)
    'c4_toe_lift_max_cm',                 # [18] 기립 기준 앞꿈치 최대 들림량 (cm)
    'c4_asym_delta_active',               # [19] 능동 구간 좌우 높이차 비대칭도
    'c4_norm_delta_nadir',                # [20] 발 길이 정규화 최하사점 수직차
    'c4_pitch_drop_active_deg'            # [21] 능동 구간 피치 드롭각 (deg)
]

# 29D 전체 피처 풀에서 상위 22개 정예 인덱스 매핑
TOP_22_INDICES_FROM_29D = [2, 17, 0, 23, 19, 11, 3, 22, 4, 14, 7, 1, 6, 15, 18, 24, 20, 8, 5, 16, 27, 13]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """3개 3D 관절 좌표 배열로부터 사이각 계산 (도 단위)"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


from src.features.squat_mediapipe33_features import extract_c4_elite_9d_features


def _extract_c4_20d_features(kps: np.ndarray) -> np.ndarray:
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)
    
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    heel_l, heel_r = kps[:, MP_LEFT_HEEL], kps[:, MP_RIGHT_HEEL]
    toe_l, toe_r = kps[:, MP_LEFT_FOOT_INDEX], kps[:, MP_RIGHT_FOOT_INDEX]
    
    # 1. Nadir b
    hip_y = (hip_l[:, 1] + hip_r[:, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=1.2)
    b = int(np.clip(np.argmax(hip_y_s), 4, T - 5))
    
    # Active motion mask
    y_min, y_max = np.min(hip_y_s), np.max(hip_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (hip_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)
        
    stand_slice = slice(0, max(2, T // 8))
    
    # 2. Relative Toe-Heel Y delta
    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0
    delta_s = gaussian_filter1d(delta_mean, sigma=1.0)
    
    delta_stand = float(np.mean(delta_s[stand_slice]))
    delta_nadir = float(delta_s[b])
    delta_drop = float(delta_nadir - delta_stand)
    delta_max_active = float(np.max(delta_s[active_mask]))
    
    # 3. Foot pitch angles
    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)
    
    pitch_stand = float(np.mean(pitch_s[stand_slice]))
    pitch_nadir = float(pitch_s[b])
    pitch_drop = float(pitch_nadir - pitch_stand)
    pitch_max = float(np.max(pitch_s[active_mask]))
    
    # 4. Bilateral Asymmetry
    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))
    
    # 5. Toe lift
    toe_lift_l = np.maximum(0.0, heel_l[:, 1] - toe_l[:, 1])
    toe_lift_r = np.maximum(0.0, heel_r[:, 1] - toe_r[:, 1])
    toe_lift_max = float(max(np.max(toe_lift_l[active_mask]), np.max(toe_lift_r[active_mask])))
    
    # 6. Foot Roll
    roll_l = np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7))
    roll_r = np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7))
    roll_nadir = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    roll_max = float(max(np.max(np.abs(roll_l)[active_mask]), np.max(np.abs(roll_r)[active_mask])))
    
    # 7. Ankle Joint Angle
    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank_mean = (ang_ank_l + ang_ank_r) / 2.0
    ang_ank_s = gaussian_filter1d(ang_ank_mean, sigma=1.0)
    ank_min = float(np.min(ang_ank_s[active_mask]))
    ank_stand = float(np.mean(ang_ank_s[stand_slice]))
    ank_drop = float(ank_stand - ank_min)
    
    # 8. Shank-Ankle Coupling Ratio
    vec_shank_l = ank_l - knee_l
    vec_shank_r = ank_r - knee_r
    shank_pitch_l = np.degrees(np.arctan2(vec_shank_l[:, 2], vec_shank_l[:, 1] + 1e-7))
    shank_pitch_r = np.degrees(np.arctan2(vec_shank_r[:, 2], vec_shank_r[:, 1] + 1e-7))
    shank_pitch = (shank_pitch_l + shank_pitch_r) / 2.0
    shank_diff = float(np.max(shank_pitch[active_mask]) - np.min(shank_pitch[active_mask]) + 1e-6)
    ank_diff = float(np.max(ang_ank_s[active_mask]) - np.min(ang_ank_s[active_mask]))
    coupling_ratio = float(ank_diff / shank_diff)
    
    # 9. Ascent Wobble & Jerk
    ascent_mask = np.zeros(T, dtype=bool)
    ascent_mask[b:] = active_mask[b:]
    if np.sum(ascent_mask) < 3:
        ascent_mask = active_mask
    delta_wobble = float(np.std(delta_s[ascent_mask]))
    
    v_delta = np.gradient(delta_s, dt)
    a_delta = np.gradient(v_delta, dt)
    j_delta = np.gradient(a_delta, dt)
    delta_jerk = float(np.mean(np.abs(j_delta[ascent_mask])))
    
    # 10. Direct heel height lift
    lh_stand = float(np.mean(heel_l[stand_slice, 1]))
    rh_stand = float(np.mean(heel_r[stand_slice, 1]))
    lh_lift = np.maximum(0.0, lh_stand - heel_l[:, 1])
    rh_lift = np.maximum(0.0, rh_stand - heel_r[:, 1])
    heel_lift_max = float(max(np.max(lh_lift[active_mask]), np.max(rh_lift[active_mask])))
    heel_lift_nadir = float((lh_lift[b] + rh_lift[b]) / 2.0)
    
    # 11. Heel-Toe distance ratio
    foot_len_l = np.linalg.norm(vec_foot_l, axis=-1)
    foot_len_r = np.linalg.norm(vec_foot_r, axis=-1)
    norm_delta_nadir = float(delta_nadir / (np.mean(foot_len_l) + 1e-7))
    norm_delta_drop = float(delta_drop / (np.mean(foot_len_l) + 1e-7))
    
    feats = [
        delta_nadir, delta_drop, delta_max_active,
        pitch_nadir, pitch_drop, pitch_max,
        asym_pitch, asym_delta,
        toe_lift_max,
        roll_nadir, roll_max,
        ank_min, ank_drop,
        coupling_ratio,
        delta_wobble, delta_jerk,
        heel_lift_max, heel_lift_nadir,
        norm_delta_nadir, norm_delta_drop
    ]
    return np.array(feats, dtype=np.float32)


def extract_c4_v4_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    순수 MediaPipe 33관절 64프레임 (64, 33, 3) 좌표로부터 c4 v4.0 22D 정예 지면 접지 피처 추출
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    if kps.ndim != 3 or kps.shape[0] != 64 or kps.shape[1] != 33:
        raise ValueError(f"입력 배열은 (64, 33, 3) 형상이어야 합니다. 현재: {kps.shape}")

    # 1. 9D 검증 피처 추출
    f9 = extract_c4_elite_9d_features(kps)
    # 2. 20D 정밀 생체역학 지면 피처 추출
    f20 = _extract_c4_20d_features(kps)
    # 3. 29D 피처 풀 결합
    full_29d = np.hstack([f9, f20])
    # 4. Top-22 정예 피처 서브셋 슬라이싱 반환
    return full_29d[TOP_22_INDICES_FROM_29D]


class BabelSquatC4ModelV4:
    """
    바벨 스쿼트 'c4. 발바닥 지면 고정' 공식 v4.0 프로덕션 모델
    - 22D 순수 MediaPipe 64F 피처셋
    - 3-State 선택적 판정 (Balanced [0.40, 0.60])
    - CatBoost(50%) + LightGBM(30%) + Random Forest(20%) 앙상블
    """

    DEFECT_CODE = "c4"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 22
    FEATURE_NAMES = FEATURE_NAMES_22_C4

    STATE_NORMAL = 0
    STATE_DEFECT = 1
    STATE_UNDETERMINED = -1

    def __init__(
        self,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60,
        min_rom_threshold: float = 0.0004
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.min_rom_threshold = min_rom_threshold
        self.feature_dim = 22
        self.feature_names = FEATURE_NAMES_22_C4

        self.model_catboost = None
        self.model_lightgbm = None
        self.model_rf = None
        self.is_fitted = False

        self._init_default_models()

    def _init_default_models(self):
        """기본 모델 파라미터 초기화"""
        if CatBoostClassifier is not None:
            self.model_catboost = CatBoostClassifier(
                iterations=350, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                random_seed=42, thread_count=-1, verbose=False, allow_writing_files=False
            )
        if lgb is not None:
            self.model_lightgbm = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03, num_leaves=31,
                colsample_bytree=0.85, subsample=0.85, random_state=42, verbose=-1, n_jobs=-1
            )
        self.model_rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """22D 피처셋으로 모델 앙상블 학습"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if self.model_catboost is not None:
            self.model_catboost.fit(X, y)
        if self.model_lightgbm is not None:
            self.model_lightgbm.fit(X, y)
        if self.model_rf is not None:
            self.model_rf.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """결함 확률 예측 (N, 2) 반환"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load_weights()를 먼저 호출하세요.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_cb = self.model_catboost.predict_proba(X)[:, 1] if self.model_catboost else 0.5
        p_lgb = self.model_lightgbm.predict_proba(X)[:, 1] if self.model_lightgbm else 0.5
        p_rf = self.model_rf.predict_proba(X)[:, 1] if self.model_rf else 0.5

        p1 = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf
        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """강제 이진 예측 (0.50 기준)"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.50).astype(int)

    def predict_selective(
        self,
        kps_33_64f: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스 (64, 33, 3)에 대한 3-State 선택적 판정 수행
        반환:
            {
                'defect_code': 'c4',
                'state': 'NORMAL' | 'DEFECT' | 'UNDETERMINED_AMBIGUOUS' | 'UNDETERMINED_INSUFFICIENT_ROM',
                'decision_code': 0 | 1 | -1,
                'probability': float,
                'tau_normal': float,
                'tau_defect': float,
                'features': np.ndarray,
                'confidence': float
            }
        """
        tn = tau_normal if tau_normal is not None else self.tau_normal
        td = tau_defect if tau_defect is not None else self.tau_defect

        kps = np.asarray(kps_33_64f, dtype=np.float32)

        # 1. 가동범위(ROM) 안전 가드
        hip_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
        rom = float(np.max(hip_y) - np.min(hip_y))
        if rom < self.min_rom_threshold:
            return {
                'defect_code': self.DEFECT_CODE,
                'state': 'UNDETERMINED_INSUFFICIENT_ROM',
                'decision_code': self.STATE_UNDETERMINED,
                'probability': 0.5,
                'tau_normal': tn,
                'tau_defect': td,
                'features': None,
                'confidence': 0.0,
                'message': f"스쿼트 가동범위 부족 (ROM={rom:.4f} < {self.min_rom_threshold})"
            }

        # 2. 22D 피처 추출 및 확률 예측
        feat_vec = extract_c4_v4_features_from_seq(kps)
        probs = self.predict_proba(feat_vec.reshape(1, -1))[0]
        prob_defect = float(probs[1])

        # 3. 3-State 판정 로직
        if prob_defect <= tn:
            state = 'NORMAL'
            code = self.STATE_NORMAL
            conf = (tn - prob_defect) / max(1e-6, tn)
            msg = "발바닥이 지면에 견고하게 고정되어 안정적인 리프팅을 수행했습니다."
        elif prob_defect >= td:
            state = 'DEFECT'
            code = self.STATE_DEFECT
            conf = (prob_defect - td) / max(1e-6, 1.0 - td)
            msg = "스쿼트 동작 중 발뒤꿈치 또는 앞꿈치가 지면에서 떨어져 접지 안정성이 저하되었습니다."
        else:
            state = 'UNDETERMINED_AMBIGUOUS'
            code = self.STATE_UNDETERMINED
            conf = 1.0 - (abs(prob_defect - 0.5) / 0.1)
            msg = f"발바닥 접지 상태가 모호한 경계 구간(P={prob_defect*100:.1f}%)에 있어 판정을 안전하게 보류합니다."

        return {
            'defect_code': self.DEFECT_CODE,
            'state': state,
            'decision_code': code,
            'probability': round(prob_defect, 4),
            'tau_normal': tn,
            'tau_defect': td,
            'features': feat_vec,
            'confidence': round(float(np.clip(conf, 0.0, 1.0)), 4),
            'message': msg
        }

    def save_weights(self, file_path: str):
        """모델 가중치 저장"""
        weights = {
            'catboost': self.model_catboost,
            'lightgbm': self.model_lightgbm,
            'rf': self.model_rf,
            'feature_names': self.feature_names,
            'feature_dim': self.feature_dim,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'min_rom_threshold': self.min_rom_threshold,
            'is_fitted': self.is_fitted
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"[*] c4 v4.0 가중치 저장 완료: {file_path}")

    def load_weights(self, file_path: str):
        """모델 가중치 로드"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {file_path}")
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        self.model_catboost = weights.get('catboost')
        self.model_lightgbm = weights.get('lightgbm')
        self.model_rf = weights.get('rf')
        self.feature_names = weights.get('feature_names', self.feature_names)
        self.feature_dim = weights.get('feature_dim', self.feature_dim)
        self.tau_normal = weights.get('tau_normal', self.tau_normal)
        self.tau_defect = weights.get('tau_defect', self.tau_defect)
        self.min_rom_threshold = weights.get('min_rom_threshold', self.min_rom_threshold)
        self.is_fitted = weights.get('is_fitted', True)
        return self
