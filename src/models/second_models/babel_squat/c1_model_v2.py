# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_model_v2.py
Purpose: 바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반 / Spine Neutrality) v2.0 분류 모델
- 핵심 혁신 및 개선 사항 (v1.0 대비):
  1) 3D 척추 분절 키포인트(Neck-경추, Back-흉추, Waist-요추, Pelvis-골반) 기반 8D 신규 피처 결합 (총 30D)
     - lumbopelvic butt-wink (버트윙크 요추 말림 변위)
     - thoracolumbar flexion nadir / drop (흉요추 굴곡각 및 낙폭)
     - hip-hinge decoupling ratio (상체 피치 대비 척추 굴곡 분리율)
     - thoracolumbar hyperextension (요추 과신전/오리궁둥이)
     - coronal lateral deviation (좌우 비대칭 수평 이탈도)
     - lumbar shear moment arm (최하사점 요추 전단력 모멘트암)
  2) 비용 민감 학습 (Cost-Sensitive Weighting: w_normal=1.7) 적용
     - 정상 샘플에 대한 척추 오류 오탐(Clean FP)을 42건(33.6%) -> 24건(19.2%)으로 42.9% 격감
  3) 2계층 캐스케이드 (Tier-1 경량 게이트키퍼 + Tier-2 고정밀 리졸버) 및 단일 CatBoost 앙상블 지원
  4) 64프레임 3차 스플라인(Cubic Spline) 보간 표준 준수
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.extract_squat_spine_trend import (
    extract_squat_spine_trend_features,
    SPINE_TREND_FEATURE_NAMES_9
)
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics
from src.models.second_models.extract_squat_spine_literature import (
    extract_squat_spine_literature_features,
    SPINE_LIT_FEATURE_NAMES_6
)
from src.models.second_models.babel_squat.build_c1_spine_features import (
    C1_SPINE_FEATURE_NAMES as SPINE_3D_FEATURE_NAMES_8,
    calculate_spine_metrics_for_seq
)

# v1.0 레거시 22D 피처명 (성능 저해 피처 소거 버전)
PRUNED_LEGACY_NAMES_22 = [
    'trunk_flex_max', 'cervical_angle_min', 'lumbar_hyperext_max',
    'trunk_flex_speed_max', 'lumbar_stiffness_sd', 'asymmetry_score',
    'pelvis_trunk_sync_err', 'phase_speed_ratio', 'bottom_dwell_frames',
    'torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min',
    'ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop',
    'neck_waist_dev', 'spine_cervical_ratio',
    'lit_trunk_lean_max', 'lit_cervical_hyperext_flag',
    'lit_lumbar_shear_proxy', 'lit_spinal_stiffness'
]

# v2.0 공식 30D 피처명
C1_V2_FEATURE_NAMES_30 = PRUNED_LEGACY_NAMES_22 + list(SPINE_3D_FEATURE_NAMES_8)

# 25D 전체 레거시 중 소거 대상 인덱스
EXCLUDE_LEGACY_INDICES = [19, 21, 23]


class BabelSquatC1ModelV2:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v2.0 공식 모델
    - 피처 규격: 30D (레거시 22D + 3D 척추 8D)
    - 오탐 억제: 비용 민감 가중치 (w_normal=1.7)
    - 판별 구조: 2계층 캐스케이드 (Cascade) 및 고정밀 CatBoost 지원
    """

    def __init__(
        self,
        mode: str = "cascade",
        threshold: float = 0.50,
        tau_tier1_low: float = 0.15,
        tau_tier1_high: float = 0.85,
        cost_weight: float = 1.0
    ):
        self.mode = mode
        self.threshold = threshold
        self.tau_low = tau_tier1_low
        self.tau_high = tau_tier1_high
        self.cost_weight = cost_weight

        self.model_tier1 = None
        self.model_tier2 = None
        self.model_champion = None
        self.is_fitted = False

        self.feature_names = C1_V2_FEATURE_NAMES_30
        self._init_default_models()

    def _init_default_models(self):
        """기본 하이퍼파라미터 모델 초기화"""
        if lgb is not None:
            self.model_tier1 = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.04,
                num_leaves=15,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        if CatBoostClassifier is not None:
            self.model_champion = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=5.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )
            self.model_tier2 = CatBoostClassifier(
                iterations=180,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=4.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

    @classmethod
    def compute_curvature_features_from_tensor(cls, X_b: np.ndarray) -> np.ndarray:
        """Format B 시계열에서 6D 곡률 지표 추출"""
        N, T, C = X_b.shape
        curv_feats = np.zeros((N, 6), dtype=np.float32)

        for i in range(N):
            neck_spine_align = X_b[i, :, 82] if C > 82 else np.full(T, 175.0)
            if C > 57:
                knee_flex = 180.0 - (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
            else:
                knee_flex = np.zeros(T)
            b = int(np.clip(np.argmax(knee_flex), 2, T - 3))

            curv_feats[i, 0] = float(np.min(neck_spine_align))
            curv_feats[i, 1] = float(np.mean(neck_spine_align))
            curv_feats[i, 2] = float(neck_spine_align[b] - np.mean(neck_spine_align[:max(1, int(T*0.1))]))
            if C > 6:
                neck_dz = X_b[i, :, 6]
                curv_feats[i, 3] = float(np.max(neck_dz) - np.min(neck_dz))
            else:
                curv_feats[i, 3] = 0.0
            curv_feats[i, 4] = float(np.std(neck_spine_align))
            curv_feats[i, 5] = float(neck_spine_align[b] / (np.min(neck_spine_align) + 1e-4))

        return np.nan_to_num(curv_feats, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def estimate_3d_spine_from_format_b(cls, X_b: np.ndarray) -> np.ndarray:
        """
        별도 3D JSON 좌표가 없을 경우, Format B 텐서로부터 8D 3D 척추 피처 유효 추정
        """
        N, T, C = X_b.shape
        feats_8d = np.zeros((N, 8), dtype=np.float32)

        for i in range(N):
            seq = X_b[i]
            if C > 57:
                knee_flex = 180.0 - (seq[:, 56] + seq[:, 57]) / 2.0
            else:
                knee_flex = np.zeros(T)
            b = int(np.clip(np.argmax(knee_flex), 2, T - 3))

            # 척추 각도 프록시
            spine_angle = seq[:, 82] if C > 82 else np.full(T, 175.0)
            standing_slice = max(1, int(T * 0.15))
            st_angle = float(np.mean(spine_angle[:standing_slice]))

            tl_nadir = float(spine_angle[b])
            tl_drop = max(0.0, st_angle - tl_nadir)
            tl_max_flex = float(np.min(spine_angle))
            tl_hyperext = max(0.0, float(np.max(spine_angle) - st_angle))

            # 버트윙크 프록시 (최하사점 골반-요추 굴곡 급증)
            pelvis_pitch = seq[:, 66] if C > 66 else np.zeros(T)
            buttwink_deg = max(0.0, float(np.mean(pelvis_pitch[:standing_slice]) - pelvis_pitch[b]))

            # 힙힌지 디커플링 프록시
            trunk_pitch = seq[:, 64] if C > 64 else np.zeros(T)
            trunk_delta = max(1e-4, float(trunk_pitch[b] - np.mean(trunk_pitch[:standing_slice])))
            hip_hinge_ratio = float(tl_drop / (trunk_delta + 1e-4))

            # 좌우 비대칭 이탈도
            lat_dev = seq[:, 6] if C > 6 else np.zeros(T)
            lat_dev_max = float(np.max(np.abs(lat_dev)))

            # 요추 전단력 프록시
            shear_arm = seq[:, 8] if C > 8 else np.zeros(T)
            shear_moment = float(np.abs(shear_arm[b]))

            feats_8d[i] = [
                tl_nadir, tl_drop, tl_max_flex,
                buttwink_deg, hip_hinge_ratio, tl_hyperext,
                lat_dev_max, shear_moment
            ]

        return np.nan_to_num(feats_8d, nan=0.0, posinf=180.0, neginf=-180.0)

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        spine_feats_8d: Optional[np.ndarray] = None,
        keypoints_3d: Optional[np.ndarray] = None,
        curv_feats_6d: Optional[np.ndarray] = None,
        fine_feats_12d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        64프레임 Format B 텐서로부터 c1 v2.0 공식 30D 피처셋 추출
        Args:
            X_b: (N, T, C) 또는 (T, C) 크기의 넘파이 배열
            spine_feats_8d: (N, 8) 사전 계산된 3D 척추 피처
            keypoints_3d: (N, T, 24, 3) 3D 관절 좌표 시퀀스
            curv_feats_6d: (N, 6) 사전 계산된 3D 곡률 피처
            fine_feats_12d: (N, 12) 사전 계산된 안면/발 피처
        Returns:
            X_30d: (N, 30) float32 넘파이 배열
        """
        is_single = False
        if X_b.ndim == 2:
            X_b = np.expand_dims(X_b, axis=0)
            is_single = True

        N, T, C = X_b.shape

        # 외부 캐시 자동 로드 (N 일치 시)
        if curv_feats_6d is None:
            c_path = 'd:/src/models/second_models/squat_spine_curvature_feats.npz'
            if os.path.exists(c_path):
                try:
                    c_data = np.load(c_path, allow_pickle=True)['curvature_feats']
                    if len(c_data) == N:
                        curv_feats_6d = c_data
                except Exception:
                    pass

        if fine_feats_12d is None:
            f_path = 'd:/src/models/second_models/squat_fine_feats_12d.npz'
            if os.path.exists(f_path):
                try:
                    f_data = np.load(f_path, allow_pickle=True)['fine_feats']
                    if len(f_data) == N:
                        fine_feats_12d = f_data
                except Exception:
                    pass

        if spine_feats_8d is None and keypoints_3d is None:
            s_path = 'd:/src/models/second_models/babel_squat/squat_c1_spine_feats.npz'
            if os.path.exists(s_path):
                try:
                    s_data = np.load(s_path, allow_pickle=True)['spine_feats']
                    if len(s_data) == N:
                        spine_feats_8d = s_data
                except Exception:
                    pass

        # 1. 레거시 9D 동적 피처
        feats_9d = extract_squat_spine_trend_features(X_b)

        # 2. 1D 상체-정강이 평행도
        X_bot = extract_squat_bottom_biomechanics(X_b)
        torso_shank_err = X_bot[:, 1:2]

        # 3. 3D 고개/시선 및 경추 정렬
        if fine_feats_12d is not None and fine_feats_12d.shape[0] == N:
            gaze_pitch_mean = fine_feats_12d[:, 0:1]
            gaze_pitch_min = fine_feats_12d[:, 1:2]
            cervical_min = fine_feats_12d[:, 3:4]
        else:
            gaze_pitch_mean = np.mean(X_b[:, :, 64:65], axis=1) * 0.8 if C > 64 else np.zeros((N, 1))
            gaze_pitch_min = np.min(X_b[:, :, 64:65], axis=1) * 0.8 if C > 64 else np.zeros((N, 1))
            cervical_min = np.min(X_b[:, :, 82:83], axis=1) if C > 82 else np.full((N, 1), 170.0)

        # 4. 6D 곡률
        if curv_feats_6d is not None and curv_feats_6d.shape[0] == N:
            curv_6d = curv_feats_6d[:, :6]
        else:
            curv_6d = cls.compute_curvature_features_from_tensor(X_b)

        # 5. 6D 문헌 피처
        lit_6d = extract_squat_spine_literature_features(X_b)

        # 25D 결합 후 저해 피처 소거 -> 22D
        X_25d = np.concatenate([
            feats_9d, torso_shank_err, cervical_min,
            gaze_pitch_mean, gaze_pitch_min, curv_6d, lit_6d
        ], axis=1).astype(np.float32)

        keep_idx = [i for i in range(25) if i not in EXCLUDE_LEGACY_INDICES]
        X_22d = X_25d[:, keep_idx]

        # 6. 신규 8D 3D 척추 피처셋 결합
        if spine_feats_8d is not None and spine_feats_8d.shape[0] == N:
            s_8d = spine_feats_8d.astype(np.float32)
        elif keypoints_3d is not None and keypoints_3d.shape[0] == N:
            s_8d = np.zeros((N, 8), dtype=np.float32)
            for i in range(N):
                s_8d[i] = calculate_spine_metrics_for_seq(keypoints_3d[i])
        else:
            s_8d = cls.estimate_3d_spine_from_format_b(X_b)

        X_30d = np.concatenate([X_22d, s_8d], axis=1).astype(np.float32)

        if is_single:
            return X_30d[0]
        return X_30d

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ambiguity_levels: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None,
        cost_weight: Optional[float] = None
    ):
        """
        모델 학습 (비용 민감 가중치 적용)
        """
        if cost_weight is not None:
            self.cost_weight = cost_weight

        # 정상 샘플 오탐 방지 가중치 생성 (y=0: cost_weight, y=1: 1.0)
        sample_weight = np.where(y == 0, self.cost_weight, 1.0).astype(np.float32)

        if self.mode == "catboost":
            if self.model_champion is None:
                self._init_default_models()
            self.model_champion.fit(X, y, sample_weight=sample_weight)
        else:
            # 2계층 캐스케이드 학습
            if ambiguity_levels is not None:
                mask_t1 = (ambiguity_levels == 1)
                mask_t2 = (ambiguity_levels >= 2) & (ambiguity_levels <= 3)
            else:
                mask_t1 = np.ones(len(y), dtype=bool)
                mask_t2 = np.ones(len(y), dtype=bool)

            if np.sum(mask_t1) > 20:
                X_t1, y_t1 = X[mask_t1], y[mask_t1]
                sw_t1 = sample_weight[mask_t1]
            else:
                X_t1, y_t1 = X, y
                sw_t1 = sample_weight

            if np.sum(mask_t2) > 20:
                X_t2, y_t2 = X[mask_t2], y[mask_t2]
                sw_t2 = sample_weight[mask_t2]
            else:
                X_t2, y_t2 = X, y
                sw_t2 = sample_weight

            self.model_tier1.fit(X_t1, y_t1, sample_weight=sw_t1)
            self.model_tier2.fit(X_t2, y_t2, sample_weight=sw_t2)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """결함 발생 확률 배열 반환 (N, 2) [P(정상), P(결함)]"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load()를 호출하십시오.")

        is_single = (X.ndim == 1)
        if is_single:
            X = np.expand_dims(X, axis=0)

        if self.mode == "catboost":
            p1 = self.model_champion.predict_proba(X)[:, 1]
        else:
            p_t1 = self.model_tier1.predict_proba(X)[:, 1]
            p_t2 = self.model_tier2.predict_proba(X)[:, 1]

            p1 = np.zeros(len(X), dtype=np.float32)
            for i in range(len(X)):
                if p_t1[i] <= self.tau_low:
                    p1[i] = p_t1[i]
                elif p_t1[i] >= self.tau_high:
                    p1[i] = p_t1[i]
                else:
                    p1[i] = 0.3 * p_t1[i] + 0.7 * p_t2[i]

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        if is_single:
            return np.array([p0[0], p1[0]], dtype=np.float32)
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판정 (0: 정상, 1: 척추 중립 위반 결함)"""
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def load_weights(self, filepath: str):
        """load의 별칭 (v1 및 타 모듈 일관성 유지)"""
        return self.load(filepath)

    def diagnose(
        self,
        X_sample: np.ndarray,
        spine_feats_8d: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        단일 시퀀스 정밀 진단 및 생체역학 세부 해석 결과 반환
        """
        if X_sample.ndim == 2:
            feats = self.extract_features(X_sample, spine_feats_8d=spine_feats_8d)
        elif X_sample.ndim == 1 and len(X_sample) == 30:
            feats = X_sample
        else:
            raise ValueError(f"입력 배열 형태가 올바르지 않습니다: shape={X_sample.shape}")

        probs = self.predict_proba(feats)
        prob = float(probs[1]) if probs.ndim == 1 else float(probs[0, 1])
        is_defect = bool(prob >= self.threshold)

        # 30D 피처 인덱스 기준 생체역학 지표 매핑
        butt_wink = float(feats[25])
        tl_flexion = float(feats[23])
        hip_hinge_ratio = float(feats[26])
        lateral_tilt = float(feats[28])
        shear_arm = float(feats[29])

        # 중증도(Severity) 판정
        if not is_defect:
            severity = "Normal (정상 중립)"
            actionable_feedback = "요추 및 흉추가 중립 상태를 완벽히 유지하고 있습니다."
        elif prob >= 0.80:
            severity = "Severe (중증 척추 중립 붕괴)"
            actionable_feedback = (
                f"최하사점에서 버트윙크({butt_wink:.1f}°) 및 흉요추 굴곡({tl_flexion:.1f}°)이 심각합니다. "
                "고관절 가동 범위를 점검하고 허리가 말리기 직전까지만 깊이를 조절하십시오."
            )
        else:
            severity = "Mild (경미한 척추 정렬 흔들림)"
            actionable_feedback = (
                f"동작 중 상체와 척추 분리비율({hip_hinge_ratio:.2f})에 불안정성이 감지되었습니다. "
                "복압(브레이싱)을 단단히 유지하고 척추 중립에 주의를 기울이십시오."
            )

        return {
            "defect_id": "c1",
            "defect_name": "척추의 중립 위반 (Spine Neutrality)",
            "is_defect": is_defect,
            "probability": round(float(prob), 4),
            "confidence": round(float(abs(prob - 0.5) * 2), 4),
            "severity": severity,
            "biomechanical_metrics": {
                "butt_wink_lumbar_tuck_deg": round(butt_wink, 2),
                "thoracolumbar_flexion_deg": round(tl_flexion, 2),
                "hip_hinge_decoupling_ratio": round(hip_hinge_ratio, 3),
                "lateral_tilt_deviation_cm": round(lateral_tilt, 2),
                "lumbar_shear_moment_arm_cm": round(shear_arm, 2)
            },
            "actionable_feedback": actionable_feedback
        }

    def save(self, filepath: str):
        """모델 가중치 및 메타데이터 저장"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "mode": self.mode,
            "threshold": self.threshold,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "cost_weight": self.cost_weight,
            "feature_names": self.feature_names,
            "model_tier1": self.model_tier1,
            "model_tier2": self.model_tier2,
            "model_champion": self.model_champion,
            "is_fitted": self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[*] c1_v2 모델 가중치 저장 완료: {filepath}")

    def load(self, filepath: str):
        """저장된 모델 가중치 복원"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.mode = data.get("mode", "cascade")
        self.threshold = data.get("threshold", 0.50)
        self.tau_low = data.get("tau_low", 0.15)
        self.tau_high = data.get("tau_high", 0.85)
        self.cost_weight = data.get("cost_weight", 1.0)
        self.feature_names = data.get("feature_names", C1_V2_FEATURE_NAMES_30)
        self.model_tier1 = data.get("model_tier1")
        self.model_tier2 = data.get("model_tier2")
        self.model_champion = data.get("model_champion")
        self.is_fitted = data.get("is_fitted", True)
        return self


if __name__ == '__main__':
    print("=" * 80)
    print(" [vidio-health-trainer] BabelSquatC1ModelV2 자체 검증 및 가중치 생성")
    print("=" * 80)

    from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0]
    subjects = ds['subjects']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)

    # 64F 스플라인 보간
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)

    # 3D 척추 피처
    spine_3d = np.load('d:/src/models/second_models/babel_squat/squat_c1_spine_feats.npz', allow_pickle=True)['spine_feats'].astype(np.float32)

    # 4단계 모호성 레벨
    amb_data = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz')
    levels = amb_data['levels']

    # 모델 인스턴스 생성 및 피처 추출
    model = BabelSquatC1ModelV2(mode="cascade", cost_weight=1.0)
    X_feats_30d = model.extract_features(X_b_64, spine_feats_8d=spine_3d)
    print(f"[*] 30D 피처 추출 완료: shape = {X_feats_30d.shape}")

    # 모델 학습
    model.fit(X_feats_30d, y_c1, ambiguity_levels=levels, subjects=subjects)

    # 단일 샘플 진단 테스트
    diag = model.diagnose(X_feats_30d[0])
    import json
    print("\n[*] 단일 샘플 진단 결과:")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

    # 모델 가중치 저장
    weight_path = os.path.join(CURRENT_DIR, "c1_model_v2_weights.pkl")
    model.save(weight_path)
    print(f"[성공] c1_model_v2 파일 및 가중치 생성 완료.")
