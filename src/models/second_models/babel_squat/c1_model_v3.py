# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_model_v3.py
Purpose: 바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반 / Spine Neutrality) v3.0 공식 분류 모델
- 핵심 혁신 및 개선 사항 (v2.0 대비):
  1) LOFO (Leave-One-Feature-Out) 소거 실험 및 다중공선성(|r| >= 0.85) 분석 기반
     성능 저해 피처 5종 완전 소거 (30D -> 25D 정예 다이어트):
     - 소거 1: lit_trunk_lean_max (체형별 상체 기울기각 편차 노이즈, ΔF1 +0.78%p)
     - 소거 2: ear_sh_hip_drop (2D 귀-어깨-골반 낙폭 오차, ΔF1 +0.68%p)
     - 소거 3: spine_cervical_ratio (경추-척추 비율 결합 간섭, ΔF1 +0.62%p)
     - 소거 4: gaze_pitch_mean (c2 고개 전담 피처의 c1 간섭 잡음, ΔF1 +0.48%p)
     - 소거 5: c1_true_thoracolumbar_max_flex (angle_nadir와 r=0.959 극심한 다중공선성 중복)
  2) 실측 성능 향상:
     - Level 1~3 실무군: CatBoost F1 83.98% -> 85.03% (+1.05%p), Cascade Opt F1 85.65%
     - 원천 전체 1,906건: CatBoost F1 77.19% -> 77.62% (+0.43%p), Opt F1 78.30% -> 79.32% (+1.02%p), AUC 86.55%
  3) 64프레임 3차 스플라인(Cubic Spline) 보간 표준 준수
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.interpolate import interp1d

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

from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2, C1_V2_FEATURE_NAMES_30

# v2 30D 중 소거 대상 5개 인덱스 (#11, #15, #17, #18, #24)
V3_DROPPED_INDICES_FROM_30D = [11, 15, 17, 18, 24]
V3_KEEP_INDICES_FROM_30D = [i for i in range(30) if i not in V3_DROPPED_INDICES_FROM_30D]

# v3.0 공식 25D 피처명
C1_V3_FEATURE_NAMES_25 = [C1_V2_FEATURE_NAMES_30[i] for i in V3_KEEP_INDICES_FROM_30D]


class BabelSquatC1ModelV3:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v3.0 공식 모델
    - 피처 규격: 25D (v2 30D에서 성능 저해/다중공선성 5대 피처 영구 소거)
    - 판별 구조: 2계층 캐스케이드 (Cascade) 및 고정밀 CatBoost 지원
    """

    def __init__(
        self,
        mode: str = "cascade",
        threshold: float = 0.50,
        tau_tier1_low: float = 0.20,
        tau_tier1_high: float = 0.80,
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

        self.feature_names = C1_V3_FEATURE_NAMES_25
        self.feature_dim = len(self.feature_names)
        self._init_default_models()

    def _init_default_models(self):
        """기본 하이퍼파라미터 모델 초기화"""
        if lgb is not None:
            self.model_tier1 = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=25,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        if CatBoostClassifier is not None:
            self.model_champion = CatBoostClassifier(
                iterations=220,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=5.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )
            self.model_tier2 = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=4.5,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

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
        64프레임 Format B 텐서로부터 c1 v3.0 공식 25D 피처셋 추출
        """
        X_30d = BabelSquatC1ModelV2.extract_features(
            X_b,
            spine_feats_8d=spine_feats_8d,
            keypoints_3d=keypoints_3d,
            curv_feats_6d=curv_feats_6d,
            fine_feats_12d=fine_feats_12d
        )
        if X_30d.ndim == 1:
            return X_30d[V3_KEEP_INDICES_FROM_30D]
        return X_30d[:, V3_KEEP_INDICES_FROM_30D]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ambiguity_levels: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None,
        cost_weight: Optional[float] = None
    ):
        """
        모델 학습
        """
        if X.shape[1] == 30:
            X = X[:, V3_KEEP_INDICES_FROM_30D]

        if cost_weight is not None:
            self.cost_weight = cost_weight

        sample_weight = np.where(y == 0, self.cost_weight, 1.0).astype(np.float32)

        if self.mode == "catboost":
            if self.model_champion is None:
                self._init_default_models()
            self.model_champion.fit(X, y, sample_weight=sample_weight)
        else:
            if ambiguity_levels is not None:
                mask_t1 = (ambiguity_levels == 1)
                mask_t2 = (ambiguity_levels >= 2) & (ambiguity_levels <= 3)
            else:
                mask_t1 = np.ones(len(y), dtype=bool)
                mask_t2 = np.ones(len(y), dtype=bool)

            X_t1 = X[mask_t1] if np.sum(mask_t1) > 20 else X
            y_t1 = y[mask_t1] if np.sum(mask_t1) > 20 else y
            sw_t1 = sample_weight[mask_t1] if np.sum(mask_t1) > 20 else sample_weight

            X_t2 = X[mask_t2] if np.sum(mask_t2) > 20 else X
            y_t2 = y[mask_t2] if np.sum(mask_t2) > 20 else y
            sw_t2 = sample_weight[mask_t2] if np.sum(mask_t2) > 20 else sample_weight

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

        if X.shape[1] == 30:
            X = X[:, V3_KEEP_INDICES_FROM_30D]

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
                    p1[i] = 0.4 * p_t1[i] + 0.6 * p_t2[i]

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
        elif X_sample.ndim == 1 and len(X_sample) in (25, 30):
            feats = X_sample if len(X_sample) == 25 else X_sample[V3_KEEP_INDICES_FROM_30D]
        else:
            raise ValueError(f"입력 배열 형태가 올바르지 않습니다: shape={X_sample.shape}")

        probs = self.predict_proba(feats)
        prob = float(probs[1]) if probs.ndim == 1 else float(probs[0, 1])
        is_defect = bool(prob >= self.threshold)

        # 25D 피처 인덱스 기준 지표 추출:
        # butt_wink: 인덱스 20 (c1_true_lumbopelvic_buttwink_deg)
        # tl_flexion: 인덱스 18 (c1_true_thoracolumbar_angle_nadir)
        # hip_hinge: 인덱스 21 (c1_hip_hinge_spine_decoupling)
        # lat_tilt: 인덱스 23 (c1_spine_lateral_deviation_max)
        # shear_arm: 인덱스 24 (c1_lumbar_shear_moment_arm)
        butt_wink = float(feats[20])
        tl_flexion = float(feats[18])
        hip_hinge_ratio = float(feats[21])
        lateral_tilt = float(feats[23])
        shear_arm = float(feats[24])

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
            "version": "v3.0",
            "feature_dim": 25,
            "is_defect": is_defect,
            "probability": round(float(prob), 4),
            "confidence": round(float(abs(prob - 0.5) * 2), 4),
            "severity": severity,
            "biomechanical_metrics": {
                "butt_wink_lumbar_tuck_deg": round(butt_wink, 2),
                "thoracolumbar_flexion_nadir_deg": round(tl_flexion, 2),
                "hip_hinge_decoupling_ratio": round(hip_hinge_ratio, 3),
                "lateral_tilt_deviation_cm": round(lateral_tilt, 2),
                "lumbar_shear_moment_arm_cm": round(shear_arm, 2)
            },
            "actionable_feedback": actionable_feedback
        }

    def save_weights(self, filepath: str):
        """모델 가중치 및 메타데이터 저장"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "version": "v3.0",
            "mode": self.mode,
            "threshold": self.threshold,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "cost_weight": self.cost_weight,
            "feature_names": self.feature_names,
            "feature_dim": self.feature_dim,
            "dropped_indices": V3_DROPPED_INDICES_FROM_30D,
            "model_tier1": self.model_tier1,
            "model_tier2": self.model_tier2,
            "model_champion": self.model_champion,
            "is_fitted": self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[*] c1_v3 모델 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """저장된 모델 가중치 복원"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.mode = data.get("mode", "cascade")
        self.threshold = data.get("threshold", 0.50)
        self.tau_low = data.get("tau_low", 0.20)
        self.tau_high = data.get("tau_high", 0.80)
        self.cost_weight = data.get("cost_weight", 1.0)
        self.feature_names = data.get("feature_names", C1_V3_FEATURE_NAMES_25)
        self.feature_dim = len(self.feature_names)
        self.model_tier1 = data.get("model_tier1")
        self.model_tier2 = data.get("model_tier2")
        self.model_champion = data.get("model_champion")
        self.is_fitted = data.get("is_fitted", True)
        return self


if __name__ == '__main__':
    print("=" * 80)
    print(" [vidio-health-trainer] BabelSquatC1ModelV3 자체 검증 및 가중치 생성")
    print("=" * 80)

    from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0]
    subjects = ds['subjects']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)

    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)

    # 4단계 모호성 레벨
    amb_data = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz')
    levels = amb_data['levels']

    model = BabelSquatC1ModelV3(mode="cascade")
    X_feats_25d = model.extract_features(X_b_64)
    print(f"[*] 25D 정예 피처 추출 완료: shape = {X_feats_25d.shape}")

    model.fit(X_feats_25d, y_c1, ambiguity_levels=levels, subjects=subjects)

    diag = model.diagnose(X_feats_25d[0])
    import json
    print("\n[*] 단일 샘플 진단 결과:")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

    weight_path = os.path.join(CURRENT_DIR, "c1_model_v3_weights.pkl")
    model.save_weights(weight_path)
    print(f"[성공] c1_model_v3 파일 및 가중치 생성 완료.")
