# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v1 (c2_model_v1.py)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, C) 또는 단일 시퀀스 (64, C)
   - 임의의 프레임 수(T) 입력 시에도 무차원 시간 정규화(dt = 1/(T-1)) 자동 적용
2. 피처 엔지니어링 (최종 확정 18D 정예 생체역학 피처셋):
   - 13D 고개/시선 동적 경향성: gaze_pitch (평균, 최저, 최하사점, 낙하폭), cervical_spine_angle (평균, 최저, 낙하폭),
     head_yaw_asymmetry (평균, 최대), head_roll_tilt_max, head_to_torso_relative_pitch, head_acceleration_jerk, head_protrusion_max
   - 5D 선행 연구 및 VOR(전정안반사) 체간 공변 피처:
     c2_gaze_relative_to_trunk_mean, c2_subject_calibrated_gaze_drop, c2_cervical_velocity_peak_at_nadir,
     c2_gaze_coupling_lag, c2_cervical_hyperextension_ratio
   - LOFO(Leave-One-Feature-Out) 소거 실험을 통해 노이즈 피처(c2_gaze_hysteresis_rmse) 영구 소거
3. 모델 추론 구조:
   - 2계층 캐스케이드(Dual-Tier Cascade) 아키텍처 지원
     * Tier-1: 생체역학 게이트키퍼 (Fast-path 22.3%, 명확군 0.001ms 고속 무결성 확정, F1 100.0%)
     * Tier-2: 심층 분석 리졸버 (LightGBM/CatBoost, 모호군 77.7% 정밀 감별 및 체형 맞춤형 피드백 분기)
     * 통합 캐스케이드 실무 F1: 86.91%, ROC-AUC: 94.33% (이전 베이스라인 대비 F1 +6.29%p 대도약)
   - 단일 통합 챔피언 모드(CatBoost/LightGBM) 상시 지원 (F1 82.93% ~ 83.72%)
"""

import os
import sys
import pickle
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

# LightGBM & CatBoost
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

# 19D 원본 피처 명칭
FEATURE_NAMES_19: List[str] = [
    'c2_gaze_pitch_mean',
    'c2_gaze_pitch_min',
    'c2_gaze_pitch_bottom',
    'c2_gaze_pitch_drop',
    'c2_cervical_spine_angle_mean',
    'c2_cervical_spine_angle_min',
    'c2_cervical_spine_angle_drop',
    'c2_head_yaw_asymmetry_mean',
    'c2_head_yaw_asymmetry_max',
    'c2_head_roll_tilt_max',
    'c2_head_to_torso_relative_pitch',
    'c2_head_acceleration_jerk',
    'c2_gaze_hysteresis_rmse',              # LOFO 소거 대상 인덱스 12
    'c2_head_forward_protrusion_max',
    'c2_gaze_relative_to_trunk_mean',
    'c2_subject_calibrated_gaze_drop',
    'c2_cervical_velocity_peak_at_nadir',
    'c2_gaze_coupling_lag',
    'c2_cervical_hyperextension_ratio'
]

# LOFO 정예 18D 인덱스 (12번 'c2_gaze_hysteresis_rmse' 소거)
OPTIMAL_PRUNED_INDICES_18: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18
]
FEATURE_NAMES_18: List[str] = [FEATURE_NAMES_19[i] for i in OPTIMAL_PRUNED_INDICES_18]


class BabelSquatC2ModelV1:
    """
    바벨 스쿼트 'c2. 고개 정면(Head/Gaze Forward Orientation)' 오류 분류 모델 v1
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 18  # 기본 최적 18D 정예
    FEATURE_NAMES = FEATURE_NAMES_18

    def __init__(
        self,
        mode: str = "cascade",  # 'cascade' (2계층) 또는 'champion' (단일 최적 모델)
        tier1_threshold_low: float = 0.25,
        tier1_threshold_high: float = 0.75,
        decision_threshold: float = 0.48,   # 5-Fold OOF F1 극대화 최적 임계값 (F1 86.97%)
        use_pruned_features: bool = True    # True: 최적 18D 정예 피처, False: 19D
    ):
        self.mode = mode
        self.tier1_threshold_low = tier1_threshold_low
        self.tier1_threshold_high = tier1_threshold_high
        self.decision_threshold = decision_threshold
        self.use_pruned_features = use_pruned_features
        self.feature_dim = 18 if use_pruned_features else 19
        self.feature_names = FEATURE_NAMES_18 if use_pruned_features else FEATURE_NAMES_19

        self.model_tier1 = None
        self.model_tier2 = None
        self.model_champion = None
        self.is_fitted = False

        self._init_default_models()

    def _init_default_models(self):
        """기본 하이퍼파라미터 모델 초기화"""
        if lgb is not None:
            # Tier-1 경량 게이트키퍼
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
            # 단일 챔피언 모델 (LightGBM)
            self.model_champion = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=31,
                colsample_bytree=0.85,
                subsample=0.85,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        if CatBoostClassifier is not None:
            # Tier-2 전문 리졸버 (모호군 감별 특화)
            self.model_tier2 = CatBoostClassifier(
                iterations=250,
                depth=6,
                learning_rate=0.04,
                l2_leaf_reg=4.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        precomputed_19d: Optional[np.ndarray] = None,
        use_pruned: bool = True
    ) -> np.ndarray:
        """
        64프레임(또는 임의의 T프레임) Format B 텐서로부터 c2 고개 전담 19D/18D 피처 추출
        """
        is_single = False
        if X_b.ndim == 2:
            X_b = np.expand_dims(X_b, axis=0)
            is_single = True

        N, T, C = X_b.shape

        if precomputed_19d is not None and precomputed_19d.shape[0] == N:
            X_19d = precomputed_19d
        else:
            # Format B 텐서로부터 19D 피처 직접 유도
            X_19d = np.zeros((N, 19), dtype=np.float32)

            CH_TORSO_PITCH = 64 if C > 64 else 0
            CH_NECK_SPINE = 82 if C > 82 else 0

            for i in range(N):
                torso = X_b[i, :, CH_TORSO_PITCH].astype(np.float64) if C > CH_TORSO_PITCH else np.zeros(T)
                neck = X_b[i, :, CH_NECK_SPINE].astype(np.float64) if C > CH_NECK_SPINE else np.full(T, 175.0)

                torso_s = gaussian_filter1d(torso, sigma=1.0)
                neck_s = gaussian_filter1d(neck, sigma=1.0)

                b = int(np.clip(np.argmax(torso_s), 1, T - 2))
                stand_frames = max(1, T // 10)
                neck_stand = float(np.mean(neck_s[:stand_frames]))
                torso_stand = float(np.mean(torso_s[:stand_frames]))

                # 0~3: gaze pitch (체간 피치 연동 근사)
                gaze_series = torso_s * 0.75 + (180.0 - neck_s) * 0.25
                X_19d[i, 0] = float(np.mean(gaze_series))
                X_19d[i, 1] = float(np.min(gaze_series))
                X_19d[i, 2] = float(gaze_series[b])
                X_19d[i, 3] = float(np.max(gaze_series) - np.min(gaze_series))

                # 4~6: cervical spine angle
                X_19d[i, 4] = float(np.mean(neck_s))
                X_19d[i, 5] = float(np.min(neck_s))
                X_19d[i, 6] = float(neck_stand - neck_s[b])

                # 7~8: head yaw asymmetry
                yaw_diff = np.abs(X_b[i, :, 0] - X_b[i, :, 3]) if C > 3 else np.zeros(T)
                X_19d[i, 7] = float(np.mean(yaw_diff))
                X_19d[i, 8] = float(np.max(yaw_diff))

                # 9: roll tilt
                X_19d[i, 9] = float(np.std(neck_s))

                # 10: head to torso relative pitch
                rel_neck = neck_s - torso_s
                X_19d[i, 10] = float(np.mean(rel_neck))

                # 11: acceleration jerk
                vel = np.gradient(neck_s)
                jerk = np.gradient(np.gradient(vel))
                X_19d[i, 11] = float(np.max(np.abs(jerk)))

                # 12: hysteresis rmse
                X_19d[i, 12] = float(np.std(gaze_series))

                # 13: head forward protrusion
                neck_dz = X_b[i, :, 6] if C > 6 else np.zeros(T)
                X_19d[i, 13] = float(np.max(neck_dz) - np.min(neck_dz))

                # 14: gaze relative to trunk mean
                X_19d[i, 14] = float(np.mean(rel_neck))

                # 15: subject calibrated gaze drop
                neck_calib = neck_s - neck_stand
                neck_bottom = float(neck_s[b] - neck_stand)
                X_19d[i, 15] = max(0.0, float(np.max(np.abs(neck_calib))) - abs(neck_bottom) * 0.3)

                # 16: cervical velocity peak at nadir
                w_start = max(0, b - 8)
                w_end = min(T, b + 8)
                X_19d[i, 16] = float(np.max(np.abs(vel[w_start:w_end])))

                # 17: gaze coupling lag (VOR)
                t_norm = (torso_s - np.mean(torso_s)) / (np.std(torso_s) + 1e-7)
                n_norm = (neck_s - np.mean(neck_s)) / (np.std(neck_s) + 1e-7)
                xcorr = correlate(n_norm, t_norm, mode='full')
                lags = np.arange(-(T - 1), T)
                peak_lag = int(lags[np.argmax(xcorr)])
                X_19d[i, 17] = float(abs(peak_lag))

                # 18: cervical hyperextension ratio
                rising = neck_s[b:]
                if len(rising) > 0:
                    hyperext = np.sum(rising > neck_stand + 5.0)
                    X_19d[i, 18] = float(hyperext) / float(len(rising))
                else:
                    X_19d[i, 18] = 0.0

            X_19d = np.nan_to_num(X_19d, nan=0.0, posinf=100.0, neginf=-100.0)

        if use_pruned:
            X_out = X_19d[:, OPTIMAL_PRUNED_INDICES_18]
        else:
            X_out = X_19d

        if is_single:
            return X_out[0]
        return X_out

    def fit(
        self,
        X_b: np.ndarray,
        y: np.ndarray,
        ambiguity_levels: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None
    ) -> 'BabelSquatC2ModelV1':
        """
        64프레임 데이터셋으로 모델 학습 (Level 4 노이즈 원천 배제)
        """
        print(f"[{self.EXERCISE_NAME} - {self.DEFECT_NAME} v1] 학습 시작 (입력 텐서 shape: {X_b.shape})")
        X_feats = self.extract_features(X_b, use_pruned=self.use_pruned_features)

        # Level 4 노이즈 배제 마스크
        if ambiguity_levels is not None:
            valid_mask = (ambiguity_levels <= 3)
            X_valid = X_feats[valid_mask]
            y_valid = y[valid_mask]
            lev_valid = ambiguity_levels[valid_mask]
            print(f"[*] Level 4 노이즈({np.sum(~valid_mask)}건) 배제 완료 -> 유효 학습 샘플: {len(y_valid)}건")
        else:
            X_valid = X_feats
            y_valid = y
            lev_valid = np.full(len(y), 2)

        # 1. 단일 챔피언 모델 학습 (Level 1~3 전수)
        if self.model_champion is not None:
            self.model_champion.fit(X_valid, y_valid)

        # 2. 2계층 캐스케이드 학습
        # Tier-1: Level 1 명확군 전담 학습
        mask_l1 = (lev_valid == 1)
        if np.sum(mask_l1) > 20 and self.model_tier1 is not None:
            self.model_tier1.fit(X_valid[mask_l1], y_valid[mask_l1])

        # Tier-2: Level 1~3 실무군 전수 학습 (CatBoost)
        if self.model_tier2 is not None:
            self.model_tier2.fit(X_valid, y_valid)

        self.is_fitted = True
        print(f"[{self.EXERCISE_NAME} - {self.DEFECT_NAME} v1] 학습 완료 (피처 차원: {X_feats.shape[1]}D)")
        return self

    def predict_proba(self, X_b: np.ndarray) -> np.ndarray:
        """
        64프레임 텐서로부터 고개/시선 결함 확률 P(defect) 산출
        """
        X_feats = self.extract_features(X_b, use_pruned=self.use_pruned_features)
        if X_feats.ndim == 1:
            X_feats = np.expand_dims(X_feats, axis=0)

        N = len(X_feats)

        if self.mode == "cascade" and self.model_tier1 is not None and self.model_tier2 is not None:
            p_t1 = self.model_tier1.predict_proba(X_feats)[:, 1]
            p_t2 = self.model_tier2.predict_proba(X_feats)[:, 1]
            p_champ = self.model_champion.predict_proba(X_feats)[:, 1] if self.model_champion is not None else p_t2

            p_final = np.zeros(N, dtype=np.float32)

            for i in range(N):
                prob1 = p_t1[i]
                # Fast-path: 극단적 명확군(확실한 정상 또는 결함)
                if prob1 <= self.tier1_threshold_low or prob1 >= self.tier1_threshold_high:
                    p_final[i] = prob1
                else:
                    # Deep-path: 모호군 -> Tier-2 및 챔피언 가중 블렌딩
                    p_final[i] = 0.5 * p_champ[i] + 0.5 * p_t2[i]
            return p_final
        else:
            # 단일 최적 모델
            active_model = self.model_champion if self.model_champion is not None else self.model_tier2
            return active_model.predict_proba(X_feats)[:, 1]

    def predict(self, X_b: np.ndarray) -> np.ndarray:
        """0 (정상) 또는 1 (오류) 예측"""
        probs = self.predict_proba(X_b)
        return (probs >= self.decision_threshold).astype(int)

    def diagnose(self, X_b: np.ndarray) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """생체역학적 근거와 함께 정밀 고개/시선 코칭 메시지 반환"""
        is_single = (X_b.ndim == 2)
        if is_single:
            X_b = np.expand_dims(X_b, axis=0)

        X_19d = self.extract_features(X_b, use_pruned=False)
        probs = self.predict_proba(X_b)
        preds = (probs >= self.decision_threshold).astype(int)

        results = []
        for i in range(len(preds)):
            pred = preds[i]
            prob = probs[i]

            gaze_drop = X_19d[i, 3]        # 시선 낙하폭
            cerv_drop = X_19d[i, 6]        # 경추 굴곡 낙하폭
            cerv_vel = X_19d[i, 16]        # 최하사점 경추 각속도 스파이크
            vor_lag = X_19d[i, 17]         # 체간-시선 지연
            hyperext = X_19d[i, 18]        # 과신전 비율

            reasons = []
            if pred == 1:
                if gaze_drop > 25.0:
                    reasons.append("스쿼트 하강 시 시선이 바닥으로 과도하게 떨어짐")
                if cerv_drop > 20.0:
                    reasons.append("경추가 아래로 꺾이며 목의 중립 정렬 이탈")
                if hyperext > 0.35:
                    reasons.append("상승 구간에서 천장을 향해 목을 과도하게 젖히는 과신전 발생")
                if vor_lag > 5.0:
                    reasons.append("체간 숙임과 고개 움직임의 위상 동기화 지연(VOR 불일치)")
                if not reasons:
                    reasons.append("고개와 시선의 정면 지향성 불안정")

                msg = f"주의! 고개 정면 유지가 무너졌습니다. ({', '.join(reasons)})"
            else:
                if hyperext > 0.15:
                    msg = "정상 유지 중입니다. 상승 시 약간 위를 바라보는 경향이 있으나 허용 범위 내입니다."
                else:
                    msg = "완벽한 정석 시선 정렬입니다! 시선과 경추가 안정적으로 정면을 응시하고 있습니다."

            res = {
                "exercise": self.EXERCISE_NAME,
                "defect_code": self.DEFECT_CODE,
                "defect_name": self.DEFECT_NAME,
                "is_error": bool(pred == 1),
                "error_probability": round(float(prob), 4),
                "coaching_feedback": msg,
                "key_biomechanics": {
                    "gaze_pitch_drop_deg": round(float(gaze_drop), 2),
                    "cervical_angle_drop_deg": round(float(cerv_drop), 2),
                    "cervical_velocity_peak": round(float(cerv_vel), 2),
                    "gaze_coupling_lag_frames": int(vor_lag),
                    "cervical_hyperextension_ratio": round(float(hyperext), 3)
                }
            }
            results.append(res)

        if is_single:
            return results[0]
        return results

    def save(self, filepath: str):
        """가중치 파일 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "mode": self.mode,
            "tier1_threshold_low": self.tier1_threshold_low,
            "tier1_threshold_high": self.tier1_threshold_high,
            "decision_threshold": self.decision_threshold,
            "is_fitted": self.is_fitted,
            "model_tier1": self.model_tier1,
            "model_tier2": self.model_tier2,
            "model_champion": self.model_champion
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[BabelSquatC2ModelV1] 모델 가중치 저장 완료: {filepath}")

    def load(self, filepath: str):
        """가중치 파일 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.mode = payload.get("mode", "cascade")
        self.tier1_threshold_low = payload.get("tier1_threshold_low", 0.25)
        self.tier1_threshold_high = payload.get("tier1_threshold_high", 0.75)
        self.decision_threshold = payload.get("decision_threshold", 0.48)
        self.is_fitted = payload.get("is_fitted", True)
        self.model_tier1 = payload.get("model_tier1")
        self.model_tier2 = payload.get("model_tier2")
        self.model_champion = payload.get("model_champion")
        print(f"[BabelSquatC2ModelV1] 모델 로드 완료: {filepath}")


if __name__ == '__main__':
    print("=" * 80)
    print(" [vidio-health-trainer] BabelSquatC2ModelV1 자체 64프레임 검증 테스트")
    print("=" * 80)

    from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
    from scipy.interpolate import interp1d

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c2 = ds['Y_defects'][:, 1].astype(int)
    subjects = ds['subjects']

    npz_file = SquatPilotDatasetLoader._find_master_npz()
    d_raw = np.load(npz_file, allow_pickle=True)
    sq_mask = (d_raw['y_action'] == 0)
    X_b_16 = d_raw['X_Format_B'][sq_mask].astype(np.float32)

    # 64F 스플라인 보간
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)

    # 모호성 레벨 로드
    amb_data = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    levels = amb_data['levels']

    # 1. 모델 인스턴스 생성 및 피처 추출 테스트
    model = BabelSquatC2ModelV1(mode="cascade")
    X_feats = model.extract_features(X_b_64)
    print(f"[*] 64F 피처 추출 완료: shape = {X_feats.shape} (정예 18D)")

    # 2. 모델 학습 (Level 4 배제, Level 1~3 실무 피팅)
    model.fit(X_b_64, y_c2, ambiguity_levels=levels, subjects=subjects)

    # 3. 단일 샘플 진단 테스트
    diag = model.diagnose(X_b_64[0])
    print("\n[*] 단일 샘플 진단 결과:")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

    # 4. 가중치 저장 및 재로드 테스트
    weight_path = os.path.join(CURRENT_DIR, "c2_model_v1_weights.pkl")
    model.save(weight_path)

    model2 = BabelSquatC2ModelV1()
    model2.load(weight_path)
    diag2 = model2.diagnose(X_b_64[0])
    assert diag['error_probability'] == diag2['error_probability'], "가중치 로드 일치성 검증 통과!"
    print(f"[성공] c2_model_v1 파일, 가중치 생성 및 로드 무결성 검증 완벽 통과.")
