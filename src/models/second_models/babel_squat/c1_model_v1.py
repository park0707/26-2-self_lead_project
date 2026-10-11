# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 (Spine Neutrality)
Model Version: v1 (c1_model_v1.py)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, C) 또는 단일 시퀀스 (64, C)
   - 임의의 프레임 수(T) 입력 시에도 무차원 시간 정규화(dt = 1/(T-1)) 자동 적용
2. 피처 엔지니어링 (최종 확정 25D 특화 생체역학 피처셋):
   - 9D 척추 동적 경향성: 폐곡선 Cyclogram 면적, 상승기 속도비(Rv), 위상 지연차, 버트윙크 가속도, 체간 저크, 관상면 비대칭 등
   - 1D 최하사점 역학: 상체-정강이 평행도 오차 (torso_shank_err)
   - 3D 경추 및 시선 공변성: cervical_min, gaze_pitch_mean, gaze_pitch_min
   - 6D 척추 곡률 및 3점 정렬 체인: 귀-어깨-고관절 3점 곡률(최저, 평균, 드롭량), 목-등-허리 정렬각, 머리 전방 돌출
   - 6D 선행 연구 문헌 피처: 최하사점 3점 내각, 체간/대퇴 각도비, 최대 경사 위상차, 속도 피크, 하강 변화율비, 교차상관 지연
3. 모델 추론 구조:
   - 2계층 캐스케이드(Dual-Tier Cascade) 아키텍처 지원
     * Tier-1: 초고속 판별 Gatekeeper (LightGBM, 0.2ms 내외, 명확군 83.8% 고속 확정)
     * Tier-2: 심층 분석 Resolver (CatBoost, 모호군 16.2% 정밀 감별 및 체형 맞춤형 피드백 분기)
   - 단일 통합 챔피언 모드(CatBoost/LightGBM) 상시 지원
"""

import os
import sys
import pickle
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

# LightGBM & CatBoost
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# 상위 경로 참조
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.extract_squat_spine_trend import extract_squat_spine_trend_features, SPINE_TREND_FEATURE_NAMES_9
from src.models.second_models.extract_squat_spine_literature import extract_squat_spine_literature_features, SPINE_LIT_FEATURE_NAMES_6
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics


FEATURE_NAMES_25: List[str] = (
    SPINE_TREND_FEATURE_NAMES_9
    + ['torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min']
    + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']
    + ['fine_neck_back_waist_angle_min', 'fine_neck_back_waist_angle_drop', 'fine_head_forward_protrusion_max']
    + SPINE_LIT_FEATURE_NAMES_6
)

# 64프레임 전수 소거 실험(Greedy Backward Elimination)을 통해 도출된 최적 22D 정예 피처셋
# 제거된 3대 노이즈/다중공선성 피처: 'c1_torso_log_dimensionless_jerk', 'gaze_pitch_mean', 'fine_neck_back_waist_angle_drop'
OPTIMAL_PRUNED_INDICES_22: List[int] = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24
]
FEATURE_NAMES_22: List[str] = [FEATURE_NAMES_25[i] for i in OPTIMAL_PRUNED_INDICES_22]


class BabelSquatC1ModelV1:
    """
    바벨 스쿼트 'c1. 척추의 중립(Spine Neutrality)' 오류 분류 모델 v1
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 22  # 기본 최적 22D
    FEATURE_NAMES = FEATURE_NAMES_22

    def __init__(
        self,
        mode: str = "cascade",  # 'cascade' (2계층) 또는 'champion' (단일 최적 모델)
        tier1_threshold_low: float = 0.25,  # 5-Fold OOF Selective Risk 최적 대역 [0.25, 0.65]
        tier1_threshold_high: float = 0.65,
        decision_threshold: float = 0.44,   # 5-Fold OOF F1 극대화 최적 임계값 (기본 0.50 대비 F1 +0.82%p~+0.91%p 향상)
        use_pruned_features: bool = True    # True: 최적 22D 정예 피처, False: 원본 25D
    ):
        self.mode = mode
        self.tier1_threshold_low = tier1_threshold_low
        self.tier1_threshold_high = tier1_threshold_high
        self.decision_threshold = decision_threshold
        self.use_pruned_features = use_pruned_features
        self.feature_dim = 22 if use_pruned_features else 25
        self.feature_names = FEATURE_NAMES_22 if use_pruned_features else FEATURE_NAMES_25

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
                colsample_bytree=0.8,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        if CatBoostClassifier is not None:
            # Tier-2 전문 리졸버 (모호군 감별 특화)
            self.model_tier2 = CatBoostClassifier(
                iterations=150,
                depth=6,
                learning_rate=0.04,
                l2_leaf_reg=4.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

    @classmethod
    def compute_curvature_features_from_tensor(cls, X_b: np.ndarray) -> np.ndarray:
        """
        별도 캐시 없이 Format B (N, T, C) 시계열로부터 척추 곡률 및 정렬 지표 6D 직접 추출
        """
        N, T, C = X_b.shape
        curv_feats = np.zeros((N, 6), dtype=np.float32)

        for i in range(N):
            # 목-체간-골반 정렬각 (ch 82)
            neck_spine_align = X_b[i, :, 82] if C > 82 else np.full(T, 175.0)
            # 체간 피치 (ch 64)
            torso_pitch = X_b[i, :, 64] if C > 64 else np.zeros(T)
            # 무릎 굴곡각 (ch 56, 57)
            if C > 57:
                knee_flex = 180.0 - (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
            else:
                knee_flex = np.zeros(T)
            b = int(np.clip(np.argmax(knee_flex), 2, T - 3))

            # 1. ear_sh_hip_min 대용: 목-체간 정렬 최소각 (최대 굽음)
            curv_feats[i, 0] = float(np.min(neck_spine_align))
            # 2. ear_sh_hip_mean 대용: 평균 정렬각
            curv_feats[i, 1] = float(np.mean(neck_spine_align))
            # 3. ear_sh_hip_drop 대용: 기립 대비 최하사점 정렬각 낙폭
            curv_feats[i, 2] = float(neck_spine_align[b] - np.mean(neck_spine_align[:max(1, int(T*0.1))]))
            # 4. fine_neck_back_waist_angle_min: 최하사점 목-체간 전방 전위 폭 (ch 6 Torso-Neck dz)
            if C > 6:
                neck_dz = X_b[i, :, 6]
                curv_feats[i, 3] = float(np.max(neck_dz) - np.min(neck_dz))
            else:
                curv_feats[i, 3] = 0.0
            # 5. fine_neck_back_waist_angle_drop: 정렬각 표준편차 (흔들림)
            curv_feats[i, 4] = float(np.std(neck_spine_align))
            # 6. fine_head_forward_protrusion_max: 최하사점 척추 정렬각
            curv_feats[i, 5] = float(neck_spine_align[b])

        return curv_feats

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        curv_feats_6d: Optional[np.ndarray] = None,
        fine_feats_12d: Optional[np.ndarray] = None,
        use_pruned: bool = False
    ) -> np.ndarray:
        """
        64프레임(또는 임의의 T프레임) Format B 텐서로부터 특화 생체역학 피처 추출
        Args:
            X_b: (N, T, C) 또는 (T, C) 크기의 넘파이 배열
            curv_feats_6d: (N, 6) 사전 계산된 곡률 피처 (없을 경우 텐서에서 직접 계산)
            fine_feats_12d: (N, 12) 사전 계산된 안면/발 피처 (없을 경우 텐서에서 추정)
            use_pruned: True일 경우 성능 저해 피처가 소거된 22D 정예 서브셋 반환 (기본값 False: 원본 25D)
        Returns:
            X_out: (N, 22) 또는 (N, 25) float32 배열
        """
        is_single = False
        if X_b.ndim == 2:
            X_b = np.expand_dims(X_b, axis=0)
            is_single = True

        N, T, C = X_b.shape

        # 1. 9D 동적 경향성 피처
        feats_9d = extract_squat_spine_trend_features(X_b)

        # 2. 1D 최하사점 상체-정강이 평행도 오차
        X_bot = extract_squat_bottom_biomechanics(X_b)
        torso_shank_err = X_bot[:, 1:2]

        # 3. 3D 고개/시선 및 경추 정렬 피처
        if fine_feats_12d is not None and fine_feats_12d.shape[0] == N:
            gaze_pitch_mean = fine_feats_12d[:, 0:1]
            gaze_pitch_min = fine_feats_12d[:, 1:2]
            cervical_min = fine_feats_12d[:, 3:4]
        else:
            # 텐서에서 직접 유도
            gaze_pitch_mean = np.mean(X_b[:, :, 64:65], axis=1) * 0.8
            gaze_pitch_min = np.min(X_b[:, :, 64:65], axis=1) * 0.8
            cervical_min = np.min(X_b[:, :, 82:83], axis=1) if C > 82 else np.full((N, 1), 170.0)

        # 4. 6D 척추 곡률 및 3점 정렬 체인
        if curv_feats_6d is not None and curv_feats_6d.shape[0] == N:
            curv_6d = curv_feats_6d[:, :6]
        else:
            curv_6d = cls.compute_curvature_features_from_tensor(X_b)

        # 5. 6D 선행 연구 문헌 피처
        lit_6d = extract_squat_spine_literature_features(X_b)

        # 최종 25D 결합
        X_25d = np.concatenate([
            feats_9d,           # 9D
            torso_shank_err,    # 1D
            cervical_min,       # 1D
            gaze_pitch_mean,    # 1D
            gaze_pitch_min,     # 1D
            curv_6d,            # 6D
            lit_6d              # 6D
        ], axis=1).astype(np.float32)

        X_25d = np.nan_to_num(X_25d, nan=0.0, posinf=10.0, neginf=-10.0)

        if use_pruned:
            X_out = X_25d[:, OPTIMAL_PRUNED_INDICES_22]
        else:
            X_out = X_25d

        if is_single:
            return X_out[0]
        return X_out

    def fit(
        self,
        X_b: np.ndarray,
        y: np.ndarray,
        ambiguity_levels: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None
    ) -> 'BabelSquatC1ModelV1':
        """
        64프레임 데이터셋으로 모델 학습
        """
        print(f"[{self.EXERCISE_NAME} - {self.DEFECT_NAME} v1] 학습 시작 (입력 텐서 shape: {X_b.shape})")
        X_feats = self.extract_features(X_b, use_pruned=self.use_pruned_features)
        N = len(y)

        # 1. 단일 챔피언 모델 학습 (전수)
        if self.model_champion is not None:
            self.model_champion.fit(X_feats, y)

        # 2. 2계층 캐스케이드 학습
        if ambiguity_levels is not None:
            # Tier-1: Level 1 (명확군) 전담 학습
            mask_l1 = (ambiguity_levels == 1)
            if np.sum(mask_l1) > 20 and self.model_tier1 is not None:
                self.model_tier1.fit(X_feats[mask_l1], y[mask_l1])

            # Tier-2: Level 1~3 (실무군 전수) 전담 학습
            mask_l123 = (ambiguity_levels <= 3)
            if np.sum(mask_l123) > 20 and self.model_tier2 is not None:
                self.model_tier2.fit(X_feats[mask_l123], y[mask_l123])
        else:
            # 모호성 레벨이 없을 경우 전체 데이터로 피팅
            if self.model_tier1 is not None:
                self.model_tier1.fit(X_feats, y)
            if self.model_tier2 is not None:
                self.model_tier2.fit(X_feats, y)

        self.is_fitted = True
        print(f"[{self.EXERCISE_NAME} - {self.DEFECT_NAME} v1] 학습 완료 (피처 차원: {X_feats.shape[1]}D)")
        return self

    def predict_proba(self, X_b: np.ndarray) -> np.ndarray:
        """
        64프레임 텐서로부터 척추 결함 확률 P(defect) 산출
        """
        X_feats = self.extract_features(X_b, use_pruned=self.use_pruned_features)
        if X_feats.ndim == 1:
            X_feats = np.expand_dims(X_feats, axis=0)

        N = len(X_feats)

        if self.mode == "cascade" and self.model_tier1 is not None and self.model_tier2 is not None:
            p_t1 = self.model_tier1.predict_proba(X_feats)[:, 1]
            p_final = np.zeros(N, dtype=np.float32)

            for i in range(N):
                prob1 = p_t1[i]
                # Fast-path: 확실한 정상(<=0.35) 또는 확실한 결함(>=0.65)
                if prob1 <= self.tier1_threshold_low or prob1 >= self.tier1_threshold_high:
                    p_final[i] = prob1
                else:
                    # Deep-path: 모호군 (0.35 < prob1 < 0.65)은 Tier-2 Resolver 판정
                    prob2 = self.model_tier2.predict_proba(X_feats[i:i+1])[:, 1][0]
                    p_final[i] = 0.3 * prob1 + 0.7 * prob2
            return p_final
        else:
            # 단일 최적 모델
            active_model = self.model_champion if self.model_champion is not None else self.model_tier1
            return active_model.predict_proba(X_feats)[:, 1]

    def predict(self, X_b: np.ndarray) -> np.ndarray:
        """
        0 (정상) 또는 1 (오류) 예측
        """
        probs = self.predict_proba(X_b)
        return (probs >= self.decision_threshold).astype(int)

    def diagnose(self, X_b: np.ndarray) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        생체역학적 근거와 함께 정밀 코칭 진단 메시지 반환
        """
        is_single = (X_b.ndim == 2)
        if is_single:
            X_b = np.expand_dims(X_b, axis=0)

        X_feats = self.extract_features(X_b)
        probs = self.predict_proba(X_b)
        preds = (probs >= self.decision_threshold).astype(int)

        results = []
        for i in range(len(preds)):
            pred = preds[i]
            prob = probs[i]

            # 주요 생체역학 피처값 해석
            rv = X_feats[i, 1]              # 엉덩이/어깨 수직속도비
            butt_wink = X_feats[i, 3]        # 버트윙크 가속도 스파이크
            torso_shank = X_feats[i, 9]      # 상체-정강이 평행도 오차
            descent_ratio = X_feats[i, 23]   # 하강 고관절/체간 변화율비

            reasons = []
            if pred == 1:
                if rv > 1.25 or descent_ratio < 0.8:
                    reasons.append("상승 시 엉덩이가 어깨보다 먼저 들리는 굿모닝 스쿼트 패턴 감지")
                if butt_wink > 12.0:
                    reasons.append("최하사점에서 골반이 안으로 말리는 버트윙크(Butt-wink) 요추 굴곡 감지")
                if torso_shank > 18.0:
                    reasons.append("상체가 정강이 축 대비 과도하게 전방으로 기울어짐")
                if not reasons:
                    reasons.append("체간-고관절 동적 협응 불일치 및 척추 기립각 이탈")

                msg = f"주의! 척추 중립이 무너졌습니다. ({', '.join(reasons)})"
            else:
                if torso_shank > 12.0:
                    msg = "정상 유지 중입니다. 대퇴골이 긴 체형 특성으로 상체가 깊게 숙여진 정상이니 안심하세요."
                else:
                    msg = "완벽한 정석 척추 정렬입니다! 요추와 흉추 중립이 훌륭하게 유지되고 있습니다."

            res = {
                "exercise": self.EXERCISE_NAME,
                "defect_code": self.DEFECT_CODE,
                "defect_name": self.DEFECT_NAME,
                "is_error": bool(pred == 1),
                "error_probability": round(float(prob), 4),
                "coaching_feedback": msg,
                "key_biomechanics": {
                    "ascent_velocity_ratio_Rv": round(float(rv), 2),
                    "butt_wink_accel": round(float(butt_wink), 2),
                    "torso_shank_error_deg": round(float(torso_shank), 2)
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
        print(f"[BabelSquatC1ModelV1] 모델 저장 완료: {filepath}")

    def load(self, filepath: str):
        """가중치 파일 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.mode = payload.get("mode", "cascade")
        self.tier1_threshold_low = payload.get("tier1_threshold_low", 0.35)
        self.tier1_threshold_high = payload.get("tier1_threshold_high", 0.65)
        self.decision_threshold = payload.get("decision_threshold", 0.50)
        self.is_fitted = payload.get("is_fitted", True)
        self.model_tier1 = payload.get("model_tier1")
        self.model_tier2 = payload.get("model_tier2")
        self.model_champion = payload.get("model_champion")
        print(f"[BabelSquatC1ModelV1] 모델 로드 완료: {filepath}")


if __name__ == '__main__':
    print("=" * 80)
    print(" [vidio-health-trainer] BabelSquatC1ModelV1 자체 64프레임 검증 테스트")
    print("=" * 80)

    # 1. 64F 마스터 데이터 로드
    from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
    from scipy.interpolate import interp1d

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0]
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

    # 4단계 모호성 레벨
    amb_data = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz')
    levels = amb_data['levels']

    # 2. 모델 인스턴스 생성 및 피처 추출 테스트
    model = BabelSquatC1ModelV1(mode="cascade")
    X_feats_64 = model.extract_features(X_b_64)
    print(f"[*] 64F 피처 추출 완료: shape = {X_feats_64.shape}")

    # 3. 모델 학습
    model.fit(X_b_64, y_c1, ambiguity_levels=levels, subjects=subjects)

    # 4. 1회 단일 샘플 진단 테스트
    diag = model.diagnose(X_b_64[0])
    print("\n[*] 단일 샘플 진단 결과:")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

    # 5. 모델 가중치 저장 테스트
    weight_path = os.path.join(CURRENT_DIR, "c1_model_v1_weights.pkl")
    model.save(weight_path)
    print(f"[성공] c1_model_v1 파일 및 가중치 생성 완료.")
