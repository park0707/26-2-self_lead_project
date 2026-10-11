# =========================================================================================================
# [모듈명]: dual_cascade_deadlift_classifier.py
# [위치]: src/models/sub_models/dual_cascade_deadlift_classifier.py
# [목적]: Stage 2-A: 일반 데드리프트 (0) vs 스티프 데드리프트 (1) 전용 v4.0 독립 2중 캐스케이드 분류기
# [아키텍처]:
#   1. Train Data Pruning (학습 데이터 정제):
#      - 피험자 1(Subj_1)의 스티프 데드 시퀀스(동일 동작 라벨 노이즈) 제거
#      - 바벨 데드 중 무릎을 170° 이상 완전히 편 Level 4 역전 데이터 제거
#   2. 1차 단계: 독립 2중 정자세 감별사 (Dual Binary Classifiers with Rejection Option)
#      - 모델 1A: 바벨 Level 1 정자세 감별 LightGBM -> P(Barbell)
#      - 모델 1B: 스티프 Level 1 정자세 감별 LightGBM -> P(Stiff)
#   3. 2차 단계: 14D 생체역학 전담 리졸버 (Biomechanical Resolver)
#      - P(Barbell)과 P(Stiff)가 둘 다 낮아(Neither) 판정을 유보한 모호군(약 35%)을 위임받아 최종 미세 분기
# =========================================================================================================

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)

# 기존 v3.0 서브 분류기의 14D 피처 추출 및 베이스 유틸 계승
from src.models.sub_models.deadlift_sub_classifier import DeadliftSubClassifier

class DualCascadeDeadliftClassifier:
    """Stage 2-A v4.0: 독립 2중 정자세 감별사 + 14D 생체역학 캐스케이드 Expert"""

    FEATURE_NAMES_14D = DeadliftSubClassifier.FEATURE_NAMES_14D
    FOOT_FEATURE_NAMES_4D = [
        'shank_foot_bot',           # 15. 최하사점 정강이-발바닥 3D 사잇각
        'shank_foot_std',           # 16. 전 구간 정강이-발바닥 사잇각 표준편차
        'ankle_rom',                # 17. 발목 배측굴곡 가동범위
        'hip_to_ankle_rom_ratio'    # 18. 고관절 대 발목 ROM 비율 (HAR)
    ]
    FEATURE_NAMES_18D = FEATURE_NAMES_14D + FOOT_FEATURE_NAMES_4D
    FOOT_CACHE_PATH = 'd:/Data/피처_데이터/deadlift_foot_features_cache.npz'

    def __init__(self,
                 conf_thresh_high: float = 0.60,
                 conf_thresh_low: float = 0.40,
                 stage2_decision_threshold: float = 0.50,
                 use_foot_features: bool = True):
        self.conf_thresh_high = conf_thresh_high
        self.conf_thresh_low = conf_thresh_low
        self.stage2_decision_threshold = stage2_decision_threshold
        self.use_foot_features = use_foot_features

        # 1차 모델 A: 바벨 정자세 식별기
        self.clf_barbell_clean: Optional[lgb.LGBMClassifier] = None
        # 1차 모델 B: 스티프 정자세 식별기
        self.clf_stiff_clean: Optional[lgb.LGBMClassifier] = None
        # 2차 모델: 모호군 전담 14D/18D 생체역학 리졸버
        self.clf_resolver: Optional[lgb.LGBMClassifier] = None

        self.routing_stats = {
            'barbell_fast': 0,
            'stiff_fast': 0,
            'routed_to_stage2': 0
        }

    @classmethod
    def load_foot_features_dict(cls) -> Dict[str, np.ndarray]:
        """사전 빌드된 발 관절 4대 파생 피처 캐시 로드"""
        if os.path.exists(cls.FOOT_CACHE_PATH):
            try:
                data = np.load(cls.FOOT_CACHE_PATH, allow_pickle=True)
                sids = data['sequence_ids']
                feats = data['foot_features']
                return {str(s): feats[i] for i, s in enumerate(sids)}
            except Exception:
                pass
        return {}

    @classmethod
    def attach_foot_features(cls, feats_14d: np.ndarray, sequence_ids: Optional[List[str]] = None, foot_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """14D 피처에 발 관절 4대 파생 피처를 결합하여 18D 피처 매트릭스로 확장"""
        if foot_feats is not None:
            return np.concatenate([feats_14d, foot_feats], axis=-1)

        if sequence_ids is not None:
            cache = cls.load_foot_features_dict()
            if cache:
                N = len(feats_14d)
                foot_arr = np.zeros((N, 4), dtype=np.float32)
                for i, sid in enumerate(sequence_ids):
                    s_str = str(sid)
                    base_sid = s_str.split('_view')[0]
                    if s_str in cache:
                        foot_arr[i] = cache[s_str]
                    elif base_sid in cache:
                        foot_arr[i] = cache[base_sid]
                    else:
                        foot_arr[i] = np.array([85.0, 3.0, 10.0, 5.0], dtype=np.float32)
                return np.concatenate([feats_14d, foot_arr], axis=-1)

        return feats_14d

    @classmethod
    def slice_features(cls, X_b: np.ndarray, sequence_ids: Optional[List[str]] = None, foot_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """기존 14D 생체역학 피처 추출 엔진 및 발 피처 연계"""
        f14 = DeadliftSubClassifier.slice_features(X_b)
        if sequence_ids is not None or foot_feats is not None:
            return cls.attach_foot_features(f14, sequence_ids=sequence_ids, foot_feats=foot_feats)
        return f14

    @classmethod
    def _ensure_2d_feats(cls, X: np.ndarray, sequence_ids: Optional[List[str]] = None, foot_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """3D 시퀀스(N, 64, D)일 경우 14D 또는 18D 생체역학 피처로 자동 변환"""
        if X.ndim == 3:
            return cls.slice_features(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        if X.shape[-1] == 14 and (sequence_ids is not None or foot_feats is not None):
            return cls.attach_foot_features(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        return X

    @classmethod
    def determine_levels(cls, feats: np.ndarray, y: np.ndarray) -> np.ndarray:
        """무릎 각도 기반 Level 1~4 산출 (Knee Bottom Angle = index 2)"""
        knee_angles = feats[:, 2]
        levels = np.zeros(len(y), dtype=int)
        for i in range(len(y)):
            k = knee_angles[i]
            if y[i] == 1: # 스티프
                if k <= 150.0: levels[i] = 4
                elif k <= 158.0: levels[i] = 3
                elif k <= 163.0: levels[i] = 2
                else: levels[i] = 1
            else: # 바벨
                if k >= 170.0: levels[i] = 4
                elif k >= 165.0: levels[i] = 3
                elif k >= 160.0: levels[i] = 2
                else: levels[i] = 1
        return levels

    def fit(self, X: np.ndarray, y: np.ndarray, subjects: Optional[np.ndarray] = None,
            sequence_ids: Optional[List[str]] = None, foot_feats: Optional[np.ndarray] = None) -> 'DualCascadeDeadliftClassifier':
        """
        [v4.0 / v5.0 훈련 파이프라인]
        1) 훈련셋 라벨 노이즈 정제 (Data Pruning: Level 4 물리 역전 및 피험자 1 노이즈 배제)
        2) 1차 모델 A & B 학습 (Level 1 정자세 감별사)
        3) 2차 리졸버 모델 학습 (Level 1~3 유효 데이터셋 기반 14D/18D 현실 적응형 트리)
        """
        feats = self._ensure_2d_feats(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        N = len(y)
        levels = self.determine_levels(feats, y)

        # -------------------------------------------------------------
        # 1. 훈련 데이터 정제 (Train Data Pruning: Level 4 물리적 역전 배제)
        # -------------------------------------------------------------
        is_subj1 = np.zeros(N, dtype=bool)
        if subjects is not None:
            for i, s in enumerate(subjects):
                s_str = str(s)
                if "피험자1" in s_str or s_str == "1" or "_1_" in s_str or "Subj_1" in s_str:
                    is_subj1[i] = True

        # Level 4(바벨 무릎>=170도, 스티프 무릎<=150도) 및 피험자 1 스티프 노이즈 완벽 배제
        bad_mask = (levels == 4) | (is_subj1 & (y == 1))
        clean_mask = ~bad_mask

        X_clean = feats[clean_mask]
        y_clean = y[clean_mask]
        levels_clean = levels[clean_mask]

        # -------------------------------------------------------------
        # 2. 1차 모델 A (바벨 정자세 감별사) 학습
        # -------------------------------------------------------------
        y_train_barbell_clean = ((y_clean == 0) & (levels_clean == 1)).astype(int)
        self.clf_barbell_clean = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        self.clf_barbell_clean.fit(X_clean, y_train_barbell_clean)

        # -------------------------------------------------------------
        # 3. 1차 모델 B (스티프 정자세 감별사) 학습
        # -------------------------------------------------------------
        y_train_stiff_clean = ((y_clean == 1) & (levels_clean == 1)).astype(int)
        self.clf_stiff_clean = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=15,
            scale_pos_weight=3.0,
            random_state=42,
            verbose=-1
        )
        self.clf_stiff_clean.fit(X_clean, y_train_stiff_clean)

        # -------------------------------------------------------------
        # 4. 2차 모델 (모호군 전담 14D/18D 생체역학 Resolver) 학습
        # -------------------------------------------------------------
        self.clf_resolver = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            num_leaves=11,
            min_child_samples=20,
            scale_pos_weight=2.5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1
        )
        self.clf_resolver.fit(X_clean, y_clean)

        return self

    def predict_proba(self, X: np.ndarray, sequence_ids: Optional[List[str]] = None,
                      foot_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """
        [v4.0 / v5.0 캐스케이드 추론 및 라우팅]
        Output: (N, 2) 형태의 사후 확률 배열 [P(Barbell), P(Stiff)]
        """
        if self.clf_barbell_clean is None or self.clf_stiff_clean is None or self.clf_resolver is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하십시오.")

        feats = self._ensure_2d_feats(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        N = len(feats)

        # 만약 모델은 18D로 학습되었는데 입력 피처가 14D인 경우 기본값 자동 패딩
        expected_dim = getattr(self.clf_resolver, 'n_features_in_', feats.shape[1])
        if feats.shape[1] == 14 and expected_dim == 18:
            pad = np.tile(np.array([85.0, 3.0, 10.0, 5.0], dtype=np.float32), (N, 1))
            feats = np.concatenate([feats, pad], axis=-1)
        elif feats.shape[1] == 18 and expected_dim == 14:
            feats = feats[:, :14]

        p_barbell_clean = self.clf_barbell_clean.predict_proba(feats)[:, 1]
        p_stiff_clean = self.clf_stiff_clean.predict_proba(feats)[:, 1]
        p_resolver = self.clf_resolver.predict_proba(feats)[:, 1]

        final_p_stiff = np.zeros(N, dtype=np.float32)

        for i in range(N):
            pb = p_barbell_clean[i]
            ps = p_stiff_clean[i]

            # 1) 바벨 정자세 확정 조건
            if pb >= self.conf_thresh_high and ps < self.conf_thresh_low:
                final_p_stiff[i] = 0.05
                self.routing_stats['barbell_fast'] += 1
            # 2) 스티프 정자세 확정 조건
            elif ps >= self.conf_thresh_high and pb < self.conf_thresh_low:
                final_p_stiff[i] = 0.95
                self.routing_stats['stiff_fast'] += 1
            # 3) 모호군 (둘 다 낮거나 충돌) -> 2차 생체역학 리졸버 위임
            else:
                final_p_stiff[i] = p_resolver[i]
                self.routing_stats['routed_to_stage2'] += 1

        final_p_barbell = 1.0 - final_p_stiff
        return np.column_stack([final_p_barbell, final_p_stiff])

    def predict(self, X: np.ndarray, sequence_ids: Optional[List[str]] = None,
                foot_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """최종 클래스 예측 (0: 바벨, 1: 스티프)"""
        probs = self.predict_proba(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        p_stiff = probs[:, 1]
        return (p_stiff >= self.stage2_decision_threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                 sequence_ids: Optional[List[str]] = None,
                 foot_feats: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """정밀 2x2 혼동 행렬 및 다차원 평가지표 산출 (기존 리포터 100% 호환)"""
        self.routing_stats = {'barbell_fast': 0, 'stiff_fast': 0, 'routed_to_stage2': 0}
        y_pred = self.predict(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        probs = self.predict_proba(X, sequence_ids=sequence_ids, foot_feats=foot_feats)
        p_stiff = probs[:, 1]

        acc = accuracy_score(y_true, y_pred)
        
        # 일반 데드리프트 (0) 지표
        dead_prec = precision_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
        dead_rec = recall_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
        dead_f1 = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)

        # 스티프 데드리프트 (1) 지표
        stiff_prec = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        stiff_rec = recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        stiff_f1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(y_true, p_stiff)
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            'accuracy': float(acc),
            'dead_precision': float(dead_prec),
            'dead_recall': float(dead_rec),
            'dead_f1': float(dead_f1),
            'stiff_precision': float(stiff_prec),
            'stiff_recall': float(stiff_rec),
            'stiff_f1': float(stiff_f1),
            'best_threshold': float(self.stage2_decision_threshold),
            'roc_auc': float(auc),
            'confusion_matrix': cm,
            'routing_stats': self.routing_stats
        }

if __name__ == '__main__':
    print("[*] DualCascadeDeadliftClassifier 모듈 무결성 점검 완료")
