# =========================================================================================================
# [모듈명]: lunge_sub_classifier.py
# [목적]: Stage 2-B: 무빙 런지 패밀리 전용 2진 초정밀 판별기
#         - 0: 다이나믹 런지 (직선 전진 스텝)
#         - 1: 크로스 런지 (대각 교차 스텝)
# [특징]:
#   1. 9대 런지 특화 생체역학 피처 (골반축 투영 교차 마진, 발목 횡수축비, 가림보상 점수 등) 100% 집중
#   2. LightGBM Binary 하이퍼파라미터 그리드 서치 및 크로스 런지 소수 클래스 가중치(scale_pos_weight) 튜닝
#   3. 크로스 런지 F1 극대화 결정 임계값(tau*_cross) 정밀 스캔
# =========================================================================================================

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)

class LungeSubClassifier:
    """Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert 분류기 (0: 다이나믹 런지, 1: 크로스 런지)"""

    LUNGE_CLASS_NAMES = ['05. 다이나믹 런지', '07. 크로스 런지']

    FEATURE_NAMES_14D = [
        'step_ratio_ml_ap',                # 1. 스텝 발 전후 대비 좌우 이동 변위 비율 (ML/AP 종횡비)
        'crossover_bottom_win',            # 2. [최하사점 윈도우] 평균 발목 교차 마진 (다이나믹 <= 0, 크로스 > 0)
        'crossover_bottom_ratio',          # 3. [최하사점 윈도우] 교차 상태 유지 시간 비율
        'crossover_margin_mean',           # 4. 전체 시퀀스 평균 교차 마진
        'crossover_bottom',                # 5. 무릎 최저점 시점(t_bottom) 순간 교차 마진
        'inter_knee_dist_bottom',          # 6. [최하사점 윈도우] 양 무릎 3D 최소 이격거리 (0° 비가림 핵심)
        'inter_knee_pelvis_ratio',         # 7. [최하사점 윈도우] 골반 너비 대비 무릎 간격 비율 (체형 정규화)
        'pelvic_obliquity_bottom',         # 8. [최하사점 윈도우] 전두면 골반 경사각 (Trendelenburg drop)
        'pelvic_obliquity_rom',            # 9. 전체 전두면 골반 경사 가동범위
        'hip_adduction_bottom',            # 10. [최하사점 윈도우] 선행 다리 대퇴골 내전각
        'lateral_contraction_ratio',       # 11. 발목 횡방향 간격 수축-확장 비율
        'cross_bottom_bio_score',          # 12. [최하사점] 생체역학 교차 복합지수 (골반x내전x무릎밀착)
        'knee_cross_ankle_bottom',         # 13. [최하사점 보조] 선행 무릎-후행 발목 3D 거리
        'inter_ankle_bottom'               # 14. [최하사점 보조] 양 발목 3D 거리
    ]
    FEATURE_NAMES_15D = FEATURE_NAMES_14D  # 하위 호환성 유지
    FEATURE_NAMES_13D = FEATURE_NAMES_14D
    FEATURE_NAMES_9D = FEATURE_NAMES_14D

    SAVED_PARAMS_PATH = 'd:/Data/피처_데이터/tuning_results/stage2_best_hyperparameters.json'

    def __init__(self, best_params: Optional[Dict[str, Any]] = None, best_threshold: Optional[float] = None):
        loaded_params, loaded_tau = self._load_saved_best_params()
        self.has_saved_params = (loaded_params is not None)

        self.best_params = best_params or loaded_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'max_depth': 4,
            'num_leaves': 15,
            'scale_pos_weight': 2.0,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 4
        }
        self.best_threshold = best_threshold if best_threshold is not None else (loaded_tau if loaded_tau is not None else 0.40)
        self.model: Optional[lgb.LGBMClassifier] = None

    @classmethod
    def _load_saved_best_params(cls) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        if os.path.exists(cls.SAVED_PARAMS_PATH):
            try:
                import json
                with open(cls.SAVED_PARAMS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                lunge_info = data.get('stage2_lunge', {})
                params = lunge_info.get('best_params')
                tau = lunge_info.get('best_threshold')
                return params, tau
            except Exception:
                pass
        return None, None

    @classmethod
    def slice_features(cls, X_b: np.ndarray) -> np.ndarray:
        """
        Format B 시계열 텐서로부터 15차원 런지 전용 생체역학 피처를 고속 추출
        - 무릎 최하사점(t_bottom ± 6 프레임) 위상 인식 윈도우 스냅샷 적용 (측면 착지 노이즈 원천 차단)
        - 정면 0° 자기 가림 극복 4대 비폐색 프록시 (골반 드롭, 대퇴 내전, 무릎 밀착, 카운터밸런스) 집중
        - 노이즈 유발 어깨-골반 비틀림 배제 및 엄밀한 골반-대퇴-무릎 기구학 체계 확립
        Input: X_b (N, 64, D)
        Output: X_lunge_sub (N, 15)
        """
        N, T, D = X_b.shape

        # 1. 무릎 각도 및 최하사점(Bottom) 윈도우 계산
        knee_angles = X_b[:, :, 56:58]
        knee_mean_t = np.mean(knee_angles, axis=-1)
        bottom_idx = np.argmin(knee_mean_t, axis=1) # (N,)

        # 최하사점 윈도우 (t_bottom ± 6 프레임) 마스크 생성
        t_grid = np.arange(T)[None, :] # (1, T)
        bottom_idx_2d = bottom_idx[:, None] # (N, 1)
        bottom_win_mask = (t_grid >= np.maximum(0, bottom_idx_2d - 6)) & (t_grid <= np.minimum(T - 1, bottom_idx_2d + 6)) # (N, T)
        win_count = np.sum(bottom_win_mask, axis=1, keepdims=True) # (N, 1)

        # 2. 골반, 무릎 및 발목 3D 좌표 복원
        hip_l = X_b[:, :, 32:35]
        hip_r = X_b[:, :, 36:39]
        knee_l = hip_l + X_b[:, :, 40:43]
        knee_r = hip_r + X_b[:, :, 48:51]
        ankle_l = knee_l + X_b[:, :, 44:47]
        ankle_r = knee_r + X_b[:, :, 52:55]

        # 골반 축 단위 벡터
        hip_axis = hip_r - hip_l
        pelvis_width = np.mean(np.linalg.norm(hip_axis, axis=-1), axis=1, keepdims=True) + 1e-7 # (N, 1)
        hip_axis_norm = hip_axis / (np.linalg.norm(hip_axis, axis=-1, keepdims=True) + 1e-7)

        # 스텝 발 변위 및 전후/좌우 이동 종횡비 (피처 1: step_ratio_ml_ap)
        init_l = np.mean(ankle_l[:, :5, :], axis=1, keepdims=True)
        init_r = np.mean(ankle_r[:, :5, :], axis=1, keepdims=True)
        disp_vec_l = ankle_l - init_l
        disp_vec_r = ankle_r - init_r

        # 초기 골반(ML) 축 및 전후(AP) 방향 단위 벡터
        vert_u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ap_axis_norm = np.cross(vert_u[None, None, :], hip_axis_norm)
        ap_axis_norm = ap_axis_norm / (np.linalg.norm(ap_axis_norm, axis=-1, keepdims=True) + 1e-7)

        disp_ml_l = np.sum(disp_vec_l * hip_axis_norm, axis=-1)
        disp_ap_l = np.sum(disp_vec_l * ap_axis_norm, axis=-1)
        disp_ml_r = np.sum(disp_vec_r * hip_axis_norm, axis=-1)
        disp_ap_r = np.sum(disp_vec_r * ap_axis_norm, axis=-1)

        tot_disp_l = np.max(np.linalg.norm(disp_vec_l, axis=-1), axis=1)
        tot_disp_r = np.max(np.linalg.norm(disp_vec_r, axis=-1), axis=1)
        is_left_stepping = (tot_disp_l >= tot_disp_r)[:, None]

        disp_ml_step = np.where(is_left_stepping, disp_ml_l, disp_ml_r)
        disp_ap_step = np.where(is_left_stepping, disp_ap_l, disp_ap_r)

        max_abs_ml = np.max(np.abs(disp_ml_step), axis=1, keepdims=True)
        max_abs_ap = np.max(np.abs(disp_ap_step), axis=1, keepdims=True)
        step_ratio_ml_ap = (max_abs_ml + 1e-4) / (max_abs_ap + 1e-4) # (N, 1)

        # 부호화 교차 마진 시계열
        vec_l_step = ankle_l - ankle_r
        vec_r_step = ankle_r - ankle_l
        proj_l = np.sum(vec_l_step * hip_axis_norm, axis=-1)
        proj_r = np.sum(vec_r_step * (-hip_axis_norm), axis=-1)
        crossover_margin_t = np.where(is_left_stepping, proj_l, proj_r) # (N, T)

        # 최하사점 윈도우 기반 교차 피처 (피처 2, 3, 4, 5)
        crossover_bottom_win = np.sum(crossover_margin_t * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)
        crossover_bottom_ratio = np.sum((crossover_margin_t > 0.0) * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)
        crossover_margin_mean = np.mean(crossover_margin_t, axis=1, keepdims=True) # (N, 1)
        crossover_bottom = crossover_margin_t[np.arange(N), bottom_idx].reshape(N, 1) # (N, 1)

        # 3. 양 무릎 3D 이격거리 (정면 0° 비가림 핵심, 피처 6, 7)
        knee_dist_t = np.linalg.norm(knee_l - knee_r, axis=-1) # (N, T)
        inter_knee_dist_bottom = np.sum(knee_dist_t * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)
        inter_knee_pelvis_ratio = inter_knee_dist_bottom / pelvis_width # (N, 1)

        # 4. 전두면 골반 경사 (Trendelenburg drop, 피처 8, 9)
        pelvic_dy = hip_r[:, :, 1] - hip_l[:, :, 1]
        pelvic_dx = hip_r[:, :, 0] - hip_l[:, :, 0]
        pelvic_obliquity_t = np.degrees(np.arctan2(pelvic_dy, np.abs(pelvic_dx) + 1e-7)) # (N, T)
        pelvic_obliquity_bottom = np.sum(np.abs(pelvic_obliquity_t) * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)
        pelvic_obliquity_rom = np.ptp(pelvic_obliquity_t, axis=1, keepdims=True) # (N, 1)

        # 5. 선행 다리 대퇴골 내전각 (Hip Adduction, 피처 10)
        femur_l = knee_l - hip_l
        femur_r = knee_r - hip_r
        proj_femur_l = np.sum(femur_l * hip_axis_norm, axis=-1) / (np.linalg.norm(femur_l, axis=-1) + 1e-7)
        proj_femur_r = np.sum(femur_r * (-hip_axis_norm), axis=-1) / (np.linalg.norm(femur_r, axis=-1) + 1e-7)
        adduction_l_deg = np.degrees(np.arcsin(np.clip(proj_femur_l, -1.0, 1.0)))
        adduction_r_deg = np.degrees(np.arcsin(np.clip(proj_femur_r, -1.0, 1.0)))
        adduction_lead_t = np.where(is_left_stepping, adduction_l_deg, adduction_r_deg)
        hip_adduction_bottom = np.sum(np.maximum(0.0, adduction_lead_t) * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)

        # 6. 발목 횡방향 간격 수축 비율 (피처 11)
        ankle_lateral_dist = np.abs(np.sum((ankle_r - ankle_l) * hip_axis_norm, axis=-1))
        lateral_dist_init = np.mean(ankle_lateral_dist[:, :5], axis=1, keepdims=True)
        lateral_dist_min = np.min(ankle_lateral_dist, axis=1, keepdims=True)
        lateral_contraction_ratio = (lateral_dist_min + 1e-4) / (lateral_dist_init + 1e-4) # (N, 1)

        # 7. 최하사점 생체역학 교차 복합지수 (어깨 비틀림 배제: 골반x내전x무릎밀착, 피처 12)
        cross_bottom_bio_score = (pelvic_obliquity_bottom / 10.0) * (hip_adduction_bottom / 10.0) * (0.3 / (inter_knee_dist_bottom + 0.05)) # (N, 1)

        # 8. 최하사점 3D 가림 보조 지표 (선행 무릎-후행 발목 및 양 발목 거리, 피처 13, 14)
        knee_cross_ankle_t = np.minimum(
            np.linalg.norm(knee_l - ankle_r, axis=-1),
            np.linalg.norm(knee_r - ankle_l, axis=-1)
        )
        knee_cross_ankle_bottom = np.sum(knee_cross_ankle_t * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)

        ankle_dist_t = np.linalg.norm(ankle_l - ankle_r, axis=-1)
        inter_ankle_bottom = np.sum(ankle_dist_t * bottom_win_mask, axis=1, keepdims=True) / win_count # (N, 1)

        feats_14d = np.concatenate([
            step_ratio_ml_ap,
            crossover_bottom_win, crossover_bottom_ratio, crossover_margin_mean, crossover_bottom,
            inter_knee_dist_bottom, inter_knee_pelvis_ratio,
            pelvic_obliquity_bottom, pelvic_obliquity_rom,
            hip_adduction_bottom,
            lateral_contraction_ratio, cross_bottom_bio_score,
            knee_cross_ankle_bottom, inter_ankle_bottom
        ], axis=1).astype(np.float32)

        return feats_14d

    @classmethod
    def _augment_cross_lunge(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_ratio: float = 0.35,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        크로스 런지 소수 클래스 집중 기구학적 데이터 증강 (지터링/노이즈 완전 배제)
        - 순수 기구학적 클래스 내 믹스업 (Intra-Class Mixup): 실제 크로스 런지 쌍의 베타(2.0, 2.0) 완만한 보간
        - 1:10 불균형을 1:3 (target_ratio=0.35) 수준으로 보강
        """
        np.random.seed(random_seed)
        dyn_idx = np.where(y_train == 0)[0]
        cr_idx = np.where(y_train == 1)[0]
        n_dyn = len(dyn_idx)
        n_cr = len(cr_idx)
        needed = int(n_dyn * target_ratio) - n_cr
        if needed <= 0 or n_cr < 2:
            return X_train, y_train

        cr_feats = X_train[cr_idx]
        new_feats = []
        for _ in range(needed):
            r1 = np.random.randint(0, n_cr)
            r2 = np.random.randint(0, n_cr)
            lam = float(np.random.beta(2.0, 2.0))
            # 순수 기구학적 보간 (노이즈/지터링 없음)
            interpolated = lam * cr_feats[r1] + (1.0 - lam) * cr_feats[r2]
            new_feats.append(interpolated)

        aug_X = np.concatenate([X_train, np.array(new_feats, dtype=np.float32)], axis=0)
        aug_y = np.concatenate([y_train, np.ones(needed, dtype=int)], axis=0)
        return aug_X, aug_y

    def tune_and_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_saved_params: bool = False
    ) -> Dict[str, Any]:
        """
        무빙 런지 2진 하이퍼파라미터 그리드 서치 및 크로스 런지 임계값(tau*_cross) 동시 최적화
        (지터링 없는 순수 기구학적 증강 + Out-of-Fold 교차 검증 기반 안전 임계값 탐색)
        """
        if use_saved_params and hasattr(self, 'has_saved_params') and self.has_saved_params:
            self.model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                n_estimators=100,
                max_depth=self.best_params.get('max_depth', 4),
                num_leaves=self.best_params.get('num_leaves', 15),
                learning_rate=self.best_params.get('learning_rate', 0.03),
                scale_pos_weight=self.best_params.get('scale_pos_weight', 2.5),
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=4
            )
            self.model.fit(X_train, y_train)
            return {
                'best_params': self.best_params,
                'best_threshold': self.best_threshold,
                'fast_mode': True
            }

        # 1. 크로스 런지 소수 클래스 기구학적 증강 (지터링 배제)
        X_tr_aug, y_tr_aug = self._augment_cross_lunge(X_train, y_train, target_ratio=0.35)

        # 이중 과보상 제거: 증강으로 이미 불균형이 완화되었으므로 중립 가중치(1.0 ~ 1.3) 사용
        param_candidates = [
            {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'scale_pos_weight': 1.0},
            {'max_depth': 4, 'num_leaves': 15, 'learning_rate': 0.03, 'scale_pos_weight': 1.0},
            {'max_depth': 4, 'num_leaves': 15, 'learning_rate': 0.03, 'scale_pos_weight': 1.2},
            {'max_depth': 5, 'num_leaves': 31, 'learning_rate': 0.03, 'scale_pos_weight': 1.0},
        ]

        best_score = -1.0
        best_cfg = param_candidates[1]
        best_tau = 0.50

        # Out-of-Fold (OOF) 3-Fold 방식으로 훈련 세트 과적합 없이 안전한 임계값 탐색 (0.46 ~ 0.58)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for cfg in param_candidates:
            oof_probs = np.zeros(len(y_tr_aug))
            for tr_idx, val_idx in skf.split(X_tr_aug, y_tr_aug):
                fold_model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    n_estimators=100,
                    max_depth=cfg['max_depth'],
                    num_leaves=cfg['num_leaves'],
                    learning_rate=cfg['learning_rate'],
                    scale_pos_weight=cfg['scale_pos_weight'],
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1,
                    n_jobs=4
                )
                fold_model.fit(X_tr_aug[tr_idx], y_tr_aug[tr_idx])
                oof_probs[val_idx] = fold_model.predict_proba(X_tr_aug[val_idx])[:, 1]

            for tau in np.arange(0.40, 0.62, 0.02):
                preds = (oof_probs >= tau).astype(int)
                f1 = f1_score(y_tr_aug, preds, zero_division=0)
                # 다이나믹 런지(0)의 재현율(Recall)이 95% 이상이어야만 안전 채택 (Neyman-Pearson 안전 제약)
                rec_dyn = np.sum((y_tr_aug == 0) & (preds == 0)) / max(1, np.sum(y_tr_aug == 0))
                penalized_score = f1 if rec_dyn >= 0.95 else (f1 * (rec_dyn / 0.95))
                if penalized_score > best_score:
                    best_score = penalized_score
                    best_cfg = cfg
                    best_tau = float(tau)

        self.best_params.update(best_cfg)
        self.best_threshold = best_tau

        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=100,
            max_depth=self.best_params['max_depth'],
            num_leaves=self.best_params['num_leaves'],
            learning_rate=self.best_params['learning_rate'],
            scale_pos_weight=self.best_params['scale_pos_weight'],
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            n_jobs=4
        )
        self.model.fit(X_tr_aug, y_tr_aug)

        return {
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'best_val_f1': best_score
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """2개 클래스 사후 확률 반환 -> Shape: (N, 2) [0: 다이나믹, 1: 크로스]"""
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """최적 임계값 적용 2진 예측 (0: 다이나믹 런지, 1: 크로스 런지)"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.best_threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """무빙 런지 서브 모델 2진 정밀 성능 지표 및 2x2 혼동 행렬 평가"""
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)

        acc = accuracy_score(y_true, preds)
        prec_cross = precision_score(y_true, preds, zero_division=0)
        rec_cross = recall_score(y_true, preds, zero_division=0)
        f1_cross = f1_score(y_true, preds, zero_division=0)

        roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
        cm = confusion_matrix(y_true, preds, labels=[0, 1])

        # 다이나믹 런지 (0) 지표
        cm_00, cm_01, cm_10, cm_11 = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        prec_dyn = cm_00 / max(1, cm_00 + cm_10)
        rec_dyn = cm_00 / max(1, cm_00 + cm_01)
        f1_dyn = 2 * (prec_dyn * rec_dyn) / max(1e-7, prec_dyn + rec_dyn)

        return {
            'accuracy': float(acc),
            'cross_precision': float(prec_cross),
            'cross_recall': float(rec_cross),
            'cross_f1': float(f1_cross),
            'dynamic_precision': float(prec_dyn),
            'dynamic_recall': float(rec_dyn),
            'dynamic_f1': float(f1_dyn),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'best_threshold': float(self.best_threshold),
            'confusion_matrix': cm.tolist(),
            'support_dynamic': int(np.sum(y_true == 0)),
            'support_cross': int(np.sum(y_true == 1))
        }
