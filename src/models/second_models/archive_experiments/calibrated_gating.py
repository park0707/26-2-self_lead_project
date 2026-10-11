# -*- coding: utf-8 -*-
"""
NovelFlow 확률 보정 및 네이만-피어슨 제약 게이팅 엔진 (Solution 1 & 2)
- Isotonic Regression 확률 보정(Probability Calibration)
- 3단계 세부 결함 모델과의 역방향 상호 검증 필터 (Back-Verification Veto)
- Neyman-Pearson 패러다임: False Positive Budget(FPR 제약) 하에서 Normal Recall 극대화
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, balanced_accuracy_score, f1_score

class CalibratedGatingEngine:
    """
    확률 보정 및 다단계 역방향 상호 검증 통합 판정 엔진
    """
    def __init__(self):
        self.calibrator_m0 = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        self.calibrators_defects = []
        
    def fit_calibrators(self, m0_raw_probs: np.ndarray, y_normal_tr: np.ndarray,
                        defects_raw_probs: np.ndarray, Y_defects_tr: np.ndarray):
        """
        Train 폴드 예측 확률을 실제 라벨과 일치하도록 Isotonic 보정기 피팅
        """
        self.calibrator_m0.fit(m0_raw_probs, y_normal_tr)
        self.calibrators_defects = []
        for j in range(defects_raw_probs.shape[1]):
            ir = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
            ir.fit(defects_raw_probs[:, j], Y_defects_tr[:, j])
            self.calibrators_defects.append(ir)
        return self

    def calibrate_m0(self, probs: np.ndarray) -> np.ndarray:
        return self.calibrator_m0.predict(probs)

    def calibrate_defects(self, probs: np.ndarray) -> np.ndarray:
        calib = np.zeros_like(probs)
        for j, ir in enumerate(self.calibrators_defects):
            calib[:, j] = ir.predict(probs[:, j])
        return calib

    @staticmethod
    def predict_with_policy(
        m0_probs: np.ndarray,
        defect_probs: np.ndarray,
        mahalanobis_dists: Optional[np.ndarray] = None,
        theta_gate: float = 0.35,
        veto_spine: float = 0.70,
        veto_knee: float = 0.65,
        veto_foot: float = 0.70,
        veto_neck: float = 0.85,
        maha_thresh: Optional[float] = None
    ) -> np.ndarray:
        """
        단일 통합 정책에 따른 최종 정상(1) vs 오류(0) 이진 판정
        
        결함 인덱스:
          0: 척추의 중립 (고신뢰도)
          1: 고개 정면 (보통 신뢰도)
          2: 발과 무릎 방향 일치 (최고 신뢰도)
          3: 발바닥 지면 고정 (고신뢰도)
        """
        N = len(m0_probs)
        preds = np.zeros(N, dtype=int)

        for i in range(N):
            # 1. M0 게이트키퍼 1차 통과 여부 검사
            if m0_probs[i] < theta_gate:
                preds[i] = 0 # 오류 확정
                continue

            # 2. [Solution 4] 마할라노비스 다양체 외곽 기각 검사
            if mahalanobis_dists is not None and maha_thresh is not None:
                if mahalanobis_dists[i] > maha_thresh:
                    preds[i] = 0 # 정상 프로토타입 범위 초과로 FP 기각
                    continue

            # 3. [Solution 1] 세부 결함 모델 역방향 상호 검증 거부권 (Veto)
            p_spine = defect_probs[i, 0]
            p_neck  = defect_probs[i, 1]
            p_knee  = defect_probs[i, 2]
            p_foot  = defect_probs[i, 3]

            # 명백한 세부 결함이 감지되면 정상 판정을 즉시 기각하고 오류로 전환
            if p_knee >= veto_knee or p_spine >= veto_spine or p_foot >= veto_foot or p_neck >= veto_neck:
                preds[i] = 0 # 결함 확정 (FP 차단)
                continue

            # 4. 모든 검증을 통과한 순수 정상만 최종 정상 승인!
            preds[i] = 1

        return preds

    @staticmethod
    def predict_soft_fusion(
        m0_probs: np.ndarray,
        defect_probs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        theta_fused: float = 0.18,
        mahalanobis_dists: Optional[np.ndarray] = None,
        maha_thresh: Optional[float] = None
    ) -> np.ndarray:
        """
        [연속형 소프트 퓨전 SOTA 판정]
        S_normal = P(normal) * (1 - max_j w_j P(defect_j))
        """
        if weights is None:
            weights = np.array([0.85, 0.40, 1.00, 0.80])
        defect_score = np.max(defect_probs * weights, axis=-1)
        score_fused = m0_probs * np.clip(1.0 - defect_score, 0.0, 1.0)
        preds = (score_fused >= theta_fused)
        if mahalanobis_dists is not None and maha_thresh is not None:
            preds = preds & (mahalanobis_dists <= maha_thresh)
        return preds.astype(int)

    @classmethod
    def optimize_neyman_pearson(
        cls,
        m0_probs: np.ndarray,
        defect_probs: np.ndarray,
        y_true: np.ndarray,
        mahalanobis_dists: Optional[np.ndarray] = None,
        target_precision: float = 0.45,
        max_fpr: float = 0.08
    ) -> Dict[str, Any]:
        """
        Neyman-Pearson 제약 최적화:
        - 제약 조건: Precision >= target_precision (또는 FPR <= max_fpr)
        - 목적 함수: Normal Recall 극대화
        """
        best_score = -1.0
        best_params = {}
        best_metrics = {}
        best_preds = None

        # 탐색 그리드
        gate_candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        veto_knee_candidates = [0.55, 0.65, 0.75, 0.85]
        veto_spine_candidates = [0.60, 0.70, 0.80]
        
        maha_percentiles = [None, 95.0, 97.5, 99.0] if mahalanobis_dists is not None else [None]

        for tg in gate_candidates:
            for vk in veto_knee_candidates:
                for vs in veto_spine_candidates:
                    for mp in maha_percentiles:
                        m_th = float(np.percentile(mahalanobis_dists[y_true == 1], mp)) if mp is not None else None
                        
                        preds = cls.predict_with_policy(
                            m0_probs=m0_probs,
                            defect_probs=defect_probs,
                            mahalanobis_dists=mahalanobis_dists,
                            theta_gate=tg,
                            veto_spine=vs,
                            veto_knee=vk,
                            veto_foot=0.70,
                            veto_neck=0.85,
                            maha_thresh=m_th
                        )

                        cm = confusion_matrix(y_true, preds)
                        tn, fp, fn, tp = cm.ravel()
                        rec = tp / max(1, tp + fn)
                        prec = tp / max(1, tp + fp)
                        fpr = fp / max(1, fp + tn)
                        bacc = (rec + (1 - fpr)) / 2.0
                        f1_norm = f1_score(y_true, preds, zero_division=0)

                        # 제약 조건 만족 여부 검사 (Precision 우선 또는 Balanced Score)
                        # 페널티 부여: Precision이 목표 미달 시 감점
                        prec_penalty = max(0.0, target_precision - prec) * 2.0
                        objective = rec - prec_penalty + 0.5 * bacc

                        if objective > best_score:
                            best_score = objective
                            best_params = {
                                'theta_gate': tg,
                                'veto_knee': vk,
                                'veto_spine': vs,
                                'veto_foot': 0.70,
                                'veto_neck': 0.85,
                                'maha_thresh': m_th,
                                'maha_percentile': mp
                            }
                            best_metrics = {
                                'recall': float(rec * 100.0),
                                'precision': float(prec * 100.0),
                                'specificity': float((1 - fpr) * 100.0),
                                'balanced_acc': float(bacc * 100.0),
                                'f1_normal': float(f1_norm * 100.0),
                                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                            }
                            best_preds = preds

        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_preds': best_preds
        }
