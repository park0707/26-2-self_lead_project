# [실행 유틸리티]: 2단계 정상(1) vs 오류(0) 극단적 불균형 전용 초정밀 평가 모듈

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)

class Stage2MetricsHelper:
    """2단계 1:17.0 극단 불균형 정상/오류 판별 모델을 위한 학술 평가 유틸리티"""

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        beta: float = 1.0
    ) -> Tuple[float, float]:
        """PR-Curve 기반 F_beta(기본 F1, 안전 F2)를 최대화하는 최적 결정 임계값 탐색"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba, pos_label=1)
        if len(thresholds) == 0:
            return 0.5, 0.0

        # F_beta = (1 + beta^2) * (prec * rec) / (beta^2 * prec + rec)
        beta_sq = beta ** 2
        f_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            denom = (beta_sq * p + r)
            if denom > 0:
                f = (1.0 + beta_sq) * (p * r) / denom
            else:
                f = 0.0
            f_scores.append(f)

        f_scores = np.array(f_scores)
        best_idx = int(np.argmax(f_scores))
        best_thresh = float(thresholds[best_idx])
        best_score = float(f_scores[best_idx])

        return best_thresh, best_score

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        inference_time_ms: float = 0.0
    ) -> Dict[str, Any]:
        """불균형 2진 분류 전문 지표 전수 정밀 계산"""
        y_pred = (y_proba >= threshold).astype(np.int64)

        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # 0: 오류(Negative), 1: 정상(Positive)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / max(1, (tp + fn)) # Positive Recall (정상 감지율)
        specificity = tn / max(1, (tn + fp)) # Negative Recall (오류 감지율)
        g_mean = float(np.sqrt(sensitivity * specificity))

        pos_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        pos_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        pos_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # F2 score (부상 방지: False Negative 최소화)
        beta = 2.0
        denom = (beta**2 * pos_prec + pos_rec)
        pos_f2 = (1 + beta**2) * (pos_prec * pos_rec) / denom if denom > 0 else 0.0

        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            pr_auc = float(average_precision_score(y_true, y_proba))
        except Exception:
            roc_auc, pr_auc = 0.5, 0.0

        fps = 1000.0 / max(1e-4, inference_time_ms) if inference_time_ms > 0 else 0.0

        return {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc),
            'cohen_kappa': float(kappa),
            'g_mean': float(g_mean),
            'sensitivity_pos_rec': float(sensitivity),
            'specificity_neg_rec': float(specificity),
            'pos_precision': float(pos_prec),
            'pos_recall': float(pos_rec),
            'pos_f1': float(pos_f1),
            'pos_f2_safety': float(pos_f2),
            'macro_f1': float(macro_f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'threshold_used': float(threshold),
            'confusion_matrix': cm.tolist(),
            'inference_time_ms': float(inference_time_ms),
            'fps': float(fps)
        }
