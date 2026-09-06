# [실행 유틸리티]: 3단계 세부 결함(Conditions) 다중 라벨 전용 초정밀 평가 모듈

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss, precision_score, recall_score,
    jaccard_score, average_precision_score, classification_report
)

class Stage3MetricsHelper:
    """3단계 다중 라벨 결함 진단 모델을 위한 학술 평가 유틸리티"""

    @staticmethod
    def evaluate_predictions(
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        Y_proba: Optional[np.ndarray] = None,
        cond_names: Optional[List[str]] = None,
        inference_time_ms: float = 0.0
    ) -> Dict[str, Any]:
        """다중 라벨 결함 진단 전문 지표 전수 정밀 계산"""
        num_samples = len(Y_true)
        if num_samples == 0:
            return {}

        num_conds = Y_true.shape[1] if Y_true.ndim > 1 else 1
        if cond_names is None or len(cond_names) != num_conds:
            cond_names = [f"Cond_{i+1}" for i in range(num_conds)]

        # 1. 종합 지표 (Overall Metrics)
        subset_acc = float(accuracy_score(Y_true, Y_pred))  # Exact Match Ratio
        h_loss = float(hamming_loss(Y_true, Y_pred))        # Bit-level error rate
        
        # Jaccard IoU (Samples & Macro)
        jaccard_samples = float(jaccard_score(Y_true, Y_pred, average='samples', zero_division=0))
        jaccard_macro = float(jaccard_score(Y_true, Y_pred, average='macro', zero_division=0))
        
        micro_f1 = float(f1_score(Y_true, Y_pred, average='micro', zero_division=0))
        macro_f1 = float(f1_score(Y_true, Y_pred, average='macro', zero_division=0))
        micro_prec = float(precision_score(Y_true, Y_pred, average='micro', zero_division=0))
        micro_rec = float(recall_score(Y_true, Y_pred, average='micro', zero_division=0))

        # 2. 결함별(Condition-wise) 세부 성적표
        per_condition = {}
        for i, c_name in enumerate(cond_names):
            y_t_col = Y_true[:, i]
            y_p_col = Y_pred[:, i]
            
            support = int(np.sum(y_t_col == 1))
            prec_c = float(precision_score(y_t_col, y_p_col, zero_division=0))
            rec_c = float(recall_score(y_t_col, y_p_col, zero_division=0))
            f1_c = float(f1_score(y_t_col, y_p_col, zero_division=0))
            
            pr_auc_c = 0.0
            if Y_proba is not None:
                try:
                    pr_auc_c = float(average_precision_score(y_t_col, Y_proba[:, i]))
                except Exception:
                    pr_auc_c = 0.0

            per_condition[c_name] = {
                'support_defects': support,
                'defect_ratio': float(support / max(1, num_samples)),
                'precision': prec_c,
                'recall': rec_c,
                'f1': f1_c,
                'pr_auc': pr_auc_c
            }

        # 3. 결함 개수별 일치도 분석 (0개 정상, 1개 결함, 2개 이상 복합 결함)
        true_counts = np.sum(Y_true, axis=1)
        pred_counts = np.sum(Y_pred, axis=1)
        zero_defect_mask = (true_counts == 0)
        zero_defect_acc = float(np.mean(pred_counts[zero_defect_mask] == 0)) if np.sum(zero_defect_mask) > 0 else 1.0

        fps = 1000.0 / max(1e-4, inference_time_ms) if inference_time_ms > 0 else 0.0

        return {
            'num_samples': num_samples,
            'num_conditions': num_conds,
            'subset_accuracy': subset_acc,
            'hamming_loss': h_loss,
            'jaccard_samples_iou': jaccard_samples,
            'jaccard_macro_iou': jaccard_macro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'micro_precision': micro_prec,
            'micro_recall': micro_rec,
            'zero_defect_accuracy': zero_defect_acc,
            'per_condition_metrics': per_condition,
            'inference_time_ms': float(inference_time_ms),
            'fps': float(fps)
        }
