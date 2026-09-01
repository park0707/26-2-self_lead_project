import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, balanced_accuracy_score, confusion_matrix
)

class EvaluationMetrics:
    """불균형 데이터셋 및 제로샷 일반화 종합 평가 지표 계산기"""

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """Macro-F1, Normal Recall, Balanced Acc, PR-AUC, ROC-AUC 종합 산출"""
        metrics = {}
        metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['normal_f1'] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['normal_recall'] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['normal_precision'] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
                metrics['pr_auc'] = float(auc(recall, precision))
            except Exception:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = 0.0
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        return metrics

    @staticmethod
    def find_optimal_threshold(y_val_true: np.ndarray, y_val_prob: np.ndarray) -> float:
        """Validation 세트에서 Normal Class F1을 극대화하는 최적 결정 임계값 산출"""
        best_thresh = 0.5
        best_f1 = -1.0
        
        for thresh in np.linspace(0.05, 0.95, 91):
            y_pred = (y_val_prob >= thresh).astype(int)
            f1 = f1_score(y_val_true, y_pred, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                
        return best_thresh
