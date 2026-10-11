# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_c4_v5_selective_abstention.py
Purpose: c4 v5.0 선택적 판정(Selective Classification) 및 5대 시점별 세부 성능 실측 감사
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c4_model_v5 import BabelSquatC4ModelV5

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
WEIGHTS_PATH = r"D:\src\models\second_models\babel_squat\c4_model_v5_weights.pkl"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c4 v5.0 선택적 판정(Selective Classification / 3-State) 전수 실측 평가", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    y_true = master['Y_defects_golden'][:, 3].astype(int)
    views = master['views']
    subjects = master['subjects']
    N = len(y_true)

    X_v5 = np.load(FEATS_PATH)['X_c4_v5_25d']
    model = BabelSquatC4ModelV5(weights_path=WEIGHTS_PATH)

    # 1. 다양한 마진별 Selective 성적표
    margins = [
        ("1. Full Decision (강제 판정)", 0.50, 0.50),
        ("2. Narrow Abstention (+/- 0.05)", 0.45, 0.55),
        ("3. Standard Abstention (+/- 0.08)", 0.42, 0.58),
        ("4. Balanced Abstention (+/- 0.10) ★공식추천", 0.40, 0.60),
        ("5. Moderate Selective (+/- 0.12)", 0.38, 0.62),
        ("6. High Confidence (+/- 0.15)", 0.35, 0.65),
        ("7. Strict Quality (+/- 0.20)", 0.30, 0.70)
    ]

    print("\n" + "=" * 115, flush=True)
    print("  [실험 1] 임계치 마진별 Risk-Coverage Trade-off 대조표 (N=2,605)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시나리오':<40} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk(오류율)':<10}", flush=True)
    print("-" * 115, flush=True)

    balanced_results = None

    for desc, tn, td in margins:
        results = model.predict_selective_batch(X_v5, tau_normal=tn, tau_defect=td)
        statuses = np.array([r['status'] for r in results])
        det_mask = (statuses != "UNDETERMINED")
        n_det = int(np.sum(det_mask))
        cov = n_det / N * 100.0
        abs_rate = 100.0 - cov

        if n_det > 0:
            y_det_true = y_true[det_mask]
            y_det_pred = (statuses[det_mask] == "DEFECT").astype(int)
            sel_f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
            sel_acc = accuracy_score(y_det_true, y_det_pred) * 100.0
            sel_risk = (1.0 - accuracy_score(y_det_true, y_det_pred)) * 100.0
        else:
            sel_f1, sel_acc, sel_risk = 0.0, 0.0, 100.0

        print(f"{desc:<40} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>9.2f}%", flush=True)

        if "공식추천" in desc:
            balanced_results = results

    # 2. 5대 시점별 세부 감사
    print("\n" + "=" * 115, flush=True)
    print("  [실험 2] 공식 프로덕션 v5.0 5대 시점별 정밀 성능 감사 (Balanced [0.40, 0.60])", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (View)':<15} | {'전체 N':<8} | {'판정 N (Cov)':<16} | {'보류 N (Abs)':<16} | {'Selective F1':<13} | {'Selective Acc':<13}", flush=True)
    print("-" * 115, flush=True)

    st_bal = np.array([r['status'] for r in balanced_results])

    for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
        m_v = (views == v)
        n_v = int(np.sum(m_v))
        st_v = st_bal[m_v]
        det_v = (st_v != "UNDETERMINED")
        n_det_v = int(np.sum(det_v))
        cov_v = n_det_v / n_v * 100.0
        abs_v = 100.0 - cov_v

        if n_det_v > 0:
            y_v_true = y_true[m_v][det_v]
            y_v_pred = (st_v[det_v] == "DEFECT").astype(int)
            f1_v = f1_score(y_v_true, y_v_pred, zero_division=0) * 100.0
            acc_v = accuracy_score(y_v_true, y_v_pred) * 100.0
        else:
            f1_v, acc_v = 0.0, 0.0

        cov_str = f"{n_det_v} ({cov_v:.1f}%)"
        abs_str = f"{n_v - n_det_v} ({abs_v:.1f}%)"
        print(f"{v:<15} | {n_v:<8} | {cov_str:<16} | {abs_str:<16} | {f1_v:>12.2f}% | {acc_v:>12.2f}%", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
