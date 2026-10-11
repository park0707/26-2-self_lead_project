# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_c2_v7_selective_abstention.py
Purpose: c2 v7.0 선택적 거부(Selective Abstention / 3-State) 전수(2,605개) 실측 성능 평가
         - 전면/측면(코 관측) vs 후방(코 미관측 -> 경추 피처 1차 판단 + 엄격 신뢰도 게이팅)
         - 5대 시점별 Coverage, Selective F1, Selective Acc, Risk 및 거부 사유(reason_code) 분포 분석
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c2_model_v7 import BabelSquatC2ModelV7

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
WEIGHTS_PATH = r"D:\src\models\second_models\babel_squat\c2_model_v7_weights.pkl"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c2 v7.0 선택적 거부(Selective Classification / 3-State) 전수 실측 평가", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_true = master['Y_defects_golden'][:, 1].astype(int)
    views = master['views']
    subjects = master['subjects']
    N = len(seq_ids)

    # 피처 로드
    d_feats = np.load(FEATS_CACHE_PATH, allow_pickle=True)
    X_v7 = d_feats['X_c2_v7_19d']
    print(f"[*] v7 19D 피처 로드 완료: {X_v7.shape}", flush=True)

    model = BabelSquatC2ModelV7(weights_path=WEIGHTS_PATH)

    # 1. 시나리오별 성능 비교표
    scenarios = [
        ("1. Full Decision (강제 판정)", 0.50, 0.50, 0.50, 0.50, 1.00, 1.00),
        ("2. Uniform Margin (전구간 [0.40, 0.60])", 0.40, 0.60, 0.40, 0.60, 0.35, 0.35),
        ("3. Rear Moderate Gate (후방 [0.35, 0.65])", 0.40, 0.60, 0.35, 0.65, 0.35, 0.30),
        ("4. Rear Standard Gate (후방 [0.30, 0.70]) ★공식추천", 0.40, 0.60, 0.30, 0.70, 0.35, 0.25),
        ("5. Rear Strict Gate (후방 [0.25, 0.75])", 0.40, 0.60, 0.25, 0.75, 0.35, 0.20),
        ("6. Rear Ultra Gate (후방 [0.20, 0.80])", 0.40, 0.60, 0.20, 0.80, 0.35, 0.15),
        ("7. Pure Face-Visible Only (후방 100% 거부)", 0.40, 0.60, 0.00, 1.00, 0.35, 0.00)
    ]

    print("\n" + "=" * 115, flush=True)
    print("  [실험 1] 코 미관측 신뢰도 게이팅 시나리오별 Risk-Coverage 분석 (N=2,605)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시나리오':<42} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk(오류율)':<10}", flush=True)
    print("-" * 115, flush=True)

    official_results = None

    for desc, fn, fd, rn, rd, f_disag, r_disag in scenarios:
        results = model.predict_selective_batch(
            X_v7,
            tau_front_normal=fn, tau_front_defect=fd,
            tau_rear_normal=rn, tau_rear_defect=rd,
            front_disag_tol=f_disag, rear_disag_tol=r_disag
        )

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

        print(f"{desc:<42} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>9.2f}%", flush=True)

        if "공식추천" in desc:
            official_results = results

    # 2. 공식 추천 설정의 5대 시점별 세부 감사
    print("\n" + "=" * 115, flush=True)
    print("  [실험 2] 공식 프로덕션 v7.0 5대 시점별 정밀 성능 감사 (Rear Standard Gate)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (View)':<15} | {'전체 N':<8} | {'판정 N (Cov)':<16} | {'보류 N (Abs)':<16} | {'Selective F1':<13} | {'Selective Acc':<13}", flush=True)
    print("-" * 115, flush=True)

    statuses_off = np.array([r['status'] for r in official_results])
    reasons_off = np.array([r['reason_code'] for r in official_results])

    for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
        m_v = (views == v)
        n_v = int(np.sum(m_v))
        st_v = statuses_off[m_v]
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

    # 3. 보류 사유(reason_code) 분포 분석
    print("\n" + "=" * 115, flush=True)
    print("  [실험 3] 판정 불가(UNDETERMINED) 사유별(reason_code) 세부 통계 (N=2,605)", flush=True)
    print("=" * 115, flush=True)
    unique_reasons, r_counts = np.unique(reasons_off, return_counts=True)
    for r_code, count in zip(unique_reasons, r_counts):
        pct = count / N * 100.0
        is_undeter = "UNDETERMINED 사유" if "CLEAR" not in r_code else "판정 확정"
        print(f"  - [{is_undeter:<13}] {r_code:<30}: {count:4d}건 ({pct:5.1f}%)", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
