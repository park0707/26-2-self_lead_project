# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_c2_v6_selective_abstention.py
Purpose: c2 v6.0 선택적 거부(Selective Abstention / 3-State) 전수(2,605개) 실측 성능 평가
         - Risk-Coverage Curve 도출 (Coverage vs Selective F1 vs Selective Risk)
         - 5대 시점별 Coverage 및 Abstention 사유(reason_code) 분포 분석
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
from src.models.second_models.babel_squat.c2_model_v6 import BabelSquatC2ModelV6

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c2_v6_15d_feats.npz"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c2 v6.0 선택적 거부(Selective Classification / 3-State) 전수 실측 평가", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_true = master['Y_defects_golden'][:, 1].astype(int)
    views = master['views']
    subjects = master['subjects']
    N = len(seq_ids)

    # 피처 로드
    d_feats = np.load(FEATS_CACHE_PATH, allow_pickle=True)
    X_v6 = d_feats['X_c2_v6_15d']
    print(f"[*] v6 피처 로드 완료: {X_v6.shape}", flush=True)

    model = BabelSquatC2ModelV6()

    # 1. 다양한 임계치 마진에 따른 Risk-Coverage Curve 실험
    # tau_normal, tau_defect
    threshold_pairs = [
        (0.50, 0.50, "Full Decision (No Abstention)"),
        (0.45, 0.55, "Narrow Abstention (+/- 0.05)"),
        (0.40, 0.60, "Balanced Abstention (+/- 0.10)"),
        (0.35, 0.65, "Standard Selective (+/- 0.15)"),
        (0.30, 0.70, "High Confidence (+/- 0.20)"),
        (0.25, 0.75, "Strict Quality (+/- 0.25)")
    ]

    print("\n" + "=" * 115, flush=True)
    print("  [실험 1] 임계치 마진별 Risk-Coverage Trade-off 대조표 (N=2,605)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'임계치 설정':<32} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk (오류율)':<12}", flush=True)
    print("-" * 115, flush=True)

    best_results = None

    for t_norm, t_def, desc in threshold_pairs:
        results = model.predict_selective_batch(
            X_v6,
            target_mode="auto",
            tau_normal=t_norm,
            tau_defect=t_def,
            disagreement_tol=0.35,
            fold_std_tol=0.25
        )

        statuses = np.array([r['status'] for r in results])
        p_defects = np.array([r['defect_probability'] if r['defect_probability'] is not None else 0.5 for r in results])

        determined_mask = (statuses != "UNDETERMINED")
        n_det = int(np.sum(determined_mask))
        cov = n_det / N * 100.0
        abs_rate = 100.0 - cov

        if n_det > 0:
            y_det_true = y_true[determined_mask]
            y_det_pred = (statuses[determined_mask] == "DEFECT").astype(int)
            sel_f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
            sel_acc = accuracy_score(y_det_true, y_det_pred) * 100.0
            sel_risk = (1.0 - accuracy_score(y_det_true, y_det_pred)) * 100.0
        else:
            sel_f1, sel_acc, sel_risk = 0.0, 0.0, 100.0

        label_str = f"[{t_norm:.2f}, {t_def:.2f}] {desc}"
        print(f"{label_str:<32} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>11.2f}%", flush=True)

        if t_norm == 0.35 and t_def == 0.65:
            best_results = results

    # 2. 표준 선택적 분류 ([0.35, 0.65]) 환경에서의 심층 분석
    print("\n" + "=" * 115, flush=True)
    print("  [실험 2] 표준 선택적 분류 ([0.35, 0.65]) 5대 시점별 Coverage 및 Selective F1 감사", flush=True)
    print("=" * 115, flush=True)

    statuses = np.array([r['status'] for r in best_results])
    reasons = np.array([r['reason_code'] for r in best_results])
    determined_mask = (statuses != "UNDETERMINED")

    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (완전 후방)', 'view2 (우측 45도)', 'view3 (완전 정면)', 'view4 (좌측 45도)', 'view5 (우측 후방 45도)']

    print(f"{'시점 (카메라 앵글)':<25} | {'전체 표본':<10} | {'판정 표본':<10} | {'Coverage':<10} | {'Selective F1':<14} | {'주요 거부 사유'}", flush=True)
    print("-" * 115, flush=True)

    for v, vn in zip(view_list, view_names):
        v_mask = (views == v)
        n_v_total = int(np.sum(v_mask))
        v_det_mask = v_mask & determined_mask
        n_v_det = int(np.sum(v_det_mask))
        v_cov = n_v_det / max(1, n_v_total) * 100.0

        if n_v_det > 0:
            v_f1 = f1_score(y_true[v_det_mask], (statuses[v_det_mask] == "DEFECT").astype(int), zero_division=0) * 100.0
        else:
            v_f1 = 0.0

        # 거부 사유 빈도
        v_und_mask = v_mask & (~determined_mask)
        v_reasons = reasons[v_und_mask]
        u_r, counts = np.unique(v_reasons, return_counts=True)
        if len(u_r) > 0:
            top_reason = f"{u_r[np.argmax(counts)]} ({np.max(counts)}건)"
        else:
            top_reason = "거부 없음"

        print(f"{vn:<25} | {n_v_total:>9d}개 | {n_v_det:>9d}개 | {v_cov:>9.1f}% | {v_f1:>13.2f}% | {top_reason}", flush=True)

    # 3. 거부 사유(Reason Code) 전체 분포
    print("\n" + "=" * 115, flush=True)
    print("  [실험 3] 전체 거부 사유(Reason Code) 분포 현황", flush=True)
    print("=" * 115, flush=True)
    all_und = (~determined_mask)
    all_reasons = reasons[all_und]
    u_reasons, r_counts = np.unique(all_reasons, return_counts=True)
    for r, c in zip(u_reasons, r_counts):
        pct = c / N * 100.0
        print(f"  • {r:<30}: {c:4d}건 ({pct:5.2f}%)")

    # 4. 안구 시선(True Gaze) 엄격 요청 시 후방 시점 자동 거부 검증
    print("\n" + "=" * 115, flush=True)
    print("  [실험 4] 안구 시선(target_mode='true_gaze') 요청 시 시점별 반응 검증", flush=True)
    print("=" * 115, flush=True)
    gaze_results = model.predict_selective_batch(X_v6, target_mode="true_gaze")
    gaze_statuses = np.array([r['status'] for r in gaze_results])
    gaze_reasons = np.array([r['reason_code'] for r in gaze_results])

    for v, vn in zip(view_list, view_names):
        v_mask = (views == v)
        v_face_not_obs = np.sum((gaze_reasons == "FACE_NOT_OBSERVABLE") & v_mask)
        v_det = np.sum((gaze_statuses != "UNDETERMINED") & v_mask)
        print(f"  • {vn:<25}: 판정 {v_det:4d}건 | FACE_NOT_OBSERVABLE 거부: {v_face_not_obs:4d}건")

    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
