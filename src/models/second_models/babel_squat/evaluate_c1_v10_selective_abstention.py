# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_c1_v10_selective_abstention.py
Purpose: c1 v10.0 선택적 거부(Selective Abstention / 3-State: 정상 / 결함 / 판정 불가) 전수(2,605개) 실측 성능 평가
         - Risk-Coverage Curve 도출 (Coverage vs Selective F1 vs Selective Risk)
         - 5대 시점별 Coverage 및 Abstention 사유(reason_code) 분포 분석
         - 이전 강제 이진(v9/v6) 대비 신뢰도 및 오류 차단 효과 비교
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
from src.models.second_models.babel_squat.c1_model_v10 import BabelSquatC1ModelV10

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_v9_28d_feats.npz"
V9_WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v9_weights.pkl"
V10_WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v10_weights.pkl"
OUT_OOF_PATH = r"D:\Data\피처_데이터\oof_c1_v10_selective_predictions.npz"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1 v10.0 선택적 거부(Selective Classification / 3-State) 전수 실측 평가", flush=True)
    print("  ★ 2,605건 전수(5대 전 시점 x 9명 피험자) 대상 Risk-Coverage Trade-off 정밀 도출", flush=True)
    print("  ★ 3계층 불확실성 필터(Hard Gate + Ensemble Disagreement + Double Threshold)", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_true = master['Y_defects_golden'][:, 0].astype(int)
    y_norm = master['y_normal_golden'].astype(int)
    views = master['views']
    subjects = master['subjects']
    N = len(seq_ids)

    # 피처 로드
    d_feats = np.load(FEATS_CACHE_PATH, allow_pickle=True)
    X_v10 = d_feats['X_c1_v9']
    print(f"[*] 28D 피처 로드 완료: {X_v10.shape}", flush=True)

    # v10 가중치 생성 및 로드
    model = BabelSquatC1ModelV10(weights_path=V9_WEIGHTS_PATH)
    model.save_weights(V10_WEIGHTS_PATH)

    # 1. 다양한 임계치 마진에 따른 Risk-Coverage Curve 실험
    threshold_pairs = [
        (0.50, 0.50, "Full Decision (강제 이진, v9 기준)"),
        (0.45, 0.55, "Narrow Abstention (+/- 0.05)"),
        (0.40, 0.60, "Balanced Abstention (+/- 0.10, 추천)"),
        (0.35, 0.65, "Standard Selective (+/- 0.15)"),
        (0.30, 0.70, "High Confidence (+/- 0.20)"),
        (0.25, 0.75, "Strict Quality (+/- 0.25)")
    ]

    print("\n" + "=" * 115, flush=True)
    print("  [실험 1] 임계치 마진별 Risk-Coverage Trade-off 대조표 (N=2,605)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'임계치 설정':<32} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk (오류율)':<12}", flush=True)
    print("-" * 115, flush=True)

    saved_results_map = {}

    for t_norm, t_def, desc in threshold_pairs:
        results = model.predict_selective_batch(
            X_v10,
            tau_normal=t_norm,
            tau_defect=t_def,
            disagreement_tol=0.35,
            fold_std_tol=0.25,
            min_rom_threshold=0.0004
        )

        statuses = np.array([r['status'] for r in results])
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
        saved_results_map[(t_norm, t_def)] = (results, cov, abs_rate, sel_f1, sel_acc, sel_risk)

    # 2. Balanced ([0.40, 0.60]) 설정 심층 분석
    primary_pair = (0.40, 0.60)
    best_results, cov_b, abs_b, f1_b, acc_b, risk_b = saved_results_map[primary_pair]

    print("\n" + "=" * 115, flush=True)
    print(f"  [실험 2] Balanced 판정 불가 ([0.40, 0.60]) 5대 시점별 Coverage 및 Selective F1 감사", flush=True)
    print("=" * 115, flush=True)

    statuses = np.array([r['status'] for r in best_results])
    reasons = np.array([r['reason_code'] for r in best_results])
    p_defects = np.array([r['defect_probability'] for r in best_results])
    confidences = np.array([r['confidence'] for r in best_results])
    determined_mask = (statuses != "UNDETERMINED")

    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (완전 후방 180°)', 'view2 (우측 45도 측면)', 'view3 (완전 정면 0°)', 'view4 (좌측 45도 측면)', 'view5 (우측 후방 45도)']

    print(f"{'시점 (카메라 앵글)':<26} | {'전체 표본':<10} | {'판정 표본':<10} | {'Coverage':<10} | {'Selective F1':<14} | {'주요 거부 사유'}", flush=True)
    print("-" * 115, flush=True)

    view_analysis = {}
    for v, vn in zip(view_list, view_names):
        v_mask = (views == v)
        n_v_total = int(np.sum(v_mask))
        v_det_mask = v_mask & determined_mask
        n_v_det = int(np.sum(v_det_mask))
        v_cov = n_v_det / max(1, n_v_total) * 100.0

        if n_v_det > 0:
            v_f1 = f1_score(y_true[v_det_mask], (statuses[v_det_mask] == "DEFECT").astype(int), zero_division=0) * 100.0
            v_acc = accuracy_score(y_true[v_det_mask], (statuses[v_det_mask] == "DEFECT").astype(int)) * 100.0
        else:
            v_f1, v_acc = 0.0, 0.0

        v_und_mask = v_mask & (~determined_mask)
        v_reasons = reasons[v_und_mask]
        u_r, counts = np.unique(v_reasons, return_counts=True)
        if len(u_r) > 0:
            top_reason = f"{u_r[np.argmax(counts)]} ({np.max(counts)}건)"
        else:
            top_reason = "거부 없음"

        view_analysis[v] = {
            'total': n_v_total, 'det': n_v_det, 'cov': round(v_cov, 1),
            'f1': round(v_f1, 2), 'acc': round(v_acc, 2), 'top_reason': top_reason
        }
        print(f"{vn:<26} | {n_v_total:>9d}개 | {n_v_det:>9d}개 | {v_cov:>9.1f}% | {v_f1:>13.2f}% | {top_reason}", flush=True)

    # 3. 거부 사유 전체 분포
    print("\n" + "=" * 115, flush=True)
    print("  [실험 3] 판정 불가(UNDETERMINED) 사유 코드별 발생 빈도 및 점유율", flush=True)
    print("=" * 115, flush=True)
    und_mask = (~determined_mask)
    tot_und = int(np.sum(und_mask))
    u_reasons, r_counts = np.unique(reasons[und_mask], return_counts=True)
    for r_code, count in sorted(zip(u_reasons, r_counts), key=lambda x: x[1], reverse=True):
        print(f"    - {r_code:<28}: {count:>4d}건 ({count/tot_und*100:>5.1f}% / 전체 표본 대비 {count/N*100:>4.1f}%)", flush=True)

    # 4. 클린 정상군 허위 오탐 구제 효과 분석
    clean_mask = (y_norm == 1)
    clean_tot = int(np.sum(clean_mask))
    clean_det_mask = clean_mask & determined_mask
    clean_det_tot = int(np.sum(clean_det_mask))
    clean_det_preds = (statuses[clean_det_mask] == "DEFECT").astype(int)
    clean_fp_sel = int(np.sum(clean_det_preds == 1))
    clean_fp_rate_sel = (clean_fp_sel / max(1, clean_det_tot)) * 100.0

    print("\n" + "=" * 115, flush=True)
    print("  [실험 4] 클린 정상군(골든 그라운드트루스 135건) 허위 결함 오탐 구제 효과", flush=True)
    print("=" * 115, flush=True)
    print(f"    - 기존 강제 이진(v9) 정상군 허위 오탐: 22건 (16.30%)", flush=True)
    print(f"    - v10 판정 불가 도입 후 판정군 허위 오탐: {clean_fp_sel}건 / {clean_det_tot}건 ({clean_fp_rate_sel:.2f}%)", flush=True)
    print(f"    - 보류(구제)된 애매한 정상 샘플 수: {clean_tot - clean_det_tot}건", flush=True)

    # 5. OOF 예측값 저장
    np.savez_compressed(
        OUT_OOF_PATH,
        statuses=statuses,
        p_defects=p_defects,
        confidences=confidences,
        reasons=reasons,
        y_true=y_true,
        views=views,
        subjects=subjects,
        sequence_ids=seq_ids,
        primary_metrics={
            'coverage': cov_b,
            'abstention': abs_b,
            'selective_f1': f1_b,
            'selective_acc': acc_b,
            'selective_risk': risk_b
        },
        view_analysis=view_analysis
    )
    print(f"\n[*] c1 v10.0 OOF 선택적 예측값 저장 완료: {OUT_OOF_PATH}", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
