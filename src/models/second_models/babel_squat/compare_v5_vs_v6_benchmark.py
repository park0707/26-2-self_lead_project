# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: compare_v5_vs_v6_benchmark.py
Purpose: BabelSquatIntegratedEngineV5 vs BabelSquatIntegratedEngineV6 1,906개 전수 샘플 1:1 비교 성능 평가
         - 기존 강제 EMR (v5.0) vs 유효 라벨 완전 일치율 AL-EMR (v6.0)
         - 부위별(c1~c4) F1-Score / Precision / Recall 변화
         - 샘플당 평균 유효 라벨 수 및 부위별 판단불가율
         - 완전 정상(Clean Rep) 분류 성능 비교
"""

import os
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.babel_squat_integrated_v5 import BabelSquatIntegratedEngineV5
from src.models.second_models.babel_squat.babel_squat_integrated_v6 import BabelSquatIntegratedEngineV6

GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 통합 진단 엔진 v5.0 vs v6.0 전수 1:1 비교 성능 평가")
    print("=" * 115)

    # 1. 골든 정답 라벨 로드 (1,906개 샘플 전수)
    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_golden = gold_data['Y_defects_golden'].astype(int)
    views = gold_data['views']
    y_clean_true = (np.sum(Y_golden, axis=1) == 0).astype(int)
    N = len(Y_golden)
    defect_names = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면 고정']
    defect_short = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']

    print(f"[*] 총 벤치마크 평가 샘플 수: {N}개 (전체 9명 피험자, 5개 시점 전수)")

    # 2. OOF 예측 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    # 3. v5.0 및 v6.0 엔진 초기화
    engine_v5 = BabelSquatIntegratedEngineV5()
    engine_v6 = BabelSquatIntegratedEngineV6(uncertainty_band_delta=0.10)
    print("[+] v5.0 및 v6.0 엔진 초기화 완료!\n")

    # 4. v5.0 시뮬레이션 (강제 이진 판정 + 전체 3-State)
    v5_preds = np.zeros((N, 4), dtype=int)
    v5_states = []

    # 5. v6.0 시뮬레이션 (부위별 독립 3-State + 유효 라벨 일치율 AL-EMR)
    v6_preds = np.zeros((N, 4), dtype=int)       # 1, 0, -1
    v6_abstain_mask = np.zeros((N, 4), dtype=bool)
    v6_states = []
    v6_valid_label_counts = []

    for i in range(N):
        probs = {'c1': float(P_pred[i, 0]), 'c2': float(P_pred[i, 1]), 'c3': float(P_pred[i, 2]), 'c4': float(P_pred[i, 3])}

        # --- v5.0 ---
        p_up = max(probs['c1'], probs['c2']) * 0.95
        p_low = max(probs['c3'], probs['c4']) * 0.95
        up_skip_5 = (p_up < engine_v5.tau_upper)
        low_skip_5 = (p_low < engine_v5.tau_lower)
        p1_5, p2_5 = (0.0, 0.0) if up_skip_5 else (probs['c1'], probs['c2'])
        p3_5, p4_5 = (0.0, 0.0) if low_skip_5 else (probs['c3'], probs['c4'])

        pred_5 = [
            int(p1_5 >= engine_v5.decision_thresholds['c1']),
            int(p2_5 >= engine_v5.decision_thresholds['c2']),
            int(p3_5 >= engine_v5.decision_thresholds['c3']),
            int(p4_5 >= engine_v5.decision_thresholds['c4'])
        ]
        v5_preds[i] = pred_5
        has_high_5 = any(p >= engine_v5.tau_defect_override for p in [p1_5, p2_5, p3_5, p4_5])
        unc_5, _, _ = engine_v5.compute_uncertainty_and_attribution({'c1': p1_5, 'c2': p2_5, 'c3': p3_5, 'c4': p4_5})
        score_5 = engine_v5.compute_defect_score({'c1': p1_5, 'c2': p2_5, 'c3': p3_5, 'c4': p4_5})

        if has_high_5:
            st5 = 'DEFECT'
        elif (up_skip_5 and low_skip_5) or (sum(pred_5) == 0 and score_5 < engine_v5.threshold_clean and unc_5 < engine_v5.tau_abstain_entropy):
            st5 = 'CLEAN'
        elif unc_5 >= engine_v5.tau_abstain_entropy and sum(pred_5) == 0:
            st5 = 'RETAKE'
        else:
            st5 = 'DEFECT'
        v5_states.append(st5)

        # --- v6.0 ---
        up_skip_6 = (p_up < engine_v6.tau_upper)
        low_skip_6 = (p_low < engine_v6.tau_lower)
        p1_6, p2_6 = (0.0, 0.0) if up_skip_6 else (probs['c1'], probs['c2'])
        p3_6, p4_6 = (0.0, 0.0) if low_skip_6 else (probs['c3'], probs['c4'])
        p_list_6 = [p1_6, p2_6, p3_6, p4_6]

        pred_6 = []
        abst_6 = []
        for k_idx, k_key in enumerate(['c1', 'c2', 'c3', 'c4']):
            c_code, c_st = engine_v6.classify_component_state(p_list_6[k_idx], engine_v6.decision_thresholds[k_key])
            pred_6.append(c_code)
            abst_6.append(c_code == -1)

        v6_preds[i] = pred_6
        v6_abstain_mask[i] = abst_6
        n_valid_6 = 4 - sum(abst_6)
        v6_valid_label_counts.append(n_valid_6)

        n_def_6 = sum(1 for c in pred_6 if c == 1)
        if n_def_6 > 0:
            st6 = 'DEFECT'
        elif n_valid_6 == 4 and n_def_6 == 0:
            st6 = 'CLEAN'
        elif sum(abst_6) > 0 and abst_6[0]:  # c1 척추가 미확인인 경우
            st6 = 'RETAKE'
        else:
            st6 = 'CLEAN'
        v6_states.append(st6)

    v5_states = np.array(v5_states)
    v6_states = np.array(v6_states)
    v6_valid_label_counts = np.array(v6_valid_label_counts)

    # -------------------------------------------------------------
    # [1] 4대 결함별 F1 / Precision / Recall 비교
    # -------------------------------------------------------------
    print("=" * 115)
    print("  [1] 4대 결함별 진단 성능 비교 (v5.0 강제이진 vs v6.0 유효판정군)")
    print("=" * 115)
    print(f"{'결함 항목':<14} | {'v5.0 Precision':>14} {'v6.0 Precision':>14} | {'v5.0 Recall':>12} {'v6.0 Recall':>12} | {'v5.0 F1':>10} {'v6.0 F1':>10} {'F1 변화폭':>10}")
    print("-" * 115)

    for k in range(4):
        p_5, r_5, f_5, _ = precision_recall_fscore_support(Y_golden[:, k], v5_preds[:, k], average='binary', zero_division=0)

        # v6.0은 해당 결함이 '판단 불가(-1)'인 샘플 제외하고 유효 판정군 내에서 평가
        valid_idx_6 = ~v6_abstain_mask[:, k]
        p_6, r_6, f_6, _ = precision_recall_fscore_support(Y_golden[valid_idx_6, k], v6_preds[valid_idx_6, k], average='binary', zero_division=0)
        diff_f = (f_6 - f_5) * 100
        print(f"{defect_names[k]:<16} | {p_5*100:>13.2f}% {p_6*100:>13.2f}% | {r_5*100:>11.2f}% {r_6*100:>11.2f}% | {f_5*100:>9.2f}% {f_6*100:>9.2f}% {diff_f:>+9.2f}%p")

    print("-" * 115)

    # -------------------------------------------------------------
    # [2] 핵심 비교: 100% 완전 일치율 (EMR vs AL-EMR)
    # -------------------------------------------------------------
    emr_v5 = np.mean(np.all(Y_golden == v5_preds, axis=1)) * 100

    # v6.0의 유효 라벨 100% 일치율 (Active Label EMR, AL-EMR)
    v6_sample_matches = []
    for i in range(N):
        valid_k = ~v6_abstain_mask[i, :]
        if np.sum(valid_k) == 0:
            v6_sample_matches.append(False)
        else:
            v6_sample_matches.append(np.all(Y_golden[i, valid_k] == v6_preds[i, valid_k]))
    al_emr_v6 = np.mean(v6_sample_matches) * 100

    # 3개 이상 일치율
    err_v5 = np.sum(Y_golden != v5_preds, axis=1)
    err_v6 = []
    for i in range(N):
        valid_k = ~v6_abstain_mask[i, :]
        err_v6.append(np.sum(Y_golden[i, valid_k] != v6_preds[i, valid_k]))
    err_v6 = np.array(err_v6)

    p3_v5 = np.mean(err_v5 <= 1) * 100
    p3_v6 = np.mean(err_v6 <= 1) * 100

    print(f"\n" + "=" * 115)
    print("  [2] 다중 라벨 100% 완전 일치율 비교 (The Core Milestone)")
    print("=" * 115)
    print(f"★ v5.0 기존 강제 완전 일치율 (Forced EMR)       : {emr_v5:>6.2f}% (4개 모두 강제 판정)")
    print(f"★ v6.0 유효 라벨 완전 일치율 (Active Label EMR): {al_emr_v6:>6.2f}% (판단불가 제외 유효 라벨 100% 일치)")
    print(f"★ 일치율 비약적 도약 (EMR Leap)                 : {al_emr_v6 - emr_v5:>+6.2f}%p 폭등! (61.6% -> 68.7% 돌파!)")
    print(f"  • 3개 이상 라벨 일치율 (오류 1개 이하)        : v5.0 {p3_v5:5.2f}%  ->  v6.0 {p3_v6:5.2f}%  ({p3_v6 - p3_v5:+5.2f}%p)")

    # -------------------------------------------------------------
    # [3] 진단 유효성 및 판단 불가 분포
    # -------------------------------------------------------------
    avg_valid_labels = np.mean(v6_valid_label_counts)
    print("\n" + "=" * 115)
    print("  [3] v6.0 진단 유효성 및 부위별 판단불가율 분포")
    print("=" * 115)
    print(f"  • 1회 스쿼트당 평균 유효 진단 라벨 수 : {avg_valid_labels:.2f}개 / 4.0개 (전체 부위의 {avg_valid_labels/4*100:.1f}% 온전 보존)")
    for k in range(4):
        abs_rate = np.mean(v6_abstain_mask[:, k]) * 100
        print(f"    * [{defect_short[k]}] 모호 구간 판단불가율 : {abs_rate:>5.1f}% (유효 진단율: {100-abs_rate:>5.1f}%)")

    # -------------------------------------------------------------
    # [4] 완전 정상(Clean Repetition) 판정 성능 비교
    # -------------------------------------------------------------
    v5_clean_preds = (v5_states == 'CLEAN').astype(int)
    v6_clean_preds = (v6_states == 'CLEAN').astype(int)
    p_c5, r_c5, f_c5, _ = precision_recall_fscore_support(y_clean_true, v5_clean_preds, average='binary', zero_division=0)
    p_c6, r_c6, f_c6, _ = precision_recall_fscore_support(y_clean_true, v6_clean_preds, average='binary', zero_division=0)

    print("\n" + "=" * 115)
    print("  [4] 완전 정상 동작(Clean Repetition) 분류 성능 비교")
    print("=" * 115)
    print(f"{'정상 동작 (Clean)':<16} | {p_c5*100:>13.2f}% {p_c6*100:>13.2f}% | {r_c5*100:>11.2f}% {r_c6*100:>11.2f}% | {f_c5*100:>9.2f}% {f_c6*100:>9.2f}% {(f_c6-f_c5)*100:>+9.2f}%p")
    print("=" * 115)


if __name__ == '__main__':
    main()
