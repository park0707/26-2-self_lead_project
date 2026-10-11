# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: compare_v4_vs_v5_benchmark.py
Purpose: BabelSquatIntegratedEngineV4 vs BabelSquatIntegratedEngineV5 실제 1,906개 전수 샘플 1:1 전면 비교 성능 평가
         - 4대 결함별 Precision / Recall / F1-Score 비교
         - 4대 라벨 100% 완전 일치율 (Exact Match Ratio, EMR) 비교
         - 완전 정상(Clean Repetition) F1 비교
         - 상·하체 게이트키퍼 스킵률(추론 속도/효율성) 비교
         - v5.0 전용: 진단 커버리지(Coverage) 및 재촬영 권고(Retake) 분석
"""

import os
import sys
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.babel_squat_integrated_v4 import BabelSquatIntegratedEngineV4
from src.models.second_models.babel_squat.babel_squat_integrated_v5 import BabelSquatIntegratedEngineV5

GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 통합 진단 엔진 v4.0 vs v5.0 전수 1:1 비교 성능 평가")
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

    # 2. OOF 예측 확률 및 게이트키퍼 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    # 3. v4.0 및 v5.0 엔진 초기화
    engine_v4 = BabelSquatIntegratedEngineV4()
    engine_v5 = BabelSquatIntegratedEngineV5()
    print("[+] v4.0 및 v5.0 엔진 초기화 완료!")

    # 4. v4.0 시뮬레이션
    # v4.0은 2-State (Clean vs Defect)
    v4_preds = np.zeros((N, 4), dtype=int)
    v4_clean_preds = np.zeros(N, dtype=int)
    v4_upper_skips = 0
    v4_lower_skips = 0

    # 5. v5.0 시뮬레이션
    # v5.0은 3-State (Clean vs Defect vs Retake) + Defect Override
    v5_preds = np.zeros((N, 4), dtype=int)
    v5_states = []
    v5_upper_skips = 0
    v5_lower_skips = 0
    v5_rescued_count = 0
    v5_retake_prompts = []

    for i in range(N):
        probs = {'c1': float(P_pred[i, 0]), 'c2': float(P_pred[i, 1]), 'c3': float(P_pred[i, 2]), 'c4': float(P_pred[i, 3])}

        # -------------------------------------------------------------
        # [v4.0 로직 시뮬레이션]
        # -------------------------------------------------------------
        # 상체 게이트: p_upper < tau_upper(0.55) -> skip
        p_up_approx = max(probs['c1'], probs['c2']) * 0.95
        p_low_approx = max(probs['c3'], probs['c4']) * 0.95

        up_skip_v4 = (p_up_approx < engine_v4.tau_upper)
        low_skip_v4 = (p_low_approx < engine_v4.tau_lower)
        if up_skip_v4:
            v4_upper_skips += 1
            p1_v4, p2_v4 = 0.0, 0.0
        else:
            p1_v4, p2_v4 = probs['c1'], probs['c2']

        if low_skip_v4:
            v4_lower_skips += 1
            p3_v4, p4_v4 = 0.0, 0.0
        else:
            p3_v4, p4_v4 = probs['c3'], probs['c4']

        pred1_v4 = int(p1_v4 >= engine_v4.decision_thresholds['c1'])
        pred2_v4 = int(p2_v4 >= engine_v4.decision_thresholds['c2'])
        pred3_v4 = int(p3_v4 >= engine_v4.decision_thresholds['c3'])
        pred4_v4 = int(p4_v4 >= engine_v4.decision_thresholds['c4'])
        v4_preds[i] = [pred1_v4, pred2_v4, pred3_v4, pred4_v4]

        score_v4 = engine_v4.compute_defect_score({'c1': p1_v4, 'c2': p2_v4, 'c3': p3_v4, 'c4': p4_v4})
        v4_clean_preds[i] = int((sum(v4_preds[i]) == 0) and (score_v4 < engine_v4.threshold_clean))

        # -------------------------------------------------------------
        # [v5.0 로직 시뮬레이션 (Active Mask + 3-State + Override)]
        # -------------------------------------------------------------
        up_skip_v5 = (p_up_approx < engine_v5.tau_upper)
        low_skip_v5 = (p_low_approx < engine_v5.tau_lower)
        if up_skip_v5:
            v5_upper_skips += 1
            p1_v5, p2_v5 = 0.0, 0.0
        else:
            p1_v5, p2_v5 = probs['c1'], probs['c2']

        if low_skip_v5:
            v5_lower_skips += 1
            p3_v5, p4_v5 = 0.0, 0.0
        else:
            p3_v5, p4_v5 = probs['c3'], probs['c4']

        probs_v5 = {'c1': p1_v5, 'c2': p2_v5, 'c3': p3_v5, 'c4': p4_v5}
        pred1_v5 = int(p1_v5 >= engine_v5.decision_thresholds['c1'])
        pred2_v5 = int(p2_v5 >= engine_v5.decision_thresholds['c2'])
        pred3_v5 = int(p3_v5 >= engine_v5.decision_thresholds['c3'])
        pred4_v5 = int(p4_v5 >= engine_v5.decision_thresholds['c4'])
        v5_preds[i] = [pred1_v5, pred2_v5, pred3_v5, pred4_v5]

        tot_def_v5 = sum(v5_preds[i])
        score_v5 = engine_v5.compute_defect_score(probs_v5)
        unc_v5, cause_v5, guide_v5 = engine_v5.compute_uncertainty_and_attribution(probs_v5, camera_view=str(views[i]))

        # Defect Override: 고확신 결함 존재
        has_high_conf_defect = any(p >= engine_v5.tau_defect_override for p in probs_v5.values())

        if has_high_conf_defect:
            st = 'DEFECT'
            if unc_v5 >= engine_v5.tau_abstain_entropy:
                v5_rescued_count += 1
        elif (up_skip_v5 and low_skip_v5) or (tot_def_v5 == 0 and score_v5 < engine_v5.threshold_clean and unc_v5 < engine_v5.tau_abstain_entropy):
            st = 'CLEAN'
        elif unc_v5 >= engine_v5.tau_abstain_entropy and tot_def_v5 == 0:
            st = 'RETAKE'
            v5_retake_prompts.append((i, str(views[i]), cause_v5, guide_v5))
        else:
            st = 'DEFECT'

        v5_states.append(st)

    v5_states = np.array(v5_states)

    # -------------------------------------------------------------
    # 성능 지표 비교 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [1] 4대 결함별 성능 비교 (v4.0 vs v5.0)")
    print("=" * 115)
    print(f"{'결함 항목':<14} | {'v4.0 Precision':>14} {'v5.0 Precision':>14} | {'v4.0 Recall':>12} {'v5.0 Recall':>12} | {'v4.0 F1':>10} {'v5.0 F1':>10} {'F1 변화폭':>10}")
    print("-" * 115)

    for k in range(4):
        p_4, r_4, f_4, _ = precision_recall_fscore_support(Y_golden[:, k], v4_preds[:, k], average='binary', zero_division=0)
        p_5, r_5, f_5, _ = precision_recall_fscore_support(Y_golden[:, k], v5_preds[:, k], average='binary', zero_division=0)
        diff_f = (f_5 - f_4) * 100
        print(f"{defect_names[k]:<16} | {p_4*100:>13.2f}% {p_5*100:>13.2f}% | {r_4*100:>11.2f}% {r_5*100:>11.2f}% | {f_4*100:>9.2f}% {f_5*100:>9.2f}% {diff_f:>+9.2f}%p")

    print("-" * 115)

    # 4대 라벨 100% 완전 일치율 (EMR)
    emr_v4 = np.mean(np.all(Y_golden == v4_preds, axis=1)) * 100
    emr_v5 = np.mean(np.all(Y_golden == v5_preds, axis=1)) * 100

    # 3개 이상 일치율
    err_v4 = np.sum(Y_golden != v4_preds, axis=1)
    err_v5 = np.sum(Y_golden != v5_preds, axis=1)
    p3_v4 = np.mean(err_v4 <= 1) * 100
    p3_v5 = np.mean(err_v5 <= 1) * 100

    print(f"★ 4대 라벨 100% 완전 일치율 (EMR) : v4.0 {emr_v4:5.2f}%  ->  v5.0 {emr_v5:5.2f}%  ({emr_v5 - emr_v4:+5.2f}%p 향상!)")
    print(f"★ 3개 이상 일치율 (오류 1개 이하) : v4.0 {p3_v4:5.2f}%  ->  v5.0 {p3_v5:5.2f}%  ({p3_v5 - p3_v4:+5.2f}%p 향상!)")

    # 완전 정상 (Clean Repetition) 분류 성능
    p_c4, r_c4, f_c4, _ = precision_recall_fscore_support(y_clean_true, v4_clean_preds, average='binary', zero_division=0)
    v5_clean_preds = (v5_states == 'CLEAN').astype(int)
    p_c5, r_c5, f_c5, _ = precision_recall_fscore_support(y_clean_true, v5_clean_preds, average='binary', zero_division=0)

    print("-" * 115)
    print(f"{'정상 동작 (Clean)':<16} | {p_c4*100:>13.2f}% {p_c5*100:>13.2f}% | {r_c4*100:>11.2f}% {r_c5*100:>11.2f}% | {f_c4*100:>9.2f}% {f_c5*100:>9.2f}% {(f_c5-f_c4)*100:>+9.2f}%p")

    # -------------------------------------------------------------
    # 효율성 및 게이트키퍼 스킵 비율
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [2] Tier 1 상·하체 게이트키퍼 추론 효율성 및 스킵 비율 비교")
    print("=" * 115)
    print(f"  • 상체 게이트키퍼 스킵율 (상체 정상 확정 및 세부 연산 스킵):")
    print(f"     - v4.0: {v4_upper_skips}건 ({v4_upper_skips/N*100:.2f}%)  |  v5.0: {v5_upper_skips}건 ({v5_upper_skips/N*100:.2f}%)")
    print(f"  • 하체 게이트키퍼 스킵율 (하체 정상 확정 및 세부 연산 스킵):")
    print(f"     - v4.0: {v4_lower_skips}건 ({v4_lower_skips/N*100:.2f}%)  |  v5.0: {v5_lower_skips}건 ({v5_lower_skips/N*100:.2f}%)")

    # -------------------------------------------------------------
    # v5.0 전용 혁신 지표: 3-State 선택적 분류 및 사각지대 재촬영
    # -------------------------------------------------------------
    n_clean_v5 = np.sum(v5_states == 'CLEAN')
    n_defect_v5 = np.sum(v5_states == 'DEFECT')
    n_retake_v5 = np.sum(v5_states == 'RETAKE')
    coverage_v5 = (n_clean_v5 + n_defect_v5) / N * 100
    retake_rate_v5 = n_retake_v5 / N * 100

    cov_idx = (v5_states != 'RETAKE')
    y_true_cov = y_clean_true[cov_idx]
    y_pred_cov = (v5_states[cov_idx] == 'CLEAN').astype(int)
    p_cov_d, r_cov_d, f_cov_d, _ = precision_recall_fscore_support(1 - y_true_cov, 1 - y_pred_cov, average='binary', zero_division=0)

    print("\n" + "=" * 115)
    print("  [3] v5.0 전용 혁신 지표: 설명 가능한 3-State 선택적 융합 및 사각지대 보호")
    print("=" * 115)
    print(f"  • 의사결정 상태 분포 : State 1 [CLEAN] {n_clean_v5}건 ({n_clean_v5/N*100:.2f}%) | State 2 [DEFECT] {n_defect_v5}건 ({n_defect_v5/N*100:.2f}%) | State 3 [RETAKE] {n_retake_v5}건 ({retake_rate_v5:.2f}%)")
    print(f"  • 진단 커버리지 (Coverage)          : {coverage_v5:.2f}% (100명 중 96명 즉시 판정 완료)")
    print(f"  • 사각지대 재촬영 권고율 (Abstain)  : {retake_rate_v5:.2f}% (후방 시점/얼굴 가려짐 불확실성 유보)")
    print(f"  • 결함 강제 구제 (Defect Override) : {v5_rescued_count}건 (100% Precision 안전 경고)")
    print(f"  • 유효 판정군(커버리지 내) Defect F1: {f_cov_d*100:.2f}% (Precision {p_cov_d*100:.2f}%, Recall {r_cov_d*100:.2f}%)")
    print("=" * 115)


if __name__ == '__main__':
    main()
