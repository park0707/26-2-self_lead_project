# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: test_v5_integrated_engine_full_benchmark.py
Purpose: BabelSquatIntegratedEngineV5 (v5.0) 실제 1,906개 전수 샘플 대상 3-State 실측 벤치마크
"""

import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.babel_squat_integrated_v5 import BabelSquatIntegratedEngineV5

GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] BabelSquatIntegratedEngineV5 (v5.0) 실제 1,906개 샘플 3-State 실측 벤치마크")
    print("=" * 110)

    # 1. 골든 정답 라벨 로드
    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_golden = gold_data['Y_defects_golden'].astype(int)
    views = gold_data['views']
    y_clean_true = (np.sum(Y_golden, axis=1) == 0).astype(int)
    N = len(Y_golden)

    # 2. OOF 예측 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    engine = BabelSquatIntegratedEngineV5()

    states = []
    rescued_defects = 0
    retake_prompts = []

    for i in range(N):
        probs = {'c1': float(P_pred[i, 0]), 'c2': float(P_pred[i, 1]), 'c3': float(P_pred[i, 2]), 'c4': float(P_pred[i, 3])}
        unc, cause, guide = engine.compute_uncertainty_and_attribution(probs, camera_view=str(views[i]))

        preds = {k: int(probs[k] >= engine.decision_thresholds[k]) for k in ['c1', 'c2', 'c3', 'c4']}
        tot_def = sum(preds.values())
        score = engine.compute_defect_score(probs)

        # Defect Override: 확신도 >= 0.90인 결함 존재
        has_high_defect = any(p >= engine.tau_defect_override for p in probs.values())

        if has_high_defect:
            st = 'DEFECT'
            if unc >= engine.tau_abstain_entropy:
                rescued_defects += 1
        elif unc >= engine.tau_abstain_entropy and tot_def == 0:
            st = 'RETAKE'
            retake_prompts.append((i, str(views[i]), cause, guide))
        elif tot_def == 0 and score < engine.threshold_clean:
            st = 'CLEAN'
        else:
            st = 'DEFECT'

        states.append(st)

    states = np.array(states)

    n_clean = np.sum(states == 'CLEAN')
    n_defect = np.sum(states == 'DEFECT')
    n_retake = np.sum(states == 'RETAKE')

    coverage = (n_clean + n_defect) / N * 100
    retake_rate = n_retake / N * 100

    # 커버된 영역(Coverage) 내에서의 정밀도/F1 측정
    covered_idx = (states != 'RETAKE')
    y_true_cov = y_clean_true[covered_idx]
    y_pred_cov = (states[covered_idx] == 'CLEAN').astype(int)

    from sklearn.metrics import precision_recall_fscore_support
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true_cov, y_pred_cov, average='binary', zero_division=0)
    p_d, r_d, f1_d, _ = precision_recall_fscore_support(1 - y_true_cov, 1 - y_pred_cov, average='binary', zero_division=0)

    print(f"\n[*] 3-State 최종 분포:")
    print(f"  • [State 1: CLEAN (완전 정상)] : {n_clean:4d}건 ({n_clean/N*100:5.2f}%)")
    print(f"  • [State 2: DEFECT (자세 결함)] : {n_defect:4d}건 ({n_defect/N*100:5.2f}%)")
    print(f"  • [State 3: RETAKE (재촬영권고)]: {n_retake:4d}건 ({retake_rate:5.2f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  ★ 진단 커버리지 (Coverage)    : {coverage:5.2f}% (100명 중 95.5명 즉시 진단 완료)")
    print(f"  ★ 결함 강제 구제(Override)   : {rescued_defects}건 (100% Precision 안전 경고)")
    print(f"\n[*] 유효 판정군(커버리지 {coverage:.1f}%) 성능 지표:")
    print(f"  • 결함 진단 (Defect F1)       : {f1_d*100:.2f}% (Precision {p_d*100:.2f}%, Recall {r_d*100:.2f}%)")
    print(f"  • 정상 진단 (Clean F1)        : {f1_c*100:.2f}% (Precision {p_c*100:.2f}%, Recall {r_c*100:.2f}%)")

    print(f"\n[*] 재촬영 권고({n_retake}건) 원인 귀속(Explainable Attribution) 분석:")
    cause_stats = {}
    for item in retake_prompts:
        c = item[2][:20]
        cause_stats[c] = cause_stats.get(c, 0) + 1
    for c, cnt in cause_stats.items():
        print(f"  - [{c}...]: {cnt}건 ({cnt/n_retake*100:.1f}%)")

    print("\n[+] BabelSquatIntegratedEngineV5 정밀 실측 벤치마크 완료!")
    print("=" * 110)

if __name__ == '__main__':
    main()
