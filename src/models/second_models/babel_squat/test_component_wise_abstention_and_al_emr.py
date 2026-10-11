# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: test_component_wise_abstention_and_al_emr.py
Purpose: 사용자 제안 '부위별 독립 판단불가 및 유효 라벨 기반 일치율(Active Label EMR)' 실증 시뮬레이션
         1) 서브모델별 애매한 구간(불확실성 구간) 판정불가(Abstain) 처리 시 개별 모델 F1/Acc 변화
         2) 판정불가 라벨을 제외한 유효 라벨 100% 일치율 (Active Label Exact Match Ratio, AL-EMR) 측정
         3) 판정불가 임계값 밴드(tau_band) 변화에 따른 EMR 상승 추이 실측
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
GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 부위별 독립 판단불가 도입 및 유효 라벨 일치율(AL-EMR) 실증 검증")
    print("=" * 115)

    # 1. 골든 정답 라벨 로드
    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_golden = gold_data['Y_defects_golden'].astype(int)
    N = len(Y_golden)
    defect_names = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']

    # 2. OOF 예측 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    base_thresholds = [0.45, 0.40, 0.50, 0.45]

    # [실험 1] 기존 강제 2진 판정 (Forced Binary) 기준 EMR
    Y_pred_forced = np.zeros_like(Y_golden)
    for k in range(4):
        Y_pred_forced[:, k] = (P_pred[:, k] >= base_thresholds[k]).astype(int)

    base_emr = np.mean(np.all(Y_golden == Y_pred_forced, axis=1)) * 100
    print(f"[*] 기존 방식 (강제 0/1 판정) 4대 라벨 100% 완전 일치율 (EMR): {base_emr:.2f}%\n")

    # [실험 2] 불확실성 밴드(Uncertainty Band) 크기에 따른 시뮬레이션
    # 기준 임계값 theta_k 주변 [theta_k - delta, theta_k + delta] 구간을 '판단불가(Abstain = -1)'로 처리
    delta_candidates = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]

    print(f"{'불확실성 밴드(±delta)':<18} | {'평균 판정불가율':>14} | {'c1 F1':>8} {'c2 F1':>8} {'c3 F1':>8} {'c4 F1':>8} | {'유효 라벨 EMR (AL-EMR)':>24} {'상승폭':>10}")
    print("-" * 115)

    for delta in delta_candidates:
        # 각 부위별 판정: 1(결함), 0(정상), -1(판단불가)
        Y_pred_tri = np.zeros_like(Y_golden)
        abstain_masks = np.zeros_like(Y_golden, dtype=bool)

        for k in range(4):
            th = base_thresholds[k]
            low_th = th - delta
            high_th = th + delta

            is_abstain = (P_pred[:, k] >= low_th) & (P_pred[:, k] <= high_th) if delta > 0 else np.zeros(N, dtype=bool)
            abstain_masks[:, k] = is_abstain

            # 판정
            Y_pred_tri[:, k] = np.where(is_abstain, -1, (P_pred[:, k] >= th).astype(int))

        # 부위별 F1 산출 (판단불가 샘플 제외 후 유효 판정군 내에서 F1 계산)
        f1_list = []
        for k in range(4):
            valid_idx = ~abstain_masks[:, k]
            if np.sum(valid_idx) > 0:
                _, _, f1, _ = precision_recall_fscore_support(Y_golden[valid_idx, k], Y_pred_tri[valid_idx, k], average='binary', zero_division=0)
            else:
                f1 = 0.0
            f1_list.append(f1 * 100)

        # 사용자 제안 핵심 평가: 유효 라벨 일치율 (Active Label EMR, AL-EMR)
        # 판정 불가가 아닌 라벨들만 대상으로 100% 일치했는지 판정!
        sample_all_matched = []
        valid_labels_per_sample = []

        for i in range(N):
            valid_k = ~abstain_masks[i, :]
            n_valid = np.sum(valid_k)
            valid_labels_per_sample.append(n_valid)

            if n_valid == 0:
                # 4개 모두 판정 불가인 극단 상황
                sample_all_matched.append(False)
            else:
                # 판정 유효한 라벨들만 참값과 100% 일치하는지 검사
                matched = np.all(Y_golden[i, valid_k] == Y_pred_tri[i, valid_k])
                sample_all_matched.append(matched)

        al_emr = np.mean(sample_all_matched) * 100
        avg_abstain_pct = np.mean(abstain_masks) * 100
        avg_valid_cnt = np.mean(valid_labels_per_sample)
        diff_emr = al_emr - base_emr

        band_str = f"±{delta:.2f} (확신구간만)" if delta > 0 else "0.00 (기존 강제판정)"
        print(f"{band_str:<18} | {avg_abstain_pct:>13.1f}% | {f1_list[0]:>7.1f}% {f1_list[1]:>7.1f}% {f1_list[2]:>7.1f}% {f1_list[3]:>7.1f}% | {al_emr:>20.2f}% (평균 {avg_valid_cnt:.1f}개 라벨) {diff_emr:>+9.2f}%p")

    print("=" * 115)


if __name__ == '__main__':
    main()
