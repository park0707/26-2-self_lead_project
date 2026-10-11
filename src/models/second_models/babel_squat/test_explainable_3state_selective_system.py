# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: test_explainable_3state_selective_system.py
Purpose: [신규 기획안 실측 검증]
         설명 가능한 3-State 선택적 분류 시스템 (Explainable 3-State with Defect Override & Cause Attribution)
         공식 5-Fold 피험자 격리 벤치마크 및 기존 Phase 1 대안 B 대비 개선 효과 정밀 검증
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.babel_squat_phase1_meta_fusion import (
    SquatPhase1MetaFusionEngine
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'explainable_3state_benchmark_results.json')

DEFECT_KEYS = ['c1_spine', 'c2_gaze', 'c3_knee', 'c4_foot']
DEFECT_LABELS = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면 고정']
CAUSE_GUIDES = {
    0: "척추 각도가 모호합니다 (촬영 가이드: 카메라를 대각선 45도 방향, 1m 높이에 두고 다시 수행해 주세요)",
    1: "시선 방향이 흐릿합니다 (촬영 가이드: 정면을 안정적으로 응시하고 다시 수행해 주세요)",
    2: "무릎-발목 정렬이 가려졌습니다 (촬영 가이드: 하체가 잘 보이도록 카메라 거리를 2m 확보해 주세요)",
    3: "발바닥 지면 고정이 불명확합니다 (촬영 가이드: 발끝까지 밝게 보이도록 조명을 확인해 주세요)"
}


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 설명 가능한 3-State 선택적 분류 시스템 (Defect Override + Cause Attribution) 실측")
    print("=" * 115)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)
    subjects = ds['subjects'].astype(int)
    N_total = len(subjects)

    y_overall_true = (np.sum(Y_defects, axis=1) > 0).astype(int)
    y_clean_true = 1 - y_overall_true

    # 2. OOF 연속 확률 로드 (최신 확정 모델들)
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)

    P_matrix = np.vstack([p_c1, p_c2, p_c3, p_c4]).T  # (1906, 4)
    max_p_defect = np.max(P_matrix, axis=1)           # 최대 결함 강도
    margins = np.abs(P_matrix - 0.50)                 # 각 관절별 확신도 마진
    min_margin = np.min(margins, axis=1)              # 최소 확신도 마진
    argmin_margin = np.argmin(margins, axis=1)        # 가장 모호한 관절 인덱스 (0:c1, 1:c2, 2:c3, 3:c4)

    # 3. 5-Fold 메타러너 OOF 예측값 산출
    X_meta = SquatPhase1MetaFusionEngine.extract_meta_features(p_c1, p_c2, p_c3, p_c4)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_clean = np.zeros(N_total, dtype=np.float32)
    for trn, val in sgkf.split(X_meta, y_clean_true, groups=subjects):
        engine_fold = SquatPhase1MetaFusionEngine(c_reg=0.5, random_state=42)
        engine_fold.train_full(X_meta[trn], y_clean_true[trn])
        oof_p_clean[val] = engine_fold.predict_proba_clean(X_meta[val])

    # 4. [기준선] 기존 Phase 1 대안 B (Naïve Selective Classification)
    t_defect = 0.15
    t_clean = 0.45

    mask_abstain_naive = (oof_p_clean >= t_defect) & (oof_p_clean < t_clean)
    mask_decided_naive = ~mask_abstain_naive
    n_abstain_naive = int(np.sum(mask_abstain_naive))
    coverage_naive = (np.sum(mask_decided_naive) / N_total) * 100.0

    pred_clean_naive = (oof_p_clean[mask_decided_naive] >= 0.50).astype(int)
    pred_defect_naive = 1 - pred_clean_naive
    y_clean_naive = y_clean_true[mask_decided_naive]
    y_defect_naive = y_overall_true[mask_decided_naive]

    f1_clean_naive = f1_score(y_clean_naive, pred_clean_naive, zero_division=0) * 100.0
    prec_clean_naive = precision_score(y_clean_naive, pred_clean_naive, zero_division=0) * 100.0
    rec_clean_naive = recall_score(y_clean_naive, pred_clean_naive, zero_division=0) * 100.0
    f1_defect_naive = f1_score(y_defect_naive, pred_defect_naive, zero_division=0) * 100.0

    print("\n" + "=" * 100)
    print("  [1] 기준선: 기존 Phase 1 대안 B (Naïve Selective Classification)")
    print("=" * 100)
    print(f"  • 모호 구간 보류(재촬영): {n_abstain_naive}건 ({100-coverage_naive:.2f}%) | 판정 커버리지: {coverage_naive:.2f}%")
    print(f"  • Clean 성능 : Precision={prec_clean_naive:.2f}%, Recall={rec_clean_naive:.2f}%, F1={f1_clean_naive:.2f}%")
    print(f"  • Defect 성능: F1={f1_defect_naive:.2f}%")
    print(f"  • 한계점: 보류된 {n_abstain_naive}건 중 실제로는 명백한 결함이 있는 건이 섞여 있어도 무조건 재촬영 처리됨.")

    # 5. [신규 기획안] Defect Override (결함 구제) 임계값별 실측 실험
    print("\n" + "=" * 100)
    print("  [2] 신규 기획안: 명백한 결함 구제 (Defect Override) 임계값별 시뮬레이션")
    print("=" * 100)

    override_thresholds = [0.95, 0.90, 0.85, 0.80]
    override_results = {}

    print(f"{'Override 임계값':<15} | {'구제된 결함':<12} | {'실제 결함 적중률':<16} | {'최종 보류수(율)':<16} | {'커버리지':<10} | {'Clean F1':<10} | {'Defect F1':<10}")
    print("-" * 100)

    for tau_ov in override_thresholds:
        # 기본 보류 마스크
        mask_abstain = (oof_p_clean >= t_defect) & (oof_p_clean < t_clean)

        # 구제 대상: 보류 구간에 걸렸으나, 단일 관절 결함 확신도 max_p >= tau_ov 인 샘플
        mask_override = mask_abstain & (max_p_defect >= tau_ov)
        n_rescued = int(np.sum(mask_override))

        # 구제된 샘플들의 실제 레이블 검증
        if n_rescued > 0:
            actual_defects_in_rescued = np.sum(y_overall_true[mask_override] == 1)
            rescue_precision = (actual_defects_in_rescued / n_rescued) * 100.0
        else:
            actual_defects_in_rescued = 0
            rescue_precision = 100.0

        # 최종 3-State 결정
        # State: 0=Clean, 1=Defect, 2=Retake(Abstain)
        final_state = np.zeros(N_total, dtype=int)
        final_state[oof_p_clean < t_defect] = 1   # 결함 확정
        final_state[oof_p_clean >= t_clean] = 0   # 정상 확정
        final_state[mask_abstain] = 2             # 판정 보류

        # 결함 구제 적용: 보류(2) -> 결함(1)으로 전환
        final_state[mask_override] = 1

        # 최종 평가 (State 0, 1만 커버리지에 포함)
        mask_decided_new = (final_state != 2)
        n_abstain_final = int(np.sum(final_state == 2))
        coverage_new = (np.sum(mask_decided_new) / N_total) * 100.0

        pred_clean_new = (final_state[mask_decided_new] == 0).astype(int)
        pred_defect_new = (final_state[mask_decided_new] == 1).astype(int)
        y_clean_new = y_clean_true[mask_decided_new]
        y_defect_new = y_overall_true[mask_decided_new]

        f1_c_new = f1_score(y_clean_new, pred_clean_new, zero_division=0) * 100.0
        prec_c_new = precision_score(y_clean_new, pred_clean_new, zero_division=0) * 100.0
        rec_c_new = recall_score(y_clean_new, pred_clean_new, zero_division=0) * 100.0
        f1_d_new = f1_score(y_defect_new, pred_defect_new, zero_division=0) * 100.0

        override_results[str(tau_ov)] = {
            'tau_override': tau_ov,
            'rescued_count': n_rescued,
            'rescue_precision': rescue_precision,
            'final_abstain_count': n_abstain_final,
            'final_coverage': coverage_new,
            'clean_f1': f1_c_new,
            'clean_precision': prec_c_new,
            'clean_recall': rec_c_new,
            'defect_f1': f1_d_new
        }

        print(f"tau >= {tau_ov:<8.2f} | {n_rescued:>4}건 구제   | {rescue_precision:>12.2f}% ({actual_defects_in_rescued}/{n_rescued}) | {n_abstain_final:>4}건 ({100-coverage_new:>4.2f}%)  | {coverage_new:>8.2f}% | {f1_c_new:>8.2f}% | {f1_d_new:>8.2f}%")

    # 6. 최적 임계값 (tau_override = 0.90) 기준 원인 귀속 (Cause Attribution) 상세 분석
    best_tau_ov = 0.90
    mask_abstain_base = (oof_p_clean >= t_defect) & (oof_p_clean < t_clean)
    mask_override_best = mask_abstain_base & (max_p_defect >= best_tau_ov)
    mask_final_abstain = mask_abstain_base & (~mask_override_best)

    n_final_abstain = int(np.sum(mask_final_abstain))
    final_abstain_indices = np.where(mask_final_abstain)[0]

    print("\n" + "=" * 100)
    print(f"  [3] 설명 가능한 재촬영(Explainable Retake): 최적 임계값(tau={best_tau_ov:.2f}) 기준 원인 분석")
    print("=" * 100)
    print(f"  • 전체 {N_total}건 중 최종 판정 보류 대상: 총 {n_final_abstain}건 ({n_final_abstain/N_total*100:.2f}%)")

    cause_counts = {}
    cause_samples = {}
    for idx in final_abstain_indices:
        culprit = int(argmin_margin[idx])
        cause_counts[culprit] = cause_counts.get(culprit, 0) + 1
        if culprit not in cause_samples:
            cause_samples[culprit] = []
        cause_samples[culprit].append({
            'idx': int(idx),
            'p_clean': float(oof_p_clean[idx]),
            'probs': [float(P_matrix[idx, k]) for k in range(4)],
            'min_margin': float(margins[idx, culprit])
        })

    print(f"\n  [원인 귀속 분포: 어떤 관절이 판정 보류를 유발했는가?]")
    print(f"  {'모호 유발 관절':<25} | {'건수':<8} | {'비율':<8} | {'맞춤형 촬영 가이드'}")
    print("  " + "-" * 90)

    attribution_summary = {}
    for k in range(4):
        cnt = cause_counts.get(k, 0)
        ratio = (cnt / n_final_abstain) * 100.0 if n_final_abstain > 0 else 0.0
        attribution_summary[DEFECT_KEYS[k]] = {
            'name': DEFECT_LABELS[k],
            'count': cnt,
            'ratio': ratio,
            'guide': CAUSE_GUIDES[k]
        }
        print(f"  {DEFECT_LABELS[k]:<22} | {cnt:>6}건 | {ratio:>6.2f}% | {CAUSE_GUIDES[k][:45]}...")

    # 7. 구제된 케이스의 실제 결함 부위 상세 분석
    rescued_indices = np.where(mask_override_best)[0]
    rescued_details = []
    print(f"\n  [결함 구제 상세: 보류될 뻔했으나 결함으로 확정 구제된 케이스 (총 {len(rescued_indices)}건)]")
    for idx in rescued_indices:
        argmax_defect = np.argmax(P_matrix[idx])
        is_true_defect = (y_overall_true[idx] == 1)
        rescued_details.append({
            'idx': int(idx),
            'p_clean': float(oof_p_clean[idx]),
            'peak_defect': DEFECT_LABELS[argmax_defect],
            'peak_prob': float(P_matrix[idx, argmax_defect]),
            'is_true_defect': bool(is_true_defect)
        })
        print(f"    - 샘플 #{idx:04d} | P(Clean)={oof_p_clean[idx]:.3f} | 확신 부위: {DEFECT_LABELS[argmax_defect]} (P={P_matrix[idx, argmax_defect]*100:.1f}%) | 실제 결함 여부: {'[O] 결함 맞음' if is_true_defect else '[X] 정상'}")

    # 8. 종합 결과 JSON 저장
    full_benchmark_result = {
        'total_samples': N_total,
        'naive_selective_baseline': {
            'abstain_count': n_abstain_naive,
            'coverage': coverage_naive,
            'clean_f1': f1_clean_naive,
            'clean_precision': prec_clean_naive,
            'clean_recall': rec_clean_naive,
            'defect_f1': f1_defect_naive
        },
        'override_threshold_simulations': override_results,
        'best_configuration': {
            'tau_defect': t_defect,
            'tau_clean': t_clean,
            'tau_override': best_tau_ov,
            'final_abstain_count': n_final_abstain,
            'final_coverage': override_results[str(best_tau_ov)]['final_coverage'],
            'clean_f1': override_results[str(best_tau_ov)]['clean_f1'],
            'clean_precision': override_results[str(best_tau_ov)]['clean_precision'],
            'clean_recall': override_results[str(best_tau_ov)]['clean_recall'],
            'defect_f1': override_results[str(best_tau_ov)]['defect_f1'],
            'rescued_count': len(rescued_indices),
            'rescue_precision': override_results[str(best_tau_ov)]['rescue_precision']
        },
        'cause_attribution_summary': attribution_summary,
        'rescued_samples': rescued_details
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_benchmark_result, f, indent=2, ensure_ascii=False)

    print(f"\n[✔] 설명 가능한 3-State 벤치마크 결과 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
