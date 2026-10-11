# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: benchmark_golden_labels_and_emr.py
Purpose: 사용자 57선 전수 조사 기반 골든 정답 라벨(Golden Ground Truth) 적용 후
         - 4대 결함별 정밀도(Precision), 재현율(Recall), F1-Score 변화
         - 완전 정상(Clean Repetition) 분류 성능 변화
         - 4대 라벨 100% 완전 일치율(Exact Match Ratio, EMR) 변화
         - 3-State 선택적 분류 시스템 커버리지 및 F1 종합 비교
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
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"
GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"


def evaluate_multilabel(Y_true, P_pred, thresholds=[0.5, 0.5, 0.5, 0.5]):
    """다중 라벨 및 EMR, 정상 F1 평가"""
    Y_pred = np.zeros_like(Y_true)
    for k in range(4):
        Y_pred[:, k] = (P_pred[:, k] >= thresholds[k]).astype(int)

    defect_names = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']
    metrics = {}

    for k, d in enumerate(defect_names):
        p, r, f1, _ = precision_recall_fscore_support(Y_true[:, k], Y_pred[:, k], average='binary', zero_division=0)
        acc = accuracy_score(Y_true[:, k], Y_pred[:, k])
        metrics[d] = {'precision': p, 'recall': r, 'f1': f1, 'acc': acc}

    # 완전 정상 (Clean Rep: 4개 결함 모두 0)
    y_clean_true = (np.sum(Y_true, axis=1) == 0).astype(int)
    y_clean_pred = (np.sum(Y_pred, axis=1) == 0).astype(int)
    p_clean, r_clean, f1_clean, _ = precision_recall_fscore_support(y_clean_true, y_clean_pred, average='binary', zero_division=0)
    metrics['clean'] = {'precision': p_clean, 'recall': r_clean, 'f1': f1_clean}

    # 4대 라벨 100% 완전 일치율 (Exact Match Ratio)
    emr = np.mean(np.all(Y_true == Y_pred, axis=1)) * 100
    metrics['emr'] = emr

    # 오류 개수 일치율 분포 (0개 틀림, 1개 틀림, 2개 틀림 ...)
    diff_counts = np.sum(Y_true != Y_pred, axis=1)
    metrics['diff_dist'] = {
        '0_error(EMR)': np.mean(diff_counts == 0) * 100,
        '1_error': np.mean(diff_counts == 1) * 100,
        '2_errors': np.mean(diff_counts == 2) * 100,
        '>=3_errors': np.mean(diff_counts >= 3) * 100,
    }

    return metrics, Y_pred


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 골든 정답 라벨 적용 전/후 5-Fold 다중 라벨 및 EMR 비교 벤치마크")
    print("=" * 115)

    # 1. 데이터 로드
    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_raw = gold_data['Y_defects_raw'].astype(int)
    Y_golden = gold_data['Y_defects_golden'].astype(int)
    is_audited = gold_data['is_audited'].astype(bool)
    defect_names = gold_data['defect_names']

    # 2. OOF 예측 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    # 3. 최적 임계값 (Phase 2에서 도출된 최적 임계값)
    thresholds = [0.45, 0.40, 0.50, 0.45]

    # [A] 전체 데이터셋 (1,906개) 기준 원천 라벨 vs 골든 라벨 평가
    m_raw_all, _ = evaluate_multilabel(Y_raw, P_pred, thresholds)
    m_gold_all, _ = evaluate_multilabel(Y_golden, P_pred, thresholds)

    # [B] 사용자 검증 대상 샘플 (213개) 기준 원천 라벨 vs 골든 라벨 평가
    m_raw_sub, _ = evaluate_multilabel(Y_raw[is_audited], P_pred[is_audited], thresholds)
    m_gold_sub, _ = evaluate_multilabel(Y_golden[is_audited], P_pred[is_audited], thresholds)

    print("\n" + "=" * 115)
    print("  [1] 전체 1,906개 샘플 대상: 원천 라벨 vs 골든 정답 라벨 성능 비교")
    print("=" * 115)
    print(f"{'결함 항목':<12} | {'원천 Precision':>14} {'골든 Precision':>14} | {'원천 Recall':>12} {'골든 Recall':>12} | {'원천 F1':>10} {'골든 F1':>10} {'변화폭':>10}")
    print("-" * 115)

    for d in defect_names:
        r_p = m_raw_all[d]['precision'] * 100
        g_p = m_gold_all[d]['precision'] * 100
        r_r = m_raw_all[d]['recall'] * 100
        g_r = m_gold_all[d]['recall'] * 100
        r_f = m_raw_all[d]['f1'] * 100
        g_f = m_gold_all[d]['f1'] * 100
        diff = g_f - r_f
        print(f"{d:<14} | {r_p:>13.2f}% {g_p:>13.2f}% | {r_r:>11.2f}% {g_r:>11.2f}% | {r_f:>9.2f}% {g_f:>9.2f}% {diff:>+9.2f}%p")

    print("-" * 115)
    # 정상 분류 성능
    c_rp = m_raw_all['clean']['precision'] * 100
    c_gp = m_gold_all['clean']['precision'] * 100
    c_rr = m_raw_all['clean']['recall'] * 100
    c_gr = m_gold_all['clean']['recall'] * 100
    c_rf = m_raw_all['clean']['f1'] * 100
    c_gf = m_gold_all['clean']['f1'] * 100
    print(f"{'정상(Clean)':<14} | {c_rp:>13.2f}% {c_gp:>13.2f}% | {c_rr:>11.2f}% {c_gr:>11.2f}% | {c_rf:>9.2f}% {c_gf:>9.2f}% {c_gf-c_rf:>+9.2f}%p")

    print("-" * 115)
    print(f"★ 4대 라벨 100% 완전 일치율 (EMR): {m_raw_all['emr']:.2f}%  ->  {m_gold_all['emr']:.2f}%  ({m_gold_all['emr']-m_raw_all['emr']:+5.2f}%p 상승!)")
    print(f"  * 0개 오류 (완전 일치) : {m_raw_all['diff_dist']['0_error(EMR)']:.2f}% -> {m_gold_all['diff_dist']['0_error(EMR)']:.2f}%")
    print(f"  * 1개 이하 오류 (3개 이상 일치) : {m_raw_all['diff_dist']['0_error(EMR)']+m_raw_all['diff_dist']['1_error']:.2f}% -> {m_gold_all['diff_dist']['0_error(EMR)']+m_gold_all['diff_dist']['1_error']:.2f}%")
    print(f"  * 2개 오류 : {m_raw_all['diff_dist']['2_errors']:.2f}% -> {m_gold_all['diff_dist']['2_errors']:.2f}%")
    print(f"  * 3개 이상 오류 : {m_raw_all['diff_dist']['>=3_errors']:.2f}% -> {m_gold_all['diff_dist']['>=3_errors']:.2f}%")

    print("\n" + "=" * 115)
    print("  [2] 사용자 직접 검증 대상 213개 샘플 심층 집중 분석")
    print("=" * 115)
    print(f"{'결함 항목':<12} | {'원천 Precision':>14} {'골든 Precision':>14} | {'원천 Recall':>12} {'골든 Recall':>12} | {'원천 F1':>10} {'골든 F1':>10} {'변화폭':>10}")
    print("-" * 115)

    for d in defect_names:
        r_p = m_raw_sub[d]['precision'] * 100
        g_p = m_gold_sub[d]['precision'] * 100
        r_r = m_raw_sub[d]['recall'] * 100
        g_r = m_gold_sub[d]['recall'] * 100
        r_f = m_raw_sub[d]['f1'] * 100
        g_f = m_gold_sub[d]['f1'] * 100
        diff = g_f - r_f
        print(f"{d:<14} | {r_p:>13.2f}% {g_p:>13.2f}% | {r_r:>11.2f}% {g_r:>11.2f}% | {r_f:>9.2f}% {g_f:>9.2f}% {diff:>+9.2f}%p")

    print("-" * 115)
    print(f"★ 검증군 4대 라벨 100% 완전 일치율 (EMR): {m_raw_sub['emr']:.2f}%  ->  {m_gold_sub['emr']:.2f}%  ({m_gold_sub['emr']-m_raw_sub['emr']:+5.2f}%p 폭발적 상승!)")
    print(f"  * 0개 오류 (완전 일치) : {m_raw_sub['diff_dist']['0_error(EMR)']:.2f}% -> {m_gold_sub['diff_dist']['0_error(EMR)']:.2f}%")
    print(f"  * 1개 이하 오류 (3개 이상 일치) : {m_raw_sub['diff_dist']['0_error(EMR)']+m_raw_sub['diff_dist']['1_error']:.2f}% -> {m_gold_sub['diff_dist']['0_error(EMR)']+m_gold_sub['diff_dist']['1_error']:.2f}%")
    print("=" * 115)


if __name__ == '__main__':
    main()
