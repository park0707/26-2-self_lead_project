# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Script: audit_stiff_d1_performance_ceiling.py
Purpose: 바벨 스티프 데드리프트 d1 (바벨-몸 밀착) 모델의 성능 한계 지점(Ceiling) 도달 여부
         실증 검증 및 4대 병목 요인(라벨 모호성, 시점 소실, 피험자 편차, 오라클 상한선) 전수 감사
Output:
  - D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_ceiling_audit_results.json
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import VIEW_ALIAS_MAP

OOF_CACHE_PKL = os.path.join(CURRENT_DIR, "sd3_v4_oof_cache.pkl")
BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd3_v4_bundle.pkl")
OUT_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_d1_ceiling_audit_results.json")


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 d1 성능 한계(Ceiling) 실증 검증 감사")
    print("=" * 80)

    with open(OOF_CACHE_PKL, 'rb') as fp:
        cache = pickle.load(fp)

    with open(BUNDLE_PATH, 'rb') as fp:
        bundle = pickle.load(fp)

    df = cache['df']
    y = cache['y']
    views = cache['views']
    subjs = cache['subjs']
    oof_gbdt = cache['oof_gbdt']
    oof_tcn = cache['oof_tcn']

    view_thresholds = bundle['view_thresholds']
    view_fusion_weights = bundle['view_fusion_weights']

    # 1. sd3_v4 융합 예측 확률 계산
    p_v4 = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        fw = view_fusion_weights[v]
        p_v4[i] = fw['gbdt'] * oof_gbdt[i] + fw['tcn'] * oof_tcn[i]

    # 2. 3-State 결정 여부 마스킹
    decided_mask = np.zeros(len(y), dtype=bool)
    pred_3s_labels = np.zeros(len(y), dtype=int)
    for idx in range(len(y)):
        v = views[idx]
        tc, td = view_thresholds[v]
        p = p_v4[idx]
        if p <= tc:
            decided_mask[idx] = True
            pred_3s_labels[idx] = 0
        elif p >= td:
            decided_mask[idx] = True
            pred_3s_labels[idx] = 1

    # 2-State 예측 (0.50 기준)
    pred_2s = (p_v4 >= 0.50).astype(int)
    is_err_2s = (pred_2s != y)

    # 3-State 확정 샘플 내 오분류
    is_err_3s = np.zeros(len(y), dtype=bool)
    for idx in range(len(y)):
        if decided_mask[idx]:
            is_err_3s[idx] = (pred_3s_labels[idx] != y[idx])

    tot_n = len(y)
    n_dec = np.sum(decided_mask)
    n_abst = tot_n - n_dec
    n_err_2s = np.sum(is_err_2s)
    n_err_3s = np.sum(is_err_3s)

    f1_2s = f1_score(y, pred_2s, zero_division=0) * 100.0
    auc_2s = roc_auc_score(y, p_v4) * 100.0
    f1_3s = f1_score(y[decided_mask], pred_3s_labels[decided_mask], zero_division=0) * 100.0
    acc_3s = accuracy_score(y[decided_mask], pred_3s_labels[decided_mask]) * 100.0

    print(f"\n[*] 1. sd3_v4 전체 기본 성적표 (N={tot_n})")
    print(f"  - 2-State (100% 전수 판정) : F1 = {f1_2s:.2f}%, AUC = {auc_2s:.2f}%, 오분류 = {n_err_2s}건 ({n_err_2s/tot_n*100:.2f}%)")
    print(f"  - 3-State (확정 판정 N={n_dec}) : F1 = {f1_3s:.2f}%, Acc = {acc_3s:.2f}%, 오분류 = {n_err_3s}건, 보류 = {n_abst}건 ({n_abst/tot_n*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 2. 오분류의 4대 원인별 심층 분해 (Decomposition)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  [*] 2. 오분류 샘플 4대 원인별 심층 분해 분석")
    print("=" * 80)

    # 원인 A: Key별(라벨 모호성 및 복합 결함 간섭) 오분류 분석
    key_breakdown = {}
    print("\n  [A. 시나리오 Key별 오분류 분포]")
    for k in sorted(df['key'].unique()):
        k_mask = (df['key'].values == k)
        k_tot = np.sum(k_mask)
        k_y = y[k_mask][0]
        k_err_2s = np.sum(is_err_2s[k_mask])
        k_err_3s = np.sum(is_err_3s[k_mask])
        k_abst = np.sum(~decided_mask[k_mask])
        k_f1 = f1_score(y[k_mask], pred_2s[k_mask], zero_division=0) * 100.0

        key_breakdown[k] = {
            'total': int(k_tot),
            'true_label': int(k_y),
            'err_2s': int(k_err_2s),
            'err_2s_rate': round(float(k_err_2s / k_tot * 100.0), 2),
            'err_3s': int(k_err_3s),
            'abstain': int(k_abst),
            'abstain_rate': round(float(k_abst / k_tot * 100.0), 2)
        }
        print(f"    * {k:7s} (y={k_y}): 총 {k_tot:3d}건 | 2S 오분류: {k_err_2s:2d}건 ({k_err_2s/k_tot*100:5.2f}%) | 3S 오분류: {k_err_3s:2d}건 | 기권: {k_abst:2d}건")

    # Tier 1 vs Tier 2 비교
    t1_mask = (df['is_tier1'].values == 1)
    t2_comp_mask = (df['key'].isin(['Key213', 'Key214']).values)

    t1_err_rate = np.mean(is_err_2s[t1_mask]) * 100.0
    t2_comp_err_rate = np.mean(is_err_2s[t2_comp_mask]) * 100.0

    print(f"\n    ★ [결정적 실증 데이터]:")
    print(f"      - Tier 1 순수 결함군 (Key 209 정상 vs Key 210 단독 결함, N=260): 오분류율 {t1_err_rate:.2f}% (정확도 {100-t1_err_rate:.2f}%)")
    print(f"      - 복합 결함군 (Key 213, 214: 기울임+이탈, N=260): 오분류율 {t2_comp_err_rate:.2f}% (정확도 {100-t2_comp_err_rate:.2f}%)")
    print(f"      => 복합 결함에서의 오분류율이 순수 결함 대비 {t2_comp_err_rate / (t1_err_rate+1e-5):.1f}배 폭증!")

    # 원인 B: 카메라 시점별 오분류 기여도
    print("\n  [B. 공식 5대 시점별 오분류 기여도]")
    view_breakdown = {}
    for vc, va in VIEW_ALIAS_MAP.items():
        v_mask = (views == vc)
        v_tot = np.sum(v_mask)
        v_err_2s = np.sum(is_err_2s[v_mask])
        v_err_3s = np.sum(is_err_3s[v_mask])
        v_abst = np.sum(~decided_mask[v_mask])
        v_f1_3s = f1_score(y[v_mask][decided_mask[v_mask]], pred_3s_labels[v_mask][decided_mask[v_mask]], zero_division=0) * 100.0

        view_breakdown[vc] = {
            'alias': va,
            'total': int(v_tot),
            'err_2s': int(v_err_2s),
            'err_2s_rate': round(float(v_err_2s / v_tot * 100.0), 2),
            'err_3s': int(v_err_3s),
            'abstain': int(v_abst),
            'f1_3s': round(float(v_f1_3s), 2)
        }
        print(f"    * {va:26s}: 총 {v_tot:3d}건 | 2S 오분류: {v_err_2s:2d}건 ({v_err_2s/v_tot*100:5.2f}%) | 3S 오분류: {v_err_3s:2d}건 | 3S F1: {v_f1_3s:5.2f}%")

    # 원인 C: 피험자별 편차 (Inter-Subject Variance)
    print("\n  [C. 피험자 9명 전원별 오분류 분포]")
    subj_breakdown = {}
    for s in sorted(df['subj'].unique()):
        s_mask = (subjs == s)
        s_tot = np.sum(s_mask)
        s_err_2s = np.sum(is_err_2s[s_mask])
        s_err_3s = np.sum(is_err_3s[s_mask])
        s_f1_3s = f1_score(y[s_mask][decided_mask[s_mask]], pred_3s_labels[s_mask][decided_mask[s_mask]], zero_division=0) * 100.0

        subj_breakdown[s] = {
            'total': int(s_tot),
            'err_2s': int(s_err_2s),
            'err_2s_rate': round(float(s_err_2s / s_tot * 100.0), 2),
            'err_3s': int(s_err_3s),
            'f1_3s': round(float(s_f1_3s), 2)
        }
        print(f"    * {s:7s}: 총 {s_tot:3d}건 | 2S 오분류: {s_err_2s:2d}건 ({s_err_2s/s_tot*100:5.2f}%) | 3S 오분류: {s_err_3s:2d}건 | 3S F1: {s_f1_3s:5.2f}%")

    # 원인 D: 마진 대역별 오분류 (Borderline Margin vs Hard Error)
    print("\n  [D. 예측 확률 마진 대역별 오분류 분포]")
    # 경계선 대역: 0.40 <= p <= 0.60
    # 확실한 정상 대역: p < 0.25 (실제 결함인 경우 극심한 FN)
    # 확실한 결함 대역: p > 0.75 (실제 정상인 경우 극심한 FP)
    borderline_mask = (p_v4 >= 0.40) & (p_v4 <= 0.60)
    hard_fp_mask = (p_v4 >= 0.75) & (y == 0)
    hard_fn_mask = (p_v4 <= 0.25) & (y == 1)

    n_borderline = np.sum(borderline_mask)
    n_borderline_err = np.sum(is_err_2s & borderline_mask)
    n_hard_fp = np.sum(hard_fp_mask)
    n_hard_fn = np.sum(hard_fn_mask)

    print(f"    * 경계선 대역 (0.40 <= P <= 0.60) : 총 {n_borderline}건 중 오분류 {n_borderline_err}건 ({n_borderline_err/n_err_2s*100:.1f}% of total errors)")
    print(f"    * 극심한 하드 FP (P >= 0.75인데 실제 정상 라벨): {n_hard_fp}건")
    print(f"    * 극심한 하드 FN (P <= 0.25인데 실제 결함 라벨): {n_hard_fn}건")

    # --------------------------------------------------------------------------
    # 3. 이론적 상한선 (Oracle & Ceiling) 실측 실험
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  [*] 3. 이론적 상한선 (Oracle Upper-Bound) 실측")
    print("=" * 80)

    # 1) 오라클 앙상블 상한선 (Oracle Ensemble Ceiling):
    # GBDT(36D)와 TCN(64F) 두 모델 중 하나라도 맞춘 경우 (모델 용량의 이론적 극대치)
    pred_gbdt = (oof_gbdt >= 0.50).astype(int)
    pred_tcn = (oof_tcn >= 0.50).astype(int)
    oracle_correct = ((pred_gbdt == y) | (pred_tcn == y))
    oracle_acc = np.mean(oracle_correct) * 100.0
    oracle_err = np.sum(~oracle_correct)

    # 오라클 F1 계산 (두 모델 중 하나라도 맞춘 쪽의 예측 선택)
    oracle_pred = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        if pred_gbdt[i] == y[i]:
            oracle_pred[i] = pred_gbdt[i]
        elif pred_tcn[i] == y[i]:
            oracle_pred[i] = pred_tcn[i]
        else:
            oracle_pred[i] = pred_gbdt[i]

    oracle_f1 = f1_score(y, oracle_pred, zero_division=0) * 100.0

    print(f"  ★ [오라클 앙상블 상한선 (GBDT or TCN 둘 중 하나라도 적중)]: F1 = {oracle_f1:.2f}%, Acc = {oracle_acc:.2f}% (잔여 불가항력 오류: {oracle_err}건 / {oracle_err/tot_n*100:.2f}%)")

    # 2) 권장 단일 시점 상한선 (View D 상한선)
    vd_mask = (views == 'view4')
    vd_f1_3s = f1_score(y[vd_mask][decided_mask[vd_mask]], pred_3s_labels[vd_mask][decided_mask[vd_mask]], zero_division=0) * 100.0
    vd_auc = roc_auc_score(y[vd_mask], p_v4[vd_mask]) * 100.0
    print(f"  ★ [최적 권장 시점(시점 D) 상한선]: 3-State F1 = {vd_f1_3s:.2f}%, AUC = {vd_auc:.2f}%")

    # 3) 노이즈 배제 골든셋 상한선 (Tier 1 Golden Upper Bound)
    t1_f1_3s = f1_score(y[t1_mask][decided_mask[t1_mask]], pred_3s_labels[t1_mask][decided_mask[t1_mask]], zero_division=0) * 100.0
    t1_auc = roc_auc_score(y[t1_mask], p_v4[t1_mask]) * 100.0
    print(f"  ★ [라벨 노이즈 배제(Tier 1 골든셋) 상한선]: 3-State F1 = {t1_f1_3s:.2f}%, AUC = {t1_auc:.2f}%")

    # 4) 멀티뷰 시점 결합 상한선 (Multi-View Consensus Ceiling)
    # 동일 시퀀스에 대해 5개 카메라 시점의 예측 확률을 앙상블했을 때의 성능
    df['pred_prob'] = p_v4
    df['pred_2s'] = pred_2s
    seq_grouped = df.groupby(['subj', 'key'])
    mv_y = []
    mv_prob = []
    for (s, k), grp in seq_grouped:
        mv_y.append(grp['y'].values[0])
        mv_prob.append(grp['pred_prob'].mean())
    mv_y = np.array(mv_y)
    mv_prob = np.array(mv_prob)
    mv_pred = (mv_prob >= 0.50).astype(int)
    mv_f1 = f1_score(mv_y, mv_pred, zero_division=0) * 100.0
    mv_auc = roc_auc_score(mv_y, mv_prob) * 100.0
    mv_acc = accuracy_score(mv_y, mv_pred) * 100.0
    print(f"  ★ [5대 시점 동시 관측 멀티뷰 합의(Multi-View Consensus) 상한선]: F1 = {mv_f1:.2f}%, Acc = {mv_acc:.2f}%, AUC = {mv_auc:.2f}% (N={len(mv_y)})")

    # 결과 JSON 저장
    audit_results = {
        'model_name': 'sd3_v4',
        'total_samples': tot_n,
        'overall_metrics': {
            'f1_2s': round(f1_2s, 2),
            'auc_2s': round(auc_2s, 2),
            'f1_3s': round(f1_3s, 2),
            'acc_3s': round(acc_3s, 2),
            'total_errors_2s': int(n_err_2s),
            'total_errors_3s': int(n_err_3s),
            'abstain_count': int(n_abst),
            'abstain_rate': round(float(n_abst / tot_n * 100.0), 2)
        },
        'key_breakdown': key_breakdown,
        'view_breakdown': view_breakdown,
        'subj_breakdown': subj_breakdown,
        'margin_breakdown': {
            'borderline_count': int(n_borderline),
            'borderline_err_count': int(n_borderline_err),
            'hard_fp_count': int(n_hard_fp),
            'hard_fn_count': int(n_hard_fn)
        },
        'ceilings': {
            'oracle_ensemble_f1': round(oracle_f1, 2),
            'oracle_ensemble_acc': round(oracle_acc, 2),
            'oracle_residual_unavoidable_errors': int(oracle_err),
            'view_d_upper_bound_f1': round(vd_f1_3s, 2),
            'view_d_auc': round(vd_auc, 2),
            'tier1_golden_f1': round(t1_f1_3s, 2),
            'tier1_golden_auc': round(t1_auc, 2),
            'multiview_consensus_f1': round(mv_f1, 2),
            'multiview_consensus_acc': round(mv_acc, 2),
            'multiview_consensus_auc': round(mv_auc, 2)
        }
    }

    with open(OUT_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(audit_results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] 한계 지점 실증 감사 결과 JSON 저장 완료: {OUT_JSON_PATH}")


if __name__ == '__main__':
    main()
