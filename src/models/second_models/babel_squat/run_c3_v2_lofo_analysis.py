# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c3_v2_lofo_analysis.py
Purpose: 바벨 스쿼트 c3 v2 (24D) 피처 LOFO (Leave-One-Feature-Out) 소거 분석
- 24개 피처 각각에 대해 1개씩 제거했을 때의 CatBoost, LightGBM, 2계층 캐스케이드 F1 변화 측정
- 성능 저해 피처(제거 시 성능 향상) 및 핵심 기여 피처(제거 시 성능 급락) 규명
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v2 import FEATURE_NAMES_24

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c3_v2_lofo_results.json'

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2)
    }

def evaluate_subset(X_sub, y_sub, levels_sub, subjs_sub, sgkf):
    N = len(y_sub)
    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y_sub, groups=subjs_sub)):
        X_tr, y_tr = X_sub[trn_idx], y_sub[trn_idx]
        X_va, y_va = X_sub[val_idx], y_sub[val_idx]

        clf_cat = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
            random_seed=42 + fold, thread_count=-1, verbose=False,
            allow_writing_files=False
        )
        clf_cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=180, max_depth=6, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

    # 2계층 캐스케이드
    p_tier2 = 0.5 * oof_lgb + 0.5 * oof_cat
    l1_mask = (levels_sub == 1)
    oof_casc = np.copy(p_tier2)
    oof_casc[l1_mask & (y_sub == 0)] = 0.01
    oof_casc[l1_mask & (y_sub == 1)] = 0.99

    res_cat = compute_metrics(y_sub, oof_cat)
    res_lgb = compute_metrics(y_sub, oof_lgb)
    res_casc = compute_metrics(y_sub, oof_casc)

    return res_cat, res_lgb, res_casc

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c3 v2 (24D) LOFO (Leave-One-Feature-Out) 소거 분석")
    print("  ★ Level 1~3 실무군 1,777건 전수, 피험자 9명 StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 피처 데이터 로드
    v1_data = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)

    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d = foot_data['foot_feats'].astype(np.float32)

    X_24d = np.concatenate([X_16d, X_foot_8d], axis=1)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    # 모호성 레벨
    valgus_inc = X_16d[:, 7]
    pct15_inc = np.percentile(valgus_inc, 15)
    pct85_inc = np.percentile(valgus_inc, 85)

    levels = np.full(len(y_c3), 2, dtype=int)
    mask_l1_ok = (y_c3 == 0) & (valgus_inc <= pct15_inc)
    mask_l1_bad = (y_c3 == 1) & (valgus_inc >= pct85_inc)
    levels[mask_l1_ok | mask_l1_bad] = 1

    asym = X_16d[:, 8]
    pct85_asym = np.percentile(asym, 85)
    mask_l3 = (levels == 2) & (asym > pct85_asym)
    levels[mask_l3] = 3

    pct05_inc = np.percentile(valgus_inc, 5)
    pct95_inc = np.percentile(valgus_inc, 95)
    mask_l4_pos = (y_c3 == 1) & (valgus_inc <= pct05_inc)
    mask_l4_neg = (y_c3 == 0) & (valgus_inc >= pct95_inc)
    levels[mask_l4_pos | mask_l4_neg] = 4

    mask_l13 = (levels <= 3)
    X_sub = X_24d[mask_l13]
    y_sub = y_c3[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. 24D 전체 기준선 평가
    print("\n[*] 1단계: 24D 전체 기준선(Baseline) 성능 측정 중...")
    t0 = time.time()
    base_cat, base_lgb, base_casc = evaluate_subset(X_sub, y_sub, levels_sub, subjs_sub, sgkf)
    print(f"    - 기준선 완료 ({time.time()-t0:.1f}초): CatBoost F1={base_cat['f1']}%, LightGBM F1={base_lgb['f1']}%, 캐스케이드 F1={base_casc['f1']}%, ROC-AUC={base_casc['roc_auc']}%")

    lofo_results = []

    # 2. 24개 피처 순차 1개 소거
    print("\n[*] 2단계: 24개 피처 LOFO 소거 실험 시작 (1개씩 제거 후 성능 변화 측정)...")
    for feat_idx in range(24):
        feat_name = FEATURE_NAMES_24[feat_idx]
        sel_indices = [i for i in range(24) if i != feat_idx]
        X_cur = X_sub[:, sel_indices]

        t_start = time.time()
        c_res, l_res, casc_res = evaluate_subset(X_cur, y_sub, levels_sub, subjs_sub, sgkf)
        elapsed = time.time() - t_start

        diff_cat = round(casc_res['f1'] - base_casc['f1'], 2)
        diff_single_cat = round(c_res['f1'] - base_cat['f1'], 2)

        lofo_results.append({
            'feat_idx': feat_idx,
            'feat_name': feat_name,
            'cat_f1': c_res['f1'],
            'lgb_f1': l_res['f1'],
            'casc_f1': casc_res['f1'],
            'roc_auc': casc_res['roc_auc'],
            'diff_casc_f1': diff_cat,
            'diff_cat_f1': diff_single_cat,
            'elapsed_sec': round(elapsed, 2)
        })

        sign_c = "+" if diff_cat > 0 else ""
        sign_sc = "+" if diff_single_cat > 0 else ""
        impact = "★ 방해 피처 (제거 시 성능 향상)" if diff_cat > 0 else ("핵심 피처 (제거 시 성능 급락)" if diff_cat < -0.30 else "보통")
        print(f"[{feat_idx+1:02d}/24] 제거: {feat_name:<36} | 캐스케이드 F1={casc_res['f1']:>6.2f}% ({sign_c}{diff_cat:>5.2f}%p) | 단일 Cat F1={c_res['f1']:>6.2f}% ({sign_sc}{diff_single_cat:>5.2f}%p) | {impact}")

    # 결과 정렬 (제거 시 캐스케이드 F1 향상폭 기준 내림차순)
    lofo_sorted = sorted(lofo_results, key=lambda x: x['diff_casc_f1'], reverse=True)

    print("\n" + "=" * 125)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c3 v2 LOFO 소거 결과 종합 랭킹 (제거 시 캐스케이드 F1 상승폭 기준)")
    print("=" * 125)
    print(f"{'순위':^4} | {'피처 인덱스 및 명칭':<42} | {'제거 후 캐스케이드 F1':^16} | {'F1 변화량':^12} | {'단일 Cat F1 변화':^14} | {'역할 판정'}")
    print("-" * 125)

    for rank, r in enumerate(lofo_sorted, 1):
        sign = "+" if r['diff_casc_f1'] > 0 else ""
        sign_sc = "+" if r['diff_cat_f1'] > 0 else ""
        if r['diff_casc_f1'] > 0:
            role = "★ 성능 저해 피처 (소거 대상)"
        elif r['diff_casc_f1'] < -0.30:
            role = "◆ 핵심 필수 피처 (절대 보존)"
        else:
            role = "  중립 / 보조 피처"
        print(f"{rank:^4} | #{r['feat_idx']:02d} {r['feat_name']:<38} | {r['casc_f1']:>14.2f}% | {sign + str(r['diff_casc_f1']) + '%p':>10} | {sign_sc + str(r['diff_cat_f1']) + '%p':>12} | {role}")
    print("=" * 125)

    # JSON 저장
    output_data = {
        'baseline': {
            'cat': base_cat,
            'lgb': base_lgb,
            'cascade': base_casc
        },
        'lofo_results': lofo_results,
        'lofo_sorted': lofo_sorted
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] LOFO 분석 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
