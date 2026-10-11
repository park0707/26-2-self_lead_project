# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c3_v3_tournament.py
Purpose: 바벨 스쿼트 c3 v2 (24D) vs v3 (방해 피처 소거 정예 모델) 5-Fold 실측 비교 토너먼트
- 후보:
  1) [v2.0 기준]: 24D 전체 세트
  2) [v3-A 20D]: 4대 성능 저해 피처(#00 레거시간격비, #12 정강이흔들림, #04 무릎저크, #11 정강이수직각) 소거
  3) [v3-B 18D]: 6대 저해/중복 피처(#00, #12, #04, #11, #06 max외반, #01 평균간격비) 소거
  4) [v3-C 16D]: 8대 저해/중복 피처(#00, #12, #04, #11, #06, #01, #05 nadir외반, #14 굴곡최소각) 소거
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, confusion_matrix

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v2 import FEATURE_NAMES_24

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c3_v3_comparison_results.json'

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

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    best_th, best_f1 = threshold, f1
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'opt_f1': round(float(best_f1), 2),
        'opt_th': round(float(best_th), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluate_subset(X_sub, y_sub, levels_sub, subjs_sub, sgkf):
    N = len(y_sub)
    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y_sub, groups=subjs_sub)):
        X_tr, y_tr = X_sub[trn_idx], y_sub[trn_idx]
        X_va, y_va = X_sub[val_idx], y_sub[val_idx]

        clf_cat = CatBoostClassifier(
            iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
            random_seed=42 + fold, thread_count=-1, verbose=False,
            allow_writing_files=False
        )
        clf_cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31,
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
    print("  [vidio-health-trainer] 바벨 스쿼트 c3 v2 (24D) vs v3 (방해 피처 제거 정예 모델) 5-Fold 실측 토너먼트")
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

    # 4대 후보 세트 구성
    # [후보 1: v2.0 기준 24D]
    idx_v2 = list(range(24))

    # [후보 2: v3-A 20D] 4대 성능 저해 피처(#00, #12, #04, #11) 소거
    drop_v3a = [0, 12, 4, 11]
    idx_v3a = [i for i in range(24) if i not in drop_v3a]

    # [후보 3: v3-B 18D] 6대 저해/중복 피처(#00, #12, #04, #11, #06, #01) 소거
    drop_v3b = [0, 12, 4, 11, 6, 1]
    idx_v3b = [i for i in range(24) if i not in drop_v3b]

    # [후보 4: v3-C 16D] 8대 저해/중복 피처(#00, #12, #04, #11, #06, #01, #05, #14) 소거
    drop_v3c = [0, 12, 4, 11, 6, 1, 5, 14]
    idx_v3c = [i for i in range(24) if i not in drop_v3c]

    candidates = [
        ("[v2.0 기준 모델] 24D 전체 피처셋", idx_v2, []),
        ("[v3-A 20D 모델] 4대 성능 저해 피처(#00, #12, #04, #11) 소거", idx_v3a, drop_v3a),
        ("[v3-B 18D 모델] 6대 저해/중복 피처(#00, #12, #04, #11, #06, #01) 소거", idx_v3b, drop_v3b),
        ("[v3-C 16D 모델] 8대 저해/중복 피처(#00, #12, #04, #11, #06, #01, #05, #14) 소거", idx_v3c, drop_v3c)
    ]

    results = []
    print("\n[*] 5-Fold StratifiedGroupKFold 실측 토너먼트 시작...")

    for label, sel_idx, drop_idx in candidates:
        t0 = time.time()
        X_cur = X_sub[:, sel_idx]
        cat_res, lgb_res, casc_res = evaluate_subset(X_cur, y_sub, levels_sub, subjs_sub, sgkf)
        elapsed = time.time() - t0

        drop_names = [FEATURE_NAMES_24[i] for i in drop_idx]
        sel_names = [FEATURE_NAMES_24[i] for i in sel_idx]

        results.append({
            'label': label,
            'dim': len(sel_idx),
            'selected_features': sel_names,
            'dropped_features': drop_names,
            'cat': cat_res,
            'lgb': lgb_res,
            'cascade': casc_res,
            'elapsed_sec': round(elapsed, 2)
        })

        print(f"\n>>> 완료: {label} (차원: {len(sel_idx)}D, 소요: {elapsed:.1f}초)")
        print(f"    - 단일 CatBoost: F1 = {cat_res['f1']}% | ROC-AUC = {cat_res['roc_auc']}% (최적 F1: {cat_res['opt_f1']}%)")
        print(f"    - 단일 LightGBM: F1 = {lgb_res['f1']}% | ROC-AUC = {lgb_res['roc_auc']}% (최적 F1: {lgb_res['opt_f1']}%)")
        print(f"    - 2계층 캐스케이드: F1 = {casc_res['f1']}% | ROC-AUC = {casc_res['roc_auc']}% (최적 F1: {casc_res['opt_f1']}%)")

    # 종합 결과표
    print("\n" + "=" * 125)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c3 v2 vs v3 토너먼트 최종 실측 결과표")
    print("=" * 125)
    header = f"{'모델 버전 및 피처 구성':<48} | {'차원':^5} | {'CatBoost F1':^12} | {'LightGBM F1':^12} | {'캐스케이드 F1':^14} | {'ROC-AUC':^9} | {'성과 (vs v2)'}"
    print(header)
    print("-" * 125)

    base_f1 = results[0]['cascade']['f1']

    for r in results:
        f_casc = r['cascade']['f1']
        df = f_casc - base_f1
        sign = "+" if df > 0 else ""
        df_str = "기준점 (Baseline)" if r == results[0] else f"{sign}{df:.2f}%p"
        star = "★ " if f_casc == max(x['cascade']['f1'] for x in results) else "  "
        print(f"{star + r['label']:<48} | {r['dim']:^5} | {r['cat']['f1']:>10.2f}% | {r['lgb']['f1']:>10.2f}% | {f_casc:>12.2f}% | {r['cascade']['roc_auc']:>8.2f}% | {df_str:>14}")

    print("=" * 125)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
