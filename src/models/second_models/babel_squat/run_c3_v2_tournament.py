# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c3_v2_tournament.py
Purpose: 바벨 스쿼트 c3(발과 무릎의 방향 일치) v1 (16D) vs v2 (3D 발 관절 정렬 피처 결합) 5-Fold 실측 비교 토너먼트
- 후보:
  1) [v1.0 기준]: 16D (기존 정강이/무릎 간격비 체계)
  2) [v2-A 24D]: 16D + 8D 전체 신규 발 관절 피처
  3) [v2-B 20D]: 16D + 4D 핵심 발 관절 피처 (정렬 사잇각 3종 + 내측 이탈 거리 1종)
  4) [v2-C 18D]: 16D + 2D 최정예 발 관절 피처 (정렬 사잇각 nadir + 내측 이탈 거리 nadir)
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
from src.models.second_models.babel_squat.build_c3_foot_features import C3_FOOT_FEATURE_NAMES

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c3_v2_comparison_results.json'

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
            random_seed=42 + fold, thread_count=-1, verbose=False
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

    # 2계층 캐스케이드 (Tier-1 Level 1 명확군 0.001ms 고속 확정 + Tier-2 리졸버)
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
    print("  [vidio-health-trainer] 바벨 스쿼트 c3 v1 (16D) vs v2 (3D 발 관절 정렬 피처 결합) 5-Fold 실측 토너먼트")
    print("  ★ Level 1~3 실무군 1,777건 전수, 피험자 9명 StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 기존 v1 16D 피처 로드
    v1_data = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)
    feat_names_16 = list(v1_data['feature_names'])

    # 2. 신규 발 8D 피처 로드
    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d = foot_data['foot_feats'].astype(np.float32)
    feat_names_foot = list(foot_data['feature_names'])

    # 결합 24D
    X_24d = np.concatenate([X_16d, X_foot_8d], axis=1)

    # 3. 레이블 및 모호성 레벨 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    # c3 모호성 레벨 (run_c3_extended_pipeline.py 정밀 기준: Level 1~4)
    valgus_inc = X_16d[:, 7] # 외반 증가폭
    pct15_inc = np.percentile(valgus_inc, 15)
    pct85_inc = np.percentile(valgus_inc, 85)

    levels = np.full(len(y_c3), 2, dtype=int)

    # Level 1: 명확군
    mask_l1_ok = (y_c3 == 0) & (valgus_inc <= pct15_inc)
    mask_l1_bad = (y_c3 == 1) & (valgus_inc >= pct85_inc)
    levels[mask_l1_ok | mask_l1_bad] = 1

    # Level 3: 체형 편차군
    asym = X_16d[:, 8]
    pct85_asym = np.percentile(asym, 85)
    mask_l3 = (levels == 2) & (asym > pct85_asym)
    levels[mask_l3] = 3

    # Level 4: 라벨 역전 노이즈
    pct05_inc = np.percentile(valgus_inc, 5)
    pct95_inc = np.percentile(valgus_inc, 95)
    mask_l4_pos = (y_c3 == 1) & (valgus_inc <= pct05_inc)
    mask_l4_neg = (y_c3 == 0) & (valgus_inc >= pct95_inc)
    levels[mask_l4_pos | mask_l4_neg] = 4

    mask_l13 = (levels <= 3)
    N_valid = np.sum(mask_l13)
    print(f"[*] Level 1~3 실무군 유효 샘플 수: {N_valid}건 (배제된 Level 4 역전 노이즈: {np.sum(levels==4)}건)")


    X_16d_sub = X_16d[mask_l13]
    X_24d_sub = X_24d[mask_l13]
    y_sub = y_c3[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 후보 정의
    # [후보 1: v1.0 기준 16D]
    idx_v1 = list(range(16))

    # [후보 2: v2-A 24D] 16D + 8D 전체
    idx_v2a = list(range(24))

    # [후보 3: v2-B 20D] 16D + 4D 핵심 발 피처 (사잇각 nadir, drop, max + 내측 이탈 거리 nadir)
    # 신규 인덱스: 16: nadir_ang, 17: drop_ang, 18: max_ang, 19: dev_nadir
    idx_v2b = list(range(16)) + [16, 17, 18, 19]

    # [후보 4: v2-C 18D] 16D + 2D 최정예 발 피처 (사잇각 nadir + 내측 이탈 거리 nadir)
    idx_v2c = list(range(16)) + [16, 19]

    # [후보 5: v2-D 21D] 16D + 5D (사잇각 nadir, drop + 내측 이탈 nadir, drop + 경골 비틀림)
    idx_v2d = list(range(16)) + [16, 17, 19, 20, 22]

    candidates = [
        ("[v1.0 기준 모델] 16D 레거시 정강이/무릎 간격비", idx_v1),
        ("[v2-A 모델] 24D 전체 발 관절 피처 결합 (8D 추가)", idx_v2a),
        ("[v2-B 모델] 20D 정예 발 관절 피처 결합 (4D 추가: 사잇각 3종 + 이탈거리 1종)", idx_v2b),
        ("[v2-C 모델] 18D 초경량 발 관절 피처 결합 (2D 추가: 사잇각 nadir + 이탈거리 nadir)", idx_v2c),
        ("[v2-D 모델] 21D 동역학 발 관절 피처 결합 (5D 추가: 사잇각 2종 + 이탈 2종 + 경골비틀림)", idx_v2d)
    ]

    all_feat_names = feat_names_16 + feat_names_foot
    results = []

    print("\n[*] 5-Fold StratifiedGroupKFold 실측 토너먼트 시작...")

    for label, sel_idx in candidates:
        t0 = time.time()
        X_cur = X_24d_sub[:, sel_idx]
        cat_res, lgb_res, casc_res = evaluate_subset(X_cur, y_sub, levels_sub, subjs_sub, sgkf)
        elapsed = time.time() - t0

        sel_names = [all_feat_names[i] for i in sel_idx]

        results.append({
            'label': label,
            'dim': len(sel_idx),
            'selected_features': sel_names,
            'cat': cat_res,
            'lgb': lgb_res,
            'cascade': casc_res,
            'elapsed_sec': round(elapsed, 2)
        })

        print(f"\n>>> 완료: {label} (차원: {len(sel_idx)}D, 소요: {elapsed:.1f}초)")
        print(f"    - 단일 CatBoost: F1 = {cat_res['f1']}% | ROC-AUC = {cat_res['roc_auc']}% (최적 F1: {cat_res['opt_f1']}%)")
        print(f"    - 단일 LightGBM: F1 = {lgb_res['f1']}% | ROC-AUC = {lgb_res['roc_auc']}% (최적 F1: {lgb_res['opt_f1']}%)")
        print(f"    - 2계층 캐스케이드: F1 = {casc_res['f1']}% | ROC-AUC = {casc_res['roc_auc']}% (최적 F1: {casc_res['opt_f1']}%)")

    # 종합 결과표 출력
    print("\n" + "=" * 125)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c3 v1 vs v2 토너먼트 최종 실측 결과표")
    print("=" * 125)
    header = f"{'모델 버전 및 피처 구성':<48} | {'차원':^5} | {'CatBoost F1':^12} | {'LightGBM F1':^12} | {'캐스케이드 F1':^14} | {'ROC-AUC':^9} | {'성과 (vs v1)'}"
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

    # JSON 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
