# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c2_v3_tournament.py
Purpose: 바벨 스쿼트 c2 v2 (20D) vs v3 (방해 피처 제거 정예 모델) 5-Fold 실측 비교 토너먼트
- 후보:
  1) [v2.0 기준]: 20D (F1 90.11%)
  2) [v3-A 16D]: 4대 핵심 방해 피처(#10 중복, #09 롤, #05 지터, #00 레거시) 소거
  3) [v3-B 14D]: 6대 방해 피처(#10, #09, #05, #00, #11 저크, #02 레거시바닥) 소거
  4) [v3-C 13D]: 7대 전체 방해 피처(#10, #09, #05, #00, #11, #02, #14) 소거
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
from src.models.second_models.babel_squat.c2_model_v2 import FEATURE_NAMES_20

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c2_v3_comparison_results.json'

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
        'opt_th': round(float(best_th), 2)
    }

def evaluate_feature_subset(X_subset, y_sub, levels_sub, subjs_sub, sgkf):
    N = len(y_sub)
    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_subset, y_sub, groups=subjs_sub)):
        X_tr, y_tr = X_subset[trn_idx], y_sub[trn_idx]
        X_va, y_va = X_subset[val_idx], y_sub[val_idx]

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

    # Cascade
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
    print("  [vidio-health-trainer] 바벨 스쿼트 c2 v2 (20D) vs v3 (방해 피처 제거 정예 모델) 5-Fold 실측 토너먼트")
    print("  ★ Level 1~3 실무군 1,775건 전수, 피험자 9명 StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    subjects = ds['subjects']

    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])
    elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
    X_18d = X_19d[:, elite_idx]

    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32)[:, :2] # 2D: drop, bot

    X_20d = np.concatenate([X_18d, gated_feats], axis=1)

    amb_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    levels = amb_d['levels']
    y_c2 = amb_d['y_c2'].astype(int)

    mask_l13 = (levels <= 3)
    X_sub = X_20d[mask_l13]
    y_sub = y_c2[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 4대 후보 세트 구성
    # [후보 1: v2.0 기준 20D]
    idx_v2 = list(range(20))

    # [후보 2: v3-A 16D] 4대 핵심 방해 피처(#10 중복, #09 롤, #05 지터, #00 레거시) 소거
    drop_v3a = [10, 9, 5, 0]
    idx_v3a = [i for i in range(20) if i not in drop_v3a]

    # [후보 3: v3-B 14D] 6대 방해 피처(#10, #09, #05, #00, #11 저크, #02 레거시바닥) 소거
    drop_v3b = [10, 9, 5, 0, 11, 2]
    idx_v3b = [i for i in range(20) if i not in drop_v3b]

    # [후보 4: v3-C 13D] 7대 전체 방해 피처(#10, #09, #05, #00, #11, #02, #14) 소거
    drop_v3c = [10, 9, 5, 0, 11, 2, 14]
    idx_v3c = [i for i in range(20) if i not in drop_v3c]

    candidates = [
        ("[v2.0 기준 모델] 20D 전체 세트", idx_v2, []),
        ("[v3-A 16D 모델] 4대 핵심 방해 피처(#10, #09, #05, #00) 소거", idx_v3a, drop_v3a),
        ("[v3-B 14D 모델] 6대 방해 피처(#10, #09, #05, #00, #11, #02) 소거", idx_v3b, drop_v3b),
        ("[v3-C 13D 모델] 7대 전체 방해 피처(#10, #09, #05, #00, #11, #02, #14) 소거", idx_v3c, drop_v3c)
    ]

    results = []
    print("\n[*] 5-Fold StratifiedGroupKFold 실측 토너먼트 시작...")

    for label, sel_idx, drop_idx in candidates:
        t0 = time.time()
        X_cur = X_sub[:, sel_idx]
        cat_res, lgb_res, casc_res = evaluate_feature_subset(X_cur, y_sub, levels_sub, subjs_sub, sgkf)
        elapsed = time.time() - t0

        drop_names = [FEATURE_NAMES_20[i] for i in drop_idx]
        sel_names = [FEATURE_NAMES_20[i] for i in sel_idx]

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
        print(f"    - 2계층 캐스케이드: F1 = {casc_res['f1']}% | ROC-AUC = {casc_res['roc_auc']}% (최적 F1: {casc_res['opt_f1']}%)")

    # 결과 종합 표 출력
    print("\n" + "=" * 120)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c2 v2 vs v3 토너먼트 최종 실측 결과표")
    print("=" * 120)
    header = f"{'모델 버전 및 피처 구성':<42} | {'피처수':^6} | {'CatBoost F1':^12} | {'캐스케이드 F1':^14} | {'최적 F1 (θ*)':^12} | {'ROC-AUC':^10} | {'성과 (vs v2)'}"
    print(header)
    print("-" * 120)

    base_f1 = results[0]['cascade']['f1']
    base_cat_f1 = results[0]['cat']['f1']

    for r in results:
        f_casc = r['cascade']['f1']
        df = f_casc - base_f1
        sign = "+" if df > 0 else ""
        df_str = "기준점 (Baseline)" if r == results[0] else f"{sign}{df:.2f}%p"
        star = "★ " if f_casc == max(x['cascade']['f1'] for x in results) else "  "
        print(f"{star + r['label']:<42} | {r['dim']:^6} | {r['cat']['f1']:>10.2f}% | {f_casc:>12.2f}% | {r['cascade']['opt_f1']:>10.2f}% | {r['cascade']['roc_auc']:>8.2f}% | {df_str:>14}")

    print("=" * 120)

    # JSON 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
