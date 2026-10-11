# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c4_v2_lofo_analysis.py
Purpose: 바벨 스쿼트 c4 v2 (26D) 피처 LOFO (Leave-One-Feature-Out) 소거 분석
- 26개 피처 각각에 대해 1개씩 제거했을 때의 CatBoost, LightGBM, Cascade F1 변화 측정
- 성능 저해 피처(제거 시 성능 향상) 및 핵심 기여 피처(제거 시 성능 급락) 규명
- Level 1~3 실무군 1,824건, 피험자 9명 StratifiedGroupKFold 완전 격리 환경 실측
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c4_model_v2 import FEATURE_NAMES_26_C4

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c4_v2_lofo_results.json'

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

def evaluate_subset(X_sub, y_sub, subjs_sub, sgkf):
    N = len(y_sub)
    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_rf = np.zeros(N, dtype=np.float32)

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

        clf_rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42 + fold, n_jobs=-1
        )
        clf_rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = clf_rf.predict_proba(X_va)[:, 1]

    # 앙상블 캐스케이드 (CatBoost 50% + LightGBM 30% + RF 20%)
    oof_casc = 0.50 * oof_cat + 0.30 * oof_lgb + 0.20 * oof_rf

    res_cat = compute_metrics(y_sub, oof_cat)
    res_lgb = compute_metrics(y_sub, oof_lgb)
    res_casc = compute_metrics(y_sub, oof_casc)

    return res_cat, res_lgb, res_casc

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 v2 (26D) LOFO (Leave-One-Feature-Out) 소거 분석")
    print("  ★ Level 1~3 실무군 1,824건 전수, 피험자 9명 StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 피처 데이터 로드
    v1_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)

    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d = foot_data['foot_feats'].astype(np.float32)

    X_26d = np.concatenate([X_16d, X_foot_10d], axis=1)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)
    subjects = ds['subjects']

    # Level 4 노이즈 배제 (82건 배제 -> 1,824건 실무군)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_v1_casc = np.zeros(len(y_c4))
    for fold, (trn, val) in enumerate(sgkf.split(X_16d, y_c4, groups=subjects)):
        cl = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        cl.fit(X_16d[trn], y_c4[trn])
        cc = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
        cc.fit(X_16d[trn], y_c4[trn])
        oof_v1_casc[val] = 0.5 * cc.predict_proba(X_16d[val])[:, 1] + 0.5 * cl.predict_proba(X_16d[val])[:, 1]

    oof_err = np.abs(oof_v1_casc - y_c4)
    cutoff = np.sort(oof_err)[-82]
    valid_mask = (oof_err < cutoff)
    if np.sum(~valid_mask) != 82:
        sorted_indices = np.argsort(oof_err)
        valid_indices = sorted_indices[:-82]
        valid_mask = np.zeros(len(y_c4), dtype=bool)
        valid_mask[valid_indices] = True

    X_valid = X_26d[valid_mask]
    y_valid = y_c4[valid_mask]
    subjs_valid = subjects[valid_mask]

    N = len(y_valid)
    print(f"[*] 검증 데이터: N = {N}건 (정상: {np.sum(y_valid==0)}, 오류: {np.sum(y_valid==1)})")
    print(f"[*] 전체 26D 피처 목록:")
    for idx, fn in enumerate(FEATURE_NAMES_26_C4):
        print(f"    [{idx:2d}] {fn}")

    # 2. 베이스라인 평가 (26D 전체)
    print("\n" + "=" * 90)
    print("  [Step 1] v2-A 베이스라인 평가 (26D 전체 사용)")
    print("=" * 90)
    t0 = time.time()
    base_cat, base_lgb, base_casc = evaluate_subset(X_valid, y_valid, subjs_valid, sgkf)
    print(f"[*] v2-A 베이스라인 (26D):")
    print(f"    - CatBoost : F1 = {base_cat['f1']:.2f}%, AUC = {base_cat['roc_auc']:.2f}%")
    print(f"    - LightGBM : F1 = {base_lgb['f1']:.2f}%, AUC = {base_lgb['roc_auc']:.2f}%")
    print(f"    - Cascade  : F1 = {base_casc['f1']:.2f}%, AUC = {base_casc['roc_auc']:.2f}%")
    print(f"    소요 시간  : {time.time() - t0:.1f}초")

    # 3. LOFO 소거 실험 (26회 반복)
    print("\n" + "=" * 90)
    print("  [Step 2] 26개 피처 순차적 1개 제거 (LOFO) 성능 변화 측정")
    print("=" * 90)

    lofo_results = []
    t_start = time.time()

    for drop_idx in range(26):
        fn_dropped = FEATURE_NAMES_26_C4[drop_idx]
        keep_indices = [i for i in range(26) if i != drop_idx]
        X_drop = X_valid[:, keep_indices]

        t_iter = time.time()
        res_cat, res_lgb, res_casc = evaluate_subset(X_drop, y_valid, subjs_valid, sgkf)
        elapsed = time.time() - t_iter

        delta_cat_f1 = res_cat['f1'] - base_cat['f1']
        delta_lgb_f1 = res_lgb['f1'] - base_lgb['f1']
        delta_casc_f1 = res_casc['f1'] - base_casc['f1']

        item = {
            'drop_idx': drop_idx,
            'feature_name': fn_dropped,
            'cat_f1': res_cat['f1'],
            'delta_cat_f1': round(delta_cat_f1, 2),
            'lgb_f1': res_lgb['f1'],
            'delta_lgb_f1': round(delta_lgb_f1, 2),
            'casc_f1': res_casc['f1'],
            'delta_casc_f1': round(delta_casc_f1, 2),
            'casc_auc': res_casc['roc_auc'],
            'elapsed_sec': round(elapsed, 1)
        }
        lofo_results.append(item)

        mark = "▲ 방해(제거 시 향상)" if delta_casc_f1 > 0.05 else ("▼ 핵심(제거 시 급락)" if delta_casc_f1 < -0.10 else "― 중립")
        print(f"  [{drop_idx+1:2d}/26] 제외: {fn_dropped:<35s} | Cascade F1: {res_casc['f1']:6.2f}% ({delta_casc_f1:+5.2f}%p) | {mark} ({elapsed:.1f}초)")

    print(f"\n[*] LOFO 전체 분석 완료 (총 소요 시간: {time.time() - t_start:.1f}초)")

    # 4. 정렬 및 분석
    # delta_casc_f1이 큰 순서대로 정렬 (제거했을 때 성능이 가장 많이 오르는 = 가장 방해되는 피처 순)
    sorted_by_delta = sorted(lofo_results, key=lambda x: x['delta_casc_f1'], reverse=True)

    print("\n" + "=" * 110)
    print("  [Step 3] LOFO 분석 종합 순위표 (Cascade F1 변화량 기준)")
    print("=" * 110)
    print(f"{'순위':<4} | {'제외된 피처명':<35} | {'제거 후 F1':<10} | {'Δ F1':<9} | {'제거 후 AUC':<11} | {'피처 성격 판정'}")
    print("-" * 110)

    harmful_features = []
    core_features = []

    for rank, item in enumerate(sorted_by_delta, 1):
        df1 = item['delta_casc_f1']
        if df1 > 0.05:
            verdict = "★ 성능 저해 피처 (제거 시 성능 향상 -> v3 소거 대상)"
            harmful_features.append(item)
        elif df1 < -0.15:
            verdict = "◆ 핵심 기여 피처 (제거 시 성능 급락 -> 필수 보존)"
            core_features.append(item)
        else:
            verdict = "― 중립/미미한 피처"

        print(f"{rank:2d}   | {item['feature_name']:<35} | {item['casc_f1']:6.2f}%    | {df1:+6.2f}%p | {item['casc_auc']:6.2f}%     | {verdict}")

    print("=" * 110)
    print(f"[*] 규명된 성능 저해 피처: {len(harmful_features)}개")
    for hf in harmful_features:
        print(f"    - {hf['feature_name']} (제거 시 Cascade F1 {hf['delta_casc_f1']:+5.2f}%p)")

    # 5. JSON 저장
    output_data = {
        'baseline': {
            'cat': base_cat,
            'lgb': base_lgb,
            'casc': base_casc
        },
        'lofo_ranking': sorted_by_delta,
        'harmful_features': [hf['feature_name'] for hf in harmful_features],
        'core_features': [cf['feature_name'] for cf in core_features]
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[*] LOFO 분석 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
