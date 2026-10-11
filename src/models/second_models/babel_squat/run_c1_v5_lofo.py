# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c1_v5_lofo.py
Purpose: c1 v5 34D 피처 전수에 대한 5-Fold LOFO(Leave-One-Feature-Out) 소거 실험
"""

import os
import sys
import time
import json
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES, C1_V4_FEATURE_NAMES_28
from src.models.second_models.babel_squat.c1_spinepose_extractor import C1_SPINEPOSE_DENSE_FEATURE_NAMES_6

OUT_JSON = 'd:/src/models/second_models/babel_squat/c1_v5_lofo_results.json'

def evaluate_subset(X, y, subjects, sgkf):
    oof = np.zeros(len(y), dtype=np.float32)
    for trn, val in sgkf.split(X, y, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn])
        oof[val] = cb.predict_proba(X[val])[:, 1]
    
    f1_50 = f1_score(y, (oof >= 0.50).astype(int), zero_division=0) * 100.0
    auc = roc_auc_score(y, oof) * 100.0
    
    best_th, best_f1 = 0.50, f1_50
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y, (oof >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t
            
    return round(float(f1_50), 2), round(float(best_f1), 2), round(float(auc), 2)

def main():
    print("=" * 95)
    print("  [vidio-health-trainer] c1 v5 34D 피처 전수 LOFO (소거 시 성능 변화) 분석 시작")
    print("=" * 95)

    ds = SquatPilotDatasetLoader.load_dataset()
    y = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_leg_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]

    v4_spine = np.load('d:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz', allow_pickle=True)['v4_spine_feats']
    v5_spine = np.load('d:/src/models/second_models/babel_squat/squat_c1_v5_spinepose_feats.npz', allow_pickle=True)['v5_spinepose_feats']

    X_34d = np.hstack([X_leg_18d, v4_spine, v5_spine]).astype(np.float32)
    feature_names = list(C1_V4_FEATURE_NAMES_28) + list(C1_SPINEPOSE_DENSE_FEATURE_NAMES_6)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[*] 1단계: Baseline (34D 전수) 5-Fold 평가 중...")
    base_f1_50, base_opt_f1, base_auc = evaluate_subset(X_34d, y, subjects, sgkf)
    print(f"    -> Baseline 34D: F1@0.50 = {base_f1_50:.2f}%, Opt F1 = {base_opt_f1:.2f}%, AUC = {base_auc:.2f}%")

    print("\n[*] 2단계: 34개 피처 개별 LOFO 소거 실험 진행 중...")
    lofo_results = {}
    t0 = time.time()

    for i, fname in enumerate(feature_names):
        keep_idx = [j for j in range(34) if j != i]
        X_sub = X_34d[:, keep_idx]
        f1_50, opt_f1, auc = evaluate_subset(X_sub, y, subjects, sgkf)
        delta_f1 = round(f1_50 - base_f1_50, 2)
        delta_opt = round(opt_f1 - base_opt_f1, 2)
        
        status = "★ 제거 시 성능 향상 (저해)" if delta_f1 > 0 else "필수 피처 (유지)"
        lofo_results[fname] = {
            'index': i,
            'f1_50': f1_50,
            'opt_f1': opt_f1,
            'auc': auc,
            'delta_f1': delta_f1,
            'delta_opt': delta_opt,
            'harmful': bool(delta_f1 > 0)
        }
        print(f"  [{i+1:2d}/34] 소거: {fname:<36} | F1: {f1_50:6.2f}% (Δ {delta_f1:+5.2f}%p) | {status}")

    # 저해 피처 랭킹
    harmful_sorted = sorted([(k, v['delta_f1'], v['delta_opt']) for k, v in lofo_results.items() if v['delta_f1'] > 0], key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 95)
    print(f"  [*] 성능 저해 피처 발견: 총 {len(harmful_sorted)}개")
    for name, df1, dopt in harmful_sorted:
        print(f"      - {name:<36}: 소거 시 ΔF1 = {df1:+5.2f}%p, ΔOpt = {dopt:+5.2f}%p")

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'baseline': {'f1_50': base_f1_50, 'opt_f1': base_opt_f1, 'auc': base_auc}, 'lofo': lofo_results}, f, indent=2, ensure_ascii=False)
    print(f"\n[*] LOFO 결과 저장 완료: {OUT_JSON} ({time.time()-t0:.1f}초)")

if __name__ == '__main__':
    main()
