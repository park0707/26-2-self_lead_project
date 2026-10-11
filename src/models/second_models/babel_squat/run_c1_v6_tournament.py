# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_v6_tournament.py
Purpose: c1 v6.0 피처 최적화 소거 모델 (v6-A 29D, v6-B 21D) vs v5.0 (34D) vs v4-A (28D) 5-Fold 실측 토너먼트
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES, V6B_KEEP_INDICES

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c1_v6_tournament_results.json'

def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    sgkf: StratifiedGroupKFold,
    clean_mask: np.ndarray = None,
    cost_weight: float = 1.0
) -> Dict[str, Any]:
    N = len(y)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for trn, val in sgkf.split(X, y, groups=subjects):
        sw_trn = np.ones(len(trn), dtype=np.float32)
        if cost_weight > 1.0:
            sw_trn[y[trn] == 0] = cost_weight

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn], sample_weight=sw_trn)
        oof_cb[val] = cb.predict_proba(X[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X[trn], y[trn], sample_weight=sw_trn)
        oof_lgb[val] = clf_l.predict_proba(X[val])[:, 1]

    oof_casc = 0.60 * oof_cb + 0.40 * oof_lgb

    def get_stats(probs):
        pred_50 = (probs >= 0.50).astype(int)
        f1_50 = f1_score(y, pred_50, zero_division=0) * 100.0
        prec_50 = precision_score(y, pred_50, zero_division=0) * 100.0
        rec_50 = recall_score(y, pred_50, zero_division=0) * 100.0
        auc = roc_auc_score(y, probs) * 100.0

        best_th, best_f1 = 0.50, f1_50
        for t in np.linspace(0.20, 0.80, 61):
            f = f1_score(y, (probs >= t).astype(int), zero_division=0) * 100.0
            if f > best_f1:
                best_f1, best_th = f, t

        clean_fps = 0
        clean_fpr = 0.0
        if clean_mask is not None:
            clean_fps = int(np.sum((pred_50 == 1) & clean_mask))
            clean_fpr = (clean_fps / max(1, np.sum(clean_mask))) * 100.0

        return {
            'f1_50': round(float(f1_50), 2),
            'opt_f1': round(float(best_f1), 2),
            'opt_th': round(float(best_th), 2),
            'prec_50': round(float(prec_50), 2),
            'rec_50': round(float(rec_50), 2),
            'auc': round(float(auc), 2),
            'clean_fps': clean_fps,
            'clean_fpr': round(float(clean_fpr), 2)
        }

    return {
        'CatBoost': get_stats(oof_cb),
        'LightGBM': get_stats(oof_lgb),
        'Cascade': get_stats(oof_casc)
    }

def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 v6.0 (소거 최적화) vs v5.0 vs v4-A 5-Fold 실측 토너먼트")
    print("=" * 115)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    Y_all = ds['Y_defects'][:, :4].astype(int)
    subjects = ds['subjects']
    clean_mask_all = (np.sum(Y_all, axis=1) == 0)

    amb_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    lvl4_mask = ~amb_d['l4_mask']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    # 1. 레거시 18D
    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]

    # 2. v4 10D
    v4_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz', allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    X_v4_28d = np.hstack([X_legacy_18d, X_v4_spine_10d])

    # 3. v5 6D
    v5_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v5_spinepose_feats.npz', allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    X_v5_34d = np.hstack([X_v4_28d, X_v5_spinepose_6d])

    # 4. v6 피처셋
    X_v6a_29d = X_v5_34d[:, V6A_KEEP_INDICES]
    X_v6b_21d = X_v5_34d[:, V6B_KEEP_INDICES]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'v4-A_Benchmark_28D': {
            'X': X_v4_28d,
            'cost_weight': 1.0,
            'desc': 'v4-A 28D 벤치마크'
        },
        'v5-A_Full_34D': {
            'X': X_v5_34d,
            'cost_weight': 1.0,
            'desc': 'v5-A 34D 전수 결합'
        },
        'v6-A_Pruned_29D': {
            'X': X_v6a_29d,
            'cost_weight': 1.0,
            'desc': '★ v6-A 29D (상위 5대 저해 소거 정예 모델)'
        },
        'v6-B_Compact_21D': {
            'X': X_v6b_21d,
            'cost_weight': 1.0,
            'desc': '★ v6-B 21D (13대 저해/중복 소거 고효율 컴팩트 모델)'
        },
        'v6-A_Cost_29D': {
            'X': X_v6a_29d,
            'cost_weight': 1.4,
            'desc': 'v6-A 29D + FP 억제 비용민감 (w=1.4)'
        }
    }

    # 1. Level 1~3 실무군 실측
    print("\n" + "=" * 115)
    print("  [Phase 1] Level 1~3 실무 유효 세트 (1,577건) 대상 5-Fold 실측 결과")
    print("=" * 115)
    print(f"{'후보 모델':<22} | {'피처수':<5} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'Cascade Opt':<11} | {'ROC-AUC':<8} | {'정상 FP'}")
    print("-" * 115)

    res_l123 = {}
    for cname, cinfo in candidates.items():
        X_sub = cinfo['X'][lvl4_mask]
        res = evaluate_features(X_sub, y_c1[lvl4_mask], subjects[lvl4_mask], sgkf, clean_mask_all[lvl4_mask], cost_weight=cinfo['cost_weight'])
        res_l123[cname] = res
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        casc_opt = res['Cascade']['opt_f1']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<22} | {cinfo['X'].shape[1]:>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {casc_opt:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    # 2. 원천 전체 1,906건 실측
    print("\n" + "=" * 115)
    print("  [Phase 2] 원천 데이터셋 전체 (1,906건) 대상 5-Fold 실측 결과")
    print("=" * 115)
    print(f"{'후보 모델':<22} | {'피처수':<5} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'Cascade Opt':<11} | {'ROC-AUC':<8} | {'클린 FP / 125건'}")
    print("-" * 115)

    res_all1906 = {}
    for cname, cinfo in candidates.items():
        res = evaluate_features(cinfo['X'], y_c1, subjects, sgkf, clean_mask_all, cost_weight=cinfo['cost_weight'])
        res_all1906[cname] = res
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        casc_opt = res['Cascade']['opt_f1']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<22} | {cinfo['X'].shape[1]:>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {casc_opt:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    out_data = {
        'level123': res_l123,
        'full1906': res_all1906
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] c1 v6 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
