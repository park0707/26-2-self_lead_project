# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_v3_tournament.py
Purpose: c1 v2-A 대비 성능 저해 피처(Noise & Multicollinearity) 소거 기반 c1 v3 모델 후보군 토너먼트 벤치마크
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
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2, C1_V2_FEATURE_NAMES_30

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c1_v3_tournament_results.json'

def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    sgkf: StratifiedGroupKFold,
    clean_mask: np.ndarray = None
) -> Dict[str, Any]:
    N = len(y)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for trn, val in sgkf.split(X, y, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn])
        oof_cb[val] = cb.predict_proba(X[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X[trn], y[trn])
        oof_lgb[val] = clf_l.predict_proba(X[val])[:, 1]

    oof_casc = 0.60 * oof_cb + 0.40 * oof_lgb

    def get_stats(probs):
        pred_50 = (probs >= 0.50).astype(int)
        f1_50 = f1_score(y, pred_50, zero_division=0) * 100.0
        prec_50 = precision_score(y, pred_50, zero_division=0) * 100.0
        rec_50 = recall_score(y, pred_50, zero_division=0) * 100.0
        auc = roc_auc_score(y, probs) * 100.0

        best_th, best_f1 = 0.50, f1_50
        for t in np.linspace(0.25, 0.75, 51):
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
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 척추의 중립 v2 vs v3 성능 저해 피처 소거 실측 토너먼트")
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
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)
    X_30d = BabelSquatC1ModelV2.extract_features(X_b_64)

    X_val = X_30d[lvl4_mask]
    y_val = y_c1[lvl4_mask]
    sub_val = subjects[lvl4_mask]
    clean_mask_val = clean_mask_all[lvl4_mask]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 후보 피처셋 구성
    feat_names = list(C1_V2_FEATURE_NAMES_30)

    # v3-A: 29D (Drop #18 lit_trunk_lean_max)
    drop_a = [18]
    idx_a = [i for i in range(30) if i not in drop_a]

    # v3-B: 28D (Drop #18 lit_trunk_lean_max, #15 ear_sh_hip_drop)
    drop_b = [18, 15]
    idx_b = [i for i in range(30) if i not in drop_b]

    # v3-C: 25D (Drop #11, #15, #17, #18, #24) - 상위 5대 저해 및 극심 다중공선성 전수 소거
    drop_c = [11, 15, 17, 18, 24]
    idx_c = [i for i in range(30) if i not in drop_c]

    # v3-D: 24D (Drop #09, #11, #15, #17, #18, #24) - 상위 6대 소거
    drop_d = [9, 11, 15, 17, 18, 24]
    idx_d = [i for i in range(30) if i not in drop_d]

    candidates = {
        'v2-A_Baseline_30D': {
            'indices': list(range(30)),
            'dropped': [],
            'desc': '기존 v2-A 30D 전수 기준선'
        },
        'v3-A_Top1Pruned_29D': {
            'indices': idx_a,
            'dropped': [feat_names[i] for i in drop_a],
            'desc': '상체 기울기 노이즈(lit_trunk_lean_max) 1개 소거'
        },
        'v3-B_Top2Pruned_28D': {
            'indices': idx_b,
            'dropped': [feat_names[i] for i in drop_b],
            'desc': '상체기울기 + 2D귀-어깨낙폭(ear_sh_hip_drop) 2개 소거'
        },
        'v3-C_Top5Pruned_25D': {
            'indices': idx_c,
            'dropped': [feat_names[i] for i in drop_c],
            'desc': '★ 5대 저해 피처(시선/비율/다중공선성) 전수 소거'
        },
        'v3-D_Top6Pruned_24D': {
            'indices': idx_d,
            'dropped': [feat_names[i] for i in drop_d],
            'desc': '6대 저해 피처 소거 정예 모델'
        }
    }

    # 1. Level 1~3 실무군 실측
    print("\n" + "=" * 115)
    print("  [Phase 1] Level 1~3 실무 유효 세트 (1,577건) 대상 5-Fold 실측 결과")
    print("=" * 115)
    print(f"{'후보 모델':<24} | {'피처수':<5} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'Cascade Opt':<11} | {'ROC-AUC':<8} | {'정상 FP'}")
    print("-" * 115)

    res_l123 = {}
    for cname, cinfo in candidates.items():
        indices = cinfo['indices']
        res = evaluate_features(X_val[:, indices], y_val, sub_val, sgkf, clean_mask_val)
        res_l123[cname] = res
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        casc_opt = res['Cascade']['opt_f1']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<24} | {len(indices):>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {casc_opt:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    # 2. 원천 전체 1,906건 실측
    print("\n" + "=" * 115)
    print("  [Phase 2] 원천 데이터셋 전체 (1,906건) 대상 5-Fold 실측 결과")
    print("=" * 115)
    print(f"{'후보 모델':<24} | {'피처수':<5} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'Cascade Opt':<11} | {'ROC-AUC':<8} | {'클린 FP / 125건'}")
    print("-" * 115)

    res_all1906 = {}
    for cname, cinfo in candidates.items():
        indices = cinfo['indices']
        res = evaluate_features(X_30d[:, indices], y_c1, subjects, sgkf, clean_mask_all)
        res_all1906[cname] = res
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        casc_opt = res['Cascade']['opt_f1']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<24} | {len(indices):>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {casc_opt:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    output_data = {
        'candidates': {k: {'indices': v['indices'], 'dropped': v['dropped'], 'desc': v['desc']} for k, v in candidates.items()},
        'results_level123': res_l123,
        'results_all1906': res_all1906
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
