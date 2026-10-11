# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_v4_tournament.py
Purpose: 제2방법(외적 기반 부호 있는 척추 곡률 및 운동역학 연쇄 벡터) 기반
         c1 v4 모델군 vs c1 v3-C 챔피언 실측 비교 벤치마크 토너먼트
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
from src.models.second_models.babel_squat.c1_model_v3 import C1_V3_FEATURE_NAMES_25, V3_KEEP_INDICES_FROM_30D

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c1_v4_tournament_results.json'

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
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 척추의 중립 v3-C vs v4.0 제2방법(벡터차이분석) 토너먼트")
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

    # 1. 기존 v3-C 25D 피처 로드
    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_v3c_25d = X_v2_30d[:, V3_KEEP_INDICES_FROM_30D]

    # 2. 신규 v4 3D 척추 피처(10D) 로드
    v4_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz', allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)

    # 3. 레거시 정예 18D (v2 22D 중 노이즈 4개 #11, 15, 17, 18 제외)
    # v2의 0~21 중 제외: 11, 15, 17, 18
    legacy_clean_idx = [i for i in range(22) if i not in [11, 15, 17, 18]]
    X_legacy_18d = X_v2_30d[:, legacy_clean_idx]

    # v4-A: 레거시 정예 18D + 신규 v4 10D = 28D
    X_v4_28d = np.hstack([X_legacy_18d, X_v4_spine_10d])

    # v4-B: v3-C 25D + 신규 v4 10D = 35D
    X_v4_35d = np.hstack([X_v3c_25d, X_v4_spine_10d])

    # v4-C: 신규 v4 척추 피처 단독 10D
    X_v4_spine_only = X_v4_spine_10d

    # v4-D: 레거시 핵심 8D + v4 척추 8D = 16D 정예
    # 레거시 핵심: trunk_flex_max(#0), cervical_angle_min(#1), lumbar_hyperext_max(#2),
    #            trunk_flex_speed_max(#3), lumbar_stiffness_sd(#4), bottom_dwell_frames(#8),
    #            torso_shank_err(#9), lit_lumbar_shear_proxy(#20)
    leg_core = [0, 1, 2, 3, 4, 8, 9, 20]
    spine_core = [0, 1, 2, 3, 4, 5, 6, 7]
    X_v4_16d = np.hstack([X_v2_30d[:, leg_core], X_v4_spine_10d[:, spine_core]])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'v3-C_Champion_25D': {
            'X': X_v3c_25d,
            'cost_weight': 1.0,
            'desc': '★ 현재 공식 챔피언 v3-C 25D'
        },
        'v4-A_Hybrid_28D': {
            'X': X_v4_28d,
            'cost_weight': 1.0,
            'desc': '레거시 정예 18D + 신규 v4 벡터연쇄 10D 전수 결합'
        },
        'v4-B_Full_35D': {
            'X': X_v4_35d,
            'cost_weight': 1.0,
            'desc': 'v3-C 25D + 신규 v4 벡터연쇄 10D 결합 (35D)'
        },
        'v4-C_SpineOnly_10D': {
            'X': X_v4_spine_only,
            'cost_weight': 1.0,
            'desc': '신규 v4 부호있는 곡률+운동역학 연쇄 단독 (10D)'
        },
        'v4-D_Elite_16D': {
            'X': X_v4_16d,
            'cost_weight': 1.0,
            'desc': '레거시 정예 8D + v4 핵심 벡터 8D 다이어트 (16D)'
        },
        'v4-A_Cost_28D': {
            'X': X_v4_28d,
            'cost_weight': 1.6,
            'desc': 'v4-A 28D + 정상 FP 억제 비용민감 (w=1.6)'
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
    print(f"\n[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
