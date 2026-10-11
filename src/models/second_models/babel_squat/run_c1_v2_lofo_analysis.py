# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_v2_lofo_analysis.py
Purpose: c1 v2-A 30D 피처셋 대상 LOFO (Leave-One-Feature-Out) 소거 실험 및
         성능 저해/방해 피처 탐색을 통한 c1 v3 모델 후보군 토너먼트 벤치마크
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
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2, C1_V2_FEATURE_NAMES_30

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c1_lofo_and_v3_results.json'

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
        # CatBoost
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn])
        oof_cb[val] = cb.predict_proba(X[val])[:, 1]

        # LightGBM
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
    print("=" * 105)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 v2-A 30D LOFO 성능 저해 피처 탐색 & v3 토너먼트")
    print("=" * 105)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    Y_all = ds['Y_defects'][:, :4].astype(int)
    subjects = ds['subjects']
    clean_mask_all = (np.sum(Y_all, axis=1) == 0) # 125건 완전 정상
    N_total = len(y_c1)

    # Level 1~3 실무군 마스크
    amb_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    lvl4_mask = ~amb_d['l4_mask']
    N_valid = int(np.sum(lvl4_mask))
    print(f"[*] 총 샘플: {N_total}건 | Level 1~3 실무군: {N_valid}건 | 클린 정상: {np.sum(clean_mask_all)}건")

    # 2. 64F 스플라인 30D 피처 추출
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)

    X_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    feat_names = list(C1_V2_FEATURE_NAMES_30)
    print(f"[*] 30D 피처 준비 완료: shape = {X_30d.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 실무군 데이터 슬라이스
    X_val = X_30d[lvl4_mask]
    y_val = y_c1[lvl4_mask]
    sub_val = subjects[lvl4_mask]
    clean_mask_val = clean_mask_all[lvl4_mask]

    # 3. [Step 1] Baseline 30D 전체 평가 (Level 1~3)
    print("\n" + "=" * 90)
    print("  [Step 1] Baseline v2-A (30D 전수) 5-Fold 교차검증 실측")
    print("=" * 90)
    t0 = time.time()
    base_res = evaluate_features(X_val, y_val, sub_val, sgkf, clean_mask_val)
    base_cb_f1 = base_res['CatBoost']['f1_50']
    base_cb_opt = base_res['CatBoost']['opt_f1']
    base_cb_auc = base_res['CatBoost']['auc']
    base_casc_f1 = base_res['Cascade']['f1_50']
    base_casc_opt = base_res['Cascade']['opt_f1']
    base_casc_auc = base_res['Cascade']['auc']

    print(f"[*] Base 30D -> CatBoost F1: {base_cb_f1:.2f}% (Opt F1: {base_cb_opt:.2f}%, AUC: {base_cb_auc:.2f}%)")
    print(f"[*] Base 30D -> Cascade  F1: {base_casc_f1:.2f}% (Opt F1: {base_casc_opt:.2f}%, AUC: {base_casc_auc:.2f}%) [{time.time()-t0:.1f}초]")

    # 4. [Step 2] LOFO (Leave-One-Feature-Out) 소거 실험
    print("\n" + "=" * 105)
    print("  [Step 2] 30대 피처별 LOFO 소거 실험 (개별 피처 소거 시 ΔF1 측정)")
    print("=" * 105)
    print(f"{'Idx':<4} | {'소거 대상 피처명':<38} | {'CatBoost F1':<12} | {'Δ F1':<9} | {'Opt F1':<9} | {'AUC':<8} | {'판정'}")
    print("-" * 105)

    lofo_results = []
    detrimental_features = [] # 소거 시 F1이 향상되는 피처들

    for i in range(30):
        keep_indices = [idx for idx in range(30) if idx != i]
        X_sub = X_val[:, keep_indices]

        res = evaluate_features(X_sub, y_val, sub_val, sgkf, clean_mask_val)
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        cb_auc = res['CatBoost']['auc']
        casc_f1 = res['Cascade']['f1_50']
        casc_opt = res['Cascade']['opt_f1']

        delta_f1 = cb_f1 - base_cb_f1
        delta_opt = cb_opt - base_cb_opt

        fname = feat_names[i]
        # 판정: ΔF1 > 0 이면 성능 저해(Harmful), ΔF1 < 0 이면 필수(Crucial)
        if delta_f1 > 0.10 or delta_opt > 0.10:
            status = "★ 성능저해 (소거 시 향상)"
            detrimental_features.append((i, fname, delta_f1, delta_opt))
        elif delta_f1 < -0.30 or delta_opt < -0.30:
            status = "■ 핵심필수 (소거 시 하락)"
        else:
            status = "─ 중립/미미"

        lofo_results.append({
            'index': i,
            'feature_name': fname,
            'cb_f1': cb_f1,
            'delta_f1': round(delta_f1, 2),
            'cb_opt': cb_opt,
            'delta_opt': round(delta_opt, 2),
            'auc': cb_auc,
            'status': status
        })

        sign = "+" if delta_f1 >= 0 else ""
        print(f"#{i:02d} | {fname:<38} | {cb_f1:6.2f}%      | {sign}{delta_f1:5.2f}%p | {cb_opt:6.2f}% | {cb_auc:5.2f}% | {status}")

    print("\n[*] LOFO 성능 저해 피처 검출 결과 (F1을 저해하던 피처들):")
    for idx, fn, df1, dopt in detrimental_features:
        print(f"    - #{idx:02d} {fn}: 소거 시 CatBoost F1 +{df1:.2f}%p (Opt F1 +{dopt:.2f}%p)")

    # 5. [Step 3] 피처 상호 상관도 분석 (Collinearity Matrix)
    corr_matrix = np.corrcoef(X_val, rowvar=False)
    high_corr_pairs = []
    for r in range(30):
        for c in range(r + 1, 30):
            if abs(corr_matrix[r, c]) >= 0.85:
                high_corr_pairs.append((r, c, feat_names[r], feat_names[c], float(corr_matrix[r, c])))

    print(f"\n[*] 고상관 피처 쌍 (|r| >= 0.85): {len(high_corr_pairs)}쌍 검출")
    for r, c, fn1, fn2, val in high_corr_pairs:
        print(f"    - #{r:02d} {fn1} <-> #{c:02d} {fn2} (r = {val:.3f})")

    # 6. [Step 4] v3 모델 후보군 설계
    # v2-A: 30D (기준선)
    # v3-A: 단일 최악 저해 피처 1개 소거
    # v3-B: LOFO 검출 성능 저해 피처 전수 소거
    # v3-C: 저해 피처 + 고상관 중복 피처 동시 소거
    # v3-D: 정예 피처셋 (핵심 생체역학 지표 보존 + 경량화)

    detrimental_indices = [x[0] for x in detrimental_features]
    top1_detrimental = max(detrimental_features, key=lambda x: x[2])[0] if detrimental_features else 0

    candidates = {}
    candidates['v2-A_Baseline_30D'] = {
        'indices': list(range(30)),
        'desc': '기존 v2-A 30D 전수'
    }

    v3_a_idx = [i for i in range(30) if i != top1_detrimental]
    candidates['v3-A_Top1Pruned'] = {
        'indices': v3_a_idx,
        'desc': f'최대 저해 피처 1개(#{top1_detrimental:02d}) 소거 ({len(v3_a_idx)}D)'
    }

    v3_b_idx = [i for i in range(30) if i not in detrimental_indices]
    candidates['v3-B_AllDetrimentalPruned'] = {
        'indices': v3_b_idx,
        'desc': f'저해 피처 {len(detrimental_indices)}개 전수 소거 ({len(v3_b_idx)}D)'
    }

    # v3-C: 고상관 중복 피처 중 저해측 소거
    redundant_to_drop = set(detrimental_indices)
    for r, c, _, _, _ in high_corr_pairs:
        if c not in redundant_to_drop and r not in redundant_to_drop:
            redundant_to_drop.add(c)
    v3_c_idx = [i for i in range(30) if i not in redundant_to_drop]
    candidates['v3-C_CollinearPruned'] = {
        'indices': v3_c_idx,
        'desc': f'저해 피처 + 고상관 중복 소거 ({len(v3_c_idx)}D)'
    }

    # v3-D: 고정밀 정예 피처셋 (3D 척추 핵심 6D + 레거시 고기여 피처 14D)
    # 3D 척추 6D: tl_nadir, tl_drop, tl_max_flex, buttwink, hip_hinge, shear_moment_arm (22, 23, 24, 25, 26, 29)
    # 레거시 정예: 0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15, 17, 18, 20
    elite_idx = sorted(list(set([0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15, 17, 18, 20, 22, 23, 24, 25, 26, 29]) - set(detrimental_indices)))
    candidates['v3-D_EliteCompact'] = {
        'indices': elite_idx,
        'desc': f'생체역학 정예 피처셋 ({len(elite_idx)}D)'
    }

    # 7. [Step 5] Level 1~3 실무군 (1,577건) 대상 최종 토너먼트
    print("\n" + "=" * 115)
    print("  [Step 5] Level 1~3 실무군 (1,577건) 대상 v2 vs v3 후보군 5-Fold 실측 토너먼트")
    print("=" * 115)
    print(f"{'후보 모델':<26} | {'피처수':<6} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'ROC-AUC':<9} | {'정상 FP (율)'}")
    print("-" * 115)

    results_valid = {}
    for cname, cinfo in candidates.items():
        idx_set = cinfo['indices']
        X_cand = X_val[:, idx_set]
        res = evaluate_features(X_cand, y_val, sub_val, sgkf, clean_mask_val)
        results_valid[cname] = {
            'feature_dim': len(idx_set),
            'indices': idx_set,
            'feature_names': [feat_names[i] for i in idx_set],
            'desc': cinfo['desc'],
            'metrics': res
        }
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<26} | {len(idx_set):>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    # 8. [Step 6] 원천 전체 (1,906건) 대상 실측 검증
    print("\n" + "=" * 115)
    print("  [Step 6] 원천 전체 (1,906건) 대상 v2 vs v3 후보군 5-Fold 실측 토너먼트")
    print("=" * 115)
    print(f"{'후보 모델':<26} | {'피처수':<6} | {'CatBoost F1':<12} | {'Opt F1':<9} | {'Cascade F1':<11} | {'ROC-AUC':<9} | {'정상 FP (율)'}")
    print("-" * 115)

    results_all = {}
    for cname, cinfo in candidates.items():
        idx_set = cinfo['indices']
        X_cand_all = X_30d[:, idx_set]
        res = evaluate_features(X_cand_all, y_c1, subjects, sgkf, clean_mask_all)
        results_all[cname] = {
            'feature_dim': len(idx_set),
            'metrics': res
        }
        cb_f1 = res['CatBoost']['f1_50']
        cb_opt = res['CatBoost']['opt_f1']
        casc_f1 = res['Cascade']['f1_50']
        auc = res['CatBoost']['auc']
        fps = res['CatBoost']['clean_fps']
        fpr = res['CatBoost']['clean_fpr']
        print(f"{cname:<26} | {len(idx_set):>4}D | {cb_f1:6.2f}%      | {cb_opt:6.2f}% | {casc_f1:6.2f}%     | {auc:6.2f}% | {fps:2d}건 ({fpr:4.1f}%)")

    # 결과 종합 저장
    full_output = {
        'baseline_30d': base_res,
        'lofo_results': lofo_results,
        'detrimental_features': [
            {'index': idx, 'name': fn, 'delta_f1': df1, 'delta_opt': dopt}
            for idx, fn, df1, dopt in detrimental_features
        ],
        'high_corr_pairs': [
            {'idx1': r, 'idx2': c, 'name1': fn1, 'name2': fn2, 'corr': val}
            for r, c, fn1, fn2, val in high_corr_pairs
        ],
        'results_level123': results_valid,
        'results_all_1906': results_all
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\n[*] LOFO 및 v3 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
