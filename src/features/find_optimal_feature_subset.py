# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: find_optimal_feature_subset.py
Purpose: Greedy Backward Elimination 및 Forward Selection을 통해
         전체 3-State F1-Score를 극대화하는 최적 v3 피처 서브셋 탐색
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v2_32d_features.npz"


def search_optimal_features():
    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    feat_names = list(data['feature_names'])
    N, D = X.shape

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    def eval_subset(feat_indices):
        oof_lgb = np.zeros(N, dtype=np.float32)
        oof_cb = np.zeros(N, dtype=np.float32)
        X_sub = X[:, feat_indices]

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects)):
            # Fast LightGBM
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=140,
                learning_rate=0.04,
                num_leaves=20,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.8,
                scale_pos_weight=pos_ratio,
                random_state=42 + fold,
                n_jobs=1,
                verbose=-1
            )
            clf_lgb.fit(X_sub[train_idx], y[train_idx])
            oof_lgb[val_idx] = clf_lgb.predict_proba(X_sub[val_idx])[:, 1]

            # Fast CatBoost
            clf_cb = CatBoostClassifier(
                iterations=140,
                learning_rate=0.045,
                depth=5,
                scale_pos_weight=pos_ratio,
                random_seed=42 + fold,
                thread_count=1,
                verbose=0
            )
            clf_cb.fit(X_sub[train_idx], y[train_idx])
            oof_cb[val_idx] = clf_cb.predict_proba(X_sub[val_idx])[:, 1]

        oof_ens = 0.5 * oof_lgb + 0.5 * oof_cb

        # 3-State F1 탐색 (최적 임계치)
        best_f1 = 0.0
        best_cov = 0.0
        best_th = (0.28, 0.56)
        for th_l in np.linspace(0.20, 0.38, 10):
            for th_h in np.linspace(0.52, 0.72, 11):
                mask = (oof_ens < th_l) | (oof_ens >= th_h)
                cov = float(np.mean(mask))
                if cov < 0.72:
                    continue
                score = f1_score(y[mask], (oof_ens[mask] >= th_h).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_cov = cov
                    best_th = (float(th_l), float(th_h))

        full_f1 = f1_score(y, (oof_ens >= 0.5).astype(int), zero_division=0)
        full_auc = roc_auc_score(y, oof_ens)
        return best_f1, 1.0 - best_cov, full_f1, full_auc, best_th

    print("[*] 32D 초기 서브셋 평가 중...")
    current_features = list(range(D))
    cur_f1, cur_rej, cur_full_f1, cur_auc, cur_th = eval_subset(current_features)
    print(f"    - 32D 초기: 3-State F1={cur_f1*100:.2f}%, 보류율={cur_rej*100:.2f}%, Full F1={cur_full_f1*100:.2f}%, AUC={cur_auc:.4f}")

    # 제거 후보 순서대로 소거 시도
    # 높은 상관관계 쌍 중 중복 제거 대상 및 ablation에서 성능 향상된 피처들
    # 3: rom, 27: zero_drift, 2: pitch_min, 5: hyperext_excess, 25: descent_smoothness, 17: z_diff, 12: pelvic_tilt, 22: lockout_rebound, 28: subject_norm_rom, 31: baseline_subtracted_peak, 29: trunk_length_scaled
    candidate_drops = [3, 27, 2, 5, 25, 17, 12, 22, 28, 31, 29, 26]

    print("\n[*] 순차적 소거(Greedy Backward Elimination) 진행 중...")
    best_subset = list(current_features)
    best_overall_f1 = cur_f1

    for drop_candidate in candidate_drops:
        if drop_candidate not in best_subset:
            continue
        test_subset = [f for f in best_subset if f != drop_candidate]
        t_f1, t_rej, t_full, t_auc, t_th = eval_subset(test_subset)
        fname = feat_names[drop_candidate]
        print(f"    - 제거 테스트: [{drop_candidate:2d}] {fname:<34} -> 3-State F1={t_f1*100:.2f}% (Δ={t_f1 - best_overall_f1:+0.2f}%p), 보류율={t_rej*100:.1f}%, AUC={t_auc:.4f}")
        if t_f1 >= best_overall_f1 - 0.002:  # 거의 동등하거나 향상되면 과감히 제거하여 모델 경량화 및 일반화 촉진
            best_subset = test_subset
            best_overall_f1 = max(best_overall_f1, t_f1)
            print(f"      ==> [채택] 피처 제거 확정! 남은 피처 수: {len(best_subset)}개 (현재 최고 F1: {best_overall_f1*100:.2f}%)")

    final_f1, final_rej, final_full, final_auc, final_th = eval_subset(best_subset)
    print("\n" + "=" * 80)
    print(f"[*] 최적 피처 서브셋 도출 완료:")
    print(f"    - 선정된 피처 수: {len(best_subset)}개 (초기 32개에서 {D - len(best_subset)}개 소거)")
    print(f"    - 3-State F1-Score: {final_f1*100:.2f}%")
    print(f"    - 판단 보류율(Rejection Rate): {final_rej*100:.2f}% (커버리지: {(1.0-final_rej)*100:.2f}%)")
    print(f"    - 100% Full F1-Score: {final_full*100:.2f}%")
    print(f"    - ROC-AUC: {final_auc:.4f}")
    print(f"    - 최적 임계치: Th_Low={final_th[0]:.3f}, Th_High={final_th[1]:.3f}")

    selected_names = [feat_names[i] for i in best_subset]
    removed_names = [feat_names[i] for i in range(D) if i not in best_subset]

    print("\n[선정된 피처 목록]:")
    for idx, name in enumerate(selected_names, 1):
        print(f"    {idx:2d}. {name}")

    print("\n[소거된 노이즈/중복 피처 목록]:")
    for idx, name in enumerate(removed_names, 1):
        print(f"    {idx:2d}. {name}")

    res = {
        'best_indices': best_subset,
        'selected_names': selected_names,
        'removed_names': removed_names,
        'metrics': {
            'three_state_f1': final_f1,
            'rejection_rate': final_rej,
            'coverage': 1.0 - final_rej,
            'full_f1': final_full,
            'auc': final_auc,
            'th_low': final_th[0],
            'th_high': final_th[1]
        }
    }
    with open(r"D:\src\models\second_models\babel_lunge\c5_optimal_feature_subset.json", 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("=" * 80)


if __name__ == '__main__':
    search_optimal_features()
