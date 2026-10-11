# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_lunge_c5_features.py
Purpose: 바벨 런지 c5 v2 32차원 피처 전수 조사
         1. 32개 피처 간 다중공선성(피어슨 상관계수 >= 0.85) 쌍 탐지
         2. 라벨과의 단독 상관도 및 Mutual Information
         3. 5-Fold 피험자 격리 기준 Feature Importance 및 Permutation Importance
         4. 피처 소거(Ablation) 시뮬레이션을 통한 최적 압축 피처셋 도출
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v2_32d_features.npz"


def audit_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c5 32차원 피처 전수 조사 및 간섭 분석")
    print("=" * 80)

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    feat_names = list(data['feature_names'])

    N, D = X.shape
    print(f"[*] 총 샘플 수: {N}개, 피처 수: {D}개, 정상: {np.sum(y==0)}개, 결함: {np.sum(y==1)}개")

    # 1. 다중공선성 분석 (상관계수 행렬)
    df = pd.DataFrame(X, columns=feat_names)
    corr_matrix = df.corr().abs()

    high_corr_pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            val = corr_matrix.iloc[i, j]
            if val >= 0.85:
                high_corr_pairs.append({
                    'f1': feat_names[i],
                    'f2': feat_names[j],
                    'corr': float(val)
                })

    print(f"\n[1] 고상관(다중공선성 R >= 0.85) 피처 쌍 탐지: 총 {len(high_corr_pairs)}쌍")
    high_corr_pairs.sort(key=lambda x: x['corr'], reverse=True)
    for p in high_corr_pairs:
        print(f"    - {p['f1']} <--> {p['f2']}: |r| = {p['corr']:.3f}")

    # 2. 라벨과의 Mutual Information 및 단독 AUC
    mi = mutual_info_classif(X, y, random_state=42)
    single_aucs = []
    for d in range(D):
        try:
            auc = roc_auc_score(y, X[:, d])
            auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
        single_aucs.append(auc)

    feat_stats = []
    for d in range(D):
        feat_stats.append({
            'idx': d,
            'name': feat_names[d],
            'mi': float(mi[d]),
            'single_auc': float(single_aucs[d]),
            'std': float(np.std(X[:, d]))
        })

    feat_stats.sort(key=lambda x: x['single_auc'], reverse=True)
    print("\n[2] 피처별 단독 AUC 및 상호정보량(MI) 상위 10개:")
    for s in feat_stats[:10]:
        print(f"    - {s['name']:<36} : AUC={s['single_auc']:.3f}, MI={s['mi']:.3f}, Std={s['std']:.3f}")

    print("\n[2-1] 피처별 단독 AUC 및 상호정보량(MI) 하위 8개 (노이즈 가능성 후보):")
    for s in feat_stats[-8:]:
        print(f"    - {s['name']:<36} : AUC={s['single_auc']:.3f}, MI={s['mi']:.3f}, Std={s['std']:.3f}")

    # 3. 5-Fold Baseline (LightGBM) 전체 F1 측정
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    def eval_features(feature_indices):
        oof_probs = np.zeros(N, dtype=np.float32)
        X_sub = X[:, feature_indices]
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects)):
            clf = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.035,
                num_leaves=24,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.8,
                scale_pos_weight=pos_ratio,
                random_state=42 + fold,
                n_jobs=1,
                verbose=-1
            )
            clf.fit(X_sub[train_idx], y[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_sub[val_idx])[:, 1]

        # 3-State F1 탐색
        best_f1 = 0.0
        best_cov = 0.0
        best_rejection = 0.0
        for th_l in np.linspace(0.20, 0.40, 11):
            for th_h in np.linspace(0.55, 0.75, 11):
                mask = (oof_probs < th_l) | (oof_probs >= th_h)
                cov = float(np.mean(mask))
                if cov < 0.70:
                    continue
                score = f1_score(y[mask], (oof_probs[mask] >= th_h).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_cov = cov
                    best_rejection = 1.0 - cov

        full_f1 = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
        full_auc = roc_auc_score(y, oof_probs)
        return best_f1, best_rejection, full_f1, full_auc

    all_indices = list(range(D))
    base_3s_f1, base_rej, base_full_f1, base_auc = eval_features(all_indices)
    print(f"\n[3] 32D 전체 피처 기준 베이스라인 성능:")
    print(f"    - 3-State F1: {base_3s_f1*100:.2f}% | 판단 보류율: {base_rej*100:.2f}% | Full F1: {base_full_f1*100:.2f}% | AUC: {base_auc:.4f}")

    # 4. Feature Ablation (단일 피처 제거 실험)
    print("\n[4] 단일 피처 소거(Leave-One-Out Ablation) 전수 조사 중...")
    ablation_results = []
    for d in range(D):
        subset = [i for i in range(D) if i != d]
        f1_sub, rej_sub, full_sub, auc_sub = eval_features(subset)
        delta_f1 = f1_sub - base_3s_f1
        ablation_results.append({
            'removed_idx': d,
            'removed_name': feat_names[d],
            'f1_without': float(f1_sub),
            'delta_3s_f1': float(delta_f1),
            'rejection': float(rej_sub),
            'full_f1': float(full_sub),
            'auc': float(auc_sub)
        })

    ablation_results.sort(key=lambda x: x['delta_3s_f1'], reverse=True)
    print("\n[*] 제거 시 성능이 가장 크게 향상되는 피처들 (성능 저해/노이즈 피처):")
    for r in ablation_results[:10]:
        print(f"    - [{r['removed_idx']:2d}] {r['removed_name']:<36}: 3-State F1={r['f1_without']*100:.2f}% (Δ = {r['delta_3s_f1']*100:+.2f}%p), 보류율={r['rejection']*100:.1f}%")

    print("\n[*] 제거 시 성능이 가장 크게 저하되는 핵심 필수 피처들:")
    for r in ablation_results[-10:]:
        print(f"    - [{r['removed_idx']:2d}] {r['removed_name']:<36}: 3-State F1={r['f1_without']*100:.2f}% (Δ = {r['delta_3s_f1']*100:+.2f}%p)")

    # 결과 요약 JSON 저장
    audit_data = {
        'high_corr_pairs': high_corr_pairs,
        'feat_stats': feat_stats,
        'baseline': {
            'three_state_f1': base_3s_f1,
            'rejection_rate': base_rej,
            'full_f1': base_full_f1,
            'auc': base_auc
        },
        'ablation': ablation_results
    }
    out_json = r"D:\src\models\second_models\babel_lunge\c5_feature_audit_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 피처 전수조사 결과 저장 완료: {out_json}")
    print("=" * 80)


if __name__ == '__main__':
    audit_features()
