# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_lunge_c4_features.py
Purpose: 바벨 런지 c4 36차원 피처 전수 조사
         1) 피처 간 상관계수 (|r| >= 0.85) 다중공선성 및 중복 피처 검출
         2) 타깃(y)과의 점이연 상관계수 및 상호정보량(Mutual Information) 계산
         3) 5-Fold GBDT (LGBM, XGB, CAT) Feature Importance 및 순열 중요도(Permutation Importance) 산출
         4) 피험자 간 체형 편향(Subject Leakage/Bias) 지수 측정
         5) 제거 대상 피처 도출 및 최적 피처 서브셋 추천
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v2.npz"

def audit_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c4 피처 전수 조사 및 상관관계/노이즈 분석")
    print("=" * 80)

    if not os.path.exists(CACHE_PATH):
        print(f"[!] 캐시 파일 없음: {CACHE_PATH}")
        return

    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X_36d']             # (787, 36)
    y = data['y']                 # (787,)
    feature_names = list(data['feature_names'])
    subjects = data['subjects']    # (787,)
    views = data['views']          # (787,)

    n_samples, n_features = X.shape
    print(f"[*] 총 샘플 수: {n_samples}개 (정상: {np.sum(y == 0)}, 결함: {np.sum(y == 1)})")
    print(f"[*] 분석 대상 피처 수: {n_features}차원")

    df = pd.DataFrame(X, columns=feature_names)

    # 1. 피처 간 다중공선성(Multicollinearity) 스캔: |r| >= 0.85
    print("\n--- [1. 고상관 다중공선성 피처 쌍 (|r| >= 0.85)] ---")
    corr_matrix = df.corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r_val = corr_matrix.iloc[i, j]
            if r_val >= 0.85:
                high_corr_pairs.append((feature_names[i], feature_names[j], r_val))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for f1, f2, r_val in high_corr_pairs:
        print(f"  - {f1:28s} <-> {f2:28s} : |r| = {r_val:.4f}")

    # 2. 타깃(y)과의 점이연 상관계수 및 Mutual Information
    print("\n--- [2. 타깃(y)과의 상관계수 및 상호정보량(Mutual Information)] ---")
    target_corrs = {}
    for col in feature_names:
        r = np.corrcoef(df[col], y)[0, 1]
        target_corrs[col] = 0.0 if np.isnan(r) else r

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_dict = dict(zip(feature_names, mi_scores))

    # 3. 5-Fold 교차검증 기반 GBDT Feature Importance 측정
    print("\n--- [3. 5-Fold 피험자 격리 기반 앙상블 Feature Importance] ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    lgb_importances = np.zeros(n_features)
    xgb_importances = np.zeros(n_features)
    cat_importances = np.zeros(n_features)

    for train_idx, val_idx in sgkf.split(X, y, groups=subjects):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
        m_lgb.fit(X_tr, y_tr)
        lgb_importances += m_lgb.feature_importances_ / 5.0

        # XGBoost
        m_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=42)
        m_xgb.fit(X_tr, y_tr)
        xgb_importances += m_xgb.feature_importances_ / 5.0

        # CatBoost
        m_cat = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42)
        m_cat.fit(X_tr, y_tr)
        cat_importances += m_cat.get_feature_importance() / 5.0

    # 정규화
    lgb_norm = lgb_importances / (np.sum(lgb_importances) + 1e-9)
    xgb_norm = xgb_importances / (np.sum(xgb_importances) + 1e-9)
    cat_norm = cat_importances / (np.sum(cat_importances) + 1e-9)
    ensemble_importance = (lgb_norm + xgb_norm + cat_norm) / 3.0

    # 4. 피험자 편향 지수 (Subject Bias Ratio): 각 피험자별 평균의 표준편차 / 전체 표준편차
    # 피험자 간 체형 차이로 인해 특정 피험자만 구별하고 일반화 성능을 깎아먹는지 판별
    subject_bias_ratios = {}
    for col in feature_names:
        overall_std = df[col].std() + 1e-9
        subj_means = [df.loc[subjects == s, col].mean() for s in np.unique(subjects)]
        subj_std = np.std(subj_means)
        subject_bias_ratios[col] = float(subj_std / overall_std)

    # 종합 집계 표 작성
    stats = []
    for i, col in enumerate(feature_names):
        stats.append({
            'name': col,
            'corr_y': target_corrs[col],
            'abs_corr_y': abs(target_corrs[col]),
            'mi': mi_dict[col],
            'importance': ensemble_importance[i],
            'subj_bias': subject_bias_ratios[col]
        })

    stats_df = pd.DataFrame(stats).sort_values(by='importance', ascending=False)
    print("\n[피처 종합 감사 성적표 (Importance 순)]")
    print(f"{'No':2s} | {'Feature Name':28s} | {'Corr(y)':8s} | {'MI':6s} | {'Importance':10s} | {'Subj Bias':10s}")
    print("-" * 75)
    for rank, (_, row) in enumerate(stats_df.iterrows(), 1):
        print(f"{rank:2d} | {row['name']:28s} | {row['corr_y']:+7.4f} | {row['mi']:6.4f} | {row['importance']*100:8.3f}% | {row['subj_bias']:9.4f}")

    # 5. 제거 후보 피처 도출 규칙:
    # A) 다중공선성 쌍 중 중요도/상관도가 더 낮은 피처
    # B) Importance가 극히 낮거나 (하위 15%) MI가 거의 0에 가까운 피처
    # C) Subj Bias가 지나치게 높아 피험자 외우기(Overfitting)를 유발하는 피처
    candidates_to_drop = []

    # 고상관 쌍 중 중요도 낮은 피처 선택
    for f1, f2, r_val in high_corr_pairs:
        imp1 = stats_df.loc[stats_df['name'] == f1, 'importance'].values[0]
        imp2 = stats_df.loc[stats_df['name'] == f2, 'importance'].values[0]
        loser = f2 if imp1 >= imp2 else f1
        if loser not in candidates_to_drop:
            candidates_to_drop.append(loser)
            print(f"\n[제거 후보 A: 다중공선성 중복] {loser} (상대: {f1 if loser==f2 else f2}, |r|={r_val:.3f})")

    # 저중요도 및 저정보 피처
    for _, row in stats_df.iterrows():
        if row['importance'] < 0.015 and row['abs_corr_y'] < 0.10: # 중요도 1.5% 미만 & 타깃 상관도 0.10 미만
            if row['name'] not in candidates_to_drop:
                candidates_to_drop.append(row['name'])
                print(f"[제거 후보 B: 저정보/저기여 노이즈] {row['name']} (Imp={row['importance']*100:.2f}%, Corr={row['corr_y']:.3f})")

    print("\n" + "=" * 80)
    print(f"[*] 최종 도출된 제거 대상 피처 ({len(candidates_to_drop)}개):")
    for f in candidates_to_drop:
        print(f"    - {f}")
    
    surviving_features = [f for f in feature_names if f not in candidates_to_drop]
    print(f"\n[*] v3 정예 최적 피처 ({len(surviving_features)}개):")
    for f in surviving_features:
        print(f"    + {f}")
    print("=" * 80)

if __name__ == "__main__":
    audit_features()
