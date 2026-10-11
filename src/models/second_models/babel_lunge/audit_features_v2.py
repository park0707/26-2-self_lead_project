# -*- coding: utf-8 -*-
"""
바벨 런지 Condition 2: v2 30D 피처 전수 조사 및 분석 스크립트
1. 피처 간 다중공선성(Multicollinearity) 분석 (|r| > 0.85)
2. 타깃(y)과의 상관관계, F-통계량, 상호정보량(Mutual Information), 단일 ROC-AUC
3. LightGBM / CatBoost Feature Importance 기여도 분석
4. 잡음/유해 피처 식별
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

def audit_features():
    cache_file = r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v2_cache.npz"
    if not os.path.exists(cache_file):
        print("[-] Cache file not found!")
        return

    data = np.load(cache_file, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    views = data["views"]

    from lunge_c2_features_v2 import FEATURE_NAMES_30D
    feature_names = FEATURE_NAMES_30D

    print(f"[*] 총 시퀀스 수: {len(y)}, 정상: {np.sum(y==0)}, 결함: {np.sum(y==1)}")
    print(f"[*] 총 피처 수: {len(feature_names)}")

    df = pd.DataFrame(X, columns=feature_names)

    # 1. 분산(Variance) 체크 - 변동성이 거의 없는 죽은 피처 확인
    variances = df.var()
    low_var_features = variances[variances < 1e-4].index.tolist()
    print(f"\n[1] 저분산 (분산 < 1e-4) 피처 수: {len(low_var_features)}")
    for f in low_var_features:
        print(f"    - {f}: var = {variances[f]:.6f}")

    # 2. 피처 간 피어슨 상관계수 행렬 (|r| >= 0.85 쌍 확인)
    corr_matrix = df.corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr_matrix.iloc[i, j]
            if r >= 0.85:
                high_corr_pairs.append((feature_names[i], feature_names[j], r))

    print(f"\n[2] 고상관 다중공선성 쌍 (|r| >= 0.85): {len(high_corr_pairs)}건")
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"    - {f1} <--> {f2} : r = {r:.4f}")

    # 3. 타깃(y)과의 선별 지표 계산
    mi = mutual_info_classif(X, y, random_state=42)
    f_vals, p_vals = f_classif(X, y)
    
    auc_scores = []
    for col in df.columns:
        score = roc_auc_score(y, df[col])
        # 방향성 보정 (0.5 이하인 경우 반대 방향)
        auc_scores.append(max(score, 1 - score))

    # 4. GBDT 중요도 분석
    lgb_model = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X, y)
    lgb_imp = lgb_model.feature_importances_ / np.sum(lgb_model.feature_importances_)

    cat_model = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, random_seed=42, verbose=0)
    cat_model.fit(X, y)
    cat_imp = cat_model.get_feature_importance() / np.sum(cat_model.get_feature_importance())

    combined_imp = 0.5 * lgb_imp + 0.5 * cat_imp

    # 종합 리포트 생성
    report_df = pd.DataFrame({
        "Feature": feature_names,
        "Variance": variances.values,
        "AUC": auc_scores,
        "MI": mi,
        "F_stat": f_vals,
        "GBDT_Imp": combined_imp
    })

    report_df = report_df.sort_values(by="GBDT_Imp", ascending=False).reset_index(drop=True)
    print("\n[3] 피처 중요도 및 판별력 종합 순위:")
    print(report_df.to_string())

    # 5. 하위 성능 저해 후보 피처 추천
    print("\n[4] 성능 저해/잡음 후보군 (중요도 하위 및 단일 AUC 0.52 미만):")
    underperforming = report_df[(report_df["AUC"] < 0.525) & (report_df["GBDT_Imp"] < 0.02)]
    print(underperforming[["Feature", "AUC", "MI", "GBDT_Imp"]].to_string())

if __name__ == "__main__":
    audit_features()
