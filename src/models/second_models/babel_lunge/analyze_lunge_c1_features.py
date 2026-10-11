# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: analyze_lunge_c1_features.py
Purpose: v2 38차원 피처 전수 조사 및 최적화 분석
         1. 피처별 기초 통계 및 결측치/분산 점검
         2. 피처-피처 간 상관관계 분석 (|r| >= 0.85 다중공선성 쌍 식별)
         3. 피처-타깃 상호정보량(Mutual Information) 및 상관계수
         4. GBDT 5-Fold 기반 Feature Importance (Gain & Split)
         5. Sequential Feature Ablation (단일 피처 제거 시 F1/AUC 변화)
         6. 최종 최적 피처 서브셋 추천
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import lightgbm as lgb

sys.path.append(r"D:\src\features")
from lunge_c1_features_v2 import LUNGE_C1_FEATURE_NAMES_V2

CACHE_PATH = r"D:\Data\피처_데이터\bl1_v2_dataset_cache.pkl"
OUTPUT_REPORT_JSON = r"D:\src\models\second_models\babel_lunge\lunge_c1_feature_analysis.json"


def run_feature_investigation():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 런지 c1 38D 피처 전수 조사 및 최적화 분석")
    print("=" * 80)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"캐시 파일을 찾을 수 없습니다: {CACHE_PATH}")

    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)

    X = np.array(cache_data["X"], dtype=np.float32)
    y = np.array(cache_data["y"], dtype=np.int32)
    subjects = np.array(cache_data.get("groups", cache_data.get("subjects")))
    views = np.array(cache_data["views"])
    feature_names = cache_data.get("feature_names", LUNGE_C1_FEATURE_NAMES_V2)

    n_samples, n_features = X.shape
    print(f"[*] 데이터셋 로드 완료: N={n_samples}, D={n_features}")
    print(f"[*] 클래스 분포: 정상(0)={np.sum(y==0)}, 결함(1)={np.sum(y==1)}")

    # 1. 기초 통계 & 결측치/분산
    df_feat = pd.DataFrame(X, columns=feature_names)
    stats = []
    low_variance_feats = []
    for col in feature_names:
        s = df_feat[col]
        mean_val = float(s.mean())
        std_val = float(s.std())
        min_val = float(s.min())
        max_val = float(s.max())
        nan_cnt = int(s.isna().sum())
        # target correlation
        corr_with_y = float(s.corr(pd.Series(y)))
        if std_val < 1e-4:
            low_variance_feats.append(col)
        stats.append({
            "feature": col,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "nan_count": nan_cnt,
            "corr_y": round(corr_with_y, 4)
        })

    print(f"[+] 저분산(Zero/Low Variance) 피처: {low_variance_feats if low_variance_feats else '없음 (양호)'}")

    # 2. 피처-피처 간 상관관계 분석 (|r| >= 0.85 식별)
    corr_matrix = df_feat.corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r_val = corr_matrix.iloc[i, j]
            if r_val >= 0.85:
                high_corr_pairs.append({
                    "feat1": feature_names[i],
                    "feat2": feature_names[j],
                    "corr": round(float(r_val), 4),
                    "feat1_corr_y": round(float(df_feat[feature_names[i]].corr(pd.Series(y))), 4),
                    "feat2_corr_y": round(float(df_feat[feature_names[j]].corr(pd.Series(y))), 4),
                })

    high_corr_pairs.sort(key=lambda x: x["corr"], reverse=True)
    print(f"\n[*] 높은 상관관계 (|r| >= 0.85) 다중공선성 피처 쌍 ({len(high_corr_pairs)}쌍 발견):")
    for pair in high_corr_pairs:
        print(f"  - {pair['feat1']} <--> {pair['feat2']}: r={pair['corr']:.4f} (y상관: {pair['feat1_corr_y']} vs {pair['feat2_corr_y']})")

    # 3. 상호정보량 (Mutual Information)
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranking = [
        {"feature": name, "mi_score": round(float(score), 4)}
        for name, score in zip(feature_names, mi_scores)
    ]
    mi_ranking.sort(key=lambda x: x["mi_score"], reverse=True)
    print("\n[*] 상호정보량(MI) 상위 10개 피처:")
    for item in mi_ranking[:10]:
        print(f"  - {item['feature']}: MI = {item['mi_score']:.4f}")

    # 4. LightGBM 5-Fold Feature Importance
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_gains = np.zeros(n_features)
    fold_splits = np.zeros(n_features)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / pos_cnt)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "scale_pos_weight": spw,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

    oof_preds_base = np.zeros(n_samples)
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        oof_preds_base[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        fold_gains += model.feature_importance(importance_type="gain")
        fold_splits += model.feature_importance(importance_type="split")

    base_f1 = f1_score(y, (oof_preds_base >= 0.5).astype(int))
    base_auc = roc_auc_score(y, oof_preds_base)
    print(f"\n[*] 38D 베이스라인 성능 (CV 5-Fold): F1={base_f1*100:.2f}%, AUC={base_auc*100:.2f}%")

    fold_gains /= 5.0
    fold_splits /= 5.0
    lgb_importance = [
        {
            "feature": name,
            "gain": round(float(g), 2),
            "split": round(float(s), 2),
            "relative_gain": round(float(g / np.sum(fold_gains) * 100), 2)
        }
        for name, g, s in zip(feature_names, fold_gains, fold_splits)
    ]
    lgb_importance.sort(key=lambda x: x["gain"], reverse=True)

    print("\n[*] LightGBM Feature Importance (Gain 상위 10개):")
    for item in lgb_importance[:10]:
        print(f"  - {item['feature']:<30}: Gain={item['gain']:>8.1f} ({item['relative_gain']:>5.2f}%) | Split={item['split']:>5.1f}")

    print("\n[*] LightGBM Feature Importance (Gain 하위 10개 - 노이즈/비기여 의심 피처):")
    for item in lgb_importance[-10:]:
        print(f"  - {item['feature']:<30}: Gain={item['gain']:>8.1f} ({item['relative_gain']:>5.2f}%) | Split={item['split']:>5.1f}")

    # 5. Single Feature Ablation Study (각 피처 1개씩 제거 시 CV F1 / AUC 변화 측정)
    print("\n[*] Single Feature Ablation Study (제거 시 성능 변화 전수 감사)...")
    ablation_results = []
    for i, col_to_remove in enumerate(feature_names):
        remaining_indices = [idx for idx in range(n_features) if idx != i]
        X_sub = X[:, remaining_indices]

        oof_preds_sub = np.zeros(n_samples)
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects)):
            X_train, y_train = X_sub[train_idx], y[train_idx]
            X_val, y_val = X_sub[val_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            oof_preds_sub[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        sub_f1 = f1_score(y, (oof_preds_sub >= 0.5).astype(int))
        sub_auc = roc_auc_score(y, oof_preds_sub)
        delta_f1 = (sub_f1 - base_f1) * 100
        delta_auc = (sub_auc - base_auc) * 100

        ablation_results.append({
            "removed_feature": col_to_remove,
            "sub_f1": round(float(sub_f1), 4),
            "sub_auc": round(float(sub_auc), 4),
            "delta_f1": round(float(delta_f1), 3),
            "delta_auc": round(float(delta_auc), 3)
        })

    # delta_f1이 양수(+)인 경우 = 해당 피처를 제거했을 때 오히려 성능이 오르는 '성능 저해 피처'
    harmful_features = [res for res in ablation_results if res["delta_f1"] > 0]
    harmful_features.sort(key=lambda x: x["delta_f1"], reverse=True)

    print(f"\n[*] 제거 시 F1 향상되는 성능 저해/노이즈 피처 ({len(harmful_features)}개 발견):")
    for item in harmful_features:
        print(f"  - 제거: {item['removed_feature']:<30} -> F1 변화: {item['delta_f1']:>+6.2f}%p | AUC 변화: {item['delta_auc']:>+6.2f}%p")

    # 결과 종합 저장
    full_report = {
        "dataset_meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "base_f1": float(base_f1),
            "base_auc": float(base_auc)
        },
        "stats": stats,
        "high_corr_pairs": high_corr_pairs,
        "mi_ranking": mi_ranking,
        "lgb_importance": lgb_importance,
        "ablation_results": ablation_results,
        "harmful_features": harmful_features
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_JSON), exist_ok=True)
    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] 전수 조사 결과 보고서 JSON 저장 완료: {OUTPUT_REPORT_JSON}")


if __name__ == "__main__":
    run_feature_investigation()
