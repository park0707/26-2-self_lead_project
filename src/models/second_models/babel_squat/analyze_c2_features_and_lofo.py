# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Alignment)
Module: analyze_c2_features_and_lofo.py
Purpose: c2 v3 13대 피처 간 다중공선성(상관관계) 분석 및 5-Fold LOFO(Leave-One-Feature-Out) 성능 저해 피처 규명
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c2_model_v3 import FEATURE_NAMES_13, V2_TO_V3_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'c2_feature_analysis_results.json')


def evaluate_features(X_mat, y_arr, subjs, sgkf):
    N = len(y_arr)
    oof_prob = np.zeros(N, dtype=np.float32)

    for trn_idx, val_idx in sgkf.split(X_mat, y_arr, groups=subjs):
        X_tr, y_tr = X_mat[trn_idx], y_arr[trn_idx]
        X_va, y_va = X_mat[val_idx], y_arr[val_idx]

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_tr, y_tr)

        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        p_cb = cb.predict_proba(X_va)[:, 1]
        p_lgb = clf.predict_proba(X_va)[:, 1]
        oof_prob[val_idx] = 0.60 * p_cb + 0.40 * p_lgb

    pred = (oof_prob >= 0.50).astype(int)
    f1 = f1_score(y_arr, pred, zero_division=0) * 100.0
    prec = precision_score(y_arr, pred, zero_division=0) * 100.0
    rec = recall_score(y_arr, pred, zero_division=0) * 100.0
    auc = roc_auc_score(y_arr, oof_prob) * 100.0

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'roc_auc': round(float(auc), 2)
    }, oof_prob


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] c2 v3 13D 피처 간 상관관계 분석 및 5-Fold LOFO 저해 피처 규명 (N=1,906)")
    print("=" * 115)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c2 = ds['Y_defects'][:, 1].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c2)

    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])
    elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
    X_18d = X_19d[:, elite_idx]

    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32)[:, :2]
    X_20d = np.concatenate([X_18d, gated_feats], axis=1)

    X_13d = X_20d[:, V2_TO_V3_INDICES]
    names_13 = FEATURE_NAMES_13

    print(f"[*] 데이터 로드 완료: N={N_total}, 피처 차원={X_13d.shape[1]}D, 결함 비율={np.mean(y_c2)*100:.2f}%")

    # 2. 피처 간 상관계수(Correlation) 분석
    df_feats = pd.DataFrame(X_13d, columns=names_13)
    corr_matrix = df_feats.corr(method='pearson')

    print("\n" + "=" * 90)
    print("  [1. 피처 간 상관계수 고간섭(|r| >= 0.70) 쌍 정밀 탐색]")
    print("=" * 90)
    high_corr_pairs = []
    for i in range(len(names_13)):
        for j in range(i + 1, len(names_13)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= 0.70:
                high_corr_pairs.append((names_13[i], names_13[j], r))
                print(f"  • {names_13[i]:<35} <---> {names_13[j]:<35} | r = {r:+.3f}")

    if not high_corr_pairs:
        print("  (고간섭 |r| >= 0.70 피처 쌍 없음)")

    # 3. Baseline (13D 전체) 5-Fold 실측
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n[*] Baseline (13D 전체) 5-Fold 평가 진행 중...")
    base_res, base_oof = evaluate_features(X_13d, y_c2, subjects, sgkf)
    print(f"  >>> Baseline 13D: F1 = {base_res['f1']}% | Precision = {base_res['precision']}% | Recall = {base_res['recall']}% | AUC = {base_res['roc_auc']}%")

    # 4. LOFO (Leave-One-Feature-Out) 소거 실험
    print("\n" + "=" * 115)
    print("  [2. 13대 피처 5-Fold LOFO(하나씩 소거) 성능 변화 분석]")
    print("=" * 115)
    print(f"{'제거 대상 피처 (LOFO)':<38} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<12} | {'ROC-AUC':<10} | {'F1 변화 (Δ)'}")
    print("-" * 115)

    lofo_results = []
    harmful_features = []

    for i, f_name in enumerate(names_13):
        keep_indices = [k for k in range(len(names_13)) if k != i]
        X_minus = X_13d[:, keep_indices]
        m_res, _ = evaluate_features(X_minus, y_c2, subjects, sgkf)

        df1 = m_res['f1'] - base_res['f1']
        sign = "+" if df1 > 0 else ""
        delta_str = f"{sign}{df1:.2f}%p"

        # 제거했을 때 오히려 F1이 오르면 성능을 깎아먹고 있던 방해 피처!
        is_harmful = df1 > 0
        tag = " ⚠️ [방해 피처]" if is_harmful else ("  (핵심 피처)" if df1 <= -0.20 else "  (중립)")
        if is_harmful:
            harmful_features.append((f_name, df1))

        print(f"{f_name:<38} | {m_res['precision']:6.2f}%    | {m_res['recall']:6.2f}%    | {m_res['f1']:6.2f}%      | {m_res['roc_auc']:6.2f}%    | {delta_str:>8} {tag}")
        lofo_results.append({
            'removed_feature': f_name,
            'metrics': m_res,
            'delta_f1': round(df1, 2),
            'is_harmful': is_harmful
        })

    print("=" * 115)

    # 5. 방해 피처 조합 소거 실험 (v4 후보 도출)
    print("\n" + "=" * 90)
    print("  [3. 식별된 방해 피처 소거 최적화 실험]")
    print("=" * 90)

    # 방해 피처들을 F1 개선폭 순으로 정렬
    harmful_features.sort(key=lambda x: x[1], reverse=True)
    print(f"[*] 개별 소거 시 성능 향상 피처 목록: {[f[0] for f in harmful_features]}")

    results_payload = {
        'high_corr_pairs': [(p[0], p[1], round(float(p[2]), 4)) for p in high_corr_pairs],
        'baseline_13d': base_res,
        'lofo_results': lofo_results,
        'harmful_candidates': harmful_features
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 분석 결과 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
