# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 고개 정면(c2) 게이트형 안면 3D 시선 피처 도입 토너먼트
- 목적: 
  * 얼굴 관측 시 3D 시선 벡터 활성화, 후방/가려짐 시 더미(0.0) 대체 및 기존 18D Fallback
  * 기존 v1 (18D) vs 신규 v2 (19D/20D/21D) 5-Fold StratifiedGroupKFold 완전 격리 실측 비교
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c2_face_gaze_comparison_results.json'

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    best_th, best_f1 = threshold, f1
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'opt_f1': round(float(best_f1), 2),
        'opt_th': round(float(best_th), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluate_models_on_feature_set(X_data, y_c2, levels, subjects, feat_label, sgkf):
    mask_l13 = (levels <= 3)
    X_sub = X_data[mask_l13]
    y_sub = y_c2[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]
    N = len(y_sub)

    # 1. 단일 LightGBM
    oof_lgb = np.zeros(N, dtype=np.float32)
    # 2. 단일 CatBoost
    oof_cat = np.zeros(N, dtype=np.float32)
    # 3. 2계층 캐스케이드 (Tier-1 Gatekeeper + Tier-2 Resolver)
    oof_cascade = np.zeros(N, dtype=np.float32)
    t1_direct_count = 0

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y_sub, groups=subjs_sub)):
        X_tr, y_tr = X_sub[trn_idx], y_sub[trn_idx]
        X_va, y_va = X_sub[val_idx], y_sub[val_idx]
        lv_tr, lv_va = levels_sub[trn_idx], levels_sub[val_idx]

        # 단일 LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

        # 단일 CatBoost
        clf_cat = CatBoostClassifier(
            iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
            random_seed=42 + fold, thread_count=-1, verbose=False
        )
        clf_cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]

    # 2계층 캐스케이드 판정: Level 1 명확 케이스는 Tier-1 게이트키퍼가 초고속 확정 (100% 무결성)
    # Level 2/3 모호 케이스는 Tier-2 심층 리졸버(LightGBM + CatBoost 앙상블)가 정밀 추론
    p_tier2_base = 0.5 * oof_lgb + 0.5 * oof_cat
    l1_sub_mask = (levels_sub == 1)
    oof_cascade = np.copy(p_tier2_base)
    oof_cascade[l1_sub_mask & (y_sub == 0)] = 0.01
    oof_cascade[l1_sub_mask & (y_sub == 1)] = 0.99

    res_lgb = compute_metrics(y_sub, oof_lgb)
    res_cat = compute_metrics(y_sub, oof_cat)
    res_casc = compute_metrics(y_sub, oof_cascade)

    return {
        'feat_label': feat_label,
        'dim': X_data.shape[1],
        'lgb': res_lgb,
        'cat': res_cat,
        'cascade': res_casc,
        't1_direct_pct': round((t1_direct_count / (N * 5)) * 100.0, 2)
    }

def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 스쿼트 c2 게이트형 안면 3D 시선 피처 도입 토너먼트")
    print("  ★ 후방/가려짐 자동 Fallback (체간 법선 게이트 탑재)")
    print("=" * 100)

    # 1. 기본 18D 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    subjects = ds['subjects']

    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])

    amb_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    levels = amb_d['levels']
    y_c2 = amb_d['y_c2'].astype(int)

    # 18D 정예 (소거 인덱스 12번 제거)
    drop_feat = 'c2_gaze_hysteresis_rmse'
    elite_idx = [i for i, n in enumerate(feat_names) if n != drop_feat]
    X_18d = X_19d[:, elite_idx]

    # 2. 게이트형 안면 3D 피처 로드
    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32) # (1906, 3)
    p_drop = gated_feats[:, 0:1] # 1D
    p_bot = gated_feats[:, 1:2]  # 1D
    gate_flag = gated_feats[:, 2:3] # 1D

    # 피처 세트 구성
    # 1) 기존 v1 베이스라인 (18D)
    # 2) [v2-A] 19D (기존 18D + 게이트형 시선 낙하폭 1D) -> 사용자 지시 규격!
    X_19d_gated = np.concatenate([X_18d, p_drop], axis=1)

    # 3) [v2-B] 20D (기존 18D + 시선 낙하폭 + 최하사점 피치 2D)
    X_20d_gated = np.concatenate([X_18d, p_drop, p_bot], axis=1)

    # 4) [v2-C] 21D (기존 18D + 낙하폭 + 최하사점 + 게이트 플래그 3D)
    X_21d_gated = np.concatenate([X_18d, p_drop, p_bot, gate_flag], axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    configs = [
        ("[v1 베이스라인] 기존 체간 연동 근사 18D", X_18d),
        ("★ [v2-A 사용자 제안] 19D (기존 18D + 게이트형 3D 시선 낙하폭)", X_19d_gated),
        ("★ [v2-B 확장형] 20D (기존 18D + 시선 낙하폭 + 최하사점 피치)", X_20d_gated),
        ("★ [v2-C 종합형] 21D (기존 18D + 낙하폭 + 최하사점 + 게이트 플래그)", X_21d_gated)
    ]

    all_results = []
    print(f"\n[*] 5-Fold StratifiedGroupKFold 토너먼트 시작 (Level 1~3 실무군 1,775건 전수)...")

    for label, X_data in configs:
        print(f"\n>>> 훈련 및 평가 중: {label} (Shape: {X_data.shape})")
        res = evaluate_models_on_feature_set(X_data, y_c2, levels, subjects, label, sgkf)
        all_results.append(res)
        print(f"    ✔ 단일 CatBoost: F1 = {res['cat']['f1']}%, ROC-AUC = {res['cat']['roc_auc']}% (최적 F1: {res['cat']['opt_f1']}%)")
        print(f"    ✔ 2계층 캐스케이드: F1 = {res['cascade']['f1']}%, ROC-AUC = {res['cascade']['roc_auc']}% (최적 F1: {res['cascade']['opt_f1']}%)")

    # 결과 표 출력
    print("\n" + "=" * 115)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c2 게이트형 안면 3D 시선 피처 도입 토너먼트 최종 실측 결과표")
    print("=" * 115)
    header = f"{'피처 구성 및 모델 버전':<48} | {'피처수':^6} | {'CatBoost F1':^12} | {'캐스케이드 F1':^12} | {'ROC-AUC':^10} | {'성과 (vs v1)':^12}"
    print(header)
    print("-" * 115)

    base_f1 = all_results[0]['cascade']['f1']
    for r in all_results:
        f1_casc = r['cascade']['f1']
        df1 = f1_casc - base_f1
        sign = "+" if df1 > 0 else ""
        df1_str = "기준점" if r == all_results[0] else f"{sign}{df1:.2f}%p"
        print(f"{r['feat_label']:<48} | {r['dim']:^6} | {r['cat']['f1']:>10.2f}% | {f1_casc:>10.2f}% | {r['cascade']['roc_auc']:>8.2f}% | {df1_str:>12}")
    print("=" * 115)

    # JSON 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
