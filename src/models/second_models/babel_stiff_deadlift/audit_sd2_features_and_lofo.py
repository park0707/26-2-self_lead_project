# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd2 (척추의 중립 위반 / Spine Neutrality)
Script: audit_sd2_features_and_lofo.py
Purpose:
  1. 30D 스티프 척추 피처 단독 변별력 감사 (정상 vs 결함 평균/SD, Cohen's d, 단독 AUC, 분산)
  2. 피처 간 다중공선성(|r| >= 0.85) 상관관계 감사
  3. LOFO (Leave-One-Feature-Out) 전수 감사: 30개 피처를 하나씩 제거하면서 5-Fold 결함 F1, AUC 변화량 측정
  4. 성능 저해(Negative Impact) 및 고상관 잉여 피처 목록 확정하여 JSON 저장
"""

import os
import sys
import time
import glob
import json
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.stiff_spine_feature_extractor import (
    extract_stiff_spine_features_30d,
    FEATURE_NAMES_30_STIFF_SPINE
)

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
OUTPUT_JSON = r"D:\src\models\second_models\babel_stiff_deadlift\sd2_feature_audit_results.json"


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    spine_cache = {}
    if os.path.exists(SPINEPOSE_CACHE):
        sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        sids = sc['sequence_ids']
        s_vision = sc['spine_9_64f_vision']
        spine_cache = {sids[i]: s_vision[i] for i in range(len(sids))}

    seq_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        m_sub = re.search(r'피험자(\d+)', sid)
        sub_num = int(m_sub.group(1)) if m_sub else 1
        m_key = re.search(r'Key(\d+)', sid)
        key_num = int(m_key.group(1)) if m_key else 209

        # Key 209, 210: 정상(0), Key 211~216: 결함(1)
        lbl = 0 if (key_num == 209 or key_num == 210) else 1

        data = np.load(f, allow_pickle=True)
        kps = data['kps_33_3d_64f'] if 'kps_33_3d_64f' in data else data['kps_33_2d_64f']
        s_pts = spine_cache.get(sid, None)

        seq_ids.append(sid)
        labels.append(lbl)
        subject_ids.append(sub_num)
        view_ids.append(vn)
        kps_list.append(kps)
        spine_list.append(s_pts)

    N = len(seq_ids)
    X_30d = np.zeros((N, 30), dtype=np.float32)
    for i in range(N):
        X_30d[i] = extract_stiff_spine_features_30d(kps_list[i], spine_list[i])

    return X_30d, np.array(labels, dtype=np.int64), np.array(subject_ids, dtype=np.int64), np.array(view_ids, dtype=object), seq_ids


def evaluate_features_subset(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[float, float, float]:
    """LightGBM + CatBoost 5-Fold 피험자 격리 빠른 평가 (F1, AUC, 정확도)"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        # LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.04, max_depth=4, num_leaves=15,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        m_lgb.fit(X_tr, y[train_idx])
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        # CatBoost
        m_cat = CatBoostClassifier(iterations=120, learning_rate=0.04, depth=4, verbose=0, random_seed=42)
        m_cat.fit(X_tr, y[train_idx])
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        oof_probs[val_idx] = 0.5 * p_lgb + 0.5 * p_cat

    bin_pred = (oof_probs >= 0.5).astype(int)
    f1 = f1_score(y, bin_pred, zero_division=0)
    auc = roc_auc_score(y, oof_probs)
    acc = accuracy_score(y, bin_pred)
    return f1, auc, acc


def run_audit():
    print("=" * 110)
    print("  [vidio-health-trainer] sd2 30D 생체역학 피처 다이어트 및 LOFO 감사 파이프라인")
    print("=" * 110)

    X_30d, y, subjects, views, seq_ids = load_dataset()
    N, D = X_30d.shape
    feature_names = FEATURE_NAMES_30_STIFF_SPINE

    print(f"[*] 총 샘플 수: {N}개 (정상: {np.sum(y==0)}개, 결함: {np.sum(y==1)}개)")
    print(f"[*] 초기 피처 수: {D}개")

    # 0. 베이스라인 전체 30D 성능 측정
    base_f1, base_auc, base_acc = evaluate_features_subset(X_30d, y, groups=subjects)
    print(f"\n[★ 베이스라인 30D 전체 성능] 결함 F1: {base_f1*100:.2f}% | ROC-AUC: {base_auc*100:.2f}% | 정확도: {base_acc*100:.2f}%")

    # ==============================================================================
    # 1. 단독 변별력 및 기본 통계 감사
    # ==============================================================================
    print("\n" + "-" * 110)
    print(f"  {'Index':5s} | {'피처명':34s} | {'분산(Var)':10s} | {'Cohen d':8s} | {'단독 AUC':8s} | {'변별력 판정'}")
    print("-" * 110)

    feature_stats = []
    zero_var_indices = []

    for j, fn in enumerate(feature_names):
        col = X_30d[:, j]
        var = float(np.var(col))
        col_c = col[y == 0]
        col_d = col[y == 1]
        mc, sc = float(np.mean(col_c)), float(np.std(col_c))
        md, sd = float(np.mean(col_d)), float(np.std(col_d))

        pooled_sd = np.sqrt((sc**2 + sd**2) / 2.0) + 1e-7
        cohen_d = float((md - mc) / pooled_sd)

        try:
            auc = float(roc_auc_score(y, col))
            eff_auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
            eff_auc = 0.5

        if var < 1e-5:
            verdict = "❌ 분산 0 (상수 피처)"
            zero_var_indices.append(j)
        elif eff_auc < 0.52 and abs(cohen_d) < 0.10:
            verdict = "⚠️ 무효/노이즈 (변별력 희박)"
        elif eff_auc >= 0.60:
            verdict = "★ 우수 변별력"
        else:
            verdict = "일반 피처"

        print(f"  [{j:2d}]   | {fn:34s} | {var:10.4f} | {cohen_d:8.3f} | {auc:8.3f} | {verdict}")
        feature_stats.append({
            'index': j, 'name': fn, 'var': var, 'cohen_d': cohen_d,
            'auc': auc, 'effective_auc': eff_auc, 'verdict': verdict
        })

    # ==============================================================================
    # 2. 다중공선성(|r| >= 0.85) 고상관 피처 쌍 탐색
    # ==============================================================================
    print("\n" + "-" * 110)
    print("  [다중공선성 감사: 피처 간 피어슨 상관계수 |r| >= 0.85 고상관 쌍]")
    print("-" * 110)

    corr_matrix = np.corrcoef(X_30d, rowvar=False)
    high_corr_pairs = []

    for i in range(D):
        for j in range(i + 1, D):
            r = corr_matrix[i, j]
            if not np.isnan(r) and abs(r) >= 0.85:
                high_corr_pairs.append({
                    'feat_a_idx': i, 'feat_a_name': feature_names[i],
                    'feat_b_idx': j, 'feat_b_name': feature_names[j],
                    'corr': float(r)
                })
                print(f"  - [{i:2d}] {feature_names[i]:32s} <──> [{j:2d}] {feature_names[j]:32s} (r = {r:+.4f})")

    if not high_corr_pairs:
        print("  - |r| >= 0.85 이상의 극단적 고상관 피처 쌍 없음 (양호)")

    # ==============================================================================
    # 3. LOFO (Leave-One-Feature-Out) 전수 감사
    # ==============================================================================
    print("\n" + "-" * 110)
    print(f"  [LOFO 감사: 피처 1개씩 제거 시 5-Fold 결함 F1 및 AUC 변화량 (Drop-One)]")
    print(f"  {'Index':5s} | {'제외된 피처명':34s} | {'F1-Score':10s} | {'Δ F1 (%p)':12s} | {'AUC':8s} | {'기여도 분류'}")
    print("-" * 110)

    lofo_results = []
    negative_impact_indices = []

    for j, fn in enumerate(feature_names):
        # j번째 피처 제외한 서브셋
        sub_indices = [k for k in range(D) if k != j]
        X_sub = X_30d[:, sub_indices]

        f1_sub, auc_sub, acc_sub = evaluate_features_subset(X_sub, y, groups=subjects)
        delta_f1 = (f1_sub - base_f1) * 100.0  # 양수면 제거 시 성능 향상 -> 즉 해당 피처가 저해 요인이었음!
        delta_auc = (auc_sub - base_auc) * 100.0

        if delta_f1 > +0.20:
            impact = "❌ 성능 저해 피처 (제거 시 F1 상승!)"
            negative_impact_indices.append(j)
        elif delta_f1 < -0.20:
            impact = "★ 필수 핵심 피처 (제거 시 성능 급락)"
        else:
            impact = "중립/미세 기여"

        print(f"  [{j:2d}]   | {fn:34s} | {f1_sub*100:9.2f}% | {delta_f1:+11.2f}%p | {auc_sub*100:7.2f}% | {impact}")
        lofo_results.append({
            'index': j, 'name': fn, 'f1_without': float(f1_sub),
            'auc_without': float(auc_sub), 'delta_f1_pct': float(delta_f1),
            'delta_auc_pct': float(delta_auc), 'impact': impact
        })

    # ==============================================================================
    # 4. 정예 피처 다이어트(v2 Diet) 선별 알고리즘
    # ==============================================================================
    print("\n" + "=" * 110)
    print("  [피처 다이어트 v2 정예 피처셋 후보 도출]")
    print("=" * 110)

    # 제거 후보:
    # 1) 분산 0인 상수 피처
    # 2) LOFO에서 delta_f1 > +0.20%p (제거 시 성능 대폭 향상)
    # 3) 고상관 쌍 중 변별력(effective_auc)이 더 낮은 피처
    prune_set = set(zero_var_indices)
    for idx in negative_impact_indices:
        prune_set.add(idx)

    for pair in high_corr_pairs:
        ia = pair['feat_a_idx']
        ib = pair['feat_b_idx']
        auc_a = feature_stats[ia]['effective_auc']
        auc_b = feature_stats[ib]['effective_auc']
        if auc_a >= auc_b:
            prune_set.add(ib)
        else:
            prune_set.add(ia)

    keep_indices = [k for k in range(D) if k not in prune_set]
    print(f"[*] 제거 대상 피처 ({len(prune_set)}개): {[feature_names[i] for i in sorted(list(prune_set))]}")
    print(f"[*] 유지/채택 피처 ({len(keep_indices)}개): {[feature_names[i] for i in keep_indices]}")

    # 다이어트 피처셋 성능 실측
    X_diet = X_30d[:, keep_indices]
    diet_f1, diet_auc, diet_acc = evaluate_features_subset(X_diet, y, groups=subjects)
    print(f"\n[★ v2 Diet 피처셋 ({len(keep_indices)}D) 실측 결과]")
    print(f"  - 결함 F1-Score : {diet_f1*100:.2f}% (베이스라인 30D 대비 {(diet_f1-base_f1)*100:+.2f}%p 도약)")
    print(f"  - ROC-AUC       : {diet_auc*100:.2f}% (베이스라인 대비 {(diet_auc-base_auc)*100:+.2f}%p 도약)")
    print(f"  - 정확도        : {diet_acc*100:.2f}%")

    audit_summary = {
        'baseline_30d': {'f1': float(base_f1), 'auc': float(base_auc), 'acc': float(base_acc)},
        'diet_v2': {
            'num_features': len(keep_indices),
            'indices': keep_indices,
            'names': [feature_names[i] for i in keep_indices],
            'pruned_indices': sorted(list(prune_set)),
            'pruned_names': [feature_names[i] for i in sorted(list(prune_set))],
            'f1': float(diet_f1), 'auc': float(diet_auc), 'acc': float(diet_acc),
            'f1_gain_pct': float((diet_f1 - base_f1) * 100.0)
        },
        'feature_stats': feature_stats,
        'high_corr_pairs': high_corr_pairs,
        'lofo_results': lofo_results
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f_out:
        json.dump(audit_summary, f_out, indent=2, ensure_ascii=False)
    print(f"\n[OK] 피처 감사 결과 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    run_audit()
