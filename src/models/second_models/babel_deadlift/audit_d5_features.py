# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: audit_d5_features.py
Purpose: d5 v5.0 모델에 투입되는 40D 피처 전수 감사 (Feature Audit)
         1. 피처 간 상호 상관관계 분석 (|r| >= 0.85 고상관 중복 피처 쌍 식별)
         2. CatBoost + LightGBM + Random Forest 피처 중요도(Feature Importance) 전수 산출
         3. 5-Fold 피험자 격리 순열 중요도 (Permutation Feature Importance, PFI) 분석
         4. 성능 저해/노이즈 후보 피처(Negative/Zero PFI) 식별 및 소거(Ablation) 영향도 실측
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from sklearn.ensemble import RandomForestClassifier

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d5_model_v4_gated import (
    FEATURE_NAMES_40_D5_GATED
)
from src.models.second_models.babel_deadlift.d5_model_v5_calibrated import (
    VIEW_OPTIMAL_THRESHOLDS
)

FEATURE_CACHE_FILE = os.path.join(CURRENT_DIR, "d5_features_40_gated_cache.npz")
OUT_AUDIT_JSON = os.path.join(CURRENT_DIR, "d5_feature_audit_results.json")


def load_cached_data():
    data = np.load(FEATURE_CACHE_FILE, allow_pickle=True)
    X = data['X']
    y = data['y']
    groups = data['groups']
    views = data['views']
    clip_names = list(data['clip_names'])
    return X, y, groups, views, clip_names


def audit_correlations(X: np.ndarray, feature_names: List[str], threshold: float = 0.85):
    """피처 간 상관계수 분석 (|r| >= threshold)"""
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr(method='pearson')

    high_corr_pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                high_corr_pairs.append({
                    "feat1_idx": i,
                    "feat1_name": feature_names[i],
                    "feat2_idx": j,
                    "feat2_name": feature_names[j],
                    "correlation": float(val)
                })

    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return corr_matrix, high_corr_pairs


def compute_tree_feature_importances(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """CatBoost, LightGBM, Random Forest 모델의 피처 중요도 전수 산출"""
    # 1. CatBoost
    cb = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.04, auto_class_weights='Balanced', verbose=False, random_seed=42, thread_count=2)
    cb.fit(X, y)
    cb_imp = cb.get_feature_importance()

    # 2. LightGBM
    pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))
    ds = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'learning_rate': 0.04, 'num_leaves': 20, 'max_depth': 5, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42, 'n_jobs': 2}
    lgb_m = lgb.train(params, ds, num_boost_round=180)
    lgb_imp = lgb_m.feature_importance(importance_type='gain')

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=180, max_depth=6, class_weight='balanced', random_state=42, n_jobs=2)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # 정규화 (합이 100이 되도록)
    cb_norm = (cb_imp / (np.sum(cb_imp) + 1e-9)) * 100.0
    lgb_norm = (lgb_imp / (np.sum(lgb_imp) + 1e-9)) * 100.0
    rf_norm = (rf_imp / (np.sum(rf_imp) + 1e-9)) * 100.0

    ensemble_imp = 0.50 * lgb_norm + 0.30 * cb_norm + 0.20 * rf_norm

    records = []
    for i, name in enumerate(feature_names):
        records.append({
            "index": i,
            "name": name,
            "lightgbm_imp": float(lgb_norm[i]),
            "catboost_imp": float(cb_norm[i]),
            "rf_imp": float(rf_norm[i]),
            "ensemble_imp": float(ensemble_imp[i])
        })

    records.sort(key=lambda x: x["ensemble_imp"], reverse=True)
    return records


def compute_permutation_importance(X: np.ndarray, y: np.ndarray, groups: np.ndarray, views: np.ndarray, feature_names: List[str]):
    """5-Fold 피험자 완전 격리 순열 중요도 (Permutation Feature Importance, PFI)"""
    sgkf = StratifiedGroupKFold(n_splits=5)

    # 5개 Fold 학습 모델 캐싱
    fold_models = []
    oof_raw_probs = np.zeros(len(y), dtype=np.float32)
    val_indices_list = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va = X[va_idx]

        m_lgb = lgb.LGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.04, num_leaves=20, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=2)
        m_cb = CatBoostClassifier(iterations=180, depth=5, learning_rate=0.04, random_seed=42, verbose=0, thread_count=2)
        m_rf = RandomForestClassifier(n_estimators=180, max_depth=6, random_state=42, n_jobs=2)

        m_lgb.fit(X_tr, y_tr)
        m_cb.fit(X_tr, y_tr)
        m_rf.fit(X_tr, y_tr)

        p = 0.50 * m_lgb.predict_proba(X_va)[:, 1] + 0.30 * m_cb.predict_proba(X_va)[:, 1] + 0.20 * m_rf.predict_proba(X_va)[:, 1]
        oof_raw_probs[va_idx] = p
        fold_models.append((m_lgb, m_cb, m_rf))
        val_indices_list.append(va_idx)

    # 베이스라인 시점별 캘리브레이션 예측
    def calc_calibrated_f1_auc(probs_raw):
        preds = np.zeros(len(y), dtype=int)
        for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
            mask_v = (views == vn)
            th = VIEW_OPTIMAL_THRESHOLDS[vn]
            preds[mask_v] = (probs_raw[mask_v] >= th).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs_raw)
        return f1, auc

    base_f1, base_auc = calc_calibrated_f1_auc(oof_raw_probs)
    print(f"[*] 감사 베이스라인 OOF 성능: F1={base_f1*100:.2f}%, AUC={base_auc:.4f}", flush=True)

    pfi_results = []
    np.random.seed(42)

    for i, name in enumerate(feature_names):
        oof_shuff_raw = np.zeros(len(y), dtype=np.float32)

        for fold, va_idx in enumerate(val_indices_list):
            m_lgb, m_cb, m_rf = fold_models[fold]
            X_va_shuff = X[va_idx].copy()
            # 해당 피처 컬럼만 랜덤 셔플
            X_va_shuff[:, i] = np.random.permutation(X_va_shuff[:, i])

            p_shuff = 0.50 * m_lgb.predict_proba(X_va_shuff)[:, 1] + 0.30 * m_cb.predict_proba(X_va_shuff)[:, 1] + 0.20 * m_rf.predict_proba(X_va_shuff)[:, 1]
            oof_shuff_raw[va_idx] = p_shuff

        shuff_f1, shuff_auc = calc_calibrated_f1_auc(oof_shuff_raw)
        # PFI: 베이스라인 - 셔플후 (양수이면 중요, 음수이면 셔플했을 때 오히려 성능이 오르는 저해 피처!)
        delta_f1 = float(base_f1 - shuff_f1)
        delta_auc = float(base_auc - shuff_auc)

        pfi_results.append({
            "index": i,
            "name": name,
            "delta_f1": delta_f1,
            "delta_auc": delta_auc,
            "shuffled_f1": float(shuff_f1),
            "shuffled_auc": float(shuff_auc),
            "is_harmful": delta_f1 < -0.001 or delta_auc < -0.002
        })

    # PFI delta_f1 내림차순 정렬
    pfi_results.sort(key=lambda x: x["delta_f1"], reverse=True)
    return pfi_results, base_f1, base_auc


def run_ablation_experiment(X: np.ndarray, y: np.ndarray, groups: np.ndarray, views: np.ndarray, 
                           feature_names: List[str], remove_indices: List[int], desc: str):
    """지정된 피처를 완전히 제거했을 때 5-Fold OOF 성능 실측"""
    keep_indices = [i for i in range(X.shape[1]) if i not in remove_indices]
    X_sub = X[:, keep_indices]

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_raw_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sub, y, groups=groups)):
        X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
        X_va = X_sub[va_idx]

        m_lgb = lgb.LGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.04, num_leaves=20, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=2)
        m_cb = CatBoostClassifier(iterations=180, depth=5, learning_rate=0.04, random_seed=42, verbose=0, thread_count=2)
        m_rf = RandomForestClassifier(n_estimators=180, max_depth=6, random_state=42, n_jobs=2)

        m_lgb.fit(X_tr, y_tr)
        m_cb.fit(X_tr, y_tr)
        m_rf.fit(X_tr, y_tr)

        p = 0.50 * m_lgb.predict_proba(X_va)[:, 1] + 0.30 * m_cb.predict_proba(X_va)[:, 1] + 0.20 * m_rf.predict_proba(X_va)[:, 1]
        oof_raw_probs[va_idx] = p

    preds = np.zeros(len(y), dtype=int)
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        mask_v = (views == vn)
        th = VIEW_OPTIMAL_THRESHOLDS[vn]
        preds[mask_v] = (oof_raw_probs[mask_v] >= th).astype(int)

    f1 = float(f1_score(y, preds, zero_division=0))
    acc = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, oof_raw_probs))

    # 5대 시점별 F1
    view_f1s = {}
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        mask_v = (views == vn)
        view_f1s[vn] = float(f1_score(y[mask_v], preds[mask_v], zero_division=0))

    return {
        "description": desc,
        "kept_feature_count": len(keep_indices),
        "removed_feature_count": len(remove_indices),
        "removed_indices": remove_indices,
        "removed_names": [feature_names[i] for i in remove_indices],
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": auc,
        "view_f1s": view_f1s
    }


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 40D 피처 전수 감사 (Feature Audit)")
    print("=" * 80)

    X, y, groups, views, clip_names = load_cached_data()
    feature_names = FEATURE_NAMES_40_D5_GATED
    print(f"[*] 데이터: {len(X)}개 시퀀스, {len(feature_names)}개 피처")

    # 1. 상관관계 감사
    print("\n--- 1. 피처 간 상호 상관관계 분석 (|r| >= 0.85) ---")
    corr_mat, high_corr_pairs = audit_correlations(X, feature_names, threshold=0.85)
    print(f"[*] 고상관 중복 피처 쌍 검출: 총 {len(high_corr_pairs)}개 쌍")
    for p in high_corr_pairs:
        print(f"  [{p['feat1_idx']:2d}] {p['feat1_name']:<34} <-> [{p['feat2_idx']:2d}] {p['feat2_name']:<34} | r = {p['correlation']:+.4f}")

    # 2. 트리 기반 중요도 산출
    print("\n--- 2. 트리 앙상블 피처 중요도 전수 산출 ---")
    tree_imp_records = compute_tree_feature_importances(X, y, feature_names)
    print(" 상위 10개 핵심 피처:")
    for r in tree_imp_records[:10]:
        print(f"  [{r['index']:2d}] {r['name']:<36} | Imp: {r['ensemble_imp']:5.2f}% (LGB: {r['lightgbm_imp']:4.1f}%, CB: {r['catboost_imp']:4.1f}%, RF: {r['rf_imp']:4.1f}%)")

    print("\n 하위 10개 최하위 피처:")
    for r in tree_imp_records[-10:]:
        print(f"  [{r['index']:2d}] {r['name']:<36} | Imp: {r['ensemble_imp']:5.2f}%")

    # 3. 5-Fold 피험자 격리 순열 중요도 (PFI) 산출
    print("\n--- 3. 5-Fold 피험자 격리 순열 중요도 (PFI) 분석 ---")
    pfi_records, base_f1, base_auc = compute_permutation_importance(X, y, groups, views, feature_names)

    print(f"[*] 베이스라인 40D F1: {base_f1*100:.2f}% | AUC: {base_auc:.4f}")
    print("\n [★ 순열 중요도 상위 10개 필수 피처 (PFI Top 10)]")
    for r in pfi_records[:10]:
        print(f"  [{r['index']:2d}] {r['name']:<36} | dF1: {r['delta_f1']*100:+5.2f}%p | dAUC: {r['delta_auc']:+7.4f}")

    harmful_features = [r for r in pfi_records if r["delta_f1"] < 0 or r["is_harmful"]]
    harmful_features.sort(key=lambda x: x["delta_f1"])

    print(f"\n [⚠️ 성능 저해 / 노이즈 후보 피처 (PFI 음수 / Negative Impact): {len(harmful_features)}개]")
    for r in harmful_features:
        print(f"  [{r['index']:2d}] {r['name']:<36} | dF1: {r['delta_f1']*100:+5.2f}%p | dAUC: {r['delta_auc']:+7.4f} (셔플 시 F1: {r['shuffled_f1']*100:.2f}%)")

    # 4. 단계별 소거(Ablation) 영향도 실측
    print("\n--- 4. 단계별 저해/중복 피처 소거(Ablation) 실측 실험 ---")
    harmful_indices = [r["index"] for r in harmful_features]

    # 실험 1: 베이스라인 (40D)
    exp0 = run_ablation_experiment(X, y, groups, views, feature_names, [], "베이스라인 (40D 전체 피처)")
    print(f" [Exp 0: 베이스라인 40D] F1: {exp0['f1_score']*100:.2f}% | Acc: {exp0['accuracy']*100:.2f}% | AUC: {exp0['roc_auc']:.4f}")

    # 실험 2: 명백한 저해 피처(delta_f1 < -0.002) 1차 소거
    severe_harmful = [r["index"] for r in harmful_features if r["delta_f1"] <= -0.002]
    exp1 = run_ablation_experiment(X, y, groups, views, feature_names, severe_harmful, f"1차 저해 피처 소거 ({len(severe_harmful)}개 제거)")
    print(f" [Exp 1: 1차 저해 소거 ({40-len(severe_harmful)}D)] F1: {exp1['f1_score']*100:.2f}% ({exp1['f1_score']-exp0['f1_score']:+.4f}) | Acc: {exp1['accuracy']*100:.2f}% | AUC: {exp1['roc_auc']:.4f}")

    # 실험 3: 모든 음수 PFI 저해 피처(delta_f1 < 0) 전수 소거
    all_harmful = [r["index"] for r in harmful_features if r["delta_f1"] < 0]
    exp2 = run_ablation_experiment(X, y, groups, views, feature_names, all_harmful, f"전체 음수 PFI 피처 전수 소거 ({len(all_harmful)}개 제거)")
    print(f" [Exp 2: 음수 PFI 전수 소거 ({40-len(all_harmful)}D)] F1: {exp2['f1_score']*100:.2f}% ({exp2['f1_score']-exp0['f1_score']:+.4f}) | Acc: {exp2['accuracy']*100:.2f}% | AUC: {exp2['roc_auc']:.4f}")

    # 실험 4: 고상관 중복 피처 중 중요도가 낮은 피처 추가 정리
    # |r| >= 0.85 쌍에서 ensemble_imp가 낮은 쪽 식별
    redundant_to_remove = set()
    imp_dict = {r["index"]: r["ensemble_imp"] for r in tree_imp_records}
    for p in high_corr_pairs:
        idx1, idx2 = p["feat1_idx"], p["feat2_idx"]
        if imp_dict[idx1] >= imp_dict[idx2]:
            redundant_to_remove.add(idx2)
        else:
            redundant_to_remove.add(idx1)

    combined_remove = sorted(list(set(severe_harmful).union(redundant_to_remove)))
    exp3 = run_ablation_experiment(X, y, groups, views, feature_names, combined_remove, f"저해 피처 + 고상관 중복 동시 소거 ({len(combined_remove)}개 제거)")
    print(f" [Exp 3: 최적 다이어트 ({40-len(combined_remove)}D)] F1: {exp3['f1_score']*100:.2f}% ({exp3['f1_score']-exp0['f1_score']:+.4f}) | Acc: {exp3['accuracy']*100:.2f}% | AUC: {exp3['roc_auc']:.4f}")

    # 실험 5: 최적 피처셋 탐색 (Greedy Forward / Backward 검증)
    best_exp = max([exp0, exp1, exp2, exp3], key=lambda x: x["f1_score"])
    print(f"\n[★ 최적 소거 실험 결과] {best_exp['description']}")
    print(f"   -> F1: {best_exp['f1_score']*100:.2f}% (베이스라인 대비 {best_exp['f1_score']-exp0['f1_score']:+.4f})")
    print(f"   -> 정확도: {best_exp['accuracy']*100:.2f}%")
    print(f"   -> ROC-AUC: {best_exp['roc_auc']:.4f}")
    print(f"   -> 유지 피처 수: {best_exp['kept_feature_count']}D (소거: {best_exp['removed_feature_count']}D)")

    out_data = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(y),
        "total_features": len(feature_names),
        "baseline_metrics": {
            "f1_score": float(base_f1),
            "roc_auc": float(base_auc)
        },
        "high_correlation_pairs": high_corr_pairs,
        "tree_importances": tree_imp_records,
        "permutation_importances": pfi_records,
        "harmful_features": harmful_features,
        "ablation_experiments": [exp0, exp1, exp2, exp3],
        "optimal_diet_result": best_exp
    }

    with open(OUT_AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 감사 결과 JSON 저장 완료: {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    main()
