# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: d1. 바벨 궤적과 몸 밀착 (sd3_v1)
Script: audit_stiff_d1_feature_interference.py
Purpose: 38D 도메인 피처 간 다중공선성(상호 간섭), 동결 모델 기반 순열 중요도(Frozen Permutation Importance),
         LOFO(Leave-One-Feature-Out) 절제 실험 및 공식 5대 시점 다차원 전수 감사
Rule: AGENTS.md 규정 (피험자 9명 전수 격리, 100% 순수 비전 좌표, 5대 시점 Taxonomy 준수)
"""

import os
import sys
import pickle
import time
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stiff_d1_features import (
    FEATURE_NAMES_38_STIFF_D1, VIEW_ALIAS_MAP
)
from train_and_eval_stiff_d1 import load_dataset
CACHE_PKL_PATH = os.path.join(CURRENT_DIR, "stiff_d1_dataset_cache.pkl")
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_d1_feature_audit_results.json")


def get_cached_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """캐시 pkl이 존재하면 즉시 로드, 없으면 load_dataset 수행"""
    if os.path.exists(CACHE_PKL_PATH):
        with open(CACHE_PKL_PATH, 'rb') as fp:
            data = pickle.load(fp)
        return data['df'], data['X']
    df, X = load_dataset()
    with open(CACHE_PKL_PATH, 'wb') as fp:
        pickle.dump({'df': df, 'X': X}, fp)
    return df, X


def fit_oof_models(X: np.ndarray, y: np.ndarray, subjs: np.ndarray, seed: int = 42) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    5-Fold StratifiedGroupKFold 모델 학습 및 검증 세트 동결(Frozen) 번들 생성
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_bundles = []
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjs)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=20,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=12,
            random_state=seed, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)

        clf_cb = CatBoostClassifier(
            iterations=60, learning_rate=0.06, depth=5, l2_leaf_reg=4.0,
            random_seed=seed, verbose=0, thread_count=1
        )
        clf_cb.fit(X_tr, y_tr)

        p_lgb = clf_lgb.predict_proba(X_va)[:, 1]
        p_cb = clf_cb.predict_proba(X_va)[:, 1]
        p_fold = 0.55 * p_lgb + 0.45 * p_cb
        oof_probs[val_idx] = p_fold

        fold_bundles.append({
            'fold': fold_idx + 1,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'lgb_model': clf_lgb,
            'cb_model': clf_cb,
            'val_probs': p_fold
        })

    return fold_bundles, oof_probs


def evaluate_predictions(oof_probs: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                         tau_clean: float = 0.36, tau_defect: float = 0.50) -> Dict[str, Any]:
    """
    전체 2-State, 3-State 및 AGENTS.md 공식 5대 시점(시점 A~E) 다차원 성능 감사 산출
    """
    pred_2s = (oof_probs >= 0.50).astype(int)
    f1_2s = f1_score(y, pred_2s, zero_division=0)
    acc_2s = accuracy_score(y, pred_2s)
    auc_2s = roc_auc_score(y, oof_probs)

    # 3-State Selective Gating
    decided_mask = (oof_probs <= tau_clean) | (oof_probs >= tau_defect)
    coverage = float(np.mean(decided_mask))
    pred_3s = (oof_probs[decided_mask] >= 0.50).astype(int)
    y_3s = y[decided_mask]
    f1_3s = f1_score(y_3s, pred_3s, zero_division=0)
    acc_3s = accuracy_score(y_3s, pred_3s)

    # 공식 5대 시점 전수 감사
    views = df['view'].values
    view_metrics = {}
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if not np.any(v_mask):
            continue
        v_probs = oof_probs[v_mask]
        v_y = y[v_mask]
        v_decided = (v_probs <= tau_clean) | (v_probs >= tau_defect)
        v_cov = float(np.mean(v_decided)) if len(v_decided) > 0 else 0.0

        if np.any(v_decided):
            v_pred_3s = (v_probs[v_decided] >= 0.50).astype(int)
            v_y_3s = v_y[v_decided]
            v_f1_3s = float(f1_score(v_y_3s, v_pred_3s, zero_division=0))
            v_acc_3s = float(accuracy_score(v_y_3s, v_pred_3s))
        else:
            v_f1_3s, v_acc_3s = 0.0, 0.0

        try:
            v_auc = float(roc_auc_score(v_y, v_probs))
        except Exception:
            v_auc = 0.5

        view_metrics[v_code] = {
            'view_alias': v_alias,
            'f1_3s': round(v_f1_3s * 100, 2),
            'acc_3s': round(v_acc_3s * 100, 2),
            'auc': round(v_auc * 100, 2),
            'coverage': round(v_cov * 100, 1)
        }

    return {
        'f1_2s': float(f1_2s),
        'acc_2s': float(acc_2s),
        'auc': float(auc_2s),
        'f1_3s': float(f1_3s),
        'acc_3s': float(acc_3s),
        'coverage': float(coverage),
        'view_metrics': view_metrics
    }


def analyze_correlations_and_collinearity(X: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """피처 간 상관계수 행렬 및 다중공선성 고상관 쌍(|r| >= 0.85) 추출"""
    df_feat = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_feat.corr().abs()

    high_corr_pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            r_val = float(corr_matrix.iloc[i, j])
            if r_val >= 0.85:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'idx_1': i,
                    'idx_2': j,
                    'abs_correlation': round(r_val, 4)
                })

    high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    return corr_matrix, high_corr_pairs


def run_frozen_permutation_importance(fold_bundles: List[Dict[str, Any]], X: np.ndarray, y: np.ndarray,
                                      df: pd.DataFrame, feature_names: List[str], base_eval: Dict[str, Any],
                                      n_repeats: int = 5, seed: int = 42) -> List[Dict[str, Any]]:
    """
    동결(Frozen) 5-Fold 모델을 기반으로 한 순열 중요도(Permutation Importance) 전수 산출
    (학습 완료된 고정 모델에 대해 검증 세트 피처만 셔플하여 예측 오차 증가량 측정)
    """
    np.random.seed(seed)
    base_f1_3s = base_eval['f1_3s']
    base_auc = base_eval['auc']

    perm_results = []
    print(f"\n[*] 38개 피처 동결 순열 중요도(Frozen Permutation Importance) 분석 시작 (n_repeats={n_repeats})...")

    for f_idx, fname in enumerate(feature_names):
        drop_f1_runs = []
        drop_auc_runs = []

        for r in range(n_repeats):
            shuffled_oof_probs = np.zeros(len(y), dtype=np.float32)

            for bundle in fold_bundles:
                val_idx = bundle['val_idx']
                X_val = X[val_idx].copy()

                # 해당 피처 컬럼만 무작위 셔플
                perm_idx = np.random.permutation(len(val_idx))
                X_val[:, f_idx] = X_val[perm_idx, f_idx]

                clf_lgb = bundle['lgb_model']
                clf_cb = bundle['cb_model']
                p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
                p_cb = clf_cb.predict_proba(X_val)[:, 1]
                shuffled_oof_probs[val_idx] = 0.55 * p_lgb + 0.45 * p_cb

            res_shuffled = evaluate_predictions(shuffled_oof_probs, y, df)
            # Permutation Importance: 피처를 망가뜨렸을 때 성능 하락 폭 (Base - Shuffled)
            # 양수(+): 해당 피처가 모델 예측에 필수적으로 기여함
            # 음수(-): 해당 피처를 망가뜨렸는데 오히려 성능이 오름 (명백한 성능 저해/간섭 피처)
            drop_f1_runs.append(base_f1_3s - res_shuffled['f1_3s'])
            drop_auc_runs.append(base_auc - res_shuffled['auc'])

        mean_drop_f1 = float(np.mean(drop_f1_runs))
        mean_drop_auc = float(np.mean(drop_auc_runs))

        # 정밀 판정 로직:
        # F1 또는 AUC 중 하나라도 0.3%p 이상 하락 시 유익 기여
        # 둘 다 -0.3%p 미만(성능 상승) 시 성능 저해 의심
        if mean_drop_f1 < -0.003 and mean_drop_auc < -0.003:
            status = "HARMFUL (성능 저해 의심)"
        elif mean_drop_f1 < -0.005 or mean_drop_auc < -0.005:
            status = "HARMFUL (성능 저해 의심)"
        elif abs(mean_drop_f1) <= 0.003 and abs(mean_drop_auc) <= 0.003:
            status = "NEUTRAL (경계/중립)"
        else:
            status = "BENEFICIAL (유익 기여)"

        perm_results.append({
            'feature_idx': f_idx,
            'feature_name': fname,
            'f1_3s_drop_p': round(mean_drop_f1 * 100, 3), # %p 단위
            'auc_drop_p': round(mean_drop_auc * 100, 3),
            'status': status
        })

        if (f_idx + 1) % 5 == 0 or (f_idx + 1) == len(feature_names):
            print(f"  [{f_idx + 1:2d}/{len(feature_names)}] {fname:42s} | F1_3S Drop: {mean_drop_f1*100:+.2f}%p | AUC Drop: {mean_drop_auc*100:+.2f}%p | {status}", flush=True)

    perm_results.sort(key=lambda x: x['f1_3s_drop_p'], reverse=True)
    return perm_results


def run_lofo_ablation_experiment(X: np.ndarray, y: np.ndarray, subjs: np.ndarray, df: pd.DataFrame,
                                 feature_names: List[str], base_eval: Dict[str, Any],
                                 candidate_indices: List[int]) -> List[Dict[str, Any]]:
    """
    후보 피처 개별 완전 배제(Leave-One-Feature-Out) 절제 실험 및 5대 시점 급락(Regression) 감사
    """
    print(f"\n[*] LOFO(Leave-One-Feature-Out) 모델 재학습 절제 실험 시작 (대상 후보: {len(candidate_indices)}개)...")
    base_f1_3s = base_eval['f1_3s']
    base_auc = base_eval['auc']
    base_cov = base_eval['coverage']

    lofo_results = []
    all_indices = set(range(len(feature_names)))

    for idx in candidate_indices:
        fname = feature_names[idx]
        kept_indices = sorted(list(all_indices - {idx}))
        X_sub = X[:, kept_indices]

        # 해당 피처를 완전히 제거하고 5-Fold 재학습
        _, oof_probs_lofo = fit_oof_models(X_sub, y, subjs, seed=42)
        res_lofo = evaluate_predictions(oof_probs_lofo, y, df)

        diff_f1_3s_p = (res_lofo['f1_3s'] - base_f1_3s) * 100
        diff_auc_p = (res_lofo['auc'] - base_auc) * 100
        cov_p = res_lofo['coverage'] * 100

        # 시점별 급락 검사 (특정 시점에서 1.5%p 이상 급락 시 RETAIN 강제)
        view_regressions = []
        for v_code, v_base in base_eval['view_metrics'].items():
            v_lofo = res_lofo['view_metrics'].get(v_code, {})
            v_drop = v_base['f1_3s'] - v_lofo.get('f1_3s', 0.0)
            if v_drop >= 1.5:
                view_regressions.append(f"{v_base['view_alias']} (-{v_drop:.1f}%p)")

        # 결정 규칙:
        # F1이 +0.3%p 이상 오르고, AUC가 훼손되지 않으며, 커버리지가 유지되고, 시점별 급락이 없을 때만 제거 추천
        if diff_f1_3s_p >= 0.3 and diff_auc_p >= -0.1 and (res_lofo['coverage'] >= base_cov - 0.015) and not view_regressions:
            decision = "REMOVE_RECOMMENDED"
        elif diff_f1_3s_p > 0.0 and not view_regressions:
            decision = "MARGINAL_IMPROVE"
        else:
            decision = "RETAIN (유지 필수)"

        lofo_results.append({
            'dropped_idx': idx,
            'dropped_feature': fname,
            'f1_3s_after': round(res_lofo['f1_3s'] * 100, 2),
            'auc_after': round(res_lofo['auc'] * 100, 2),
            'diff_f1_3s_p': round(diff_f1_3s_p, 3),
            'diff_auc_p': round(diff_auc_p, 3),
            'coverage': round(cov_p, 1),
            'view_regressions': view_regressions,
            'decision': decision
        })
        print(f"  [Drop #{idx:02d}] {fname:40s} -> F1_3S: {res_lofo['f1_3s']*100:.2f}% ({diff_f1_3s_p:+.2f}%p) | AUC: {res_lofo['auc']*100:.2f}% ({diff_auc_p:+.2f}%p) | {decision}", flush=True)

    return lofo_results


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 스티프 데드리프트 d1 (sd3_v1) 38D 피처 간섭 및 성능 저해 전수 감사")
    print("=" * 80)

    # 1. 1,035개 데이터셋 및 38D 피처 로드 (캐시 pkl 활용)
    df, X = get_cached_dataset()
    y = df['y'].values
    subjs = df['subj'].values
    feature_names = FEATURE_NAMES_38_STIFF_D1

    # 2. 38D 전체 5-Fold 베이스라인 모델 학습 및 동결
    print("\n[*] 38D 전체 피처셋 5-Fold StratifiedGroupKFold 베이스라인 모델 학습 및 검증 동결 중...")
    t0 = time.time()
    fold_bundles, base_oof_probs = fit_oof_models(X, y, subjs, seed=42)
    base_eval = evaluate_predictions(base_oof_probs, y, df)
    print(f"[+] 38D 베이스라인 결과 (소요 시간: {time.time()-t0:.1f}초):")
    print(f"    - 2-State F1: {base_eval['f1_2s']*100:.2f}% | 정확도: {base_eval['acc_2s']*100:.2f}% | AUC: {base_eval['auc']*100:.2f}%")
    print(f"    - 3-State F1: {base_eval['f1_3s']*100:.2f}% | 정확도: {base_eval['acc_3s']*100:.2f}% | 커버리지: {base_eval['coverage']*100:.2f}%")
    print("\n    [공식 5대 시점별 베이스라인 F1/AUC/Coverage]")
    for v_code, vm in base_eval['view_metrics'].items():
        print(f"      * {vm['view_alias']:26s}: F1_3S={vm['f1_3s']:.2f}% | AUC={vm['auc']:.2f}% | Cov={vm['coverage']:.1f}%")

    # 3. 상관계수 및 다중공선성 분석
    print("\n[*] 피처 간 상관계수 행렬 및 다중공선성(|r| >= 0.85) 쌍 탐색...")
    corr_matrix, high_corr_pairs = analyze_correlations_and_collinearity(X, feature_names)
    print(f"[+] |r| >= 0.85 초고상관 피처 쌍 총 {len(high_corr_pairs)}개 검출:")
    for p in high_corr_pairs[:10]:
        print(f"    - [{p['abs_correlation']:.3f}] #{p['idx_1']:02d} {p['feature_1']} <--> #{p['idx_2']:02d} {p['feature_2']}")

    # 4. 순열 중요도 (Frozen Permutation Importance) 전수 조사
    perm_results = run_frozen_permutation_importance(fold_bundles, X, y, df, feature_names, base_eval, n_repeats=2, seed=42)

    # 5. 성능 저해 후보 피처 탐색 (f1_3s_drop_p <= 0.0)
    harmful_candidates = [p for p in perm_results if p['f1_3s_drop_p'] <= 0.0]
    print(f"\n[+] F1_3S 순열 중요도 <= 0.0 인 피처 (제거 시 성능 개선 가능성 후보): 총 {len(harmful_candidates)}개")
    for hc in harmful_candidates:
        print(f"    - #{hc['feature_idx']:02d} {hc['feature_name']}: F1 Drop={hc['f1_3s_drop_p']:+.3f}%p, AUC Drop={hc['auc_drop_p']:+.3f}%p ({hc['status']})")

    # 6. LOFO (Leave-One-Feature-Out) 절제 실험
    candidate_indices = [hc['feature_idx'] for hc in harmful_candidates]
    if not candidate_indices:
        # 하위 5개 피처를 점검
        candidate_indices = [p['feature_idx'] for p in perm_results[-5:]]

    lofo_results = run_lofo_ablation_experiment(X, y, subjs, df, feature_names, base_eval, candidate_indices)

    # 7. 제거 권장 피처 동시 제거 Pruning 실험
    prune_candidates = [r['dropped_idx'] for r in lofo_results if r['decision'] in ['REMOVE_RECOMMENDED', 'MARGINAL_IMPROVE']]
    pruning_eval = None
    if prune_candidates:
        print(f"\n[*] 성능 개선 후보 피처 {len(prune_candidates)}개 동시 제거 후 종합 검증...")
        kept_pruned = sorted(list(set(range(len(feature_names))) - set(prune_candidates)))
        X_pruned = X[:, kept_pruned]
        _, oof_probs_pruned = fit_oof_models(X_pruned, y, subjs, seed=42)
        pruning_eval = evaluate_predictions(oof_probs_pruned, y, df)
        print(f"[+] Pruned ({len(kept_pruned)}D) 결과:")
        print(f"    - 3-State F1: {pruning_eval['f1_3s']*100:.2f}% (원본 대비 {(pruning_eval['f1_3s'] - base_eval['f1_3s'])*100:+.2f}%p)")
        print(f"    - AUC: {pruning_eval['auc']*100:.2f}% (원본 대비 {(pruning_eval['auc'] - base_eval['auc'])*100:+.2f}%p)")
        print(f"    - Coverage: {pruning_eval['coverage']*100:.2f}%")
        print("\n    [Pruned 모델 공식 5대 시점별 F1/AUC/Coverage]")
        for v_code, vm in pruning_eval['view_metrics'].items():
            base_vm = base_eval['view_metrics'][v_code]
            f1_diff = vm['f1_3s'] - base_vm['f1_3s']
            print(f"      * {vm['view_alias']:26s}: F1_3S={vm['f1_3s']:.2f}% ({f1_diff:+.2f}%p) | AUC={vm['auc']:.2f}% | Cov={vm['coverage']:.1f}%")

    # 8. 최종 결과 직렬화
    audit_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(y),
        'base_eval': base_eval,
        'high_corr_pairs_count': len(high_corr_pairs),
        'high_corr_pairs': high_corr_pairs,
        'permutation_importance': perm_results,
        'harmful_candidates_count': len(harmful_candidates),
        'lofo_results': lofo_results,
        'pruning_eval': pruning_eval,
        'prune_candidate_indices': prune_candidates,
        'prune_candidate_names': [feature_names[i] for i in prune_candidates]
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(audit_summary, fp, ensure_ascii=False, indent=2)

    print(f"\n[+] 전수 감사 결과 JSON 저장 완료: {OUTPUT_JSON_PATH}")


if __name__ == '__main__':
    main()
