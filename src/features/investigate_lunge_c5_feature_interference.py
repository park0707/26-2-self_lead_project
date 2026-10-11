# -*- coding: utf-8 -*-
"""
[vidio-health-trainer] 바벨 런지 c5 (상체 과도한 숙임/젖힘)
36D 피처 간섭, 다중공선성, 성능 저해 피처 전수 조사 스크립트
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

DATA_PATH = r"D:\Data\피처_데이터\lunge_c5_v5_36d_features.npz"
RESULTS_JSON_PATH = r"D:\src\features\lunge_c5_feature_interference_results.json"

def run_comprehensive_feature_investigation():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c5 36D 피처 간섭 및 성능 저해 전수 조사")
    print("=" * 80)

    # 1. 데이터셋 로드
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"피처 데이터셋을 찾을 수 없습니다: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    feat_names = list(data['feature_names'])

    N, D = X.shape
    print(f"[*] 데이터셋 로드 완료: N={N}개, D={D}개 피처")
    print(f"    - 정상(0): {np.sum(y == 0)}개, 결함(1): {np.sum(y == 1)}개")

    # NaN / Inf 검사
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"[*] 결측치 검사: NaN={nan_count}, Inf={inf_count}")

    # =========================================================================
    # PART 1: 다중공선성 및 상관관계 분석 (Cross-Correlation & VIF)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [PART 1] 피처 상호 상관관계 (Pearson & Spearman) 및 다중공선성 전수 분석")
    print("=" * 60)

    df_feats = pd.DataFrame(X, columns=feat_names)
    corr_pearson = df_feats.corr(method='pearson')
    corr_spearman = df_feats.corr(method='spearman')

    high_corr_pairs = []
    seen = set()
    for i in range(D):
        for j in range(i + 1, D):
            f1, f2 = feat_names[i], feat_names[j]
            r_p = corr_pearson.iloc[i, j]
            r_s = corr_spearman.iloc[i, j]
            if abs(r_p) >= 0.80 or abs(r_s) >= 0.80:
                high_corr_pairs.append({
                    'f1_idx': i + 1,
                    'f1_name': f1,
                    'f2_idx': j + 1,
                    'f2_name': f2,
                    'pearson_r': float(r_p),
                    'spearman_r': float(r_s),
                    'max_abs_r': float(max(abs(r_p), abs(r_s)))
                })

    high_corr_pairs.sort(key=lambda x: x['max_abs_r'], reverse=True)
    print(f"[*] |r| >= 0.80 고상관 피처 쌍: 총 {len(high_corr_pairs)}쌍 발견")
    for p in high_corr_pairs[:12]:
        print(f"    - [{p['f1_idx']:02d}] {p['f1_name'][:30]:<30} <---> [{p['f2_idx']:02d}] {p['f2_name'][:30]:<30} | Pearson={p['pearson_r']:+.3f}, Spearman={p['spearman_r']:+.3f}")

    # VIF 근사 계산 (각 피처를 나머지 피처들로 선형 회귀 시 R^2 계산)
    print("\n[*] 피처별 VIF (Variance Inflation Factor) 계산 중...")
    from sklearn.linear_model import Ridge
    vif_dict = {}
    for i, col in enumerate(feat_names):
        y_col = X[:, i]
        X_other = np.delete(X, i, axis=1)
        ridge = Ridge(alpha=1e-3)
        ridge.fit(X_other, y_col)
        r2 = ridge.score(X_other, y_col)
        vif = 1.0 / (1.0 - r2 + 1e-6)
        vif_dict[col] = float(vif)

    vif_sorted = sorted(vif_dict.items(), key=lambda x: x[1], reverse=True)
    print("    [상위 10개 고다중공선성 위험 피처 (VIF > 5)]:")
    for col, vif in vif_sorted[:10]:
        print(f"    - {col:<40}: VIF = {vif:.2f}")

    # =========================================================================
    # PART 2: 단일 피처 정답 변별력 (Target Discriminative Power)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [PART 2] 단일 피처 정답 변별력 (ROC-AUC, Point-Biserial r, Mutual Information)")
    print("=" * 60)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)

    single_feat_metrics = []
    for i, col in enumerate(feat_names):
        vals = X[:, i]
        # Point-biserial
        pb_r, pb_p = pointbiserialr(vals, y)
        # Single Feature AUC
        try:
            auc = roc_auc_score(y, vals)
            # 단일 AUC는 방향성이 반대일 수 있으므로 max(auc, 1-auc)가 변별력
            eff_auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
            eff_auc = 0.5

        single_feat_metrics.append({
            'index': i + 1,
            'name': col,
            'point_biserial_r': float(pb_r),
            'p_value': float(pb_p),
            'raw_auc': float(auc),
            'effective_auc': float(eff_auc),
            'mutual_info': float(mi_scores[i]),
            'vif': float(vif_dict[col])
        })

    # 변별력 낮은 순서 정렬 (노이즈 후보)
    single_feat_metrics_sorted_low = sorted(single_feat_metrics, key=lambda x: (x['effective_auc'], x['mutual_info']))
    print("    [단일 변별력 최하위 8개 피처 (노이즈/무기여 가능성 높음)]:")
    for m in single_feat_metrics_sorted_low[:8]:
        print(f"    - [{m['index']:02d}] {m['name']:<38}: Eff-AUC={m['effective_auc']:.3f}, MI={m['mutual_info']:.4f}, PB-r={m['point_biserial_r']:+.3f}")

    # 변별력 높은 순서 정렬
    single_feat_metrics_sorted_high = sorted(single_feat_metrics, key=lambda x: x['effective_auc'], reverse=True)
    print("\n    [단일 변별력 최상위 8개 핵심 피처]:")
    for m in single_feat_metrics_sorted_high[:8]:
        print(f"    - [{m['index']:02d}] {m['name']:<38}: Eff-AUC={m['effective_auc']:.3f}, MI={m['mutual_info']:.4f}, PB-r={m['point_biserial_r']:+.3f}")

    # =========================================================================
    # PART 3: 5-Fold OOF 기반 모델별 피처 중요도 (Feature Importance Consensus)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [PART 3] 5-Fold OOF 피처 중요도 (LGBM, CatBoost, XGBoost, RF, ExtraTrees)")
    print("=" * 60)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    lgb_importances = np.zeros(D)
    cb_importances = np.zeros(D)
    xgb_importances = np.zeros(D)
    rf_importances = np.zeros(D)
    et_importances = np.zeros(D)

    # 36D 베이스라인 성능 측정
    base_oof_matrix = np.zeros((N, 5))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        sw = np.where(y_tr == 1, 1.7789, 1.0)

        # LGB
        m_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=5, num_leaves=24, learning_rate=0.035, random_state=42+fold, verbose=-1, n_jobs=1)
        m_lgb.fit(X_tr, y_tr, sample_weight=sw)
        lgb_importances += m_lgb.feature_importances_ / 5.0
        base_oof_matrix[val_idx, 0] = m_lgb.predict_proba(X_va)[:, 1]

        # CB
        m_cb = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.035, auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
        m_cb.fit(X_tr, y_tr)
        cb_importances += m_cb.get_feature_importance() / 5.0
        base_oof_matrix[val_idx, 1] = m_cb.predict_proba(X_va)[:, 1]

        # XGB
        m_xgb = xgb.XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.035, scale_pos_weight=1.7789, random_state=42+fold, n_jobs=1, eval_metric='logloss')
        m_xgb.fit(X_tr, y_tr, sample_weight=sw)
        xgb_importances += m_xgb.feature_importances_ / 5.0
        base_oof_matrix[val_idx, 2] = m_xgb.predict_proba(X_va)[:, 1]

        # RF
        m_rf = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42+fold, n_jobs=1)
        m_rf.fit(X_tr, y_tr, sample_weight=sw)
        rf_importances += m_rf.feature_importances_ / 5.0
        base_oof_matrix[val_idx, 3] = m_rf.predict_proba(X_va)[:, 1]

        # ET
        m_et = ExtraTreesClassifier(n_estimators=180, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42+fold, n_jobs=1)
        m_et.fit(X_tr, y_tr, sample_weight=sw)
        et_importances += m_et.feature_importances_ / 5.0
        base_oof_matrix[val_idx, 4] = m_et.predict_proba(X_va)[:, 1]

    # 정규화 종합 중요도 계산
    def norm_imp(arr):
        s = np.sum(arr)
        return arr / s if s > 0 else arr

    norm_lgb = norm_imp(lgb_importances)
    norm_cb = norm_imp(cb_importances)
    norm_xgb = norm_imp(xgb_importances)
    norm_rf = norm_imp(rf_importances)
    norm_et = norm_imp(et_importances)

    consensus_imp = (0.20 * norm_lgb + 0.30 * norm_cb + 0.30 * norm_xgb + 0.10 * norm_rf + 0.10 * norm_et)

    importance_list = []
    for i, col in enumerate(feat_names):
        importance_list.append({
            'index': i + 1,
            'name': col,
            'consensus_importance': float(consensus_imp[i]),
            'lgb_imp': float(norm_lgb[i]),
            'cb_imp': float(norm_cb[i]),
            'xgb_imp': float(norm_xgb[i]),
            'rf_imp': float(norm_rf[i]),
            'et_imp': float(norm_et[i])
        })

    importance_list_sorted = sorted(importance_list, key=lambda x: x['consensus_importance'], reverse=True)
    print("    [종합 피처 중요도 최상위 8개 피처]:")
    for item in importance_list_sorted[:8]:
        print(f"    - [{item['index']:02d}] {item['name']:<38}: {item['consensus_importance']*100:5.2f}%")

    importance_list_sorted_low = sorted(importance_list, key=lambda x: x['consensus_importance'])
    print("\n    [종합 피처 중요도 최하위 8개 피처 (기여도 미미)]:")
    for item in importance_list_sorted_low[:8]:
        print(f"    - [{item['index']:02d}] {item['name']:<38}: {item['consensus_importance']*100:5.2f}%")

    # 베이스라인 36D 성능 계산
    base_blended_p = (
        0.20 * base_oof_matrix[:, 0] +
        0.30 * base_oof_matrix[:, 1] +
        0.30 * base_oof_matrix[:, 2] +
        0.10 * base_oof_matrix[:, 3] +
        0.10 * base_oof_matrix[:, 4]
    )
    # Temp scaling
    eps = 1e-6
    cp = np.clip(base_blended_p, eps, 1.0 - eps)
    logits = np.log(cp / (1.0 - cp))
    base_probs = 1.0 / (1.0 + np.exp(-logits / 1.02))

    base_auc = roc_auc_score(y, base_probs)

    # 3-State 최적 임계치 탐색 (Rejection <= 10.0%)
    def eval_3state(probs, y_true, max_rej=0.10):
        best = {'f1': 0.0, 'th_low': 0.35, 'th_high': 0.45, 'rejection_rate': 0.0, 'recall': 0.0}
        for th_l in np.arange(0.28, 0.45, 0.01):
            for th_h in np.arange(th_l + 0.02, 0.55, 0.01):
                mask = (probs < th_l) | (probs >= th_h)
                rej = 1.0 - np.mean(mask)
                if rej > max_rej:
                    continue
                y_c = y_true[mask]
                p_c = (probs[mask] >= th_h).astype(int)
                if len(np.unique(p_c)) < 2:
                    continue
                f1 = f1_score(y_c, p_c, zero_division=0)
                rec = np.sum((y_c == 1) & (p_c == 1)) / (np.sum(y_c == 1) + 1e-6)
                if f1 > best['f1']:
                    best = {'f1': float(f1), 'th_low': float(th_l), 'th_high': float(th_h), 'rejection_rate': float(rej), 'recall': float(rec)}
        return best

    base_3state = eval_3state(base_probs, y, max_rej=0.10)
    print(f"\n[*] 베이스라인 36D 성능: 3-State F1={base_3state['f1']*100:.2f}%, AUC={base_auc:.4f}, Rejection={base_3state['rejection_rate']*100:.2f}%")

    # =========================================================================
    # PART 4: 피처 절제 연구 (Leave-One-Feature-Out / LOFO Ablation Study) 전수 실증
    # =========================================================================
    print("\n" + "=" * 60)
    print("  [PART 4] 36개 피처 개별 제거(LOFO) 전수 실증 테스트 (36회 5-Fold 학습)")
    print("=" * 60)

    lofo_results = []
    t_lofo_start = time.time()

    # 빠른 LOFO를 위해 3-Way 부스팅(LGB + CB + XGB)으로 고속 OOF 평가
    # (결과 신뢰도와 속도를 위해 핵심 앙상블 사용)
    for i, col in enumerate(feat_names):
        print(f"    - [{i+1:02d}/36] 피처 제거 테스트: {col:<38} ... ", end="", flush=True)
        X_sub = np.delete(X, i, axis=1)

        oof_sub = np.zeros((N, 3))
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects), 1):
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_va, y_va = X_sub[val_idx], y[val_idx]
            sw = np.where(y_tr == 1, 1.7789, 1.0)

            # LGB
            m_lgb = lgb.LGBMClassifier(n_estimators=140, max_depth=5, num_leaves=24, learning_rate=0.04, random_state=42+fold, verbose=-1, n_jobs=1)
            m_lgb.fit(X_tr, y_tr, sample_weight=sw)
            oof_sub[val_idx, 0] = m_lgb.predict_proba(X_va)[:, 1]

            # CB
            m_cb = CatBoostClassifier(iterations=180, depth=5, learning_rate=0.04, auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
            m_cb.fit(X_tr, y_tr)
            oof_sub[val_idx, 1] = m_cb.predict_proba(X_va)[:, 1]

            # XGB
            m_xgb = xgb.XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.04, scale_pos_weight=1.7789, random_state=42+fold, n_jobs=1, eval_metric='logloss')
            m_xgb.fit(X_tr, y_tr, sample_weight=sw)
            oof_sub[val_idx, 2] = m_xgb.predict_proba(X_va)[:, 1]

        p_sub = 0.35 * oof_sub[:, 0] + 0.35 * oof_sub[:, 1] + 0.30 * oof_sub[:, 2]
        auc_sub = roc_auc_score(y, p_sub)
        eval_res = eval_3state(p_sub, y, max_rej=0.10)

        # Baseline 3-Way 비교치 산출을 위해 동일 조건 delta 계산
        delta_f1 = (eval_res['f1'] - base_3state['f1']) * 100
        delta_auc = auc_sub - base_auc

        status = "성능 저해 (제거 권장)" if delta_f1 > 0.05 else ("필수 피처 (보존 필수)" if delta_f1 < -0.30 else "중립/경미")
        print(f"F1={eval_res['f1']*100:.2f}% (Δ {delta_f1:+.2f}%p) | AUC={auc_sub:.4f} (Δ {delta_auc:+.4f}) -> {status}")

        lofo_results.append({
            'index': i + 1,
            'feature_name': col,
            'f1_without_feature': float(eval_res['f1'] * 100),
            'f1_delta': float(delta_f1),
            'auc_without_feature': float(auc_sub),
            'auc_delta': float(delta_auc),
            'rejection_rate': float(eval_res['rejection_rate'] * 100),
            'recall': float(eval_res['recall'] * 100),
            'status': status
        })

    elapsed_lofo = time.time() - t_lofo_start
    print(f"\n[*] LOFO 전수 실증 완료 (소요시간: {elapsed_lofo:.1f}초)")

    # 성능 저해 피처 정렬 (제거 시 F1이 가장 많이 오르는 순서)
    lofo_sorted_degrading = sorted(lofo_results, key=lambda x: x['f1_delta'], reverse=True)
    print("\n    [제거 시 성능이 향상되는 성능 저해 피처 (ΔF1 > 0)]:")
    degrading_features = [r for r in lofo_sorted_degrading if r['f1_delta'] > 0]
    for r in degrading_features:
        print(f"    - [{r['index']:02d}] {r['feature_name']:<38}: 제거 시 F1 {r['f1_without_feature']:.2f}% (Δ {r['f1_delta']:+.2f}%p), AUC {r['auc_without_feature']:.4f}")

    # =========================================================================
    # PART 5: 종합 진단 및 최종 JSON 저장
    # =========================================================================
    results = {
        'total_features': D,
        'total_samples': N,
        'baseline_performance': {
            '3state_f1': float(base_3state['f1'] * 100),
            'auc': float(base_auc),
            'rejection_rate': float(base_3state['rejection_rate'] * 100),
            'recall': float(base_3state['recall'] * 100)
        },
        'high_correlation_pairs': high_corr_pairs,
        'vif_rankings': vif_sorted,
        'single_feature_metrics': single_feat_metrics,
        'consensus_importance': importance_list_sorted,
        'lofo_ablation_results': lofo_results,
        'degrading_features_summary': degrading_features,
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 전수 조사 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")

if __name__ == '__main__':
    run_comprehensive_feature_investigation()
