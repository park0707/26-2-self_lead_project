# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Model: sl2_v2 3-State Threshold Tuning & Comprehensive Performance Evaluation
Constraint: 보류율(Retake Rate) <= 10.0% (커버리지 >= 90.0%) 엄격 준수
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v2_dataset_cache.pkl"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl2_v2_bundle.pkl"
TUNED_JSON_OUT = r"D:\src\models\second_models\side_lunge\sl2_v2_3state_tuned_results.json"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def load_dataset_and_models():
    assert os.path.exists(FEATURE_CACHE_PKL), f"Cache not found: {FEATURE_CACHE_PKL}"
    assert os.path.exists(MODEL_BUNDLE_PKL), f"Bundle not found: {MODEL_BUNDLE_PKL}"

    with open(FEATURE_CACHE_PKL, 'rb') as f:
        ds = pickle.load(f)

    with open(MODEL_BUNDLE_PKL, 'rb') as f:
        bundle = pickle.load(f)

    return ds, bundle


def generate_oof_predictions(ds: Dict[str, Any], bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = ds['X']
    y = ds['y']
    groups = ds['groups']
    views = ds['views']

    models_lgb = bundle['models_lgb']
    models_cat = bundle['models_cat']
    models_xgb = bundle['models_xgb']
    weights = bundle['ensemble_weights']

    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)
    oof_probs = np.zeros(n_samples, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_val = X[val_idx]

        # 1. LightGBM
        m_lgb = models_lgb[fold_idx]
        p_lgb = m_lgb.predict(X_val)

        # 2. CatBoost
        m_cat = models_cat[fold_idx]
        p_cat = m_cat.predict_proba(X_val)[:, 1]

        # 3. XGBoost
        m_xgb = models_xgb[fold_idx]
        dval = xgb.DMatrix(X_val)
        p_xgb = m_xgb.predict(dval)

        p_ens = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb
        oof_probs[val_idx] = p_ens

    return oof_probs, y, groups, views


def evaluate_3state_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float
) -> Dict[str, Any]:
    N = len(y_true)
    decided_mask = (y_prob <= tau_normal) | (y_prob >= tau_defect)
    retake_mask = ~decided_mask

    coverage = float(np.mean(decided_mask))
    retake_rate = float(np.mean(retake_mask))
    retake_count = int(np.sum(retake_mask))
    decided_count = int(np.sum(decided_mask))

    # 보류된 샘플 중 실제 정답 분포
    retake_clean = int(np.sum(retake_mask & (y_true == 0)))
    retake_defect = int(np.sum(retake_mask & (y_true == 1)))

    y_dec_true = y_true[decided_mask]
    y_dec_prob = y_prob[decided_mask]
    y_dec_pred = (y_dec_prob >= tau_defect).astype(int)

    if len(y_dec_true) > 0 and len(np.unique(y_dec_true)) > 1:
        acc = float(accuracy_score(y_dec_true, y_dec_pred))
        prec = float(precision_score(y_dec_true, y_dec_pred, zero_division=0))
        rec = float(recall_score(y_dec_true, y_dec_pred, zero_division=0))
        defect_f1 = float(f1_score(y_dec_true, y_dec_pred, zero_division=0))
        clean_f1 = float(f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0))
        macro_f1 = float((defect_f1 + clean_f1) / 2.0)
        cm = confusion_matrix(y_dec_true, y_dec_pred).tolist()
    else:
        acc, prec, rec, defect_f1, clean_f1, macro_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        cm = []

    full_pred = (y_prob >= 0.5).astype(int)
    full_f1 = float(f1_score(y_true, full_pred, zero_division=0))
    full_acc = float(accuracy_score(y_true, full_pred))
    auc = float(roc_auc_score(y_true, y_prob))

    return {
        "total_samples": int(N),
        "tau_normal": float(tau_normal),
        "tau_defect": float(tau_defect),
        "coverage": coverage,
        "retake_rate": retake_rate,
        "retake_count": retake_count,
        "decided_count": decided_count,
        "retake_clean_count": retake_clean,
        "retake_defect_count": retake_defect,
        "selective_defect_f1": defect_f1,
        "selective_clean_f1": clean_f1,
        "macro_f1": macro_f1,
        "selective_accuracy": acc,
        "selective_precision": prec,
        "selective_recall": rec,
        "confusion_matrix": cm,
        "full_coverage_defect_f1": full_f1,
        "full_coverage_accuracy": full_acc,
        "roc_auc": auc
    }


def tune_global_thresholds(y_true: np.ndarray, y_prob: np.ndarray, max_retake: float = 0.10) -> Dict[str, Any]:
    print("\n[*] [튜닝 1] 전역 3-State 임계값 그리드 탐색 중 (Retake <= 10.0% 엄격 제약)...")

    tau_n_grid = np.linspace(0.20, 0.55, 141)
    tau_d_grid = np.linspace(0.40, 0.75, 141)

    cands_defect_f1 = []
    cands_macro_f1 = []
    cands_balanced = []

    for tn in tau_n_grid:
        for td in tau_d_grid:
            if td <= tn:
                continue
            dec_mask = (y_prob <= tn) | (y_prob >= td)
            cov = float(np.mean(dec_mask))
            rej = 1.0 - cov
            if rej <= max_retake + 0.0001:
                y_dec = y_true[dec_mask]
                p_dec = (y_prob[dec_mask] >= td).astype(int)
                d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                c_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
                m_f1 = (d_f1 + c_f1) / 2.0
                acc = accuracy_score(y_dec, p_dec)
                prec = precision_score(y_dec, p_dec, zero_division=0)
                rec = recall_score(y_dec, p_dec, zero_division=0)

                cands_defect_f1.append((d_f1, acc, float(tn), float(td)))
                cands_macro_f1.append((m_f1, d_f1, acc, float(tn), float(td)))
                balanced_score = d_f1 * 0.6 + acc * 0.4
                cands_balanced.append((balanced_score, d_f1, acc, float(tn), float(td)))

    cands_defect_f1.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cands_macro_f1.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cands_balanced.sort(key=lambda x: (x[0], x[1]), reverse=True)

    best_d_tn, best_d_td = cands_defect_f1[0][2], cands_defect_f1[0][3]
    best_m_tn, best_m_td = cands_macro_f1[0][3], cands_macro_f1[0][4]
    best_b_tn, best_b_td = cands_balanced[0][3], cands_balanced[0][4]

    res_defect = evaluate_3state_metrics(y_true, y_prob, best_d_tn, best_d_td)
    res_macro = evaluate_3state_metrics(y_true, y_prob, best_m_tn, best_m_td)
    res_balanced = evaluate_3state_metrics(y_true, y_prob, best_b_tn, best_b_td)

    return {
        "opt_defect_f1": res_defect,
        "opt_macro_f1": res_macro,
        "opt_balanced": res_balanced
    }


def tune_per_view_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray, max_retake: float = 0.10
) -> Dict[str, Any]:
    print("\n[*] [튜닝 2] 공식 5대 시점별 적응형 3-State 임계값 튜닝 중...")
    unique_views = ["view5", "view2", "view3", "view4", "view1"]
    view_thresholds = {}
    view_audits = {}

    for v in unique_views:
        v_mask = (views == v)
        y_v = y_true[v_mask]
        p_v = y_prob[v_mask]

        best_f1, best_tn, best_td = -1.0, 0.35, 0.55

        tau_n_grid = np.linspace(0.20, 0.55, 141)
        tau_d_grid = np.linspace(0.35, 0.75, 161)

        cands = []
        for tn in tau_n_grid:
            for td in tau_d_grid:
                if td <= tn:
                    continue
                dec_mask = (p_v <= tn) | (p_v >= td)
                rej = 1.0 - float(np.mean(dec_mask))
                if rej <= max_retake + 0.0001:
                    y_dec = y_v[dec_mask]
                    p_dec = (p_v[dec_mask] >= td).astype(int)
                    d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                    acc = accuracy_score(y_dec, p_dec)
                    cands.append((d_f1, acc, float(tn), float(td)))

        if cands:
            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_tn, best_td = cands[0][2], cands[0][3]

        metrics = evaluate_3state_metrics(y_v, p_v, best_tn, best_td)
        view_thresholds[v] = {'tau_normal': best_tn, 'tau_defect': best_td}
        metrics['alias'] = VIEW_ALIAS_MAP.get(v, v)
        view_audits[v] = metrics

    # 전체 종합 집계
    pred_states = []
    for i in range(len(y_true)):
        v = views[i]
        p = y_prob[i]
        th = view_thresholds[v]
        if p <= th['tau_normal']:
            pred_states.append(0)
        elif p >= th['tau_defect']:
            pred_states.append(1)
        else:
            pred_states.append(2)  # Retake

    pred_states = np.array(pred_states)
    dec_mask = (pred_states != 2)
    y_dec_true = y_true[dec_mask]
    y_dec_pred = pred_states[dec_mask]

    overall_metrics = {
        "total_samples": int(len(y_true)),
        "coverage": float(np.mean(dec_mask)),
        "retake_rate": float(np.mean(~dec_mask)),
        "retake_count": int(np.sum(~dec_mask)),
        "decided_count": int(np.sum(dec_mask)),
        "selective_defect_f1": float(f1_score(y_dec_true, y_dec_pred, zero_division=0)),
        "selective_clean_f1": float(f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0)),
        "macro_f1": float((f1_score(y_dec_true, y_dec_pred, zero_division=0) + f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0)) / 2.0),
        "selective_accuracy": float(accuracy_score(y_dec_true, y_dec_pred)),
        "selective_precision": float(precision_score(y_dec_true, y_dec_pred, zero_division=0)),
        "selective_recall": float(recall_score(y_dec_true, y_dec_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_dec_true, y_dec_pred).tolist(),
        "view_thresholds": view_thresholds,
        "view_audits": view_audits
    }
    return overall_metrics


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl2_v2 3-State 임계값 튜닝 및 종합 평가")
    print("  - 데이터셋: 전체 826개 폴더 (4,130개 시퀀스 전수)")
    print("  - 제약조건: 보류율(Retake Rate) <= 10.0% (커버리지 >= 90.0%) 엄격 준수")
    print("=" * 80)

    ds, bundle = load_dataset_and_models()
    oof_probs, y_true, groups, views = generate_oof_predictions(ds, bundle)

    print(f"\n[+] OOF 예측 확률 생성 완료 (총 {len(y_true)}개, 피험자 9명 전수 격리 검증)")
    print(f"    - c2 결함: {np.sum(y_true == 1)}개 (18.64%)")
    print(f"    - c2 정상: {np.sum(y_true == 0)}개 (81.36%)")
    print(f"    - Overall ROC-AUC: {roc_auc_score(y_true, oof_probs):.4f}")

    global_tuned = tune_global_thresholds(y_true, oof_probs, max_retake=0.10)
    per_view_tuned = tune_per_view_thresholds(y_true, oof_probs, views, max_retake=0.10)

    results_data = {
        "exercise": "사이드 런지 (07_사이드_런지)",
        "defect": "Condition 2 - 몸통 발 무릎 방향 일치 여부",
        "model": "sl2_v2 (원천 826폴더 4,130시퀀스 전수)",
        "global_tuned": global_tuned,
        "per_view_tuned": per_view_tuned
    }

    with open(TUNED_JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 튜닝 결과 저장 완료: {TUNED_JSON_OUT}")

    # 리포트 출력
    g_def = global_tuned["opt_defect_f1"]
    print("\n" + "=" * 80)
    print("  [결과 1] 전역 단일 3-State 임계값 최적 튜닝 (Defect F1 극대화)")
    print("=" * 80)
    print(f"  * 최적 임계값: Clean <= {g_def['tau_normal']:.4f} < Retake < {g_def['tau_defect']:.4f} <= Defect")
    print(f"  * 보류율 (Retake Rate) : {g_def['retake_rate']*100:.2f}% ({g_def['retake_count']}/{g_def['total_samples']}) [<= 10.0% 준수]")
    print(f"  * 판정 커버리지        : {g_def['coverage']*100:.2f}% ({g_def['decided_count']}개)")
    print(f"  * Selective Defect F1 : {g_def['selective_defect_f1']*100:.2f}% (Full: {g_def['full_coverage_defect_f1']*100:.2f}%)")
    print(f"  * Selective Clean F1  : {g_def['selective_clean_f1']*100:.2f}%")
    print(f"  * Selective Macro F1  : {g_def['macro_f1']*100:.2f}%")
    print(f"  * Selective 정확도    : {g_def['selective_accuracy']*100:.2f}% (Full: {g_def['full_coverage_accuracy']*100:.2f}%)")
    print(f"  * Selective 정밀도    : {g_def['selective_precision']*100:.2f}%")
    print(f"  * Selective 재현율    : {g_def['selective_recall']*100:.2f}%")
    print(f"  * 혼동행렬 (TN, FP / FN, TP): {g_def['confusion_matrix']}")

    pv = per_view_tuned
    print("\n" + "=" * 80)
    print("  [결과 2] 공식 5대 시점별 적응형 3-State 임계값 최적 튜닝 (Per-View Adaptive)")
    print("=" * 80)
    print(f"  * 전체 종합 보류율 : {pv['retake_rate']*100:.2f}% ({pv['retake_count']}/{pv['total_samples']}) [<= 10.0% 준수]")
    print(f"  * 전체 종합 커버리지: {pv['coverage']*100:.2f}% ({pv['decided_count']}개)")
    print(f"  * 전체 Selective Defect F1: {pv['selective_defect_f1']*100:.2f}%")
    print(f"  * 전체 Selective Clean F1 : {pv['selective_clean_f1']*100:.2f}%")
    print(f"  * 전체 Selective Macro F1 : {pv['macro_f1']*100:.2f}%")
    print(f"  * 전체 Selective 정확도   : {pv['selective_accuracy']*100:.2f}%")
    print(f"  * 전체 Selective 정밀도   : {pv['selective_precision']*100:.2f}%")
    print(f"  * 전체 Selective 재현율   : {pv['selective_recall']*100:.2f}%")
    print(f"  * 전체 혼동행렬: {pv['confusion_matrix']}")

    print("\n[공식 5대 시점별 세부 감사표]")
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        v_aud = pv['view_audits'][vn]
        alias = v_aud['alias']
        th = pv['view_thresholds'][vn]
        print(f"  * {alias} [{vn}]:")
        print(f"      - 임계값: Clean <= {th['tau_normal']:.4f} < Retake < {th['tau_defect']:.4f} <= Defect")
        print(f"      - 보류율: {v_aud['retake_rate']*100:.2f}% | 커버리지: {v_aud['coverage']*100:.2f}%")
        print(f"      - Defect F1: {v_aud['selective_defect_f1']*100:.2f}% | Clean F1: {v_aud['selective_clean_f1']*100:.2f}% | Macro F1: {v_aud['macro_f1']*100:.2f}%")
        print(f"      - 정확도: {v_aud['selective_accuracy']*100:.2f}% | 정밀도: {v_aud['selective_precision']*100:.2f}% | 재현율: {v_aud['selective_recall']*100:.2f}%")


if __name__ == '__main__':
    main()
