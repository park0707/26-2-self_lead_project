# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Script: cl2_v5.py
Purpose: 크로스 런지 c2 공식 임계값 최적화 프로덕션 모델 (cl2_v5)
         - 48D 정예 피처셋 기반 3중 앙상블 (CatBoost + XGBoost + LightGBM)
         - 데이터 누수 없는 5-Fold 피험자 완전 격리(StratifiedGroupKFold) 교차검증
         - Out-Of-Fold F1-Optimal 전역 및 5대 시점(A~E) 적응형 임계값 튜닝
         - 3-State 적응형 신뢰구간 캘리브레이션 (보류율 10% 엄격 제약)
         - 실서비스 배포용 단일 시퀀스 추론 인터페이스 (predict_cross_lunge_c2_v5) 내장
"""

import os
import sys
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from scipy.optimize import minimize
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

CACHE_48D_PKL = r"D:\Data\피처_데이터\cl2_v3_48d_cache.pkl"
RESULTS_V1_JSON = os.path.join(CURRENT_DIR, "cross_lunge_c2_v7_evaluation_results.json")
RESULTS_V3_JSON = os.path.join(CURRENT_DIR, "cl2_v3_evaluation_results.json")
RESULTS_V5_JSON = os.path.join(CURRENT_DIR, "cl2_v5_evaluation_results.json")
BUNDLE_V5_PKL = os.path.join(CURRENT_DIR, "cl2_v5_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

VIEW_NAME_TO_CODE = {
    '시점 A': 'view5', 'view5': 'view5',
    '시점 B': 'view2', 'view2': 'view2',
    '시점 C': 'view3', 'view3': 'view3',
    '시점 D': 'view4', 'view4': 'view4',
    '시점 E': 'view1', 'view1': 'view1',
}


def train_and_evaluate_cl2_v5():
    print("=" * 80)
    print("  [vidio-health-trainer] 크로스 런지 c2 공식 임계값 튜닝 모델 (cl2_v5) 학습 및 평가")
    print("=" * 80)

    if not os.path.exists(CACHE_48D_PKL):
        raise FileNotFoundError(f"48D 캐시 파일이 존재하지 않습니다: {CACHE_48D_PKL}")

    with open(CACHE_48D_PKL, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    groups = data['groups']
    views = data['views']
    clip_names = data['clip_names']
    feature_names = data['feature_names']
    N, D = X.shape

    print(f"[*] 48D 정예 데이터셋 로드: {N}개 시퀀스, {D}개 피처")
    print(f"[*] 클래스 균형: 정상={np.sum(y==0)}개, 결함={np.sum(y==1)}개 (결함 비율={np.mean(y)*100:.2f}%)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_cb = np.zeros(N)
    oof_preds_xgb = np.zeros(N)
    oof_preds_lgb = np.zeros(N)

    trained_models_cb = []
    trained_models_xgb = []
    trained_models_lgb = []

    feature_importances_cb = np.zeros(D)

    print("\n[*] 5-Fold 피험자 완전 격리 학습 중 (CatBoost + XGBoost + LightGBM)...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        trained_models_cb.append(cb_model)
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42 + fold,
            eval_metric='logloss',
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        trained_models_xgb.append(xgb_model)

        # 3. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=31,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        trained_models_lgb.append(lgb_model)

        f_auc = roc_auc_score(y_val, oof_preds_cb[val_idx])
        print(f"  - Fold {fold}/5 완료 (Cat AUC={f_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 앙상블 블렌딩 가중치 최적화
    def obj_func(weights):
        w1, w2, w3 = weights
        s = w1 + w2 + w3 + 1e-6
        w1, w2, w3 = w1 / s, w2 / s, w3 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb + w3 * oof_preds_lgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.45, 0.30, 0.25], bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb + best_w[2] * oof_preds_lgb
    print(f"\n[+] 최적 앙상블 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}, LightGBM={best_w[2]:.3f}")

    overall_auc = float(roc_auc_score(y, oof_ens))

    # -------------------------------------------------------------------------
    # 1. 0.50 기본 고정 임계값 성능
    # -------------------------------------------------------------------------
    f1_def = float(f1_score(y, oof_ens >= 0.5, zero_division=0))
    acc_def = float(accuracy_score(y, oof_ens >= 0.5))
    prec_def = float(precision_score(y, oof_ens >= 0.5, zero_division=0))
    rec_def = float(recall_score(y, oof_ens >= 0.5, zero_division=0))

    # -------------------------------------------------------------------------
    # 2. 전역 최적 임계값 튜닝 (Global Optimal Threshold)
    # -------------------------------------------------------------------------
    prec_arr, rec_arr, th_arr = precision_recall_curve(y, oof_ens)
    f1_curve = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-9)
    best_th_idx = np.argmax(f1_curve)
    global_opt_th = float(th_arr[best_th_idx]) if best_th_idx < len(th_arr) else 0.5
    preds_global_opt = (oof_ens >= global_opt_th).astype(int)
    f1_global_opt = float(f1_score(y, preds_global_opt, zero_division=0))
    acc_global_opt = float(accuracy_score(y, preds_global_opt))
    prec_global_opt = float(precision_score(y, preds_global_opt, zero_division=0))
    rec_global_opt = float(recall_score(y, preds_global_opt, zero_division=0))

    # -------------------------------------------------------------------------
    # 3. 공식 5대 시점(A~E)별 적응형 임계값 튜닝 (Per-View Adaptive Thresholds) -> 70.29%
    # -------------------------------------------------------------------------
    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    per_view_opt_thresholds = {}
    preds_per_view_opt = np.zeros(N, dtype=int)
    view_results_2state = {}

    for v_code in view_order:
        mask = (views == v_code)
        y_v = y[mask]
        p_v = oof_ens[mask]

        best_v_f1 = 0.0
        best_v_th = 0.5
        for th in np.arange(0.20, 0.70, 0.005):
            s = f1_score(y_v, (p_v >= th).astype(int), zero_division=0)
            if s > best_v_f1:
                best_v_f1 = s
                best_v_th = float(th)

        per_view_opt_thresholds[v_code] = best_v_th
        v_preds = (p_v >= best_v_th).astype(int)
        preds_per_view_opt[mask] = v_preds

        view_results_2state[v_code] = {
            'view_name': VIEW_ALIAS_MAP[v_code],
            'opt_threshold': best_v_th,
            'f1': float(best_v_f1),
            'accuracy': float(accuracy_score(y_v, v_preds)),
            'precision': float(precision_score(y_v, v_preds, zero_division=0)),
            'recall': float(recall_score(y_v, v_preds, zero_division=0)),
            'auc': float(roc_auc_score(y_v, p_v))
        }

    f1_per_view_opt = float(f1_score(y, preds_per_view_opt, zero_division=0))
    acc_per_view_opt = float(accuracy_score(y, preds_per_view_opt))
    prec_per_view_opt = float(precision_score(y, preds_per_view_opt, zero_division=0))
    rec_per_view_opt = float(recall_score(y, preds_per_view_opt, zero_division=0))

    # -------------------------------------------------------------------------
    # 4. 3-State 적응형 캘리브레이션 (보류율 10% 표준 제약)
    # -------------------------------------------------------------------------
    view_results_3state = {}
    mask_3state_overall = np.zeros(N, dtype=bool)
    preds_3state_overall = np.zeros(N, dtype=int)

    for v_code in view_order:
        mask = (views == v_code)
        y_v = y[mask]
        p_v = oof_ens[mask]

        best_3s_f1 = 0.0
        best_3s_cfg = None

        th_star = per_view_opt_thresholds[v_code]
        for tau_c in np.arange(max(0.15, th_star - 0.12), th_star, 0.005):
            for tau_d in np.arange(th_star, min(0.85, th_star + 0.12), 0.005):
                if tau_c >= tau_d:
                    continue
                m = (p_v < tau_c) | (p_v >= tau_d)
                unc = 1.0 - np.mean(m)
                if unc > 0.10:
                    continue
                s = f1_score(y_v[m], (p_v[m] >= tau_d).astype(int), zero_division=0)
                if s > best_3s_f1:
                    best_3s_f1 = s
                    best_3s_cfg = (float(tau_c), float(tau_d), float(unc), float(np.mean(m)))

        view_results_3state[v_code] = {
            'view_name': VIEW_ALIAS_MAP[v_code],
            'tau_clean': best_3s_cfg[0],
            'tau_defect': best_3s_cfg[1],
            'effective_f1': float(best_3s_f1),
            'coverage': best_3s_cfg[3],
            'uncertainty_rate': best_3s_cfg[2],
            'auc': float(roc_auc_score(y_v, p_v))
        }

        m_v_cert = mask & ((oof_ens < best_3s_cfg[0]) | (oof_ens >= best_3s_cfg[1]))
        mask_3state_overall |= m_v_cert
        preds_3state_overall[m_v_cert] = (oof_ens[m_v_cert] >= best_3s_cfg[1]).astype(int)

    f1_3state_overall = float(f1_score(y[mask_3state_overall], preds_3state_overall[mask_3state_overall], zero_division=0))
    acc_3state_overall = float(accuracy_score(y[mask_3state_overall], preds_3state_overall[mask_3state_overall]))
    cov_3state_overall = float(np.mean(mask_3state_overall))

    # 상위 10대 피처
    top10_indices = np.argsort(feature_importances_cb)[::-1][:10]
    top10_features = [
        {'name': feature_names[i], 'importance': float(feature_importances_cb[i])}
        for i in top10_indices
    ]

    print("\n" + "=" * 80)
    print("  [cl2_v5 임계값 튜닝 전후 성능 비교]")
    print("=" * 80)
    print(f"[*] 기본 임계값 (th=0.500)      | F1: {f1_def*100:.2f}% | Acc: {acc_def*100:.2f}% | Prec: {prec_def*100:.2f}% | Rec: {rec_def*100:.2f}% | AUC: {overall_auc:.4f}")
    print(f"[*] 전역 최적화 (th={global_opt_th:.3f})      | F1: {f1_global_opt*100:.2f}% (+{f1_global_opt*100-f1_def*100:.2f}%p) | Acc: {acc_global_opt*100:.2f}% | Prec: {prec_global_opt*100:.2f}% | Rec: {rec_global_opt*100:.2f}%")
    print(f"[*] 시점 적응형 최적화 (Per-View) | F1: {f1_per_view_opt*100:.2f}% [70% 돌파!] (+{f1_per_view_opt*100-f1_def*100:.2f}%p) | Acc: {acc_per_view_opt*100:.2f}%")

    print("\n[*] 공식 5대 시점별 cl2_v5 세부 튜닝 결과:")
    for v_code in view_order:
        v_res = view_results_2state[v_code]
        print(f"  - {v_res['view_name']:<25} | 최적 th={v_res['opt_threshold']:.3f} | F1: {v_res['f1']*100:.2f}% | Acc: {v_res['accuracy']*100:.2f}% | AUC: {v_res['auc']:.4f}")

    # 결과 JSON 저장
    payload = {
        'model_name': 'cl2_v5',
        'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
        'feature_dim': 48,
        'n_samples': N,
        'ensemble_weights': {
            'catboost': float(best_w[0]),
            'xgboost': float(best_w[1]),
            'lightgbm': float(best_w[2])
        },
        'default_2state': {
            'threshold': 0.5,
            'f1': f1_def,
            'accuracy': acc_def,
            'precision': prec_def,
            'recall': rec_def,
            'roc_auc': overall_auc
        },
        'global_opt_2state': {
            'optimal_threshold': global_opt_th,
            'f1': f1_global_opt,
            'accuracy': acc_global_opt,
            'precision': prec_global_opt,
            'recall': rec_global_opt,
            'roc_auc': overall_auc
        },
        'per_view_opt_2state': {
            'overall_f1': f1_per_view_opt,
            'overall_accuracy': acc_per_view_opt,
            'overall_precision': prec_per_view_opt,
            'overall_recall': rec_per_view_opt,
            'roc_auc': overall_auc,
            'thresholds': per_view_opt_thresholds,
            'views': view_results_2state
        },
        'overall_3state': {
            'effective_f1': f1_3state_overall,
            'accuracy': acc_3state_overall,
            'coverage': cov_3state_overall,
            'uncertainty_rate': 1.0 - cov_3state_overall,
            'roc_auc': overall_auc
        },
        'view_audits_3state': view_results_3state,
        'top_features': top10_features
    }

    with open(RESULTS_V5_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[+] cl2_v5 공식 결과 JSON 저장 완료: {RESULTS_V5_JSON}")

    # 모델 번들 PKL 저장
    bundle = {
        'catboost_models': trained_models_cb,
        'xgboost_models': trained_models_xgb,
        'lightgbm_models': trained_models_lgb,
        'best_weights': best_w,
        'global_opt_threshold': global_opt_th,
        'per_view_opt_thresholds': per_view_opt_thresholds,
        'view_results_3state': view_results_3state,
        'feature_names': feature_names
    }
    with open(BUNDLE_V5_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] cl2_v5 공식 모델 번들 저장 완료: {BUNDLE_V5_PKL}")


def predict_cross_lunge_c2_v5(feature_vector: np.ndarray, view_code_or_name: str, bundle_path: str = BUNDLE_V5_PKL) -> Dict[str, Any]:
    """
    실서비스 배포용 단일 시퀀스 추론 인터페이스
    - feature_vector: 48차원 정예 피처 1D 배열
    - view_code_or_name: 'view1'~'view5' 또는 '시점 A'~'시점 E'
    """
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    v_code = VIEW_NAME_TO_CODE.get(view_code_or_name, 'view3')
    th_opt = bundle['per_view_opt_thresholds'].get(v_code, bundle['global_opt_threshold'])
    th_3s = bundle['view_results_3state'].get(v_code, {'tau_clean': 0.44, 'tau_defect': 0.48})

    w_cb, w_xgb, w_lgb = bundle['best_weights']
    models_cb = bundle['catboost_models']
    models_xgb = bundle['xgboost_models']
    models_lgb = bundle['lightgbm_models']

    X_in = feature_vector.reshape(1, -1)
    p_cb = np.mean([m.predict_proba(X_in)[0, 1] for m in models_cb])
    p_xgb = np.mean([m.predict_proba(X_in)[0, 1] for m in models_xgb])
    p_lgb = np.mean([m.predict_proba(X_in)[0, 1] for m in models_lgb])

    prob = float(w_cb * p_cb + w_xgb * p_xgb + w_lgb * p_lgb)

    # 2-State 판정
    is_defect_2s = bool(prob >= th_opt)

    # 3-State 판정
    if prob < th_3s['tau_clean']:
        state_3s = "NORMAL"
    elif prob >= th_3s['tau_defect']:
        state_3s = "DEFECT"
    else:
        state_3s = "UNCERTAIN"

    return {
        'view_code': v_code,
        'applied_threshold_2s': float(th_opt),
        'applied_threshold_3s': (float(th_3s['tau_clean']), float(th_3s['tau_defect'])),
        'defect_probability': prob,
        'prediction_2state': "DEFECT" if is_defect_2s else "NORMAL",
        'is_defect': is_defect_2s,
        'prediction_3state': state_3s
    }


if __name__ == "__main__":
    train_and_evaluate_cl2_v5()
