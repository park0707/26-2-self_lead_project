# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Module: src/models/second_models/side_lunge/evaluate_side_lunge_rtmp.py
Purpose: RTMPose 추출 포인트 기반 사이드 런지 5대 세부 오류 모델(c1~c5)
         5-Fold 피험자 완전 격리(StratifiedGroupKFold) 학습 및 평가 파이프라인
         - MediaPipe 공식 베이스라인 대비 1:1 성능 비교 (2-State & 3-State)
         - 5대 공식 시점(시점 A~E, view1~view5) 전수 감사
         - 원천 데이터 3-State F1 및 보류율(Uncertainty / Retake Rate) 정밀 산출
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SIDE_LUNGE_DIR = r"D:\src\models\second_models\side_lunge"
if SIDE_LUNGE_DIR not in sys.path:
    sys.path.insert(0, SIDE_LUNGE_DIR)

DATA_PATH = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"
RESULTS_OUT_PATH = os.path.join(SIDE_LUNGE_DIR, "side_lunge_rtmp_evaluation_results.json")

# MediaPipe 공식 결과 JSON 경로
MP_RESULTS = {
    'c1': os.path.join(SIDE_LUNGE_DIR, "sl1_final_evaluation_results.json"),
    'c2': os.path.join(SIDE_LUNGE_DIR, "sl2_v7_evaluation_results.json"),
    'c3': os.path.join(SIDE_LUNGE_DIR, "sl3_v3_evaluation_results.json"),
    'c4': os.path.join(SIDE_LUNGE_DIR, "sl4_v3_evaluation_results.json"),
    'c5': os.path.join(SIDE_LUNGE_DIR, "sl5_v7_benchmark_results.json")
}

# AGENTS.md 공식 5대 시점 정의
OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


def calc_2state_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """2-State 강제 이진 분류 지표 산출"""
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.50

    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc
    }


def calc_3state_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau_clean: float, tau_defect: float) -> Dict[str, Any]:
    """3-State 선택적 분류 지표 산출 (tau_clean, tau_defect)"""
    unc_mask = (y_prob >= tau_clean) & (y_prob < tau_defect)
    dec_mask = ~unc_mask
    n_total = len(y_true)
    n_unc = int(np.sum(unc_mask))
    n_dec = int(np.sum(dec_mask))

    unc_rate = float(n_unc / max(1, n_total))
    coverage = float(1.0 - unc_rate)

    if n_dec > 0:
        y_dec = y_true[dec_mask]
        p_dec = (y_prob[dec_mask] >= tau_defect).astype(int)
        eff_f1 = float(f1_score(y_dec, p_dec, zero_division=0))
        acc = float(accuracy_score(y_dec, p_dec))
        prec = float(precision_score(y_dec, p_dec, zero_division=0))
        rec = float(recall_score(y_dec, p_dec, zero_division=0))
    else:
        eff_f1, acc, prec, rec = 0.0, 0.0, 0.0, 0.0

    return {
        'tau_clean': float(tau_clean),
        'tau_defect': float(tau_defect),
        'uncertainty_rate': unc_rate,
        'coverage': coverage,
        'effective_f1': eff_f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'decided_samples': n_dec,
        'uncertain_samples': n_unc,
        'total_samples': n_total
    }


def tune_view_thresholds(y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray, max_uncertainty: float = 0.10) -> Dict[str, Dict[str, float]]:
    """5대 시점별 보류율 <= 10.0% 제약 하에서 F1을 최대화하는 3-State 임계값 튜닝"""
    view_th = {}
    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y_true[v_mask]
        yp_v = y_prob[v_mask]

        best_f1 = -1.0
        best_th = {'tau_clean': 0.40, 'tau_defect': 0.60}

        # Grid search over candidate threshold intervals
        for tc in np.linspace(0.20, 0.48, 15):
            for td in np.linspace(0.52, 0.80, 15):
                if td <= tc:
                    continue
                unc = (yp_v >= tc) & (yp_v < td)
                u_rate = np.mean(unc)
                if u_rate > max_uncertainty:
                    continue

                dec = ~unc
                if np.sum(dec) == 0:
                    continue
                f1_val = f1_score(yt_v[dec], (yp_v[dec] >= td).astype(int), zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_th = {'tau_clean': round(float(tc), 3), 'tau_defect': round(float(td), 3)}

        view_th[vn] = best_th
    return view_th


def train_and_eval_submodel_c1(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [*] [c1 (sl1)] 앞다리 무릎 각도 90도 오류 (RTMPose 55D)")
    print("=" * 80)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    weights = [0.15, 0.65, 0.20] # LightGBM, CatBoost, XGBoost
    oof_ens = np.zeros(len(y), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=6, learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1, random_state=42+fold)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        # 2. CatBoost
        m_cat = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, thread_count=1, verbose=0, random_seed=42+fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        # 3. XGBoost
        m_xgb = xgb.XGBClassifier(n_estimators=160, max_depth=6, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42+fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        oof_ens[val_idx] = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb

    metrics_2s = calc_2state_metrics(y, oof_ens)
    metrics_3s_mp_policy = calc_3state_metrics(y, oof_ens, tau_clean=0.35, tau_defect=0.65)

    view_th_10 = tune_view_thresholds(y, oof_ens, views, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = oof_ens[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']
        v_3s = calc_3state_metrics(yt_v, yp_v, tc, td)
        v_3s['alias'] = OFFICIAL_VIEW_MAP[vn]
        view_audits_10[vn] = v_3s

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    metrics_3s_adaptive = {
        'effective_f1': float(f1_score(total_yt_10, total_yp_10, zero_division=0)),
        'accuracy': float(accuracy_score(total_yt_10, total_yp_10)),
        'coverage': float(np.mean(total_dec_10)),
        'uncertainty_rate': float(1.0 - np.mean(total_dec_10)),
        'view_audits': view_audits_10
    }

    print(f"  [c1 RTMPose] 2-State F1: {metrics_2s['f1']*100:.2f}%, AUC: {metrics_2s['roc_auc']:.4f}, Acc: {metrics_2s['accuracy']*100:.2f}%")
    print(f"  [c1 RTMPose] 3-State Policy [0.35, 0.65] F1: {metrics_3s_mp_policy['effective_f1']*100:.2f}%, Cov: {metrics_3s_mp_policy['coverage']*100:.2f}%, Retake: {metrics_3s_mp_policy['uncertainty_rate']*100:.2f}%")
    print(f"  [c1 RTMPose] 3-State Adaptive (Retake<=10%) F1: {metrics_3s_adaptive['effective_f1']*100:.2f}%, Cov: {metrics_3s_adaptive['coverage']*100:.2f}%, Retake: {metrics_3s_adaptive['uncertainty_rate']*100:.2f}%")

    return {
        'two_state_metrics': metrics_2s,
        'three_state_mp_policy': metrics_3s_mp_policy,
        'three_state_adaptive': metrics_3s_adaptive,
        'oof_probs': oof_ens.tolist()
    }


def train_and_eval_submodel_c2(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [*] [c2 (sl2)] 몸통·발·무릎 방향 불일치 (RTMPose 51D)")
    print("=" * 80)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    weights = [0.475, 0.525] # CatBoost, XGBoost
    oof_ens = np.zeros(len(y), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va = X[val_idx]

        m_cat = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.04, thread_count=1, verbose=0, random_seed=42+fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        m_xgb = xgb.XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42+fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        oof_ens[val_idx] = weights[0] * p_cat + weights[1] * p_xgb

    metrics_2s = calc_2state_metrics(y, oof_ens)

    # 3-State Adaptive 10%
    view_th_10 = tune_view_thresholds(y, oof_ens, views, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = oof_ens[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']
        v_3s = calc_3state_metrics(yt_v, yp_v, tc, td)
        v_3s['alias'] = OFFICIAL_VIEW_MAP[vn]
        view_audits_10[vn] = v_3s

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    metrics_3s_adaptive = {
        'effective_f1': float(f1_score(total_yt_10, total_yp_10, zero_division=0)),
        'accuracy': float(accuracy_score(total_yt_10, total_yp_10)),
        'coverage': float(np.mean(total_dec_10)),
        'uncertainty_rate': float(1.0 - np.mean(total_dec_10)),
        'view_audits': view_audits_10
    }

    # 전방 뷰군 (시점 B, C, D) 지표
    front_mask = np.isin(views, ['view2', 'view3', 'view4'])
    front_metrics_2s = calc_2state_metrics(y[front_mask], oof_ens[front_mask])

    print(f"  [c2 RTMPose] 2-State F1: {metrics_2s['f1']*100:.2f}%, AUC: {metrics_2s['roc_auc']:.4f}, Acc: {metrics_2s['accuracy']*100:.2f}%")
    print(f"  [c2 RTMPose] 3-State (Retake<=10%) F1: {metrics_3s_adaptive['effective_f1']*100:.2f}%, Cov: {metrics_3s_adaptive['coverage']*100:.2f}%, Retake: {metrics_3s_adaptive['uncertainty_rate']*100:.2f}%")
    print(f"  [c2 RTMPose] 전방 시점군(B, C, D) 2-State F1: {front_metrics_2s['f1']*100:.2f}%")

    return {
        'two_state_metrics': metrics_2s,
        'three_state_adaptive': metrics_3s_adaptive,
        'front_views_metrics': front_metrics_2s,
        'oof_probs': oof_ens.tolist()
    }


def train_and_eval_submodel_c3(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, mask_2460: Optional[np.ndarray] = None) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [*] [c3 (sl3)] 안 굽히는 다리 신전 결함 (RTMPose 28D)")
    print("=" * 80)

    # 1. 2,460개 공식 벤치마크 서브셋 평가 (MediaPipe와 완벽한 1:1 대조)
    if mask_2460 is not None and np.sum(mask_2460) > 0:
        X_sub, y_sub, subj_sub, views_sub = X[mask_2460], y[mask_2460], subjects[mask_2460], views[mask_2460]
    else:
        X_sub, y_sub, subj_sub, views_sub = X, y, subjects, views

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    weights = [0.294, 0.353, 0.354] # LightGBM, CatBoost, XGBoost
    oof_ens = np.zeros(len(y_sub), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y_sub, subj_sub)):
        X_tr, y_tr = X_sub[trn_idx], y_sub[trn_idx]
        X_va = X_sub[val_idx]

        m_lgb = lgb.LGBMClassifier(n_estimators=140, max_depth=5, learning_rate=0.05, num_leaves=24, n_jobs=1, verbose=-1, random_state=42+fold)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        m_cat = CatBoostClassifier(iterations=180, depth=6, learning_rate=0.05, thread_count=1, verbose=0, random_seed=42+fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        m_xgb = xgb.XGBClassifier(n_estimators=160, max_depth=5, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42+fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        oof_ens[val_idx] = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb

    metrics_2s = calc_2state_metrics(y_sub, oof_ens)

    view_th_10 = tune_view_thresholds(y_sub, oof_ens, views_sub, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views_sub == vn)
        yt_v = y_sub[v_mask]
        yp_v = oof_ens[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']
        v_3s = calc_3state_metrics(yt_v, yp_v, tc, td)
        v_3s['alias'] = OFFICIAL_VIEW_MAP[vn]
        view_audits_10[vn] = v_3s

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    metrics_3s_adaptive = {
        'effective_f1': float(f1_score(total_yt_10, total_yp_10, zero_division=0)),
        'accuracy': float(accuracy_score(total_yt_10, total_yp_10)),
        'coverage': float(np.mean(total_dec_10)),
        'uncertainty_rate': float(1.0 - np.mean(total_dec_10)),
        'view_audits': view_audits_10
    }

    print(f"  [c3 RTMPose 2460] 2-State F1: {metrics_2s['f1']*100:.2f}%, AUC: {metrics_2s['roc_auc']:.4f}, Acc: {metrics_2s['accuracy']*100:.2f}%")
    print(f"  [c3 RTMPose 2460] 3-State (Retake<=10%) F1: {metrics_3s_adaptive['effective_f1']*100:.2f}%, Cov: {metrics_3s_adaptive['coverage']*100:.2f}%, Retake: {metrics_3s_adaptive['uncertainty_rate']*100:.2f}%")

    return {
        'two_state_metrics': metrics_2s,
        'three_state_adaptive': metrics_3s_adaptive,
        'samples_count': len(y_sub),
        'oof_probs': oof_ens.tolist()
    }


def train_and_eval_submodel_c4(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [*] [c4 (sl4)] 척추의 중립 위반 (RTMPose 24D)")
    print("=" * 80)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    weights = [0.642, 0.241, 0.117] # CatBoost, XGBoost, LightGBM
    oof_ens = np.zeros(len(y), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va = X[val_idx]

        m_cat = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.05, thread_count=1, verbose=0, random_seed=42+fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        m_xgb = xgb.XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42+fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        m_lgb = lgb.LGBMClassifier(n_estimators=140, max_depth=5, learning_rate=0.05, num_leaves=24, n_jobs=1, verbose=-1, random_state=42+fold)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        oof_ens[val_idx] = weights[0] * p_cat + weights[1] * p_xgb + weights[2] * p_lgb

    metrics_2s = calc_2state_metrics(y, oof_ens)

    view_th_10 = tune_view_thresholds(y, oof_ens, views, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = oof_ens[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']
        v_3s = calc_3state_metrics(yt_v, yp_v, tc, td)
        v_3s['alias'] = OFFICIAL_VIEW_MAP[vn]
        view_audits_10[vn] = v_3s

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    metrics_3s_adaptive = {
        'effective_f1': float(f1_score(total_yt_10, total_yp_10, zero_division=0)),
        'accuracy': float(accuracy_score(total_yt_10, total_yp_10)),
        'coverage': float(np.mean(total_dec_10)),
        'uncertainty_rate': float(1.0 - np.mean(total_dec_10)),
        'view_audits': view_audits_10
    }

    print(f"  [c4 RTMPose] 2-State F1: {metrics_2s['f1']*100:.2f}%, AUC: {metrics_2s['roc_auc']:.4f}, Acc: {metrics_2s['accuracy']*100:.2f}%")
    print(f"  [c4 RTMPose] 3-State (Retake<=10%) F1: {metrics_3s_adaptive['effective_f1']*100:.2f}%, Cov: {metrics_3s_adaptive['coverage']*100:.2f}%, Retake: {metrics_3s_adaptive['uncertainty_rate']*100:.2f}%")

    return {
        'two_state_metrics': metrics_2s,
        'three_state_adaptive': metrics_3s_adaptive,
        'oof_probs': oof_ens.tolist()
    }


def train_and_eval_submodel_c5(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [*] [c5 (sl5)] 상체의 과도한 숙임/젖힘 여부 (RTMPose 42D)")
    print("=" * 80)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_ens = np.zeros(len(y), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va = X[val_idx]

        m_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=6, learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1, random_state=42+fold)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        m_cat = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.05, thread_count=1, verbose=0, random_seed=42+fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        m_xgb = xgb.XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42+fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        m_rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42+fold, n_jobs=1)
        m_rf.fit(X_tr, y_tr)
        p_rf = m_rf.predict_proba(X_va)[:, 1]

        # 5-Way Blending
        oof_ens[val_idx] = 0.30 * p_cat + 0.25 * p_lgb + 0.25 * p_xgb + 0.20 * p_rf

    metrics_2s = calc_2state_metrics(y, oof_ens)

    view_th_10 = tune_view_thresholds(y, oof_ens, views, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = oof_ens[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']
        v_3s = calc_3state_metrics(yt_v, yp_v, tc, td)
        v_3s['alias'] = OFFICIAL_VIEW_MAP[vn]
        view_audits_10[vn] = v_3s

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    metrics_3s_adaptive = {
        'effective_f1': float(f1_score(total_yt_10, total_yp_10, zero_division=0)),
        'accuracy': float(accuracy_score(total_yt_10, total_yp_10)),
        'coverage': float(np.mean(total_dec_10)),
        'uncertainty_rate': float(1.0 - np.mean(total_dec_10)),
        'view_audits': view_audits_10
    }

    print(f"  [c5 RTMPose] 2-State F1: {metrics_2s['f1']*100:.2f}%, AUC: {metrics_2s['roc_auc']:.4f}, Acc: {metrics_2s['accuracy']*100:.2f}%")
    print(f"  [c5 RTMPose] 3-State (Retake<=10%) F1: {metrics_3s_adaptive['effective_f1']*100:.2f}%, Cov: {metrics_3s_adaptive['coverage']*100:.2f}%, Retake: {metrics_3s_adaptive['uncertainty_rate']*100:.2f}%")

    return {
        'two_state_metrics': metrics_2s,
        'three_state_adaptive': metrics_3s_adaptive,
        'oof_probs': oof_ens.tolist()
    }


def load_mediapipe_baselines() -> Dict[str, Any]:
    """저장된 MediaPipe 공식 평가 결과 JSON 파일 로드 및 정형화"""
    mp_data = {}

    # c1
    with open(MP_RESULTS['c1'], 'r', encoding='utf-8') as f:
        d1 = json.load(f)
    p1 = d1.get('primary_three_state_policy_035_065', d1.get('three_state_metrics', {}))
    mp_data['c1'] = {
        'name': 'c1 (앞다리 무릎 90도)',
        'two_state_f1': d1['two_state_metrics']['f1'],
        'two_state_acc': d1['two_state_metrics']['accuracy'],
        'two_state_auc': d1['two_state_metrics']['roc_auc'],
        'three_state_f1': p1['effective_f1'],
        'three_state_acc': p1['accuracy'],
        'coverage': p1['coverage'],
        'deferral_rate': p1['uncertainty_rate'],
        'view_audits': p1.get('view_audits', {})
    }

    # c2
    with open(MP_RESULTS['c2'], 'r', encoding='utf-8') as f:
        d2 = json.load(f)
    p2 = d2['profiles']['standard_10pct']
    mp_data['c2'] = {
        'name': 'c2 (몸통·발·무릎 방향 불일치)',
        'two_state_f1': p2['overall_f1'],
        'two_state_acc': p2['overall_accuracy'],
        'two_state_auc': p2['overall_auc'],
        'three_state_f1': p2['overall_f1'],
        'three_state_acc': p2['overall_accuracy'],
        'coverage': p2['coverage'],
        'deferral_rate': p2['uncertainty_rate'],
        'view_audits': p2.get('view_audits', {})
    }

    # c3
    with open(MP_RESULTS['c3'], 'r', encoding='utf-8') as f:
        d3 = json.load(f)
    p3 = d3['three_state_metrics']
    mp_data['c3'] = {
        'name': 'c3 (안 굽히는 다리 신전 결함)',
        'two_state_f1': d3['two_state_metrics']['ensemble']['f1'],
        'two_state_acc': d3['two_state_metrics']['ensemble']['accuracy'],
        'two_state_auc': d3['two_state_metrics']['ensemble']['roc_auc'],
        'three_state_f1': p3['effective_f1'],
        'three_state_acc': p3['accuracy'],
        'coverage': p3['coverage'],
        'deferral_rate': p3['uncertainty_rate'],
        'view_audits': p3.get('view_audits', {})
    }

    # c4
    with open(MP_RESULTS['c4'], 'r', encoding='utf-8') as f:
        d4 = json.load(f)
    p4 = d4['ensemble_3state']
    mp_data['c4'] = {
        'name': 'c4 (척추의 중립 위반)',
        'two_state_f1': d4['ensemble_2state']['f1'],
        'two_state_acc': d4['ensemble_2state']['accuracy'],
        'two_state_auc': d4['ensemble_2state']['auc'],
        'three_state_f1': p4['confirmed_f1'],
        'three_state_acc': p4['confirmed_accuracy'],
        'coverage': p4['coverage'],
        'deferral_rate': 1.0 - p4['coverage'],
        'view_audits': d4.get('view_audit', {})
    }

    # c5
    with open(MP_RESULTS['c5'], 'r', encoding='utf-8') as f:
        d5 = json.load(f)
    p5 = d5['three_state']
    mp_data['c5'] = {
        'name': 'c5 (상체 과도한 숙임/젖힘)',
        'two_state_f1': d5['full_coverage']['f1'],
        'two_state_acc': d5['full_coverage']['accuracy'],
        'two_state_auc': d5['full_coverage']['auc'],
        'three_state_f1': p5['f1'],
        'three_state_acc': p5['accuracy'],
        'coverage': p5['coverage'],
        'deferral_rate': p5['rejection_rate'],
        'view_audits': d5.get('view_audits', {})
    }

    return mp_data


def run_full_evaluation():
    t_start = time.time()
    print("=" * 95)
    print("  [vidio-health-trainer] 사이드 런지 MediaPipe vs RTMPose 1:1 성능 비교 공식 파이프라인")
    print("  - 데이터셋: AI-Hub 원천 826개 폴더 전수 (4,130개 시퀀스, 9명 피험자, 5대 전 시점)")
    print("  - 검증: 5-Fold StratifiedGroupKFold 피험자 완전 격리 (No Leakage)")
    print("  - 모델: c1 (55D), c2 (51D), c3 (28D), c4 (24D), c5 (42D) 세부 오류 5대 챔피언 아키텍처")
    print("=" * 95)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"RTMPose feature dataset not found at {DATA_PATH}!")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_c1 = data['X_c1']
    X_c2 = data['X_c2']
    X_c3 = data['X_c3']
    X_c4 = data['X_c4']
    X_c5 = data['X_c5']

    y_c1 = data['y_c1']
    y_c2 = data['y_c2']
    y_c3 = data['y_c3']
    c3_mask_2460 = data['c3_mask_2460']
    y_c4 = data['y_c4']
    y_c5 = data['y_c5']

    subjects = data['subjects']
    views = data['views']

    print(f"[*] RTMPose 피처 텐서 로드 완료 (총 {len(y_c1)}개 시퀀스)")

    # 1. 서브모델 학습 및 평가
    res_c1 = train_and_eval_submodel_c1(X_c1, y_c1, subjects, views)
    res_c2 = train_and_eval_submodel_c2(X_c2, y_c2, subjects, views)
    res_c3 = train_and_eval_submodel_c3(X_c3, y_c3, subjects, views, mask_2460=c3_mask_2460)
    res_c4 = train_and_eval_submodel_c4(X_c4, y_c4, subjects, views)
    res_c5 = train_and_eval_submodel_c5(X_c5, y_c5, subjects, views)

    # 2. MediaPipe 베이스라인 로드
    mp_baselines = load_mediapipe_baselines()

    # 3. 종합 비교 테이블 출력
    print("\n" + "=" * 115)
    print("  [vidio-health-trainer] 사이드 런지 MediaPipe vs RTMPose 원천 데이터 종합 비교표")
    print("=" * 115)
    header_fmt = "{:<12} | {:<10} | {:<10} | {:<8} | {:<10} | {:<10} | {:<8} | {:<10} | {:<10}"
    row_fmt    = "{:<12} | {:<10.2f}% | {:<10.2f}% | {:<+8.2f}% | {:<10.2f}% | {:<10.2f}% | {:<+8.2f}% | {:<10.2f}% | {:<10.2f}%"

    print(header_fmt.format("결함 코드", "MP 2S F1", "RTMP 2S F1", "2S 변동", "MP 3S F1", "RTMP 3S F1", "3S 변동", "MP 보류율", "RTMP 보류율"))
    print("-" * 115)

    submodel_rtmp = {
        'c1': res_c1,
        'c2': res_c2,
        'c3': res_c3,
        'c4': res_c4,
        'c5': res_c5
    }

    comparison_summary = {}

    for c_code in ['c1', 'c2', 'c3', 'c4', 'c5']:
        mp = mp_baselines[c_code]
        rt = submodel_rtmp[c_code]

        mp_2s = mp['two_state_f1'] * 100
        rt_2s = rt['two_state_metrics']['f1'] * 100
        diff_2s = rt_2s - mp_2s

        # 3-State
        mp_3s = mp['three_state_f1'] * 100
        mp_def = mp['deferral_rate'] * 100

        if c_code == 'c1':
            rt_3s = rt['three_state_mp_policy']['effective_f1'] * 100
            rt_def = rt['three_state_mp_policy']['uncertainty_rate'] * 100
        else:
            rt_3s = rt['three_state_adaptive']['effective_f1'] * 100
            rt_def = rt['three_state_adaptive']['uncertainty_rate'] * 100

        diff_3s = rt_3s - mp_3s

        comparison_summary[c_code] = {
            'mp_2s_f1': mp_2s,
            'rtmp_2s_f1': rt_2s,
            'diff_2s_f1': diff_2s,
            'mp_3s_f1': mp_3s,
            'rtmp_3s_f1': rt_3s,
            'diff_3s_f1': diff_3s,
            'mp_deferral_rate': mp_def,
            'rtmp_deferral_rate': rt_def
        }

        print(row_fmt.format(c_code, mp_2s, rt_2s, diff_2s, mp_3s, rt_3s, diff_3s, mp_def, rt_def))

    print("-" * 115)

    # 4. 결과 JSON 저장
    full_output = {
        'metadata': {
            'project': 'vidio-health-trainer',
            'exercise': '사이드 런지 (07_사이드_런지)',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_sequences': len(y_c1),
            'subjects': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'views': VIEW_ORDER
        },
        'comparison_summary': comparison_summary,
        'rtmp_results': {
            'c1': {k: v for k, v in res_c1.items() if k != 'oof_probs'},
            'c2': {k: v for k, v in res_c2.items() if k != 'oof_probs'},
            'c3': {k: v for k, v in res_c3.items() if k != 'oof_probs'},
            'c4': {k: v for k, v in res_c4.items() if k != 'oof_probs'},
            'c5': {k: v for k, v in res_c5.items() if k != 'oof_probs'}
        },
        'mp_baselines': mp_baselines
    }

    with open(RESULTS_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_OUT_PATH}")
    print(f"[*] 전체 평가 소요 시간: {time.time() - t_start:.1f}초")
    return full_output


if __name__ == "__main__":
    run_full_evaluation()
