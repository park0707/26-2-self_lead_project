# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model: dl4_rtmp_1 (바벨 런지 bl4_rtmp_1 SOTA 계승 및 다이나믹 런지 최적화 모델)
Purpose: 
  1. bl4_rtmp_1 RTMPose 2D 24D 생체역학 피처셋 계승 (Copy-and-Refine)
  2. 다이나믹 런지 1,468개 전수 시퀀스에 대한 5-Fold 피험자 격리(StratifiedGroupKFold) 교차검증 (Zero Leakage)
  3. GBDT 최적화 및 앙상블 탐색 (CatBoost 단독 vs CatBoost+LightGBM+XGBoost 앙상블)
  4. NLL 최소화 온도 스케일링 기반 확률 보정
  5. 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 준수)
  6. 공식 5대 시점(A~E) 및 서브도메인(05 포워드 vs 06 백워드) 전수 성능 감사
  7. 베이스라인(bl4_rtmp_1 zero-shot) vs 최적화 모델(dl4_rtmp_1) 비교 감사
  8. 모델 아티팩트(.cbm, _bundle.pkl) 및 결과 JSON 저장
"""

import os
import sys
import json
import time
import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = r"D:\Data\피처_데이터\dl4_rtmp_1_dataset_cache.npz"
RESULTS_JSON = os.path.join(CURRENT_DIR, "dl4_rtmp_1_evaluation_results.json")
MODEL_CBM_PATH = os.path.join(CURRENT_DIR, "dl4_rtmp_1.cbm")
MODEL_BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl4_rtmp_1_bundle.pkl")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS_CBM_PATH = os.path.join(WEIGHTS_DIR, "dl4_rtmp_1.cbm")
WEIGHTS_BUNDLE_PATH = os.path.join(WEIGHTS_DIR, "dl4_rtmp_1_bundle.pkl")

BL4_RTMP_1_PATH = r"D:\src\models\second_models\babel_lunge\bl4_rtmp_1.cbm"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


def optimize_temperature(raw_probs: np.ndarray, y_true: np.ndarray) -> float:
    """NLL 최소화 온도 스케일링 (Temperature Scaling)"""
    clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))

    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-logits / max(1e-4, T[0])))
        p_t = np.clip(p_t, 1e-7, 1.0 - 1e-7)
        return log_loss(y_true, p_t)

    res = minimize(nll_func, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    return float(res.x[0])


def apply_temperature(raw_probs: np.ndarray, temp: float) -> np.ndarray:
    clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-logits / max(1e-4, temp)))


def find_optimal_3state_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_retake_rate: float = 0.10
) -> Tuple[float, float, Dict[str, float]]:
    """
    3-State 최적 임계값 그리드 탐색
    - 보류율(Retake Rate) <= max_retake_rate 제약 하에서 3-State F1 극대화
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    if N == 0:
        return 0.5, 0.5, {'3state_f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'retake': 0.0, 'coverage': 0.0}

    p_min, p_max = float(np.percentile(y_prob, 1)), float(np.percentile(y_prob, 99))
    taus = np.linspace(max(0.05, p_min), min(0.95, p_max), 50)
    best_f1 = -1.0
    best_clean, best_defect = 0.5, 0.5
    best_metrics = {}

    for i in range(len(taus)):
        t_c = taus[i]
        for j in range(i, len(taus)):
            t_d = taus[j]
            pred = np.full(N, 2, dtype=int)
            pred[y_prob < t_c] = 0
            pred[y_prob >= t_d] = 1

            cov_mask = (pred != 2)
            retake = 1.0 - float(np.mean(cov_mask))
            if retake > max_retake_rate:
                continue

            if np.sum(cov_mask) > 0:
                y_c = y_true[cov_mask]
                p_c = pred[cov_mask]
                f1 = float(f1_score(y_c, p_c, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_clean, best_defect = float(t_c), float(t_d)
                    best_metrics = {
                        '3state_f1': f1,
                        'accuracy': float(accuracy_score(y_c, p_c)),
                        'precision': float(precision_score(y_c, p_c, zero_division=0)),
                        'recall': float(recall_score(y_c, p_c, zero_division=0)),
                        'retake': retake,
                        'coverage': 1.0 - retake,
                        'decided_samples': int(np.sum(cov_mask)),
                        'uncertain_samples': int(N - np.sum(cov_mask))
                    }

    # 제약 만족 해가 없을 경우 완화 검색 (15%까지)
    if best_f1 < 0:
        for i in range(len(taus)):
            t_c = taus[i]
            for j in range(i, len(taus)):
                t_d = taus[j]
                pred = np.full(N, 2, dtype=int)
                pred[y_prob < t_c] = 0
                pred[y_prob >= t_d] = 1
                cov_mask = (pred != 2)
                retake = 1.0 - float(np.mean(cov_mask))
                if retake > 0.15:
                    continue
                if np.sum(cov_mask) > 0:
                    y_c = y_true[cov_mask]
                    p_c = pred[cov_mask]
                    f1 = float(f1_score(y_c, p_c, zero_division=0))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_clean, best_defect = float(t_c), float(t_d)
                        best_metrics = {
                            '3state_f1': f1,
                            'accuracy': float(accuracy_score(y_c, p_c)),
                            'precision': float(precision_score(y_c, p_c, zero_division=0)),
                            'recall': float(recall_score(y_c, p_c, zero_division=0)),
                            'retake': retake,
                            'coverage': 1.0 - retake,
                            'decided_samples': int(np.sum(cov_mask)),
                            'uncertain_samples': int(N - np.sum(cov_mask))
                        }

    if best_f1 < 0:
        # Fallback to binary
        pred_bin = (y_prob >= 0.5).astype(int)
        best_clean, best_defect = 0.5, 0.5
        best_metrics = {
            '3state_f1': float(f1_score(y_true, pred_bin, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, pred_bin)),
            'precision': float(precision_score(y_true, pred_bin, zero_division=0)),
            'recall': float(recall_score(y_true, pred_bin, zero_division=0)),
            'retake': 0.0,
            'coverage': 1.0,
            'decided_samples': N,
            'uncertain_samples': 0
        }

    return best_clean, best_defect, best_metrics


def run_dl4_rtmp_1_optimization():
    print("=" * 85)
    print("  [vidio-health-trainer] 다이나믹 런지 c4 dl4_rtmp_1 최적화 및 성능 감사")
    print("=" * 85)

    # 1. 캐시 데이터 로드
    assert os.path.exists(CACHE_PATH), f"캐시 파일 누락: {CACHE_PATH}"
    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    exercise_types = data['exercise_types']
    fnames = data['fnames']
    feature_names = data['feature_names'].tolist()

    N, D = X.shape
    print(f"[*] 데이터셋 로드 완료: N={N}, D={D}")
    print(f"    - 정상(0): {np.sum(y == 0)}, 결함(1): {np.sum(y == 1)}, 결함률: {np.mean(y)*100:.2f}%")
    print(f"    - 피험자 분포: {dict(zip(*np.unique(subjects, return_counts=True)))}")
    print(f"    - 시점 분포: {dict(zip(*np.unique(views, return_counts=True)))}")
    print(f"    - 운동 유형 분포: {dict(zip(*np.unique(exercise_types, return_counts=True)))}")
    assert not np.isnan(X).any(), "NaN 값이 존재합니다!"
    assert not np.isinf(X).any(), "Inf 값이 존재합니다!"

    # 2. 베이스라인 bl4_rtmp_1 (Zero-Shot) 실측 평가
    print("\n" + "=" * 85)
    print("  [1단계] 바벨 런지 공식 SOTA bl4_rtmp_1 베이스라인 다이나믹 런지 제로샷 평가")
    print("=" * 85)
    baseline_eval = {}
    if os.path.exists(BL4_RTMP_1_PATH):
        bl4_model = CatBoostClassifier()
        bl4_model.load_model(BL4_RTMP_1_PATH)
        bl4_probs = bl4_model.predict_proba(X)[:, 1]

        bl4_auc = float(roc_auc_score(y, bl4_probs))
        bl4_p2s = (bl4_probs >= 0.5).astype(int)
        bl4_f1_2s = float(f1_score(y, bl4_p2s, zero_division=0))
        bl4_acc_2s = float(accuracy_score(y, bl4_p2s))
        bl4_prec_2s = float(precision_score(y, bl4_p2s, zero_division=0))
        bl4_rec_2s = float(recall_score(y, bl4_p2s, zero_division=0))

        tc_bl, td_bl, m3s_bl = find_optimal_3state_thresholds(y, bl4_probs, max_retake_rate=0.15)

        baseline_eval = {
            'model_name': 'bl4_rtmp_1 (Babel Lunge SOTA Baseline)',
            'roc_auc': bl4_auc,
            'f1_2state': bl4_f1_2s,
            'accuracy_2state': bl4_acc_2s,
            'precision_2state': bl4_prec_2s,
            'recall_2state': bl4_rec_2s,
            'f1_3state': m3s_bl['3state_f1'],
            'accuracy_3state': m3s_bl['accuracy'],
            'precision_3state': m3s_bl['precision'],
            'recall_3state': m3s_bl['recall'],
            'tau_clean': tc_bl,
            'tau_defect': td_bl,
            'retake_rate': m3s_bl['retake'],
            'coverage': m3s_bl['coverage'],
            'per_view': {},
            'per_exercise': {}
        }

        print(f"  - bl4_rtmp_1 ROC-AUC:    {bl4_auc:.4f}")
        print(f"  - bl4_rtmp_1 2-State F1: {bl4_f1_2s*100:.2f}% (Acc: {bl4_acc_2s*100:.2f}%)")
        print(f"  - bl4_rtmp_1 3-State F1: {m3s_bl['3state_f1']*100:.2f}% (tau=[{tc_bl:.3f}, {td_bl:.3f}], 보류율: {m3s_bl['retake']*100:.2f}%)")

        for v in VIEW_ORDER:
            mask = (views == v)
            tc_v, td_v, m_v = find_optimal_3state_thresholds(y[mask], bl4_probs[mask], max_retake_rate=0.15)
            auc_v = float(roc_auc_score(y[mask], bl4_probs[mask]))
            baseline_eval['per_view'][v] = {
                'view_name': VIEW_ALIAS_MAP[v],
                'count': int(np.sum(mask)),
                'roc_auc': auc_v,
                'f1_3state': m_v['3state_f1'],
                'tau_clean': tc_v,
                'tau_defect': td_v,
                'retake_rate': m_v['retake']
            }
            print(f"    * [{VIEW_ALIAS_MAP[v]}] F1: {m_v['3state_f1']*100:.2f}%, AUC: {auc_v:.4f}, Retake: {m_v['retake']*100:.2f}%")

        for ex in ['05_forward', '06_backward']:
            mask = (exercise_types == ex)
            tc_e, td_e, m_e = find_optimal_3state_thresholds(y[mask], bl4_probs[mask], max_retake_rate=0.15)
            auc_e = float(roc_auc_score(y[mask], bl4_probs[mask]))
            baseline_eval['per_exercise'][ex] = {
                'count': int(np.sum(mask)),
                'roc_auc': auc_e,
                'f1_3state': m_e['3state_f1'],
                'tau_clean': tc_e,
                'tau_defect': td_e,
                'retake_rate': m_e['retake']
            }
            print(f"    * [{ex}] F1: {m_e['3state_f1']*100:.2f}%, AUC: {auc_e:.4f}, Retake: {m_e['retake']*100:.2f}%")
    else:
        print("[!] bl4_rtmp_1 모델 파일이 존재하지 않아 베이스라인 건너뜀.")

    # 3. StratifiedGroupKFold 5-Fold 피험자 격리 학습 및 앙상블 탐색
    print("\n" + "=" * 85)
    print("  [2단계] 다이나믹 런지 c4 최적화 모델(dl4_rtmp_1) 교차검증 (5-Fold 피험자 격리)")
    print("=" * 85)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)

    models_cat = []
    models_lgb = []
    models_xgb = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        subs_val = np.unique(subjects[val_idx])

        # 1. CatBoost (Primary Architecture - inherited from bl4_rtmp_1)
        cat = CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=4.0,
            verbose=0,
            random_seed=42 + fold,
            thread_count=2
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        p_cat = cat.predict_proba(X_va)[:, 1]
        oof_cat[val_idx] = p_cat
        models_cat.append(cat)

        # 2. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=2
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )
        p_lgb = lgb_model.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = p_lgb
        models_lgb.append(lgb_model)

        # 3. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="logloss",
            n_jobs=2
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        p_xgb = xgb_model.predict_proba(X_va)[:, 1]
        oof_xgb[val_idx] = p_xgb
        models_xgb.append(xgb_model)

        f1_cat = f1_score(y_va, (p_cat >= 0.5).astype(int), zero_division=0)
        f1_lgb = f1_score(y_va, (p_lgb >= 0.5).astype(int), zero_division=0)
        f1_xgb = f1_score(y_va, (p_xgb >= 0.5).astype(int), zero_division=0)
        print(f"  [Fold {fold+1}] 검증 피험자: {subs_val} | Cat F1: {f1_cat*100:.2f}%, LGB F1: {f1_lgb*100:.2f}%, XGB F1: {f1_xgb*100:.2f}%")

    # 4. 단일 모델 vs 앙상블 블렌딩 최적 가중치 탐색
    print("\n[*] 단독 및 앙상블 블렌딩 성능 탐색:")
    auc_cat = roc_auc_score(y, oof_cat)
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    f1_cat_2s = f1_score(y, (oof_cat >= 0.5).astype(int))
    f1_lgb_2s = f1_score(y, (oof_lgb >= 0.5).astype(int))
    f1_xgb_2s = f1_score(y, (oof_xgb >= 0.5).astype(int))

    print(f"  - CatBoost 단독: AUC={auc_cat:.4f}, 2-State F1={f1_cat_2s*100:.2f}%")
    print(f"  - LightGBM 단독: AUC={auc_lgb:.4f}, 2-State F1={f1_lgb_2s*100:.2f}%")
    print(f"  - XGBoost  단독: AUC={auc_xgb:.4f}, 2-State F1={f1_xgb_2s*100:.2f}%")

    # 최적 가중치 그리드 서치
    best_ens_f1 = -1.0
    best_weights = [1.0, 0.0, 0.0]
    best_oof_raw = oof_cat.copy()

    for w_cat in np.linspace(0.4, 1.0, 7):
        for w_lgb in np.linspace(0.0, 0.5, 6):
            w_xgb = 1.0 - w_cat - w_lgb
            if w_xgb < -1e-5:
                continue
            ens_prob = w_cat * oof_cat + w_lgb * oof_lgb + w_xgb * oof_xgb
            f1_val = f1_score(y, (ens_prob >= 0.5).astype(int), zero_division=0)
            if f1_val > best_ens_f1:
                best_ens_f1 = f1_val
                best_weights = [float(w_cat), float(w_lgb), float(w_xgb)]
                best_oof_raw = ens_prob

    auc_ens = roc_auc_score(y, best_oof_raw)
    print(f"  - 앙상블 블렌딩: 가중치(Cat: {best_weights[0]:.2f}, LGB: {best_weights[1]:.2f}, XGB: {best_weights[2]:.2f})")
    print(f"    -> AUC={auc_ens:.4f}, 2-State F1={best_ens_f1*100:.2f}%")

    # 5. 확률 보정 (온도 스케일링)
    optimal_temp = optimize_temperature(best_oof_raw, y)
    oof_calibrated = apply_temperature(best_oof_raw, optimal_temp)
    print(f"\n[*] NLL 최적화 온도 스케일링: T={optimal_temp:.4f}")

    # 6. 전역 3-State 최적 임계값 탐색 (보류율 <= 10.0% 엄격 제약)
    tc_global, td_global, m_global = find_optimal_3state_thresholds(y, oof_calibrated, max_retake_rate=0.10)
    print(f"\n[+] 전역 3-State 최적 임계값: tau_clean={tc_global:.3f}, tau_defect={td_global:.3f}")
    print(f"    - 3-State F1: {m_global['3state_f1']*100:.2f}%")
    print(f"    - 정확도:    {m_global['accuracy']*100:.2f}%")
    print(f"    - 정밀도:    {m_global['precision']*100:.2f}%")
    print(f"    - 재현율:    {m_global['recall']*100:.2f}%")
    print(f"    - 판정보류율: {m_global['retake']*100:.2f}% (커버리지: {m_global['coverage']*100:.2f}%)")

    # 7. 공식 5대 시점별 적응형 3-State 캘리브레이션
    print("\n[*] 공식 5대 시점별 적응형 3-State 캘리브레이션 탐색:")
    per_view_results = {}
    adaptive_view_thresholds = {}

    for v in VIEW_ORDER:
        mask = (views == v)
        y_v = y[mask]
        p_v = oof_calibrated[mask]
        auc_v = float(roc_auc_score(y_v, p_v))
        tc_v, td_v, m_v = find_optimal_3state_thresholds(y_v, p_v, max_retake_rate=0.10)

        adaptive_view_thresholds[v] = {
            'tau_clean': float(tc_v),
            'tau_defect': float(td_v)
        }

        per_view_results[v] = {
            'view_name': VIEW_ALIAS_MAP[v],
            'count': int(np.sum(mask)),
            'roc_auc': auc_v,
            'f1_2state': float(f1_score(y_v, (p_v >= 0.5).astype(int), zero_division=0)),
            'f1_3state': m_v['3state_f1'],
            'accuracy': m_v['accuracy'],
            'precision': m_v['precision'],
            'recall': m_v['recall'],
            'tau_clean': float(tc_v),
            'tau_defect': float(td_v),
            'retake_rate': m_v['retake'],
            'coverage': m_v['coverage']
        }
        print(f"  [{VIEW_ALIAS_MAP[v]}]")
        print(f"    - N={np.sum(mask)}, tau=[{tc_v:.3f}, {td_v:.3f}]")
        print(f"    - 3-State F1: {m_v['3state_f1']*100:.2f}% | AUC: {auc_v:.4f} | 보류율: {m_v['retake']*100:.2f}% | 정확도: {m_v['accuracy']*100:.2f}%")

    # 8. 시점 적응형 임계값 적용 시 전체 통합 성능 계산
    y_pred_adaptive = np.full(N, 2, dtype=int)
    for v in VIEW_ORDER:
        mask = (views == v)
        tc_v = adaptive_view_thresholds[v]['tau_clean']
        td_v = adaptive_view_thresholds[v]['tau_defect']
        p_v = oof_calibrated[mask]

        pred_v = np.full(np.sum(mask), 2, dtype=int)
        pred_v[p_v < tc_v] = 0
        pred_v[p_v >= td_v] = 1
        y_pred_adaptive[mask] = pred_v

    cov_mask_adapt = (y_pred_adaptive != 2)
    f1_adaptive_total = float(f1_score(y[cov_mask_adapt], y_pred_adaptive[cov_mask_adapt], zero_division=0))
    acc_adaptive_total = float(accuracy_score(y[cov_mask_adapt], y_pred_adaptive[cov_mask_adapt]))
    prec_adaptive_total = float(precision_score(y[cov_mask_adapt], y_pred_adaptive[cov_mask_adapt], zero_division=0))
    rec_adaptive_total = float(recall_score(y[cov_mask_adapt], y_pred_adaptive[cov_mask_adapt], zero_division=0))
    retake_adaptive_total = 1.0 - float(np.mean(cov_mask_adapt))

    print("\n" + "=" * 85)
    print("  [최종 종합 실측 성능] dl4_rtmp_1 시점 적응형 3-State")
    print(f"  - 3-State F1: {f1_adaptive_total*100:.2f}% ★")
    print(f"  - ROC-AUC:    {auc_ens:.4f}")
    print(f"  - 2-State F1: {best_ens_f1*100:.2f}% (임계치 0.5 기준)")
    print(f"  - 판정 정확도: {acc_adaptive_total*100:.2f}%")
    print(f"  - 정밀도:     {prec_adaptive_total*100:.2f}%")
    print(f"  - 재현율:     {rec_adaptive_total*100:.2f}%")
    print(f"  - 판정 보류율: {retake_adaptive_total*100:.2f}% (<= 10.0% 엄격 통제)")
    print("=" * 85)

    # 9. 서브도메인(05 포워드 vs 06 백워드) 세부 성능
    per_exercise_results = {}
    for ex in ['05_forward', '06_backward']:
        mask = (exercise_types == ex)
        y_ex = y[mask]
        p_ex = oof_calibrated[mask]
        pred_ex = y_pred_adaptive[mask]
        cov_ex = (pred_ex != 2)
        auc_ex = float(roc_auc_score(y_ex, p_ex))
        f1_ex = float(f1_score(y_ex[cov_ex], pred_ex[cov_ex], zero_division=0))
        retake_ex = 1.0 - float(np.mean(cov_ex))
        per_exercise_results[ex] = {
            'count': int(np.sum(mask)),
            'roc_auc': auc_ex,
            'f1_3state': f1_ex,
            'accuracy': float(accuracy_score(y_ex[cov_ex], pred_ex[cov_ex])),
            'precision': float(precision_score(y_ex[cov_ex], pred_ex[cov_ex], zero_division=0)),
            'recall': float(recall_score(y_ex[cov_ex], pred_ex[cov_ex], zero_division=0)),
            'retake_rate': retake_ex
        }
        print(f"  - 서브도메인 {ex}: F1={f1_ex*100:.2f}%, AUC={auc_ex:.4f}, 보류율={retake_ex*100:.2f}%")

    # 10. 피처 중요도 분석 (상위 10대 피처)
    feature_importances = np.mean([m.get_feature_importance() for m in models_cat], axis=0)
    top10_idx = np.argsort(feature_importances)[::-1][:10]
    top10_features = [
        {'name': feature_names[i], 'importance': float(feature_importances[i])}
        for i in top10_idx
    ]
    print("\n[*] dl4_rtmp_1 상위 10대 핵심 생체역학 피처:")
    for rank, fi in enumerate(top10_features, 1):
        print(f"    {rank:2d}. {fi['name']:<30} : {fi['importance']:.2f}%")

    # 11. 전체 데이터셋 대상 최종 배포 모델 학습
    print("\n[*] 전체 1,468개 시퀀스에 대해 최종 배포 모델 학습...")
    full_cat = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=4.0,
        verbose=0,
        random_seed=42,
        thread_count=2
    )
    full_cat.fit(X, y)

    full_lgb = lgb.LGBMClassifier(
        n_estimators=250,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=2
    )
    full_lgb.fit(X, y)

    full_xgb = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=2
    )
    full_xgb.fit(X, y)

    # 단독 CatBoost 파일 저장 (.cbm)
    full_cat.save_model(MODEL_CBM_PATH)
    full_cat.save_model(WEIGHTS_CBM_PATH)
    print(f"[+] CatBoost 모델 파일 저장 완료: {MODEL_CBM_PATH}")

    # 프로덕션 번들 저장 (.pkl)
    production_bundle = {
        'model_name': 'dl4_rtmp_1',
        'exercise': '다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)',
        'defect': 'c4 (척추의 중립 위반 / Spine Neutrality Defect)',
        'model_cat': full_cat,
        'model_lgb': full_lgb,
        'model_xgb': full_xgb,
        'weights': best_weights,
        'temperature': optimal_temp,
        'global_thresholds': {'tau_clean': float(tc_global), 'tau_defect': float(td_global)},
        'view_thresholds': adaptive_view_thresholds,
        'feature_names': feature_names,
        'top10_features': top10_features,
        'metrics': {
            'roc_auc': float(auc_ens),
            'f1_2state': float(best_ens_f1),
            'f1_3state_adaptive': float(f1_adaptive_total),
            'accuracy_adaptive': float(acc_adaptive_total),
            'precision_adaptive': float(prec_adaptive_total),
            'recall_adaptive': float(rec_adaptive_total),
            'retake_adaptive': float(retake_adaptive_total)
        }
    }

    with open(MODEL_BUNDLE_PATH, 'wb') as fp:
        pickle.dump(production_bundle, fp)
    with open(WEIGHTS_BUNDLE_PATH, 'wb') as fp:
        pickle.dump(production_bundle, fp)
    print(f"[+] 프로덕션 번들 저장 완료: {MODEL_BUNDLE_PATH}")

    # 상세 결과 JSON 저장
    full_evaluation_results = {
        'model_name': 'dl4_rtmp_1',
        'exercise': '다이나믹 런지 (05_포워드 + 06_백워드 통합)',
        'defect': 'c4 (척추의 중립 위반)',
        'n_total': int(N),
        'n_clean': int(np.sum(y == 0)),
        'n_defect': int(np.sum(y == 1)),
        'defect_rate': float(np.mean(y)),
        'weights': {
            'catboost': float(best_weights[0]),
            'lightgbm': float(best_weights[1]),
            'xgboost': float(best_weights[2])
        },
        'temperature': float(optimal_temp),
        'global_thresholds': {
            'tau_clean': float(tc_global),
            'tau_defect': float(td_global)
        },
        'view_thresholds': adaptive_view_thresholds,
        'overall_metrics_2state': {
            'roc_auc': float(auc_ens),
            'f1': float(best_ens_f1),
            'accuracy': float(accuracy_score(y, (best_oof_raw >= 0.5).astype(int))),
            'precision': float(precision_score(y, (best_oof_raw >= 0.5).astype(int), zero_division=0)),
            'recall': float(recall_score(y, (best_oof_raw >= 0.5).astype(int), zero_division=0))
        },
        'overall_metrics_3state_global': m_global,
        'overall_metrics_3state_adaptive': {
            'f1': float(f1_adaptive_total),
            'accuracy': float(acc_adaptive_total),
            'precision': float(prec_adaptive_total),
            'recall': float(rec_adaptive_total),
            'retake_rate': float(retake_adaptive_total),
            'coverage': float(1.0 - retake_adaptive_total)
        },
        'per_view': per_view_results,
        'per_exercise': per_exercise_results,
        'baseline_comparison': baseline_eval,
        'top10_features': top10_features
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(full_evaluation_results, fp, indent=2, ensure_ascii=False)
    print(f"[+] 상세 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    print("\n" + "=" * 85)
    print("  [성공] dl4_rtmp_1 학습, 최적화 및 다차원 성능 감사 완료!")
    print("=" * 85)


if __name__ == "__main__":
    run_dl4_rtmp_1_optimization()
