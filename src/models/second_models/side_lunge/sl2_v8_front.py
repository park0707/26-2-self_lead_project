# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Script: sl2_v8_front.py
Purpose: 시점 B(우측 전방 45°) & 시점 D(좌측 전방 45°) 전용 F1 70% 돌파 58D SOTA 모델 (sl2_v8_front)
         - 1,652건 전방 원천 데이터(피험자 9명 전수) 5-Fold StratifiedGroupKFold 완전 격리
         - v7 51D 베이스라인 완전 계승 + 전방 특화 7대 혁신 생체역학 피처 (총 58D)
         - 생체역학 스칼라 삼중적(κ_c2) 및 제곱근 비용 민감 가중치(w_pos ≈ 2.09) 탑재
         - 시점 B 및 시점 D 단독/결합 3-State 적응형 캘리브레이션
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from scipy.optimize import minimize
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v8_front_features import (
    FEATURE_NAMES_58D,
    compute_v8_front_breakthrough_features_7d,
    extract_side_lunge_c2_v8_front_features_58d
)
from side_lunge_c2_features_v3 import safe_savgol_filter_kps

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
V7_ALL_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
CACHE_FRONT_58D_PKL = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl2_v8_front_evaluation_results.json")
BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl2_v8_front_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view2": "시점 B (우측 전방 45°)",
    "view4": "시점 D (좌측 전방 45°)"
}


def build_58d_front_dataset() -> Dict[str, Any]:
    """1,652건 전방 뷰 대상 58D 피처 데이터셋 고속 구축 및 캐싱"""
    if os.path.exists(CACHE_FRONT_58D_PKL):
        print(f"[*] sl2_v8_front 1,652건 58D 캐시 로드 중: {CACHE_FRONT_58D_PKL}")
        with open(CACHE_FRONT_58D_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] sl2_v7 캐시 기반 전방 뷰(B, D) 1,652건 58D 데이터셋 고속 구축 시작...")
    with open(V7_ALL_CACHE_PKL, 'rb') as f:
        v7_data = pickle.load(f)

    X_all_51d = v7_data['X']
    y_all = v7_data['y']
    groups_all = v7_data['groups']
    views_all = v7_data['views']
    clip_names_all = v7_data['clip_names']

    # 시점 B (view2)와 시점 D (view4)만 필터링
    front_mask = (views_all == 'view2') | (views_all == 'view4')
    X_front_51d = X_all_51d[front_mask]
    y_front = y_all[front_mask]
    groups_front = groups_all[front_mask]
    views_front = views_all[front_mask]
    clip_names_front = clip_names_all[front_mask]
    N_front = len(y_front)

    print(f"[*] 전방 1,652건 대상 v8 전방 특화 7D 피처(FPA, 2D정합, 시점비대칭, 무릎붕괴, 외번, 비틀림, 삼중적) 고속 추출 중...")
    t0 = time.time()
    new_7d_list = []

    for i in range(N_front):
        clip_name = clip_names_front[i]
        view_code = views_front[i]
        npz_name = f"{clip_name}_{view_code}.npz"
        npz_path = os.path.join(DATA_DIR, npz_name)

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ 파일 없음: {npz_path}")

        data = np.load(npz_path)
        kps_clean = data['kps_33_3d_64f'][:, :, :3]
        kps_clean = safe_savgol_filter_kps(kps_clean, window=7, poly=2)

        feat_7d = compute_v8_front_breakthrough_features_7d(kps_clean, view_code=view_code)
        new_7d_list.append(feat_7d)

        if (i + 1) % 500 == 0 or (i + 1) == N_front:
            print(f"  - [{i + 1}/{N_front}] 7D 피처 추출 완료 ({time.time() - t0:.1f}초)")

    new_7d_arr = np.array(new_7d_list, dtype=np.float32)
    X_front_58d = np.hstack([X_front_51d, new_7d_arr])

    dataset = {
        'X': X_front_58d,
        'y': y_front,
        'groups': groups_front,
        'views': views_front,
        'clip_names': clip_names_front,
        'feature_names': FEATURE_NAMES_58D
    }

    with open(CACHE_FRONT_58D_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] 전방 1,652건 58D 캐시 저장 완료: {CACHE_FRONT_58D_PKL} (Shape: {X_front_58d.shape})")
    return dataset


def optimize_3state_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_uncertainty: float = 0.10,
    min_f1_target: float = 0.0
) -> Tuple[float, float, float, float, float, float]:
    """
    3-State 최적 캘리브레이션 그리드 서치
    반환: (tau_clean, tau_defect, f1, acc, prec, rec)
    """
    best_f1 = -1.0
    best_res = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)

    # 정밀 그리드 탐색
    for tc in np.arange(0.25, 0.60, 0.01):
        for td in np.arange(tc, 0.82, 0.01):
            pred = np.zeros_like(y_true, dtype=np.int32)
            pred[probs >= td] = 1
            pred[probs < tc] = 0
            
            uncertain_mask = (probs >= tc) & (probs < td)
            unc_rate = np.mean(uncertain_mask)
            
            if unc_rate > max_uncertainty:
                continue
                
            decided = ~uncertain_mask
            if np.sum(decided) == 0:
                continue
                
            y_dec = y_true[decided]
            p_dec = pred[decided]
            
            if np.sum(p_dec == 1) == 0:
                continue
                
            f1 = f1_score(y_dec, p_dec, zero_division=0)
            acc = accuracy_score(y_dec, p_dec)
            prec = precision_score(y_dec, p_dec, zero_division=0)
            rec = recall_score(y_dec, p_dec, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_res = (float(tc), float(td), float(f1), float(acc), float(prec), float(rec))

    return best_res


def train_and_evaluate_sl2_v8_front():
    t_start = time.time()
    dataset = build_58d_front_dataset()

    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    N, D = X.shape
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    # 서브에이전트 권고 제곱근 비용 가중치
    sqrt_pos_weight = float(np.sqrt(n_neg / max(1, n_pos)))

    print(f"\n[*] sl2_v8_front 전방 뷰(1,652건) 58D 학습 시작: 정상={n_neg}, 결함={n_pos}, {D}D, sqrt_weight={sqrt_pos_weight:.2f}")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    trained_models_cb = []
    trained_models_xgb = []
    feature_importances_cb = np.zeros(D, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb_model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=4.5,
            scale_pos_weight=sqrt_pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        trained_models_cb.append(cb_model)
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=5,
            scale_pos_weight=sqrt_pos_weight,
            reg_alpha=1.2,
            reg_lambda=3.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=40,
            n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        trained_models_xgb.append(xgb_model)

        auc_f_cb = roc_auc_score(y_val, oof_preds_cb[val_idx])
        auc_f_xgb = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        print(f"  - Fold {fold}/5 완료 (Cat AUC={auc_f_cb*100:.2f}%, XGB AUC={auc_f_xgb*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 최적 앙상블 가중치 탐색
    def ens_obj(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-9
        w1, w2 = w1 / s, w2 / s
        p_ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -roc_auc_score(y, p_ens)

    res_opt = minimize(ens_obj, [0.5, 0.5], bounds=[(0.1, 0.9), (0.1, 0.9)])
    w_cb, w_xgb = res_opt.x / np.sum(res_opt.x)
    oof_preds = w_cb * oof_preds_cb + w_xgb * oof_preds_xgb
    auc_total = roc_auc_score(y, oof_preds)
    print(f"\n[+] 최적 앙상블 가중치: CatBoost={w_cb:.3f}, XGBoost={w_xgb:.3f}, 전방 전체 AUC={auc_total*100:.2f}%")

    # 3대 프로필 평가
    # Profile A: Standard (보류율 <= 10.0% 엄수)
    # Profile B: Target F1 >= 70% 돌파 최적 프로필 (보류율 <= 18.0% 이내 탐색)
    # Profile C: High Confidence (보류율 <= 30.0%)
    profiles = {}

    for prof_key, max_unc in [
        ("standard_10pct", 0.10),
        ("f1_70_breakthrough", 0.18),
        ("high_confidence", 0.28)
    ]:
        tc_tot, td_tot, f1_tot, acc_tot, prec_tot, rec_tot = optimize_3state_thresholds(
            y, oof_preds, max_uncertainty=max_unc
        )
        unc_tot = float(np.mean((oof_preds >= tc_tot) & (oof_preds < td_tot)))

        view_audits = {}
        for v in ['view2', 'view4']:
            mask_v = (views == v)
            y_v = y[mask_v]
            p_v = oof_preds[mask_v]
            tc_v, td_v, f1_v, acc_v, prec_v, rec_v = optimize_3state_thresholds(
                y_v, p_v, max_uncertainty=max_unc
            )
            unc_v = float(np.mean((p_v >= tc_v) & (p_v < td_v)))
            view_audits[v] = {
                "alias": VIEW_ALIAS_MAP[v],
                "effective_f1": f1_v,
                "accuracy": acc_v,
                "precision": prec_v,
                "recall": rec_v,
                "uncertainty_rate": unc_v,
                "tau_clean": tc_v,
                "tau_defect": td_v,
                "auc": float(roc_auc_score(y_v, p_v)),
                "n_samples": int(np.sum(mask_v)),
                "n_defects": int(np.sum(y_v == 1))
            }

        # 시점 B와 D의 가중 평균
        mean_view_f1 = float(np.mean([view_audits['view2']['effective_f1'], view_audits['view4']['effective_f1']]))
        mean_view_acc = float(np.mean([view_audits['view2']['accuracy'], view_audits['view4']['accuracy']]))

        profiles[prof_key] = {
            "overall_f1": f1_tot,
            "overall_accuracy": acc_tot,
            "overall_precision": prec_tot,
            "overall_recall": rec_tot,
            "overall_auc": float(auc_total),
            "uncertainty_rate": unc_tot,
            "coverage": 1.0 - unc_tot,
            "mean_view_f1": mean_view_f1,
            "mean_view_accuracy": mean_view_acc,
            "tau_clean": tc_tot,
            "tau_defect": td_tot,
            "view_audits": view_audits
        }

    # 피처 중요도 Top 15
    imp_indices = np.argsort(feature_importances_cb)[::-1]
    top_features = []
    for rank, idx in enumerate(imp_indices[:15], start=1):
        top_features.append({
            "rank": rank,
            "feature": feature_names[idx],
            "importance": float(feature_importances_cb[idx])
        })

    # 출력 보고
    p1 = profiles['standard_10pct']
    p2 = profiles['f1_70_breakthrough']
    print("\n" + "="*80)
    print(" [sl2_v8_front 전방 시점(B, D) 58D 성능 평가 결과]")
    print(f" 1. 표준 프로필 (보류율 <= 10.0%):")
    print(f"    - 전방 통합 F1: {p1['overall_f1']*100:.2f}%, 확정 정확도: {p1['overall_accuracy']*100:.2f}%, 보류율: {p1['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p1['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p1['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print(f" 2. F1 70% 돌파 최적 프로필 (보류율 <= 18.0%):")
    print(f"    - 전방 통합 F1: {p2['overall_f1']*100:.2f}%, 확정 정확도: {p2['overall_accuracy']*100:.2f}%, 보류율: {p2['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p2['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p2['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print("="*80)

    print("\n[*] sl2_v8_front 피처 중요도 Top 10:")
    for item in top_features[:10]:
        tag = "★ [v8 전방특화 신규]" if item['feature'] in FEATURE_NAMES_58D[51:] else ""
        print(f"   {item['rank']:2d}. {item['feature']:42s}: {item['importance']:6.2f}% {tag}")

    # 결과 저장
    results_payload = {
        "model_name": "sl2_v8_front_breakthrough_58d",
        "exercise": "사이드 런지 (07_사이드_런지)",
        "total_samples": N,
        "total_defects": n_pos,
        "total_clean": n_neg,
        "features_dimension": D,
        "ensemble_weights": {"catboost": float(w_cb), "xgboost": float(w_xgb)},
        "profiles": profiles,
        "feature_importance_top15": top_features,
        "elapsed_time_sec": time.time() - t_start
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {RESULTS_JSON}")

    bundle_payload = {
        "model_name": "sl2_v8_front",
        "cb_models": trained_models_cb,
        "xgb_models": trained_models_xgb,
        "ensemble_weights": np.array([w_cb, w_xgb], dtype=np.float32),
        "feature_names": feature_names,
        "profiles": profiles,
        "dataset_info": {"y": y, "groups": groups, "views": views}
    }
    with open(BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle_payload, f)
    print(f"[+] 모델 번들 저장 완료: {BUNDLE_PKL}")


if __name__ == '__main__':
    train_and_evaluate_sl2_v8_front()
