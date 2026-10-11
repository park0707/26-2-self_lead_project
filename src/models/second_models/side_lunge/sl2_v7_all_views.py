# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Model: sl2_v7_all_views.py
Purpose: 사용자 제안 3대 혁신 기하 알고리즘 통합 SOTA 모델 (sl2_v7, 51D)
         1. 카메라 광축 폐색 역투영 (3D)
         2. 하강 전 구간 다단계 시계열 벡터 (5D)
         3. 정강이 비틀림 기반 족부 역학적 복원 (4D)
         전체 원천 데이터(4,130건) 5대 시점 전수 학습 및 평가 (보류율 10% 모드 및 고신뢰 모드)
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

from side_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    V7_NEW_FEATURE_NAMES_12D,
    compute_v7_breakthrough_features_12d
)
from side_lunge_c2_features_v3 import safe_savgol_filter_kps

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
V6_CACHE_39D_PKL = r"D:\Data\피처_데이터\sl2_v6_all_views_39d_cache.pkl"
CACHE_ALL_51D_PKL = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_v7_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl2_v7_bundle.pkl"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_51d_all_views_dataset() -> Dict[str, Any]:
    """sl2_v6의 39D 캐시 + v7 신규 12D 피처 고속 결합"""
    if os.path.exists(CACHE_ALL_51D_PKL):
        print(f"[*] sl2_v7 4,130건 51D 캐시 로드 중: {CACHE_ALL_51D_PKL}")
        with open(CACHE_ALL_51D_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] sl2_v6 캐시 기반 4,130건 전수 51D 피처 데이터셋 고속 구축 시작...")
    with open(V6_CACHE_39D_PKL, 'rb') as f:
        v6_data = pickle.load(f)

    X_base = v6_data['X'] # (4130, 39)
    y = v6_data['y']
    groups = v6_data['groups']
    views = v6_data['views']
    clip_names = v6_data['clip_names']
    N = len(y)

    new_12d_list = []
    print(f"[*] 4,130건 대상 v7 혁신 12D 피처(역투영 3D + 시계열 5D + 역학복원 4D) 고속 추출 중...")
    t0 = time.time()

    for i in range(N):
        clip_name = clip_names[i]
        view_code = views[i]
        npz_name = f"{clip_name}_{view_code}.npz"
        npz_path = os.path.join(DATA_DIR, npz_name)

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ 파일 없음: {npz_path}")

        data = np.load(npz_path)
        kps_clean = data['kps_33_3d_64f'][:, :, :3]
        kps_clean = safe_savgol_filter_kps(kps_clean, window=7, poly=2)

        feat_12d = compute_v7_breakthrough_features_12d(kps_clean, view_code=view_code)
        new_12d_list.append(feat_12d)

        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"  - [{i + 1}/{N}] 12D 피처 추출 완료 ({time.time() - t0:.1f}초)")

    new_12d_arr = np.array(new_12d_list, dtype=np.float32)
    X_51d = np.hstack([X_base, new_12d_arr])

    dataset = {
        'X': X_51d,
        'y': y,
        'groups': groups,
        'views': views,
        'clip_names': clip_names,
        'feature_names': FEATURE_NAMES_51D
    }

    with open(CACHE_ALL_51D_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] 전체 4,130건 51D 캐시 저장 완료: {CACHE_ALL_51D_PKL} (Shape: {X_51d.shape})")
    return dataset


def train_and_evaluate_sl2_v7():
    t_start = time.time()
    dataset = build_51d_all_views_dataset()

    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    N, D = X.shape
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"\n[*] sl2_v7 전체 원천 데이터(4,130건) 51D 학습 시작: 정상={n_neg}, 결함={n_pos}, {D}D")
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
            iterations=650,
            learning_rate=0.038,
            depth=6,
            l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        trained_models_cb.append(cb_model)
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=550,
            learning_rate=0.038,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        trained_models_xgb.append(xgb_model)

        fold_auc = roc_auc_score(y_val, oof_preds_cb[val_idx])
        fold_f1 = f1_score(y_val, (oof_preds_cb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  - Fold {fold}/5 완료 (Cat F1={fold_f1*100:.2f}%, AUC={fold_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 최적 가중치 앙상블 블렌딩
    def obj_func(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-6
        w1, w2 = w1 / s, w2 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.5, 0.5], bounds=[(0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb
    print(f"\n[+] 최적 앙상블 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}")

    # 3-State 적응형 캘리브레이션 함수
    def optimize_view_threshold(y_true, y_prob, max_uncertainty=0.10):
        best_cfg = None
        best_f1 = -1.0

        for tau_c in np.arange(0.18, 0.55, 0.015):
            for tau_d in np.arange(0.45, 0.80, 0.015):
                if tau_c >= tau_d:
                    continue
                mask_cert = (y_prob < tau_c) | (y_prob >= tau_d)
                unc_rate = 1.0 - np.mean(mask_cert)

                if max_uncertainty is not None and unc_rate > max_uncertainty:
                    continue

                if np.sum(mask_cert) == 0:
                    continue

                y_eval_t = y_true[mask_cert]
                y_eval_p = (y_prob[mask_cert] >= tau_d).astype(int)
                score = f1_score(y_eval_t, y_eval_p, zero_division=0)

                if score > best_f1:
                    best_f1 = score
                    best_cfg = {
                        'tau_clean': float(tau_c),
                        'tau_defect': float(tau_d),
                        'effective_f1': float(score),
                        'accuracy': float(accuracy_score(y_eval_t, y_eval_p)),
                        'precision': float(precision_score(y_eval_t, y_eval_p, zero_division=0)),
                        'recall': float(recall_score(y_eval_t, y_eval_p, zero_division=0)),
                        'uncertainty_rate': float(unc_rate)
                    }
        return best_cfg

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']

    # 1) 프로파일 1: 보류율 <= 10.0% 표준 모드
    p1_view_audits = {}
    p1_cert_mask = np.zeros(N, dtype=bool)
    p1_final_preds = np.zeros(N, dtype=int)

    for v_code in view_order:
        v_idx = np.where(views == v_code)[0]
        y_v = y[v_idx]
        prob_v = oof_ens[v_idx]

        cfg = optimize_view_threshold(y_v, prob_v, max_uncertainty=0.10)
        v_cert = (prob_v < cfg['tau_clean']) | (prob_v >= cfg['tau_defect'])
        p1_cert_mask[v_idx] = v_cert
        p1_final_preds[v_idx] = (prob_v >= cfg['tau_defect']).astype(int)

        p1_view_audits[v_code] = {
            **cfg,
            'alias': VIEW_ALIAS_MAP[v_code],
            'n_samples': len(v_idx),
            'n_defects': int(np.sum(y_v == 1))
        }

    p1_y_cert = y[p1_cert_mask]
    p1_p_cert = p1_final_preds[p1_cert_mask]
    p1_overall = {
        'overall_f1': float(f1_score(p1_y_cert, p1_p_cert, zero_division=0)),
        'overall_accuracy': float(accuracy_score(p1_y_cert, p1_p_cert)),
        'overall_precision': float(precision_score(p1_y_cert, p1_p_cert, zero_division=0)),
        'overall_recall': float(recall_score(p1_y_cert, p1_p_cert, zero_division=0)),
        'overall_auc': float(roc_auc_score(y, oof_ens)),
        'uncertainty_rate': float(1.0 - np.mean(p1_cert_mask)),
        'coverage': float(np.mean(p1_cert_mask)),
        'view_audits': p1_view_audits
    }

    # 2) 프로파일 2: 고신뢰 정밀 모드
    p2_view_audits = {}
    p2_cert_mask = np.zeros(N, dtype=bool)
    p2_final_preds = np.zeros(N, dtype=int)

    for v_code in view_order:
        v_idx = np.where(views == v_code)[0]
        y_v = y[v_idx]
        prob_v = oof_ens[v_idx]

        cfg = optimize_view_threshold(y_v, prob_v, max_uncertainty=None)
        v_cert = (prob_v < cfg['tau_clean']) | (prob_v >= cfg['tau_defect'])
        p2_cert_mask[v_idx] = v_cert
        p2_final_preds[v_idx] = (prob_v >= cfg['tau_defect']).astype(int)

        p2_view_audits[v_code] = {
            **cfg,
            'alias': VIEW_ALIAS_MAP[v_code],
            'n_samples': len(v_idx),
            'n_defects': int(np.sum(y_v == 1))
        }

    p2_y_cert = y[p2_cert_mask]
    p2_p_cert = p2_final_preds[p2_cert_mask]
    p2_overall = {
        'overall_f1': float(f1_score(p2_y_cert, p2_p_cert, zero_division=0)),
        'overall_accuracy': float(accuracy_score(p2_y_cert, p2_p_cert)),
        'overall_precision': float(precision_score(p2_y_cert, p2_p_cert, zero_division=0)),
        'overall_recall': float(recall_score(p2_y_cert, p2_p_cert, zero_division=0)),
        'overall_auc': float(roc_auc_score(y, oof_ens)),
        'uncertainty_rate': float(1.0 - np.mean(p2_cert_mask)),
        'coverage': float(np.mean(p2_cert_mask)),
        'view_audits': p2_view_audits
    }

    # 3) 전방 시점(B+D, 1,652건) 전용 서브메트릭 산출
    front_mask = np.isin(views, ['view2', 'view4'])
    front_cert_mask = p1_cert_mask & front_mask
    front_p1_f1 = float(f1_score(y[front_cert_mask], p1_final_preds[front_cert_mask], zero_division=0))
    front_p1_acc = float(accuracy_score(y[front_cert_mask], p1_final_preds[front_cert_mask]))

    print("\n" + "=" * 80)
    print(" [sl2_v7 전체 원천 데이터(4,130건) 5대 시점 3-State 평가 결과 (보류율 <= 10.0%)]")
    print(f"  - 전체 통합 3-State F1 : {p1_overall['overall_f1']*100:.2f}%")
    print(f"  - 전체 확정 정확도(ACC): {p1_overall['overall_accuracy']*100:.2f}%")
    print(f"  - 전체 정밀도 / 재현율 : {p1_overall['overall_precision']*100:.2f}% / {p1_overall['overall_recall']*100:.2f}%")
    print(f"  - 전체 평균 보류율     : {p1_overall['uncertainty_rate']*100:.2f}% (목표 10% 엄수)")
    print(f"  - (전방 B+D 서브 F1)   : {front_p1_f1*100:.2f}% (정확도: {front_p1_acc*100:.2f}%)")
    print("-" * 80)
    print(" [5대 공식 시점별 세부 성적표 (보류율 <= 10.0%)]")
    for v_code in view_order:
        aud = p1_view_audits[v_code]
        print(f"  * [{v_code}] {aud['alias']:24s}: F1={aud['effective_f1']*100:5.2f}%, 정확도={aud['accuracy']*100:5.2f}%, 정밀도={aud['precision']*100:5.2f}%, 재현율={aud['recall']*100:5.2f}%, 보류율={aud['uncertainty_rate']*100:4.1f}%")
    print("=" * 80)

    # 피처 중요도 Top 15 정리
    top_indices = np.argsort(feature_importances_cb)[::-1][:15]
    top_features = []
    print("\n[*] sl2_v7 피처 중요도 Top 15:")
    for rank, idx in enumerate(top_indices, start=1):
        fname = feature_names[idx]
        imp = float(feature_importances_cb[idx])
        top_features.append({'rank': rank, 'feature': fname, 'importance': imp})
        is_v7 = "★ [v7 신규]" if fname in V7_NEW_FEATURE_NAMES_12D else ""
        print(f"  {rank:2d}. {fname:42s}: {imp:6.2f}% {is_v7}")

    results_data = {
        'model_name': 'sl2_v7_breakthrough_51d',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'total_samples': N,
        'total_defects': n_pos,
        'total_clean': n_neg,
        'features_dimension': D,
        'profiles': {
            'standard_10pct': p1_overall,
            'high_confidence': p2_overall
        },
        'front_views_submetrics': {
            'f1': front_p1_f1,
            'accuracy': front_p1_acc
        },
        'feature_importance_top15': top_features,
        'v7_new_features_analysis': [
            {'feature': f, 'importance': float(feature_importances_cb[feature_names.index(f)])}
            for f in V7_NEW_FEATURE_NAMES_12D
        ],
        'ensemble_weights': {
            'catboost': float(best_w[0]),
            'xgboost': float(best_w[1])
        },
        'elapsed_time_sec': float(time.time() - t_start)
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 모델 번들 저장
    bundle = {
        'model_name': 'sl2_v7_breakthrough_51d',
        'cb_models': trained_models_cb,
        'xgb_models': trained_models_xgb,
        'ensemble_weights': best_w,
        'feature_names': feature_names,
        'profiles': {
            'standard_10pct': p1_overall,
            'high_confidence': p2_overall
        }
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 모델 가중치 번들 저장 완료: {MODEL_BUNDLE_PKL}")


if __name__ == '__main__':
    train_and_evaluate_sl2_v7()
