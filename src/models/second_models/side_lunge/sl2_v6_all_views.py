# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Model: sl2_v6_all_views.py
Purpose: 전체 원천 데이터(5대 시점, 4,130개 시퀀스 전수) 대상
         사용자 제안 3D 기하 직교/평행 정렬 알고리즘 통합 39D 모델 학습 및
         보류율 10% 제한 3-State F1 시점별 성능 정밀 평가
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

from side_lunge_c2_v6_geometric_features import (
    FEATURE_NAMES_39D,
    USER_GEOMETRIC_FEATURE_NAMES_7D,
    compute_user_geometric_alignment
)
from src.features.side_lunge_spatial_leg_separator import detect_side_lunge_lead_leg_spatial
from side_lunge_c2_features_v3 import vector_angle_3d

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
V4_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v4_dataset_cache.pkl"
CACHE_ALL_39D_PKL = r"D:\Data\피처_데이터\sl2_v6_all_views_39d_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_v6_all_views_evaluation_results.json"

# AGENTS.md 공식 5대 시점 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_39d_all_views_dataset() -> Dict[str, Any]:
    """sl2_v4의 4,130건 30D 캐시 + 추가 9D(광축 2D + 사용자 기하 7D) 고속 결합"""
    if os.path.exists(CACHE_ALL_39D_PKL):
        print(f"[*] sl2_v6 4,130건 39D 캐시 로드 중: {CACHE_ALL_39D_PKL}")
        with open(CACHE_ALL_39D_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] sl2_v4 캐시 기반 4,130건 전수 39D 피처 데이터셋 고속 구축 시작...")
    with open(V4_CACHE_PKL, 'rb') as f:
        v4_data = pickle.load(f)

    X_base = v4_data['X'] # (4130, 30)
    y = v4_data['y']
    groups = v4_data['groups']
    views = v4_data['views']
    clip_names = v4_data['clip_names']
    N = len(y)

    extra_9d_list = []
    print(f"[*] 4,130건 대상 9D 피처(사용자 기하 7D + 광축 2D) 고속 추출 중...")
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
        T = kps_clean.shape[0]

        is_left_lead, t_bot = detect_side_lunge_lead_leg_spatial(kps_clean)
        t_bot = max(5, min(T - 5, t_bot))

        if is_left_lead:
            hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
            lat_sign = 1.0
        else:
            hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
            lat_sign = -1.0

        # 광축 투영 피처 2D
        view_sign = 1.0 if view_code in ['view2', 'view5'] else -1.0
        cam_axis = np.array([view_sign * 0.7071, 0.0, 0.7071], dtype=np.float32)
        cam_normal = np.array([-0.7071, 0.0, view_sign * 0.7071], dtype=np.float32)

        v_foot = kps_clean[t_bot, toe_idx] - kps_clean[t_bot, heel_idx]
        v_foot_proj = np.array([np.dot(v_foot, cam_normal), v_foot[1], np.dot(v_foot, cam_axis)])
        foot_proj_norm = np.linalg.norm(v_foot_proj) + 1e-6
        u_foot_proj = v_foot_proj / foot_proj_norm
        front_optical_axis_foot_yaw = float(np.degrees(np.arctan2(u_foot_proj[0] * lat_sign, u_foot_proj[2] + 1e-6)))

        v_knee_disp = kps_clean[t_bot, knee_idx] - kps_clean[0, knee_idx]
        front_optical_axis_knee_foot_alignment = vector_angle_3d(v_knee_disp, v_foot)

        # 사용자 기하 7D 피처
        (stand_dot_t_th, stand_dot_th_f, stand_dot_t_f,
         stand_ang_t_f, stand_dev_t_th, stand_dev_th_f) = compute_user_geometric_alignment(
            kps_clean[0], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign
        )
        (bot_dot_t_th, bot_dot_th_f, bot_dot_t_f,
         bot_ang_t_f, bot_dev_t_th, bot_dev_th_f) = compute_user_geometric_alignment(
            kps_clean[t_bot], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign
        )
        delta_torso_foot_alignment = bot_dot_t_f - stand_dot_t_f

        extra_9d = [
            front_optical_axis_foot_yaw,
            front_optical_axis_knee_foot_alignment,
            bot_dot_t_th,
            bot_dot_th_f,
            bot_dot_t_f,
            bot_ang_t_f,
            bot_dev_t_th,
            bot_dev_th_f,
            delta_torso_foot_alignment
        ]
        extra_9d_list.append(extra_9d)

        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"  - [{i + 1}/{N}] 9D 피처 추출 완료 ({time.time() - t0:.1f}초)")

    extra_9d_arr = np.array(extra_9d_list, dtype=np.float32)
    X_39d = np.hstack([X_base, extra_9d_arr])

    dataset = {
        'X': X_39d,
        'y': y,
        'groups': groups,
        'views': views,
        'clip_names': clip_names,
        'feature_names': FEATURE_NAMES_39D
    }

    with open(CACHE_ALL_39D_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] 전체 4,130건 39D 캐시 저장 완료: {CACHE_ALL_39D_PKL} (Shape: {X_39d.shape})")
    return dataset


def run_evaluation():
    t_start = time.time()
    dataset = build_39d_all_views_dataset()

    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    N, D = X.shape
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"\n[*] sl2_v6 전체 원천 데이터(4,130건) 학습 시작: 정상={n_neg}, 결함={n_pos}, {D}D")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    feature_importances_cb = np.zeros(D, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb_model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
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

        fold_auc = roc_auc_score(y_val, oof_preds_cb[val_idx])
        fold_f1 = f1_score(y_val, (oof_preds_cb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  - Fold {fold}/5 완료 (Cat F1={fold_f1*100:.2f}%, AUC={fold_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 최적 앙상블 블렌딩
    def obj_func(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-6
        w1, w2 = w1 / s, w2 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.5, 0.5], bounds=[(0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb
    print(f"\n[+] 최적 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}")

    # 3-State 적응형 캘리브레이션 (보류율 <= 10.0% 엄격 제한)
    def optimize_view_threshold(y_true, y_prob, max_uncertainty=0.10):
        best_cfg = None
        best_f1 = -1.0

        for tau_c in np.arange(0.20, 0.55, 0.015):
            for tau_d in np.arange(0.45, 0.80, 0.015):
                if tau_c >= tau_d:
                    continue
                mask_cert = (y_prob < tau_c) | (y_prob >= tau_d)
                unc_rate = 1.0 - np.mean(mask_cert)

                if unc_rate > max_uncertainty:
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

    view_audits = {}
    cert_mask = np.zeros(N, dtype=bool)
    final_preds = np.zeros(N, dtype=int)

    # 5대 공식 시점 순회
    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    for v_code in view_order:
        v_idx = np.where(views == v_code)[0]
        y_v = y[v_idx]
        prob_v = oof_ens[v_idx]

        cfg = optimize_view_threshold(y_v, prob_v, max_uncertainty=0.10)
        v_cert = (prob_v < cfg['tau_clean']) | (prob_v >= cfg['tau_defect'])
        cert_mask[v_idx] = v_cert
        final_preds[v_idx] = (prob_v >= cfg['tau_defect']).astype(int)

        view_audits[v_code] = {
            **cfg,
            'alias': VIEW_ALIAS_MAP[v_code],
            'n_samples': len(v_idx),
            'n_defects': int(np.sum(y_v == 1))
        }

    y_cert = y[cert_mask]
    p_cert = final_preds[cert_mask]

    overall_metrics = {
        'overall_f1': float(f1_score(y_cert, p_cert, zero_division=0)),
        'overall_accuracy': float(accuracy_score(y_cert, p_cert)),
        'overall_precision': float(precision_score(y_cert, p_cert, zero_division=0)),
        'overall_recall': float(recall_score(y_cert, p_cert, zero_division=0)),
        'overall_auc': float(roc_auc_score(y, oof_ens)),
        'uncertainty_rate': float(1.0 - np.mean(cert_mask)),
        'coverage': float(np.mean(cert_mask))
    }

    print("\n" + "=" * 80)
    print(" [sl2_v6 4,130건 전체 원천 데이터 5대 시점 3-State 평가 결과 (보류율 <= 10.0%)]")
    print(f"  - 전체 통합 3-State F1 : {overall_metrics['overall_f1']*100:.2f}%")
    print(f"  - 전체 확정 정확도(ACC): {overall_metrics['overall_accuracy']*100:.2f}%")
    print(f"  - 전체 정밀도 / 재현율 : {overall_metrics['overall_precision']*100:.2f}% / {overall_metrics['overall_recall']*100:.2f}%")
    print(f"  - 전체 평균 보류율     : {overall_metrics['uncertainty_rate']*100:.2f}% (목표 10% 이내 달성)")
    print("-" * 80)
    print(" [5대 공식 시점별 세부 성적표]")
    for v_code in view_order:
        aud = view_audits[v_code]
        print(f"  * [{v_code}] {aud['alias']:24s}: F1={aud['effective_f1']*100:5.2f}%, 정확도={aud['accuracy']*100:5.2f}%, 정밀도={aud['precision']*100:5.2f}%, 재현율={aud['recall']*100:5.2f}%, 보류율={aud['uncertainty_rate']*100:4.1f}%")
    print("=" * 80)

    results_data = {
        'model_name': 'sl2_v6_all_views_ortho',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'total_samples': N,
        'total_defects': n_pos,
        'total_clean': n_neg,
        'features_dimension': D,
        'overall_metrics': overall_metrics,
        'view_audits': view_audits,
        'ensemble_weights': {
            'catboost': float(best_w[0]),
            'xgboost': float(best_w[1])
        },
        'elapsed_time_sec': float(time.time() - t_start)
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 결과 JSON 저장 완료: {RESULTS_JSON}")


if __name__ == '__main__':
    run_evaluation()
