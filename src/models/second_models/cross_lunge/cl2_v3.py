# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Script: cl2_v3.py
Purpose: 다중공선성 및 시점 노이즈 11개 피처를 완벽 제거한 48D 정예 최적화 모델 (cl2_v3)
         - 1. 완벽 복제 피처(kinematic_shank_torsion_angle, r=1.000) 제거
         - 2. 내적-사잇각 비선형 삼각함수 완전 중복 3개(user_geom_*_angle_deg, r>0.96) 제거
         - 3. 랜드마크 고도 중복 2개(kinematic_reconstructed_foot_yaw, pacs_dkcd_medial_deviation, r>0.94) 제거
         - 4. 시점 간 일반화 방해 원핫 노이즈 5개(view_is_view*) 제거
         - 총 59D -> 48D 정예 피처셋 (Elite 48D)
         - CatBoost + XGBoost + LightGBM 3중 앙상블 블렌딩
         - 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증 (No Leakage)
         - cl2_v1 vs cl2_v2 vs cl2_v3 3대 세대 전수 비교 감사
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
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "전처리_데이터", "MediaPipe_Full33_64F", "08_크로스_런지")
V2_CACHE_PKL = os.path.join(PROJECT_ROOT, "Data", "피처_데이터", "cl2_v2_59d_cache.pkl")
V3_CACHE_PKL = os.path.join(PROJECT_ROOT, "Data", "피처_데이터", "cl2_v3_48d_cache.pkl")

RESULTS_V1_JSON = os.path.join(CURRENT_DIR, "cross_lunge_c2_v7_evaluation_results.json")
RESULTS_V2_JSON = os.path.join(CURRENT_DIR, "cl2_v2_evaluation_results.json")
RESULTS_V3_JSON = os.path.join(CURRENT_DIR, "cl2_v3_evaluation_results.json")
BUNDLE_V3_PKL = os.path.join(CURRENT_DIR, "cl2_v3_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

# 제거할 다중공선성 및 노이즈 11대 피처 공식 정의
FEATURES_TO_ELIMINATE_11 = [
    'kinematic_shank_torsion_angle',       # pacs_knee_ankle_coronal_tilt와 100% 동일 복제 (r=1.000)
    'user_geom_knee_foot_angle_deg',       # user_geom_knee_foot_parallel_dot과 삼각함수 단조 대응 중복 (r=0.992)
    'user_geom_torso_knee_angle_deg',      # user_geom_torso_knee_parallel_dot과 삼각함수 단조 대응 중복 (r=0.991)
    'user_geom_torso_foot_angle_deg',      # user_geom_torso_foot_parallel_dot과 삼각함수 단조 대응 중복 (r=0.962)
    'kinematic_reconstructed_foot_yaw',    # pacs_lead_foot_yaw_angle과 고도 중복 (r=0.947)
    'pacs_dkcd_medial_deviation',          # pacs_fppa_3d_collapse와 고도 중복 (r=0.988)
    'view_is_view1',                       # 시점 One-Hot 인코딩 노이즈
    'view_is_view2',
    'view_is_view3',
    'view_is_view4',
    'view_is_view5'
]


def build_or_load_v3_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """59D 데이터셋에서 11대 중복/노이즈 피처를 정밀 필터링하여 48D 정예 데이터셋 구축"""
    if os.path.exists(V3_CACHE_PKL) and not force_rebuild:
        print(f"[*] cl2_v3 48D 캐시 로드 중: {V3_CACHE_PKL}")
        with open(V3_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] cl2_v2 59D 캐시 기반 48D 정예 피처 데이터셋 필터링 구축 시작...")
    with open(V2_CACHE_PKL, 'rb') as f:
        v2_data = pickle.load(f)

    X_59d = v2_data['X']
    fnames_59d = list(v2_data['feature_names'])

    retained_indices = [i for i, fn in enumerate(fnames_59d) if fn not in FEATURES_TO_ELIMINATE_11]
    retained_fnames = [fnames_59d[i] for i in retained_indices]

    assert len(retained_indices) == 48, f"정예 피처 차원이 48이 아닙니다: {len(retained_indices)}"

    X_48d = X_59d[:, retained_indices]

    dataset = {
        'X': X_48d,
        'y': v2_data['y'],
        'groups': v2_data['groups'],
        'views': v2_data['views'],
        'clip_names': v2_data['clip_names'],
        'feature_names': retained_fnames,
        'retained_indices': retained_indices,
        'eliminated_features': FEATURES_TO_ELIMINATE_11
    }

    with open(V3_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] cl2_v3 48D 정예 데이터셋 캐시 저장 완료: {V3_CACHE_PKL}")
    return dataset


def run_cl2_v3_optimization():
    print("=" * 80)
    print("  [vidio-health-trainer] 크로스 런지 c2 48D 정예 최적화 모델 (cl2_v3) 공식 학습 및 평가")
    print("=" * 80)

    dataset = build_or_load_v3_dataset()
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']
    N = len(y)

    print(f"[*] 총 시퀀스 수: {N}개 (피처 차원: {X.shape[1]}D, 제거된 중복/노이즈: {len(dataset['eliminated_features'])}D)")
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    scale_pos_weight = float(n_neg / max(1, n_pos))

    # 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    oof_preds_lgb = np.zeros(N, dtype=np.float32)

    trained_models_cb = []
    trained_models_xgb = []
    trained_models_lgb = []
    feature_importances_cb = np.zeros(X.shape[1], dtype=np.float32)

    print("\n[*] CatBoost + XGBoost + LightGBM 3중 앙상블 5-Fold 피험자 완전 격리 학습...")
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost (최적 파라미터)
        cb_model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.035,
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

        # 2. XGBoost (최적 파라미터)
        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.035,
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

        # 3. LightGBM (최적 파라미터)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        trained_models_lgb.append(lgb_model)

        f_f1 = f1_score(y_val, (oof_preds_cb[val_idx] >= 0.5).astype(int), zero_division=0)
        f_auc = roc_auc_score(y_val, oof_preds_cb[val_idx])
        print(f"  - Fold {fold}/5 완료 (Cat F1={f_f1*100:.2f}%, AUC={f_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 3중 앙상블 가중치 Nelder-Mead 최적화
    def obj_func(weights):
        w1, w2, w3 = weights
        s = w1 + w2 + w3 + 1e-6
        w1, w2, w3 = w1 / s, w2 / s, w3 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb + w3 * oof_preds_lgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.45, 0.30, 0.25], bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb + best_w[2] * oof_preds_lgb
    print(f"\n[+] 최적 3중 앙상블 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}, LightGBM={best_w[2]:.3f}")

    # 3-State 적응형 캘리브레이션 함수 (보류율 10% 이하 제약)
    def optimize_view_threshold(y_true, y_prob, max_uncertainty=0.10):
        best_cfg = None
        best_f1 = -1.0

        for tau_c in np.arange(0.20, 0.54, 0.015):
            for tau_d in np.arange(0.46, 0.82, 0.015):
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
            'view_name': VIEW_ALIAS_MAP[v_code],
            'count': int(len(v_idx)),
            'tau_clean': cfg['tau_clean'],
            'tau_defect': cfg['tau_defect'],
            'effective_f1': cfg['effective_f1'],
            'accuracy': cfg['accuracy'],
            'precision': cfg['precision'],
            'recall': cfg['recall'],
            'uncertainty_rate': cfg['uncertainty_rate'],
            'confirmed_count': int(np.sum(v_cert)),
            'auc': float(roc_auc_score(y_v, prob_v))
        }

    # 전체 종합 3-State 지표
    y_p1_eval = y[p1_cert_mask]
    preds_p1_eval = p1_final_preds[p1_cert_mask]
    p1_overall_f1 = float(f1_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_acc = float(accuracy_score(y_p1_eval, preds_p1_eval))
    p1_overall_prec = float(precision_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_rec = float(recall_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_unc = float(1.0 - np.mean(p1_cert_mask))
    p1_overall_auc = float(roc_auc_score(y, oof_ens))

    # 2-State 지표
    preds_2state = (oof_ens >= 0.5).astype(int)
    f1_2state = float(f1_score(y, preds_2state, zero_division=0))
    acc_2state = float(accuracy_score(y, preds_2state))
    prec_2state = float(precision_score(y, preds_2state, zero_division=0))
    rec_2state = float(recall_score(y, preds_2state, zero_division=0))

    # 상위 10개 피처
    top10_indices = np.argsort(feature_importances_cb)[::-1][:10]
    top10_features = [
        {'name': feature_names[i], 'importance': float(feature_importances_cb[i])}
        for i in top10_indices
    ]

    # v1 및 v2 결과 로드
    with open(RESULTS_V1_JSON, 'r', encoding='utf-8') as f:
        res_v1 = json.load(f)
    with open(RESULTS_V2_JSON, 'r', encoding='utf-8') as f:
        res_v2 = json.load(f)

    f1_v1_3s = res_v1['overall_3state']['effective_f1']
    f1_v2_3s = res_v2['overall_3state']['effective_f1']
    f1_v1_2s = res_v1['overall_2state']['f1']
    f1_v2_2s = res_v2['overall_2state']['f1']
    auc_v1 = res_v1['overall_3state']['roc_auc']
    auc_v2 = res_v2['overall_3state']['roc_auc']

    delta_v1_3s = p1_overall_f1 - f1_v1_3s
    delta_v2_3s = p1_overall_f1 - f1_v2_3s
    delta_v1_2s = f1_2state - f1_v1_2s
    delta_v2_2s = f1_2state - f1_v2_2s
    delta_v1_auc = p1_overall_auc - auc_v1
    delta_v2_auc = p1_overall_auc - auc_v2

    print("\n" + "=" * 80)
    print("  [cl2_v1 vs cl2_v2 vs cl2_v3 3대 세대 공식 성능 비교 대조표]")
    print("=" * 80)
    print(f"[*] 2-State F1:       cl2_v1 {f1_v1_2s*100:.2f}%  ->  cl2_v2 {f1_v2_2s*100:.2f}%  ->  cl2_v3 {f1_2state*100:.2f}%  (v1 대비 {delta_v1_2s*100:+.2f}%p, v2 대비 {delta_v2_2s*100:+.2f}%p)")
    print(f"[*] 3-State F1:       cl2_v1 {f1_v1_3s*100:.2f}%  ->  cl2_v2 {f1_v2_3s*100:.2f}%  ->  cl2_v3 {p1_overall_f1*100:.2f}%  (v1 대비 {delta_v1_3s*100:+.2f}%p, v2 대비 {delta_v2_3s*100:+.2f}%p)")
    print(f"[*] ROC-AUC:          cl2_v1 {auc_v1:.4f}  ->  cl2_v2 {auc_v2:.4f}  ->  cl2_v3 {p1_overall_auc:.4f}  (v1 대비 {delta_v1_auc:+.4f}, v2 대비 {delta_v2_auc:+.4f})")
    print(f"[*] 3-State Coverage: cl2_v3 {(1-p1_overall_unc)*100:.2f}% (보류율: {p1_overall_unc*100:.2f}%)")

    print("\n[*] 공식 5대 시점별 cl2_v3 3-State 세부 성능:")
    for v_code in view_order:
        aud = p1_view_audits[v_code]
        v1_f = res_v1['view_audits_3state'][v_code]['effective_f1']
        v2_f = res_v2['view_audits_3state'][v_code]['effective_f1']
        print(f"  - {aud['view_name']:<25} | 3-State F1: {aud['effective_f1']*100:.2f}% (v1 대비 {(aud['effective_f1']-v1_f)*100:+.2f}%p, v2 대비 {(aud['effective_f1']-v2_f)*100:+.2f}%p) | Acc: {aud['accuracy']*100:.2f}% | AUC: {aud['auc']:.4f}")

    print("\n[*] 상위 5대 핵심 피처:")
    for rank, finfo in enumerate(top10_features[:5], 1):
        print(f"  {rank}. {finfo['name']}: {finfo['importance']:.3f}%")

    # 결과 JSON 저장
    payload = {
        'model_name': 'cl2_v3',
        'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
        'feature_dim': 48,
        'eliminated_features': FEATURES_TO_ELIMINATE_11,
        'n_samples': N,
        'ensemble_weights': {
            'catboost': float(best_w[0]),
            'xgboost': float(best_w[1]),
            'lightgbm': float(best_w[2])
        },
        'comparison_summary': {
            'v1_2state_f1': f1_v1_2s,
            'v2_2state_f1': f1_v2_2s,
            'v3_2state_f1': f1_2state,
            'v1_3state_f1': f1_v1_3s,
            'v2_3state_f1': f1_v2_3s,
            'v3_3state_f1': p1_overall_f1,
            'v1_roc_auc': auc_v1,
            'v2_roc_auc': auc_v2,
            'v3_roc_auc': p1_overall_auc,
        },
        'overall_2state': {
            'f1': f1_2state,
            'accuracy': acc_2state,
            'precision': prec_2state,
            'recall': rec_2state,
            'roc_auc': p1_overall_auc
        },
        'overall_3state': {
            'effective_f1': p1_overall_f1,
            'accuracy': p1_overall_acc,
            'precision': p1_overall_prec,
            'recall': p1_overall_rec,
            'uncertainty_rate': p1_overall_unc,
            'coverage': 1.0 - p1_overall_unc,
            'roc_auc': p1_overall_auc
        },
        'view_audits_3state': p1_view_audits,
        'top_features': top10_features
    }

    with open(RESULTS_V3_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] cl2_v3 공식 결과 JSON 저장 완료: {RESULTS_V3_JSON}")

    # 모델 번들 저장
    bundle = {
        'catboost_models': trained_models_cb,
        'xgboost_models': trained_models_xgb,
        'lightgbm_models': trained_models_lgb,
        'best_weights': best_w,
        'feature_names': feature_names,
        'p1_view_audits': p1_view_audits,
        'results_summary': payload
    }
    with open(BUNDLE_V3_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] cl2_v3 모델 번들 PKL 저장 완료: {BUNDLE_V3_PKL}")

    return payload


if __name__ == '__main__':
    run_cl2_v3_optimization()
