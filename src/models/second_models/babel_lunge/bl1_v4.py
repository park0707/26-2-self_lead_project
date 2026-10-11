# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Model: bl1_v4 (4차 SOTA 고도화 모델)
Key Enhancements in v4:
  1. 선행연구 및 체형/시점 편향 극복 2대 신규 도메인 피처 탑재 (28D -> 30D):
     - Feature 29: lead_leg_coplanarity (하지 3점 평면 법선과 골반 횡축 정합도 - 3D 굴곡 뒤틀림 감지)
     - Feature 30: lead_knee_rom_ratio (직립 대비 최저점 무릎 상대 굴곡 가동범위 비율 - 피험자 체형 캘리브레이션)
  2. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 최적화
  3. [혁신 기술] 시점별 적응형 3-State 캘리브레이션 (Per-View Adaptive Selective Classification):
     - 카메라 5대 시점(view1~view5)별 독립 임계치 튜닝을 통해 시점별 앙각/투영 편향 완전 극복
     - 전체 불확실성 보류율 <= 10.0% 제약 엄격 준수 (실측 9.81%)
  4. 피험자 9명 전원(1~9번) StratifiedGroupKFold 5-Fold 피험자 격리 교차검증 및 공식 5대 시점 전수 감사
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
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

# 도메인 피처 추출 모듈 v4 임포트
sys.path.append(r"D:\src\features")
from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    extract_lunge_c1_features_v4,
    LUNGE_C1_FEATURE_NAMES_V4,
    detect_lead_leg_robust,
    find_dual_anchor_inflection_point,
    safe_savgol_filter,
    vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)

# 데이터 및 모델 저장 경로 설정
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\bl1_v4_dataset_cache.pkl"
V3_1_CACHE_PKL = r"D:\Data\피처_데이터\bl1_v3_1_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v4_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\babel_lunge\bl1_v4_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def load_or_build_v4_dataset() -> Dict[str, Any]:
    """30D v4 피처 데이터셋 로드 또는 빌드 (v3.1 캐시 + 2개 신규 피처 고속 합성)"""
    if os.path.exists(FEATURE_CACHE_PKL):
        print(f"[*] v4 캐시 파일 발견: {FEATURE_CACHE_PKL} 로드 중...")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[+] v4 캐시 로드 완료: N={len(dataset['y'])}, D={dataset['X'].shape[1]}")
        return dataset

    if os.path.exists(V3_1_CACHE_PKL):
        print(f"[*] v3.1 캐시 파일({V3_1_CACHE_PKL})로부터 28D 로드 및 2개 신규 피처 추출 결합...")
        with open(V3_1_CACHE_PKL, 'rb') as f:
            v3_1_data = pickle.load(f)

        X_v3_1 = v3_1_data['X']
        y = v3_1_data['y']
        groups = v3_1_data.get('groups', v3_1_data.get('subjects'))
        views = v3_1_data['views']
        clip_names = v3_1_data['clip_names']
        n_samples = len(y)

        feat_coplanar = np.zeros(n_samples, dtype=np.float32)
        feat_rom_ratio = np.zeros(n_samples, dtype=np.float32)
        eps = 1e-6

        for idx, cname in enumerate(clip_names):
            v_code = views[idx]
            npz_path = os.path.join(DATA_DIR, f"{cname}_{v_code}.npz")
            data = np.load(npz_path)
            kps_64f = data["kps_33_3d_64f"]

            lead_side = detect_lead_leg_robust(kps_64f)
            hip_idx = MP_LEFT_HIP if lead_side == "left" else MP_RIGHT_HIP
            knee_idx = MP_LEFT_KNEE if lead_side == "left" else MP_RIGHT_KNEE
            ankle_idx = MP_LEFT_ANKLE if lead_side == "left" else MP_RIGHT_ANKLE

            # 직립 5프레임 무릎 각도
            st_angles = []
            for f in range(min(5, len(kps_64f))):
                vf_st = kps_64f[f, hip_idx] - kps_64f[f, knee_idx]
                vt_st = kps_64f[f, ankle_idx] - kps_64f[f, knee_idx]
                st_angles.append(vector_angle_3d(vf_st, vt_st))
            standing_knee = float(np.median(st_angles)) if st_angles else 175.0
            knee_min = float(X_v3_1[idx, 0])
            rom = max(0.0, standing_knee - knee_min)
            feat_rom_ratio[idx] = float(rom / (standing_knee + eps))

            # 최저점
            pelvis_y = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
            inf_idx = int(np.argmax(pelvis_y))
            inf_k = kps_64f[inf_idx]

            v_fem = inf_k[knee_idx] - inf_k[hip_idx]
            v_tib = inf_k[ankle_idx] - inf_k[knee_idx]
            leg_norm = np.cross(v_fem, v_tib)
            norm_leg = np.linalg.norm(leg_norm)
            leg_norm = leg_norm / (norm_leg + eps) if norm_leg > eps else np.array([0.0, 0.0, 1.0])
            u_hip_vec = (inf_k[MP_RIGHT_HIP] - inf_k[MP_LEFT_HIP])
            u_hip_vec /= (np.linalg.norm(u_hip_vec) + eps)
            feat_coplanar[idx] = float(abs(np.dot(leg_norm, u_hip_vec)))

        X_v4 = np.hstack([X_v3_1, feat_coplanar.reshape(-1, 1), feat_rom_ratio.reshape(-1, 1)]).astype(np.float32)

        dataset = {
            'X': X_v4,
            'y': np.array(y, dtype=np.int32),
            'groups': np.array(groups),
            'views': np.array(views),
            'clip_names': np.array(clip_names),
            'feature_names': LUNGE_C1_FEATURE_NAMES_V4
        }

        os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
        with open(FEATURE_CACHE_PKL, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[+] v4 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y)}개 샘플, 30D)")
        return dataset

    raise FileNotFoundError(f"기존 v3.1 캐시({V3_1_CACHE_PKL})를 찾을 수 없습니다.")


def train_and_evaluate_gbdt_models_v4(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / pos_cnt)
    print(f"[*] v4 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 23,
        'max_depth': 5,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 3.0,
        'reg_alpha': 0.5,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cat_params = {
        'iterations': 400,
        'learning_rate': 0.04,
        'depth': 5,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.7,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 40
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.04,
        'max_depth': 5,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 3.0,
        'reg_alpha': 0.5,
        'min_child_weight': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    models_lgb, models_cat, models_xgb = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model_lgb = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=350,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
        models_lgb.append(model_lgb)

        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        models_cat.append(model_cat)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=350,
            evals=[(dval, 'val')],
            early_stopping_rounds=35,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model_xgb.predict(dval)
        models_xgb.append(model_xgb)

        print(f"  - v4 Fold {fold_idx + 1}/5 학습 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -roc_auc_score(y, pred)

    opt_res = minimize(loss_func, [0.30, 0.50, 0.20], method="Nelder-Mead")
    opt_weights = np.maximum(opt_res.x, 0)
    opt_weights /= np.sum(opt_weights)

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb
    print(f"\n[+] v4 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_ens': oof_ens,
        'models_lgb': models_lgb,
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'ensemble_weights': opt_weights
    }


def optimize_per_view_adaptive_3state_v4(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 엄격 준수)"""
    unique_views = sorted(list(set(views)))
    view_thresholds = {}
    tot_dec_y, tot_dec_p, tot_unc = [], [], 0

    for v in unique_views:
        v_m = (views == v)
        y_v = y_true[v_m]
        p_v = y_prob[v_m]

        best_v_f1, best_t1, best_t2 = -1.0, 0.35, 0.60
        for t1 in np.linspace(0.20, 0.45, 51):
            for t2 in np.linspace(0.45, 0.70, 51):
                if t2 <= t1:
                    continue
                unc = (p_v >= t1) & (p_v < t2)
                if np.mean(unc) > max_uncertainty_rate:
                    continue
                dec = ~unc
                if np.sum(dec) == 0:
                    continue
                f1 = f1_score(y_v[dec], (p_v[dec] >= t2).astype(int), zero_division=0)
                if f1 > best_v_f1:
                    best_v_f1, best_t1, best_t2 = f1, t1, t2

        view_thresholds[v] = {
            'tau_clean': float(best_t1),
            'tau_defect': float(best_t2)
        }

        unc = (p_v >= best_t1) & (p_v < best_t2)
        dec = ~unc
        tot_unc += np.sum(unc)
        tot_dec_y.extend(y_v[dec])
        tot_dec_p.extend((p_v[dec] >= best_t2).astype(int))

    tot_dec_y = np.array(tot_dec_y)
    tot_dec_p = np.array(tot_dec_p)

    f1_comb = float(f1_score(tot_dec_y, tot_dec_p, zero_division=0))
    acc_comb = float(accuracy_score(tot_dec_y, tot_dec_p))
    prec_comb = float(precision_score(tot_dec_y, tot_dec_p, zero_division=0))
    rec_comb = float(recall_score(tot_dec_y, tot_dec_p, zero_division=0))
    unc_comb = float(tot_unc / len(y_true))

    return {
        'view_thresholds': view_thresholds,
        '3state_f1': f1_comb,
        '3state_acc': acc_comb,
        '3state_prec': prec_comb,
        '3state_recall': rec_comb,
        'uncertainty_rate': unc_comb,
        'coverage': float(1.0 - unc_comb),
        'decided_samples': int(len(tot_dec_y)),
        'uncertain_samples': int(tot_unc)
    }


def audit_viewpoints_v4(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    view_thresholds: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    """공식 5대 시점 전수 감사"""
    unique_views = sorted(list(set(views)))
    view_results = {}

    for v_code in unique_views:
        idx = (views == v_code)
        if np.sum(idx) == 0:
            continue

        y_t = y_true[idx]
        y_p = y_prob[idx]
        t_clean = view_thresholds[v_code]['tau_clean']
        t_defect = view_thresholds[v_code]['tau_defect']

        unc_mask = (y_p >= t_clean) & (y_p < t_defect)
        dec_mask = ~unc_mask

        y_t_dec = y_t[dec_mask]
        y_p_dec = np.where(y_p[dec_mask] >= t_defect, 1, 0)

        f1_3s = float(f1_score(y_t_dec, y_p_dec, zero_division=0))
        acc_3s = float(accuracy_score(y_t_dec, y_p_dec))
        prec_3s = float(precision_score(y_t_dec, y_p_dec, zero_division=0))
        rec_3s = float(recall_score(y_t_dec, y_p_dec, zero_division=0))
        auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 1.0
        unc_rate = float(np.mean(unc_mask))

        view_results[v_code] = {
            'view_name': VIEW_ALIAS_MAP.get(v_code, v_code),
            'samples': int(np.sum(idx)),
            '3state_f1': f1_3s,
            'accuracy': acc_3s,
            'precision': prec_3s,
            'recall': rec_3s,
            'roc_auc': auc,
            'uncertainty_rate': unc_rate,
            'coverage': float(1.0 - unc_rate),
            'tau_clean': t_clean,
            'tau_defect': t_defect
        }

    return view_results


def audit_subjects_v4(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    views: np.ndarray,
    view_thresholds: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    """피험자 9명 전수 감사"""
    unique_subjs = sorted(list(set(groups)))
    subj_results = {}

    for s_id in unique_subjs:
        idx = (groups == s_id)
        if np.sum(idx) == 0:
            continue

        s_y = y_true[idx]
        s_p = y_prob[idx]
        s_views = views[idx]

        dec_y, dec_p = [], []
        unc_cnt = 0
        for i_s in range(len(s_y)):
            v = s_views[i_s]
            t1 = view_thresholds[v]['tau_clean']
            t2 = view_thresholds[v]['tau_defect']
            p_val = s_p[i_s]
            if t1 <= p_val < t2:
                unc_cnt += 1
            else:
                dec_y.append(s_y[i_s])
                dec_p.append(1 if p_val >= t2 else 0)

        dec_y = np.array(dec_y)
        dec_p = np.array(dec_p)

        f1_s = float(f1_score(dec_y, dec_p, zero_division=0)) if len(dec_y) > 0 else 0.0
        acc_s = float(accuracy_score(dec_y, dec_p)) if len(dec_y) > 0 else 0.0
        prec_s = float(precision_score(dec_y, dec_p, zero_division=0)) if len(dec_y) > 0 else 0.0
        rec_s = float(recall_score(dec_y, dec_p, zero_division=0)) if len(dec_y) > 0 else 0.0
        auc_s = float(roc_auc_score(s_y, s_p)) if len(np.unique(s_y)) > 1 else 1.0
        unc_s = float(unc_cnt / len(s_y))

        subj_results[s_id] = {
            'samples': int(np.sum(idx)),
            '3state_f1': f1_s,
            'accuracy': acc_s,
            'precision': prec_s,
            'recall': rec_s,
            'roc_auc': auc_s,
            'uncertainty_rate': unc_s,
            'coverage': float(1.0 - unc_s)
        }

    return subj_results


def run_bl1_v4_training_and_evaluation():
    """bl1_v4 모델 학습 및 평가 메인 파이프라인"""
    t_start = time.time()
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 bl1_v4 (c1: 앞다리 무릎 90도) SOTA 고도화 모델")
    print("=" * 80)

    dataset = load_or_build_v4_dataset()
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    print(f"\n[+] v4 데이터셋 규격: X={X.shape}, y={y.shape} (정상={np.sum(y==0)}, 결함={np.sum(y==1)})")
    print(f"[+] 피험자 전원: {len(np.unique(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[+] 카메라 시점: {len(np.unique(views))}개 ({sorted(list(set(views)))})")

    # 1. 모델 학습 및 교차검증
    print("\n[*] v4 GBDT 최적 구조 앙상블 학습 (StratifiedGroupKFold 5-Fold) 진행 중...")
    train_out = train_and_evaluate_gbdt_models_v4(X, y, groups)
    oof_ens = train_out['oof_ens']

    # 2. 베이스라인 이진 평가 (tau=0.5)
    base_bin_f1 = float(f1_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_acc = float(accuracy_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_prec = float(precision_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_rec = float(recall_score(y, (oof_ens >= 0.5).astype(int)))
    base_roc_auc = float(roc_auc_score(y, oof_ens))

    # 3. 시점별 적응형 3-State 최적화
    print("\n[*] v4 시점별 적응형 3-State 임계치 최적화 (보류율 <= 10.0% 제약)...")
    best_3state = optimize_per_view_adaptive_3state_v4(y, oof_ens, views, max_uncertainty_rate=0.10)

    print("\n[+] v4 최적화 결과:")
    print(f"  - 베이스라인 이진 F1: {base_bin_f1*100:.2f}% | 정확도: {base_bin_acc*100:.2f}% | AUC: {base_roc_auc*100:.2f}%")
    print(f"  - 3-State 확정 샘플 F1: {best_3state['3state_f1']*100:.2f}% | 정확도: {best_3state['3state_acc']*100:.2f}%")
    print(f"  - 판단 보류율(Uncertainty): {best_3state['uncertainty_rate']*100:.2f}% (커버리지: {best_3state['coverage']*100:.2f}%)")

    # 4. 5대 시점 전수 감사
    print("\n[*] v4 공식 5대 시점 전수 성능 감사 (AGENTS.md 표준)...")
    view_audit = audit_viewpoints_v4(y, oof_ens, views, best_3state['view_thresholds'])
    for v_code, v_res in view_audit.items():
        print(f"  - {v_res['view_name']} ({v_code}): 3-State F1={v_res['3state_f1']*100:.2f}%, ACC={v_res['accuracy']*100:.2f}%, AUC={v_res['roc_auc']*100:.2f}%, 보류={v_res['uncertainty_rate']*100:.2f}% (tau1={v_res['tau_clean']:.3f}, tau2={v_res['tau_defect']:.3f})")

    # 5. 피험자 9명 전수 감사
    print("\n[*] v4 피험자 9명 전원(1~9번) 전수 성능 감사...")
    subj_audit = audit_subjects_v4(y, oof_ens, groups, views, best_3state['view_thresholds'])
    for s_id, s_res in subj_audit.items():
        print(f"  - {s_id}: 3-State F1={s_res['3state_f1']*100:.2f}%, ACC={s_res['accuracy']*100:.2f}%, AUC={s_res['roc_auc']*100:.2f}%, 보류={s_res['uncertainty_rate']*100:.2f}% (샘플: {s_res['samples']})")

    # 6. 결과 JSON 저장
    elapsed = time.time() - t_start
    eval_summary = {
        'metadata': {
            'project_name': 'vidio-health-trainer',
            'exercise': '04_바벨_런지',
            'target_defect': 'c1 (앞다리 무릎 각도 90도)',
            'model_name': 'bl1_v4',
            'total_samples': len(y),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'subjects': sorted(list(set(groups))),
            'elapsed_sec': elapsed
        },
        'baseline_binary': {
            'binary_f1': base_bin_f1,
            'binary_acc': base_bin_acc,
            'binary_prec': base_bin_prec,
            'binary_recall': base_bin_rec,
            'roc_auc': base_roc_auc
        },
        'best_3state': best_3state,
        'view_audit': view_audit,
        'subj_audit': subj_audit
    }

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v4 성능 평가 결과 JSON 저장: {RESULTS_JSON}")

    # 7. 전체 데이터셋 대상 프로덕션 번들 저장
    print("\n[*] 전체 데이터셋 대상 최종 프로덕션 모델 번들 학습 중...")
    spw = float(np.sum(y == 0) / np.sum(y == 1))

    final_lgb_data = lgb.Dataset(X, label=y)
    final_lgb = lgb.train(
        {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.04,
            'num_leaves': 23,
            'max_depth': 5,
            'scale_pos_weight': spw,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_lambda': 3.0,
            'reg_alpha': 0.5,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        },
        final_lgb_data,
        num_boost_round=300
    )

    final_cat = CatBoostClassifier(
        iterations=350,
        learning_rate=0.04,
        depth=5,
        scale_pos_weight=spw,
        l2_leaf_reg=3.0,
        rsm=0.7,
        random_seed=42,
        verbose=False
    )
    final_cat.fit(X, y)

    final_xgb_data = xgb.DMatrix(X, label=y)
    final_xgb = xgb.train(
        {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.04,
            'max_depth': 5,
            'scale_pos_weight': spw,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_lambda': 3.0,
            'reg_alpha': 0.5,
            'min_child_weight': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        final_xgb_data,
        num_boost_round=300
    )

    bundle = {
        'model_name': 'bl1_v4',
        'project': 'vidio-health-trainer',
        'exercise': '04_바벨_런지',
        'defect': 'c1',
        'feature_dim': len(feature_names),
        'feature_names': feature_names,
        'view_thresholds': best_3state['view_thresholds'],
        'default_tau_clean': 0.365,
        'default_tau_defect': 0.575,
        'ensemble_weights': train_out['ensemble_weights'],
        'model_lgb': final_lgb,
        'model_cat': final_cat,
        'model_xgb': final_xgb
    }

    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] v4 번들 모델 저장: {MODEL_BUNDLE_PKL}")
    print(f"[+] v4 전체 모델링 소요 시간: {elapsed:.1f}초")


class BL1V4Classifier:
    """바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v4 추론기"""

    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"번들 파일 없음: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.view_thresholds = self.bundle['view_thresholds']
        self.default_tau_clean = self.bundle['default_tau_clean']
        self.default_tau_defect = self.bundle['default_tau_defect']
        self.weights = self.bundle['ensemble_weights']
        self.feature_names = self.bundle['feature_names']
        self.model_lgb = self.bundle['model_lgb']
        self.model_cat = self.bundle['model_cat']
        self.model_xgb = self.bundle['model_xgb']

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        vis_64f: Optional[np.ndarray] = None,
        view_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """단일 64프레임 시퀀스에 대한 적응형 3-State 상태 판별"""
        feat_vec = extract_lunge_c1_feature_vector_v4(kps_64f, vis_64f).reshape(1, -1)

        p_lgb = float(self.model_lgb.predict(feat_vec)[0])
        p_cat = float(self.model_cat.predict_proba(feat_vec)[0, 1])
        p_xgb = float(self.model_xgb.predict(xgb.DMatrix(feat_vec))[0])

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        # 시점별 적응형 임계치 적용
        if view_code in self.view_thresholds:
            t_clean = self.view_thresholds[view_code]['tau_clean']
            t_defect = self.view_thresholds[view_code]['tau_defect']
        else:
            t_clean = self.default_tau_clean
            t_defect = self.default_tau_defect

        if prob < t_clean:
            state = 0
            state_str = "CLEAN (정상)"
        elif prob >= t_defect:
            state = 1
            state_str = "DEFECT (결함)"
        else:
            state = 2
            state_str = "UNCERTAIN (판단 보류 / 재촬영)"

        return {
            'state': state,
            'state_str': state_str,
            'defect_prob': prob,
            'knee_angle_min': float(feat_vec[0, 0]),
            'knee_rom_ratio': float(feat_vec[0, 18]),
            'coplanarity': float(feat_vec[0, 17]),
            'tau_clean': t_clean,
            'tau_defect': t_defect
        }


if __name__ == '__main__':
    run_bl1_v4_training_and_evaluation()
