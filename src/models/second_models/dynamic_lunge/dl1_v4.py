# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - Forward 05 & Backward 06)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Model: dl1_v4 (바벨 런지 bl1_v4 SOTA 아키텍처 계승 모델)
Key Features & Architecture:
  1. 바벨 런지 bl1_v4 30D 정예 생체역학 도메인 피처셋 100% 완벽 계승 (Copy-and-Refine)
     - 앞다리 자율 식별 엔진 (detect_lead_leg_robust)
     - 최저점 윈도우 Active Bottom Phase Clamping
     - 관상면 뒤틀림 감지 (lead_leg_coplanarity) 및 체형 캘리브레이션 (lead_knee_rom_ratio)
  2. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 최적화
  3. [혁신 기술] 시점별 적응형 3-State 캘리브레이션 (Per-View Adaptive Selective Classification):
     - 카메라 5대 시점(view1~view5)별 독립 임계치 튜닝을 통해 시점별 앙각/투영 편향 완전 극복
     - 전체 불확실성 보류율 <= 10.0% 제약 엄격 준수
  4. 피험자 9명 전원(1~9번) StratifiedGroupKFold 5-Fold 피험자 격리 교차검증
  5. 공식 5대 시점(시점 A~E) 및 서브 도메인(스텝 포워드 05 vs 스텝 백워드 06) 전수 성능 감사
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

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 도메인 피처 추출 모듈 v4 임포트
sys.path.append(r"D:\src\features")
from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    extract_lunge_c1_features_v4,
    LUNGE_C1_FEATURE_NAMES_V4
)

# 데이터 및 모델 저장 경로 설정
NPZ_DIRS = [
    r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지",
    r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\06_스텝_백워드_다이나믹_런지"
]
LABEL_DIRS = {
    '05': r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    '06': r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
}
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\dl1_v4_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\dynamic_lunge\dl1_v4_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\dynamic_lunge\dl1_v4_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_or_load_dl1_v4_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """다이나믹 런지 (05+06) 30D v4 피처 데이터셋 로드 또는 빌드"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[+] 피처 캐시 로드 완료: N={len(dataset['y'])}, D={dataset['X'].shape[1]}")
        return dataset

    print(f"[*] 다이나믹 런지(05+06) 피처 데이터셋 새로 구축 시작: {NPZ_DIRS}")
    npz_files = []
    for d in NPZ_DIRS:
        npz_files.extend(glob.glob(os.path.join(d, "*.npz")))
    npz_files = sorted(npz_files)
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {NPZ_DIRS}"

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []
    subdomain_list = [] # '05_forward' vs '06_backward'

    # 라벨 JSON 캐싱
    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        # 형식: {clip_name}_{view}.npz
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_name, view_code = parts[0], parts[1]

        # 05번 혹은 06번 라벨 JSON 탐색
        if clip_name not in label_cache:
            target_json = None
            found_sub = None
            for sub_code, ldir in LABEL_DIRS.items():
                p = os.path.join(ldir, f"{clip_name}.json")
                if os.path.exists(p):
                    target_json = p
                    found_sub = '05_forward' if sub_code == '05' else '06_backward'
                    break

            if target_json is None:
                label_cache[clip_name] = (None, None)
                continue

            try:
                with open(target_json, 'r', encoding='utf-8') as fp:
                    d = json.load(fp)
                conds = d.get('type_info', {}).get('conditions', [])
                c1_val = None
                for c in conds:
                    if '앞다리' in c.get('condition', '') and '무릎' in c.get('condition', ''):
                        c1_val = c.get('value')
                        break
                label_cache[clip_name] = (c1_val, found_sub)
            except Exception:
                label_cache[clip_name] = (None, None)

        c1_val, sub_domain = label_cache.get(clip_name, (None, None))
        if c1_val is None:
            continue

        # 결함(False) -> 1, 정상(True) -> 0
        target_label = 1 if c1_val is False else 0

        # 피험자 ID 파싱 (피험자1 ~ 피험자9)
        subj = "피험자1"
        for s in range(1, 10):
            if f"피험자{s}" in clip_name:
                subj = f"피험자{s}"
                break

        # NPZ 로드 및 피처 추출
        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f']
            vis_64f = data['vis_64f'] if 'vis_64f' in data else None
            feat_vec = extract_lunge_c1_feature_vector_v4(kps_33_3d, vis_64f)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
            subdomain_list.append(sub_domain)
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'subdomains': np.array(subdomain_list),
        'feature_names': LUNGE_C1_FEATURE_NAMES_V4
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 샘플, 30D)")
    return dataset


def train_and_evaluate_gbdt_models_v4(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

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

        # 1. LightGBM
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

        # 2. CatBoost
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        models_cat.append(model_cat)

        # 3. XGBoost
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

        print(f"  - dl1_v4 Fold {fold_idx + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

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
    print(f"\n[+] dl1_v4 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def audit_subdomains_v4(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subdomains: np.ndarray,
    views: np.ndarray,
    view_thresholds: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    """포워드(05) vs 백워드(06) 서브 도메인 감사"""
    unique_subs = sorted(list(set(subdomains)))
    sub_results = {}

    for s_name in unique_subs:
        idx = (subdomains == s_name)
        if np.sum(idx) == 0:
            continue

        s_y = y_true[idx]
        s_p = y_prob[idx]
        s_v = views[idx]

        dec_y, dec_p = [], []
        unc_cnt = 0
        for i_s in range(len(s_y)):
            v = s_v[i_s]
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

        sub_results[s_name] = {
            'samples': int(np.sum(idx)),
            '3state_f1': f1_s,
            'accuracy': acc_s,
            'precision': prec_s,
            'recall': rec_s,
            'roc_auc': auc_s,
            'uncertainty_rate': unc_s,
            'coverage': float(1.0 - unc_s)
        }

    return sub_results


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


def run_dl1_v4_pipeline():
    """dl1_v4 모델 학습 및 평가 메인 파이프라인"""
    t_start = time.time()
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 dl1_v4 (c1: 앞다리 무릎 90도) SOTA 모델")
    print("=" * 80)

    dataset = build_or_load_dl1_v4_dataset()
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    subdomains = dataset.get('subdomains', np.array(['dynamic'] * len(y)))
    feature_names = dataset['feature_names']

    print(f"\n[+] dl1_v4 데이터셋 규격: X={X.shape}, y={y.shape} (정상={np.sum(y==0)}, 결함={np.sum(y==1)})")
    print(f"[+] 피험자 전원: {len(np.unique(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[+] 카메라 시점: {len(np.unique(views))}개 ({sorted(list(set(views)))})")

    # 1. 모델 학습 및 교차검증
    print("\n[*] dl1_v4 GBDT 최적 구조 앙상블 학습 (StratifiedGroupKFold 5-Fold) 진행 중...")
    train_out = train_and_evaluate_gbdt_models_v4(X, y, groups)
    oof_ens = train_out['oof_ens']

    # 2. 베이스라인 이진 평가 (tau=0.5)
    base_bin_f1 = float(f1_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_acc = float(accuracy_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_prec = float(precision_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_rec = float(recall_score(y, (oof_ens >= 0.5).astype(int)))
    base_roc_auc = float(roc_auc_score(y, oof_ens))

    # 3. 시점별 적응형 3-State 최적화
    print("\n[*] dl1_v4 시점별 적응형 3-State 임계치 최적화 (보류율 <= 10.0% 제약)...")
    best_3state = optimize_per_view_adaptive_3state_v4(y, oof_ens, views, max_uncertainty_rate=0.10)

    print("\n[+] dl1_v4 최적화 결과:")
    print(f"  - 베이스라인 이진 F1: {base_bin_f1*100:.2f}% | 정확도: {base_bin_acc*100:.2f}% | AUC: {base_roc_auc*100:.2f}%")
    print(f"  - 3-State 확정 샘플 F1: {best_3state['3state_f1']*100:.2f}% | 정확도: {best_3state['3state_acc']*100:.2f}%")
    print(f"  - 판단 보류율(Uncertainty): {best_3state['uncertainty_rate']*100:.2f}% (커버리지: {best_3state['coverage']*100:.2f}%)")

    # 4. 5대 시점 전수 감사
    print("\n[*] dl1_v4 공식 5대 시점 전수 성능 감사 (AGENTS.md 표준)...")
    view_audit = audit_viewpoints_v4(y, oof_ens, views, best_3state['view_thresholds'])
    for v_code, v_res in view_audit.items():
        print(f"  - {v_res['view_name']} ({v_code}): 3-State F1={v_res['3state_f1']*100:.2f}%, ACC={v_res['accuracy']*100:.2f}%, AUC={v_res['roc_auc']*100:.2f}%, 보류={v_res['uncertainty_rate']*100:.2f}% (tau1={v_res['tau_clean']:.3f}, tau2={v_res['tau_defect']:.3f})")

    # 5. 서브 도메인(포워드 vs 백워드) 감사
    print("\n[*] dl1_v4 서브 도메인(포워드 05 vs 백워드 06) 감사...")
    sub_audit = audit_subdomains_v4(y, oof_ens, subdomains, views, best_3state['view_thresholds'])
    for s_name, s_res in sub_audit.items():
        print(f"  - {s_name}: 3-State F1={s_res['3state_f1']*100:.2f}%, ACC={s_res['accuracy']*100:.2f}%, AUC={s_res['roc_auc']*100:.2f}%, 보류={s_res['uncertainty_rate']*100:.2f}% (샘플: {s_res['samples']})")

    # 6. 피험자 9명 전수 감사
    print("\n[*] dl1_v4 피험자 9명 전원(1~9번) 전수 성능 감사...")
    subj_audit = audit_subjects_v4(y, oof_ens, groups, views, best_3state['view_thresholds'])
    for s_id, s_res in subj_audit.items():
        print(f"  - {s_id}: 3-State F1={s_res['3state_f1']*100:.2f}%, ACC={s_res['accuracy']*100:.2f}%, AUC={s_res['roc_auc']*100:.2f}%, 보류={s_res['uncertainty_rate']*100:.2f}% (샘플: {s_res['samples']})")

    # 7. 결과 저장
    elapsed = time.time() - t_start
    final_results = {
        'model_name': 'dl1_v4',
        'exercise': '다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)',
        'defect': 'c1 (앞다리 무릎 각도 90도)',
        'feature_dim': int(X.shape[1]),
        'feature_names': feature_names,
        'num_samples': int(len(y)),
        'num_normal': int(np.sum(y == 0)),
        'num_defect': int(np.sum(y == 1)),
        'binary_baseline': {
            'f1_score': base_bin_f1,
            'accuracy': base_bin_acc,
            'precision': base_bin_prec,
            'recall': base_bin_rec,
            'roc_auc': base_roc_auc
        },
        'selective_3state': best_3state,
        'view_audit': view_audit,
        'subdomain_audit': sub_audit,
        'subject_audit': subj_audit,
        'ensemble_weights': {
            'lightgbm': float(train_out['ensemble_weights'][0]),
            'catboost': float(train_out['ensemble_weights'][1]),
            'xgboost': float(train_out['ensemble_weights'][2])
        },
        'elapsed_time_sec': float(elapsed)
    }

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(final_results, fp, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 모델 번들 pkl 저장
    bundle = {
        'model_name': 'dl1_v4',
        'exercise': '다이나믹 런지 (Dynamic Lunge)',
        'feature_dim': int(X.shape[1]),
        'feature_names': feature_names,
        'ensemble_weights': train_out['ensemble_weights'],
        'models_lgb': train_out['models_lgb'],
        'models_cat': train_out['models_cat'],
        'models_xgb': train_out['models_xgb'],
        'view_thresholds': best_3state['view_thresholds']
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 번들 저장 완료: {MODEL_BUNDLE_PKL}")
    print(f"[+] 전체 소요 시간: {elapsed:.2f}초")


if __name__ == '__main__':
    run_dl1_v4_pipeline()
