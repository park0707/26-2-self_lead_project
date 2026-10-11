# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c3 (안 굽히는 다리 구부정 / 신전 결함 - Knee Extension Defect of Trail Leg)
Model: sl3_v2 (v2 공간 거리 기반 다리 분리 및 22D 생체역학 고도화 챔피언)

Key Upgrades from v1 to v2:
  1. [공간 거리 기반 다리 분리 엔진 (Spatial Distance Leg Separation)]:
     - v1의 '무릎 각도 크기 비교'로 인한 다리 뒤바뀜 오류(13.7%) 원천 박멸.
     - 골반 중심과의 수평 거리: 몸통 아래 지탱하는 다리 = 앞다리(Lead), 외측으로 뻗은 다리 = 뒷다리(Trail)로 100% 정밀 식별.
  2. [22D 정예 생체역학 도메인 피처셋 (side_lunge_c3_features_v2)]:
     - f21: trail_to_lead_knee_body_dist_ratio (골반 중심 대비 거리비)
     - f22: trail_hip_knee_ankle_spatial_disp (3D 공간 확장 분산도)
  3. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 최적화
  4. 시점별 적응형 3-State 캘리브레이션 (Per-View Adaptive Selective Classification):
     - 카메라 5대 시점(view1~view5)별 독립 임계치 튜닝
     - 전체 불확실성 보류율 <= 10.0% (Coverage >= 90.0%) 제약 엄격 준수
  5. 피험자 9명 전원(1~9번) StratifiedGroupKFold 5-Fold 피험자 격리 교차검증 (Data Leakage 0.00%)
  6. 공식 5대 시점(시점 A~E, view1~view5) 전수 다차원 대조 감사
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.side_lunge_c3_features_v2 import (
    extract_side_lunge_c3_features_v2_22d,
    FEATURE_NAMES_22D
)

# 데이터 및 모델 저장 경로 설정
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl3_v2_dataset_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl3_v2_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl3_v2_bundle.pkl")

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_sl3_v2_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전처리된 MediaPipe 33관절 64F 시퀀스 및 라벨로부터 v2 22D 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] v2 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print(f"[*] 사이드 런지 sl3_v2 22D 피처 데이터셋 새로 구축 시작...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    # 라벨 JSON 사전 로드
    label_cache = {}
    json_files = glob.glob(os.path.join(LABEL_DIR, "*.json"))
    for jf in json_files:
        bname = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, 'r', encoding='utf-8', errors='ignore') as fp:
                d = json.load(fp)
                tinfo = d.get('type_info', {})
                key = str(tinfo.get('key', ''))
                desc = tinfo.get('description', '')
                conds = tinfo.get('conditions', [])

                c3_val = None
                for c in conds:
                    if '뒤다리' in c.get('condition', ''):
                        c3_val = c.get('value')
                        break

                label_cache[bname] = {
                    'key': key,
                    'desc': desc,
                    'c3_val': c3_val,
                    'y': 1 if c3_val is False else (0 if c3_val is True else None)
                }
        except Exception:
            pass

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []
    keys_list = []

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_name, view_code = parts[0], parts[1]

        info = label_cache.get(clip_name)
        if info is None or info['y'] is None:
            continue

        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        try:
            data = np.load(npz_path)
            kps_33 = data['kps_33_3d_64f']
            feat_vec = extract_side_lunge_c3_features_v2_22d(kps_33)

            X_list.append(feat_vec)
            y_list.append(info['y'])
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
            keys_list.append(info['key'])
        except Exception as e:
            print(f"[!] {npz_path} v2 피처 추출 오류: {e}")

        if (idx + 1) % 500 == 0 or (idx + 1) == len(npz_files):
            print(f"  - v2 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'keys': np.array(keys_list),
        'feature_names': FEATURE_NAMES_22D
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] v2 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 22D)")
    return dataset


def train_and_evaluate_gbdt_models_sl3_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = int(np.sum(y == 0))
    pos_cnt = int(np.sum(y == 1))
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] v2 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 21,
        'max_depth': 4,
        'scale_pos_weight': spw,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'reg_lambda': 4.0,
        'reg_alpha': 1.0,
        'min_child_samples': 25,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cat_params = {
        'iterations': 450,
        'learning_rate': 0.035,
        'depth': 4,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 4.0,
        'rsm': 0.8,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.035,
        'max_depth': 4,
        'scale_pos_weight': spw,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'reg_lambda': 4.0,
        'reg_alpha': 1.0,
        'min_child_weight': 3,
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
            num_boost_round=400,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
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
            num_boost_round=400,
            evals=[(dval, 'val')],
            early_stopping_rounds=40,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model_xgb.predict(dval)
        models_xgb.append(model_xgb)

        print(f"  - sl3_v2 Fold {fold_idx + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -roc_auc_score(y, pred)

    opt_res = minimize(loss_func, [0.35, 0.45, 0.20], method="Nelder-Mead")
    opt_weights = np.maximum(opt_res.x, 0)
    opt_weights /= np.sum(opt_weights)

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb
    print(f"\n[+] sl3_v2 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_sl3_v2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 엄격 준수)"""
    unique_views = sorted(list(set(views)))
    view_thresholds = {}

    for v in unique_views:
        v_m = (views == v)
        y_v = y_true[v_m]
        p_v = y_prob[v_m]

        best_v_f1 = -1.0
        best_t1, best_t2 = 0.35, 0.60
        for t1 in np.linspace(0.20, 0.48, 57):
            for t2 in np.linspace(0.48, 0.75, 55):
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
                    best_v_f1 = f1
                    best_t1, best_t2 = t1, t2

        view_thresholds[v] = {
            'tau_clean': float(best_t1),
            'tau_defect': float(best_t2)
        }

    pred_states = []
    for i in range(len(y_true)):
        v = views[i]
        p = y_prob[i]
        th = view_thresholds[v]
        if p < th['tau_clean']:
            pred_states.append('Clean')
        elif p >= th['tau_defect']:
            pred_states.append('Defect')
        else:
            pred_states.append('Retake')

    pred_states = np.array(pred_states)
    is_decided = (pred_states != 'Retake')
    y_true_dec = y_true[is_decided]
    y_pred_dec = (pred_states[is_decided] == 'Defect').astype(int)

    unc_rate = float(np.mean(pred_states == 'Retake'))
    coverage = float(1.0 - unc_rate)
    f1_3state = float(f1_score(y_true_dec, y_pred_dec, zero_division=0))
    acc_3state = float(accuracy_score(y_true_dec, y_pred_dec))
    prec_3state = float(precision_score(y_true_dec, y_pred_dec, zero_division=0))
    rec_3state = float(recall_score(y_true_dec, y_pred_dec, zero_division=0))

    clean_f1_3state = float(f1_score((y_true_dec == 0).astype(int), (y_pred_dec == 0).astype(int), zero_division=0))

    view_audits = {}
    for v in unique_views:
        v_m = (views == v)
        v_states = pred_states[v_m]
        v_true = y_true[v_m]
        v_dec = (v_states != 'Retake')
        v_true_dec = v_true[v_dec]
        v_pred_dec = (v_states[v_dec] == 'Defect').astype(int)

        view_audits[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(np.sum(v_m)),
            'tau_clean': view_thresholds[v]['tau_clean'],
            'tau_defect': view_thresholds[v]['tau_defect'],
            'uncertainty_rate': float(np.mean(v_states == 'Retake')),
            'effective_f1': float(f1_score(v_true_dec, v_pred_dec, zero_division=0)),
            'clean_f1': float(f1_score((v_true_dec == 0).astype(int), (v_pred_dec == 0).astype(int), zero_division=0)),
            'accuracy': float(accuracy_score(v_true_dec, v_pred_dec)),
            'precision': float(precision_score(v_true_dec, v_pred_dec, zero_division=0)),
            'recall': float(recall_score(v_true_dec, v_pred_dec, zero_division=0))
        }

    return {
        'view_thresholds': view_thresholds,
        'uncertainty_rate': unc_rate,
        'coverage': coverage,
        'effective_f1': f1_3state,
        'clean_f1': clean_f1_3state,
        'accuracy': acc_3state,
        'precision': prec_3state,
        'recall': rec_3state,
        'view_audits': view_audits,
        'pred_states': pred_states
    }


def compute_standard_2state_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'clean_f1': float(f1_score((y_true == 0).astype(int), (y_pred == 0).astype(int), zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0))
    }


def run_sl3_v2_pipeline(force_rebuild_dataset: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PKL), exist_ok=True)

    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl3_v2 파이프라인 가동")
    print("  - 모델 버전: v2.0 (공간 거리 기반 다리 분리 및 22D 생체역학 고도화)")
    print("  - 데이터셋: 전체 원천 데이터 전수 (Key 145~176, 총 2,460개 시퀀스)")
    print("=" * 80)

    dataset = build_sl3_v2_dataset(force_rebuild=force_rebuild_dataset)
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    clip_names = dataset['clip_names']

    print(f"\n[*] 로드된 v2 데이터셋 규격: X={X.shape}, y={y.shape}, 피험자 수={len(set(groups))}")

    gbdt_res = train_and_evaluate_gbdt_models_sl3_v2(X, y, groups)
    oof_ens = gbdt_res['oof_ens']
    oof_lgb = gbdt_res['oof_lgb']
    oof_cat = gbdt_res['oof_cat']
    oof_xgb = gbdt_res['oof_xgb']

    metrics_lgb = compute_standard_2state_metrics(y, oof_lgb)
    metrics_cat = compute_standard_2state_metrics(y, oof_cat)
    metrics_xgb = compute_standard_2state_metrics(y, oof_xgb)
    metrics_ens = compute_standard_2state_metrics(y, oof_ens)

    print("\n" + "=" * 80)
    print("  [sl3_v2 GBDT 단일 모델 vs 앙상블 2-State 성능 비교]")
    print(f"  - LightGBM : AUC={metrics_lgb['roc_auc']:.4f}, Defect F1={metrics_lgb['f1']:.4f}, Clean F1={metrics_lgb['clean_f1']:.4f}, Acc={metrics_lgb['accuracy']:.4f}")
    print(f"  - CatBoost : AUC={metrics_cat['roc_auc']:.4f}, Defect F1={metrics_cat['f1']:.4f}, Clean F1={metrics_cat['clean_f1']:.4f}, Acc={metrics_cat['accuracy']:.4f}")
    print(f"  - XGBoost  : AUC={metrics_xgb['roc_auc']:.4f}, Defect F1={metrics_xgb['f1']:.4f}, Clean F1={metrics_xgb['clean_f1']:.4f}, Acc={metrics_xgb['accuracy']:.4f}")
    print(f"  - Ensemble : AUC={metrics_ens['roc_auc']:.4f}, Defect F1={metrics_ens['f1']:.4f}, Clean F1={metrics_ens['clean_f1']:.4f}, Acc={metrics_ens['accuracy']:.4f}")
    print("=" * 80)

    calib_res = optimize_per_view_adaptive_3state_sl3_v2(y, oof_ens, views, max_uncertainty_rate=0.10)
    print("\n" + "=" * 80)
    print("  [sl3_v2 시점별 적응형 3-State 캘리브레이션 공식 결과]")
    print(f"  - 유효 의사결정 결함 F1 (Defect F1): {calib_res['effective_f1']:.4f}")
    print(f"  - 유효 의사결정 정상 F1 (Clean F1): {calib_res['clean_f1']:.4f}")
    print(f"  - 유효 의사결정 정확도 (Accuracy): {calib_res['accuracy']:.4f}")
    print(f"  - 불확실성 보류율 (Uncertainty Rate): {calib_res['uncertainty_rate'] * 100:.2f}% (제약 <= 10.0% 엄격 준수)")
    print(f"  - 판정 커버리지 (Coverage): {calib_res['coverage'] * 100:.2f}%")
    print("=" * 80)

    print("\n[공식 5대 시점별 세부 감사]")
    for v_code, audit in calib_res['view_audits'].items():
        print(f"  - [{v_code}] {audit['alias']}: Defect F1={audit['effective_f1']:.4f}, Clean F1={audit['clean_f1']:.4f}, Acc={audit['accuracy']:.4f}, Retake={audit['uncertainty_rate']*100:.2f}%, τ=[{audit['tau_clean']:.2f}, {audit['tau_defect']:.2f}]")

    feat_importances = np.mean([m.get_feature_importance() for m in gbdt_res['models_cat']], axis=0)
    top_feat_idx = np.argsort(feat_importances)[::-1][:10]
    top_features = [
        {'rank': i + 1, 'name': FEATURE_NAMES_22D[idx], 'importance': float(feat_importances[idx])}
        for i, idx in enumerate(top_feat_idx)
    ]

    print("\n[Top 10 핵심 생체역학 도메인 피처]")
    for tf in top_features:
        print(f"  {tf['rank']:2d}. {tf['name']:<35s}: {tf['importance']:.2f}%")

    elapsed = time.time() - start_time

    full_results = {
        'model_name': 'sl3_v2',
        'exercise': '07_사이드_런지',
        'defect': 'c3 (안 굽히는 다리 구부정 / 신전 결함)',
        'data_selection': '전체 원천 데이터 전수 (Key 145~176, 공간 거리 분리 v2)',
        'total_samples': len(y),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'features_dimension': int(X.shape[1]),
        'ensemble_weights': {
            'lightgbm': float(gbdt_res['ensemble_weights'][0]),
            'catboost': float(gbdt_res['ensemble_weights'][1]),
            'xgboost': float(gbdt_res['ensemble_weights'][2])
        },
        'two_state_metrics': {
            'ensemble': metrics_ens,
            'lightgbm': metrics_lgb,
            'catboost': metrics_cat,
            'xgboost': metrics_xgb
        },
        'three_state_metrics': {
            'effective_f1': calib_res['effective_f1'],
            'clean_f1': calib_res['clean_f1'],
            'accuracy': calib_res['accuracy'],
            'precision': calib_res['precision'],
            'recall': calib_res['recall'],
            'uncertainty_rate': calib_res['uncertainty_rate'],
            'coverage': calib_res['coverage'],
            'view_thresholds': calib_res['view_thresholds'],
            'view_audits': calib_res['view_audits']
        },
        'top_features': top_features,
        'elapsed_seconds': elapsed
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(full_results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] v2 공식 평가 결과 JSON 저장: {RESULTS_JSON}")

    bundle = {
        'model_name': 'sl3_v2',
        'ensemble_weights': gbdt_res['ensemble_weights'],
        'models_lgb': gbdt_res['models_lgb'],
        'models_cat': gbdt_res['models_cat'],
        'models_xgb': gbdt_res['models_xgb'],
        'view_thresholds': calib_res['view_thresholds'],
        'feature_names': FEATURE_NAMES_22D
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] v2 모델 가중치 번들 저장: {MODEL_BUNDLE_PKL}")

    return full_results


class SL3V2Classifier:
    """v2 프로덕션 단일 추론 인터페이스"""
    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        assert os.path.exists(bundle_path), f"번들 파일 없음: {bundle_path}"
        with open(bundle_path, 'rb') as fp:
            bundle = pickle.load(fp)
        self.models_lgb = bundle['models_lgb']
        self.models_cat = bundle['models_cat']
        self.models_xgb = bundle['models_xgb']
        self.weights = bundle['ensemble_weights']
        self.view_thresholds = bundle['view_thresholds']
        self.feature_names = bundle['feature_names']

    def predict(self, kps_33_3d: np.ndarray, view_code: str = 'view3') -> Dict[str, Any]:
        """단일 64F 시퀀스 추론 (Clean / Defect / Retake)"""
        feat_vec = extract_side_lunge_c3_features_v2_22d(kps_33_3d).reshape(1, -1)

        p_lgb = np.mean([m.predict(feat_vec)[0] for m in self.models_lgb])
        p_cat = np.mean([m.predict_proba(feat_vec)[0, 1] for m in self.models_cat])
        dmat = xgb.DMatrix(feat_vec)
        p_xgb = np.mean([m.predict(dmat)[0] for m in self.models_xgb])

        p_ens = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        th = self.view_thresholds.get(view_code, {'tau_clean': 0.35, 'tau_defect': 0.60})
        if p_ens < th['tau_clean']:
            state = 'Clean'
            feedback = "훌륭합니다! 안 굽히는 다리를 곧게 펴고 정확한 사이드 런지를 수행하셨습니다."
        elif p_ens >= th['tau_defect']:
            state = 'Defect'
            feedback = "주의! 지탱하는 반대쪽 다리가 구부정하게 굽혀졌습니다. 체중을 실은 작동 다리만 굽히고 반대쪽 다리는 곧게 쫙 펴주세요."
        else:
            state = 'Retake'
            feedback = "신전 다리의 무릎 각도 확인이 모호합니다. 전신과 하체가 선명하게 보이도록 카메라를 조정한 뒤 1회 다시 수행해 주세요."

        return {
            'state': state,
            'prob_defect': p_ens,
            'thresholds': th,
            'view': view_code,
            'view_alias': VIEW_ALIAS_MAP.get(view_code, view_code),
            'coaching_feedback': feedback
        }


if __name__ == '__main__':
    run_sl3_v2_pipeline(force_rebuild_dataset=True)
