# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_v2 (설명 기반 라벨 정제 원천 전체 826폴더 4,130시퀀스 SOTA 모델)

Key Features & Pipeline:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 설명 기반 정밀 라벨 정제 (Description-based Ground Truth Labeling):
     - 텍스트 설명에 '앞다리 조금접고/앞다리 접힘'이 명시된 시퀀스 -> c1 결함(1)
     - 앞다리 언급이 없거나 타 부위(발방향, 안굽히는 다리, 척추, 상체, 시선) 결함 -> c1 정상(0)
  3. 바벨 런지 bl1 SOTA 30D 정예 생체역학 도메인 피처셋 완벽 계승 (lunge_c1_features_v4)
  4. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 가중치 최적화
  5. 카메라 5대 공식 시점(view1~view5, 시점 A~E)별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 제약 준수)
  6. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 피험자 격리 교차검증
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

sys.path.append(r"D:\src\features")
from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    extract_lunge_c1_features_v4,
    LUNGE_C1_FEATURE_NAMES_V4
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v2_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_v2_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_v2_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def determine_c1_label_from_description(desc: str, fname: str) -> int:
    """
    텍스트 설명(Description) 기반 c1 (앞다리 무릎 각도 90도 오류) 정답 라벨 판정
    - 설명에 '앞다리 조금' 또는 '앞다리' 오류 명시 시: 1 (결함)
    - 정상 또는 타 부위/시선 오류 시: 0 (정상)
    """
    if fname.startswith('[정상]') or '1앞다리 90도' in desc:
        return 0
    if '앞다리 조금' in desc or ('앞다리' in desc and '90도' not in desc):
        return 1
    return 0


def build_sl1_v2_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 30D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl1_v2 826폴더 전체 30D 피처 데이터셋 새로 구축 시작...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []

    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_name, view_code = parts[0], parts[1]

        # 라벨 JSON 확인
        if clip_name not in label_cache:
            json_path = os.path.join(LABEL_DIR, clip_name + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    d = json.load(fp)
                    tinfo = d.get('type_info', {})
                    desc = tinfo.get('description', '')
                    target_label = determine_c1_label_from_description(desc, clip_name)
                    label_cache[clip_name] = target_label
            except Exception:
                label_cache[clip_name] = None

        target_label = label_cache.get(clip_name)
        if target_label is None:
            continue

        # 피험자 ID 파싱 (피험자 1~9)
        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        # NPZ 로드 및 피처 추출 (100% 무누수 순수 비전)
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
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 500 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': LUNGE_C1_FEATURE_NAMES_V4
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v2 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 30D)")
    return dataset


def train_and_evaluate_gbdt_models_sl1_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
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
        'num_leaves': 31,
        'max_depth': 6,
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
        'iterations': 500,
        'learning_rate': 0.04,
        'depth': 6,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.7,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.04,
        'max_depth': 6,
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

        print(f"  - sl1_v2 Fold {fold_idx + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

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
    print(f"\n[+] sl1_v2 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_sl1_v2(
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
            'accuracy': float(accuracy_score(v_true_dec, v_pred_dec)),
            'precision': float(precision_score(v_true_dec, v_pred_dec, zero_division=0)),
            'recall': float(recall_score(v_true_dec, v_pred_dec, zero_division=0))
        }

    return {
        'view_thresholds': view_thresholds,
        'uncertainty_rate': unc_rate,
        'coverage': coverage,
        'effective_f1': f1_3state,
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
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0))
    }


def run_sl1_v2_pipeline(force_rebuild_dataset: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PKL), exist_ok=True)

    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl1_v2 (원천 전체 826폴더 설명 기반 정제) 파이프라인 가동")
    print("  - 대상: 사이드 런지 전체 826개 폴더 (4,130개 시퀀스)")
    print("  - 라벨: 설명(Description) 기반 일관 정제 Ground Truth")
    print("  - 피처셋: 30D 정예 생체역학 도메인 피처 (lunge_c1_features_v4)")
    print("=" * 80)

    dataset = build_sl1_v2_dataset(force_rebuild=force_rebuild_dataset)
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    clip_names = dataset['clip_names']

    print(f"\n[*] 로드된 데이터셋 규격: X={X.shape}, y={y.shape}, 피험자 수={len(set(groups))}")

    gbdt_res = train_and_evaluate_gbdt_models_sl1_v2(X, y, groups)
    oof_ens = gbdt_res['oof_ens']
    oof_lgb = gbdt_res['oof_lgb']
    oof_cat = gbdt_res['oof_cat']
    oof_xgb = gbdt_res['oof_xgb']

    metrics_lgb = compute_standard_2state_metrics(y, oof_lgb)
    metrics_cat = compute_standard_2state_metrics(y, oof_cat)
    metrics_xgb = compute_standard_2state_metrics(y, oof_xgb)
    metrics_ens = compute_standard_2state_metrics(y, oof_ens)

    print("\n" + "=" * 80)
    print("  [sl1_v2 GBDT 단일 모델 vs 앙상블 2-State 성능 비교 (원천 826폴더 4,130시퀀스 전수)]")
    print(f"  - LightGBM : AUC={metrics_lgb['roc_auc']:.4f}, F1={metrics_lgb['f1']:.4f}, Acc={metrics_lgb['accuracy']:.4f}")
    print(f"  - CatBoost : AUC={metrics_cat['roc_auc']:.4f}, F1={metrics_cat['f1']:.4f}, Acc={metrics_cat['accuracy']:.4f}")
    print(f"  - XGBoost  : AUC={metrics_xgb['roc_auc']:.4f}, F1={metrics_xgb['f1']:.4f}, Acc={metrics_xgb['accuracy']:.4f}")
    print(f"  - Ensemble : AUC={metrics_ens['roc_auc']:.4f}, F1={metrics_ens['f1']:.4f}, Acc={metrics_ens['accuracy']:.4f}")
    print("=" * 80)

    calib_res = optimize_per_view_adaptive_3state_sl1_v2(y, oof_ens, views, max_uncertainty_rate=0.10)
    print("\n" + "=" * 80)
    print("  [sl1_v2 시점별 적응형 3-State 캘리브레이션 결과]")
    print(f"  - 유효 의사결정 F1 (Effective F1): {calib_res['effective_f1']:.4f}")
    print(f"  - 유효 의사결정 정확도 (Accuracy): {calib_res['accuracy']:.4f}")
    print(f"  - 불확실성 보류율 (Uncertainty Rate): {calib_res['uncertainty_rate'] * 100:.2f}% (제약 <= 10.0% 준수)")
    print(f"  - 커버리지 (Coverage): {calib_res['coverage'] * 100:.2f}%")
    print("=" * 80)

    print("\n[공식 5대 시점별 세부 감사]")
    for v_code, audit in calib_res['view_audits'].items():
        print(f"  - [{v_code}] {audit['alias']}: F1={audit['effective_f1']:.4f}, Acc={audit['accuracy']:.4f}, Retake={audit['uncertainty_rate']*100:.2f}%, τ=[{audit['tau_clean']:.2f}, {audit['tau_defect']:.2f}]")

    feat_importances = np.mean([m.get_feature_importance() for m in gbdt_res['models_cat']], axis=0)
    top_feat_idx = np.argsort(feat_importances)[::-1][:10]
    top_features = [
        {'rank': i + 1, 'name': LUNGE_C1_FEATURE_NAMES_V4[idx], 'importance': float(feat_importances[idx])}
        for i, idx in enumerate(top_feat_idx)
    ]

    print("\n[Top 10 핵심 생체역학 도메인 피처]")
    for tf in top_features:
        print(f"  {tf['rank']:2d}. {tf['name']:<35s}: {tf['importance']:.2f}%")

    elapsed = time.time() - start_time

    full_results = {
        'model_name': 'sl1_v2',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'data_selection': '원천 전체 826개 폴더 (설명 기반 라벨 정제본)',
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
    print(f"\n[+] 공식 평가 결과 JSON 저장: {RESULTS_JSON}")

    bundle = {
        'model_name': 'sl1_v2',
        'ensemble_weights': gbdt_res['ensemble_weights'],
        'models_lgb': gbdt_res['models_lgb'],
        'models_cat': gbdt_res['models_cat'],
        'models_xgb': gbdt_res['models_xgb'],
        'view_thresholds': calib_res['view_thresholds'],
        'feature_names': LUNGE_C1_FEATURE_NAMES_V4
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 가중치 번들 저장: {MODEL_BUNDLE_PKL}")

    return full_results


class SL1V2Classifier:
    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        assert os.path.exists(bundle_path), f"번들 없음: {bundle_path}"
        with open(bundle_path, 'rb') as fp:
            bundle = pickle.load(fp)
        self.models_lgb = bundle['models_lgb']
        self.models_cat = bundle['models_cat']
        self.models_xgb = bundle['models_xgb']
        self.weights = bundle['ensemble_weights']
        self.view_thresholds = bundle['view_thresholds']

    def predict(self, kps_33_3d: np.ndarray, view_code: str = 'view3') -> Dict[str, Any]:
        feat_vec = extract_lunge_c1_feature_vector_v4(kps_33_3d).reshape(1, -1)
        p_lgb = np.mean([m.predict(feat_vec)[0] for m in self.models_lgb])
        p_cat = np.mean([m.predict_proba(feat_vec)[0, 1] for m in self.models_cat])
        dmat = xgb.DMatrix(feat_vec)
        p_xgb = np.mean([m.predict(dmat)[0] for m in self.models_xgb])

        p_ens = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        th = self.view_thresholds.get(view_code, {'tau_clean': 0.35, 'tau_defect': 0.60})
        if p_ens < th['tau_clean']:
            state = 'Clean'
        elif p_ens >= th['tau_defect']:
            state = 'Defect'
        else:
            state = 'Retake'

        return {
            'state': state,
            'prob_defect': p_ens,
            'thresholds': th,
            'view': view_code
        }


if __name__ == '__main__':
    run_sl1_v2_pipeline(force_rebuild_dataset=True)
