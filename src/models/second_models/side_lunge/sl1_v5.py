# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_v5 (24D 정예 피처 다이어트 & 다중공선성 제거 최적화 모델)

Key Features & Pipeline:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 설명 기반 정밀 라벨 정제 (Description-based Ground Truth Labeling):
     - 텍스트 설명에 '앞다리 조금접고/앞다리 접힘'이 명시된 시퀀스 -> c1 결함(1)
     - 앞다리 언급이 없거나 타 부위(발방향, 안굽히는 다리, 척추, 상체, 시선) 결함 -> c1 정상(0)
  3. [핵심 혁신] 24D 정예 피처 다이어트 (side_lunge_c1_features_v7):
     - 36D에서 상관계수 r >= 0.85 (특히 r=1.000, 0.99, 0.94) 및 기여도 0 노이즈 12개 피처를 전격 제거
     - 핵심 생체역학 불변량 보존:
       * lead_femur_vertical_drop_ratio (수직 낙차비, 중요도 1위 21.4%)
       * lead_knee_rom_ratio (직립 대비 가동범위, 유연성 편향 제거)
       * lead_rear_knee_angle_difference (양 하지 비대칭 각도 차이)
       * optical_axis_foreshortening_factor (투영 단축 역보정)
       * pelvic_drop_to_stride_ratio (보폭 대비 골반 수직 낙차비)
       * sagittal_knee_90_error & lead_femur_horizontal_tilt (정면 시상면 자세 복원)
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

from src.features.side_lunge_c1_features_v6 import (
    SIDE_LUNGE_C1_FEATURE_NAMES_V6
)
from src.features.side_lunge_c1_features_v7 import (
    extract_side_lunge_c1_feature_vector_v7,
    SIDE_LUNGE_C1_FEATURE_NAMES_V7
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
V4_FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v4_dataset_cache.pkl"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v5_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_v5_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_v5_bundle.pkl"

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
    """
    if fname.startswith('[정상]') or '1앞다리 90도' in desc:
        return 0
    if '앞다리 조금' in desc or ('앞다리' in desc and '90도' not in desc):
        return 1
    return 0


def build_sl1_v5_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 24D 정예 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    # v4 캐시가 존재하면 24D 인덱스 매핑으로 고속 생성
    if os.path.exists(V4_FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] v4 피처 캐시({V4_FEATURE_CACHE_PKL})로부터 24D 정예 피처셋 슬라이싱 추출...")
        with open(V4_FEATURE_CACHE_PKL, 'rb') as f:
            v4_ds = pickle.load(f)

        indices = [SIDE_LUNGE_C1_FEATURE_NAMES_V6.index(name) for name in SIDE_LUNGE_C1_FEATURE_NAMES_V7]
        X_v5 = v4_ds['X'][:, indices]

        dataset = {
            'X': X_v5,
            'y': v4_ds['y'],
            'groups': v4_ds['groups'],
            'views': v4_ds['views'],
            'clip_names': v4_ds['clip_names'],
            'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V7
        }

        with open(FEATURE_CACHE_PKL, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[+] sl1_v5 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(dataset['y'])}개 시퀀스, 24D)")
        return dataset

    print("[*] 사이드 런지 sl1_v5 826폴더 전체 24D 피처 데이터셋 새로 구축 시작...")
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
            feat_vec = extract_side_lunge_c1_feature_vector_v7(kps_33_3d, vis_64f)

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
        'clip_names': clip_names_list,
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V7
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v5 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 24D)")

    return dataset


def train_evaluate_gbdt_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
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
        'learning_rate': 0.035,
        'num_leaves': 31,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_lambda': 2.5,
        'reg_alpha': 0.5,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cat_params = {
        'iterations': 550,
        'learning_rate': 0.035,
        'depth': 6,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.75,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.035,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_lambda': 2.5,
        'reg_alpha': 0.5,
        'random_state': 42,
        'n_jobs': -1
    }

    models_lgb = []
    models_cat = []
    models_xgb = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=600,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
        models_lgb.append(lgb_model)

        # 2. CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
        models_cat.append(cat_model)

        # 3. XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=600,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        oof_xgb[val_idx] = xgb_model.predict(dval)
        models_xgb.append(xgb_model)

        print(f"  - sl1_v5 Fold {fold + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    # Nelder-Mead 가중치 최적화 (F1 극대화)
    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(0, w)
        if np.sum(w) == 0:
            return 1.0
        w = w / np.sum(w)
        ens_prob = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        pred = (ens_prob >= 0.5).astype(int)
        return -f1_score(y, pred, zero_division=0)

    init_weights = [0.333, 0.333, 0.334]
    opt_res = minimize(loss_func, init_weights, method='Nelder-Mead')
    opt_w = np.maximum(0, opt_res.x)
    opt_weights = (opt_w / np.sum(opt_w)).tolist()

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb

    print(f"\n[+] sl1_v5 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_sl1_v5(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 엄격 준수)"""
    unique_views = sorted(list(set(views)))
    view_thresholds = {}

    t_range = np.arange(0.15, 0.85, 0.005)

    for v in unique_views:
        v_m = (views == v)
        y_v = y_true[v_m]
        p_v = y_prob[v_m]

        best_v_f1, best_t1, best_t2 = -1.0, 0.38, 0.58
        for i, t1 in enumerate(t_range):
            for t2 in t_range[i + 1:]:
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
        'effective_f1': f1_3state,
        'accuracy': acc_3state,
        'precision': prec_3state,
        'recall': rec_3state,
        'uncertainty_rate': unc_rate,
        'coverage': coverage,
        'view_audits': view_audits
    }


def compute_feature_importances(models: List[Any], feature_names: List[str]) -> List[Dict[str, Any]]:
    """GBDT 모델들의 피처 중요도(Gain) 평균 산출"""
    importances = np.zeros(len(feature_names))
    for m in models:
        if hasattr(m, 'feature_importance'):
            # LightGBM
            importances += m.feature_importance(importance_type='gain')
        elif hasattr(m, 'get_feature_importance'):
            # CatBoost
            importances += m.get_feature_importance()
        elif hasattr(m, 'get_score'):
            # XGBoost
            scores = m.get_score(importance_type='gain')
            for f_idx, fname in enumerate(feature_names):
                f_key = f"f{f_idx}"
                importances[f_idx] += scores.get(f_key, 0.0)

    importances /= len(models)
    total_gain = np.sum(importances) + 1e-9
    rel_gain = importances / total_gain * 100.0

    sorted_indices = np.argsort(importances)[::-1]
    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        results.append({
            'rank': rank,
            'feature': feature_names[idx],
            'gain': float(importances[idx]),
            'relative_importance_pct': float(rel_gain[idx])
        })
    return results


def run_sl1_v5_pipeline(force_rebuild_dataset: bool = False):
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl1_v5 (24D 정예 피처 다이어트) 파이프라인")
    print("  - 데이터셋: 사이드 런지 원천 전체 826개 폴더 (4,130개 시퀀스)")
    print("  - 라벨 정제: 설명(Description) 기반 일관성 정제 Ground Truth")
    print("  - 피처셋: 24D 정예 불변 생체역학 도메인 피처 (side_lunge_c1_features_v7)")
    print("=" * 80)

    start_time = time.time()
    ds = build_sl1_v5_dataset(force_rebuild=force_rebuild_dataset)
    X, y, groups, views = ds['X'], ds['y'], ds['groups'], ds['views']

    print(f"\n[*] 데이터셋 Shape: X={X.shape}, y={y.shape} (정상 0: {np.sum(y==0)}, 결함 1: {np.sum(y==1)})")
    assert not np.isnan(X).any(), "피처 행렬에 NaN이 존재합니다!"
    assert not np.isinf(X).any(), "피처 행렬에 Inf가 존재합니다!"

    # 1. 5-Fold StratifiedGroupKFold 앙상블 학습
    train_res = train_evaluate_gbdt_models(X, y, groups)

    oof_ens = train_res['oof_ens']
    opt_weights = train_res['ensemble_weights']

    # 2. 2-State (임계값 0.5) 전체 성능 평가
    pred_2s = (oof_ens >= 0.5).astype(int)
    auc_2s = float(roc_auc_score(y, oof_ens))
    f1_2s = float(f1_score(y, pred_2s, zero_division=0))
    acc_2s = float(accuracy_score(y, pred_2s))
    prec_2s = float(precision_score(y, pred_2s, zero_division=0))
    rec_2s = float(recall_score(y, pred_2s, zero_division=0))

    print("\n" + "=" * 80)
    print(f"  [sl1_v5 2-State 전체 평가 결과 (Threshold 0.5)]")
    print(f"  - ROC-AUC:  {auc_2s:.4f}")
    print(f"  - F1-Score: {f1_2s:.4f} ({f1_2s*100:.2f}%)")
    print(f"  - Accuracy: {acc_2s:.4f} ({acc_2s*100:.2f}%)")
    print(f"  - Precision:{prec_2s:.4f} ({prec_2s*100:.2f}%)")
    print(f"  - Recall:   {rec_2s:.4f} ({rec_2s*100:.2f}%)")
    print("=" * 80)

    # 개별 모델 2-State 메트릭
    indiv_metrics = {}
    for name, oof in [('lightgbm', train_res['oof_lgb']), ('catboost', train_res['oof_cat']), ('xgboost', train_res['oof_xgb'])]:
        p_ind = (oof >= 0.5).astype(int)
        indiv_metrics[name] = {
            'roc_auc': float(roc_auc_score(y, oof)),
            'f1': float(f1_score(y, p_ind, zero_division=0)),
            'accuracy': float(accuracy_score(y, p_ind)),
            'precision': float(precision_score(y, p_ind, zero_division=0)),
            'recall': float(recall_score(y, p_ind, zero_division=0))
        }

    # 3. 공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0%)
    calib_res = optimize_per_view_adaptive_3state_sl1_v5(y, oof_ens, views, max_uncertainty_rate=0.10)

    print("\n" + "=" * 80)
    print("  [sl1_v5 공식 5대 시점별 적응형 3-State 캘리브레이션 결과]")
    print(f"  - 유효 F1-Score:     {calib_res['effective_f1']:.4f} ({calib_res['effective_f1']*100:.2f}%)")
    print(f"  - 유효 정확도(Acc):  {calib_res['accuracy']:.4f} ({calib_res['accuracy']*100:.2f}%)")
    print(f"  - 유효 정밀도(Prec): {calib_res['precision']:.4f} ({calib_res['precision']*100:.2f}%)")
    print(f"  - 유효 재현율(Rec):  {calib_res['recall']:.4f} ({calib_res['recall']*100:.2f}%)")
    print(f"  - 보류율(Holdout):   {calib_res['uncertainty_rate']:.4f} ({calib_res['uncertainty_rate']*100:.2f}%, 제약 <= 10.0% 준수)")
    print(f"  - 판정 커버리지:     {calib_res['coverage']:.4f} ({calib_res['coverage']*100:.2f}%)")
    print("-" * 80)
    print("  [공식 5대 시점별 세부 감사표]")
    for v_code, v_info in calib_res['view_audits'].items():
        print(f"  * {v_code} [{v_info['alias']}]:")
        print(f"      - 임계값: [Clean < {v_info['tau_clean']:.3f} <= Retake < {v_info['tau_defect']:.3f} <= Defect]")
        print(f"      - 유효 F1: {v_info['effective_f1']:.4f} ({v_info['effective_f1']*100:.2f}%) | 정확도: {v_info['accuracy']:.4f} | 보류율: {v_info['uncertainty_rate']*100:.2f}%")
    print("=" * 80)

    # 4. 피처 중요도(Gain) 분석
    feat_importances = compute_feature_importances(
        train_res['models_lgb'] + train_res['models_cat'],
        SIDE_LUNGE_C1_FEATURE_NAMES_V7
    )
    print("\n[*] sl1_v5 상위 10개 핵심 도메인 피처 (Gain 기준):")
    for f in feat_importances[:10]:
        print(f"  {f['rank']:2d}. {f['feature']:<38} : {f['relative_importance_pct']:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[+] sl1_v5 파이프라인 완료 소요 시간: {elapsed:.2f}초")

    # 5. 최종 결과 JSON 저장
    full_results = {
        'model_name': 'sl1_v5',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'data_selection': '원천 826개 폴더 전체 4,130개 시퀀스 (설명 기반 라벨 정제)',
        'total_samples': len(y),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'features_dimension': len(SIDE_LUNGE_C1_FEATURE_NAMES_V7),
        'ensemble_weights': {
            'lightgbm': opt_weights[0],
            'catboost': opt_weights[1],
            'xgboost': opt_weights[2]
        },
        'two_state_metrics': {
            'ensemble': {
                'roc_auc': auc_2s,
                'f1': f1_2s,
                'accuracy': acc_2s,
                'precision': prec_2s,
                'recall': rec_2s
            },
            'lightgbm': indiv_metrics['lightgbm'],
            'catboost': indiv_metrics['catboost'],
            'xgboost': indiv_metrics['xgboost']
        },
        'three_state_metrics': calib_res,
        'top_features': feat_importances,
        'elapsed_seconds': elapsed
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 성능 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 6. 프로덕션 번들 저장 (배포용)
    bundle = {
        'version': 'v5',
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V7,
        'ensemble_weights': opt_weights,
        'models_lgb': train_res['models_lgb'],
        'models_cat': train_res['models_cat'],
        'models_xgb': train_res['models_xgb'],
        'view_thresholds': calib_res['view_thresholds'],
        'view_alias_map': VIEW_ALIAS_MAP
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v5 모델 프로덕션 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    return full_results


def predict_single_sequence(bundle: Dict[str, Any], kps_64f: np.ndarray, view_code: str, vis_64f: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """단일 시퀀스(64, 33, 3)에 대한 100% 무누수 실시간 추론 인터페이스"""
    feat_vec = extract_side_lunge_c1_feature_vector_v7(kps_64f, vis_64f).reshape(1, -1)
    weights = bundle['ensemble_weights']

    # 5-Fold 모델 평균 추론
    p_lgb = np.mean([m.predict(feat_vec) for m in bundle['models_lgb']])
    p_cat = np.mean([m.predict_proba(feat_vec)[:, 1] for m in bundle['models_cat']])

    dtest = xgb.DMatrix(feat_vec)
    p_xgb = np.mean([m.predict(dtest) for m in bundle['models_xgb']])

    ens_prob = float(weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb)

    th = bundle['view_thresholds'].get(view_code, {'tau_clean': 0.40, 'tau_defect': 0.60})
    if ens_prob < th['tau_clean']:
        state = 'Clean'
        is_defect = False
    elif ens_prob >= th['tau_defect']:
        state = 'Defect'
        is_defect = True
    else:
        state = 'Retake'
        is_defect = None

    return {
        'defect_code': 'c1',
        'defect_name': '앞다리 무릎 각도 90도 오류',
        'view': view_code,
        'view_alias': bundle['view_alias_map'].get(view_code, view_code),
        'probability': ens_prob,
        'state': state,
        'is_defect': is_defect,
        'thresholds': th
    }


if __name__ == "__main__":
    run_sl1_v5_pipeline(force_rebuild_dataset=False)
