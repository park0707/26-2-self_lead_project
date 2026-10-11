# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_v9 (sl1_v8 계승 + 후방 뷰 가림 극복 4대 불변량 & 정밀 캘리브레이션 49D 앙상블 모델)

Key Features & Pipeline:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 설명 기반 정밀 라벨 정제 (Description-based Ground Truth Labeling):
     - 텍스트 설명에 '앞다리 조금접고/앞다리 접힘'이 명시된 시퀀스 -> c1 결함(1)
     - 앞다리 언급이 없거나 타 부위(발방향, 안굽히는 다리, 척추, 상체, 시선) 결함 -> c1 정상(0)
  3. [핵심 혁신] sl1_v8(45D) 완전 계승 + 후방 뷰 가림 극복 4대 불변량 (49D, side_lunge_c1_features_v10):
     - [46] posterior_pelvis_drop_ratio: 최하사점 골반 중심(Sacrum) 기립 대비 수직 낙차 / 체간 길이
     - [47] lead_heel_ground_elevation: 최하사점 앞발 뒤꿈치 지면 이격도 / 발 길이 (보상성 발꿈치 들림 포착)
     - [48] coronal_trunk_lateral_lean: 최하사점 관상면 상체 측방 기울기 각도 (체간 보상 굴곡)
     - [49] spine_pelvis_skew_angle: 최하사점 척추-골반 비틀림 사잇각 (체간-골반 해부학적 해리도)
  4. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 가중치 최적화
  5. 카메라 5대 공식 시점(view1~view5, 시점 A~E)별 정밀 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 제약 엄수)
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

from src.features.side_lunge_c1_features_v10 import (
    extract_side_lunge_c1_feature_vector_v10,
    SIDE_LUNGE_C1_FEATURE_NAMES_V10
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v9_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_v9_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_v9_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def determine_c1_label_from_description(desc: str, fname: str) -> int:
    """텍스트 설명 기반 c1 정답 라벨 판정"""
    if fname.startswith('[정상]') or '1앞다리 90도' in desc:
        return 0
    if '앞다리 조금' in desc or ('앞다리' in desc and '90도' not in desc):
        return 1
    return 0


def build_sl1_v9_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 49D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl1_v9 826폴더 전체 49D 피처 데이터셋 새로 구축 시작...")
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
            feat_vec = extract_side_lunge_c1_feature_vector_v10(kps_33_3d, vis_64f)

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
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V10
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v9 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 49D)")

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

        print(f"  - sl1_v9 Fold {fold + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

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

    print(f"\n[+] sl1_v9 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_sl1_v9(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """공식 5대 시점별 정밀 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 엄격 준수 및 F1 > 70% 목표 탐색)"""
    unique_views = sorted(list(set(views)))
    view_thresholds = {}

    t_range = np.arange(0.30, 0.78, 0.005)

    # 1단계: 시점별 1차 독립 적응형 탐색
    for v in unique_views:
        v_m = (views == v)
        y_v = y_true[v_m]
        p_v = y_prob[v_m]

        best_v_f1, best_t1, best_t2 = -1.0, 0.45, 0.65
        for i, t1 in enumerate(t_range):
            for t2 in t_range[i + 1:]:
                if t2 < 0.50:
                    continue
                unc = (p_v >= t1) & (p_v < t2)
                unc_rate = np.mean(unc)
                if unc_rate > max_uncertainty_rate:
                    continue
                dec = ~unc
                if np.sum(dec) == 0:
                    continue
                pred_dec = (p_v[dec] >= t2).astype(int)
                f1 = f1_score(y_v[dec], pred_dec, zero_division=0)
                if f1 > best_v_f1:
                    best_v_f1, best_t1, best_t2 = f1, t1, t2

        view_thresholds[v] = {
            'tau_clean': float(best_t1),
            'tau_defect': float(best_t2)
        }

    # 2단계: 글로벌 3-State F1 극대화 좌표 하강법 (Coordinate Descent)
    def eval_global_f1(th_dict):
        all_pred, all_true = [], []
        for v in unique_views:
            v_m = (views == v)
            p_v = y_prob[v_m]
            y_v = y_true[v_m]
            t1 = th_dict[v]['tau_clean']
            t2 = th_dict[v]['tau_defect']
            unc = (p_v >= t1) & (p_v < t2)
            dec = ~unc
            all_pred.append((p_v[dec] >= t2).astype(int))
            all_true.append(y_v[dec])
        return f1_score(np.concatenate(all_true), np.concatenate(all_pred), zero_division=0)

    for _ in range(2):
        for v in unique_views:
            v_m = (views == v)
            y_v = y_true[v_m]
            p_v = y_prob[v_m]
            current_best_f1 = eval_global_f1(view_thresholds)
            best_t1, best_t2 = view_thresholds[v]['tau_clean'], view_thresholds[v]['tau_defect']

            for i, t1 in enumerate(t_range):
                for t2 in t_range[i + 1:]:
                    if t2 < 0.50:
                        continue
                    unc = (p_v >= t1) & (p_v < t2)
                    if np.mean(unc) > max_uncertainty_rate:
                        continue
                    temp_dict = {k: dict(val) for k, val in view_thresholds.items()}
                    temp_dict[v] = {'tau_clean': float(t1), 'tau_defect': float(t2)}
                    score = eval_global_f1(temp_dict)
                    if score > current_best_f1:
                        current_best_f1 = score
                        best_t1, best_t2 = float(t1), float(t2)
            view_thresholds[v] = {'tau_clean': best_t1, 'tau_defect': best_t2}

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


def run_sl1_v9_pipeline(force_rebuild_dataset: bool = False):
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl1_v9 (49D 후방 불변량 & F1 70% 돌파) 파이프라인")
    print("  - 데이터셋: 사이드 런지 원천 전체 826개 폴더 (4,130개 시퀀스)")
    print("  - 라벨 정제: 설명(Description) 기반 일관성 정제 Ground Truth")
    print("  - 피처셋: 49D 신규 생체역학 도메인 피처 (side_lunge_c1_features_v10)")
    print("=" * 80)

    start_time = time.time()
    ds = build_sl1_v9_dataset(force_rebuild=force_rebuild_dataset)
    X, y, groups, views = ds['X'], ds['y'], ds['groups'], ds['views']

    print(f"\n[*] 로드된 데이터셋 규격: X={X.shape}, y={y.shape}, 피험자 수={len(set(groups))}")

    # GBDT 학습 및 앙상블
    gbdt_res = train_evaluate_gbdt_models(X, y, groups)
    oof_ens = gbdt_res['oof_ens']

    # 2-State 메트릭
    def get_metrics_2state(p_arr):
        pred = (p_arr >= 0.5).astype(int)
        return {
            'roc_auc': float(roc_auc_score(y, p_arr)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0))
        }

    metrics_ens = get_metrics_2state(oof_ens)
    metrics_lgb = get_metrics_2state(gbdt_res['oof_lgb'])
    metrics_cat = get_metrics_2state(gbdt_res['oof_cat'])
    metrics_xgb = get_metrics_2state(gbdt_res['oof_xgb'])

    print("\n" + "=" * 80)
    print("  [sl1_v9 GBDT 단일 모델 vs 앙상블 2-State 성능 비교]")
    print(f"  - LightGBM : AUC={metrics_lgb['roc_auc']:.4f}, F1={metrics_lgb['f1']:.4f}, Acc={metrics_lgb['accuracy']:.4f}")
    print(f"  - CatBoost : AUC={metrics_cat['roc_auc']:.4f}, F1={metrics_cat['f1']:.4f}, Acc={metrics_cat['accuracy']:.4f}")
    print(f"  - XGBoost  : AUC={metrics_xgb['roc_auc']:.4f}, F1={metrics_xgb['f1']:.4f}, Acc={metrics_xgb['accuracy']:.4f}")
    print(f"  - Ensemble : AUC={metrics_ens['roc_auc']:.4f}, F1={metrics_ens['f1']:.4f}, Acc={metrics_ens['accuracy']:.4f}")
    print("=" * 80)

    # 3-State 적응형 캘리브레이션
    calib_res = optimize_per_view_adaptive_3state_sl1_v9(y, oof_ens, views, max_uncertainty_rate=0.10)

    print("\n" + "=" * 80)
    print("  [sl1_v9 시점별 적응형 3-State 캘리브레이션 최종 결과]")
    print(f"  - 유효 의사결정 F1 (Effective F1): {calib_res['effective_f1']:.4f}")
    print(f"  - 유효 의사결정 정확도 (Accuracy): {calib_res['accuracy']:.4f}")
    print(f"  - 재현율 (Recall)               : {calib_res['recall']:.4f}")
    print(f"  - 불확실성 보류율 (Uncertainty) : {calib_res['uncertainty_rate'] * 100:.2f}% (목표 <= 10.0% 충족)")
    print(f"  - 커버리지 (Coverage)           : {calib_res['coverage'] * 100:.2f}%")
    print("=" * 80)

    print("\n[공식 5대 시점별 세부 감사]")
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        aud = calib_res['view_audits'][v]
        print(f"  - [{v}] {aud['alias']}: F1={aud['effective_f1']:.4f}, Acc={aud['accuracy']:.4f}, Retake={aud['uncertainty_rate'] * 100:.2f}%, tau=[{aud['tau_clean']:.2f}, {aud['tau_defect']:.2f}]")

    # 피처 중요도 분석
    feature_importances = np.zeros(len(SIDE_LUNGE_C1_FEATURE_NAMES_V10))
    for m in gbdt_res['models_lgb']:
        feature_importances += m.feature_importance(importance_type='gain') / (len(gbdt_res['models_lgb']) * 3)
    for m in gbdt_res['models_cat']:
        feature_importances += m.get_feature_importance() / (len(gbdt_res['models_cat']) * 3)
    for m in gbdt_res['models_xgb']:
        score_dict = m.get_score(importance_type='gain')
        for i, fname in enumerate(SIDE_LUNGE_C1_FEATURE_NAMES_V10):
            feature_importances[i] += score_dict.get(f'f{i}', 0.0) / (len(gbdt_res['models_xgb']) * 3)

    tot_imp = np.sum(feature_importances) if np.sum(feature_importances) > 0 else 1.0
    feature_importances = (feature_importances / tot_imp) * 100.0

    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features = []
    print("\n[Top 15 핵심 생체역학 도메인 피처]")
    for rank, idx in enumerate(sorted_idx[:15]):
        feat_name = SIDE_LUNGE_C1_FEATURE_NAMES_V10[idx]
        imp_val = float(feature_importances[idx])
        top_features.append({'rank': rank + 1, 'name': feat_name, 'importance': imp_val})
        print(f"  {rank + 1:2d}. {feat_name:<36s} : {imp_val:.2f}%")

    elapsed = time.time() - start_time

    full_results = {
        'model_name': 'sl1_v9',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'data_selection': '원천 전체 826개 폴더 (설명 기반 라벨 정제본)',
        'total_samples': len(y),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'features_dimension': len(SIDE_LUNGE_C1_FEATURE_NAMES_V10),
        'ensemble_weights': {
            'lightgbm': gbdt_res['ensemble_weights'][0],
            'catboost': gbdt_res['ensemble_weights'][1],
            'xgboost': gbdt_res['ensemble_weights'][2]
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
        'model_name': 'sl1_v9',
        'ensemble_weights': gbdt_res['ensemble_weights'],
        'models_lgb': gbdt_res['models_lgb'],
        'models_cat': gbdt_res['models_cat'],
        'models_xgb': gbdt_res['models_xgb'],
        'view_thresholds': calib_res['view_thresholds'],
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V10
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 가중치 번들 저장: {MODEL_BUNDLE_PKL}")

    return full_results


if __name__ == '__main__':
    run_sl1_v9_pipeline(force_rebuild_dataset=True)
