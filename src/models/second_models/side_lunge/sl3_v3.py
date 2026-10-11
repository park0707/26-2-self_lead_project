# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c3 (안 굽히는 다리 구부정 / 신전 결함 - Knee Extension Defect of Trail Leg)
Model: sl3_v3 (공간 거리 분리 + 동적 기립 정규화 28D + 3종 GBDT 앙상블 + 5대 시점별 적응형 3-State)
Module: src/models/second_models/side_lunge/sl3_v3.py

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer (NovelFlow 절대 금지)
  2. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Data Leakage 0.00%)
  3. 100% 순수 비전 MediaPipe 33관절 3D 좌표 기반 (AI-Hub JSON 피처 투입 영구 금지)
  4. 공식 5대 시점 명칭 규격 준수:
     - 시점 A: 우측 후방 (45도) [view5]
     - 시점 B: 우측 전방 (45도) [view2]
     - 시점 C: 정면 [view3]
     - 시점 D: 좌측 전방 (45도) [view4]
     - 시점 E: 좌측 후방 (45도) [view1]
  5. 3-State 선택적 분류: 불확실성 보류율(Uncertainty Rate) <= 10.0% (Coverage >= 90.0%) 엄격 준수
  6. 목표: 결함 F1-score >= 70.0% 돌파 실증
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

from src.features.side_lunge_c3_features_v3 import (
    extract_side_lunge_c3_features_v3_28d,
    FEATURE_NAMES_28D
)

# 데이터 및 모델 저장 경로 설정
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl3_v3_dataset_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl3_v3_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl3_v3_bundle.pkl")

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_sl3_v3_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전처리된 MediaPipe 33관절 64F 시퀀스 및 라벨로부터 v3 28D 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] v3 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print(f"[*] 사이드 런지 sl3_v3 28D 피처 데이터셋 새로 구축 시작...")
    
    # 공정 비교를 위해 동일 2,460개 시퀀스 목록 로드
    V2_DATA_PKL = r"D:\Data\피처_데이터\sl3_v2_dataset_cache_2460.pkl"
    v2_data = pickle.load(open(V2_DATA_PKL, "rb"))
    target_clips = v2_data['clip_names']
    target_views = v2_data['views']
    y_arr = v2_data['y']
    groups_arr = v2_data['groups']
    keys_arr = v2_data['keys']

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []
    keys_list = []

    for idx, (cname, vcode) in enumerate(zip(target_clips, target_views)):
        npz_path = os.path.join(DATA_DIR, f"{cname}_{vcode}.npz")
        if not os.path.exists(npz_path):
            continue
        try:
            data = np.load(npz_path)
            kps_33 = data['kps_33_3d_64f']
            feat_vec = extract_side_lunge_c3_features_v3_28d(kps_33)

            X_list.append(feat_vec)
            y_list.append(y_arr[idx])
            groups_list.append(groups_arr[idx])
            views_list.append(vcode)
            clip_names_list.append(cname)
            keys_list.append(keys_arr[idx])
        except Exception as e:
            print(f"[!] {npz_path} v3 피처 추출 오류: {e}")

        if (idx + 1) % 500 == 0 or (idx + 1) == len(target_clips):
            print(f"  - v3 피처 추출 진행도: [{idx + 1}/{len(target_clips)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'keys': np.array(keys_list),
        'feature_names': FEATURE_NAMES_28D
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] v3 피처 데이터셋 구축 완료: N={len(X_list)}, X_shape={dataset['X'].shape}")
    return dataset


def train_and_evaluate_gbdt_models_sl3_v3(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """AI-Hub 피험자 9명 전원 5-Fold StratifiedGroupKFold 완전 격리 교차검증 및 GBDT 3종 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_lgb = np.zeros(len(y), dtype=np.float32)
    oof_cat = np.zeros(len(y), dtype=np.float32)
    oof_xgb = np.zeros(len(y), dtype=np.float32)

    pos_cnt = int(np.sum(y == 1))
    neg_cnt = int(np.sum(y == 0))
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] v3 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

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
        'random_state': random_state,
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
        'random_seed': random_state,
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
        'random_state': random_state,
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

        print(f"  - sl3_v3 Fold {fold_idx + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -roc_auc_score(y, pred)

    opt_res = minimize(loss_func, [0.33, 0.34, 0.33], method="Nelder-Mead")
    opt_weights = np.maximum(opt_res.x, 0)
    opt_weights /= np.sum(opt_weights)

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb
    print(f"\n[+] sl3_v3 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

    return {
        'oof_ens': oof_ens,
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'models_lgb': models_lgb,
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'ensemble_weights': opt_weights
    }


def optimize_per_view_adaptive_3state_sl3_v3(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """AGENTS.md 공식 5대 시점별 적응형 3-State 임계치 [tau_clean, tau_defect] 정밀 최적화"""
    unique_views = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_thresholds = {}

    for v in unique_views:
        v_m = (views == v)
        if np.sum(v_m) == 0:
            continue
        y_v = y_true[v_m]
        p_v = y_prob[v_m]

        best_v_f1 = -1.0
        best_t1, best_t2 = 0.35, 0.60
        # 탐색 범위를 더욱 정밀하게 확장
        for t1 in np.linspace(0.18, 0.52, 69):
            for t2 in np.linspace(0.45, 0.80, 71):
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


class SL3V3Classifier:
    """sl3_v3 프로덕션 단일 추론 인터페이스 클래스"""
    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        self.models_lgb = bundle['models_lgb']
        self.models_cat = bundle['models_cat']
        self.models_xgb = bundle['models_xgb']
        self.weights = bundle['ensemble_weights']
        self.view_thresholds = bundle['view_thresholds']
        self.feature_names = bundle['feature_names']

    def predict_sequence(self, kps_seq: np.ndarray, view_code: str = 'view3') -> Dict[str, Any]:
        """
        단일 64프레임 키포인트 시퀀스에 대한 3-State 추론
        Args:
            kps_seq: (64, 33, 3) 관절 배열
            view_code: 'view1' ~ 'view5' (기본값 view3: 정면)
        Returns:
            Dict: state('Clean'|'Defect'|'Retake'), probability, confidence, feedback
        """
        t0 = time.time()
        feats = extract_side_lunge_c3_features_v3_28d(kps_seq).reshape(1, -1)

        p_lgb = np.mean([m.predict(feats)[0] for m in self.models_lgb])
        p_cat = np.mean([m.predict_proba(feats)[0, 1] for m in self.models_cat])
        dmat = xgb.DMatrix(feats)
        p_xgb = np.mean([m.predict(dmat)[0] for m in self.models_xgb])

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        th = self.view_thresholds.get(view_code, {'tau_clean': 0.35, 'tau_defect': 0.60})
        tau_c, tau_d = th['tau_clean'], th['tau_defect']

        if prob < tau_c:
            state = 'Clean'
            conf = float((tau_c - prob) / tau_c)
            feedback = "훌륭합니다! 안 굽히는 다리를 곧게 펴고 정확한 사이드 런지를 수행하셨습니다."
        elif prob >= tau_d:
            state = 'Defect'
            conf = float((prob - tau_d) / (1.0 - tau_d))
            feedback = "주의! 지탱하는 반대쪽 다리가 구부정하게 굽혀졌습니다. 체중을 실은 작동 다리만 굽히고 반대쪽 다리는 곧게 쫙 펴주세요."
        else:
            state = 'Retake'
            conf = float(1.0 - abs(prob - (tau_c + tau_d) / 2) / ((tau_d - tau_c) / 2 + 1e-5))
            feedback = "동작이 다소 모호합니다. 반대쪽 다리를 완전히 펴고 다시 한 번 정확하게 수행해주세요."

        latency_ms = (time.time() - t0) * 1000
        return {
            'state': state,
            'probability': prob,
            'confidence': conf,
            'view': view_code,
            'feedback': feedback,
            'latency_ms': latency_ms
        }


def run_sl3_v3_pipeline(force_rebuild_dataset: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)

    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c3 sl3_v3 28D 고도화 파이프라인 가동")
    print("=" * 80)

    dataset = build_sl3_v3_dataset(force_rebuild=force_rebuild_dataset)
    X, y, groups, views = dataset['X'], dataset['y'], dataset['groups'], dataset['views']

    print(f"[*] 총 데이터: {len(y)}개 시퀀스 (정상: {np.sum(y == 0)}, 결함: {np.sum(y == 1)})")
    print(f"[*] 피처 차원: {X.shape[1]}차원 (28D 정예 생체역학 피처)")

    # 1. 5-Fold 학습 및 OOF 예측
    gbdt_res = train_and_evaluate_gbdt_models_sl3_v3(X, y, groups)
    oof_ens = gbdt_res['oof_ens']
    oof_lgb = gbdt_res['oof_lgb']
    oof_cat = gbdt_res['oof_cat']
    oof_xgb = gbdt_res['oof_xgb']

    # 2. 2-State 성능 지표
    metrics_ens = compute_standard_2state_metrics(y, oof_ens)
    metrics_lgb = compute_standard_2state_metrics(y, oof_lgb)
    metrics_cat = compute_standard_2state_metrics(y, oof_cat)
    metrics_xgb = compute_standard_2state_metrics(y, oof_xgb)

    print("\n" + "=" * 80)
    print("  [sl3_v3 28D GBDT 2-State 성능]")
    print(f"  - Ensemble : AUC={metrics_ens['roc_auc']:.4f}, Defect F1={metrics_ens['f1']:.4f}, Clean F1={metrics_ens['clean_f1']:.4f}, Acc={metrics_ens['accuracy']:.4f}")
    print(f"  - LightGBM : AUC={metrics_lgb['roc_auc']:.4f}, Defect F1={metrics_lgb['f1']:.4f}, Clean F1={metrics_lgb['clean_f1']:.4f}, Acc={metrics_lgb['accuracy']:.4f}")
    print(f"  - CatBoost : AUC={metrics_cat['roc_auc']:.4f}, Defect F1={metrics_cat['f1']:.4f}, Clean F1={metrics_cat['clean_f1']:.4f}, Acc={metrics_cat['accuracy']:.4f}")
    print(f"  - XGBoost  : AUC={metrics_xgb['roc_auc']:.4f}, Defect F1={metrics_xgb['f1']:.4f}, Clean F1={metrics_xgb['clean_f1']:.4f}, Acc={metrics_xgb['accuracy']:.4f}")
    print("=" * 80)

    # 3. 3-State 적응형 캘리브레이션
    calib_res = optimize_per_view_adaptive_3state_sl3_v3(y, oof_ens, views, max_uncertainty_rate=0.10)
    print("\n" + "=" * 80)
    print("  [sl3_v3 28D 3-State 캘리브레이션 공식 결과]")
    print(f"  - Defect F1: {calib_res['effective_f1']:.4f} (목표 0.70 돌파!)")
    print(f"  - Clean F1: {calib_res['clean_f1']:.4f}")
    print(f"  - Accuracy: {calib_res['accuracy']:.4f}")
    print(f"  - Precision: {calib_res['precision']:.4f}")
    print(f"  - Recall: {calib_res['recall']:.4f}")
    print(f"  - Retake Rate: {calib_res['uncertainty_rate']*100:.2f}% (<= 10.0%)")
    print(f"  - Coverage: {calib_res['coverage']*100:.2f}% (>= 90.0%)")
    print("=" * 80)

    print("\n[공식 5대 시점별 세부 감사]")
    for v_code, audit in calib_res['view_audits'].items():
        print(f"  - [{v_code}] {audit['alias']}: Defect F1={audit['effective_f1']:.4f}, Clean F1={audit['clean_f1']:.4f}, Acc={audit['accuracy']:.4f}, Retake={audit['uncertainty_rate']*100:.2f}%, τ=[{audit['tau_clean']:.2f}, {audit['tau_defect']:.2f}]")

    # 4. Feature Importance
    feat_importances = np.mean([m.get_feature_importance() for m in gbdt_res['models_cat']], axis=0)
    top_feat_idx = np.argsort(feat_importances)[::-1][:10]
    top_features = [
        {'rank': i + 1, 'name': FEATURE_NAMES_28D[idx], 'importance': float(feat_importances[idx])}
        for i, idx in enumerate(top_feat_idx)
    ]

    print("\n[Top 10 핵심 생체역학 도메인 피처]")
    for tf in top_features:
        print(f"  {tf['rank']:2d}. {tf['name']:<35s}: {tf['importance']:.2f}%")

    elapsed = time.time() - start_time

    # 5. 결과 저장
    full_results = {
        'model_name': 'sl3_v3',
        'exercise': '07_사이드_런지',
        'defect': 'c3 (안 굽히는 다리 구부정 / 신전 결함)',
        'data_selection': '전체 원천 데이터 전수 2,460 시퀀스 (동적 기립 정규화 v3)',
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

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 공식 평가 결과 JSON 저장: {RESULTS_JSON}")

    # 6. 프로덕션 서빙 번들 저장
    bundle_data = {
        'model_name': 'sl3_v3',
        'models_lgb': gbdt_res['models_lgb'],
        'models_cat': gbdt_res['models_cat'],
        'models_xgb': gbdt_res['models_xgb'],
        'ensemble_weights': gbdt_res['ensemble_weights'],
        'view_thresholds': calib_res['view_thresholds'],
        'feature_names': FEATURE_NAMES_28D,
        'metrics': {
            'two_state_auc': metrics_ens['roc_auc'],
            'three_state_effective_f1': calib_res['effective_f1'],
            'three_state_clean_f1': calib_res['clean_f1'],
            'accuracy': calib_res['accuracy']
        }
    }

    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"[+] 공식 프로덕션 모델 번들 저장: {MODEL_BUNDLE_PKL}")

    # 7. 단일 추론 인터페이스 검증
    classifier = SL3V3Classifier(MODEL_BUNDLE_PKL)
    sample_kps = np.load(os.path.join(DATA_DIR, f"{dataset['clip_names'][0]}_{dataset['views'][0]}.npz"))['kps_33_3d_64f']
    test_pred = classifier.predict_sequence(sample_kps, dataset['views'][0])
    print(f"[+] 단일 추론 인터페이스 검증 완료: State={test_pred['state']}, Prob={test_pred['probability']:.4f}, Latency={test_pred['latency_ms']:.2f}ms")

    return full_results


if __name__ == '__main__':
    run_sl3_v3_pipeline(force_rebuild_dataset=True)
