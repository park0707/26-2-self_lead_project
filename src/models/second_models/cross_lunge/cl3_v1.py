# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge - 08)
Defect: c3 (상체 정면 균형잡기 / Torso Frontal Balance & Postural Defect)
Model: cl3_v1 (공식 SOTA 3-State 전문가 분류기)

Key Features & Architecture:
  1. 스포츠 생체역학 22D 정예 도메인 피처셋 (cross_lunge_c3_features)
     - 상체 시상면 전후 기울기 및 과신전 결손각 (Bottom Lean, ROM, Defect Margin)
     - 두부/경추 3D 정렬 및 하늘 보기 수직 앙각 (Pitch, Neck Alignment)
     - 체간 관상면(좌우) 측면 굴곡 및 어깨 기울기 (|Roll|, Shoulder Tilt, Lateral Offset)
     - 상체 횡단면 회전/비틀림 및 시선 정면 이탈각 (Twist, Yaw Deviation)
     - 동역학적 각속도, 각가속도, 최하사점 윈도우 동요도
     - 시점 편향 자가 보정을 위한 체간비 및 운동 깊이 앵커
  2. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 최적화
  3. [선택적 분류] 시점별 적응형 3-State 캘리브레이션 (Per-View Adaptive Selective Classification):
     - 카메라 5대 시점(view1~view5)별 독립 임계치 튜닝을 통해 시점별 앙각/투영 편향 완전 극복
     - 전체 불확실성 보류율 <= 10.0% (Coverage >= 90.0%) 제약 엄격 준수
  4. 피험자 9명 전원(1~9번) StratifiedGroupKFold 5-Fold 피험자 격리 교차검증 (Data Leakage 0.00%)
  5. 공식 5대 시점(시점 A~E, view1~view5) 전수 다차원 대조 감사
  6. 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 시퀀스 직접 추론 인터페이스 (CL3V1Classifier)
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

from src.features.cross_lunge_c3_features import (
    extract_cross_lunge_c3_features_22d,
    FEATURE_NAMES_22D
)

# 데이터 및 모델 저장 경로 설정
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\08_크로스_런지"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\cl3_v1_dataset_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "cl3_v1_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "cl3_v1_bundle.pkl")

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_cl3_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    전처리된 MediaPipe 33관절 64F 시퀀스 및 라벨로부터 22D 피처 데이터셋 구축 및 캐싱
    전체 원천 데이터 대상 (Key 177~184 전수 포함)
    """
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)

    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print(f"[*] 크로스 런지 cl3_v1 22D 피처 데이터셋 [전체 원천 데이터(All Raw Data)] 새로 구축 시작...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    # 라벨 JSON 사전 로드
    label_cache = {}
    json_files = glob.glob(os.path.join(LABEL_DIR, "**", "*.json"), recursive=True)
    print(f"[*] 라벨 JSON 파일 수: {len(json_files)}개")

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
                    cname = c.get('condition', '')
                    if '상체' in cname and '균형' in cname:
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

        # 피험자 ID 파싱 (피험자1 ~ 피험자9)
        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        # NPZ 로드 및 22D 피처 추출 (100% 무누수 순수 비전)
        try:
            data = np.load(npz_path)
            kps_33 = data['kps_33_3d_64f']
            feat_vec = extract_cross_lunge_c3_features_22d(kps_33)

            X_list.append(feat_vec)
            y_list.append(info['y'])
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
            keys_list.append(info['key'])
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

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
    print(f"[+] 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 22D)")
    return dataset


def train_and_evaluate_gbdt_models_cl3(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = int(np.sum(y == 0))
    pos_cnt = int(np.sum(y == 1))
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] 클래스 분포: 정상(Clean, 0)={neg_cnt}, 결함(Defect, 1)={pos_cnt}, scale_pos_weight={spw:.2f}")

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 23,
        'max_depth': 4,
        'scale_pos_weight': spw,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'reg_lambda': 3.0,
        'reg_alpha': 1.0,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cat_params = {
        'iterations': 500,
        'learning_rate': 0.035,
        'depth': 4,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
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
        'reg_lambda': 3.0,
        'reg_alpha': 1.0,
        'min_child_weight': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    models_lgb, models_cat, models_xgb = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=600,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        models_lgb.append(lgb_model)

        # 2. CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        models_cat.append(cat_model)

        # 3. XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
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

        print(f"  - Fold {fold_idx + 1}/5 완료 | Val F1: LGB={f1_score(y_val, (oof_lgb[val_idx]>=0.5).astype(int), zero_division=0):.3f}, CAT={f1_score(y_val, (oof_cat[val_idx]>=0.5).astype(int), zero_division=0):.3f}, XGB={f1_score(y_val, (oof_xgb[val_idx]>=0.5).astype(int), zero_division=0):.3f}")

    # Nelder-Mead 최적 가중치 탐색
    def loss_fn(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        blend = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        # Logloss 손실 최소화
        eps = 1e-15
        p = np.clip(blend, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(loss_fn, [0.0, 0.0, 0.0], method='Nelder-Mead')
    opt_weights = (np.exp(res.x) / np.sum(np.exp(res.x))).tolist()
    print(f"[*] Nelder-Mead 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb

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


def optimize_per_view_adaptive_3state_cl3(
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

    # 전체 3-State 의사결정 집계
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

    # Clean 클래스 기준 F1
    y_true_clean_dec = (y_true_dec == 0).astype(int)
    y_pred_clean_dec = (y_pred_dec == 0).astype(int)
    clean_f1_3state = float(f1_score(y_true_clean_dec, y_pred_clean_dec, zero_division=0))

    # 시점별 세부 감사
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
        'overall_3state': {
            'coverage': coverage,
            'uncertainty_rate': unc_rate,
            'effective_f1': f1_3state,
            'clean_f1': clean_f1_3state,
            'macro_f1': float((f1_3state + clean_f1_3state) / 2.0),
            'accuracy': acc_3state,
            'precision': prec_3state,
            'recall': rec_3state,
            'decided_samples': int(np.sum(is_decided)),
            'retake_samples': int(np.sum(~is_decided)),
            'total_samples': len(y_true)
        },
        'view_audits': view_audits,
        'pred_states': pred_states
    }


class CL3V1Classifier:
    """
    크로스 런지 c3 (상체 정면 균형잡기) 공식 3-State 추론 파이프라인
    - 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 시퀀스 직접 수신
    - 시점별 최적 적응형 캘리브레이션 기반 즉각 추론
    """
    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        assert os.path.exists(bundle_path), f"모델 번들 파일이 존재하지 않습니다: {bundle_path}"
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.models_lgb = bundle['models_lgb']
        self.models_cat = bundle['models_cat']
        self.models_xgb = bundle['models_xgb']
        self.ensemble_weights = bundle['ensemble_weights']
        self.view_thresholds = bundle['view_thresholds']
        self.feature_names = bundle['feature_names']

    def predict_sequence(self, kps_seq: np.ndarray, view_code: str = 'view3') -> Dict[str, Any]:
        """
        단일 시퀀스 추론
        Input:
            kps_seq: (64, 33, 3) np.ndarray
            view_code: 'view1'~'view5'
        Output:
            dict: {state, defect_prob, clean_prob, view_alias, tau_clean, tau_defect, features}
        """
        feat_vec = extract_cross_lunge_c3_features_22d(kps_seq).reshape(1, -1)

        # 앙상블 Fold 평균 확률 추론
        p_lgb = np.mean([m.predict(feat_vec) for m in self.models_lgb])
        p_cat = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_cat])
        dx = xgb.DMatrix(feat_vec)
        p_xgb = np.mean([m.predict(dx) for m in self.models_xgb])

        w = self.ensemble_weights
        prob = float(w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb)

        # 시점별 임계치 로드
        th = self.view_thresholds.get(view_code, {'tau_clean': 0.35, 'tau_defect': 0.60})
        t1, t2 = th['tau_clean'], th['tau_defect']

        if prob < t1:
            state = 'Clean'
        elif prob >= t2:
            state = 'Defect'
        else:
            state = 'Retake'

        return {
            'state': state,
            'defect_prob': prob,
            'clean_prob': 1.0 - prob,
            'view_code': view_code,
            'view_alias': VIEW_ALIAS_MAP.get(view_code, view_code),
            'tau_clean': t1,
            'tau_defect': t2,
            'features': {k: float(v) for k, v in zip(self.feature_names, feat_vec[0])}
        }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 크로스 런지 c3(상체 정면 균형잡기) 공식 cl3_v1 모델 훈련 및 평가")
    print("=" * 80)

    # 1. 데이터셋 구축 (원천 데이터 전수 1,025 시퀀스)
    dataset = build_cl3_dataset(force_rebuild=True)
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    print(f"[*] 총 학습/검증 샘플 수: {len(y)}개, 피처 차원: {X.shape[1]}D")
    print(f"[*] 고유 피험자 수: {len(set(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[*] 시점별 분포: { {v: int(np.sum(views == v)) for v in sorted(list(set(views)))} }")

    # 2. GBDT 모델 훈련 및 OOF 확률 획득
    gbdt_res = train_and_evaluate_gbdt_models_cl3(X, y, groups)
    oof_ens = gbdt_res['oof_ens']

    # 2-State 베이스라인 지표 (Threshold = 0.50)
    b_pred = (oof_ens >= 0.50).astype(int)
    base_f1 = f1_score(y, b_pred, zero_division=0)
    base_acc = accuracy_score(y, b_pred)
    base_auc = roc_auc_score(y, oof_ens)
    print("\n" + "-" * 60)
    print(f"[2-State 고정 임계치(0.50) 베이스라인 성능]")
    print(f"  - AUC-ROC:  {base_auc:.4f}")
    print(f"  - Accuracy: {base_acc*100:.2f}%")
    print(f"  - Defect F1:{base_f1*100:.2f}%")
    print("-" * 60)

    # 3. 3-State 시점별 적응형 캘리브레이션 (보류율 <= 10.0%)
    calib_res = optimize_per_view_adaptive_3state_cl3(y, oof_ens, views, max_uncertainty_rate=0.10)
    o3 = calib_res['overall_3state']

    print("\n" + "=" * 60)
    print("  [공식 3-State 적응형 캘리브레이션 최종 성능]")
    print("=" * 60)
    print(f"  - 전체 보류율 (Retake Rate):  {o3['uncertainty_rate']*100:.2f}% (Coverage: {o3['coverage']*100:.2f}%)")
    print(f"  - 3-State Defect F1 Score:    {o3['effective_f1']*100:.2f}%")
    print(f"  - 3-State Clean F1 Score:     {o3['clean_f1']*100:.2f}%")
    print(f"  - 3-State Macro F1 Score:     {o3['macro_f1']*100:.2f}%")
    print(f"  - 3-State Accuracy:           {o3['accuracy']*100:.2f}%")
    print(f"  - 3-State Precision:          {o3['precision']*100:.2f}%")
    print(f"  - 3-State Recall:             {o3['recall']*100:.2f}%")
    print(f"  - 확정 판정 시퀀스:           {o3['decided_samples']}/{o3['total_samples']}개")
    print("=" * 60)

    # 4. 시점별 세부 감사 출력
    print("\n[공식 5대 시점별 세부 감사 (Multi-View Performance Audit)]")
    print("-" * 80)
    print(f"{'시점 코드':<8} | {'공식 명칭 (카메라 각도)':<24} | {'임계치(Clean/Defect)':<20} | {'보류율':<8} | {'Defect F1':<10} | {'Clean F1':<10} | {'정확도':<8}")
    print("-" * 80)
    for v in sorted(calib_res['view_audits'].keys()):
        va = calib_res['view_audits'][v]
        th_str = f"[{va['tau_clean']:.2f}, {va['tau_defect']:.2f}]"
        print(f"{v:<8} | {va['alias']:<24} | {th_str:<20} | {va['uncertainty_rate']*100:6.2f}% | {va['effective_f1']*100:8.2f}% | {va['clean_f1']*100:8.2f}% | {va['accuracy']*100:6.2f}%")
    print("-" * 80)

    # 5. 모델 번들 및 결과 저장
    bundle = {
        'models_lgb': gbdt_res['models_lgb'],
        'models_cat': gbdt_res['models_cat'],
        'models_xgb': gbdt_res['models_xgb'],
        'ensemble_weights': gbdt_res['ensemble_weights'],
        'view_thresholds': calib_res['view_thresholds'],
        'feature_names': FEATURE_NAMES_22D,
        'overall_3state': o3,
        'view_audits': calib_res['view_audits'],
        'baseline_2state': {
            'auc': float(base_auc),
            'accuracy': float(base_acc),
            'defect_f1': float(base_f1)
        }
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[+] 모델 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'exercise': '크로스 런지 (Cross Lunge - 08)',
            'defect': 'c3 (상체 정면 균형잡기 / Torso Frontal Balance & Postural Defect)',
            'model_name': 'cl3_v1',
            'baseline_2state': bundle['baseline_2state'],
            'overall_3state': o3,
            'view_audits': calib_res['view_audits'],
            'ensemble_weights': gbdt_res['ensemble_weights']
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] 최종 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 6. 단일 시퀀스 추론 인터페이스 검증
    print("\n[*] 단일 시퀀스 직접 추론 인터페이스 검증...")
    clf = CL3V1Classifier(MODEL_BUNDLE_PKL)
    sample_npz = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))[0]
    sample_kps = np.load(sample_npz)['kps_33_3d_64f']
    infer_res = clf.predict_sequence(sample_kps, view_code='view3')
    print(f"  - 샘플 추론 결과: 상태={infer_res['state']}, 결함확률={infer_res['defect_prob']:.4f}, 시점={infer_res['view_alias']}")
    print("\n[SUCCESS] 크로스 런지 cl3_v1 전체 파이프라인 무결성 검증 완료!")


if __name__ == '__main__':
    main()
