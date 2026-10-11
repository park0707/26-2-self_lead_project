# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Model: bl1_v3 (3차 최적화 모델)
Key Enhancements in v3:
  1. 피처 전수 조사(Ablation & Correlation Matrix) 기반 12개 고상관/성능 저해 피처 제거 (38D -> 26D 정예 피처셋)
  2. 모델 구조 최적화: 트리 깊이 제어(max_depth=5), L1/L2 복합 규제화(alpha=0.5, lambda=3.0), 컬럼 서브샘플링(colsample=0.7)
  3. GBDT 3종(LightGBM, CatBoost, XGBoost) Nelder-Mead 최적 앙상블 가중치 적용
  4. 3-State 선택적 분류(Selective Classification) 임계치 정밀 튜닝 (보류율 <= 10.0% 엄격 준수)
  5. 피험자 9명 전원(1~9번) StratifiedGroupKFold 5-Fold 피험자 격리 교차검증 및 공식 5대 시점 전수 감사
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

# 도메인 피처 추출 모듈 v3 임포트
sys.path.append(r"D:\src\features")
from lunge_c1_features_v3 import (
    extract_lunge_c1_feature_vector_v3,
    LUNGE_C1_FEATURE_NAMES_V3
)

# 데이터 및 모델 저장 경로 설정
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\bl1_v3_dataset_cache.pkl"
V2_CACHE_PKL = r"D:\Data\피처_데이터\bl1_v2_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v3_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\babel_lunge\bl1_v3_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def load_or_build_v3_dataset() -> Dict[str, Any]:
    """26D v3 피처 데이터셋 로드 또는 빌드 (v2 캐시에서 고속 프로젝션 지원)"""
    if os.path.exists(FEATURE_CACHE_PKL):
        print(f"[*] v3 캐시 파일 발견: {FEATURE_CACHE_PKL} 로드 중...")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[+] v3 캐시 로드 완료: N={len(dataset['y'])}, D={dataset['X'].shape[1]}")
        return dataset

    # v2 캐시가 존재하면 v3 피처 인덱스만 슬라이싱하여 초고속 생성
    if os.path.exists(V2_CACHE_PKL):
        print(f"[*] v2 캐시 파일({V2_CACHE_PKL})로부터 v3 26D 피처셋 고속 프로젝션 수행...")
        with open(V2_CACHE_PKL, 'rb') as f:
            v2_data = pickle.load(f)

        v2_names = v2_data.get("feature_names")
        v3_indices = [v2_names.index(name) for name in LUNGE_C1_FEATURE_NAMES_V3]

        X_v3 = v2_data["X"][:, v3_indices]
        dataset = {
            'X': np.array(X_v3, dtype=np.float32),
            'y': np.array(v2_data['y'], dtype=np.int32),
            'groups': np.array(v2_data.get('groups', v2_data.get('subjects'))),
            'views': np.array(v2_data['views']),
            'clip_names': np.array(v2_data['clip_names']),
            'feature_names': LUNGE_C1_FEATURE_NAMES_V3
        }

        os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
        with open(FEATURE_CACHE_PKL, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[+] v3 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(dataset['y'])}개 샘플, 26D)")
        return dataset

    # 캐시가 없는 경우 원천 NPZ에서 직접 추출
    print("[*] v3 원천 NPZ 파일로부터 26D 피처 추출 파이프라인 가동...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"NPZ 파일을 찾을 수 없습니다: {DATA_DIR}")

    X_list, y_list, groups_list, views_list, clip_names_list = [], [], [], [], []
    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.basename(npz_path)
        clip_name = fname.replace('.npz', '')

        view_code = "view3"
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if v in fname:
                view_code = v
                break

        if clip_name not in label_cache:
            json_path = os.path.join(LABEL_DIR, f"{clip_name}.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as fp:
                    d = json.load(fp)
                    conds = d.get('type_info', {}).get('conditions', [])
                    c1_val = None
                    for c in conds:
                        if '앞다리' in c.get('condition', '') and '무릎' in c.get('condition', ''):
                            c1_val = c.get('value')
                            break
                    label_cache[clip_name] = c1_val
            except Exception:
                label_cache[clip_name] = None

        c1_val = label_cache.get(clip_name)
        if c1_val is None:
            continue

        target_label = 1 if c1_val is False else 0
        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f']
            vis_64f = data.get('vis_64f', None)
            feat_vec = extract_lunge_c1_feature_vector_v3(kps_33_3d, vis_64f)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
        except Exception as e:
            print(f"[!] {npz_path} v3 피처 추출 오류: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(npz_files):
            print(f"  - v3 피처 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': LUNGE_C1_FEATURE_NAMES_V3
    }

    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] v3 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 샘플)")
    return dataset


def train_and_evaluate_gbdt_models_v3(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 최적화 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / pos_cnt)
    print(f"[*] v3 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

    # 1. LightGBM 최적 구조 하이퍼파라미터
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

    # 2. CatBoost 최적 구조 하이퍼파라미터
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

    # 3. XGBoost 최적 구조 하이퍼파라미터
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

        # LightGBM 학습
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

        # CatBoost 학습
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        models_cat.append(model_cat)

        # XGBoost 학습
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

        print(f"  - v3 Fold {fold_idx + 1}/5 학습 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    # Nelder-Mead 최적 가중치 탐색
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
    print(f"\n[+] v3 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_3state_selective_classification_v3(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """3-State 선택적 분류 임계치 최적화 (보류율 <= 10.0% 엄격 준수)"""
    best_res = None
    best_f1 = -1.0

    tau_clean_cands = np.linspace(0.25, 0.45, 41)
    tau_defect_cands = np.linspace(0.45, 0.65, 41)

    for tau_c in tau_clean_cands:
        for tau_d in tau_defect_cands:
            if tau_d <= tau_c:
                continue

            clean_mask = y_prob < tau_c
            defect_mask = y_prob >= tau_d
            uncertain_mask = (~clean_mask) & (~defect_mask)

            unc_rate = np.mean(uncertain_mask)
            if unc_rate > max_uncertainty_rate:
                continue

            decided_mask = ~uncertain_mask
            if np.sum(decided_mask) == 0:
                continue

            y_t_dec = y_true[decided_mask]
            y_p_dec = np.where(y_prob[decided_mask] >= tau_d, 1, 0)

            f1 = f1_score(y_t_dec, y_p_dec, zero_division=0)
            acc = accuracy_score(y_t_dec, y_p_dec)

            if f1 > best_f1:
                best_f1 = f1
                best_res = {
                    'tau_clean': float(tau_c),
                    'tau_defect': float(tau_d),
                    '3state_f1': float(f1),
                    '3state_acc': float(acc),
                    '3state_prec': float(precision_score(y_t_dec, y_p_dec, zero_division=0)),
                    '3state_recall': float(recall_score(y_t_dec, y_p_dec, zero_division=0)),
                    'uncertainty_rate': float(unc_rate),
                    'coverage': float(1.0 - unc_rate),
                    'decided_samples': int(np.sum(decided_mask)),
                    'uncertain_samples': int(np.sum(uncertain_mask)),
                    'clean_count': int(np.sum(clean_mask)),
                    'defect_count': int(np.sum(defect_mask))
                }

    return best_res


def audit_viewpoints_v3(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    tau_clean: float,
    tau_defect: float
) -> Dict[str, Dict[str, Any]]:
    """공식 5대 시점(시점 A~E / view1~view5) 전수 성능 감사"""
    unique_views = sorted(list(set(views)))
    view_results = {}

    for v_code in unique_views:
        idx = (views == v_code)
        if np.sum(idx) == 0:
            continue

        y_t = y_true[idx]
        y_p = y_prob[idx]

        clean_mask = y_p < tau_clean
        defect_mask = y_p >= tau_defect
        uncertain_mask = (~clean_mask) & (~defect_mask)
        decided_mask = ~uncertain_mask

        y_t_dec = y_t[decided_mask]
        y_p_dec = np.where(y_p[decided_mask] >= tau_defect, 1, 0)

        f1_3s = float(f1_score(y_t_dec, y_p_dec, zero_division=0))
        acc_3s = float(accuracy_score(y_t_dec, y_p_dec))
        prec_3s = float(precision_score(y_t_dec, y_p_dec, zero_division=0))
        rec_3s = float(recall_score(y_t_dec, y_p_dec, zero_division=0))
        auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 1.0
        unc_rate = float(np.mean(uncertain_mask))

        view_results[v_code] = {
            'view_name': VIEW_ALIAS_MAP.get(v_code, v_code),
            'samples': int(np.sum(idx)),
            '3state_f1': f1_3s,
            'accuracy': acc_3s,
            'precision': prec_3s,
            'recall': rec_3s,
            'roc_auc': auc,
            'uncertainty_rate': unc_rate,
            'coverage': float(1.0 - unc_rate)
        }

    return view_results


def run_bl1_v3_training_and_evaluation():
    """bl1_v3 모델 학습 및 평가 메인 파이프라인"""
    t_start = time.time()
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 bl1_v3 (c1: 앞다리 무릎 90도) SOTA 최적화 모델")
    print("=" * 80)

    dataset = load_or_build_v3_dataset()
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    print(f"\n[+] v3 데이터셋 규격: X={X.shape}, y={y.shape} (정상={np.sum(y==0)}, 결함={np.sum(y==1)})")
    print(f"[+] 피험자 전원: {len(np.unique(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[+] 카메라 시점: {len(np.unique(views))}개 ({sorted(list(set(views)))})")

    # 1. 모델 학습 및 교차검증
    print("\n[*] v3 GBDT 최적 구조 앙상블 학습 (StratifiedGroupKFold 5-Fold) 진행 중...")
    train_out = train_and_evaluate_gbdt_models_v3(X, y, groups)
    oof_ens = train_out['oof_ens']

    # 2. 베이스라인 이진 평가 (tau=0.5)
    base_bin_f1 = float(f1_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_acc = float(accuracy_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_prec = float(precision_score(y, (oof_ens >= 0.5).astype(int)))
    base_bin_rec = float(recall_score(y, (oof_ens >= 0.5).astype(int)))
    base_roc_auc = float(roc_auc_score(y, oof_ens))

    # 3. 3-State 최적화
    print("\n[*] v3 3-State 임계치 최적화 (최대 보류율 <= 10.0% 제약)...")
    best_3state = optimize_3state_selective_classification_v3(y, oof_ens, max_uncertainty_rate=0.10)

    print("\n[+] v3 최적화 결과:")
    print(f"  - 베이스라인 이진 F1: {base_bin_f1*100:.2f}% | 정확도: {base_bin_acc*100:.2f}% | AUC: {base_roc_auc*100:.2f}%")
    print(f"  - 3-State 확정 샘플 F1: {best_3state['3state_f1']*100:.2f}% | 정확도: {best_3state['3state_acc']*100:.2f}%")
    print(f"  - 판단 보류율(Uncertainty): {best_3state['uncertainty_rate']*100:.2f}% (커버리지: {best_3state['coverage']*100:.2f}%)")
    print(f"  - 최적 임계치: tau_clean={best_3state['tau_clean']:.3f}, tau_defect={best_3state['tau_defect']:.3f}")

    # 4. 5대 시점 전수 감사
    print("\n[*] v3 공식 5대 시점 전수 성능 감사 (AGENTS.md 표준)...")
    view_audit = audit_viewpoints_v3(
        y, oof_ens, views, best_3state['tau_clean'], best_3state['tau_defect']
    )
    for v_code, v_res in view_audit.items():
        print(f"  - {v_res['view_name']} ({v_code}): 3-State F1={v_res['3state_f1']*100:.2f}%, ACC={v_res['accuracy']*100:.2f}%, AUC={v_res['roc_auc']*100:.2f}%, 보류={v_res['uncertainty_rate']*100:.2f}% (샘플: {v_res['samples']})")

    # 5. 결과 JSON 저장
    elapsed = time.time() - t_start
    eval_summary = {
        'metadata': {
            'project_name': 'vidio-health-trainer',
            'exercise': '04_바벨_런지',
            'target_defect': 'c1 (앞다리 무릎 각도 90도)',
            'model_name': 'bl1_v3',
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
        'view_audit': view_audit
    }

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v3 성능 평가 결과 JSON 저장: {RESULTS_JSON}")

    # 6. 전체 데이터셋 대상 최종 프로덕션 번들 저장
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
        'model_name': 'bl1_v3',
        'project': 'vidio-health-trainer',
        'exercise': '04_바벨_런지',
        'defect': 'c1',
        'feature_dim': len(feature_names),
        'feature_names': feature_names,
        'tau_clean': best_3state['tau_clean'],
        'tau_defect': best_3state['tau_defect'],
        'ensemble_weights': train_out['ensemble_weights'],
        'model_lgb': final_lgb,
        'model_cat': final_cat,
        'model_xgb': final_xgb
    }

    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] v3 번들 모델 저장: {MODEL_BUNDLE_PKL}")
    print(f"[+] v3 전체 모델링 소요 시간: {elapsed:.1f}초")


# ---------------------------------------------------------------------------
# 실서비스 서브모델 추론 인터페이스 클래스 (BL1V3Classifier)
# ---------------------------------------------------------------------------
class BL1V3Classifier:
    """바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v3 추론기"""

    def __init__(self, bundle_path: str = MODEL_BUNDLE_PKL):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"번들 파일 없음: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.tau_clean = self.bundle['tau_clean']
        self.tau_defect = self.bundle['tau_defect']
        self.weights = self.bundle['ensemble_weights']
        self.feature_names = self.bundle['feature_names']
        self.model_lgb = self.bundle['model_lgb']
        self.model_cat = self.bundle['model_cat']
        self.model_xgb = self.bundle['model_xgb']

    def predict_sequence(self, kps_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """단일 64프레임 시퀀스에 대한 3-State 상태 판별"""
        feat_vec = extract_lunge_c1_feature_vector_v3(kps_64f, vis_64f).reshape(1, -1)

        p_lgb = self.model_lgb.predict(feat_vec)[0]
        p_cat = self.model_cat.predict_proba(feat_vec)[0, 1]
        p_xgb = self.model_xgb.predict(xgb.DMatrix(feat_vec))[0]

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        if prob < self.tau_clean:
            state = 0
            state_str = "CLEAN (정상)"
        elif prob >= self.tau_defect:
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
            'knee_90_error': float(feat_vec[0, 2]),
            'sagittal_90_error': float(feat_vec[0, 5]),
            'tau_clean': self.tau_clean,
            'tau_defect': self.tau_defect
        }


if __name__ == '__main__':
    run_bl1_v3_training_and_evaluation()
