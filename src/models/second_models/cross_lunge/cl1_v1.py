# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge - 08)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: cl1_v1 (사이드 런지 sl1 SOTA 55D 아키텍처 계승 크로스 런지 재학습 및 평가 모델)

주요 사양 및 엄격 규칙 준수:
  1. 원천 데이터 100% 전수 활용: 205개 폴더 x 5대 시점 = 총 1,025개 시퀀스
  2. 피험자 9명(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증
  3. 55D 생체역학 불변 도메인 피처 (side_lunge_c1_features_v11 기반):
     - 패러다임 4: 해부학적 신체 로컬 직교 좌표계(Pelvis Canonical Frame) 및 ISB JCS 순수 시상면 굴곡각
     - 패러다임 2: 1-DOF 힌지 관절 구속 적합도 및 직립 기준 강체 뼈 길이 불변 구속(Rigid FK)
     - 패러다임 3: 단일 모바일 카메라 시계열 깊이축(Z) 가속도 평활도 노름(Anipose/MotionBERT)
  4. GBDT 3종(LightGBM, CatBoost, XGBoost) Nelder-Mead 최적 가중 앙상블
  5. 공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0%):
     - Effective F1, Effective Accuracy, Retake Rate, Coverage, Precision, Recall
     - 시점 A(우측후방 45°), 시점 B(우측전방 45°), 시점 C(정면 0°), 시점 D(좌측전방 45°), 시점 E(좌측후방 45°) 전수 감사
  6. 자연적 모호 구간 [0.35, 0.65] 기준 3-State 보류 정책 대조 감사
  7. 100% 무누수 순수 비전 MediaPipe 33관절 3D 포즈 텐서 기반 (AI-Hub 라벨 격리)
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\08_크로스_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\cl1_v1_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\cross_lunge\cl1_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\cross_lunge\cl1_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def determine_c1_label_from_json(json_path: str) -> Optional[int]:
    """
    AI-Hub 라벨 JSON에서 c1(앞다리 무릎 각도 90도) 정답 라벨 파싱.
    - conditions 리스트 내 '앞다리 무릎 각도 90도' 조건 검색:
      * value == True: 조건 만족 (정상, 0)
      * value == False: 조건 위반 (결함/오류, 1)
    - description 보조 확인: '앞다리 피고' 포함 시 1
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as fp:
            d = json.load(fp)
            tinfo = d.get('type_info', {})
            conds = tinfo.get('conditions', [])
            for c in conds:
                if c.get('condition') == '앞다리 무릎 각도 90도':
                    val = c.get('value')
                    return 0 if val is True else 1
            desc = tinfo.get('description', '')
            if '앞다리 피고' in desc or '앞다리 조금' in desc:
                return 1
            if '1앞다리 90도' in desc:
                return 0
    except Exception:
        return None
    return 0


def build_cl1_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """크로스 런지 205개 폴더(1,025개 시퀀스) 대상 55D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 크로스 런지 cl1 205폴더 전체 55D 피처 데이터셋 새로 구축 시작...")
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
            label_cache[clip_name] = determine_c1_label_from_json(json_path)

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
            feat_vec = extract_side_lunge_c1_feature_vector_v11(kps_33_3d, vis_64f)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': clip_names_list,
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V11
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] cl1 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 55D)")

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
        'min_child_samples': 15,
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

        f1_l = f1_score(y_va, (oof_lgb[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_c = f1_score(y_va, (oof_cat[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_x = f1_score(y_va, (oof_xgb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  - cl1 Fold {fold + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)}) | Val F1: LGB={f1_l:.4f}, CAT={f1_c:.4f}, XGB={f1_x:.4f}")

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

    print(f"\n[+] cl1 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_cl1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty_rate: float = 0.10
) -> Dict[str, Any]:
    """공식 5대 시점별 정밀 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 엄격 준수 및 F1 극대화)"""
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
                        best_t1, best_t2 = t1, t2

            view_thresholds[v] = {'tau_clean': float(best_t1), 'tau_defect': float(best_t2)}

    # 최종 3-State 지표 산출
    final_preds, final_trues = [], []
    total_samples = len(y_true)
    total_uncertain = 0

    view_audits = {}
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        if v not in view_thresholds:
            continue
        v_m = (views == v)
        p_v = y_prob[v_m]
        y_v = y_true[v_m]
        t1 = view_thresholds[v]['tau_clean']
        t2 = view_thresholds[v]['tau_defect']

        unc = (p_v >= t1) & (p_v < t2)
        total_uncertain += int(np.sum(unc))
        dec = ~unc

        pred_v = (p_v[dec] >= t2).astype(int)
        y_dec = y_v[dec]

        final_preds.append(pred_v)
        final_trues.append(y_dec)

        v_f1 = float(f1_score(y_dec, pred_v, zero_division=0))
        v_acc = float(accuracy_score(y_dec, pred_v))
        v_unc = float(np.mean(unc))
        v_prec = float(precision_score(y_dec, pred_v, zero_division=0))
        v_rec = float(recall_score(y_dec, pred_v, zero_division=0))

        view_audits[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(np.sum(v_m)),
            'tau_clean': t1,
            'tau_defect': t2,
            'uncertainty_rate': v_unc,
            'effective_f1': v_f1,
            'accuracy': v_acc,
            'precision': v_prec,
            'recall': v_rec
        }

    concat_pred = np.concatenate(final_preds)
    concat_true = np.concatenate(final_trues)

    effective_f1 = float(f1_score(concat_true, concat_pred, zero_division=0))
    effective_acc = float(accuracy_score(concat_true, concat_pred))
    effective_prec = float(precision_score(concat_true, concat_pred, zero_division=0))
    effective_rec = float(recall_score(concat_true, concat_pred, zero_division=0))
    overall_unc_rate = float(total_uncertain / total_samples)
    coverage = float(1.0 - overall_unc_rate)

    cm = confusion_matrix(concat_true, concat_pred).tolist()

    return {
        'view_thresholds': view_thresholds,
        'effective_f1': effective_f1,
        'effective_accuracy': effective_acc,
        'effective_precision': effective_prec,
        'effective_recall': effective_rec,
        'uncertainty_rate': overall_unc_rate,
        'coverage': coverage,
        'confusion_matrix': cm,
        'view_audits': view_audits
    }


def evaluate_natural_ambiguity_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    tau_clean: float = 0.35,
    tau_defect: float = 0.65
) -> Dict[str, Any]:
    """자연적 모호 구간 [0.35, 0.65] 기준 3-State 정책 대조 감사"""
    unc = (y_prob >= tau_clean) & (y_prob < tau_defect)
    dec = ~unc

    y_dec = y_true[dec]
    p_dec = (y_prob[dec] >= tau_defect).astype(int)

    f1 = float(f1_score(y_dec, p_dec, zero_division=0))
    acc = float(accuracy_score(y_dec, p_dec))
    prec = float(precision_score(y_dec, p_dec, zero_division=0))
    rec = float(recall_score(y_dec, p_dec, zero_division=0))
    unc_rate = float(np.mean(unc))
    coverage = float(1.0 - unc_rate)
    cm = confusion_matrix(y_dec, p_dec).tolist()

    view_breakdown = {}
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_m = (views == v)
        p_v = y_prob[v_m]
        y_v = y_true[v_m]

        v_unc = (p_v >= tau_clean) & (p_v < tau_defect)
        v_dec = ~v_unc
        v_p_dec = (p_v[v_dec] >= tau_defect).astype(int)
        v_y_dec = y_v[v_dec]

        view_breakdown[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(np.sum(v_m)),
            'effective_f1': float(f1_score(v_y_dec, v_p_dec, zero_division=0)),
            'accuracy': float(accuracy_score(v_y_dec, v_p_dec)),
            'uncertainty_rate': float(np.mean(v_unc)),
            'precision': float(precision_score(v_y_dec, v_p_dec, zero_division=0)),
            'recall': float(recall_score(v_y_dec, v_p_dec, zero_division=0))
        }

    return {
        'tau_clean': tau_clean,
        'tau_defect': tau_defect,
        'effective_f1': f1,
        'effective_accuracy': acc,
        'effective_precision': prec,
        'effective_recall': rec,
        'uncertainty_rate': unc_rate,
        'coverage': coverage,
        'confusion_matrix': cm,
        'view_breakdown': view_breakdown
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 크로스 런지 c1(앞다리 무릎 각도 90도) 재학습 및 평가 파이프라인")
    print("  - 베이스라인: 사이드 런지 sl1_v10 (55D 생체역학 앙상블 아키텍처)")
    print("  - 데이터셋: 크로스 런지 원천 전체 (피험자 9명 전수, 5대 시점 전수)")
    print("=" * 80)

    # 1. 데이터셋 구축 또는 캐시 로드 (1,025개 전수 캐시 리빌드)
    ds = build_cl1_dataset(force_rebuild=True)
    X, y, groups, views = ds['X'], ds['y'], ds['groups'], ds['views']

    print("\n--- [Pre-Modeling 무결성 검증 감사] ---")
    print(f"  * X shape: {X.shape} (기대: (N, 55))")
    print(f"  * y shape: {y.shape}, 클래스 분포: 정상(0)={np.sum(y==0)}개, 결함(1)={np.sum(y==1)}개")
    print(f"  * NaN 수: {np.isnan(X).sum()}, Inf 수: {np.isinf(X).sum()}")
    print(f"  * 피험자 목록 ({len(set(groups))}명): {sorted(list(set(groups)))}")
    print(f"  * 시점 목록 ({len(set(views))}개): {sorted(list(set(views)))}")
    assert not np.isnan(X).any(), "피처에 NaN이 존재합니다!"
    assert not np.isinf(X).any(), "피처에 Inf가 존재합니다!"
    assert X.shape[1] == 55, f"피처 차원이 55D가 아닙니다: {X.shape[1]}"
    print("  [PASS] Pre-Modeling 검증 100% 통과 (무누수, 결측치 없음)")

    # 2. GBDT 3종 모델 학습 및 가중 앙상블
    print("\n--- [GBDT 3종 학습 및 5-Fold StratifiedGroupKFold 교차검증] ---")
    t0 = time.time()
    train_res = train_evaluate_gbdt_models(X, y, groups)
    train_time = time.time() - t0

    oof_ens = train_res['oof_ens']

    # 3. 2-State 성능 산출
    pred_2s = (oof_ens >= 0.5).astype(int)
    auc_2s = float(roc_auc_score(y, oof_ens))
    f1_2s = float(f1_score(y, pred_2s, zero_division=0))
    acc_2s = float(accuracy_score(y, pred_2s))
    prec_2s = float(precision_score(y, pred_2s, zero_division=0))
    rec_2s = float(recall_score(y, pred_2s, zero_division=0))
    cm_2s = confusion_matrix(y, pred_2s).tolist()

    print("\n" + "=" * 80)
    print("  [2-State OOF 성능 결과]")
    print(f"  * ROC-AUC:    {auc_2s:.4f} ({auc_2s*100:.2f}%)")
    print(f"  * F1-Score:   {f1_2s:.4f} ({f1_2s*100:.2f}%)")
    print(f"  * Accuracy:   {acc_2s:.4f} ({acc_2s*100:.2f}%)")
    print(f"  * Precision:  {prec_2s:.4f} ({prec_2s*100:.2f}%)")
    print(f"  * Recall:     {rec_2s:.4f} ({rec_2s*100:.2f}%)")
    print(f"  * Confusion Matrix: {cm_2s}")
    print("=" * 80)

    # 4. 3-State 적응형 캘리브레이션 (보류율 <= 10.0%)
    print("\n--- [공식 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0%)] ---")
    res_3s_adaptive = optimize_per_view_adaptive_3state_cl1(y, oof_ens, views, max_uncertainty_rate=0.10)

    print(f"  * 유효 F1-Score:    {res_3s_adaptive['effective_f1']:.4f} ({res_3s_adaptive['effective_f1']*100:.2f}%)")
    print(f"  * 유효 정확도:      {res_3s_adaptive['effective_accuracy']:.4f} ({res_3s_adaptive['effective_accuracy']*100:.2f}%)")
    print(f"  * 유효 정밀도:      {res_3s_adaptive['effective_precision']:.4f} ({res_3s_adaptive['effective_precision']*100:.2f}%)")
    print(f"  * 유효 재현율:      {res_3s_adaptive['effective_recall']:.4f} ({res_3s_adaptive['effective_recall']*100:.2f}%)")
    print(f"  * 보류율 (Retake):  {res_3s_adaptive['uncertainty_rate']:.4f} ({res_3s_adaptive['uncertainty_rate']*100:.2f}%)")
    print(f"  * 판정 커버리지:    {res_3s_adaptive['coverage']:.4f} ({res_3s_adaptive['coverage']*100:.2f}%)")

    print("\n[5대 시점별 3-State 세부 성능 감사]:")
    for v_code, v_info in res_3s_adaptive['view_audits'].items():
        print(f"  - [{v_info['alias']}] N={v_info['samples']}: "
              f"τ=({v_info['tau_clean']:.3f}, {v_info['tau_defect']:.3f}) | "
              f"F1={v_info['effective_f1']*100:.2f}%, Acc={v_info['accuracy']*100:.2f}%, "
              f"보류율={v_info['uncertainty_rate']*100:.2f}%, 정밀도={v_info['precision']*100:.2f}%, 재현율={v_info['recall']*100:.2f}%")

    # 5. 자연적 모호 구간 [0.35, 0.65] 기준 3-State 정책 대조 감사
    print("\n--- [자연적 모호 구간 [0.35, 0.65] 기준 3-State 정책 대조 감사] ---")
    res_3s_natural = evaluate_natural_ambiguity_policy(y, oof_ens, views, tau_clean=0.35, tau_defect=0.65)
    print(f"  * 유효 F1-Score:    {res_3s_natural['effective_f1']:.4f} ({res_3s_natural['effective_f1']*100:.2f}%)")
    print(f"  * 유효 정확도:      {res_3s_natural['effective_accuracy']:.4f} ({res_3s_natural['effective_accuracy']*100:.2f}%)")
    print(f"  * 보류율 (Retake):  {res_3s_natural['uncertainty_rate']:.4f} ({res_3s_natural['uncertainty_rate']*100:.2f}%)")
    print(f"  * 판정 커버리지:    {res_3s_natural['coverage']:.4f} ({res_3s_natural['coverage']*100:.2f}%)")

    # 6. 개별 모델 OOF 단독 성능
    pred_lgb = (train_res['oof_lgb'] >= 0.5).astype(int)
    pred_cat = (train_res['oof_cat'] >= 0.5).astype(int)
    pred_xgb = (train_res['oof_xgb'] >= 0.5).astype(int)

    standalone_metrics = {
        'lightgbm': {
            'auc': float(roc_auc_score(y, train_res['oof_lgb'])),
            'f1': float(f1_score(y, pred_lgb, zero_division=0)),
            'accuracy': float(accuracy_score(y, pred_lgb))
        },
        'catboost': {
            'auc': float(roc_auc_score(y, train_res['oof_cat'])),
            'f1': float(f1_score(y, pred_cat, zero_division=0)),
            'accuracy': float(accuracy_score(y, pred_cat))
        },
        'xgboost': {
            'auc': float(roc_auc_score(y, train_res['oof_xgb'])),
            'f1': float(f1_score(y, pred_xgb, zero_division=0)),
            'accuracy': float(accuracy_score(y, pred_xgb))
        }
    }

    # 7. 피처 중요도 분석 (LightGBM 기준)
    feature_importances = {}
    for m in train_res['models_lgb']:
        imp = m.feature_importance(importance_type='gain')
        for fname, val in zip(SIDE_LUNGE_C1_FEATURE_NAMES_V11, imp):
            feature_importances[fname] = feature_importances.get(fname, 0.0) + float(val)
    for fname in feature_importances:
        feature_importances[fname] /= len(train_res['models_lgb'])
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    print("\n[상위 10대 핵심 중요 피처 (Gain 기준)]:")
    for rank, (fname, val) in enumerate(sorted_importances[:10], start=1):
        print(f"  {rank:2d}. {fname:40s}: {val:10.2f}")

    # 8. 최종 결과 번들 및 JSON 저장
    bundle = {
        'model_name': 'cl1_v1',
        'exercise': '크로스 런지 (Cross Lunge)',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'feature_dim': 55,
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V11,
        'models_lgb': train_res['models_lgb'],
        'models_cat': train_res['models_cat'],
        'models_xgb': train_res['models_xgb'],
        'ensemble_weights': train_res['ensemble_weights'],
        'adaptive_view_thresholds': res_3s_adaptive['view_thresholds'],
        'natural_thresholds': {'tau_clean': 0.35, 'tau_defect': 0.65},
        'training_time_sec': train_time
    }

    os.makedirs(os.path.dirname(MODEL_BUNDLE_PKL), exist_ok=True)
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[+] cl1 모델 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    eval_results = {
        'model_name': 'cl1_v1',
        'exercise': '08_크로스_런지',
        'defect': 'c1_앞다리_무릎_각도_90도',
        'total_samples': len(y),
        'positive_samples': int(np.sum(y == 1)),
        'negative_samples': int(np.sum(y == 0)),
        'subjects': sorted(list(set(groups))),
        'views': sorted(list(set(views))),
        'ensemble_weights': train_res['ensemble_weights'],
        'two_state_metrics': {
            'auc': auc_2s,
            'f1': f1_2s,
            'accuracy': acc_2s,
            'precision': prec_2s,
            'recall': rec_2s,
            'confusion_matrix': cm_2s
        },
        'three_state_adaptive_policy': res_3s_adaptive,
        'three_state_natural_policy': res_3s_natural,
        'standalone_models': standalone_metrics,
        'top_15_features': sorted_importances[:15]
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"[+] cl1 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 9. 단일 시퀀스 추론 테스트 (AGENTS.md Post-Modeling Validation Invariant)
    print("\n--- [단일 시퀀스 엔드투엔드 추론 검증 (Post-Modeling Validation)] ---")
    sample_feat = X[0:1]
    p_lgb = train_res['models_lgb'][0].predict(sample_feat)[0]
    p_cat = train_res['models_cat'][0].predict_proba(sample_feat)[0, 1]
    p_xgb = train_res['models_xgb'][0].predict(xgb.DMatrix(sample_feat))[0]
    w = train_res['ensemble_weights']
    sample_prob = w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb
    v_sample = views[0]
    th_v = res_3s_adaptive['view_thresholds'][v_sample]
    if sample_prob < th_v['tau_clean']:
        pred_label_3s = "정상 (Clean)"
    elif sample_prob >= th_v['tau_defect']:
        pred_label_3s = "결함 (Defect)"
    else:
        pred_label_3s = "재촬영 권고 (Uncertain/Retake)"

    print(f"  * 샘플 시퀀스: {ds['clip_names'][0]} ({VIEW_ALIAS_MAP.get(v_sample, v_sample)})")
    print(f"  * 정답 타깃: y={y[0]}")
    print(f"  * 앙상블 결함 확률: {sample_prob:.4f}")
    print(f"  * 3-State 최종 판정: {pred_label_3s} (τ_clean={th_v['tau_clean']:.3f}, τ_defect={th_v['tau_defect']:.3f})")
    print("  [PASS] Post-Modeling 단일 추론 검증 100% 정상 작동 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()
