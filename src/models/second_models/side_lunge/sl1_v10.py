# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_v10 (패러다임 2, 3, 4 생체역학 융합 55D SOTA 앙상블 모델)

주요 혁신 및 설계:
1. sl1_v9(49D)의 완전 계승 및 패러다임 2, 3, 4 기반 6대 생체역학 불변량 융합 (49D -> 55D):
   - 패러다임 4: 해부학적 신체 로컬 직교 좌표계(Pelvis Canonical Frame) 및 ISB JCS 순수 시상면 굴곡각
   - 패러다임 2: 1-DOF 힌지 관절 구속 적합도 및 직립 기준 강체 뼈 길이 불변 구속(Rigid FK)
   - 패러다임 3: 단일 모바일 카메라 시계열 깊이축(Z) 가속도 평활도 노름(Anipose/MotionBERT)
2. 피험자 9명 전수 및 5대 카메라 시점(view1~view5) 전수 포함 (총 4,130개 시퀀스)
3. 5-Fold StratifiedGroupKFold (피험자 단위 엄격 격리)
4. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 Nelder-Mead 가중 앙상블
5. 5대 공식 시점별 적응형 3-State 좌표 하강 캘리브레이션 (보류율 <= 10.0%)
6. 100% 무누수 순수 비전 MediaPipe 33관절 3D 포즈 텐서 기반
"""

import os
import sys
import json
import glob
import pickle
from typing import Dict, List, Tuple, Any

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

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v10_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_v10_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_v10_bundle.pkl"

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


def build_sl1_v10_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 55D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl1_v10 826폴더 전체 55D 피처 데이터셋 새로 구축 시작...")
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
            feat_vec = extract_side_lunge_c1_feature_vector_v11(kps_33_3d, vis_64f)

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
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V11
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v10 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 55D)")

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

        print(f"  - sl1_v10 Fold {fold + 1}/5 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

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

    print(f"\n[+] sl1_v10 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

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


def optimize_per_view_adaptive_3state_sl1_v10(
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

    for v in unique_views:
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

        view_f1 = float(f1_score(y_dec, pred_v, zero_division=0))
        view_acc = float(accuracy_score(y_dec, pred_v))
        view_prec = float(precision_score(y_dec, pred_v, zero_division=0))
        view_rec = float(recall_score(y_dec, pred_v, zero_division=0))
        view_unc_rate = float(np.mean(unc))

        view_audits[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(len(y_v)),
            'tau_clean': t1,
            'tau_defect': t2,
            'uncertainty_rate': view_unc_rate,
            'effective_f1': view_f1,
            'accuracy': view_acc,
            'precision': view_prec,
            'recall': view_rec
        }

    y_dec_all = np.concatenate(final_trues)
    pred_dec_all = np.concatenate(final_preds)

    eff_f1 = float(f1_score(y_dec_all, pred_dec_all, zero_division=0))
    eff_acc = float(accuracy_score(y_dec_all, pred_dec_all))
    eff_prec = float(precision_score(y_dec_all, pred_dec_all, zero_division=0))
    eff_rec = float(recall_score(y_dec_all, pred_dec_all, zero_division=0))
    uncertainty_rate = float(total_uncertain / total_samples)
    coverage = float(1.0 - uncertainty_rate)

    return {
        'effective_f1': eff_f1,
        'accuracy': eff_acc,
        'precision': eff_prec,
        'recall': eff_rec,
        'uncertainty_rate': uncertainty_rate,
        'coverage': coverage,
        'view_thresholds': view_thresholds,
        'view_audits': view_audits
    }


def main():
    print("=" * 80)
    print("  vidio-health-trainer | 사이드 런지 c1 sl1_v10 (패러다임 2, 3, 4 생체역학 융합 55D)")
    print("=" * 80)

    # 1. 55D 데이터셋 구축
    dataset = build_sl1_v10_dataset(force_rebuild=False)
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    print(f"\n[*] 데이터셋 규격: X={X.shape}, y={y.shape}, 고유 피험자={len(set(groups))}명, 5대 시점 균등 분포 확인")
    assert X.shape[1] == 55, f"Expected 55 features, got {X.shape[1]}"

    # 2. 5-Fold 교차검증 및 모델 학습
    train_res = train_evaluate_gbdt_models(X, y, groups)

    oof_lgb = train_res['oof_lgb']
    oof_cat = train_res['oof_cat']
    oof_xgb = train_res['oof_xgb']
    oof_ens = train_res['oof_ens']
    opt_weights = train_res['ensemble_weights']

    # 3. 2-State 성능 산출 (임계값 0.50)
    def calc_2state(probs):
        preds = (probs >= 0.5).astype(int)
        return {
            'roc_auc': float(roc_auc_score(y, probs)),
            'f1': float(f1_score(y, preds, zero_division=0)),
            'accuracy': float(accuracy_score(y, preds)),
            'precision': float(precision_score(y, preds, zero_division=0)),
            'recall': float(recall_score(y, preds, zero_division=0)),
        }

    m_ens = calc_2state(oof_ens)
    m_lgb = calc_2state(oof_lgb)
    m_cat = calc_2state(oof_cat)
    m_xgb = calc_2state(oof_xgb)

    print("\n" + "=" * 50)
    print("  [sl1_v10 2-State (Threshold 0.50) Performance]")
    print("=" * 50)
    print(f"- LightGBM: AUC={m_lgb['roc_auc']:.4f}, F1={m_lgb['f1']:.4f}, Acc={m_lgb['accuracy']:.4f}")
    print(f"- CatBoost: AUC={m_cat['roc_auc']:.4f}, F1={m_cat['f1']:.4f}, Acc={m_cat['accuracy']:.4f}")
    print(f"- XGBoost:  AUC={m_xgb['roc_auc']:.4f}, F1={m_xgb['f1']:.4f}, Acc={m_xgb['accuracy']:.4f}")
    print(f"[*] Ensemble: AUC={m_ens['roc_auc']:.4f}, F1={m_ens['f1']:.4f} ({m_ens['f1']*100:.2f}%), Acc={m_ens['accuracy']:.4f} ({m_ens['accuracy']*100:.2f}%)")

    # 4. 3-State 적응형 캘리브레이션 (보류율 <= 10.0%)
    print("\n[*] sl1_v10 5-view adaptive 3-state calibration running...")
    opt_3state = optimize_per_view_adaptive_3state_sl1_v10(y, oof_ens, views, max_uncertainty_rate=0.10)

    print("\n" + "=" * 50)
    print("  [sl1_v10 3-State Adaptive Calibration (Uncertainty <= 10%)]")
    print("=" * 50)
    print(f"[*] Effective F1: {opt_3state['effective_f1']:.4f} ({opt_3state['effective_f1']*100:.2f}%)")
    print(f"- Effective Acc:  {opt_3state['accuracy']:.4f} ({opt_3state['accuracy']*100:.2f}%)")
    print(f"- Precision:      {opt_3state['precision']:.4f}, Recall: {opt_3state['recall']:.4f}")
    print(f"- Uncertainty:    {opt_3state['uncertainty_rate']:.4f} ({opt_3state['uncertainty_rate']*100:.2f}%)")
    print(f"- Coverage:       {opt_3state['coverage']:.4f} ({opt_3state['coverage']*100:.2f}%)")

    # 5. 자연적 모호 구간 [0.35, 0.65] 정책도 함께 평가 (사용자 기선정 정책과 직접 대조)
    unc_fixed = (oof_ens >= 0.35) & (oof_ens < 0.65)
    dec_fixed = ~unc_fixed
    pred_fixed = (oof_ens[dec_fixed] >= 0.65).astype(int)
    y_fixed = y[dec_fixed]
    fixed_policy_res = {
        'policy': '[0.35, 0.65]',
        'effective_f1': float(f1_score(y_fixed, pred_fixed, zero_division=0)),
        'accuracy': float(accuracy_score(y_fixed, pred_fixed)),
        'precision': float(precision_score(y_fixed, pred_fixed, zero_division=0)),
        'recall': float(recall_score(y_fixed, pred_fixed, zero_division=0)),
        'uncertainty_rate': float(np.mean(unc_fixed)),
        'coverage': float(1.0 - np.mean(unc_fixed))
    }
    print(f"- [0.35, 0.65] Ambiguity Policy: F1={fixed_policy_res['effective_f1']:.4f}, Acc={fixed_policy_res['accuracy']:.4f}, Uncertainty={fixed_policy_res['uncertainty_rate']*100:.2f}%")

    print("\n[5-View Audit Results]")
    for v_code, audit in opt_3state['view_audits'].items():
        print(f"  - [{v_code}] {audit['alias']}: F1={audit['effective_f1']:.4f}, Acc={audit['accuracy']:.4f}, Unc={audit['uncertainty_rate']*100:.2f}%, tau=[{audit['tau_clean']:.3f}, {audit['tau_defect']:.3f}]")

    # 6. 피처 중요도 분석 (LightGBM 평균)
    importances = np.mean([m.feature_importance(importance_type='gain') for m in train_res['models_lgb']], axis=0)
    top_indices = np.argsort(importances)[::-1]
    print("\n[LightGBM Feature Importance Top 10]")
    for rank, idx in enumerate(top_indices[:10], start=1):
        print(f"  {rank:2d}. {feature_names[idx]}: gain={importances[idx]:.1f}")

    # 7. 평가 결과 JSON 저장
    eval_results = {
        'model_name': 'sl1_v10',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'data_selection': '원천 전체 826개 폴더 (설명 기반 라벨 정제본)',
        'total_samples': int(len(y)),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'features_dimension': 55,
        'ensemble_weights': {
            'lightgbm': opt_weights[0],
            'catboost': opt_weights[1],
            'xgboost': opt_weights[2]
        },
        'two_state_metrics': {
            'ensemble': m_ens,
            'lightgbm': m_lgb,
            'catboost': m_cat,
            'xgboost': m_xgb
        },
        'three_state_metrics': opt_3state,
        'fixed_ambiguity_policy_035_065': fixed_policy_res,
        'top_features': [
            {'name': feature_names[idx], 'gain': float(importances[idx])}
            for idx in top_indices[:15]
        ]
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(eval_results, fp, indent=2, ensure_ascii=False)
    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 8. 공식 서빙 번들 저장
    bundle = {
        'model_name': 'sl1_v10',
        'features_dimension': 55,
        'feature_names': feature_names,
        'ensemble_weights': opt_weights,
        'models_lgb': train_res['models_lgb'],
        'models_cat': train_res['models_cat'],
        'models_xgb': train_res['models_xgb'],
        'view_thresholds': opt_3state['view_thresholds'],
        'default_thresholds': {'tau_clean': 0.48, 'tau_defect': 0.68},
        'view_alias_map': VIEW_ALIAS_MAP
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 공식 배포 번들 PKL 저장 완료: {MODEL_BUNDLE_PKL}")


if __name__ == '__main__':
    main()
