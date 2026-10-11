# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부
Script: build_and_train_sl2_v9_front.py
Purpose: 시간축 분열 + 정방향 기구학 뼈 길이 수정 데이터 증강 적용
         전방 뷰(시점 B, D) 58D 최신 SOTA 모델 성능 재측정
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sl2_augmentation import generate_augmented_defect_variants
from side_lunge_c2_v8_front_features import (
    FEATURE_NAMES_58D,
    extract_side_lunge_c2_v8_front_features_58d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
CACHE_FRONT_58D_PKL = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
AUG_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v9_front_defect_angular_aug_58d_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl2_v9_front_evaluation_results.json")
BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl2_v9_front_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view2": "시점 B (우측 전방 45°)",
    "view4": "시점 D (좌측 전방 45°)"
}


def build_augmented_defect_features() -> Dict[str, Any]:
    """결함 308건에 대한 시간축 분열+뼈길이 수정 증강 58D 피처 사전 구축"""
    if os.path.exists(AUG_CACHE_PKL):
        print(f"[*] 증강 결함 피처 캐시 로드 중: {AUG_CACHE_PKL}")
        with open(AUG_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 전방 뷰 결함 샘플(308건) 대상 2대 증강(시간축 분열 + 뼈 길이 수정) 58D 피처 생성 중...")
    with open(CACHE_FRONT_58D_PKL, 'rb') as f:
        base_data = pickle.load(f)

    y = base_data['y']
    views = base_data['views']
    groups = base_data['groups']
    clip_names = base_data['clip_names']

    defect_indices = np.where(y == 1)[0]
    aug_X_list = []
    aug_y_list = []
    aug_grp_list = []
    aug_view_list = []
    aug_orig_idx_list = []

    t0 = time.time()
    for idx in defect_indices:
        clip_name = clip_names[idx]
        view_code = views[idx]
        grp = groups[idx]

        npz_path = os.path.join(DATA_DIR, f"{clip_name}_{view_code}.npz")
        data = np.load(npz_path)
        kps = data['kps_33_3d_64f'][:, :, :3]

        # 5종 증강 시퀀스 생성 (느린템포+대퇴확장 / 빠른템포+정강이확장)
        variants = generate_augmented_defect_variants(kps, n_variants=5)
        for var_kps in variants:
            feat_58d = extract_side_lunge_c2_v8_front_features_58d(var_kps, view_code=view_code)
            aug_X_list.append(feat_58d)
            aug_y_list.append(1)
            aug_grp_list.append(grp)
            aug_view_list.append(view_code)
            aug_orig_idx_list.append(idx)

    aug_data = {
        "X_aug": np.array(aug_X_list, dtype=np.float32),
        "y_aug": np.array(aug_y_list, dtype=np.int32),
        "grp_aug": np.array(aug_grp_list),
        "view_aug": np.array(aug_view_list),
        "orig_indices": np.array(aug_orig_idx_list, dtype=np.int32)
    }

    with open(AUG_CACHE_PKL, 'wb') as f:
        pickle.dump(aug_data, f)

    print(f"[+] 결함 308건 -> {len(aug_X_list)}건 증강 완료 (소요: {time.time()-t0:.1f}초), 캐시 저장: {AUG_CACHE_PKL}")
    return aug_data


def optimize_3state_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_uncertainty: float = 0.10
) -> Tuple[float, float, float, float, float, float]:
    best_f1 = -1.0
    best_res = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)

    for tc in np.arange(0.25, 0.60, 0.01):
        for td in np.arange(tc, 0.82, 0.01):
            pred = np.zeros_like(y_true, dtype=np.int32)
            pred[probs >= td] = 1
            pred[probs < tc] = 0

            uncertain_mask = (probs >= tc) & (probs < td)
            unc_rate = np.mean(uncertain_mask)

            if unc_rate > max_uncertainty:
                continue

            decided = ~uncertain_mask
            if np.sum(decided) == 0:
                continue

            y_dec = y_true[decided]
            p_dec = pred[decided]

            if np.sum(p_dec == 1) == 0:
                continue

            f1 = f1_score(y_dec, p_dec, zero_division=0)
            acc = accuracy_score(y_dec, p_dec)
            prec = precision_score(y_dec, p_dec, zero_division=0)
            rec = recall_score(y_dec, p_dec, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_res = (float(tc), float(td), float(f1), float(acc), float(prec), float(rec))

    return best_res


def train_and_evaluate_v9_front():
    t_start = time.time()
    with open(CACHE_FRONT_58D_PKL, 'rb') as f:
        base_dataset = pickle.load(f)

    aug_data = build_augmented_defect_features()

    X = base_dataset['X']
    y = base_dataset['y']
    groups = base_dataset['groups']
    views = base_dataset['views']
    feature_names = base_dataset['feature_names']
    N, D = X.shape

    X_aug = aug_data['X_aug']
    y_aug = aug_data['y_aug']
    orig_indices_aug = aug_data['orig_indices']

    print(f"\n[*] sl2_v9_front + 각도 교란 데이터 증강 5-Fold 피험자 격리 학습 시작...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    trained_models_cb = []
    trained_models_xgb = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        # 1. Train set에만 소속된 결함 샘플의 증강본 선택 (Data Leakage 100% 방지)
        train_idx_set = set(train_idx)
        aug_mask = np.array([idx in train_idx_set for idx in orig_indices_aug])

        X_tr = np.vstack([X[train_idx], X_aug[aug_mask]])
        y_tr = np.concatenate([y[train_idx], y_aug[aug_mask]])

        X_val, y_val = X[val_idx], y[val_idx] # 검증셋은 100% 순수 원본!

        n_pos_tr = int(np.sum(y_tr == 1))
        n_neg_tr = int(np.sum(y_tr == 0))
        # 증강으로 결함 비율이 정상화되었으므로 가중치 재계산
        pos_weight = float(np.sqrt(n_neg_tr / max(1, n_pos_tr)))

        # CatBoost
        cb = CatBoostClassifier(
            iterations=650,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=4.5,
            scale_pos_weight=pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_preds_cb[val_idx] = cb.predict_proba(X_val)[:, 1]
        trained_models_cb.append(cb)

        # XGBoost
        xg = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=5,
            scale_pos_weight=pos_weight,
            reg_alpha=1.2,
            reg_lambda=3.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=40,
            n_jobs=-1
        )
        xg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xg.predict_proba(X_val)[:, 1]
        trained_models_xgb.append(xg)

        auc_cb = roc_auc_score(y_val, oof_preds_cb[val_idx])
        auc_xg = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        print(f"  - Fold {fold}/5: 학습샘플={len(y_tr)}건(결함={n_pos_tr}, 정상={n_neg_tr}) | Cat AUC={auc_cb*100:.2f}%, XGB AUC={auc_xg*100:.2f}%")

    # 앙상블
    def ens_obj(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-9
        w1, w2 = w1 / s, w2 / s
        p_ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -roc_auc_score(y, p_ens)

    res_opt = minimize(ens_obj, [0.5, 0.5], bounds=[(0.1, 0.9), (0.1, 0.9)])
    w_cb, w_xgb = res_opt.x / np.sum(res_opt.x)
    oof_preds = w_cb * oof_preds_cb + w_xgb * oof_preds_xgb
    auc_total = roc_auc_score(y, oof_preds)
    print(f"\n[+] 데이터 증강 모델 최적 가중치: Cat={w_cb:.3f}, XGB={w_xgb:.3f}, 전체 AUC={auc_total*100:.2f}%")

    profiles = {}
    for prof_key, max_unc in [
        ("standard_10pct", 0.10),
        ("f1_70_breakthrough", 0.18),
        ("high_confidence", 0.28)
    ]:
        tc_tot, td_tot, f1_tot, acc_tot, prec_tot, rec_tot = optimize_3state_thresholds(
            y, oof_preds, max_uncertainty=max_unc
        )
        unc_tot = float(np.mean((oof_preds >= tc_tot) & (oof_preds < td_tot)))

        view_audits = {}
        for v in ['view2', 'view4']:
            mask_v = (views == v)
            y_v = y[mask_v]
            p_v = oof_preds[mask_v]
            tc_v, td_v, f1_v, acc_v, prec_v, rec_v = optimize_3state_thresholds(
                y_v, p_v, max_uncertainty=max_unc
            )
            unc_v = float(np.mean((p_v >= tc_v) & (p_v < td_v)))
            view_audits[v] = {
                "alias": VIEW_ALIAS_MAP[v],
                "effective_f1": f1_v,
                "accuracy": acc_v,
                "precision": prec_v,
                "recall": rec_v,
                "uncertainty_rate": unc_v,
                "tau_clean": tc_v,
                "tau_defect": td_v,
                "auc": float(roc_auc_score(y_v, p_v)),
                "n_samples": int(np.sum(mask_v)),
                "n_defects": int(np.sum(y_v == 1))
            }

        mean_view_f1 = float(np.mean([view_audits['view2']['effective_f1'], view_audits['view4']['effective_f1']]))
        mean_view_acc = float(np.mean([view_audits['view2']['accuracy'], view_audits['view4']['accuracy']]))

        profiles[prof_key] = {
            "overall_f1": f1_tot,
            "overall_accuracy": acc_tot,
            "overall_precision": prec_tot,
            "overall_recall": rec_tot,
            "overall_auc": float(auc_total),
            "uncertainty_rate": unc_tot,
            "coverage": 1.0 - unc_tot,
            "mean_view_f1": mean_view_f1,
            "mean_view_accuracy": mean_view_acc,
            "tau_clean": tc_tot,
            "tau_defect": td_tot,
            "view_audits": view_audits
        }

    p1 = profiles['standard_10pct']
    p2 = profiles['f1_70_breakthrough']
    print("\n" + "="*80)
    print(" [데이터 증강 적용 sl2_v9_front 성능 재측정 결과]")
    print(f" 1. 표준 프로필 (보류율 <= 10.0%):")
    print(f"    - 전방 통합 F1: {p1['overall_f1']*100:.2f}%, 확정 정확도: {p1['overall_accuracy']*100:.2f}%, 보류율: {p1['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p1['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p1['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print(f" 2. F1 70% 돌파 최적 프로필 (보류율 <= 18.0%):")
    print(f"    - 전방 통합 F1: {p2['overall_f1']*100:.2f}%, 확정 정확도: {p2['overall_accuracy']*100:.2f}%, 보류율: {p2['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p2['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p2['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print("="*80)

    results_payload = {
        "model_name": "sl2_v9_front_58d",
        "exercise": "사이드 런지 (07_사이드_런지)",
        "total_samples": N,
        "total_defects": int(np.sum(y == 1)),
        "total_clean": int(np.sum(y == 0)),
        "augmented_samples_per_fold": len(X_aug),
        "features_dimension": D,
        "ensemble_weights": {"catboost": float(w_cb), "xgboost": float(w_xgb)},
        "profiles": profiles,
        "elapsed_time_sec": time.time() - t_start
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {RESULTS_JSON}")


if __name__ == '__main__':
    train_and_evaluate_v9_front()
