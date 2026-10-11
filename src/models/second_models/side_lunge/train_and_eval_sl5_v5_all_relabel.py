# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: train_and_eval_sl5_v5_all_relabel.py
Purpose: 사이드 런지 5번째 오류 '상체의 과조한 숙임/젖힘 여부' sl5_v5 모델을
         원천 데이터 826개 폴더 전체(4,130개 시퀀스)를 대상으로 설명(description) 기반
         라벨 노이즈 교정 데이터셋을 활용하여 전수 재학습 및 다차원 성능 평가
         - 5-Way 앙상블 (LightGBM + CatBoost + XGBoost + RF + ExtraTrees)
         - 5-Fold 피험자 1~9 전원 엄격 격리 교차검증 (StratifiedGroupKFold)
         - 전체 3-State F1 최적화 (보류율 10% 이하 엄격 준수)
         - 기존 정제 492폴더 버전 vs 전체 826폴더 설명 기반 정리 버전 성능 대조
         - AGENTS.md 공식 5대 시점 전수 성능 감사
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import pickle
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\side_lunge_c5_v5_all_relabel_36d_features.npz"
CLEAN_492_BENCHMARK_PATH = r"D:\src\models\second_models\side_lunge\sl5_v5_benchmark_results.json"
BL5_BENCHMARK_PATH = r"D:\src\models\second_models\babel_lunge\bl5_v5_benchmark_results.json"
MODEL_DIR = r"D:\src\models\second_models\side_lunge"
WEIGHT_DIR = os.path.join(MODEL_DIR, "weights")
BUNDLE_SAVE_PATH = os.path.join(WEIGHT_DIR, "sl5_v5_all_relabel_bundle.pkl")
BENCHMARK_SAVE_PATH = os.path.join(MODEL_DIR, "sl5_v5_all_relabel_benchmark_results.json")

os.makedirs(WEIGHT_DIR, exist_ok=True)

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_all_relabel_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(FEATURE_CACHE_PATH):
        raise FileNotFoundError(f"피처 캐시 파일이 존재하지 않습니다: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    keys = data['keys'].astype(str)
    filenames = data['filenames'].astype(str)
    feat_names = list(data['feature_names'])

    print(f"[*] 사이드 런지 전체 826폴더 설명 기반 정리 데이터셋 로드 성공:")
    print(f"    - 총 샘플 수: {X.shape[0]}개 (피처 차원: {X.shape[1]}D)")
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)")
    print(f"    - 피험자: {np.unique(subjects).tolist()} ({len(np.unique(subjects))}명 전수)")
    print(f"    - 시점: {np.unique(views).tolist()} (5대 시점 전수)")
    return X, y, subjects, views, keys, filenames, feat_names


def train_and_eval_pipeline():
    print("=" * 80)
    print("  [vidio-health-trainer] sl5_v5 전체 설명 기반 라벨 정리 재학습 및 평가 파이프라인")
    print("=" * 80)

    X, y, subjects, views, keys, filenames, feat_names = load_all_relabel_dataset()
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_matrix = np.zeros((N, 5), dtype=np.float32)

    lgb_models = []
    cb_models = []
    xgb_models = []
    rf_models = []
    et_models = []

    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    print("\n[*] 5-Fold 피험자 전수 격리 학습 개시 (LightGBM + CatBoost + XGBoost + RF + ExtraTrees)...")
    t0 = time.time()

    femur_ratio_idx = feat_names.index("lunge_femur_to_torso_ratio") if "lunge_femur_to_torso_ratio" in feat_names else 0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        subjs_val = np.unique(subjects[val_idx]).tolist()

        sample_weights = np.ones(len(train_idx), dtype=np.float32)
        for i in range(len(train_idx)):
            if X_train[i, femur_ratio_idx] > 1.05 and y_train[i] == 0:
                sample_weights[i] *= 1.5

        # 1. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.032,
            num_leaves=26,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.18,
            reg_lambda=1.5,
            scale_pos_weight=pos_ratio,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_lgb = lgb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 0] = pred_lgb
        lgb_models.append(lgb_clf)

        # 2. CatBoost
        cb_clf = CatBoostClassifier(
            iterations=240,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=4.8,
            scale_pos_weight=pos_ratio,
            random_seed=42 + fold,
            thread_count=1,
            verbose=0
        )
        cb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_cb = cb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 1] = pred_cb
        cb_models.append(cb_clf)

        # 3. XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.032,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.18,
            reg_lambda=1.5,
            scale_pos_weight=pos_ratio,
            random_state=42 + fold,
            n_jobs=1,
            eval_metric='logloss'
        )
        xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_xgb = xgb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 2] = pred_xgb
        xgb_models.append(xgb_clf)

        # 4. Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        rf_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_rf = rf_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 3] = pred_rf
        rf_models.append(rf_clf)

        # 5. Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        et_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_et = et_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 4] = pred_et
        et_models.append(et_clf)

        fold_mean_p = np.mean(oof_matrix[val_idx], axis=1)
        val_f1 = f1_score(y_val, (fold_mean_p >= 0.5).astype(int), zero_division=0)
        val_auc = roc_auc_score(y_val, fold_mean_p) if len(np.unique(y_val)) > 1 else 0.5
        print(f"    [Fold {fold}/5] 검증 피험자: {subjs_val} | 샘플: {len(val_idx)}개 | Base평균 F1: {val_f1*100:.2f}% | AUC: {val_auc:.4f}")

    elapsed_train = time.time() - t0
    print(f"[*] 5-Fold Base 학습 완료 (소요시간: {elapsed_train:.1f}초)")

    # 5-Way SOTA 앙상블 블렌딩 & Temperature Scaling
    print("\n[*] 5-Way SOTA 앙상블 블렌딩 (LGB 0.20 + CB 0.30 + XGB 0.30 + RF 0.10 + ET 0.10)...")
    oof_p = (
        0.20 * oof_matrix[:, 0] +
        0.30 * oof_matrix[:, 1] +
        0.30 * oof_matrix[:, 2] +
        0.10 * oof_matrix[:, 3] +
        0.10 * oof_matrix[:, 4]
    )

    best_temp = 1.02
    eps = 1e-6
    cp = np.clip(oof_p, eps, 1.0 - eps)
    logits = np.log(cp / (1.0 - cp))
    oof_final_probs = 1.0 / (1.0 + np.exp(-logits / best_temp))

    # 1. 100% Full-Coverage Baseline
    full_preds = (oof_final_probs >= 0.5).astype(int)
    full_f1 = f1_score(y, full_preds, zero_division=0)
    full_prec = precision_score(y, full_preds, zero_division=0)
    full_rec = recall_score(y, full_preds, zero_division=0)
    full_acc = accuracy_score(y, full_preds)
    full_auc = roc_auc_score(y, oof_final_probs)

    print("\n" + "=" * 80)
    print("  [sl5_v5 전체 설명기반정리 OOF 풀 커버리지 (100% Coverage, Th=0.50) 성능]")
    print("=" * 80)
    print(f"  - Full F1-Score : {full_f1*100:.2f}%")
    print(f"  - Full Precision: {full_prec*100:.2f}%")
    print(f"  - Full Recall   : {full_rec*100:.2f}%")
    print(f"  - Full Accuracy : {full_acc*100:.2f}%")
    print(f"  - ROC-AUC       : {full_auc:.4f}")

    # 2. 3-State F1 최적화 (보류율 최대 10% 이하 제약: Coverage >= 90.0%)
    print("\n[*] 전체 3-State F1-Score 최적 임계치 그리드 탐색 (보류율 <= 10.0% 엄격 제약)...")
    best_f1_score = -1.0
    best_th_l = 0.40
    best_th_h = 0.60
    best_coverage = 1.0
    best_c_f1 = 0.0
    best_c_prec = 0.0
    best_c_rec = 0.0
    best_c_acc = 0.0
    best_cm = None
    best_confirmed = 0
    best_caution = 0

    th_candidates = np.linspace(0.20, 0.70, 51)
    for tl in th_candidates:
        for th in th_candidates:
            if th < tl + 0.02:
                continue

            conf_mask = (oof_final_probs < tl) | (oof_final_probs >= th)
            cov = float(np.mean(conf_mask))

            if cov < 0.90:
                continue

            y_conf = y[conf_mask]
            p_conf = (oof_final_probs[conf_mask] >= th).astype(int)

            f1 = f1_score(y_conf, p_conf, zero_division=0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_th_l = float(tl)
                best_th_h = float(th)
                best_coverage = float(cov)
                best_c_f1 = float(f1)
                best_c_prec = float(precision_score(y_conf, p_conf, zero_division=0))
                best_c_rec = float(recall_score(y_conf, p_conf, zero_division=0))
                best_c_acc = float(accuracy_score(y_conf, p_conf))
                best_cm = confusion_matrix(y_conf, p_conf).tolist()
                best_confirmed = int(np.sum(conf_mask))
                best_caution = int(N - best_confirmed)

    print("\n" + "=" * 80)
    print("  [sl5_v5 전체 설명기반정리 최적 3-State 신뢰 분류 구간 탐색 결과]")
    print("=" * 80)
    print(f"  - 최적 판정 구간 : Normal < {best_th_l:.3f} | Caution [{best_th_l:.3f}, {best_th_h:.3f}) | Defect >= {best_th_h:.3f}")
    print(f"  - 커버리지 (Coverage)   : {best_coverage*100:.2f}% (확정 {best_confirmed}건 / 보류 {best_caution}건)")
    print(f"  - 보류율 (Rejection Rate): {(1.0 - best_coverage)*100:.2f}% (<= 10.0% 엄격 충족)")
    print(f"  - 3-State F1-Score      : {best_c_f1*100:.2f}% (100% Full 대비 +{(best_c_f1 - full_f1)*100:.2f}%p)")
    print(f"  - 3-State Precision     : {best_c_prec*100:.2f}%")
    print(f"  - 3-State Recall        : {best_c_rec*100:.2f}%")
    print(f"  - 3-State Accuracy      : {best_c_acc*100:.2f}%")
    print(f"  - Confusion Matrix      : {best_cm}")

    # 3. AGENTS.md 공식 5대 시점 전수 감사
    print("\n" + "=" * 80)
    print("  [AGENTS.md 공식 5대 시점(view1~view5) 전수 성능 감사]")
    print("=" * 80)

    view_audit_results = {}
    for vn, vname in OFFICIAL_VIEW_MAP.items():
        vmask = (views == vn)
        n_v = np.sum(vmask)
        if n_v == 0:
            continue

        y_v = y[vmask]
        prob_v = oof_final_probs[vmask]

        conf_v = (prob_v < best_th_l) | (prob_v >= best_th_h)
        v_cov = float(np.mean(conf_v))
        v_rej = float(1.0 - v_cov)

        y_v_conf = y_v[conf_v]
        p_v_conf = (prob_v[conf_v] >= best_th_h).astype(int)

        v_f1 = float(f1_score(y_v_conf, p_v_conf, zero_division=0))
        v_prec = float(precision_score(y_v_conf, p_v_conf, zero_division=0))
        v_rec = float(recall_score(y_v_conf, p_v_conf, zero_division=0))
        v_acc = float(accuracy_score(y_v_conf, p_v_conf))
        v_auc = float(roc_auc_score(y_v, prob_v)) if len(np.unique(y_v)) > 1 else 0.5
        v_cm = confusion_matrix(y_v_conf, p_v_conf).tolist()

        view_audit_results[vn] = {
            "view_name": vname,
            "samples": int(n_v),
            "confirmed": int(np.sum(conf_v)),
            "caution": int(n_v - np.sum(conf_v)),
            "coverage": v_cov,
            "rejection_rate": v_rej,
            "f1": v_f1,
            "precision": v_prec,
            "recall": v_rec,
            "accuracy": v_acc,
            "auc": v_auc,
            "cm": v_cm
        }

        print(f"  [{vname}] (총 {n_v}건)")
        print(f"    - 커버리지: {v_cov*100:.1f}% (확정 {np.sum(conf_v)} / 보류 {n_v - np.sum(conf_v)})")
        print(f"    - 3-State F1: {v_f1*100:.2f}% | Precision: {v_prec*100:.2f}% | Recall: {v_rec*100:.2f}%")
        print(f"    - 정확도: {v_acc*100:.2f}% | AUC: {v_auc:.4f}")

    # 4. 이전 정제 492폴더 버전 및 바벨런지 대조
    clean_492_f1 = None
    if os.path.exists(CLEAN_492_BENCHMARK_PATH):
        try:
            with open(CLEAN_492_BENCHMARK_PATH, 'r', encoding='utf-8') as f:
                c_data = json.load(f)
            clean_492_f1 = c_data.get("best_3state", {}).get("f1")
        except Exception:
            pass

    bl5_f1 = None
    if os.path.exists(BL5_BENCHMARK_PATH):
        try:
            with open(BL5_BENCHMARK_PATH, 'r', encoding='utf-8') as f:
                bl_data = json.load(f)
            bl5_f1 = bl_data.get("best_3state", {}).get("f1")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("  [이전 정제 492폴더 버전 vs 전체 826폴더 설명 기반 정리 버전 대조]")
    print("=" * 80)
    if clean_492_f1 is not None:
        diff_clean = (best_c_f1 - clean_492_f1) * 100
        sign = "+" if diff_clean >= 0 else ""
        print(f"  - 정제 492폴더 버전 3S-F1     : {clean_492_f1*100:.2f}% (2,460건)")
        print(f"  - 전체 826폴더 설명정리 3S-F1 : {best_c_f1*100:.2f}% ({N}건) (격차: {sign}{diff_clean:.2f}%p)")

    # 5. 가중치 번들 저장
    bundle = {
        "version": "sl5_v5_all_relabel",
        "models": {
            "lgb": lgb_models,
            "cb": cb_models,
            "xgb": xgb_models,
            "rf": rf_models,
            "et": et_models
        },
        "blend_weights": [0.20, 0.30, 0.30, 0.10, 0.10],
        "temperature": best_temp,
        "th_low": best_th_l,
        "th_high": best_th_h,
        "feature_names": feat_names,
        "femur_ratio_idx": femur_ratio_idx,
        "official_view_map": OFFICIAL_VIEW_MAP
    }
    with open(BUNDLE_SAVE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[+] 가중치 번들 저장 완료: {BUNDLE_SAVE_PATH}")

    # 6. 벤치마크 결과 JSON 저장
    benchmark_res = {
        "version": "sl5_v5_all_relabel",
        "dataset_summary": {
            "total_samples": int(N),
            "normal_samples": int(np.sum(y == 0)),
            "defect_samples": int(np.sum(y == 1)),
            "defect_ratio": float(np.mean(y == 1)),
            "subjects": [int(s) for s in np.unique(subjects)],
            "views": [str(v) for v in np.unique(views)]
        },
        "full_coverage": {
            "f1": float(full_f1),
            "precision": float(full_prec),
            "recall": float(full_rec),
            "accuracy": float(full_acc),
            "auc": float(full_auc)
        },
        "best_3state": {
            "f1": float(best_c_f1),
            "th_low": float(best_th_l),
            "th_high": float(best_th_h),
            "coverage": float(best_coverage),
            "rejection_rate": float(1.0 - best_coverage),
            "precision": float(best_c_prec),
            "recall": float(best_c_rec),
            "accuracy": float(best_c_acc),
            "cm": best_cm,
            "total_confirmed": int(best_confirmed),
            "total_caution": int(best_caution)
        },
        "view_audit": view_audit_results,
        "baseline_comparison": {
            "clean_492_f1": float(clean_492_f1) if clean_492_f1 is not None else None,
            "bl5_v5_f1": float(bl5_f1) if bl5_f1 is not None else None,
            "diff_vs_clean": float(best_c_f1 - clean_492_f1) if clean_492_f1 is not None else None
        }
    }
    with open(BENCHMARK_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_res, f, ensure_ascii=False, indent=2)
    print(f"[+] 벤치마크 JSON 저장 완료: {BENCHMARK_SAVE_PATH}")


if __name__ == '__main__':
    train_and_eval_pipeline()
