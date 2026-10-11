# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: train_and_eval_bl5_v3.py
Purpose: 바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' bl5_v3 모델 학습,
         피처 전수조사 최적화 28D 적용 및 모델 구조 최적화 (4대 앙상블 + 하이퍼파라미터 튜닝),
         5-Fold 피험자 전수 격리 교차검증, 전체 3-State F1 최적화, 판단 보류율 산출 및 5대 시점 감사
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v3_28d_features.npz"
MODEL_DIR = r"D:\src\models\second_models\babel_lunge"
WEIGHT_DIR = os.path.join(MODEL_DIR, "weights")
BUNDLE_SAVE_PATH = os.path.join(WEIGHT_DIR, "bl5_v3_bundle.pkl")
BENCHMARK_SAVE_PATH = os.path.join(MODEL_DIR, "bl5_v3_benchmark_results.json")

os.makedirs(WEIGHT_DIR, exist_ok=True)

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_v3_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(FEATURE_CACHE_PATH):
        raise FileNotFoundError(f"피처 캐시 파일이 존재하지 않습니다: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    feat_names = list(data['feature_names'])

    print(f"[*] v3 최적화 데이터셋 로드 성공:")
    print(f"    - 총 샘플 수: {X.shape[0]}개 (피처 차원: {X.shape[1]}D)")
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)")
    print(f"    - 피험자: {np.unique(subjects).tolist()} ({len(np.unique(subjects))}명 전수)")
    print(f"    - 시점: {np.unique(views).tolist()} (5대 시점 전수)")
    return X, y, subjects, views, feat_names


def train_and_eval_v3_pipeline():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_v3 피처/모델 최적화 및 전체 3-State F1 평가 파이프라인")
    print("=" * 80)

    X, y, subjects, views, feat_names = load_v3_dataset()
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(N, dtype=np.float32)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_rf = np.zeros(N, dtype=np.float32)
    oof_preds_et = np.zeros(N, dtype=np.float32)

    lgb_models = []
    cb_models = []
    rf_models = []
    et_models = []

    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    print("\n[*] 5-Fold 피험자 격리 학습 개시 (LightGBM + CatBoost + RandomForest + ExtraTrees 최적화)...")
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        subjs_val = np.unique(subjects[val_idx]).tolist()

        # 1. LightGBM 최적화 (28D 정규화 및 피처 서브샘플링)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=22,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_ratio,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train)
        val_pred_lgb = lgb_clf.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_pred_lgb
        lgb_models.append(lgb_clf)

        # 2. CatBoost 최적화
        cb_clf = CatBoostClassifier(
            iterations=200,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=4.0,
            scale_pos_weight=pos_ratio,
            random_seed=42 + fold,
            thread_count=1,
            verbose=0
        )
        cb_clf.fit(X_train, y_train)
        val_pred_cb = cb_clf.predict_proba(X_val)[:, 1]
        oof_preds_cb[val_idx] = val_pred_cb
        cb_models.append(cb_clf)

        # 3. Random Forest 최적화
        rf_clf = RandomForestClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        rf_clf.fit(X_train, y_train)
        val_pred_rf = rf_clf.predict_proba(X_val)[:, 1]
        oof_preds_rf[val_idx] = val_pred_rf
        rf_models.append(rf_clf)

        # 4. Extra Trees 최적화
        et_clf = ExtraTreesClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        et_clf.fit(X_train, y_train)
        val_pred_et = et_clf.predict_proba(X_val)[:, 1]
        oof_preds_et[val_idx] = val_pred_et
        et_models.append(et_clf)

        val_pred_ens = 0.35 * val_pred_lgb + 0.35 * val_pred_cb + 0.15 * val_pred_rf + 0.15 * val_pred_et
        val_f1 = f1_score(y_val, (val_pred_ens >= 0.5).astype(int), zero_division=0)
        val_auc = roc_auc_score(y_val, val_pred_ens) if len(np.unique(y_val)) > 1 else 0.5
        print(f"    [Fold {fold}/5] 검증 피험자: {subjs_val} | 샘플: {len(val_idx)}개 | 앙상블 F1: {val_f1*100:.2f}% | AUC: {val_auc:.4f}")

    elapsed_train = time.time() - t0
    print(f"[*] 5-Fold 학습 완료 (소요시간: {elapsed_train:.1f}초)")

    oof_ensemble_probs = (
        0.35 * oof_preds_lgb +
        0.35 * oof_preds_cb +
        0.15 * oof_preds_rf +
        0.15 * oof_preds_et
    )

    # 1. 100% Full-Coverage Baseline
    full_preds = (oof_ensemble_probs >= 0.5).astype(int)
    full_f1 = f1_score(y, full_preds, zero_division=0)
    full_prec = precision_score(y, full_preds, zero_division=0)
    full_rec = recall_score(y, full_preds, zero_division=0)
    full_acc = accuracy_score(y, full_preds)
    full_auc = roc_auc_score(y, oof_ensemble_probs)

    print("\n" + "=" * 80)
    print("  [v3 OOF 풀 커버리지 (100% Coverage, Th=0.50) 성능]")
    print("=" * 80)
    print(f"  - Full F1-Score : {full_f1*100:.2f}%")
    print(f"  - Full Precision: {full_prec*100:.2f}%")
    print(f"  - Full Recall   : {full_rec*100:.2f}%")
    print(f"  - Full Accuracy : {full_acc*100:.2f}%")
    print(f"  - ROC-AUC       : {full_auc:.4f}")

    # 2. 전체 3-State F1 최적화 및 판단 보류율 산출
    print("\n[*] 전체 3-State F1-Score 최적 임계치 그리드 서치 탐색...")
    best_3state = {
        'f1': 0.0,
        'th_low': 0.30,
        'th_high': 0.60,
        'coverage': 0.0,
        'rejection_rate': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'cm': None
    }

    for th_l in np.linspace(0.20, 0.42, 23):
        for th_h in np.linspace(0.52, 0.78, 27):
            if th_h <= th_l + 0.12:
                continue

            high_conf_mask = (oof_ensemble_probs < th_l) | (oof_ensemble_probs >= th_h)
            cov = float(np.mean(high_conf_mask))

            if cov < 0.72:  # 최소 72% 이상 커버리지 유지 (보류율 최대 28% 이내)
                continue

            y_conf = y[high_conf_mask]
            pred_conf = (oof_ensemble_probs[high_conf_mask] >= th_h).astype(int)

            score_f1 = f1_score(y_conf, pred_conf, zero_division=0)
            score_rec = recall_score(y_conf, pred_conf, zero_division=0)
            score_prec = precision_score(y_conf, pred_conf, zero_division=0)
            score_acc = accuracy_score(y_conf, pred_conf)

            obj_score = score_f1 * 0.7 + score_rec * 0.15 + score_prec * 0.15
            if obj_score > (best_3state['f1'] * 0.7 + best_3state['recall'] * 0.15 + best_3state['precision'] * 0.15):
                best_3state = {
                    'f1': float(score_f1),
                    'th_low': float(th_l),
                    'th_high': float(th_h),
                    'coverage': float(cov),
                    'rejection_rate': float(1.0 - cov),
                    'precision': float(score_prec),
                    'recall': float(score_rec),
                    'accuracy': float(score_acc),
                    'cm': confusion_matrix(y_conf, pred_conf).tolist()
                }

    print("\n" + "=" * 80)
    print("  [v3 최적 전체 3-State F1-Score 및 판단 보류율 성과 보고]")
    print("=" * 80)
    print(f"  ★ 최적 전체 3-State F1-Score : {best_3state['f1']*100:.2f}%")
    print(f"  ★ 판단 보류율 (Rejection Rate) : {best_3state['rejection_rate']*100:.2f}% (커버리지: {best_3state['coverage']*100:.2f}%)")
    print(f"    - 3-State Precision         : {best_3state['precision']*100:.2f}%")
    print(f"    - 3-State Recall            : {best_3state['recall']*100:.2f}%")
    print(f"    - 3-State Accuracy          : {best_3state['accuracy']*100:.2f}%")
    print(f"    - 최적 임계치 밴드          : Th_Low = {best_3state['th_low']:.3f}, Th_High = {best_3state['th_high']:.3f}")
    print(f"    - 3-State 혼동행렬 (TN, FP / FN, TP): {best_3state['cm']}")

    # 3. 5대 공식 시점별 세부 성능 감사
    print("\n" + "=" * 80)
    print("  [AGENTS.md 규정: 5대 공식 시점별 3-State 및 Full F1 전수 감사]")
    print("=" * 80)

    view_audit_results = {}
    for v_code, v_name in OFFICIAL_VIEW_MAP.items():
        v_mask = (views == v_code)
        if not np.any(v_mask):
            continue

        y_v = y[v_mask]
        probs_v = oof_ensemble_probs[v_mask]

        full_preds_v = (probs_v >= 0.5).astype(int)
        v_full_f1 = f1_score(y_v, full_preds_v, zero_division=0)
        v_full_rec = recall_score(y_v, full_preds_v, zero_division=0)
        v_full_prec = precision_score(y_v, full_preds_v, zero_division=0)
        v_full_acc = accuracy_score(y_v, full_preds_v)

        th_l = best_3state['th_low']
        th_h = best_3state['th_high']
        conf_mask_v = (probs_v < th_l) | (probs_v >= th_h)
        v_cov = float(np.mean(conf_mask_v))
        v_rej = float(1.0 - v_cov)

        if np.sum(conf_mask_v) > 0:
            y_v_conf = y_v[conf_mask_v]
            pred_v_conf = (probs_v[conf_mask_v] >= th_h).astype(int)
            v_3s_f1 = f1_score(y_v_conf, pred_v_conf, zero_division=0)
            v_3s_rec = recall_score(y_v_conf, pred_v_conf, zero_division=0)
            v_3s_prec = precision_score(y_v_conf, pred_v_conf, zero_division=0)
            v_3s_acc = accuracy_score(y_v_conf, pred_v_conf)
        else:
            v_3s_f1 = v_3s_rec = v_3s_prec = v_3s_acc = 0.0

        view_audit_results[v_code] = {
            'view_name': v_name,
            'total_samples': int(len(y_v)),
            'defects': int(np.sum(y_v == 1)),
            'full_coverage_f1': float(v_full_f1),
            'full_coverage_recall': float(v_full_rec),
            'full_coverage_precision': float(v_full_prec),
            'full_coverage_accuracy': float(v_full_acc),
            'three_state_f1': float(v_3s_f1),
            'three_state_recall': float(v_3s_rec),
            'three_state_precision': float(v_3s_prec),
            'three_state_accuracy': float(v_3s_acc),
            'coverage': float(v_cov),
            'rejection_rate': float(v_rej)
        }

        print(f"  * {v_name} ({v_code}): 총 {len(y_v)}개 (결함 {np.sum(y_v==1)}개)")
        print(f"    - [Full 100%]  F1: {v_full_f1*100:5.2f}% | Rec: {v_full_rec*100:5.2f}% | Prec: {v_full_prec*100:5.2f}% | Acc: {v_full_acc*100:5.2f}%")
        print(f"    - [3-State]    F1: {v_3s_f1*100:5.2f}% | Rec: {v_3s_rec*100:5.2f}% | Prec: {v_3s_prec*100:5.2f}% | Acc: {v_3s_acc*100:5.2f}% (보류율: {v_rej*100:4.1f}%)")

    # 4. 피처 중요도 분석
    importances = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
    top_indices = np.argsort(importances)[::-1]
    top_features = []
    print("\n" + "=" * 80)
    print("  [v3 상위 10대 최적 도메인 피처 중요도 분석 (LightGBM)]")
    print("=" * 80)
    for rank, idx in enumerate(top_indices[:10], 1):
        fname = feat_names[idx]
        imp = importances[idx]
        top_features.append({'rank': rank, 'feature': fname, 'importance': float(imp)})
        print(f"  {rank:2d}. {fname:<36} : {imp:.2f}")

    # 5. 가중치 번들 저장 (bl5_v3_bundle.pkl)
    bundle_data = {
        'version': 'bl5_v3',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': feat_names,
        'lgb_models': lgb_models,
        'cb_models': cb_models,
        'rf_models': rf_models,
        'et_models': et_models,
        'ensemble_weights': {'lgb': 0.35, 'cb': 0.35, 'rf': 0.15, 'et': 0.15},
        'best_3state': best_3state,
        'full_metrics': {
            'f1': float(full_f1),
            'precision': float(full_prec),
            'recall': float(full_rec),
            'accuracy': float(full_acc),
            'auc': float(full_auc)
        },
        'view_audit': view_audit_results,
        'top_features': top_features
    }

    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"\n[+] v3 모델 가중치 번들 저장 완료: {BUNDLE_SAVE_PATH} ({os.path.getsize(BUNDLE_SAVE_PATH)/1024/1024:.2f} MB)")

    # 6. 벤치마크 결과 JSON 저장
    with open(BENCHMARK_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(bundle_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[+] v3 벤치마크 결과 JSON 저장 완료: {BENCHMARK_SAVE_PATH}")
    print("=" * 80)

    return bundle_data


if __name__ == '__main__':
    train_and_eval_v3_pipeline()
