# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: recalibrate_bl5_v3_rejection_10pct.py
Purpose: bl5_v3 모델의 판단 보류율을 최대 10% (커버리지 90% 이상)로 엄격히 제한한 상태에서
         전체 3-State F1-Score를 극대화하는 최적 임계치 재보정 및 5대 시점 감사
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pickle
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v3_28d_features.npz"
BUNDLE_SAVE_PATH = r"D:\src\models\second_models\babel_lunge\weights\bl5_v3_bundle.pkl"
BENCHMARK_SAVE_PATH = r"D:\src\models\second_models\babel_lunge\bl5_v3_benchmark_results.json"

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def recalibrate():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_v3 판단 보류율 최대 10% (커버리지 >= 90%) 재조정 파이프라인")
    print("=" * 80)

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    feat_names = list(data['feature_names'])
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    oof_preds_lgb = np.zeros(N, dtype=np.float32)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_rf = np.zeros(N, dtype=np.float32)
    oof_preds_et = np.zeros(N, dtype=np.float32)

    lgb_models = []
    cb_models = []
    rf_models = []
    et_models = []

    print("[*] 5-Fold OOF 예측 획득 중...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
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
        oof_preds_lgb[val_idx] = lgb_clf.predict_proba(X_val)[:, 1]
        lgb_models.append(lgb_clf)

        # 2. CatBoost
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
        oof_preds_cb[val_idx] = cb_clf.predict_proba(X_val)[:, 1]
        cb_models.append(cb_clf)

        # 3. Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        rf_clf.fit(X_train, y_train)
        oof_preds_rf[val_idx] = rf_clf.predict_proba(X_val)[:, 1]
        rf_models.append(rf_clf)

        # 4. Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        et_clf.fit(X_train, y_train)
        oof_preds_et[val_idx] = et_clf.predict_proba(X_val)[:, 1]
        et_models.append(et_clf)

    oof_ens = (
        0.35 * oof_preds_lgb +
        0.35 * oof_preds_cb +
        0.15 * oof_preds_rf +
        0.15 * oof_preds_et
    )

    # 100% Full-Coverage 지표
    full_preds = (oof_ens >= 0.5).astype(int)
    full_f1 = f1_score(y, full_preds, zero_division=0)
    full_prec = precision_score(y, full_preds, zero_division=0)
    full_rec = recall_score(y, full_preds, zero_division=0)
    full_acc = accuracy_score(y, full_preds)
    full_auc = roc_auc_score(y, oof_ens)

    print("\n" + "=" * 80)
    print("  [100% Full-Coverage (미기각 베이스라인)]")
    print(f"  - Full F1-Score : {full_f1*100:.2f}%")
    print(f"  - Full Precision: {full_prec*100:.2f}%")
    print(f"  - Full Recall   : {full_rec*100:.2f}%")
    print(f"  - Full Accuracy : {full_acc*100:.2f}%")
    print(f"  - ROC-AUC       : {full_auc:.4f}")

    # 판단 보류율 최대 10% (커버리지 >= 90.0%) 제약 하 최적 탐색
    print("\n[*] 판단 보류율 <= 10.0% (Coverage >= 90.0%) 제약 하 최적 3-State 그리드 서치...")
    best_3state = {
        'f1': 0.0,
        'th_low': 0.45,
        'th_high': 0.55,
        'coverage': 0.0,
        'rejection_rate': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'cm': None
    }

    # 세밀한 임계치 탐색
    for th_l in np.linspace(0.35, 0.49, 141):
        for th_h in np.linspace(0.51, 0.65, 141):
            if th_h <= th_l:
                continue

            high_conf_mask = (oof_ens < th_l) | (oof_ens >= th_h)
            cov = float(np.mean(high_conf_mask))
            rej = 1.0 - cov

            # 핵심 제약: 판단 보류율 최대 10.0% (즉 cov >= 0.90)
            if rej > 0.100001 or cov < 0.899999:
                continue

            y_conf = y[high_conf_mask]
            pred_conf = (oof_ens[high_conf_mask] >= th_h).astype(int)

            score_f1 = f1_score(y_conf, pred_conf, zero_division=0)
            score_rec = recall_score(y_conf, pred_conf, zero_division=0)
            score_prec = precision_score(y_conf, pred_conf, zero_division=0)
            score_acc = accuracy_score(y_conf, pred_conf)

            # F1 최적화
            if score_f1 > best_3state['f1']:
                best_3state = {
                    'f1': float(score_f1),
                    'th_low': float(th_l),
                    'th_high': float(th_h),
                    'coverage': float(cov),
                    'rejection_rate': float(rej),
                    'precision': float(score_prec),
                    'recall': float(score_rec),
                    'accuracy': float(score_acc),
                    'cm': confusion_matrix(y_conf, pred_conf).tolist()
                }

    print("\n" + "=" * 80)
    print("  [재조정된 최종 bl5_v3 3-State 성과 보고 (판단 보류율 <= 10%)]")
    print("=" * 80)
    print(f"  ★ 최종 전체 3-State F1-Score : {best_3state['f1']*100:.2f}%")
    print(f"  ★ 판단 보류율 (Rejection Rate) : {best_3state['rejection_rate']*100:.2f}% (최대 10% 이하 완벽 준수)")
    print(f"  ★ 신뢰도 커버리지 (Coverage)  : {best_3state['coverage']*100:.2f}%")
    print(f"    - 3-State Precision         : {best_3state['precision']*100:.2f}%")
    print(f"    - 3-State Recall            : {best_3state['recall']*100:.2f}%")
    print(f"    - 3-State Accuracy          : {best_3state['accuracy']*100:.2f}%")
    print(f"    - 최적 임계치 밴드          : Th_Low = {best_3state['th_low']:.4f}, Th_High = {best_3state['th_high']:.4f}")
    print(f"    - 3-State 혼동행렬 (TN, FP / FN, TP): {best_3state['cm']}")

    # 5대 공식 시점별 전수 감사
    print("\n" + "=" * 80)
    print("  [AGENTS.md 규정: 5대 공식 시점별 전수 감사 (보류율 <= 10% 기준)]")
    print("=" * 80)

    view_audit_results = {}
    th_l = best_3state['th_low']
    th_h = best_3state['th_high']

    for v_code, v_name in OFFICIAL_VIEW_MAP.items():
        v_mask = (views == v_code)
        if not np.any(v_mask):
            continue

        y_v = y[v_mask]
        probs_v = oof_ens[v_mask]

        full_preds_v = (probs_v >= 0.5).astype(int)
        v_full_f1 = f1_score(y_v, full_preds_v, zero_division=0)
        v_full_rec = recall_score(y_v, full_preds_v, zero_division=0)
        v_full_prec = precision_score(y_v, full_preds_v, zero_division=0)
        v_full_acc = accuracy_score(y_v, full_preds_v)

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

    # 상위 10대 피처 중요도
    importances = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
    top_indices = np.argsort(importances)[::-1]
    top_features = []
    for rank, idx in enumerate(top_indices[:10], 1):
        fname = feat_names[idx]
        imp = importances[idx]
        top_features.append({'rank': rank, 'feature': fname, 'importance': float(imp)})

    # 가중치 번들 업데이트 저장
    bundle_data = {
        'version': 'bl5_v3_final_10pct_rejection',
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
    print(f"\n[+] 최종 v3 모델 가중치 번들 저장 완료: {BUNDLE_SAVE_PATH}")

    with open(BENCHMARK_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(bundle_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[+] 최종 v3 벤치마크 결과 JSON 저장 완료: {BENCHMARK_SAVE_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    recalibrate()
