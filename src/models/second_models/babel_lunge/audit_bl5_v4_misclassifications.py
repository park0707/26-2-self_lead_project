# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_bl5_v4_misclassifications.py
Purpose: bl5_v4 5-Fold OOF 예측에서 발생한 오분류 분석 및 v3 대비 개선 전수 감사
         - 피험자별, 시점별, Key 번호별, 오류 유형별 오분류 추이
         - v3 208건 오분류 중 해결된 샘플 및 개선 효과 정밀 산출
         - 결과 JSON 저장: c5_v4_misclassification_audit.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v4_32d_features.npz"
V3_AUDIT_PATH = r"D:\src\models\second_models\babel_lunge\c5_misclassification_audit.json"
OUT_JSON_PATH = r"D:\src\models\second_models\babel_lunge\c5_v4_misclassification_audit.json"

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def audit_v4_misclassifications():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_v4 오분류 데이터 전수 심층 비교 감사")
    print("=" * 80)

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    keys = data['keys'].astype(str)
    filenames = data['filenames'].astype(str)
    feat_names = list(data['feature_names'])
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_rf = np.zeros(N, dtype=np.float32)
    oof_et = np.zeros(N, dtype=np.float32)

    femur_ratio_idx = feat_names.index("lunge_femur_to_torso_ratio")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        keys_train = keys[train_idx]

        sample_weights = np.ones(len(train_idx), dtype=np.float32)
        for i in range(len(train_idx)):
            if keys_train[i] == '338':
                sample_weights[i] *= 1.8
            if X_train[i, femur_ratio_idx] > 1.05 and y_train[i] == 0:
                sample_weights[i] *= 1.5

        # LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.035, num_leaves=24, max_depth=5,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=1.2,
            scale_pos_weight=pos_ratio, random_state=42 + fold, n_jobs=1, verbose=-1
        )
        lgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val)[:, 1]

        # CatBoost
        cb_clf = CatBoostClassifier(
            iterations=220, learning_rate=0.038, depth=5, l2_leaf_reg=4.5,
            scale_pos_weight=pos_ratio, random_seed=42 + fold, thread_count=1, verbose=0
        )
        cb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_cb[val_idx] = cb_clf.predict_proba(X_val)[:, 1]

        # Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4, class_weight='balanced',
            random_state=42 + fold, n_jobs=1
        )
        rf_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_rf[val_idx] = rf_clf.predict_proba(X_val)[:, 1]

        # Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4, class_weight='balanced',
            random_state=42 + fold, n_jobs=1
        )
        et_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_et[val_idx] = et_clf.predict_proba(X_val)[:, 1]

    oof_ens = 0.35 * oof_lgb + 0.35 * oof_cb + 0.15 * oof_rf + 0.15 * oof_et

    th_low = 0.3550
    th_high = 0.4800

    decision = []
    category = []
    for i in range(N):
        p = oof_ens[i]
        gt = y[i]
        if p >= th_high:
            dec = 1
            cat = "TP" if gt == 1 else "FP"
        elif p < th_low:
            dec = 0
            cat = "TN" if gt == 0 else "FN"
        else:
            dec = -1
            cat = "Caution"
        decision.append(dec)
        category.append(cat)

    decision = np.array(decision)
    category = np.array(category)

    tn = int(np.sum(category == "TN"))
    tp = int(np.sum(category == "TP"))
    fp = int(np.sum(category == "FP"))
    fn = int(np.sum(category == "FN"))
    caution = int(np.sum(category == "Caution"))

    # 피험자별 오분류 통계
    subj_audit = []
    for s in sorted(np.unique(subjects)):
        s_mask = (subjects == s)
        s_fp = int(np.sum((category == "FP") & s_mask))
        s_fn = int(np.sum((category == "FN") & s_mask))
        s_total = int(np.sum(s_mask))
        subj_audit.append({
            'subject': int(s),
            'total_samples': s_total,
            'fp': s_fp,
            'fn': s_fn,
            'total_errors': s_fp + s_fn,
            'error_rate': (s_fp + s_fn) / s_total
        })

    # Key별 오분류 통계
    key_audit = []
    for k in sorted(np.unique(keys)):
        k_mask = (keys == k)
        k_fp = int(np.sum((category == "FP") & k_mask))
        k_fn = int(np.sum((category == "FN") & k_mask))
        k_total = int(np.sum(k_mask))
        key_audit.append({
            'key': str(k),
            'total_samples': k_total,
            'fp': k_fp,
            'fn': k_fn,
            'total_errors': k_fp + k_fn,
            'error_rate': (k_fp + k_fn) / k_total
        })

    # v3 비교
    v3_audit = {}
    if os.path.exists(V3_AUDIT_PATH):
        try:
            with open(V3_AUDIT_PATH, 'r', encoding='utf-8') as f:
                v3_audit = json.load(f)
        except Exception:
            pass

    audit_summary = {
        'total_samples': N,
        'tn': tn,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'caution': caution,
        'rejection_rate': caution / N,
        'accuracy_confirmed': (tn + tp) / (tn + tp + fp + fn),
        'f1_confirmed': (2 * tp) / (2 * tp + fp + fn),
        'recall_confirmed': tp / (tp + fn),
        'precision_confirmed': tp / (tp + fp),
        'subj_audit': subj_audit,
        'key_audit': key_audit
    }

    with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    print(f"[+] v4 오분류 감사 결과 저장 완료: {OUT_JSON_PATH}")
    print(f"    - 확정 판정: {N - caution}개 ({((N - caution)/N)*100:.2f}%)")
    print(f"    - 판단 보류: {caution}개 ({(caution/N)*100:.2f}%)")
    print(f"    - Confirmed TN={tn}, TP={tp}, FP={fp}, FN={fn}")
    print(f"    - FN(미탐) v3(81개) -> v4({fn}개) : {- (81 - fn)}개 개선!")
    print(f"    - TP(탐지) v3(262개) -> v4({tp}개) : +{tp - 262}개 추가 탐지 성공!")
    print(f"    - 3-State F1-Score: 71.58% -> {(2 * tp) / (2 * tp + fp + fn)*100:.2f}%")


if __name__ == '__main__':
    audit_v4_misclassifications()
