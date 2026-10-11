# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_bl5_v3_misclassifications.py
Purpose: bl5_v3 5-Fold OOF 예측에서 발생한 오분류(FP, FN) 샘플 208건 전수 조사
         - 피험자별, 시점별, Key 번호별, 오류 조건별 오분류 분포 통계
         - FP (정상인데 결함 판정) 원인 분석: 피처 분포, 정면 원근 왜곡, 체형 편차 등
         - FN (결함인데 정상 판정) 원인 분석: 미세 결함, 라벨 불일치, 최하사점 지연 등
         - 결과 JSON 저장: c5_misclassification_audit.json
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v3_28d_features.npz"
OUT_JSON_PATH = r"D:\src\models\second_models\babel_lunge\c5_misclassification_audit.json"

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def audit_misclassifications():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_v3 오분류 데이터 208건 전수 심층 감사")
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

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.035, num_leaves=22, max_depth=5,
            subsample=0.85, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_ratio, random_state=42 + fold, n_jobs=1, verbose=-1
        )
        clf_lgb.fit(X_train, y_train)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cb = CatBoostClassifier(
            iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4.0,
            scale_pos_weight=pos_ratio, random_seed=42 + fold, thread_count=1, verbose=0
        )
        clf_cb.fit(X_train, y_train)
        oof_cb[val_idx] = clf_cb.predict_proba(X_val)[:, 1]

        clf_rf = RandomForestClassifier(
            n_estimators=180, max_depth=7, min_samples_split=4,
            class_weight='balanced', random_state=42 + fold, n_jobs=1
        )
        clf_rf.fit(X_train, y_train)
        oof_rf[val_idx] = clf_rf.predict_proba(X_val)[:, 1]

        clf_et = ExtraTreesClassifier(
            n_estimators=180, max_depth=7, min_samples_split=4,
            class_weight='balanced', random_state=42 + fold, n_jobs=1
        )
        clf_et.fit(X_train, y_train)
        oof_et[val_idx] = clf_et.predict_proba(X_val)[:, 1]

    oof_ens = 0.35 * oof_lgb + 0.35 * oof_cb + 0.15 * oof_rf + 0.15 * oof_et

    # 최적 임계치 밴드 (Th_low=0.3820, Th_high=0.5280)
    th_low = 0.3820
    th_high = 0.5280

    conf_mask = (oof_ens < th_low) | (oof_ens >= th_high)
    pred_3s = np.full(N, -1, dtype=int)  # -1: 판단 보류 (Caution)
    pred_3s[oof_ens >= th_high] = 1
    pred_3s[oof_ens < th_low] = 0

    # 확정 구간 내 오분류 샘플 분리
    fp_indices = np.where((y == 0) & (pred_3s == 1))[0]  # False Positive (정상 -> 결함 오진)
    fn_indices = np.where((y == 1) & (pred_3s == 0))[0]  # False Negative (결함 -> 정상 오진)
    caution_indices = np.where(pred_3s == -1)[0]         # 판단 보류
    tp_indices = np.where((y == 1) & (pred_3s == 1))[0]
    tn_indices = np.where((y == 0) & (pred_3s == 0))[0]

    print(f"[*] 확정 판정 총 샘플 수: {len(y) - len(caution_indices)}개 / 1,081개 (커버리지: {np.mean(conf_mask)*100:.2f}%)")
    print(f"    - True Negative (TN) : {len(tn_indices)}개")
    print(f"    - True Positive (TP) : {len(tp_indices)}개")
    print(f"    - False Positive (FP): {len(fp_indices)}개 (정상 오탐)")
    print(f"    - False Negative (FN): {len(fn_indices)}개 (결함 미탐)")
    print(f"    - 판단 보류 (Caution): {len(caution_indices)}개 ({len(caution_indices)/N*100:.2f}%)")

    # 1. 피험자별 오분류 분포 조사
    subj_df = pd.DataFrame({
        'subj': subjects,
        'view': views,
        'key': keys,
        'y_true': y,
        'y_pred': pred_3s,
        'prob': oof_ens
    })

    print("\n" + "=" * 80)
    print("  [1] 피험자별 오분류(FP/FN) 집중도 분석")
    print("=" * 80)
    subj_audit = []
    for s in np.unique(subjects):
        s_mask = (subjects == s)
        s_fp = np.sum((subjects == s) & (y == 0) & (pred_3s == 1))
        s_fn = np.sum((subjects == s) & (y == 1) & (pred_3s == 0))
        s_tot = np.sum(s_mask)
        s_err = s_fp + s_fn
        s_err_rate = s_err / s_tot if s_tot > 0 else 0.0
        subj_audit.append({
            'subject': int(s),
            'total_samples': int(s_tot),
            'fp': int(s_fp),
            'fn': int(s_fn),
            'total_errors': int(s_err),
            'error_rate': float(s_err_rate)
        })
        print(f"  * 피험자 {s:2d}: 총 {s_tot:3d}개 중 오류 {s_err:2d}개 (FP: {s_fp:2d}, FN: {s_fn:2d}) -> 오분류율: {s_err_rate*100:5.2f}%")

    # 2. 시점별 오분류 분포 조사
    print("\n" + "=" * 80)
    print("  [2] 시점별 오분류(FP/FN) 분포 분석")
    print("=" * 80)
    view_audit = []
    for v_code, v_name in OFFICIAL_VIEW_MAP.items():
        v_mask = (views == v_code)
        v_fp = np.sum((views == v_code) & (y == 0) & (pred_3s == 1))
        v_fn = np.sum((views == v_code) & (y == 1) & (pred_3s == 0))
        v_tot = np.sum(v_mask)
        v_err = v_fp + v_fn
        v_err_rate = v_err / v_tot if v_tot > 0 else 0.0
        view_audit.append({
            'view_code': v_code,
            'view_name': v_name,
            'total_samples': int(v_tot),
            'fp': int(v_fp),
            'fn': int(v_fn),
            'total_errors': int(v_err),
            'error_rate': float(v_err_rate)
        })
        print(f"  * {v_name:<24} ({v_code}): 총 {v_tot:3d}개 중 오류 {v_err:2d}개 (FP: {v_fp:2d}, FN: {v_fn:2d}) -> 오분류율: {v_err_rate*100:5.2f}%")

    # 3. Key(동작 변형)별 오분류 분포
    print("\n" + "=" * 80)
    print("  [3] Key별 오분류 분포 분석")
    print("=" * 80)
    key_audit = []
    for k in sorted(np.unique(keys)):
        k_mask = (keys == k)
        k_fp = np.sum((keys == k) & (y == 0) & (pred_3s == 1))
        k_fn = np.sum((keys == k) & (y == 1) & (pred_3s == 0))
        k_tot = np.sum(k_mask)
        k_err = k_fp + k_fn
        k_audit_item = {
            'key': str(k),
            'total_samples': int(k_tot),
            'fp': int(k_fp),
            'fn': int(k_fn),
            'total_errors': int(k_err)
        }
        key_audit.append(k_audit_item)
        if k_err > 5:
            print(f"  * Key {k}: 총 {k_tot:3d}개 중 오류 {k_err:2d}개 (FP: {k_fp:2d}, FN: {k_fn:2d})")

    # 4. FP vs TN 피처 통계 대조 (정상인데 왜 결함으로 오탐되었는가?)
    print("\n" + "=" * 80)
    print("  [4] False Positive (정상 오탐 127건) 원인 규명: TN(정상 정답)과의 피처 평균 대조")
    print("=" * 80)
    fp_X = X[fp_indices]
    tn_X = X[tn_indices]
    fn_X = X[fn_indices]
    tp_X = X[tp_indices]

    fp_vs_tn_diffs = []
    for d in range(X.shape[1]):
        tn_m = float(np.mean(tn_X[:, d]))
        fp_m = float(np.mean(fp_X[:, d]))
        diff = fp_m - tn_m
        norm_diff = diff / (float(np.std(tn_X[:, d])) + 1e-6)
        fp_vs_tn_diffs.append({
            'idx': d,
            'name': feat_names[d],
            'tn_mean': tn_m,
            'fp_mean': fp_m,
            'diff': diff,
            'norm_diff': float(norm_diff)
        })

    fp_vs_tn_diffs.sort(key=lambda x: abs(x['norm_diff']), reverse=True)
    print("  [*] FP에서 TN 대비 가장 크게 왜곡된 피처 Top 8:")
    for item in fp_vs_tn_diffs[:8]:
        print(f"      - {item['name']:<36}: TN={item['tn_mean']:7.2f}, FP={item['fp_mean']:7.2f} (z-편차: {item['norm_diff']:+5.2f}σ)")

    # 5. FN vs TP 피처 통계 대조 (결함인데 왜 정상으로 놓쳤는가?)
    print("\n" + "=" * 80)
    print("  [5] False Negative (결함 미탐 81건) 원인 규명: TP(결함 정답)과의 피처 평균 대조")
    print("=" * 80)
    fn_vs_tp_diffs = []
    for d in range(X.shape[1]):
        tp_m = float(np.mean(tp_X[:, d]))
        fn_m = float(np.mean(fn_X[:, d]))
        diff = fn_m - tp_m
        norm_diff = diff / (float(np.std(tp_X[:, d])) + 1e-6)
        fn_vs_tp_diffs.append({
            'idx': d,
            'name': feat_names[d],
            'tp_mean': tp_m,
            'fn_mean': fn_m,
            'diff': diff,
            'norm_diff': float(norm_diff)
        })

    fn_vs_tp_diffs.sort(key=lambda x: abs(x['norm_diff']), reverse=True)
    print("  [*] FN에서 TP 대비 가장 크게 왜곡된 피처 Top 8:")
    for item in fn_vs_tp_diffs[:8]:
        print(f"      - {item['name']:<36}: TP={item['tp_mean']:7.2f}, FN={item['fn_mean']:7.2f} (z-편차: {item['norm_diff']:+5.2f}σ)")

    # 대표적인 오분류 파일명 목록 (Top 10 FP / FN)
    top_fp_samples = [
        {'filename': filenames[i], 'subj': int(subjects[i]), 'view': views[i], 'prob': float(oof_ens[i])}
        for i in fp_indices[:15]
    ]
    top_fn_samples = [
        {'filename': filenames[i], 'subj': int(subjects[i]), 'view': views[i], 'prob': float(oof_ens[i])}
        for i in fn_indices[:15]
    ]

    audit_result = {
        'total_samples': N,
        'tn': len(tn_indices),
        'tp': len(tp_indices),
        'fp': len(fp_indices),
        'fn': len(fn_indices),
        'caution': len(caution_indices),
        'subj_audit': subj_audit,
        'view_audit': view_audit,
        'key_audit': key_audit,
        'fp_vs_tn_top_diffs': fp_vs_tn_diffs[:10],
        'fn_vs_tp_top_diffs': fn_vs_tp_diffs[:10],
        'top_fp_samples': top_fp_samples,
        'top_fn_samples': top_fn_samples
    }

    with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit_result, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 오분류 감사 결과 저장 완료: {OUT_JSON_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    audit_misclassifications()
