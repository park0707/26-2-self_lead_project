# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: analyze_c3_v3_misclassifications.py
Purpose: v3 챔피언(11D CatBoost)의 1,085개 전수 OOF 예측에서 발생한 오분류(FP, FN) 및
         판단 보류(Undetermined) 샘플 전수 추출 및 통계적/생체역학적 원인 분석
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v3 import FEATURE_NAMES_11D

CACHE_11D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_11d.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
JSON_OUT_PATH = os.path.join(RESULTS_DIR, "bl3_v3_misclassification_analysis.json")

def main():
    print("=" * 80)
    print(" [v3 챔피언] 1,085개 전수 OOF 오분류(FP, FN) 및 보류 데이터 정밀 추출 및 분석")
    print("=" * 80)
    
    data = np.load(CACHE_11D_PATH, allow_pickle=True)
    X = data['X'] # (1085, 11)
    y = data['y'] # (1085,)
    groups = data['groups'] # (1085,)
    views = data['views']
    fnames = data['filenames']
    
    n_samples = len(y)
    scale_pos = np.sum(y == 0) / np.sum(y == 1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(n_samples, dtype=np.float32)
    
    for tr_idx, val_idx in cv.split(X, y, groups):
        cb = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb.fit(X[tr_idx], y[tr_idx])
        oof_probs[val_idx] = cb.predict_proba(X[val_idx])[:, 1]
        
    # v3 최적 임계값 적용: tau_low = 0.35, tau_high = 0.63
    tau_low = 0.35
    tau_high = 0.63
    
    preds_3state = np.full(n_samples, -1, dtype=np.int32)
    preds_3state[oof_probs < tau_low] = 0
    preds_3state[oof_probs > tau_high] = 1
    
    # 카테고리 분류
    # TP: y == 1 & pred == 1
    # TN: y == 0 & pred == 0
    # FP: y == 0 & pred == 1 (정상 -> 결함 오탐)
    # FN: y == 1 & pred == 0 (결함 -> 정상 미탐)
    # UNDET: pred == -1 (판단 보류)
    
    tp_mask = (y == 1) & (preds_3state == 1)
    tn_mask = (y == 0) & (preds_3state == 0)
    fp_mask = (y == 0) & (preds_3state == 1)
    fn_mask = (y == 1) & (preds_3state == 0)
    undet_mask = (preds_3state == -1)
    
    n_tp = int(np.sum(tp_mask))
    n_tn = int(np.sum(tn_mask))
    n_fp = int(np.sum(fp_mask))
    n_fn = int(np.sum(fn_mask))
    n_undet = int(np.sum(undet_mask))
    
    print(f"\n[*] 전체 1,085개 표본 분류 결과 집계:")
    print(f"  - TN (정상 적중): {n_tn}건 ({n_tn/n_samples*100:.1f}%)")
    print(f"  - TP (결함 적중): {n_tp}건 ({n_tp/n_samples*100:.1f}%)")
    print(f"  - FP (정상 -> 결함 오탐): {n_fp}건 ({n_fp/n_samples*100:.1f}%)")
    print(f"  - FN (결함 -> 정상 미탐): {n_fn}건 ({n_fn/n_samples*100:.1f}%)")
    print(f"  - UNDETERMINED (판단 보류): {n_undet}건 ({n_undet/n_samples*100:.1f}%)")
    print(f"  - 총 확정 판정(Determined): {n_tp + n_tn + n_fp + n_fn}건")
    print(f"  - 확정 판정 내 오류 수: {n_fp + n_fn}건 (오류율 {(n_fp + n_fn)/(n_tp + n_tn + n_fp + n_fn)*100:.2f}%)")
    
    df_all = pd.DataFrame(X, columns=FEATURE_NAMES_11D)
    df_all['y_true'] = y
    df_all['prob_defect'] = oof_probs
    df_all['pred_3state'] = preds_3state
    df_all['view'] = views
    df_all['subject'] = groups
    df_all['filename'] = fnames
    
    # 1. FP 샘플 분석 (정상인데 결함으로 오탐)
    df_fp = df_all[fp_mask].copy()
    print("\n" + "=" * 80)
    print(f" [FP 분석] 정상인데 결함으로 오탐된 샘플 ({len(df_fp)}건):")
    print("=" * 80)
    print(f"  - 평균 뒤다리 무릎 각도 (rear_knee_angle_bottom): {df_fp['rear_knee_angle_bottom'].mean():.1f}° (정상 기준 90° 대비)")
    print(f"  - 평균 90도 오차 (rear_knee_target_dev_bottom): {df_fp['rear_knee_target_dev_bottom'].mean():.1f}°")
    print(f"  - 평균 무릎 안정성 떨림 (rear_knee_stability_std): {df_fp['rear_knee_stability_std'].mean():.2f}° (전체 정상 평균: {df_all[y==0]['rear_knee_stability_std'].mean():.2f}°)")
    print(f"  - 평균 예측 결함 확률: {df_fp['prob_defect'].mean():.3f}")
    print(f"  - 피험자별 분포: {dict(df_fp['subject'].value_counts())}")
    
    # 2. FN 샘플 분석 (결함인데 정상으로 미탐)
    df_fn = df_all[fn_mask].copy()
    print("\n" + "=" * 80)
    print(f" [FN 분석] 결함인데 정상으로 미탐된 샘플 ({len(df_fn)}건):")
    print("=" * 80)
    print(f"  - 평균 뒤다리 무릎 각도 (rear_knee_angle_bottom): {df_fn['rear_knee_angle_bottom'].mean():.1f}°")
    print(f"  - 평균 90도 오차 (rear_knee_target_dev_bottom): {df_fn['rear_knee_target_dev_bottom'].mean():.1f}°")
    print(f"  - 평균 앞/뒤 무릎 각도 비율 (lead_rear_knee_ratio): {df_fn['lead_rear_knee_ratio'].mean():.2f}")
    print(f"  - 평균 예측 결함 확률: {df_fn['prob_defect'].mean():.3f}")
    print(f"  - 피험자별 분포: {dict(df_fn['subject'].value_counts())}")
    
    # 3. UNDET 샘플 분석 (판단 보류)
    df_undet = df_all[undet_mask].copy()
    print("\n" + "=" * 80)
    print(f" [UNDET 분석] 판단 보류 샘플 ({len(df_undet)}건):")
    print("=" * 80)
    print(f"  - 실제 라벨 구성: 정상 {int(np.sum(df_undet['y_true']==0))}건 vs 결함 {int(np.sum(df_undet['y_true']==1))}건")
    print(f"  - 평균 예측 결함 확률: {df_undet['prob_defect'].mean():.3f} (확률 범위: {df_undet['prob_defect'].min():.3f} ~ {df_undet['prob_defect'].max():.3f})")
    print(f"  - 평균 뒤다리 무릎 각도: {df_undet['rear_knee_angle_bottom'].mean():.1f}°")
    print(f"  - 평균 90도 오차: {df_undet['rear_knee_target_dev_bottom'].mean():.1f}°")
    
    # 대표적 FP 샘플 TOP 5 (가장 결함 확률이 높았던 정상)
    top_fp = df_fp.sort_values(by='prob_defect', ascending=False).head(5)
    print("\n[*] 대표적 FP 샘플 TOP 5 (정상 라벨이나 극단적 결함 판정):")
    for _, r in top_fp.iterrows():
        print(f"  - 파일: {r['filename']} | P(결함)={r['prob_defect']:.3f} | 뒤무릎각={r['rear_knee_angle_bottom']:.1f}° | 편차={r['rear_knee_target_dev_bottom']:.1f}° | 떨림={r['rear_knee_stability_std']:.2f}° | 대퇴피치={r['rear_femur_vertical_pitch']:.1f}°")
        
    # 대표적 FN 샘플 TOP 5 (가장 정상 확률이 높았던 결함)
    top_fn = df_fn.sort_values(by='prob_defect', ascending=True).head(5)
    print("\n[*] 대표적 FN 샘플 TOP 5 (결함 라벨이나 극단적 정상 판정):")
    for _, r in top_fn.iterrows():
        print(f"  - 파일: {r['filename']} | P(결함)={r['prob_defect']:.3f} | 뒤무릎각={r['rear_knee_angle_bottom']:.1f}° | 편차={r['rear_knee_target_dev_bottom']:.1f}° | 앞/뒤비율={r['lead_rear_knee_ratio']:.2f} | 대퇴피치={r['rear_femur_vertical_pitch']:.1f}°")
        
    # 결과 JSON 저장
    analysis_payload = {
        'total_samples': n_samples,
        'tn': n_tn,
        'tp': n_tp,
        'fp': n_fp,
        'fn': n_fn,
        'undetermined': n_undet,
        'determined_accuracy': (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn) * 100.0,
        'recall': n_tp / (n_tp + n_fn) * 100.0,
        'precision': n_tp / (n_tp + n_fp) * 100.0,
        'fp_stats': {
            'mean_rear_knee_angle': float(df_fp['rear_knee_angle_bottom'].mean()),
            'mean_target_dev': float(df_fp['rear_knee_target_dev_bottom'].mean()),
            'mean_stability': float(df_fp['rear_knee_stability_std'].mean()),
            'mean_prob': float(df_fp['prob_defect'].mean()),
            'subject_dist': {str(k): int(v) for k, v in df_fp['subject'].value_counts().items()}
        },
        'fn_stats': {
            'mean_rear_knee_angle': float(df_fn['rear_knee_angle_bottom'].mean()),
            'mean_target_dev': float(df_fn['rear_knee_target_dev_bottom'].mean()),
            'mean_lead_rear_ratio': float(df_fn['lead_rear_knee_ratio'].mean()),
            'mean_prob': float(df_fn['prob_defect'].mean()),
            'subject_dist': {str(k): int(v) for k, v in df_fn['subject'].value_counts().items()}
        },
        'undet_stats': {
            'normal_count': int(np.sum(df_undet['y_true']==0)),
            'defect_count': int(np.sum(df_undet['y_true']==1)),
            'mean_prob': float(df_undet['prob_defect'].mean()),
            'mean_target_dev': float(df_undet['rear_knee_target_dev_bottom'].mean())
        },
        'top_fp_samples': [
            {'filename': str(r['filename']), 'prob': float(r['prob_defect']), 'angle': float(r['rear_knee_angle_bottom']), 'dev': float(r['rear_knee_target_dev_bottom']), 'stability': float(r['rear_knee_stability_std']), 'sub': int(r['subject'])}
            for _, r in top_fp.iterrows()
        ],
        'top_fn_samples': [
            {'filename': str(r['filename']), 'prob': float(r['prob_defect']), 'angle': float(r['rear_knee_angle_bottom']), 'dev': float(r['rear_knee_target_dev_bottom']), 'ratio': float(r['lead_rear_knee_ratio']), 'sub': int(r['subject'])}
            for _, r in top_fn.iterrows()
        ]
    }
    
    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(analysis_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved detailed misclassification analysis to: {JSON_OUT_PATH}")

if __name__ == "__main__":
    main()
