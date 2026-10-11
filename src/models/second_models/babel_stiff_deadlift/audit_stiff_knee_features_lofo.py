# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_stiff_knee_features_lofo.py
Purpose:
  1. 14대 생체역학 피처 각각의 단독 변별력 전수 감사 (정상 vs 결함 통계, Cohen's d, 단일 피처 ROC-AUC)
  2. Stream 1 (TCN) 단독 성능 vs Stream 2 (LightGBM) 단독 성능 비교 감사
  3. LOFO (Leave-One-Feature-Out) 전수 감사: 14개 피처를 하나씩 제거하면서 Macro F1, AUC, Clean F1 변화 측정
  4. 성능을 저해(Negative Impact)하거나 기여하지 않는 불량/노이즈 피처 전수 식별
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_deadlift_knee_ambiguity_levels.npz"

def audit_features():
    print("=" * 110)
    print("  [vidio-health-trainer] 스티프 데드리프트 14대 피처 전수 변별력 및 LOFO 감사 파이프라인")
    print("=" * 110)

    # 1. 데이터 로드
    data = np.load(CACHE_PATH, allow_pickle=True)
    labels = data['labels'] # 0: 정상, 1: 결함
    l4_mask = data['l4_mask']
    feats_14d = data['feats_14d']
    series_12ch = data['series_12ch']
    feature_names = list(data['feature_names'])
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']

    # Level 4 제외한 Level 1~3 유효 데이터
    valid_mask = ~l4_mask
    valid_idx = np.where(valid_mask)[0]
    X_val = feats_14d[valid_idx]
    y_val = labels[valid_idx]
    subs_val = subject_ids[valid_idx]
    views_val = view_ids[valid_idx]

    N = len(y_val)
    n_clean = np.sum(y_val == 0)
    n_defect = np.sum(y_val == 1)
    print(f"[*] 감사 대상 데이터셋: Level 1~3 유효 샘플 {N}개 (정상: {n_clean}개, 결함: {n_defect}개)")

    # ==============================================================================
    # 감사 1: 14대 피처 개별 통계 및 단독 ROC-AUC 전수 검사
    # ==============================================================================
    print("\n" + "-" * 110)
    print(f"  {'피처명':25s} | {'정상 평균 (SD)':20s} | {'결함 평균 (SD)':20s} | {'Cohen d':8s} | {'단독 AUC':8s} | {'판정'}")
    print("-" * 110)

    stats_summary = []

    for j, fn in enumerate(feature_names):
        col = X_val[:, j]
        col_clean = col[y_val == 0]
        col_defect = col[y_val == 1]

        m_c, s_c = np.mean(col_clean), np.std(col_clean)
        m_d, s_d = np.mean(col_defect), np.std(col_defect)

        # Cohen's d (효과 크기)
        pooled_sd = np.sqrt((s_c**2 + s_d**2) / 2.0) + 1e-7
        cohen_d = (m_d - m_c) / pooled_sd

        # 단독 ROC-AUC (피처 자체를 판별기로 사용했을 때)
        try:
            auc = roc_auc_score(y_val, col)
            # 만약 역상관(auc < 0.5)이면 1-auc로 방향성 보정 AUC 계산
            effective_auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
            effective_auc = 0.5

        # 판정
        if effective_auc < 0.52 and abs(cohen_d) < 0.10:
            verdict = "❌ 무효/노이즈 (변별력 전무)"
        elif auc < 0.50:
            verdict = f"⚠️ 역상관 피처 (AUC={auc:.3f})"
        elif effective_auc >= 0.60:
            verdict = "★ 우수 기여 피처"
        else:
            verdict = "▲ 미약 기여 피처"

        stats_summary.append({
            'feature': fn,
            'mean_clean': m_c, 'sd_clean': s_c,
            'mean_defect': m_d, 'sd_defect': s_d,
            'cohen_d': cohen_d,
            'raw_auc': auc,
            'effective_auc': effective_auc,
            'verdict': verdict
        })

        str_c = f"{m_c:7.2f} (±{s_c:5.2f})"
        str_d = f"{m_d:7.2f} (±{s_d:5.2f})"
        print(f"  {fn:25s} | {str_c:20s} | {str_d:20s} | {cohen_d:8.3f} | {auc:8.3f} | {verdict}")

    # ==============================================================================
    # 감사 2: LightGBM 베이스라인 (14D 전체) 5-Fold 교차검증 성능
    # ==============================================================================
    print("\n" + "-" * 110)
    print("  [감사 2: LightGBM 5-Fold 베이스라인 성능 실측 (14D 전체 투입 시)]")
    print("-" * 110)

    def eval_lgb(features_matrix, y_true, groups):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y_true), dtype=np.float32)

        for tr_i, val_i in sgkf.split(features_matrix, y_true, groups):
            pos_w = float(np.sum(y_true[tr_i] == 0)) / max(1, np.sum(y_true[tr_i] == 1))
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.03,
                scale_pos_weight=pos_w * 0.9,
                min_child_samples=15,
                reg_alpha=0.1,
                reg_lambda=0.5,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )
            clf.fit(features_matrix[tr_i], y_true[tr_i])
            oof_preds[val_i] = clf.predict_proba(features_matrix[val_i])[:, 1]

        binary_preds = (oof_preds >= 0.50).astype(int)
        macro_f1 = f1_score(y_true, binary_preds, average='macro')
        clean_f1 = f1_score(y_true, binary_preds, pos_label=0)
        defect_f1 = f1_score(y_true, binary_preds, pos_label=1)
        auc = roc_auc_score(y_true, oof_preds)
        acc = accuracy_score(y_true, binary_preds)
        tn, fp, fn, tp = confusion_matrix(y_true, binary_preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'macro_f1': macro_f1,
            'clean_f1': clean_f1,
            'defect_f1': defect_f1,
            'auc': auc,
            'acc': acc,
            'spec': spec,
            'oof_preds': oof_preds
        }

    base_res = eval_lgb(X_val, y_val, subs_val)
    print(f"  * 14D 전체 베이스라인: Macro F1 = {base_res['macro_f1']*100:.2f}% | AUC = {base_res['auc']*100:.2f}% | Clean F1 = {base_res['clean_f1']*100:.2f}% | Defect F1 = {base_res['defect_f1']*100:.2f}% | Specificity = {base_res['spec']*100:.2f}%")

    # ==============================================================================
    # 감사 3: LOFO (Leave-One-Feature-Out) 전수 감사
    # ==============================================================================
    print("\n" + "-" * 110)
    print(f"  {'제외 피처':25s} | {'Macro F1':9s} | {'Δ F1':8s} | {'AUC':8s} | {'Δ AUC':8s} | {'Clean F1':9s} | {'영향도 판정'}")
    print("-" * 110)

    lofo_results = []

    for j, fn in enumerate(feature_names):
        # j번째 피처 제외한 서브셋
        subset_idx = [k for k in range(len(feature_names)) if k != j]
        X_sub = X_val[:, subset_idx]

        sub_res = eval_lgb(X_sub, y_val, subs_val)

        delta_f1 = (sub_res['macro_f1'] - base_res['macro_f1']) * 100.0
        delta_auc = (sub_res['auc'] - base_res['auc']) * 100.0

        # 판정: 해당 피처를 뺐더니 성능이 올랐다면 -> 방해 피처 (Negative Feature)!
        if delta_f1 > 0.5:
            impact = "⛔ 성능 방해 피처 (제외 시 F1 대폭 상승!)"
        elif delta_f1 > 0.0:
            impact = "⚠️ 미약 방해/노이즈 (제외 시 소폭 상승)"
        elif delta_f1 < -1.0:
            impact = "★ 필수 핵심 피처 (제외 시 F1 급락)"
        elif delta_f1 < 0.0:
            impact = "✔ 유효 기여 피처 (제외 시 F1 소폭 하락)"
        else:
            impact = "― 영향 없음 (Neutral)"

        lofo_results.append({
            'excluded_feature': fn,
            'macro_f1': sub_res['macro_f1'],
            'delta_f1': delta_f1,
            'auc': sub_res['auc'],
            'delta_auc': delta_auc,
            'clean_f1': sub_res['clean_f1'],
            'impact': impact
        })

        print(f"  {fn:25s} | {sub_res['macro_f1']*100:8.2f}% | {delta_f1:+7.2f}%p | {sub_res['auc']*100:7.2f}% | {delta_auc:+7.2f}%p | {sub_res['clean_f1']*100:8.2f}% | {impact}")

    # ==============================================================================
    # 감사 4: 방해 피처 전수 제거 후 정예 피처셋(Curated Feature Set) 재평가
    # ==============================================================================
    print("\n" + "-" * 110)
    print("  [감사 4: 방해 피처 제거 후 정예 피처셋 최적화 검증]")
    print("-" * 110)

    # 방해 피처 (delta_f1 > 0 또는 delta_auc > 0 인 피처) 식별
    harmful_features = [r['excluded_feature'] for r in lofo_results if r['delta_f1'] > 0.0]
    print(f"[*] 식별된 성능 방해/노이즈 피처 ({len(harmful_features)}개): {harmful_features}")

    curated_idx = [k for k, fn in enumerate(feature_names) if fn not in harmful_features]
    curated_names = [feature_names[k] for k in curated_idx]
    print(f"[*] 정예 피처셋 ({len(curated_names)}개): {curated_names}")

    curated_res = eval_lgb(X_val[:, curated_idx], y_val, subs_val)
    print(f"\n[★ 정예 피처셋 교차검증 성능]:")
    print(f"  - Macro F1:    {curated_res['macro_f1']*100:.2f}% (기존 대비 {(curated_res['macro_f1'] - base_res['macro_f1'])*100:+.2f}%p)")
    print(f"  - ROC-AUC:     {curated_res['auc']*100:.2f}% (기존 대비 {(curated_res['auc'] - base_res['auc'])*100:+.2f}%p)")
    print(f"  - Clean F1:    {curated_res['clean_f1']*100:.2f}% (기존 대비 {(curated_res['clean_f1'] - base_res['clean_f1'])*100:+.2f}%p)")
    print(f"  - Defect F1:   {curated_res['defect_f1']*100:.2f}% (기존 대비 {(curated_res['defect_f1'] - base_res['defect_f1'])*100:+.2f}%p)")
    print(f"  - Specificity: {curated_res['spec']*100:.2f}% (기존 대비 {(curated_res['spec'] - base_res['spec'])*100:+.2f}%p)")

    # JSON 저장
    audit_data = {
        'baseline': {
            'macro_f1': float(base_res['macro_f1']),
            'auc': float(base_res['auc']),
            'clean_f1': float(base_res['clean_f1']),
            'defect_f1': float(base_res['defect_f1']),
            'specificity': float(base_res['spec'])
        },
        'feature_stats': stats_summary,
        'lofo_results': lofo_results,
        'harmful_features': harmful_features,
        'curated_features': curated_names,
        'curated_results': {
            'macro_f1': float(curated_res['macro_f1']),
            'auc': float(curated_res['auc']),
            'clean_f1': float(curated_res['clean_f1']),
            'defect_f1': float(curated_res['defect_f1']),
            'specificity': float(curated_res['spec'])
        }
    }

    def convert_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    audit_data = convert_types(audit_data)

    out_json = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_knee_lofo_audit_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] LOFO 감사 결과 JSON 저장 완료: {out_json}")

if __name__ == '__main__':
    audit_features()
