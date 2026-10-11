# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_lunge_c4_misclassifications.py
Purpose: bl4_v3 모델의 5-Fold OOF 예측에서 발생한 오분류(FP, FN, Retake) 샘플 전수 감사
         1) 오분류 샘플 목록 및 메타데이터(피험자, 시점, 결함 코드, 확률값) 집계
         2) 위양성(FP)과 위음성(FN)의 발생 원인별 생체역학적/기하학적 통계 분석
         3) 시점별(view1~5), 피험자별(피험자1~9), 모호성 레벨별 오분류율 상관관계 도출
         4) 오분류 해결을 위한 신규 피처 및 알고리즘 개선 방향 제시
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:/")
from src.features.lunge_c4_features_v3 import LUNGE_C4_FEATURE_NAMES_V3

CACHE_V3_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
BUNDLE_V3_PATH = r"D:\src\models\second_models\babel_lunge\bl4_v3_weights.pkl"
RESULTS_V3_PATH = r"D:\src\models\second_models\babel_lunge\bl4_v3_evaluation_results.json"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def audit_misclassifications():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 런지 c4 bl4_v3 오분류 데이터 전수 조사 및 원인 분석")
    print("=" * 85)

    data = np.load(CACHE_V3_PATH, allow_pickle=True)
    X_24d = data['X_24d']
    X_temp = data['X_temp_14d']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    clip_ids = data['clip_ids']
    ambiguities = data['ambiguities']
    keys = data['keys']

    from src.models.second_models.babel_lunge.bl4_v3 import MultiScaleSETCN, MultiScaleConvBlock, SEBlock1D
    import sys
    main_mod = sys.modules['__main__']
    setattr(main_mod, 'MultiScaleSETCN', MultiScaleSETCN)
    setattr(main_mod, 'MultiScaleConvBlock', MultiScaleConvBlock)
    setattr(main_mod, 'SEBlock1D', SEBlock1D)

    with open(BUNDLE_V3_PATH, 'rb') as f:
        bundle = pickle.load(f)

    with open(RESULTS_V3_PATH, 'r', encoding='utf-8') as f:
        v3_res = json.load(f)

    tau_clean = v3_res['thresholds']['tau_clean']
    tau_def = v3_res['thresholds']['tau_def']

    # 5-Fold 모델 예측 재생성
    from sklearn.model_selection import StratifiedGroupKFold
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_models = bundle['fold_models']
    weights = bundle['weights']

    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_24d, y, groups=subjects)):
        scaler = fold_models['scaler'][fold]
        tcn = fold_models['tcn'][fold].to(device)
        lgb = fold_models['lgb'][fold]
        xgb = fold_models['xgb'][fold]
        cat = fold_models['cat'][fold]

        X_val_sc = scaler.transform(X_24d[val_idx])
        T_val = torch.from_numpy(X_temp[val_idx]).float().to(device)

        tcn.eval()
        with torch.no_grad():
            p_tcn = F.softmax(tcn(T_val), dim=-1)[:, 1].cpu().numpy()

        p_lgb = lgb.predict_proba(X_val_sc)[:, 1]
        p_xgb = xgb.predict_proba(X_val_sc)[:, 1]
        p_cat = cat.predict_proba(X_val_sc)[:, 1]

        p_comb = (
            weights['w_tcn'] * p_tcn +
            weights['w_lgb'] * p_lgb +
            weights['w_xgb'] * p_xgb +
            weights['w_cat'] * p_cat
        )
        oof_probs[val_idx] = p_comb

    # 판정 분류
    preds_3s = []
    types = []

    for i in range(len(y)):
        p = oof_probs[i]
        true_y = y[i]

        if p <= tau_clean:
            pred_3s = 0
            if true_y == 0:
                t = "TN (True Clean)"
            else:
                t = "FN (False Negative - 결함 누락)"
        elif p >= tau_def:
            pred_3s = 1
            if true_y == 1:
                t = "TP (True Defect)"
            else:
                t = "FP (False Positive - 오경보)"
        else:
            pred_3s = -1
            t = "Retake (신뢰 보류)"

        preds_3s.append(pred_3s)
        types.append(t)

    df = pd.DataFrame({
        'clip_id': clip_ids,
        'subject': subjects,
        'view': views,
        'view_name': [VIEW_ALIAS_MAP.get(v, v) for v in views],
        'key': keys,
        'true_y': y,
        'prob': oof_probs,
        'pred_3s': preds_3s,
        'type': types,
        'ambiguity': ambiguities
    })

    # 피처 데이터도 결합
    for idx, fname in enumerate(LUNGE_C4_FEATURE_NAMES_V3):
        df[fname] = X_24d[:, idx]

    # 통계 요약
    print("\n--- [1. 전체 판정 결과 분포] ---")
    type_counts = df['type'].value_counts()
    for t_name, cnt in type_counts.items():
        print(f"  • {t_name:35s}: {cnt:4d}건 ({cnt/len(df)*100:5.2f}%)")

    # FP 분석
    df_fp = df[df['type'].str.startswith('FP')]
    df_fn = df[df['type'].str.startswith('FN')]
    df_retake = df[df['type'].str.startswith('Retake')]

    print(f"\n--- [2. 위양성(FP) 오분류 분석: 총 {len(df_fp)}건 (정상인데 결함으로 판정)] ---")
    print("  [피험자별 FP 분포]")
    print(df_fp['subject'].value_counts().to_string())
    print("\n  [시점별 FP 분포]")
    print(df_fp['view_name'].value_counts().to_string())

    print(f"\n--- [3. 위음성(FN) 오분류 분석: 총 {len(df_fn)}건 (결함인데 정상으로 판정)] ---")
    print("  [피험자별 FN 분포]")
    print(df_fn['subject'].value_counts().to_string())
    print("\n  [시점별 FN 분포]")
    print(df_fn['view_name'].value_counts().to_string())
    print("\n  [결함 세부 유형(Key)별 FN 분포]")
    print(df_fn['key'].value_counts().head(10).to_string())

    # 피처 평균값 비교: True Clean(TN) vs FP vs FN vs TP
    print("\n--- [4. 핵심 피처 평균값 대조표 (오분류 메커니즘 분석)] ---")
    feat_subset = [
        'ear_shoulder_asymmetry', 'shoulder_roll_max', 'head_trunk_yaw_diff',
        'local_trunk_roll_inv', 'trunk_roll_max', 'cervical_yaw_max',
        'cervical_pitch_max', 'cervical_pitch_min', 'counter_tilt_compensation'
    ]
    mean_table = df.groupby('type')[feat_subset].mean()
    print(mean_table.T.to_string())

    # 저장
    df.to_csv(r"D:\src\models\second_models\babel_lunge\bl4_v3_misclassifications.csv", index=False, encoding='utf-8-sig')
    print(f"\n[+] 전수 오분류 분석 CSV 저장 완료: D:\\src\\models\\second_models\\babel_lunge\\bl4_v3_misclassifications.csv")

if __name__ == "__main__":
    audit_misclassifications()
