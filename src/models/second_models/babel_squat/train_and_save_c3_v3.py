# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c3_v3.py
Purpose: 바벨 스쿼트 c3 v3.0 최종 모델 학습 및 가중치 저장 (Level 1~3 실무군 1,777건)
"""

import os
import sys
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v3 import BabelSquatC3ModelV3, V2_TO_V3_INDICES

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c3 v3.0 최종 프로덕션 모델 학습 및 저장")
    print("=" * 80)

    # 1. 데이터 로드
    v1_data = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)

    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d = foot_data['foot_feats'].astype(np.float32)

    X_24d = np.concatenate([X_16d, X_foot_8d], axis=1)
    X_20d = X_24d[:, V2_TO_V3_INDICES]

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)

    valgus_inc = X_16d[:, 7]
    pct15_inc = np.percentile(valgus_inc, 15)
    pct85_inc = np.percentile(valgus_inc, 85)

    levels = np.full(len(y_c3), 2, dtype=int)
    mask_l1_ok = (y_c3 == 0) & (valgus_inc <= pct15_inc)
    mask_l1_bad = (y_c3 == 1) & (valgus_inc >= pct85_inc)
    levels[mask_l1_ok | mask_l1_bad] = 1

    asym = X_16d[:, 8]
    pct85_asym = np.percentile(asym, 85)
    mask_l3 = (levels == 2) & (asym > pct85_asym)
    levels[mask_l3] = 3

    pct05_inc = np.percentile(valgus_inc, 5)
    pct95_inc = np.percentile(valgus_inc, 95)
    mask_l4_pos = (y_c3 == 1) & (valgus_inc <= pct05_inc)
    mask_l4_neg = (y_c3 == 0) & (valgus_inc >= pct95_inc)
    levels[mask_l4_pos | mask_l4_neg] = 4

    mask_l13 = (levels <= 3)
    X_train = X_20d[mask_l13]
    y_train = y_c3[mask_l13]
    levels_train = levels[mask_l13]

    print(f"[*] 학습 데이터: N = {len(y_train)}건 (정상: {np.sum(y_train==0)}, 오류: {np.sum(y_train==1)})")
    print(f"[*] 피처 차원: {X_train.shape[1]}D")

    # 2. 모델 인스턴스화 및 학습
    model = BabelSquatC3ModelV3(mode="cascade")
    model.fit(X_train, y_train, levels=levels_train)

    # 3. 가중치 저장
    weights_path = 'd:/src/models/second_models/babel_squat/c3_model_v3_weights.pkl'
    model.save_weights(weights_path)

    # 4. 검증 로드 및 추론 테스트
    loaded_model = BabelSquatC3ModelV3(mode="cascade")
    loaded_model.load_weights(weights_path)
    test_probs = loaded_model.predict_proba(X_train[:10])
    test_preds = loaded_model.predict(X_train[:10])
    print(f"[*] 샘플 10건 예측 확률:\n{np.round(test_probs[:, 1], 3)}")
    print(f"[*] 샘플 10건 예측 레이블: {test_preds}")
    print(f"[*] 샘플 10건 실제 레이블: {y_train[:10]}")
    print("[*] c3 v3.0 모델 학습 및 가중치 저장 완료!")

if __name__ == '__main__':
    main()
