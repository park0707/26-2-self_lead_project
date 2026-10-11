# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c4_v1.py
Purpose: 바벨 스쿼트 c4 v1.0 모델 학습 및 가중치 저장 (Level 1~3 실무군 1,824건)
"""

import os
import sys
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c4_model_v1 import BabelSquatC4ModelV1

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 v1.0 프로덕션 모델 학습 및 저장")
    print("=" * 80)

    # 1. 데이터 로드
    c4_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d = c4_data['feats'].astype(np.float32)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)

    heel_lift = X_16d[:, 3]
    stance_drift = X_16d[:, 6]

    pct15_lift = np.percentile(heel_lift, 15)
    pct85_lift = np.percentile(heel_lift, 85)

    levels = np.full(len(y_c4), 2, dtype=int)
    mask_l1_ok = (y_c4 == 0) & (heel_lift <= pct15_lift)
    mask_l1_bad = (y_c4 == 1) & (heel_lift >= pct85_lift)
    levels[mask_l1_ok | mask_l1_bad] = 1

    pct85_drift = np.percentile(stance_drift, 85)
    mask_l3 = (levels == 2) & (stance_drift > pct85_drift)
    levels[mask_l3] = 3

    pct05_lift = np.percentile(heel_lift, 5)
    pct95_lift = np.percentile(heel_lift, 95)
    mask_l4_pos = (y_c4 == 1) & (heel_lift <= pct05_lift)
    mask_l4_neg = (y_c4 == 0) & (heel_lift >= pct95_lift)
    levels[mask_l4_pos | mask_l4_neg] = 4

    mask_l13 = (levels <= 3)
    X_train = X_16d[mask_l13]
    y_train = y_c4[mask_l13]
    levels_train = levels[mask_l13]

    print(f"[*] 학습 데이터: N = {len(y_train)}건 (정상: {np.sum(y_train==0)}, 오류: {np.sum(y_train==1)})")
    print(f"[*] 피처 차원: {X_train.shape[1]}D")

    # 2. 모델 인스턴스화 및 학습
    model = BabelSquatC4ModelV1(mode="cascade")
    model.fit(X_train, y_train, levels=levels_train)

    # 3. 가중치 저장
    weights_path = 'd:/src/models/second_models/babel_squat/c4_model_v1_weights.pkl'
    model.save_weights(weights_path)

    # 4. 검증 로드 및 추론 테스트
    loaded_model = BabelSquatC4ModelV1(mode="cascade")
    loaded_model.load_weights(weights_path)
    test_probs = loaded_model.predict_proba(X_train[:10])
    test_preds = loaded_model.predict(X_train[:10])
    print(f"[*] 샘플 10건 예측 확률:\n{np.round(test_probs[:, 1], 3)}")
    print(f"[*] 샘플 10건 예측 레이블: {test_preds}")
    print(f"[*] 샘플 10건 실제 레이블: {y_train[:10]}")
    print("[*] c4 v1.0 모델 학습 및 가중치 저장 완료!")

if __name__ == '__main__':
    main()
