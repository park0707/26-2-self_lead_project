# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c2_v3.py
Purpose: 바벨 스쿼트 c2 v3.0 최종 모델 학습 및 가중치 저장 (Level 1~3 실무군 1,775건)
"""

import os
import sys
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c2_model_v3 import BabelSquatC2ModelV3, V2_TO_V3_INDICES

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c2 v3.0 최종 프로덕션 모델 학습 및 저장")
    print("=" * 80)

    # 1. 데이터 로드
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])
    elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
    X_18d = X_19d[:, elite_idx]

    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32)[:, :2]

    X_20d = np.concatenate([X_18d, gated_feats], axis=1)
    X_13d = X_20d[:, V2_TO_V3_INDICES]

    amb_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    levels = amb_d['levels']
    y_c2 = amb_d['y_c2'].astype(int)

    mask_l13 = (levels <= 3)
    X_train = X_13d[mask_l13]
    y_train = y_c2[mask_l13]
    levels_train = levels[mask_l13]

    print(f"[*] 학습 데이터: N = {len(y_train)}건 (정상: {np.sum(y_train==0)}, 오류: {np.sum(y_train==1)})")
    print(f"[*] 피처 차원: {X_train.shape[1]}D")

    # 2. 모델 인스턴스화 및 학습
    model = BabelSquatC2ModelV3(mode="cascade")
    model.fit(X_train, y_train, levels=levels_train)

    # 3. 가중치 저장
    weights_path = 'd:/src/models/second_models/babel_squat/c2_model_v3_weights.pkl'
    model.save_weights(weights_path)

    # 4. 검증 로드 및 추론 테스트
    loaded_model = BabelSquatC2ModelV3(mode="cascade")
    loaded_model.load_weights(weights_path)
    test_probs = loaded_model.predict_proba(X_train[:10])
    test_preds = loaded_model.predict(X_train[:10])
    print(f"[*] 샘플 10건 예측 확률:\n{np.round(test_probs[:, 1], 3)}")
    print(f"[*] 샘플 10건 예측 레이블: {test_preds}")
    print(f"[*] 샘플 10건 실제 레이블: {y_train[:10]}")
    print("[*] v3.0 모델 학습 및 가중치 저장 성공 완료!")

if __name__ == '__main__':
    main()
