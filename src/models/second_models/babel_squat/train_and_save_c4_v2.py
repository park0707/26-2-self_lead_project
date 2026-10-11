# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c4_v2.py
Purpose: 바벨 스쿼트 c4 v2.0 프로덕션 모델 학습 및 가중치 저장 (Level 1~3 실무군 1,824건)
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c4_model_v2 import BabelSquatC4ModelV2

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 v2.0 프로덕션 모델 학습 및 저장")
    print("=" * 80)

    # 1. 피처 로드
    v1_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)

    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d = foot_data['foot_feats'].astype(np.float32)

    X_26d = np.concatenate([X_16d, X_foot_10d], axis=1)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)

    # Level 4 노이즈 배제 인덱스 로드 (v1 기반 82건 배제하여 1,824건 실무군 구성)
    tour_json = 'd:/src/models/second_models/babel_squat/c4_v2_comparison_results.json'
    # tournament에서 산출된 것과 일치하게 필터링
    # v1 레거시 모델 OOF 에러 절대값 상위 82건 배제
    from sklearn.model_selection import StratifiedGroupKFold
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    subjects = ds['subjects']
    oof_casc = np.zeros(len(y_c4))

    for fold, (trn, val) in enumerate(sgkf.split(X_16d, y_c4, groups=subjects)):
        cl = lgb.LGBMClassifier(n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        cl.fit(X_16d[trn], y_c4[trn])
        p_l = cl.predict_proba(X_16d[val])[:, 1]

        cc = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
        cc.fit(X_16d[trn], y_c4[trn])
        p_c = cc.predict_proba(X_16d[val])[:, 1]

        cr = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42+fold, n_jobs=-1)
        cr.fit(X_16d[trn], y_c4[trn])
        p_r = cr.predict_proba(X_16d[val])[:, 1]

        oof_casc[val] = 0.50 * p_c + 0.30 * p_l + 0.20 * p_r

    oof_err = np.abs(oof_casc - y_c4)
    cutoff = np.sort(oof_err)[-82]
    valid_mask = (oof_err < cutoff)
    if np.sum(~valid_mask) != 82:
        sorted_indices = np.argsort(oof_err)
        valid_indices = sorted_indices[:-82]
        valid_mask = np.zeros(len(y_c4), dtype=bool)
        valid_mask[valid_indices] = True

    X_train = X_26d[valid_mask]
    y_train = y_c4[valid_mask]

    print(f"[*] 학습 데이터: N = {len(y_train)}건 (정상: {np.sum(y_train==0)}, 오류: {np.sum(y_train==1)})")
    print(f"[*] 피처 차원: {X_train.shape[1]}D")

    # 2. 모델 인스턴스화 및 학습
    model = BabelSquatC4ModelV2(mode="cascade")
    model.fit(X_train, y_train)

    # 3. 가중치 저장
    weights_path = 'd:/src/models/second_models/babel_squat/c4_model_v2_weights.pkl'
    model.save_weights(weights_path)

    # 4. 검증 로드 및 추론 테스트
    loaded_model = BabelSquatC4ModelV2(mode="cascade")
    loaded_model.load_weights(weights_path)
    test_probs = loaded_model.predict_proba(X_train[:10])
    test_preds = loaded_model.predict(X_train[:10])
    print(f"[*] 샘플 10건 예측 확률:\n{np.round(test_probs[:, 1], 3)}")
    print(f"[*] 샘플 10건 예측 레이블: {test_preds}")
    print(f"[*] 샘플 10건 실제 레이블: {y_train[:10]}")
    print("[*] c4 v2.0 모델 학습 및 가중치 저장 완료!")

if __name__ == '__main__':
    main()
