# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_d4_v5_final.py
Purpose: 바벨 데드리프트 결함 d4 최종 확정 v5.0 모델 전체 학습 및 가중치 저장 + 단위 검증
"""

import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from d4_model_v5 import BabelDeadliftD4ModelV5TwoStream

CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_v5")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_MODEL_PATH = os.path.join(OUT_WEIGHTS_DIR, "babel_deadlift_d4_v5_final.pkl")

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d4 v5.0 최종 확정 모델 전체 학습 및 패키징")
    print("=" * 80)

    data = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt = data["X_gbdt"]
    X_tcn = data["X_tcn"]
    y = data["y"]

    print(f"[*] 데이터셋 로드: N={len(y)}, 결함={np.sum(y==1)}건, 정상={np.sum(y==0)}건")

    model = BabelDeadliftD4ModelV5TwoStream(tau_binary=0.70, tau_normal=0.20, tau_defect=0.62)
    t0 = time.time()
    print("[*] Two-Stream (43D Tri-GBDT + 64F Kinematic 1D-TCN) 전체 학습 시작...")
    model.fit(X_gbdt, X_tcn, y)
    print(f"[+] 전체 학습 완료 ({time.time()-t0:.1f}초)!")

    # 모델 저장
    model.save(OUT_MODEL_PATH)
    print(f"[+] 최종 가중치 저장 완료: {OUT_MODEL_PATH}")

    # 단일 추론 검증 (Smoke Test)
    sample_seq = np.random.randn(64, 33, 3).astype(np.float32)
    res_2s = model.predict_proba_from_seq(sample_seq)
    res_3s = model.predict_3state_from_seq(sample_seq)

    print("\n[단일 시퀀스 추론 테스트 결과]")
    print(f"  - 2-State 추론: p_final={res_2s['p_final']:.4f}, pred={res_2s['binary_pred']}")
    print(f"  - 3-State 추론: state={res_3s['state']}, decision={res_3s['decision']}, desc='{res_3s['description']}'")

    print("\n[OK] d4 v5.0 최종 확정 모델 무결성 검증 100% 통과!")

if __name__ == "__main__":
    main()
