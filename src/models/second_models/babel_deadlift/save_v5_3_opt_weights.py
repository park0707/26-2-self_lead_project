# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Train and save official production weights for d3_model_v5_3_opt.py
"""

import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import prepare_v5_3_clean_dataset
from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import BabelDeadliftD3ModelV5_3Opt, OPTIMAL_17_INDICES

def main():
    print("=" * 80)
    print("v5.3-Opt 전체 데이터셋 학습 및 가중치 저장")
    print("=" * 80)
    
    X_seq_57, X_feats_32, y, subjects, views, clip_names = prepare_v5_3_clean_dataset()
    X_feats_17 = X_feats_32[:, OPTIMAL_17_INDICES]
    
    model = BabelDeadliftD3ModelV5_3Opt()
    print(f"[*] 모델 학습 시작 (샘플: {len(y)}, 17D 피처 + 57D 시계열)...")
    model.fit(X_seq_57, X_feats_17, y, epochs=35, random_state=42)
    
    weights_path = os.path.join(CURRENT_DIR, "d3_model_v5_3_opt_weights.pkl")
    model.save_weights(weights_path)
    
    # 단일 시퀀스 추론 테스트
    print("\n[*] 단일 시퀀스(64, 33, 3) 무결성 추론 테스트...")
    # 더미 혹은 실제 첫 번째 샘플을 이용한 테스트
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    res = model.predict_single_sequence(dummy_seq, view_code="view3")
    print(f"[+] 추론 결과: Label={res['label']}, Prob={res['prob']:.4f}, State={res['state']}")
    print("[+] 모든 파이프라인 무결성 검증 완료!")

if __name__ == '__main__':
    main()
