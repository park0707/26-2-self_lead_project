# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_d4_v4_unit.py
Purpose: d4_model_v4.py 모듈의 43D 피처 추출기, 12x64 TCN 입력 생성기,
         KinematicTCN 신경망 및 Asymmetric Focal Loss의 무결성 전수 단위 테스트
"""

import sys
import os
import glob
import numpy as np
import torch

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d4_model_v4 import (
    extract_d4_v4_features_from_seq,
    extract_tcn_temporal_features_from_seq,
    KinematicTCN,
    AsymmetricFocalLoss,
    BabelDeadliftD4ModelV4GBDT,
    TwoStreamCalibratedFusionModel,
    FEATURE_NAMES_43_D4
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"

def test_unit():
    print("=" * 60)
    print("  [Harness Engineering] d4 v4.0 단위 테스트 시작")
    print("=" * 60)

    # 1. 실제 샘플 로드
    npz_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
    assert len(npz_files) > 0, "샘플 npz 파일이 존재하지 않습니다."
    
    sample_data = np.load(npz_files[0])
    kps = sample_data["kps_33_3d_64f"]
    print(f"[1] 원천 데이터 로드 성공: Shape = {kps.shape}")

    # 2. 43D 피처 추출 검증
    feats_43 = extract_d4_v4_features_from_seq(kps)
    assert feats_43.shape == (43,), f"피처 차원 불일치: expected (43,), got {feats_43.shape}"
    assert not np.isnan(feats_43).any(), "피처에 NaN이 존재합니다."
    assert not np.isinf(feats_43).any(), "피처에 Inf가 존재합니다."
    print(f"[2] 43D 피처 추출 성공: {feats_43.shape}, NaN/Inf=0개")
    print(f"    - 신규 6대 피처: HS-VVR={feats_43[37]:.3f}, OnsetLag={feats_43[38]:.3f}, EarlyImpulse={feats_43[39]:.3f}, AccLead={feats_43[40]:.3f}, BarDecouple={feats_43[41]:.3f}, MomentArm={feats_43[42]:.3f}")

    # 3. TCN 시계열 입력 검증
    tcn_in = extract_tcn_temporal_features_from_seq(kps)
    assert tcn_in.shape == (12, 64), f"TCN 입력 차원 불일치: expected (12, 64), got {tcn_in.shape}"
    assert not np.isnan(tcn_in).any(), "TCN 입력에 NaN이 존재합니다."
    print(f"[3] 1D-TCN (12, 64) 시계열 텐서 생성 성공: {tcn_in.shape}")

    # 4. KinematicTCN 순방향/역방향 테스트
    model_tcn = KinematicTCN(in_features=12, hidden_dim=32)
    dummy_batch = torch.tensor(np.stack([tcn_in, tcn_in]), dtype=torch.float32)  # (2, 12, 64)
    logits = model_tcn(dummy_batch)
    assert logits.shape == (2, 1), f"TCN 로짓 Shape 불일치: expected (2, 1), got {logits.shape}"
    print(f"[4] KinematicTCN Forward Pass 성공: output={logits.squeeze().tolist()}")

    # 5. Asymmetric Focal Loss 테스트
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5)
    dummy_targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = criterion(logits, dummy_targets)
    assert not torch.isnan(loss), "Loss에 NaN이 발생했습니다."
    loss.backward()
    print(f"[5] Asymmetric Focal Loss & Backward 계산 성공: loss={loss.item():.4f}")

    # 6. GBDT 모델 피팅 테스트 (간이 10개 샘플)
    dummy_X = np.random.randn(10, 43).astype(np.float32)
    dummy_y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int32)
    gbdt = BabelDeadliftD4ModelV4GBDT()
    gbdt.fit(dummy_X, dummy_y)
    probs = gbdt.predict_proba(dummy_X)
    assert len(probs) == 10, "GBDT 확률 길이 불일치"
    assert not np.isnan(probs).any(), "GBDT 확률에 NaN이 존재합니다."
    print(f"[6] 43D Tri-Ensemble GBDT 피팅 및 추론 성공: probs min={probs.min():.3f}, max={probs.max():.3f}")

    # 7. Two-Stream Fusion 모델 테스트
    fusion = TwoStreamCalibratedFusionModel(tau_normal=0.30, tau_defect=0.50)
    fusion.fit_meta(probs, probs, dummy_y)
    final_p = fusion.predict_proba(probs, probs)
    state_res = fusion.predict_3state(final_p, probs, probs)
    assert len(state_res) == 10, "3-State 결과 길이 불일치"
    print(f"[7] Two-Stream Calibrated Fusion 테스트 성공: state sample={state_res[0]}")

    print("\n>>> [PASSED] d4 v4.0 모든 핵심 모듈 및 단위 테스트 100% 통과! <<<")

if __name__ == "__main__":
    test_unit()
