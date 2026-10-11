# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect)
Version: v4.0 (Official Production Standard: sd3_v4)
Module: sd3_v4_classifier.py
Purpose: 전략 4 (Multi-View Dynamic Meta Fusion / 36D GBDT + Dilated-TCN 2-Stream)
         판단 보류율 <= 10.0% (커버리지 90.05%) 제약 최적화 3-State 추론 및 진단 인터페이스
Rule: AGENTS.md 규정 및 공식 5대 시점 체계, 순수 비전 MediaPipe 33관절(64F) 입력 준수
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_v3_features,
    FEATURE_NAMES_36_STIFF_D1,
    VIEW_ALIAS_MAP
)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd3_v4_bundle.pkl")


# ------------------------------------------------------------------------------
# Dilated-TCN 시계열 네트워크 정의
# ------------------------------------------------------------------------------
class DilatedTCN(nn.Module):
    def __init__(self, in_channels: int = 36, hidden_dim: int = 32):
        super(DilatedTCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(res1))) + res1
        res3 = self.relu(self.bn3(self.conv3(res2))) + res2
        out = self.pool(res3).squeeze(-1)
        logits = self.fc(out).squeeze(-1)
        return logits


class SD3V4Classifier:
    """
    바벨 스티프 데드리프트 3번 오류(sd3_v4: 바벨 궤적과 몸 밀착 위반) 공식 v4 분류기
    - 전략 4: Multi-View Dynamic Meta Fusion (36D GBDT Tabular + Dilated-TCN Sequence)
    - 판단 보류율 최대 10.0% 제약 하 3-State F1 81.00%, 커버리지 90.05% 달성
    - Tier 1 골든셋(명확한 단독 결함) F1 87.04%, AUC 93.08% 달성
    - 정면(시점 C) 블라인드 스팟 극복 (F1 63.37% / 정확도 76.88%)
    """
    KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            default_path = os.path.join(CURRENT_DIR, "weights", "sd3_v4_bundle.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "weights", "sd3_bundle.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "sd3_model_v4_weights.pkl")
            bundle_path = default_path

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd3_v4 weights bundle not found: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = self.bundle.get('model_name', 'sd3_v4')
        self.defect_code = 'sd3_v4'
        self.defect_name = self.bundle.get('defect_name', '바벨 궤적과 몸 밀착 위반 (Multi-View Meta Fusion / 36D GBDT + Dilated-TCN)')
        self.strategy_name = self.bundle.get('strategy_name', '전략 4: Multi-View Meta Fusion')

        # 1. GBDT 모델 (LightGBM + CatBoost)
        self.lgb_model = self.bundle['lgb_model']
        self.cb_model = self.bundle['cb_model']
        self.gbdt_weights = self.bundle.get('gbdt_sub_weights', (0.55, 0.45))

        # 2. Dilated-TCN 모델
        self.tcn_model = DilatedTCN(in_channels=36, hidden_dim=32)
        self.tcn_model.load_state_dict(self.bundle['tcn_state_dict'])
        self.tcn_model.eval()

        # 3. 시점별 퓨전 가중치 및 적응형 임계치 (보류율 <= 10.0% 제약 최적화)
        self.view_fusion_weights = self.bundle.get('view_fusion_weights', {
            'view3': {'gbdt': 0.55, 'tcn': 0.45},
            'view2': {'gbdt': 0.75, 'tcn': 0.25},
            'view4': {'gbdt': 0.75, 'tcn': 0.25},
            'view1': {'gbdt': 0.65, 'tcn': 0.35},
            'view5': {'gbdt': 0.65, 'tcn': 0.35}
        })

        self.view_thresholds = self.bundle.get('view_thresholds', {
            'view4': (0.39, 0.48),
            'view2': (0.39, 0.48),
            'view1': (0.36, 0.49),
            'view5': (0.36, 0.49),
            'view3': (0.38, 0.49)
        })

        self.opt_3s = self.bundle.get('optimal_3state', {})
        self.coverage = float(self.opt_3s.get('coverage', 90.05))
        self.abstain_rate = float(self.opt_3s.get('abstain_rate', 9.95))

        self.feature_names = self.bundle.get('feature_names', FEATURE_NAMES_36_STIFF_D1)
        self.view_alias_map = self.bundle.get('view_alias_map', VIEW_ALIAS_MAP)

    def _prepare_tensor(self, x_seq: np.ndarray) -> np.ndarray:
        """입력 배열을 (64, 33, 3) 3D 규격으로 정합 및 NaN/Inf 안전 방어"""
        arr = np.array(x_seq, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            if arr.shape == (64, 99):
                arr = arr.reshape(64, 33, 3)
            elif arr.shape == (64, 98):
                pad = np.zeros((64, 1), dtype=np.float32)
                arr = np.hstack([arr, pad]).reshape(64, 33, 3)
            else:
                raise ValueError(f"Unexpected 2D sequence shape: {arr.shape}, expected (64, 99)")
        elif arr.ndim == 3:
            if arr.shape != (64, 33, 3):
                raise ValueError(f"Input shape must be (64, 33, 3), got {arr.shape}")
        else:
            raise ValueError(f"Input dimension must be 2 or 3, got {arr.ndim}")
        return arr

    def predict_proba(self, x_seq: np.ndarray, view_name: str = "view4") -> float:
        """
        단일 시퀀스에 대해 Multi-View Meta Fusion 결함 확률 P(Defect) 반환
        """
        kps_64f = self._prepare_tensor(x_seq)

        # 1. Stream 1: 36D 생체역학 강체 복원 도메인 피처 기반 GBDT
        feat_36 = extract_stiff_d1_v3_features(kps_64f, view_name=view_name)
        feat_36 = np.nan_to_num(feat_36, nan=0.0, posinf=0.0, neginf=0.0)
        X_gbdt = feat_36.reshape(1, -1)
        p_lgb = float(self.lgb_model.predict_proba(X_gbdt)[0, 1])
        p_cb = float(self.cb_model.predict_proba(X_gbdt)[0, 1])
        p_gbdt = float(self.gbdt_weights[0] * p_lgb + self.gbdt_weights[1] * p_cb)

        # 2. Stream 2: 12개 핵심 관절 시계열 기반 Dilated-TCN
        seq_12j = kps_64f[:, self.KEY_JOINTS, :3].reshape(64, 36)
        x_tcn_tensor = torch.tensor(seq_12j.T, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            tcn_logits = self.tcn_model(x_tcn_tensor)
            p_tcn = float(torch.sigmoid(tcn_logits).item())

        # 3. 시점 적응형 Dynamic Meta Fusion
        fw = self.view_fusion_weights.get(view_name, {'gbdt': 0.70, 'tcn': 0.30})
        w_gbdt = fw['gbdt']
        w_tcn = fw['tcn']
        p_fused = w_gbdt * p_gbdt + w_tcn * p_tcn

        return float(np.clip(p_fused, 0.0, 1.0))

    def diagnose(self, x_seq: np.ndarray, view_name: str = "view4") -> Dict[str, Any]:
        """
        단일 시퀀스에 대한 3-State 최종 진단 결과 반환
        """
        p = self.predict_proba(x_seq, view_name=view_name)
        state_2s = 1 if p >= 0.50 else 0

        # 시점별 최적화 임계값 적용 (보류율 <= 10.0% 제약)
        tc, td = self.view_thresholds.get(view_name, (0.38, 0.49))

        if p <= tc:
            state_3s = 0
            label = "정상 확정 (CLEAN)"
            feedback = "바벨이 정강이와 허벅지에 가깝게 수직으로 완벽하게 밀착되어 이동했습니다."
        elif p >= td:
            state_3s = 1
            label = "결함 확정 (DEFECT)"
            feedback = "바벨이 몸(정강이/허벅지)에서 멀리 떨어졌습니다. 광배근에 힘을 주어 바벨을 몸 쪽으로 당겨 밀착시키세요."
        else:
            state_3s = 2
            label = "판단 불가 / 재촬영 (UNCERTAIN)"
            feedback = "카메라 각도 또는 가림으로 바벨 밀착 궤적이 모호합니다. 측면 45도 시점(시점 D 또는 시점 B)에서 1회 다시 수행해 주세요."

        return {
            'defect_code': self.defect_code,
            'defect_name': self.defect_name,
            'strategy_name': self.strategy_name,
            'probability': p,
            'margin': float(abs(p - 0.50)),
            'state_2state': state_2s,
            'state_3state': state_3s,
            'state_label': label,
            'feedback': feedback,
            'view_name': view_name,
            'view_alias': self.view_alias_map.get(view_name, view_name),
            'features_dim': len(self.feature_names),
            'thresholds': {
                'tau_clean': tc,
                'tau_defect': td,
                'coverage': self.coverage,
                'abstain_rate': self.abstain_rate
            }
        }


if __name__ == '__main__':
    clf = SD3V4Classifier()
    print("[+] SD3V4Classifier 로딩 및 초기화 성공!")
    print(f"  - Model: {clf.model_name} ({clf.defect_name})")
    print(f"  - Strategy: {clf.strategy_name}")
    print(f"  - Overall Coverage: {clf.coverage:.2f}% (판단 보류율: {clf.abstain_rate:.2f}% <= 10.0%)")
    print(f"  - 시점별 임계치: {clf.view_thresholds}")

    dummy = np.zeros((64, 33, 3), dtype=np.float32)
    diag = clf.diagnose(dummy, view_name="view4")
    print("\n[+] 더미 진단 테스트 성공:")
    print(" ", diag)
