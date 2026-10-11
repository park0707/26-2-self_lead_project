"""
sd1_v8_classifier.py
바벨 스티프 데드리프트 1번 결함 (sd1: 무릎 구부린 채 고정 위반)
v8.0 공식 프로덕션 추론기 (30D GBDT + Dilated-TCN 2-Stream Multi-View Dynamic Fusion)
규정 준수: vidio-health-trainer, 순수 비전 MediaPipe 33관절(64F) 입력, 3-State Selective Gating
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

# Windows 환경
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd1_v8_features import (
    extract_sd1_v8_features,
    extract_sd1_temporal_12j,
    FEATURE_NAMES_30D_SD1
)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_CODE_MAP = {
    "시점 A": "view5", "시점 B": "view2", "시점 C": "view3", "시점 D": "view4", "시점 E": "view1",
    "view5": "view5", "view2": "view2", "view3": "view3", "view4": "view4", "view1": "view1"
}

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v8_bundle.pkl")


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


class SD1V8Classifier:
    """
    바벨 스티프 데드리프트 1번 오류(sd1_v8: 무릎 구부린 채 고정 위반) 공식 v8 분류기
    - 30D 생체역학 도메인 피처 (KADR, CRP, 위상 분리, 동적 시상면 투영 등)
    - 12관절 3D 궤적 기반 3-Layer Dilated-TCN 시계열 딥러닝 스트림
    - 5대 시점별 Dynamic Meta Fusion 및 3-State 완충망 (보류율 9.57%, 커버리지 90.43%)
    """
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd1_v8 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = self.bundle.get('model_name', 'sd1_v8')
        self.defect_name = self.bundle.get('defect_name', '무릎 구부린 채 고정 위반 (30D GBDT + Dilated-TCN 2-Stream)')
        
        # 1. GBDT 모델
        self.lgb_model = self.bundle['lgb_model']
        self.cb_model = self.bundle['cb_model']
        self.gbdt_weights = self.bundle.get('gbdt_sub_weights', (0.55, 0.45))

        # 2. Dilated-TCN 모델
        self.tcn_model = DilatedTCN(in_channels=36, hidden_dim=32)
        self.tcn_model.load_state_dict(self.bundle['tcn_state_dict'])
        self.tcn_model.eval()

        # 3. 시점별 퓨전 가중치 및 임계치
        self.view_fusion_weights = self.bundle.get('view_fusion_weights', {
            'view1': {'gbdt': 0.50, 'tcn': 0.50},
            'view2': {'gbdt': 0.45, 'tcn': 0.55},
            'view3': {'gbdt': 0.45, 'tcn': 0.55},
            'view4': {'gbdt': 0.70, 'tcn': 0.30},
            'view5': {'gbdt': 0.10, 'tcn': 0.90}
        })
        self.tau_clean = float(self.bundle['thresholds']['tau_clean'])
        self.tau_defect = float(self.bundle['thresholds']['tau_defect'])

    def _prepare_tensor(self, kps_sequence: np.ndarray) -> np.ndarray:
        arr = np.array(kps_sequence, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape != (64, 33, 3):
            raise ValueError(f"입력 규격은 (64, 33, 3)이어야 합니다. 입력된 규격: {arr.shape}")
        return arr

    def predict_proba(self, kps_sequence: np.ndarray, view_name: str = "view4") -> float:
        """
        단일 시퀀스(64, 33, 3)에 대해 무릎 고정 위반 결함 확률 P(Defect) 반환
        """
        kps = self._prepare_tensor(kps_sequence)
        if view_name not in VIEW_CODE_MAP:
            raise ValueError(f"지원되지 않는 카메라 시점입니다: '{view_name}'. 지원 목록: {list(VIEW_CODE_MAP.keys())}")
        v_code = VIEW_CODE_MAP[view_name]

        # 1. Stream 1: 30D 생체역학 GBDT
        feat_30d = extract_sd1_v8_features(kps, view_name=v_code).reshape(1, -1)
        p_lgb = float(self.lgb_model.predict_proba(feat_30d)[0, 1])
        p_cb = float(self.cb_model.predict_proba(feat_30d)[0, 1])
        p_gbdt = self.gbdt_weights[0] * p_lgb + self.gbdt_weights[1] * p_cb

        # 2. Stream 2: Dilated-TCN 시계열
        seq_12j = extract_sd1_temporal_12j(kps) # (64, 36)
        x_tcn = torch.tensor(seq_12j.T, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_tcn = float(torch.sigmoid(self.tcn_model(x_tcn)).item())

        # 3. 시점 적응형 Dynamic Meta Fusion
        fw = self.view_fusion_weights.get(v_code, {'gbdt': 0.60, 'tcn': 0.40})
        p_fused = fw['gbdt'] * p_gbdt + fw['tcn'] * p_tcn
        return float(np.clip(p_fused, 0.0, 1.0))

    def diagnose(self, kps_sequence: np.ndarray, view_name: str = "view4") -> Dict[str, Any]:
        """
        단일 시퀀스에 대한 3-State 최종 진단 결과 반환
        """
        kps = self._prepare_tensor(kps_sequence)
        v_code = VIEW_CODE_MAP[view_name]
        v_alias = VIEW_ALIAS_MAP.get(v_code, v_code)

        p = self.predict_proba(kps, view_name=v_code)

        if p < self.tau_clean:
            decision_state = "CLEAN"
            decision_label = "정상 확정 (무릎 고정 완벽)"
            feedback = "무릎 각도를 약 15도로 견고하게 고정한 채 고관절 힌지만 순수하게 수행했습니다. 훌륭한 테크닉입니다!"
        elif p >= self.tau_defect:
            decision_state = "DEFECT"
            decision_label = "결함 확정 (무릎 구부림/주저앉음 발생)"
            feedback = "동작 중 무릎이 스쿼트처럼 추가로 구부러지거나 주저앉았습니다. 최저점에서도 무릎 각도를 고정하고 엉덩이만 뒤로 밀어주세요."
        else:
            decision_state = "RETAKE"
            decision_label = "판단 보류 / 재촬영 안내 (Uncertain)"
            feedback = "무릎 관절 궤적이 모호합니다. 무릎과 정강이가 선명하게 보이도록 측면 45도(시점 D)에서 1회 다시 수행해 주세요."

        return {
            'model_name': self.model_name,
            'defect_code': 'sd1_v8',
            'decision_state': decision_state,
            'decision_label': decision_label,
            'defect_probability': round(p, 4),
            'clean_probability': round(1.0 - p, 4),
            'feedback_message': feedback,
            'camera_view': {
                'view_code': v_code,
                'view_alias': v_alias
            },
            'thresholds': {
                'tau_clean': self.tau_clean,
                'tau_defect': self.tau_defect
            }
        }
