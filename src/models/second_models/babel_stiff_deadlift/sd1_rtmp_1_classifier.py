# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Version: v1.0-RTM (sd1_rtmp_1)
Module: sd1_rtmp_1_classifier.py
Purpose: RTMPose-m (Halpe26) 최적화 공식 프로덕션 추론기 (SD1RTMP1Classifier)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd1_rtmp_1_features import (
    extract_sd1_rtmp_1_features,
    extract_sd1_rtmp_1_temporal,
    FEATURE_NAMES_30D_SD1_RTMP_1,
    VIEW_INDEX_MAP
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
    "시점 A: 우측 후방 (45도)": "view5", "시점 B: 우측 전방 (45도)": "view2",
    "시점 C: 정면": "view3", "시점 D: 좌측 전방 (45도)": "view4",
    "시점 E: 좌측 후방 (45도)": "view1",
    "view5": "view5", "view2": "view2", "view3": "view3", "view4": "view4", "view1": "view1"
}

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_rtmp_1_bundle.pkl")


class DilatedTCN24(nn.Module):
    def __init__(self, in_channels: int = 24, hidden_dim: int = 32):
        super(DilatedTCN24, self).__init__()
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


class SD1RTMP1Classifier:
    """sd1_rtmp_1 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd1_rtmp_1 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "sd1_rtmp_1"
        self.defect_name = "무릎 구부린 채 고정 위반 (RTMPose 2D 생체역학 GBDT + TCN v1.0)"

        self.lgb_models = self.bundle['lgb_models']
        self.cb_models = self.bundle['cb_models']

        self.tcn_model = DilatedTCN24(in_channels=24, hidden_dim=32)
        if self.bundle.get('tcn_state_dict') is not None:
            self.tcn_model.load_state_dict(self.bundle['tcn_state_dict'])
        self.tcn_model.eval()

        self.view_fusion_weights = self.bundle.get('view_fusion_weights', {})
        self.tau_clean = float(self.bundle['thresholds']['tau_clean'])
        self.tau_defect = float(self.bundle['thresholds']['tau_defect'])

    def _prepare_tensor(self, kps_sequence: np.ndarray) -> np.ndarray:
        arr = np.array(kps_sequence, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 3 or arr.shape[0] != 64 or arr.shape[1] not in [26, 33]:
            raise ValueError(f"입력 형상은 (64, 26, 3) 또는 (64, 33, 3)이어야 합니다. 입력된 형상: {arr.shape}")
        return arr

    def predict_proba(self, kps_sequence: np.ndarray, view_name: str = "view4") -> float:
        kps = self._prepare_tensor(kps_sequence)
        if view_name not in VIEW_CODE_MAP:
            raise ValueError(f"지원되지 않는 카메라 시점입니다: '{view_name}'")
        v_code = VIEW_CODE_MAP[view_name]

        feat_30d = extract_sd1_rtmp_1_features(kps, view_name=v_code).reshape(1, -1)
        p_lgb = np.mean([clf.predict_proba(feat_30d)[0, 1] for clf in self.lgb_models])
        p_cb = np.mean([clf.predict_proba(feat_30d)[0, 1] for clf in self.cb_models])
        p_gbdt = 0.55 * p_lgb + 0.45 * p_cb

        seq_24 = extract_sd1_rtmp_1_temporal(kps)
        x_tcn = torch.tensor(seq_24.T, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_tcn = float(torch.sigmoid(self.tcn_model(x_tcn)).item())

        fw = self.view_fusion_weights.get(v_code, {'gbdt': 0.60, 'tcn': 0.40})
        p_fused = fw['gbdt'] * p_gbdt + fw['tcn'] * p_tcn
        return float(np.clip(p_fused, 0.0, 1.0))

    def diagnose(self, kps_sequence: np.ndarray, view_name: str = "view4") -> Dict[str, Any]:
        kps = self._prepare_tensor(kps_sequence)
        v_code = VIEW_CODE_MAP[view_name]
        v_alias = VIEW_ALIAS_MAP.get(v_code, v_code)

        p = self.predict_proba(kps, view_name=v_code)

        if p < self.tau_clean:
            decision_state = "CLEAN"
            decision_label = "정상 확정 (무릎 고정 양호)"
            feedback = "무릎 각도를 견고하게 고정한 채 순수 고관절 힌지(Hip Hinge)로 스티프 데드리프트를 완벽히 수행했습니다."
        elif p >= self.tau_defect:
            decision_state = "DEFECT"
            decision_label = "결함 확정 (무릎 추가 굴곡/주저앉음 발생)"
            feedback = "동작 중 무릎이 스쿼트처럼 추가로 구부러지거나 주저앉았습니다. 최저점에서도 무릎 각도를 고정하고 엉덩이만 뒤로 밀어주세요."
        else:
            decision_state = "RETAKE"
            decision_label = "판단 보류 / 재촬영 안내 (Uncertain)"
            feedback = "무릎 관절 움직임이 모호하거나 가림이 있습니다. 무릎과 정강이가 선명하게 보이도록 측면 45도(시점 D)에서 1회 다시 수행해 주세요."

        return {
            'model_name': self.model_name,
            'defect_code': 'sd1_rtmp_1',
            'decision_state': decision_state,
            'decision_label': decision_label,
            'defect_probability': round(p, 4),
            'clean_probability': round(1.0 - p, 4),
            'feedback_message': feedback,
            'camera_view': {'view_code': v_code, 'view_alias': v_alias},
            'thresholds': {'tau_clean': self.tau_clean, 'tau_defect': self.tau_defect}
        }
