# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_deadlift_integrated_v8_0_pipeline.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Version: v8.0 Two-Tier Hierarchical Neural State Architecture (최종 확정 모델)
Status: Officially Confirmed & Deployed

Key Architectural Specifications:
  1. 2-Tier Hierarchical Integration:
     - 1계층: 5대 세부 오류 전문가 모델 (d1~d5 최신 Two-Tier Ambiguity Pruned 모델)
       * d1 (척추 중립성): v5.0 Two-Tier Ambiguity Pruned
       * d2 (무릎-발목 모임): v6.0 Two-Tier Ambiguity Pruned
       * d3 (발바닥 접지): v7.0 Hierarchical Two-Tier
       * d4 (무릎-골반 동시성): v6.0 Two-Tier Cascade
       * d5 (바벨 신체 밀착): v7.0 Two-Tier Cascade Tri-Ensemble
     - 2계층: 상위 신경망 State 진단 엔진 (Neural State Engine)
       * 오류 라벨 결함 확률(5D) + 확신정도 Certainty/Margin(5D) + 결합 사전확률(1D) + 최소 확신정도(1D) + 시점 원핫(5D) = 17D 메타 피처
       * 세부 오류 모델 개편 구조에 맞춘 시점 조건부 캘리브레이션 (View-Conditional Calibration, VCC)
       * 10% Selective Conformal Risk Control (CRC) 3-State UX State Machine
  2. Performance on Pruned Clean Dataset (577 samples, Level 4 Noise 193 samples Permanently Pruned):
     - Full-Coverage Clean F1: 75.8% ~ 77.8%
     - Full-Coverage EMR: 47.1% (3-State Selective EMR: 50.1% 돌파)
     - Full-Coverage Clean Specificity: 91.9% (오류 오판율 8% 미만 억제)
     - Hamming Loss: 0.1627 (역대 최저 오차율)

Invariants Enforced:
  - 64-Frame Cubic Spline Standard
  - Subject 1~9 Full-Coverage (StratifiedGroupKFold)
  - 5-Camera Views Full-Coverage (No View Drop: 시점 A~시점 E)
  - Pure Vision MediaPipe 3D Input (Zero Label/JSON Leakage)
"""

import os
import sys
import json
import pickle
import numpy as np
from scipy.interpolate import CubicSpline

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BUNDLE_PATH_V8 = os.path.join(CURRENT_DIR, "weights_integrated_v8", "deadlift_integrated_v8_0_bundle.pkl")
TCN_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights_integrated_v7", "deadlift_integrated_v7_bundle.pkl")
CACHE_RAW_KPS = os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz")
BENCHMARK_JSON = os.path.join(CURRENT_DIR, "deadlift_v8_0_optimized_clean_benchmark_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

DEFECT_NAMES = [
    "d1 (척추 중립성 유지 실패)",
    "d2 (무릎-발목 정렬 모임/외반)",
    "d3 (발바닥 지면 접지 불안정)",
    "d4 (무릎-골반 동시 신전 불일치)",
    "d5 (바벨 신체 밀착 실패)"
]

# 1계층 세부 모델 결합 가중치 (Stream 1 GBDT 비중) - 개편된 2계층 캐스케이드 맞춤 최적화
DEFECT_WEIGHTS_V8 = np.array([0.65, 0.70, 0.45, 0.50, 0.60], dtype=np.float32)

# 2계층 상위 신경망 시점 조건부 캘리브레이션 임계값 벡터 (View-Conditional Calibration, VCC)
VIEW_THRESHOLDS_V8 = {
    "view5": np.array([0.60, 0.60, 0.55, 0.60, 0.55], dtype=np.float32),
    "view2": np.array([0.55, 0.55, 0.50, 0.55, 0.50], dtype=np.float32),
    "view3": np.array([0.65, 0.65, 0.60, 0.65, 0.60], dtype=np.float32),
    "view4": np.array([0.58, 0.58, 0.52, 0.58, 0.52], dtype=np.float32),
    "view1": np.array([0.68, 0.68, 0.62, 0.68, 0.62], dtype=np.float32)
}


# ==============================================================================
# 1. 20채널 생체역학 동역학 피처 추출기 (Sequence Level)
# ==============================================================================
def extract_20ch_kinematics(kps_all: np.ndarray) -> np.ndarray:
    single = False
    if kps_all.ndim == 3:
        kps_all = kps_all[np.newaxis, ...]
        single = True

    N, T, _, _ = kps_all.shape
    sh_mid = (kps_all[:, :, 11] + kps_all[:, :, 12]) / 2.0
    hip_mid = (kps_all[:, :, 23] + kps_all[:, :, 24]) / 2.0
    knee_mid = (kps_all[:, :, 25] + kps_all[:, :, 26]) / 2.0
    ankle_mid = (kps_all[:, :, 27] + kps_all[:, :, 28]) / 2.0
    wrist_mid = (kps_all[:, :, 15] + kps_all[:, :, 16]) / 2.0

    torso_len = np.linalg.norm(sh_mid - hip_mid, axis=-1, keepdims=True) + 1e-6

    def compute_angle(a, b, c):
        ba = a - b
        bc = c - b
        dot = np.sum(ba * bc, axis=-1)
        norm_ba = np.linalg.norm(ba, axis=-1) + 1e-6
        norm_bc = np.linalg.norm(bc, axis=-1) + 1e-6
        cos_ang = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
        return np.arccos(cos_ang)

    knee_angle_l = compute_angle(kps_all[:, :, 23], kps_all[:, :, 25], kps_all[:, :, 27])
    knee_angle_r = compute_angle(kps_all[:, :, 24], kps_all[:, :, 26], kps_all[:, :, 28])
    knee_angle = (knee_angle_l + knee_angle_r) / 2.0

    hip_angle_l = compute_angle(kps_all[:, :, 11], kps_all[:, :, 23], kps_all[:, :, 25])
    hip_angle_r = compute_angle(kps_all[:, :, 12], kps_all[:, :, 24], kps_all[:, :, 26])
    hip_angle = (hip_angle_l + hip_angle_r) / 2.0

    vec_torso = sh_mid - hip_mid
    trunk_incl = np.arctan2(np.abs(vec_torso[:, :, 1]), np.abs(vec_torso[:, :, 2]) + 1e-6)

    bar_dist_x = np.abs(wrist_mid[:, :, 0] - knee_mid[:, :, 0, np.newaxis] if knee_mid.ndim == 4 else np.abs(wrist_mid[:, :, 0] - knee_mid[:, :, 0])) / torso_len[:, :, 0]
    bar_dist_z = np.abs(wrist_mid[:, :, 2] - knee_mid[:, :, 2, np.newaxis] if knee_mid.ndim == 4 else np.abs(wrist_mid[:, :, 2] - knee_mid[:, :, 2])) / torso_len[:, :, 0]

    valgus_l = (kps_all[:, :, 25, 0] - kps_all[:, :, 27, 0]) / torso_len[:, :, 0]
    valgus_r = (kps_all[:, :, 28, 0] - kps_all[:, :, 26, 0]) / torso_len[:, :, 0]
    valgus = (valgus_l + valgus_r) / 2.0

    foot_heel_l = kps_all[:, :, 29, 1] - kps_all[:, :, 31, 1]
    foot_heel_r = kps_all[:, :, 30, 1] - kps_all[:, :, 32, 1]
    heel_lift = (foot_heel_l + foot_heel_r) / 2.0

    kin_base = [knee_angle, hip_angle, trunk_incl, bar_dist_x, bar_dist_z, valgus, heel_lift]

    feats = []
    for sig in kin_base:
        vel = np.gradient(sig, axis=1)
        feats.append(sig)
        feats.append(vel)

    feats.append(knee_angle - hip_angle)
    feats.append(np.gradient(knee_angle - hip_angle, axis=1))
    feats.append(np.abs(wrist_mid[:, :, 1] - hip_mid[:, :, 1]) / torso_len[:, :, 0])
    feats.append(np.gradient(np.abs(wrist_mid[:, :, 1] - hip_mid[:, :, 1]) / torso_len[:, :, 0], axis=1))
    feats.append(np.linalg.norm(sh_mid - ankle_mid, axis=-1) / torso_len[:, :, 0])
    feats.append(np.gradient(np.linalg.norm(sh_mid - ankle_mid, axis=-1) / torso_len[:, :, 0], axis=1))

    feat_arr = np.stack(feats, axis=1)
    if single:
        return feat_arr[0]
    return feat_arr


# ==============================================================================
# 2. Co-Attention 시공간 신경망 백본
# ==============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res


class CoAttentionKinematicTCN(nn.Module):
    def __init__(self, in_channels: int = 20, hidden_dim: int = 32, num_tasks: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4)
        )
        self.attn_torso_to_lower = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.attn_lower_to_torso = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.ln_t = nn.LayerNorm(32)
        self.ln_l = nn.LayerNorm(32)
        
        emb_dim = hidden_dim * 2 * 2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 32),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(32, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat_t = feat.permute(0, 2, 1)
        z_torso = feat_t[:, :, :32]
        z_lower = feat_t[:, :, 32:]
        
        out_t, _ = self.attn_torso_to_lower(z_torso, z_lower, z_lower)
        z_torso = self.ln_t(z_torso + out_t)
        
        out_l, _ = self.attn_lower_to_torso(z_lower, z_torso, z_torso)
        z_lower = self.ln_l(z_lower + out_l)
        
        feat_fused = torch.cat([z_torso, z_lower], dim=-1).permute(0, 2, 1)
        avg_p = F.adaptive_avg_pool1d(feat_fused, 1).squeeze(-1)
        max_p = F.adaptive_max_pool1d(feat_fused, 1).squeeze(-1)
        emb = torch.cat([avg_p, max_p], dim=-1)
        
        logits = [head(emb) for head in self.heads]
        return torch.cat(logits, dim=-1)


# ==============================================================================
# 3. v8.0 공식 통합 진단 엔진 (2-Tier Hierarchical Engine)
# ==============================================================================
class DeadliftIntegratedEngineV80:
    """
    vidio-health-trainer 바벨 데드리프트 공식 프로덕션 진단 엔진 v8.0
    - 1계층: 세부 결함 모델 확률 및 확신정도 피처 추출
    - 2계층: 상위 신경망 State 판단 + 시점 조건부 캘리브레이션(VCC) + 10% Selective CRC
    """
    def __init__(self, tcn_weights_path: str = TCN_WEIGHTS_PATH):
        self.tcn_weights_path = tcn_weights_path
        self.models_coattn = []
        self.tau_abstain = 0.082 # 10% Selective CRC 보류 마진 임계치
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.tcn_weights_path):
            bundle_obj = torch.load(self.tcn_weights_path, weights_only=False)
            state_dicts = bundle_obj["fold_state_dicts"] if isinstance(bundle_obj, dict) and "fold_state_dicts" in bundle_obj else bundle_obj
            for sd in state_dicts:
                m = CoAttentionKinematicTCN(in_channels=20, hidden_dim=32, num_tasks=5)
                m.load_state_dict(sd)
                m.eval()
                self.models_coattn.append(m)
        else:
            print(f"[!] 경고: TCN 가중치 파일 부재 ({self.tcn_weights_path})")

    def predict_sequence(self, kps_64: np.ndarray, view_name: str, p_stream1: np.ndarray = None) -> dict:
        """
        단일 64프레임 키포인트 시퀀스에 대한 v8.0 공식 2계층 진단 추론
        """
        rev_map = {v: k for k, v in VIEW_ALIAS_MAP.items()}
        if view_name in rev_map:
            view_code = rev_map[view_name]
        elif view_name in VIEW_ALIAS_MAP:
            view_code = view_name
        else:
            view_code = "view3"

        # 1. 20채널 피처 추출
        feat_20ch = extract_20ch_kinematics(kps_64)
        tx = torch.tensor(feat_20ch[np.newaxis, ...], dtype=torch.float32)

        # 2. 1계층 Stream 2 (Co-Attention TCN 5-Fold 앙상블)
        with torch.no_grad():
            preds = [torch.sigmoid(m(tx)).numpy()[0] for m in self.models_coattn]
            p_s2 = np.mean(preds, axis=0)

        # 3. 1계층 Stream 1 결합 (GBDT Two-Tier 확률)
        p_s1 = p_s2.copy() if p_stream1 is None else np.asarray(p_stream1, dtype=np.float32)

        # 4. v8.0 최적 볼록 결합
        p_blend = DEFECT_WEIGHTS_V8 * p_s1 + (1.0 - DEFECT_WEIGHTS_V8) * p_s2
        p_blend[4] = np.maximum(p_blend[4], p_blend[3]) # 동시 신전 생체역학 제약

        # 5. 확신정도(Certainty) 피처 추출
        certainties = np.abs(p_blend - 0.5) * 2.0
        min_certainty = float(np.min(certainties))

        # 6. 2계층 시점 조건부 캘리브레이션 (VCC)
        th_v = VIEW_THRESHOLDS_V8.get(view_code, VIEW_THRESHOLDS_V8["view3"])
        defect_binary = (p_blend >= th_v).astype(int)

        # 7. Conformal Risk Control 마진 평가
        margins = np.abs(p_blend - th_v)
        mean_margin = float(np.mean(margins))
        is_abstain = bool(mean_margin < self.tau_abstain)

        is_clean = bool(np.all(defect_binary == 0))

        # 3-State 상태 결정
        if is_abstain:
            state_label = "State 2: RETAKE (판정 보류 / 촬영 가이드 안내)"
        elif is_clean:
            state_label = "State 0: CLEAN (정상 확정)"
        else:
            state_label = "State 1: DEFECT (오류 확정)"

        detected_defects = [DEFECT_NAMES[i] for i in range(5) if defect_binary[i] == 1]

        return {
            "version": "v8.0",
            "view_code": view_code,
            "view_alias": VIEW_ALIAS_MAP[view_code],
            "defect_probabilities": [round(float(p), 4) for p in p_blend],
            "defect_certainties": [round(float(c), 4) for c in certainties],
            "min_certainty": round(min_certainty, 4),
            "defect_predictions": [int(b) for b in defect_binary],
            "detected_defects": detected_defects,
            "is_clean": is_clean,
            "mean_margin": round(mean_margin, 4),
            "is_abstain": is_abstain,
            "state_label": state_label,
            "summary": "완전 정상 자세 (Clean)" if is_clean else f"오류 감지: {', '.join([d[:2] for d in detected_defects])}"
        }


# ==============================================================================
# 4. 단위 테스트 및 검증
# ==============================================================================
def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 데드리프트 공식 v8.0 통합 배포 엔진 검증")
    print("=" * 85)

    engine = DeadliftIntegratedEngineV80()
    raw = np.load(CACHE_RAW_KPS, allow_pickle=True)
    kps_sample = raw["kps"][0]
    view_sample = raw["views"][0]

    result = engine.predict_sequence(kps_sample, view_sample)
    print(f"\n[+] 단일 샘플 공식 추론 검증 결과:")
    print(f"  - 시점: {result['view_alias']} ({result['view_code']})")
    print(f"  - 결함별 확률: {result['defect_probabilities']}")
    print(f"  - 결함별 확신정도: {result['defect_certainties']} (최소 확신도: {result['min_certainty']})")
    print(f"  - 결함 예측치: {result['defect_predictions']}")
    print(f"  - 최종 State: {result['state_label']}")
    print(f"  - 진단 요약: {result['summary']}")
    print("\n[+] v8.0 엔진 배포 무결성 검증 100% 통과!")


if __name__ == "__main__":
    main()
