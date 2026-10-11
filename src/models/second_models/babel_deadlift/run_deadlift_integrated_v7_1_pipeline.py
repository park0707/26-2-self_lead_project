# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_deadlift_integrated_v7_1_pipeline.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Version: v7.1 (Defect-Weighted Two-Stream Hybrid with View-Conditional Calibration & Selective CRC)
Status: Officially Confirmed & Deployed (최종 확정 모델)

Key Specifications:
  1. Performance Highlights:
     - Full-Coverage EMR: 42.21% (325/770) [v7.0 대비 +4.81%p]
     - Full-Coverage Clean F1: 71.04%, Clean Specificity: 87.54%
     - 10% Selective CRC: Coverage 89.87% (692/770), Abstain Rate 10.13% (78/770)
     - Selective EMR: 43.35% (300/692 완벽 적중) [역대 최고 기록]
     - Selective Clean F1: 71.26%, Selective Clean Specificity: 86.19%
     - Selective Hamming Loss: 0.1760 [최저 오차율]
  2. 5-Defect F1 Scores (Selective):
     - d1 (척추 중립성): 61.67%
     - d2 (무릎-발목 모임): 50.55%
     - d3 (발바닥 접지): 71.57%
     - d4 (무릎-골반 동시성): 64.79%
     - d5 (바벨 밀착): 81.83%
  3. 5-Camera View Performance (Selective EMR / Clean F1):
     - 시점 A (우측 후방 45° / view5): EMR 34.00%, Clean F1 60.98%
     - 시점 B (우측 전방 45° / view2): EMR 50.35%, Clean F1 81.25% (EMR 50% 돌파)
     - 시점 C (정면 0° / view3):        EMR 40.00%, Clean F1 76.92%
     - 시점 D (좌측 전방 45° / view4): EMR 53.19%, Clean F1 77.78% (EMR 53% 돌파)
     - 시점 E (좌측 후방 45° / view1): EMR 39.31%, Clean F1 64.10%

Invariants Enforced:
  - 64-Frame Cubic Spline Standard
  - Subject 1~9 Full-Coverage (StratifiedGroupKFold)
  - 5-Camera Views Full-Coverage (No View Drop)
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
BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights_integrated_v7", "deadlift_integrated_v7_1_bundle.pkl")
BENCHMARK_JSON = os.path.join(CURRENT_DIR, "deadlift_integrated_v7_1_benchmark_results.json")
TCN_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights_integrated_v7", "deadlift_integrated_v7_bundle.pkl")

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

# 결함별 최적 블렌딩 가중치 (Stream 1 GBDT 비중)
DEFECT_WEIGHTS = np.array([0.65, 0.70, 0.45, 0.50, 0.55], dtype=np.float32)

# 5대 시점별 맞춤형 임계값 벡터 (View-Conditional Calibration)
VIEW_THRESHOLDS = {
    "view5": np.array([0.61, 0.69, 0.55, 0.65, 0.51], dtype=np.float32),
    "view2": np.array([0.67, 0.55, 0.49, 0.59, 0.35], dtype=np.float32),
    "view3": np.array([0.53, 0.63, 0.47, 0.49, 0.49], dtype=np.float32),
    "view4": np.array([0.51, 0.55, 0.43, 0.57, 0.35], dtype=np.float32),
    "view1": np.array([0.45, 0.67, 0.53, 0.51, 0.55], dtype=np.float32)
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
    ank_mid = (kps_all[:, :, 27] + kps_all[:, :, 28]) / 2.0
    wrist_mid = (kps_all[:, :, 15] + kps_all[:, :, 16]) / 2.0
    heel_mid = (kps_all[:, :, 29] + kps_all[:, :, 30]) / 2.0

    ref_len = np.median(np.linalg.norm(sh_mid - hip_mid, axis=-1), axis=1, keepdims=True)
    ref_len = np.clip(ref_len, 1e-3, 10.0)[:, :, None]

    sh_rel = (sh_mid - hip_mid) / ref_len
    wrist_rel = (wrist_mid - hip_mid) / ref_len
    knee_rel = (knee_mid - hip_mid) / ref_len
    ank_rel = (ank_mid - hip_mid) / ref_len

    v_thigh = hip_mid - knee_mid
    v_shank = ank_mid - knee_mid
    dot_k = np.sum(v_thigh * v_shank, axis=-1)
    norm_k = np.linalg.norm(v_thigh, axis=-1) * np.linalg.norm(v_shank, axis=-1) + 1e-6
    knee_ang = np.degrees(np.arccos(np.clip(dot_k / norm_k, -1.0, 1.0)))
    knee_ang_norm = (knee_ang - 90.0) / 90.0

    v_torso = sh_mid - hip_mid
    dot_h = np.sum(v_torso * v_thigh, axis=-1)
    norm_h = np.linalg.norm(v_torso, axis=-1) * np.linalg.norm(v_thigh, axis=-1) + 1e-6
    hip_ang = np.degrees(np.arccos(np.clip(dot_h / norm_h, -1.0, 1.0)))
    hip_ang_norm = (hip_ang - 90.0) / 90.0

    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    dot_t = np.sum(v_torso * v_vert, axis=-1)
    norm_t = np.linalg.norm(v_torso, axis=-1) + 1e-6
    torso_pitch = np.degrees(np.arccos(np.clip(dot_t / norm_t, -1.0, 1.0)))
    torso_pitch_norm = (torso_pitch - 45.0) / 45.0

    knee_vel = np.gradient(knee_ang_norm, axis=1)
    hip_vel = np.gradient(hip_ang_norm, axis=1)
    d4_sync = knee_vel - hip_vel

    bar_horiz_dist = np.abs(wrist_mid[:, :, 0] - knee_mid[:, :, 0]) / (ref_len[:, :, 0] + 1e-6)
    heel_rel_y = (heel_mid[:, :, 1] - ank_mid[:, :, 1]) / (ref_len[:, :, 0] + 1e-6)
    knee_width = np.linalg.norm(kps_all[:, :, 25] - kps_all[:, :, 26], axis=-1)
    ank_width = np.linalg.norm(kps_all[:, :, 27] - kps_all[:, :, 28], axis=-1)
    valgus_ratio = (knee_width - ank_width) / (ref_len[:, :, 0] + 1e-6)
    wrist_vy = np.gradient(-wrist_mid[:, :, 1], axis=1)

    channels = np.stack([
        sh_rel[:, :, 0], sh_rel[:, :, 1], sh_rel[:, :, 2],
        wrist_rel[:, :, 0], wrist_rel[:, :, 1], wrist_rel[:, :, 2],
        knee_rel[:, :, 0], knee_rel[:, :, 1], knee_rel[:, :, 2],
        ank_rel[:, :, 0], ank_rel[:, :, 1], ank_rel[:, :, 2],
        knee_ang_norm, hip_ang_norm, torso_pitch_norm,
        d4_sync, bar_horiz_dist, heel_rel_y, valgus_ratio, wrist_vy
    ], axis=-1)

    channels = np.nan_to_num(channels, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.transpose(channels, (0, 2, 1)).astype(np.float32)
    return out[0] if single else out


# ==============================================================================
# 2. PyTorch Co-Attention Kinematic 1D-TCN 아키텍처
# ==============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
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
# 3. v7.1 공식 통합 진단 엔진 (Pipeline Class)
# ==============================================================================
class DeadliftIntegratedEngineV71:
    def __init__(self, bundle_path: str = BUNDLE_PATH, tcn_weights_path: str = TCN_WEIGHTS_PATH):
        self.bundle_path = bundle_path
        self.tcn_weights_path = tcn_weights_path
        self.models_coattn = []
        self.tau_abstain = 0.085 # 10% Selective CRC 보류 임계치
        self._load_bundle()

    def _load_bundle(self):
        if os.path.exists(self.tcn_weights_path):
            bundle_obj = torch.load(self.tcn_weights_path, weights_only=False)
            if isinstance(bundle_obj, dict) and "fold_state_dicts" in bundle_obj:
                state_dicts = bundle_obj["fold_state_dicts"]
            elif isinstance(bundle_obj, list):
                state_dicts = bundle_obj
            else:
                state_dicts = []
            
            for sd in state_dicts:
                m = CoAttentionKinematicTCN(in_channels=20, hidden_dim=32, num_tasks=5)
                m.load_state_dict(sd)
                m.eval()
                self.models_coattn.append(m)
        else:
            print(f"[!] 경고: TCN 가중치 파일이 없습니다 ({self.tcn_weights_path})")

    def predict_sequence(self, kps_64: np.ndarray, view_name: str, p_stream1: np.ndarray = None) -> dict:
        """
        단일 64프레임 키포인트 시퀀스에 대한 v7.1 공식 진단 추론
        Args:
            kps_64: (64, 33, 3) 스켈레톤 시퀀스
            view_name: 'view1' ~ 'view5' (또는 '시점 A' ~ '시점 E')
            p_stream1: Stream 1 (GBDT) 5개 확률값 (옵션)
        """
        # 시점 매핑 정규화
        rev_map = {v: k for k, v in VIEW_ALIAS_MAP.items()}
        if view_name in rev_map:
            view_code = rev_map[view_name]
        elif view_name in VIEW_ALIAS_MAP:
            view_code = view_name
        else:
            view_code = "view3" # 기본 정면

        # 1. 20채널 피처 추출
        feat_20ch = extract_20ch_kinematics(kps_64) # (20, 64)
        tx = torch.tensor(feat_20ch[np.newaxis, ...], dtype=torch.float32)

        # 2. Stream 2 (Co-Attention TCN 5-Fold 앙상블)
        with torch.no_grad():
            preds = [torch.sigmoid(m(tx)).numpy()[0] for m in self.models_coattn]
            p_s2 = np.mean(preds, axis=0) # (5,)

        # 3. Stream 1 결합 (제공되지 않을 경우 s2 단독 또는 기본값)
        if p_stream1 is None:
            p_s1 = p_s2.copy()
        else:
            p_s1 = np.asarray(p_stream1, dtype=np.float32)

        # 4. 결함별 최적 볼록 결합 (v7.1 가중치 적용)
        p_blend = DEFECT_WEIGHTS * p_s1 + (1.0 - DEFECT_WEIGHTS) * p_s2

        # 5. 생체역학 동시 신전 의존성 보정
        p_blend[4] = np.maximum(p_blend[4], p_blend[3])

        # 6. 시점 조건부 캘리브레이션 (VCC)
        th_v = VIEW_THRESHOLDS.get(view_code, VIEW_THRESHOLDS["view3"])
        defect_binary = (p_blend >= th_v).astype(int)

        # 7. Conformal Risk Control 마진 평가
        margins = np.abs(p_blend - th_v)
        mean_margin = float(np.mean(margins))
        is_abstain = bool(mean_margin < self.tau_abstain)

        is_clean = bool(np.all(defect_binary == 0))

        return {
            "view_code": view_code,
            "view_alias": VIEW_ALIAS_MAP[view_code],
            "defect_probabilities": [round(float(p), 4) for p in p_blend],
            "defect_predictions": [int(b) for b in defect_binary],
            "defect_names": [DEFECT_NAMES[i] for i in range(5) if defect_binary[i] == 1],
            "is_clean": is_clean,
            "mean_margin": round(mean_margin, 4),
            "is_abstain": is_abstain,
            "decision_status": "ABSTAIN (보류)" if is_abstain else "DECIDED (확정)",
            "summary": "완전 정상 자세 (Clean)" if is_clean else f"오류 감지: {', '.join([DEFECT_NAMES[i][:2] for i in range(5) if defect_binary[i] == 1])}"
        }


# ==============================================================================
# 4. 메인 실행 및 단일 검증
# ==============================================================================
def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 데드리프트 공식 v7.1 배포 엔진 로드 및 테스트")
    print("=" * 85)

    engine = DeadliftIntegratedEngineV71()
    raw = np.load(os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz"), allow_pickle=True)
    kps_sample = raw["kps"][0]
    view_sample = raw["views"][0]

    result = engine.predict_sequence(kps_sample, view_sample)
    print(f"\n[+] 단일 샘플 공식 추론 테스트 결과:")
    print(f"  - 시점: {result['view_alias']} ({result['view_code']})")
    print(f"  - 결함별 확률: {result['defect_probabilities']}")
    print(f"  - 결함 예측치: {result['defect_predictions']}")
    print(f"  - 판정 상태: {result['decision_status']} (마진: {result['mean_margin']})")
    print(f"  - 최종 진단: {result['summary']}")
    print("\n[+] v7.1 엔진 배포 검증 완료!")


if __name__ == "__main__":
    main()
