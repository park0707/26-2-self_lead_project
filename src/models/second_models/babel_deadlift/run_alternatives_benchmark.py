# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_alternatives_benchmark.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Purpose:
  엄격한 벤치마크 검증: 3가지 대안 아키텍처 vs v5.0 베이스라인 비교 평가
  
  대안 1 (temp 1): End-to-End Multi-Task Kinematic 1D-TCN
    - 64프레임 20채널 생체역학 시계열 -> 3블록 확장 인과/대칭 1D-TCN 백본
    - 5개 결함 병렬 멀티태스크 헤드 + Asymmetric Focal Loss (ASL)
    
  대안 2 (temp 2): Biomechanical Phase-Gated Multi-Head Network
    - 운동학적 위상 분할 (First Pull: 바닥~무릎 / Second Pull: 무릎~락아웃 / Descent: 하강)
    - 위상별 특화 특징 풀링(Phase-specific pooling) 및 생체역학적 게이팅 헤드
    
  대안 3 (temp 3): Prototypical Contrastive Discrepancy Network (CoRe)
    - 훈련 폴드 내 정상(Clean) 표준 템플릿과의 동역학적 시계열 편차 벡터 추출
    - 결함별 이탈도(Discrepancy) 기반 다중 라벨 판정
    
  검증 규격:
    - AI-Hub 770건 전수 (피험자 1~9번 전원, 5개 시점 view1~view5 전수)
    - 5-Fold StratifiedGroupKFold (피험자 완전 격리, Clean 층화)
    - EMR, Clean F1, Clean Spec, d1~d5 F1, 5대 시점별 세부 성능 전수 감사
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, hamming_loss
)
import lightgbm as lgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_RAW_KPS = os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz")
CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
OUT_JSON = os.path.join(CURRENT_DIR, "deadlift_alternatives_benchmark_results.json")

VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


# ==============================================================================
# 1. 20채널 생체역학 시계열 피처 추출기
# ==============================================================================
def extract_20ch_kinematics(kps_all: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 원천 3D 스켈레톤 -> (N, 20, 64) 정규화 생체역학 시계열 텐서
    """
    N, T, _, _ = kps_all.shape
    sh_mid = (kps_all[:, :, 11] + kps_all[:, :, 12]) / 2.0
    hip_mid = (kps_all[:, :, 23] + kps_all[:, :, 24]) / 2.0
    knee_mid = (kps_all[:, :, 25] + kps_all[:, :, 26]) / 2.0
    ank_mid = (kps_all[:, :, 27] + kps_all[:, :, 28]) / 2.0
    wrist_mid = (kps_all[:, :, 15] + kps_all[:, :, 16]) / 2.0
    heel_mid = (kps_all[:, :, 29] + kps_all[:, :, 30]) / 2.0
    
    # 기준 길이: 체간 길이 (골반-어깨 중앙)
    ref_len = np.median(np.linalg.norm(sh_mid - hip_mid, axis=-1), axis=1, keepdims=True)
    ref_len = np.clip(ref_len, 1e-3, 10.0)[:, :, None] # (N, 1, 1)
    
    # 상대 좌표 (3D x 4 = 12D)
    sh_rel = (sh_mid - hip_mid) / ref_len
    wrist_rel = (wrist_mid - hip_mid) / ref_len
    knee_rel = (knee_mid - hip_mid) / ref_len
    ank_rel = (ank_mid - hip_mid) / ref_len
    
    # 무릎 각도
    v_thigh = hip_mid - knee_mid
    v_shank = ank_mid - knee_mid
    dot_k = np.sum(v_thigh * v_shank, axis=-1)
    norm_k = np.linalg.norm(v_thigh, axis=-1) * np.linalg.norm(v_shank, axis=-1) + 1e-6
    knee_ang = np.degrees(np.arccos(np.clip(dot_k / norm_k, -1.0, 1.0)))
    knee_ang_norm = (knee_ang - 90.0) / 90.0
    
    # 골반 각도
    v_torso = sh_mid - hip_mid
    dot_h = np.sum(v_torso * v_thigh, axis=-1)
    norm_h = np.linalg.norm(v_torso, axis=-1) * np.linalg.norm(v_thigh, axis=-1) + 1e-6
    hip_ang = np.degrees(np.arccos(np.clip(dot_h / norm_h, -1.0, 1.0)))
    hip_ang_norm = (hip_ang - 90.0) / 90.0
    
    # 상체 경사각 (수직 대비)
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    dot_t = np.sum(v_torso * v_vert, axis=-1)
    norm_t = np.linalg.norm(v_torso, axis=-1) + 1e-6
    torso_pitch = np.degrees(np.arccos(np.clip(dot_t / norm_t, -1.0, 1.0)))
    torso_pitch_norm = (torso_pitch - 45.0) / 45.0
    
    # 각속도 및 d4 동시성 차이
    knee_vel = np.gradient(knee_ang_norm, axis=1)
    hip_vel = np.gradient(hip_ang_norm, axis=1)
    d4_sync = knee_vel - hip_vel
    
    # d5 바벨 수평 이격거리
    bar_horiz_dist = np.abs(wrist_mid[:, :, 0] - knee_mid[:, :, 0]) / (ref_len[:, :, 0] + 1e-6)
    
    # d3 발뒤꿈치 들림 (Heel lift)
    heel_rel_y = (heel_mid[:, :, 1] - ank_mid[:, :, 1]) / (ref_len[:, :, 0] + 1e-6)
    
    # d2 무릎 안쪽 모임 (Valgus)
    knee_width = np.linalg.norm(kps_all[:, :, 25] - kps_all[:, :, 26], axis=-1)
    ank_width = np.linalg.norm(kps_all[:, :, 27] - kps_all[:, :, 28], axis=-1)
    valgus_ratio = (knee_width - ank_width) / (ref_len[:, :, 0] + 1e-6)
    
    # 바벨 수직 이동 속도
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
    return np.transpose(channels, (0, 2, 1)).astype(np.float32) # (N, 20, 64)


# ==============================================================================
# 2. 손실 함수: Multi-Label Asymmetric Focal Loss (ASL)
# ==============================================================================
class MultiLabelAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=2.5, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 5), targets: (B, 5)
        probs = torch.sigmoid(logits)
        
        # Positive loss
        loss_pos = -targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs.clamp(min=self.eps))
        
        # Negative loss with probability shifting/clipping
        p_neg = (probs - self.clip).clamp(min=0.0)
        loss_neg = -(1.0 - targets) * torch.pow(p_neg, self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))
        
        return torch.mean(loss_pos + loss_neg)


# ==============================================================================
# 3. 모델 아키텍처 1: Multi-Task End-to-End Kinematic 1D-TCN (temp 1)
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


class MultiTaskKinematicTCN(nn.Module):
    def __init__(self, in_channels: int = 20, hidden_dim: int = 32, num_tasks: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4)
        )
        # Shared embedding dim: 64 * 2 (avg + max pool) = 128
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
        # x: (B, C, T)
        feat = self.backbone(x) # (B, 64, T)
        avg_p = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        max_p = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        emb = torch.cat([avg_p, max_p], dim=-1) # (B, 128)
        
        logits = [head(emb) for head in self.heads]
        return torch.cat(logits, dim=-1) # (B, 5)


# ==============================================================================
# 4. 모델 아키텍처 2: Biomechanical Phase-Gated Network (temp 2)
# ==============================================================================
class PhaseGatedKinematicNet(nn.Module):
    def __init__(self, in_channels: int = 20, hidden_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4)
        )
        # Phase 1, Phase 2, Global pooling -> 각 64D
        ch = hidden_dim * 2
        # Head d1: Spine (Phase 1 + Phase 2)
        self.head_d1 = nn.Sequential(nn.Linear(ch * 2, 32), nn.GELU(), nn.Linear(32, 1))
        # Head d2: Knee-Foot alignment (Global)
        self.head_d2 = nn.Sequential(nn.Linear(ch, 32), nn.GELU(), nn.Linear(32, 1))
        # Head d3: Foot contact (Phase 1 + Global)
        self.head_d3 = nn.Sequential(nn.Linear(ch * 2, 32), nn.GELU(), nn.Linear(32, 1))
        # Head d4: Knee-Hip synchrony (Phase 1 + Phase 2)
        self.head_d4 = nn.Sequential(nn.Linear(ch * 2, 32), nn.GELU(), nn.Linear(32, 1))
        # Head d5: Barbell proximity (Phase 2 + Global)
        self.head_d5 = nn.Sequential(nn.Linear(ch * 2, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, phase_masks: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 64)
        # phase_masks: (B, 3, 64) -> [Phase 1 mask, Phase 2 mask, Global mask]
        feat = self.backbone(x) # (B, 64, 64)
        
        # Masked average pooling
        p1_mask = phase_masks[:, 0:1, :] # (B, 1, 64)
        p2_mask = phase_masks[:, 1:2, :]
        p3_mask = phase_masks[:, 2:3, :]
        
        z_p1 = torch.sum(feat * p1_mask, dim=-1) / (torch.sum(p1_mask, dim=-1) + 1e-4) # (B, 64)
        z_p2 = torch.sum(feat * p2_mask, dim=-1) / (torch.sum(p2_mask, dim=-1) + 1e-4) # (B, 64)
        z_glob = torch.mean(feat, dim=-1) # (B, 64)
        
        out_d1 = self.head_d1(torch.cat([z_p1, z_p2], dim=-1))
        out_d2 = self.head_d2(z_glob)
        out_d3 = self.head_d3(torch.cat([z_p1, z_glob], dim=-1))
        out_d4 = self.head_d4(torch.cat([z_p1, z_p2], dim=-1))
        out_d5 = self.head_d5(torch.cat([z_p2, z_glob], dim=-1))
        
        return torch.cat([out_d1, out_d2, out_d3, out_d4, out_d5], dim=-1) # (B, 5)


def compute_deadlift_phase_masks(wrist_vy: np.ndarray, wrist_y: np.ndarray) -> np.ndarray:
    """
    각 시퀀스별 운동학적 위상 마스크 생성 (N, 3, 64)
    Phase 1: First Pull (바닥 -> 무릎 통과 구간, 프레임 0 ~ 24경)
    Phase 2: Second Pull (무릎 통과 -> 락아웃 정점, 프레임 24 ~ 40경)
    Phase 3: 하강기 (락아웃 -> 바닥, 프레임 40 ~ 64)
    """
    N, T = wrist_vy.shape
    masks = np.zeros((N, 3, T), dtype=np.float32)
    for i in range(N):
        # 락아웃 정점: 바벨이 가장 높이 올라간 지점 (y좌표가 가장 작은 지점)
        apex_idx = int(np.argmin(wrist_y[i]))
        apex_idx = np.clip(apex_idx, 20, 50)
        
        # 무릎 통과 지점: 시작부터 정점 사이의 50% 지점
        mid_idx = apex_idx // 2
        
        masks[i, 0, 0:mid_idx] = 1.0       # Phase 1
        masks[i, 1, mid_idx:apex_idx] = 1.0 # Phase 2
        masks[i, 2, apex_idx:] = 1.0        # Phase 3
    return masks


# ==============================================================================
# 5. 모델 아키텍처 3: Prototypical Contrastive Discrepancy Network (CoRe, temp 3)
# ==============================================================================
def compute_prototype_discrepancies(X_feats: np.ndarray, train_idx: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    훈련 폴드 내의 Clean 정상 템플릿과의 동역학적 L1/L2 편차 벡터 계산
    X_feats: (N, 20, 64)
    Clean 템플릿들: y_train == [0,0,0,0,0] 인 샘플들
    반환: (N, 20 * 2 = 40D) 채널별 평균 절대 편차(MAE) 및 최대 편차(MaxDev)
    """
    clean_indices = train_idx[np.all(y_train == 0, axis=-1)]
    if len(clean_indices) == 0:
        clean_indices = train_idx[:10] # 폴백
    
    clean_templates = X_feats[clean_indices] # (K, 20, 64)
    # 표준 정상 프로토타입 (K개 템플릿의 평균 및 표준편차)
    proto_mean = np.mean(clean_templates, axis=0, keepdims=True) # (1, 20, 64)
    proto_std = np.std(clean_templates, axis=0, keepdims=True) + 1e-4
    
    # 각 시퀀스와 정상 프로토타입 간의 정규화 편차
    dev = np.abs(X_feats - proto_mean) / proto_std # (N, 20, 64)
    
    mae_per_channel = np.mean(dev, axis=-1) # (N, 20)
    max_per_channel = np.max(dev, axis=-1)  # (N, 20)
    
    return np.column_stack([mae_per_channel, max_per_channel]).astype(np.float32) # (N, 40)


# ==============================================================================
# 6. 공통 평가 헬퍼 함수
# ==============================================================================
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, views: np.ndarray) -> dict:
    emr = float(np.mean(np.all(y_true == y_pred, axis=-1)))
    hl = float(hamming_loss(y_true, y_pred))
    
    clean_true = np.all(y_true == 0, axis=-1).astype(int)
    clean_pred = np.all(y_pred == 0, axis=-1).astype(int)
    
    clean_spec = float(recall_score(1 - clean_true, 1 - clean_pred, zero_division=0)) # True Negative Rate
    clean_f1 = float(f1_score(clean_true, clean_pred, zero_division=0))
    clean_prec = float(precision_score(clean_true, clean_pred, zero_division=0))
    clean_rec = float(recall_score(clean_true, clean_pred, zero_division=0))
    
    defect_f1 = [float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)) for i in range(5)]
    
    view_res = {}
    for vn in VIEW_LIST:
        m = (views == vn)
        if np.sum(m) > 0:
            view_res[vn] = {
                "alias": VIEW_ALIAS_MAP[vn],
                "samples": int(np.sum(m)),
                "emr": float(np.mean(np.all(y_true[m] == y_pred[m], axis=-1))),
                "clean_f1": float(f1_score(clean_true[m], clean_pred[m], zero_division=0))
            }
            
    return {
        "emr": emr,
        "hamming_loss": hl,
        "clean_spec": clean_spec,
        "clean_f1": clean_f1,
        "clean_prec": clean_prec,
        "clean_rec": clean_rec,
        "defect_f1": defect_f1,
        "views": view_res
    }


# ==============================================================================
# 메인 벤치마크 루프
# ==============================================================================
def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 3대 대안 아키텍처 엄격 벤치마크")
    print("  - 데이터셋: 770건 전수 (피험자 1~9번 전원, 5대 전 시점 전수)")
    print("  - 검증 규격: 5-Fold StratifiedGroupKFold (피험자 완전 격리, Clean 층화)")
    print("=" * 80, flush=True)

    # 1. 데이터 로드
    raw_data = np.load(CACHE_RAW_KPS, allow_pickle=True)
    kps = raw_data["kps"]          # (770, 64, 33, 3)
    y_all = raw_data["y"]          # (770, 5)
    subjects = raw_data["subjects"] # (770,)
    views = raw_data["views"]       # (770,)
    N = len(y_all)
    y_clean = np.all(y_all == 0, axis=-1).astype(int)

    print(f"[*] 원천 데이터 로드 완료: N={N}, 정상(Clean)={np.sum(y_clean)}건, 결함={N - np.sum(y_clean)}건")

    # 2. 20채널 생체역학 텐서 추출
    X_kin = extract_20ch_kinematics(kps) # (770, 20, 64)
    wrist_y = kps[:, :, 15, 1]           # (770, 64)
    wrist_vy = X_kin[:, 19, :]           # (770, 64)
    phase_masks = compute_deadlift_phase_masks(wrist_vy, wrist_y) # (770, 3, 64)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------------------------
    # 벤치마크 1: temp 1 (Multi-Task Kinematic 1D-TCN)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [대안 1 (temp 1)] End-to-End Multi-Task Kinematic 1D-TCN 5-Fold 학습 시작...")
    print("#" * 80, flush=True)

    oof_probs_temp1 = np.zeros((N, 5), dtype=np.float32)
    t0_t1 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_kin, y_clean, groups=subjects), 1):
        torch.manual_seed(42 + fold)
        model = MultiTaskKinematicTCN(in_channels=20, hidden_dim=32, num_tasks=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = MultiLabelAsymmetricLoss(gamma_pos=1.0, gamma_neg=2.5)

        tx = torch.tensor(X_kin[train_idx], dtype=torch.float32)
        ty = torch.tensor(y_all[train_idx], dtype=torch.float32)
        vx = torch.tensor(X_kin[val_idx], dtype=torch.float32)

        dataset = TensorDataset(tx, ty)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(25):
            for bx, by in loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(vx)
            oof_probs_temp1[val_idx] = torch.sigmoid(v_logits).numpy()

        print(f"  - Fold {fold}/5 완료 ({time.time() - t0_t1:.1f}s)", flush=True)

    # Threshold 0.5 이진화
    y_pred_temp1 = (oof_probs_temp1 >= 0.50).astype(int)
    eval_temp1 = evaluate_predictions(y_all, y_pred_temp1, views)

    # --------------------------------------------------------------------------
    # 벤치마크 2: temp 2 (Biomechanical Phase-Gated Multi-Head Network)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [대안 2 (temp 2)] Biomechanical Phase-Gated Network 5-Fold 학습 시작...")
    print("#" * 80, flush=True)

    oof_probs_temp2 = np.zeros((N, 5), dtype=np.float32)
    t0_t2 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_kin, y_clean, groups=subjects), 1):
        torch.manual_seed(42 + fold)
        model = PhaseGatedKinematicNet(in_channels=20, hidden_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = MultiLabelAsymmetricLoss(gamma_pos=1.0, gamma_neg=2.5)

        tx = torch.tensor(X_kin[train_idx], dtype=torch.float32)
        tm = torch.tensor(phase_masks[train_idx], dtype=torch.float32)
        ty = torch.tensor(y_all[train_idx], dtype=torch.float32)
        vx = torch.tensor(X_kin[val_idx], dtype=torch.float32)
        vm = torch.tensor(phase_masks[val_idx], dtype=torch.float32)

        dataset = TensorDataset(tx, tm, ty)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(25):
            for bx, bm, by in loader:
                optimizer.zero_grad()
                logits = model(bx, bm)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(vx, vm)
            oof_probs_temp2[val_idx] = torch.sigmoid(v_logits).numpy()

        print(f"  - Fold {fold}/5 완료 ({time.time() - t0_t2:.1f}s)", flush=True)

    y_pred_temp2 = (oof_probs_temp2 >= 0.50).astype(int)
    eval_temp2 = evaluate_predictions(y_all, y_pred_temp2, views)

    # --------------------------------------------------------------------------
    # 벤치마크 3: temp 3 (Prototypical Contrastive Discrepancy Network)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [대안 3 (temp 3)] Prototypical Contrastive Discrepancy Network 5-Fold 학습 시작...")
    print("#" * 80, flush=True)

    oof_probs_temp3 = np.zeros((N, 5), dtype=np.float32)
    t0_t3 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_kin, y_clean, groups=subjects), 1):
        # 훈련 폴드 Clean 템플릿 기반 40D 편차 피처 생성
        X_discrepancy = compute_prototype_discrepancies(X_kin, train_idx, y_all[train_idx])
        
        # 5개 결함 독립 GBDT 학습
        for di in range(5):
            clf = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.04,
                class_weight='balanced', subsample=0.85, colsample_bytree=0.85,
                random_state=42 + fold, verbose=-1
            )
            clf.fit(X_discrepancy[train_idx], y_all[train_idx, di])
            oof_probs_temp3[val_idx, di] = clf.predict_proba(X_discrepancy[val_idx])[:, 1]

        print(f"  - Fold {fold}/5 완료 ({time.time() - t0_t3:.1f}s)", flush=True)

    y_pred_temp3 = (oof_probs_temp3 >= 0.50).astype(int)
    eval_temp3 = evaluate_predictions(y_all, y_pred_temp3, views)

    # --------------------------------------------------------------------------
    # 4. 종합 비교 결과 집계 및 저장
    # --------------------------------------------------------------------------
    # v5.0 기준 데이터 (캐시된 JSON 로드)
    with open(os.path.join(CURRENT_DIR, "deadlift_integrated_v5_benchmark_results.json"), "r", encoding="utf-8") as f:
        v5_json = json.load(f)
    v5_res = v5_json["integrated_v5_structured_failsafe"]

    summary = {
        "dataset_samples": N,
        "clean_samples_true": int(np.sum(y_clean)),
        "baseline_v5": {
            "name": "v5.0 Structured Energy & Fail-Safe",
            "emr": v5_res["emr"],
            "clean_spec": v5_res["clean_spec"],
            "clean_f1": v5_res["clean_f1"],
            "hamming_loss": v5_res["hamming_loss"],
            "defect_f1": v5_res["defect_f1"]
        },
        "temp1_multitask_tcn": {
            "name": "temp 1: End-to-End Multi-Task Kinematic 1D-TCN",
            **eval_temp1
        },
        "temp2_phase_gated": {
            "name": "temp 2: Biomechanical Phase-Gated Network",
            **eval_temp2
        },
        "temp3_prototypical_core": {
            "name": "temp 3: Prototypical Contrastive Discrepancy (CoRe)",
            **eval_temp3
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  [종합 벤치마크 감사 결과]")
    print("=" * 80)
    print(f"{'지표 구분':<30} | {'v5.0 (현재 채택)':<15} | {'temp 1 (E2E TCN)':<15} | {'temp 2 (Phase)':<15} | {'temp 3 (CoRe)':<15}")
    print("-" * 100)
    print(f"{'전수 EMR (완전 일치율)':<30} | {v5_res['emr']*100:>13.2f}% | {eval_temp1['emr']*100:>13.2f}% | {eval_temp2['emr']*100:>13.2f}% | {eval_temp3['emr']*100:>13.2f}%")
    print(f"{'Clean Specificity (정상 특이도)':<30} | {v5_res['clean_spec']*100:>13.2f}% | {eval_temp1['clean_spec']*100:>13.2f}% | {eval_temp2['clean_spec']*100:>13.2f}% | {eval_temp3['clean_spec']*100:>13.2f}%")
    print(f"{'Clean F1-Score':<30} | {v5_res['clean_f1']*100:>13.2f}% | {eval_temp1['clean_f1']*100:>13.2f}% | {eval_temp2['clean_f1']*100:>13.2f}% | {eval_temp3['clean_f1']*100:>13.2f}%")
    print(f"{'Hamming Loss (오분류율)':<30} | {v5_res['hamming_loss']:>14.4f} | {eval_temp1['hamming_loss']:>14.4f} | {eval_temp2['hamming_loss']:>14.4f} | {eval_temp3['hamming_loss']:>14.4f}")
    print("-" * 100)
    for di, dname in enumerate(["d1 (척추 중립)", "d2 (무릎 모임)", "d3 (발바닥 접지)", "d4 (무릎-골반 동시성)", "d5 (바벨 밀착)"]):
        v5_d = v5_res['defect_f1'][di] * 100
        t1_d = eval_temp1['defect_f1'][di] * 100
        t2_d = eval_temp2['defect_f1'][di] * 100
        t3_d = eval_temp3['defect_f1'][di] * 100
        print(f"{dname:<30} | {v5_d:>13.2f}% | {t1_d:>13.2f}% | {t2_d:>13.2f}% | {t3_d:>13.2f}%")
    print("=" * 80)
    print(f"[*] 벤치마크 결과가 성공적으로 저장되었습니다: {OUT_JSON}")


if __name__ == "__main__":
    main()
