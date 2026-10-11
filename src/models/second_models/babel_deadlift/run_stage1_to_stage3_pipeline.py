# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_stage1_to_stage3_pipeline.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Purpose:
  차세대 3단계 성능 고도화 파이프라인 (Stage 1 -> Stage 2 -> Stage 3 종합 실증)
  
  [1단계]: 모델 재학습 없는 컨포멀 리스크 제어 (Conformal Risk Control, CRC) 10% 보류 최적화
  [2단계]: 생체역학 불변 데이터 증강 (뼈 길이 스케일링 ±8% + DTW 템포 왜곡 0.88~1.12배)
  [3단계]: 시공간 관절 교차 어텐션 (Spatio-Temporal Co-Attention: Torso <-> Lower Body)
  
  검증 규격:
    - AI-Hub 770건 전수 (피험자 1~9번 전원, 5개 시점 view1~view5 전수)
    - 5-Fold StratifiedGroupKFold (피험자 완전 격리, Clean 층화)
"""

import os
import sys
import time
import json
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
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, hamming_loss
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_RAW_KPS = os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz")
CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
OUT_JSON = os.path.join(CURRENT_DIR, "deadlift_stage1_to_stage3_benchmark_results.json")
OUT_BUNDLE = os.path.join(CURRENT_DIR, "weights_integrated_v6", "deadlift_coattention_aug_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]


# ==============================================================================
# [2단계]: 생체역학 불변 데이터 증강 함수
# ==============================================================================
def scale_bone_lengths(kps: np.ndarray, s_femur: float = 1.05, s_shank: float = 0.95, s_torso: float = 1.03) -> np.ndarray:
    """
    (64, 33, 3) 스켈레톤의 뼈 길이만 신축 (관절 각도 100% 불변)
    """
    kps_out = kps.copy()
    hip_l = kps[:, 23]
    knee_l = kps[:, 25]
    ank_l = kps[:, 27]

    delta_knee_l = (knee_l - hip_l) * (s_femur - 1.0)
    kps_out[:, 25] += delta_knee_l
    delta_ank_l = delta_knee_l + (ank_l - knee_l) * (s_shank - 1.0)
    kps_out[:, 27] += delta_ank_l
    kps_out[:, 29] += delta_ank_l # heel
    kps_out[:, 31] += delta_ank_l # foot

    hip_r = kps[:, 24]
    knee_r = kps[:, 26]
    ank_r = kps[:, 28]

    delta_knee_r = (knee_r - hip_r) * (s_femur - 1.0)
    kps_out[:, 26] += delta_knee_r
    delta_ank_r = delta_knee_r + (ank_r - knee_r) * (s_shank - 1.0)
    kps_out[:, 28] += delta_ank_r
    kps_out[:, 30] += delta_ank_r # heel
    kps_out[:, 32] += delta_ank_r # foot

    sh_l = kps[:, 11]
    sh_r = kps[:, 12]
    delta_sh_l = (sh_l - hip_l) * (s_torso - 1.0)
    delta_sh_r = (sh_r - hip_r) * (s_torso - 1.0)

    kps_out[:, 11] += delta_sh_l
    kps_out[:, 13] += delta_sh_l # elbow
    kps_out[:, 15] += delta_sh_l # wrist
    kps_out[:, 12] += delta_sh_r
    kps_out[:, 14] += delta_sh_r
    kps_out[:, 16] += delta_sh_r

    return kps_out.astype(np.float32)


def warp_tempo_dtw(kps: np.ndarray, warp_factor: float = 1.10) -> np.ndarray:
    """
    (64, 33, 3) 스켈레톤의 시간축을 0.88~1.12배 왜곡 후 64프레임 3차 스플라인 복원
    """
    T = kps.shape[0] # 64
    orig_t = np.linspace(0.0, 1.0, T)
    warped_t = np.power(orig_t, warp_factor)
    warped_t = (warped_t - warped_t[0]) / (warped_t[-1] - warped_t[0] + 1e-6)

    kps_flat = kps.reshape(T, -1)
    cs = CubicSpline(warped_t, kps_flat, axis=0)
    kps_warped = cs(orig_t).reshape(T, 33, 3)
    return kps_warped.astype(np.float32)


# ==============================================================================
# 20채널 생체역학 시계열 피처 추출기
# ==============================================================================
def extract_20ch_kinematics(kps_all: np.ndarray) -> np.ndarray:
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
    return np.transpose(channels, (0, 2, 1)).astype(np.float32)


# ==============================================================================
# [3단계]: Spatio-Temporal Co-Attention 1D-TCN 아키텍처
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
        
        # Dual-stream Co-Attention: 체간(Torso 32ch) <-> 하지(Lower 32ch)
        self.attn_torso_to_lower = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.attn_lower_to_torso = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.ln_t = nn.LayerNorm(32)
        self.ln_l = nn.LayerNorm(32)
        
        emb_dim = hidden_dim * 2 * 2 # 128
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 32),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(32, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x) # (B, 64, 64)
        
        # Co-Attention
        feat_t = feat.permute(0, 2, 1) # (B, 64, 64)
        z_torso = feat_t[:, :, :32]    # (B, 64, 32)
        z_lower = feat_t[:, :, 32:]    # (B, 64, 32)
        
        out_t, _ = self.attn_torso_to_lower(z_torso, z_lower, z_lower)
        z_torso = self.ln_t(z_torso + out_t)
        
        out_l, _ = self.attn_lower_to_torso(z_lower, z_torso, z_torso)
        z_lower = self.ln_l(z_lower + out_l)
        
        feat_fused = torch.cat([z_torso, z_lower], dim=-1).permute(0, 2, 1) # (B, 64, 64)
        
        avg_p = F.adaptive_avg_pool1d(feat_fused, 1).squeeze(-1)
        max_p = F.adaptive_max_pool1d(feat_fused, 1).squeeze(-1)
        emb = torch.cat([avg_p, max_p], dim=-1) # (B, 128)
        
        logits = [head(emb) for head in self.heads]
        return torch.cat(logits, dim=-1)


class MultiLabelAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=2.5, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss_pos = -targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs.clamp(min=self.eps))
        p_neg = (probs - self.clip).clamp(min=0.0)
        loss_neg = -(1.0 - targets) * torch.pow(p_neg, self.gamma_neg) * torch.log((1.0 - p_neg).clamp(min=self.eps))
        return torch.mean(loss_pos + loss_neg)


# ==============================================================================
# 메인 통합 실행 함수
# ==============================================================================
def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 데드리프트 차세대 3단계 성능 고도화 파이프라인")
    print("  - 1단계: Conformal Risk Control (CRC) 10% 보류 최적화")
    print("  - 2단계: 생체역학 불변 데이터 증강 (뼈 길이 ±8% + DTW 템포 0.88~1.12배, 3배 확장)")
    print("  - 3단계: 시공간 관절 교차 어텐션 (Spatio-Temporal Co-Attention 1D-TCN)")
    print("  - 데이터셋: 770건 전수 (피험자 1~9번 전원, 5개 전 시점 전수)")
    print("  - 검증 규격: 5-Fold StratifiedGroupKFold (피험자 완전 격리, Clean 층화)")
    print("=" * 85, flush=True)

    # 1. 데이터 로드
    raw_kps_data = np.load(CACHE_RAW_KPS, allow_pickle=True)
    kps = raw_kps_data["kps"]          # (770, 64, 33, 3)
    y_all = raw_kps_data["y"]          # (770, 5)
    subjects = raw_kps_data["subjects"] # (770,)
    views = raw_kps_data["views"]       # (770,)
    N = len(y_all)
    y_clean = np.all(y_all == 0, axis=-1).astype(int)

    meta_data = np.load(CACHE_META, allow_pickle=True)
    oof_indep = meta_data["oof_probs"] # (770, 5)

    # 원본 20채널 피처 추출
    X_kin_orig = extract_20ch_kinematics(kps) # (770, 20, 64)

    # ==========================================================================
    # [1단계 실행]: v6.0 베이스라인 위 Conformal Risk Control (CRC)
    # ==========================================================================
    print("\n" + "#" * 85)
    print("  [1단계]: 모델 재학습 없는 컨포멀 리스크 제어 (Conformal Risk Control, CRC) 평가")
    print("#" * 85, flush=True)

    oof_tcn_v6 = np.load(os.path.join(CURRENT_DIR, "temp1_oof_probs.npy"))
    p_blend_v6 = 0.50 * oof_indep + 0.50 * oof_tcn_v6
    p_blend_v6[:, 4] = np.maximum(p_blend_v6[:, 4], p_blend_v6[:, 3])

    th_calib = np.array([0.58, 0.58, 0.46, 0.50, 0.50])
    y_pred_calib = (p_blend_v6 >= th_calib).astype(int)

    margins_v6 = np.abs(p_blend_v6 - th_calib)
    mean_margins_v6 = np.mean(margins_v6, axis=-1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_decided_stage1 = np.zeros(N, dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(p_blend_v6, y_clean, groups=subjects), 1):
        calib_scores = mean_margins_v6[train_idx]
        tau_fold = np.quantile(calib_scores, 0.10) # 10% 보류 임계치
        oof_decided_stage1[val_idx] = (mean_margins_v6[val_idx] >= tau_fold)

    y_dec_s1 = y_all[oof_decided_stage1]
    p_dec_s1 = y_pred_calib[oof_decided_stage1]

    emr_s1 = float(np.mean(np.all(y_dec_s1 == p_dec_s1, axis=-1)))
    c_true_s1 = np.all(y_dec_s1 == 0, axis=-1).astype(int)
    c_pred_s1 = np.all(p_dec_s1 == 0, axis=-1).astype(int)
    cf1_s1 = float(f1_score(c_true_s1, c_pred_s1, zero_division=0))
    cspec_s1 = float(recall_score(1 - c_true_s1, 1 - c_pred_s1, zero_division=0))
    cprec_s1 = float(precision_score(c_true_s1, c_pred_s1, zero_division=0))
    crec_s1 = float(recall_score(c_true_s1, c_pred_s1, zero_division=0))
    hl_s1 = float(hamming_loss(y_dec_s1, p_dec_s1))
    def_f1s_s1 = [float(f1_score(y_dec_s1[:, i], p_dec_s1[:, i], zero_division=0)) for i in range(5)]

    print(f"[*] 1단계 CRC 완료: 커버리지={np.mean(oof_decided_stage1)*100:.2f}%, 보류율={100-np.mean(oof_decided_stage1)*100:.2f}%")
    print(f"    - 선택적 EMR: {emr_s1*100:.2f}% ({np.sum(np.all(y_dec_s1 == p_dec_s1, axis=-1))}/{len(y_dec_s1)}건)")
    print(f"    - Clean F1-Score: {cf1_s1*100:.2f}%, Clean Specificity: {cspec_s1*100:.2f}%")

    # ==========================================================================
    # [2단계 + 3단계 실행]: 뼈 길이 & DTW 템포 증강 + Co-Attention TCN 학습
    # ==========================================================================
    print("\n" + "#" * 85)
    print("  [2단계 + 3단계]: 데이터 3배 증강 + Co-Attention 1D-TCN 5-Fold 학습 시작...")
    print("#" * 85, flush=True)

    oof_probs_coattn = np.zeros((N, 5), dtype=np.float32)
    bundle_models = []

    t0_train = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_kin_orig, y_clean, groups=subjects), 1):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)

        # ----------------------------------------------------------------------
        # [2단계]: 훈련 세트 3배 증강 (원본 1배 + 뼈길이 스케일 1배 + DTW 템포 1배)
        # ----------------------------------------------------------------------
        kps_tr_orig = kps[train_idx] # (M, 64, 33, 3)
        y_tr_orig = y_all[train_idx] # (M, 5)

        M = len(train_idx)
        # 뼈 길이 스케일링 증강
        kps_tr_bone = np.zeros_like(kps_tr_orig)
        for i in range(M):
            sf = float(np.random.uniform(0.92, 1.08))
            ss = float(np.random.uniform(0.92, 1.08))
            st = float(np.random.uniform(0.92, 1.08))
            kps_tr_bone[i] = scale_bone_lengths(kps_tr_orig[i], sf, ss, st)

        # DTW 템포 워핑 증강
        kps_tr_tempo = np.zeros_like(kps_tr_orig)
        for i in range(M):
            wf = float(np.random.uniform(0.88, 1.12))
            kps_tr_tempo[i] = warp_tempo_dtw(kps_tr_orig[i], wf)

        # 20채널 피처 추출
        X_tr_bone = extract_20ch_kinematics(kps_tr_bone)
        X_tr_tempo = extract_20ch_kinematics(kps_tr_tempo)
        X_tr_orig = X_kin_orig[train_idx]

        # 3배 결합 (M * 3, 20, 64)
        X_tr_aug = np.concatenate([X_tr_orig, X_tr_bone, X_tr_tempo], axis=0)
        y_tr_aug = np.concatenate([y_tr_orig, y_tr_orig, y_tr_orig], axis=0)

        # 검증 세트는 100% 원본 유지
        X_va = X_kin_orig[val_idx]
        y_va = y_all[val_idx]

        # ----------------------------------------------------------------------
        # [3단계]: Co-Attention Kinematic TCN 학습
        # ----------------------------------------------------------------------
        model = CoAttentionKinematicTCN(in_channels=20, hidden_dim=32, num_tasks=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=1e-4)
        criterion = MultiLabelAsymmetricLoss(gamma_pos=1.0, gamma_neg=2.5)

        tx = torch.tensor(X_tr_aug, dtype=torch.float32)
        ty = torch.tensor(y_tr_aug, dtype=torch.float32)
        vx = torch.tensor(X_va, dtype=torch.float32)

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
            oof_probs_coattn[val_idx] = torch.sigmoid(model(vx)).numpy()

        bundle_models.append(model.state_dict())
        print(f"[*] Fold {fold}/5 완료: 증강 훈련 데이터={len(X_tr_aug)}건, 소요 시간={time.time() - t0_train:.1f}s", flush=True)

    # 4. 하이브리드 블렌딩 (Stream 1 Expert + Stream 2 Co-Attention TCN)
    p_blend_s3 = 0.50 * oof_indep + 0.50 * oof_probs_coattn
    p_blend_s3[:, 4] = np.maximum(p_blend_s3[:, 4], p_blend_s3[:, 3])

    # 5. Full Coverage (100% 전수 채점) 평가
    y_pred_s3_05 = (p_blend_s3 >= 0.50).astype(int)
    emr_s3_05 = float(np.mean(np.all(y_all == y_pred_s3_05, axis=-1)))
    clean_true_s3 = np.all(y_all == 0, axis=-1).astype(int)
    clean_pred_s3 = np.all(y_pred_s3_05 == 0, axis=-1).astype(int)
    cf1_s3_05 = float(f1_score(clean_true_s3, clean_pred_s3, zero_division=0))
    cspec_s3_05 = float(recall_score(1 - clean_true_s3, 1 - clean_pred_s3, zero_division=0))
    hl_s3_05 = float(hamming_loss(y_all, y_pred_s3_05))
    def_f1s_s3_05 = [float(f1_score(y_all[:, i], y_pred_s3_05[:, i], zero_division=0)) for i in range(5)]

    # 6. 보정 임계치 평가
    y_pred_s3_calib = np.zeros_like(p_blend_s3, dtype=int)
    for di in range(5):
        y_pred_s3_calib[:, di] = (p_blend_s3[:, di] >= th_calib[di]).astype(int)
    emr_s3_calib = float(np.mean(np.all(y_all == y_pred_s3_calib, axis=-1)))
    clean_pred_s3_calib = np.all(y_pred_s3_calib == 0, axis=-1).astype(int)
    cf1_s3_calib = float(f1_score(clean_true_s3, clean_pred_s3_calib, zero_division=0))
    cspec_s3_calib = float(recall_score(1 - clean_true_s3, 1 - clean_pred_s3_calib, zero_division=0))
    hl_s3_calib = float(hamming_loss(y_all, y_pred_s3_calib))
    def_f1s_s3_calib = [float(f1_score(y_all[:, i], y_pred_s3_calib[:, i], zero_division=0)) for i in range(5)]

    # 7. [1단계 Conformal Risk Control을 3단계 모델에 연결]: 10% 보류 평가
    margins_s3 = np.abs(p_blend_s3 - th_calib)
    mean_margins_s3 = np.mean(margins_s3, axis=-1)

    oof_decided_s3 = np.zeros(N, dtype=bool)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(p_blend_s3, y_clean, groups=subjects), 1):
        calib_scores = mean_margins_s3[train_idx]
        tau_fold = np.quantile(calib_scores, 0.10) # 10% 보류
        oof_decided_s3[val_idx] = (mean_margins_s3[val_idx] >= tau_fold)

    y_dec_final = y_all[oof_decided_s3]
    p_dec_final = y_pred_s3_calib[oof_decided_s3]
    views_final = views[oof_decided_s3]

    emr_final = float(np.mean(np.all(y_dec_final == p_dec_final, axis=-1)))
    c_true_final = np.all(y_dec_final == 0, axis=-1).astype(int)
    c_pred_final = np.all(p_dec_final == 0, axis=-1).astype(int)
    cf1_final = float(f1_score(c_true_final, c_pred_final, zero_division=0))
    cspec_final = float(recall_score(1 - c_true_final, 1 - c_pred_final, zero_division=0))
    cprec_final = float(precision_score(c_true_final, c_pred_final, zero_division=0))
    crec_final = float(recall_score(c_true_final, c_pred_final, zero_division=0))
    hl_final = float(hamming_loss(y_dec_final, p_dec_final))
    def_f1s_final = [float(f1_score(y_dec_final[:, i], p_dec_final[:, i], zero_division=0)) for i in range(5)]

    views_eval_final = {}
    for vn in VIEW_LIST:
        m = (views_final == vn)
        views_eval_final[vn] = {
            "alias": VIEW_ALIAS_MAP[vn],
            "total_samples": int(np.sum(views == vn)),
            "decided_samples": int(np.sum(m)),
            "emr": float(np.mean(np.all(y_dec_final[m] == p_dec_final[m], axis=-1))),
            "clean_f1": float(f1_score(c_true_final[m], c_pred_final[m], zero_division=0))
        }

    # 8. 종합 요약 및 JSON 저장
    benchmark_summary = {
        "dataset_samples": N,
        "clean_samples_true": int(np.sum(y_clean)),
        "baseline_v5": {
            "name": "v5.0 Structured Energy & Fail-Safe",
            "emr": 0.2987,
            "clean_spec": 0.6250,
            "clean_f1": 0.5848,
            "hamming_loss": 0.2532
        },
        "baseline_v6_full": {
            "name": "v6.0 Two-Stream Hybrid (Full-Coverage)",
            "emr": 0.3519,
            "clean_spec": 0.8443,
            "clean_f1": 0.6327,
            "hamming_loss": 0.2088
        },
        "stage1_conformal_crc_v6": {
            "name": "Stage 1: Conformal Risk Control on v6.0",
            "coverage": float(np.mean(oof_decided_stage1)),
            "abstain_rate": float(1.0 - np.mean(oof_decided_stage1)),
            "decided_samples": int(np.sum(oof_decided_stage1)),
            "emr": emr_s1,
            "clean_f1": cf1_s1,
            "clean_spec": cspec_s1,
            "hamming_loss": hl_s1,
            "defect_f1": def_f1s_s1
        },
        "stage2_stage3_coattn_full": {
            "name": "Stage 2+3: Augmentation + Co-Attention (Full-Coverage)",
            "emr_05": emr_s3_05,
            "emr_calib": emr_s3_calib,
            "clean_f1_05": cf1_s3_05,
            "clean_f1_calib": cf1_s3_calib,
            "clean_spec_calib": cspec_s3_calib,
            "hamming_loss": hl_s3_calib,
            "defect_f1": def_f1s_s3_calib
        },
        "stage1_to_stage3_final_integrated": {
            "name": "Final Integrated: Augmentation + Co-Attention + CRC 10% Selective",
            "coverage": float(np.mean(oof_decided_s3)),
            "abstain_rate": float(1.0 - np.mean(oof_decided_s3)),
            "decided_samples": int(np.sum(oof_decided_s3)),
            "emr": emr_final,
            "clean_f1": cf1_final,
            "clean_spec": cspec_final,
            "clean_prec": cprec_final,
            "clean_rec": crec_final,
            "hamming_loss": hl_final,
            "defect_f1": def_f1s_final,
            "exact_matches": int(np.sum(np.all(y_dec_final == p_dec_final, axis=-1))),
            "views": views_eval_final
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    torch.save(bundle_models, OUT_BUNDLE)

    print("\n" + "=" * 85)
    print("  [1단계~3단계 종합 고도화 실증 결과]")
    print("=" * 85)
    print(f"  * 1단계 (CRC 10% 보류 on v6.0)     : EMR {emr_s1*100:.2f}%, Clean F1 {cf1_s1*100:.2f}%, Spec {cspec_s1*100:.2f}% (보류율 {100-np.mean(oof_decided_stage1)*100:.1f}%)")
    print(f"  * 2+3단계 (증강+Co-Attention Full) : EMR {emr_s3_calib*100:.2f}%, Clean F1 {cf1_s3_calib*100:.2f}%, Spec {cspec_s3_calib*100:.2f}% (전수 100% 채점)")
    print(f"  * 최종 통합 (증강+Co-Attention+CRC): EMR {emr_final*100:.2f}%, Clean F1 {cf1_final*100:.2f}%, Spec {cspec_final*100:.2f}% (보류율 {100-np.mean(oof_decided_s3)*100:.1f}%)")
    print("-" * 85)
    print("  * 5대 결함 최종 F1:")
    def_names = ["d1 (척추 중립)", "d2 (무릎 모임)", "d3 (발바닥 접지)", "d4 (무릎-골반 동시성)", "d5 (바벨 밀착)"]
    for dn, f1 in zip(def_names, def_f1s_final):
        print(f"    - {dn:<22}: {f1 * 100:.2f}%")
    print("-" * 85)
    print("  * 5대 시점별 최종 EMR:")
    for vn in VIEW_LIST:
        vr = views_eval_final[vn]
        print(f"    - {vr['alias']:<25}: EMR {vr['emr']*100:.2f}%, Clean F1 {vr['clean_f1']*100:.2f}% (채점 {vr['decided_samples']}/{vr['total_samples']}건)")
    print("=" * 85)
    print(f"[*] 벤치마크 JSON 저장 완료: {OUT_JSON}")
    print(f"[*] 모델 번들 저장 완료: {OUT_BUNDLE}")


if __name__ == "__main__":
    main()
