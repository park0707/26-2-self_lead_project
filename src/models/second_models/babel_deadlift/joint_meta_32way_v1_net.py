# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: joint_meta_32way_v1_net.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Purpose:
  스쿼트 v12.3 SOTA 아키텍처 계승 데드리프트 통합 메타 신경망 v1.0
  - [해결책 1]: 32상태 조건부 상호작용 Factorized Pairwise Energy Head (Unary 5D + Pairwise 10D MLP + Clean Boost 1D)
  - [해결책 2]: Soft-EMR + Soft-Clean F1 복합 손실 함수
  - [해결책 3]: 48D 도메인/시점 피처 기반 FiLM(Feature-wise Linear Modulation) Gated Residual 블록
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

# 32개 결합 상태와 5개 이진 결함(d1, d2, d3, d4, d5) 간의 불리언 매핑 행렬 (32, 5)
STATE_TO_DEFECTS_MAP_32 = np.zeros((32, 5), dtype=np.float32)
for s in range(32):
    STATE_TO_DEFECTS_MAP_32[s, 0] = (s & 16) >> 4  # d1 (척추)
    STATE_TO_DEFECTS_MAP_32[s, 1] = (s & 8)  >> 3  # d2 (무릎방향)
    STATE_TO_DEFECTS_MAP_32[s, 2] = (s & 4)  >> 2  # d3 (지면접지)
    STATE_TO_DEFECTS_MAP_32[s, 3] = (s & 2)  >> 1  # d4 (동시펴짐)
    STATE_TO_DEFECTS_MAP_32[s, 4] = (s & 1)       # d5 (바벨밀착)


def defect_tuple_to_idx(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
    """5개 이진 결함 플래그를 32개 상태 인덱스(0~31)로 변환"""
    return (int(d1) << 4) | (int(d2) << 3) | (int(d3) << 2) | (int(d4) << 1) | int(d5)


def idx_to_defect_tuple(s: int) -> Tuple[int, int, int, int, int]:
    """32개 상태 인덱스(0~31)를 5개 이진 결함 튜플로 변환"""
    return (
        int(STATE_TO_DEFECTS_MAP_32[s, 0]),
        int(STATE_TO_DEFECTS_MAP_32[s, 1]),
        int(STATE_TO_DEFECTS_MAP_32[s, 2]),
        int(STATE_TO_DEFECTS_MAP_32[s, 3]),
        int(STATE_TO_DEFECTS_MAP_32[s, 4])
    )


class FiLMGatedResidualBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Gated Residual 블록
    48D 조건 벡터 cond로부터 gamma(스케일), beta(시프트), gate(게이트) 생성
    h_mod = sigmoid(gate) * ((1 + gamma) * h + beta)
    h_out = LayerNorm(h + Dropout(h_mod))
    """
    def __init__(self, in_features: int, hidden_dim: int, dropout_rate: float = 0.10):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 3)  # gamma, beta, gate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film = self.film_gen(cond)
        gamma, beta, gate = film.chunk(3, dim=-1)
        h_mod = torch.sigmoid(gate) * ((1.0 + gamma) * h + beta)
        return self.norm(h + self.dropout(h_mod))


class FactorizedPairwiseEnergyHead32(nn.Module):
    """
    32상태 조건부 상호작용 에너지 헤드
    - Unary logits (5D)
    - Pairwise interactions (10D MLP: 5개 결함 간 10개 조합)
    - Clean Boost (1D: State 0 Clean 전용 에너지 보정)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_labels = 5
        self.n_pairs = 10

        self.unary = nn.Linear(in_dim, 5)
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(in_dim, self.n_pairs)
        )
        self.clean_boost = nn.Linear(in_dim, 1)
        self.register_buffer('states', torch.from_numpy(STATE_TO_DEFECTS_MAP_32))
        
        # 10대 쌍별 조합 인덱스
        self.pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # (d1, d2), (d1, d3), (d1, d4), (d1, d5)
            (1, 2), (1, 3), (1, 4),          # (d2, d3), (d2, d4), (d2, d5)
            (2, 3), (2, 4),                  # (d3, d4), (d3, d5)
            (3, 4)                           # (d4, d5)
        ]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.unary(h)          # [B, 5]
        w = self.pair_mlp(h)       # [B, 10]
        c0 = self.clean_boost(h)   # [B, 1]

        # 1. Unary 에너지: [B, 32]
        energy = torch.matmul(u, self.states.t())

        # 2. State 0 (Clean) Boost
        energy[:, 0:1] = energy[:, 0:1] + c0

        # 3. 10대 Pairwise 결합 에너지 가산
        for idx, (i, j) in enumerate(self.pairs):
            pair_mask = self.states[:, i] * self.states[:, j]  # [32]
            energy = energy + w[:, idx:idx+1] * pair_mask.unsqueeze(0)

        return energy, u


class JointMeta32WayMoENetV1(nn.Module):
    """
    바벨 데드리프트 통합 진단 메타 신경망 v1.0
    - 48D 통합 메타 입력 텐서
    - FiLM Gated Residual 블록
    - 32-Way Factorized Pairwise Energy Head
    - 5-Way Marginal Auxiliary Head
    """
    def __init__(self, in_features: int = 48, hidden_dim: int = 64, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        # Trunk 인코더
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 32
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU()
        )

        # FiLM Gated Residual 블록 (cond=48D 도메인 피처)
        self.film_block = FiLMGatedResidualBlock(in_features=in_features, hidden_dim=hidden_dim // 2, dropout_rate=0.10)

        # 32-Way Factorized Pairwise Energy Head
        self.energy_head = FactorizedPairwiseEnergyHead32(in_dim=hidden_dim // 2)

        # 5-Way Marginal Auxiliary Head
        self.head_marginal_5 = nn.Linear(hidden_dim // 2, 5)

        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP_32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        h_fused = self.film_block(h, cond=x)
        logits_32, _ = self.energy_head(h_fused)
        logits_5 = self.head_marginal_5(h_fused)
        return logits_32, logits_5

    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            l32, l5 = self.forward(x)
            p32 = F.softmax(l32, dim=-1)
            p5 = torch.sigmoid(l5)
        return p32, p5


class SoftEMRConsistencyLoss32(nn.Module):
    """
    Soft-EMR + Soft-Clean F1 + Cross-Entropy + Marginal BCE 복합 손실함수
    """
    def __init__(
        self,
        w_ce: float = 0.45,
        w_bce: float = 0.25,
        w_cons: float = 0.10,
        w_emr: float = 0.10,
        w_clean: float = 0.10,
        class_weights_32: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = w_ce
        self.w_bce = w_bce
        self.w_cons = w_cons
        self.w_emr = w_emr
        self.w_clean = w_clean

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_32)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP_32))

    def forward(
        self,
        logits_32: torch.Tensor,
        logits_5: torch.Tensor,
        target_s: torch.Tensor,
        target_d5: torch.Tensor
    ) -> torch.Tensor:
        # 1. 32-Way Cross Entropy
        loss_ce = self.ce_loss(logits_32, target_s)

        # 2. 5-Way Marginal BCE
        loss_bce = self.bce_loss(logits_5, target_d5)

        p_joint = F.softmax(logits_32, dim=-1)
        p_marg_from_joint = torch.matmul(p_joint, self.M_matrix)
        p_marg_direct = torch.sigmoid(logits_5)

        # 3. Consistency Loss
        loss_cons = F.mse_loss(p_marg_from_joint, p_marg_direct)

        # 4. Soft-EMR Loss (정답 상태 확률의 기하/산술 평균 최대화)
        target_one_hot = F.one_hot(target_s, num_classes=32).float()
        prob_correct_state = torch.sum(p_joint * target_one_hot, dim=-1)
        loss_emr = 1.0 - torch.mean(prob_correct_state)

        # 5. Soft-Clean F1 Loss (State 0 정밀도/재현율 최적화)
        is_clean_true = (target_s == 0).float()
        p_clean_pred = p_joint[:, 0]
        tp_clean = torch.sum(p_clean_pred * is_clean_true)
        fp_clean = torch.sum(p_clean_pred * (1.0 - is_clean_true))
        fn_clean = torch.sum((1.0 - p_clean_pred) * is_clean_true)
        soft_clean_f1 = (2.0 * tp_clean) / (2.0 * tp_clean + fp_clean + fn_clean + 1e-6)
        loss_clean = 1.0 - soft_clean_f1

        total_loss = (
            self.w_ce * loss_ce +
            self.w_bce * loss_bce +
            self.w_cons * loss_cons +
            self.w_emr * loss_emr +
            self.w_clean * loss_clean
        )
        return total_loss


def decode_hybrid_32(p_joint: np.ndarray, p_marginal: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Joint 32-Way 확률과 Marginal 5-Way 확률 앙상블 블렌딩 디코딩
    """
    p_marg_from_joint = np.matmul(p_joint, STATE_TO_DEFECTS_MAP_32)
    p_final = alpha * p_marg_from_joint + (1.0 - alpha) * p_marginal
    return p_final
