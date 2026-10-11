# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: joint_meta_16way_v12_3_net.py
Purpose: v12.3 [전체 해결책 통합 마스터]
         - [해결책 1]: Factorized Pairwise Energy Head (Unary 4D + Pairwise 6D MLP)
         - [해결책 2]: Soft-EMR + Soft-Clean F1 복합 손실
         - [해결책 3]: 39D 도메인 토큰화 & FiLM(Feature-wise Linear Modulation) Gated Residual 중간 융합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

STATE_TO_DEFECTS_MAP = np.zeros((16, 4), dtype=np.float32)
for s in range(16):
    STATE_TO_DEFECTS_MAP[s, 0] = (s & 8) >> 3  # c1
    STATE_TO_DEFECTS_MAP[s, 1] = (s & 4) >> 2  # c2
    STATE_TO_DEFECTS_MAP[s, 2] = (s & 2) >> 1  # c3
    STATE_TO_DEFECTS_MAP[s, 3] = (s & 1)       # c4


class FiLMGatedResidualBlock(nn.Module):
    """
    [해결책 3] Feature-wise Linear Modulation (FiLM) Gated Residual 블록
    도메인 피처 cond로부터 gamma(스케일), beta(시프트), gate(게이트)를 생성하여 은닉 표현 h를 선형 변조
    h_mod = sigma(gate) * ((1 + gamma) * h + beta)
    h_out = h + Dropout(h_mod)
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


class FactorizedPairwiseEnergyHead(nn.Module):
    """
    [해결책 1] 16상태 조건부 상호작용 에너지 헤드
    - Unary logits (4D) + Pairwise interactions (6D) + Clean Boost (1D)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_labels = 4
        self.n_pairs = 6
        
        self.unary = nn.Linear(in_dim, 4)
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(in_dim, self.n_pairs)
        )
        self.clean_boost = nn.Linear(in_dim, 1)
        self.register_buffer('states', torch.from_numpy(STATE_TO_DEFECTS_MAP))
        self.pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.unary(h)          # [B, 4]
        w = self.pair_mlp(h)       # [B, 6]
        c0 = self.clean_boost(h)   # [B, 1]
        
        # 1. Unary energy: [B, 16]
        energy = torch.matmul(u, self.states.t())
        
        # State 0 (Clean) boost
        energy[:, 0:1] = energy[:, 0:1] + c0
        
        # 2. Pairwise energy: [B, 16]
        for idx, (i, j) in enumerate(self.pairs):
            pair_mask = self.states[:, i] * self.states[:, j]
            energy = energy + w[:, idx:idx+1] * pair_mask.unsqueeze(0)
            
        return energy, u


class FiLMStructuredExpertSubNet(nn.Module):
    """v12.3 FiLM 중간 융합 + FactorizedPairwiseEnergyHead 장착 전문가 서브 네트워크"""
    def __init__(self, in_features: int, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU()
        )
        # [해결책 3] FiLM Gated Residual 블록 (cond=39D 도메인 피처)
        self.film_block = FiLMGatedResidualBlock(in_features=in_features, hidden_dim=hidden_dim // 2, dropout_rate=0.10)
        # [해결책 1] Factorized Pairwise Energy Head
        self.energy_head = FactorizedPairwiseEnergyHead(in_dim=hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        h_fused = self.film_block(h, cond=x)
        return self.energy_head(h_fused)


class JointMeta16WayMoENetV12_3(nn.Module):
    """
    vidio-health-trainer v12.3
    View-Conditioned MoE + FiLM Gated Residual + Factorized Pairwise Energy Head
    """
    def __init__(self, in_features: int = 39, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        self.expert_front = FiLMStructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_rear = FiLMStructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_kinetic = FiLMStructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 3)
        )

        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l16_front, l4_front = self.expert_front(x)
        l16_rear, l4_rear = self.expert_rear(x)
        l16_kin, l4_kin = self.expert_kinetic(x)

        gate_logits = self.gate_net(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        g1 = gate_weights[:, 0:1]
        g2 = gate_weights[:, 1:2]
        g3 = gate_weights[:, 2:3]

        fused_l16 = g1 * l16_front + g2 * l16_rear + g3 * l16_kin
        fused_l4 = g1 * l4_front + g2 * l4_rear + g3 * l4_kin

        return fused_l16, fused_l4, gate_weights


class CompositeMetricAlignedLossV12_3(nn.Module):
    """
    [해결책 2] 지표 정렬 복합 목적함수 v12.3
    - CE (16상태) + Reverse-ASL (lambda=3.0) + Marginal BCE + Soft-EMR + Soft-Clean F1 + Gate Balance
    """
    def __init__(
        self,
        weight_ce: float = 0.40,
        weight_clean: float = 0.15,
        weight_bce: float = 0.15,
        weight_soft_emr: float = 0.15,
        weight_soft_clean_f1: float = 0.10,
        weight_consistency: float = 0.025,
        weight_gate_balance: float = 0.025,
        lambda_fc: float = 3.0,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        class_weights_16: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = weight_ce
        self.w_clean = weight_clean
        self.w_bce = weight_bce
        self.w_emr = weight_soft_emr
        self.w_f1 = weight_soft_clean_f1
        self.w_cons = weight_consistency
        self.w_gate = weight_gate_balance

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_16)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_fc = lambda_fc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(
        self,
        logits_16: torch.Tensor,
        logits_4: torch.Tensor,
        targets_16: torch.Tensor,
        targets_4: torch.Tensor,
        gate_weights: torch.Tensor,
        y_clean: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-7

        loss_ce = self.ce_loss(logits_16, targets_16)

        p16 = torch.softmax(logits_16, dim=-1)
        p_clean = p16[:, 0].clamp(eps, 1.0 - eps)
        y_c = y_clean.float()
        loss_pos = -y_c * (1.0 - p_clean).pow(self.gamma_pos) * torch.log(p_clean)
        loss_neg = -self.lambda_fc * (1.0 - y_c) * p_clean.pow(self.gamma_neg) * torch.log(1.0 - p_clean)
        loss_fc = (loss_pos + loss_neg).mean()

        loss_bce = self.bce_loss(logits_4, targets_4)

        p4 = torch.matmul(p16, self.M_matrix).clamp(eps, 1.0 - eps)
        correct_prob = targets_4 * p4 + (1.0 - targets_4) * (1.0 - p4)
        sample_emr_prob = correct_prob.prod(dim=-1).clamp(eps, 1.0)
        loss_soft_emr = -torch.log(sample_emr_prob).mean()

        tp_clean = (y_c * p_clean).sum()
        fp_clean = ((1.0 - y_c) * p_clean).sum()
        fn_clean = (y_c * (1.0 - p_clean)).sum()
        soft_f1 = (2.0 * tp_clean + eps) / (2.0 * tp_clean + fp_clean + fn_clean + eps)
        loss_soft_clean_f1 = 1.0 - soft_f1

        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4, p4_from_bce)

        mean_gates = torch.mean(gate_weights, dim=0)
        uniform = torch.full_like(mean_gates, 1.0 / 3.0)
        loss_gate = F.kl_div(mean_gates.log(), uniform, reduction='sum')

        total = (
            self.w_ce   * loss_ce
            + self.w_clean * loss_fc
            + self.w_bce  * loss_bce
            + self.w_emr  * loss_soft_emr
            + self.w_f1   * loss_soft_clean_f1
            + self.w_cons * loss_cons
            + self.w_gate * loss_gate
        )

        return total, {
            'loss_total':   float(total.item()),
            'loss_ce':      float(loss_ce.item()),
            'loss_fc':      float(loss_fc.item()),
            'loss_bce':     float(loss_bce.item()),
            'loss_emr':     float(loss_soft_emr.item()),
            'loss_f1':      float(loss_soft_clean_f1.item()),
            'loss_cons':    float(loss_cons.item()),
            'loss_gate':    float(loss_gate.item())
        }
