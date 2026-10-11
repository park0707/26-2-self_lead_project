# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: joint_meta_16way_v12_1_net.py
Purpose: v12.1 [해결책 1] Factorized Pairwise Energy Head 장착 MoE 메타 신경망
         - Unary Head (4D): c1~c4 개별 결함 logit
         - Pairwise MLP (6D): 6대 생체역학 동반 결함 상호작용 에너지
         - Clean Baseline Boost (1D): 무결점 정상 상태 전용 안정화 항
         - Exact 16-State Energy Engine: E(y|h) = sum(u_i*y_i) + sum(w_ij*y_i*y_j) + c_0
         - 3대 전문가 (Front, Rear, Kinetic Chain) 모두 FactorizedPairwiseEnergyHead 장착
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
        
        # 1. Unary Head
        self.unary = nn.Linear(in_dim, 4)
        
        # 2. Pairwise MLP (입력 동작 특징에 따른 가변 상호작용 세기)
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(in_dim, self.n_pairs)
        )
        
        # 3. Clean baseline energy boost
        self.clean_boost = nn.Linear(in_dim, 1)
        
        # 4. 16상태 이진 버퍼 등록 [16, 4]
        self.register_buffer('states', torch.from_numpy(STATE_TO_DEFECTS_MAP))
        self.pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, in_dim] 잠재 벡터
        Returns:
            energy: [B, 16] 16개 결합 상태의 총 에너지
            u: [B, 4] 4대 결함 개별 Unary 로짓
        """
        u = self.unary(h)          # [B, 4]
        w = self.pair_mlp(h)       # [B, 6]
        c0 = self.clean_boost(h)   # [B, 1]
        
        # 1. Unary energy: sum_i (S_si * u_i) -> [B, 16]
        energy = torch.matmul(u, self.states.t())
        
        # State 0 (Clean [0,0,0,0])에 baseline boost 결합
        energy[:, 0:1] = energy[:, 0:1] + c0
        
        # 2. Pairwise interactions: sum_{i<j} (S_si * S_sj * w_ij)
        for idx, (i, j) in enumerate(self.pairs):
            pair_mask = self.states[:, i] * self.states[:, j] # [16]
            energy = energy + w[:, idx:idx+1] * pair_mask.unsqueeze(0)
            
        return energy, u


class StructuredExpertSubNetV12(nn.Module):
    """v12.1 FactorizedPairwiseEnergyHead 장착 전문가 서브 네트워크"""
    def __init__(self, in_features: int, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU()
        )
        self.energy_head = FactorizedPairwiseEnergyHead(in_dim=hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.energy_head(h)


class JointMeta16WayMoENetV12_1(nn.Module):
    """
    vidio-health-trainer v12.1
    View-Conditioned Mixture-of-Experts (MoE) + Factorized Pairwise Energy Head
    """
    def __init__(self, in_features: int = 39, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        self.expert_front = StructuredExpertSubNetV12(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_rear = StructuredExpertSubNetV12(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_kinetic = StructuredExpertSubNetV12(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

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


class MoEConsistencyLossV12_1(nn.Module):
    """
    vidio-health-trainer v12.1 학습 손실
    - 16-State Weighted CE Loss
    - Reverse-ASL (False Clean 억제, lambda_fc=3.0)
    - 4-Way Marginal BCE Loss
    - Consistency Loss (P16 @ M vs Sigmoid(L4))
    - Gate Balance Regularizer
    """
    def __init__(
        self,
        weight_ce: float = 0.50,
        weight_clean: float = 0.20,
        weight_bce: float = 0.20,
        weight_consistency: float = 0.05,
        weight_gate_balance: float = 0.05,
        lambda_fc: float = 3.0,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        class_weights_16: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = weight_ce
        self.w_clean = weight_clean
        self.w_bce = weight_bce
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
        loss_ce = self.ce_loss(logits_16, targets_16)

        p16 = torch.softmax(logits_16, dim=-1)
        p_clean = p16[:, 0].clamp(1e-7, 1.0 - 1e-7)
        y = y_clean.float()
        loss_pos = -y * (1.0 - p_clean).pow(self.gamma_pos) * torch.log(p_clean)
        loss_neg = -self.lambda_fc * (1.0 - y) * p_clean.pow(self.gamma_neg) * torch.log(1.0 - p_clean)
        loss_fc = (loss_pos + loss_neg).mean()

        loss_bce = self.bce_loss(logits_4, targets_4)

        p4_from_16 = torch.matmul(p16, self.M_matrix)
        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4_from_16, p4_from_bce)

        mean_gates = torch.mean(gate_weights, dim=0)
        uniform = torch.full_like(mean_gates, 1.0 / 3.0)
        loss_gate = F.kl_div(mean_gates.log(), uniform, reduction='sum')

        total = (
            self.w_ce   * loss_ce
            + self.w_clean * loss_fc
            + self.w_bce  * loss_bce
            + self.w_cons * loss_cons
            + self.w_gate * loss_gate
        )

        return total, {
            'loss_total':   float(total.item()),
            'loss_ce':      float(loss_ce.item()),
            'loss_fc':      float(loss_fc.item()),
            'loss_bce':     float(loss_bce.item()),
            'loss_cons':    float(loss_cons.item()),
            'loss_gate':    float(loss_gate.item())
        }
