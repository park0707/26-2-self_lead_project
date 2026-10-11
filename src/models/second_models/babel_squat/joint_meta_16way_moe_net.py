# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: joint_meta_16way_moe_net.py
Purpose: 해결책 6 - View-Conditioned Mixture-of-Experts (MoE) 16-Way 메타 신경망
         - 3대 전문 서브 네트워크 (Front/Oblique, Rear, Kinetic Chain)
         - 1대 시점 및 품질 인지형 게이팅 네트워크 (Gating Network)
         - 16-Way Label Powerset Softmax + 4-Way Marginal Sigmoid + 일관성 손실
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


class ExpertSubNet(nn.Module):
    """개별 전문가 서브 네트워크"""
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
        self.head_16 = nn.Linear(hidden_dim // 2, 16)
        self.head_4 = nn.Linear(hidden_dim // 2, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.head_16(h), self.head_4(h)


class JointMeta16WayMoENet(nn.Module):
    """
    39D 입력 기반 View-Conditioned Mixture-of-Experts (MoE) 메타러너
    - Expert 1: Front / Oblique 전담 (시점 B, C, D 특화)
    - Expert 2: Rear 전담 (시점 A, E 특화)
    - Expert 3: Kinetic Chain & 전신 동역학 전담
    - Gating Net: 시점(v), 관측품질(q), 확률(x)을 조건으로 전문가 가중치 g_m 동적 산출
    """
    def __init__(self, in_features: int = 39, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        # 3대 서브 전문가
        self.expert_front = ExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_rear = ExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_kinetic = ExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        # 게이팅 네트워크 (Gating Network)
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 3)
        )

        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_l16: (Batch, 16) 결합 상태 로짓
            fused_l4:  (Batch, 4)  개별 결함 보조 로짓
            gate_weights: (Batch, 3) 전문가별 가중치 [g_front, g_rear, g_kinetic]
        """
        # 1. 3대 전문가 개별 로짓 산출
        l16_front, l4_front = self.expert_front(x)
        l16_rear, l4_rear = self.expert_rear(x)
        l16_kin, l4_kin = self.expert_kinetic(x)

        # 2. 게이팅 가중치 산출 (Softmax over 3 experts)
        gate_logits = self.gate_net(x)
        gate_weights = torch.softmax(gate_logits, dim=-1) # (Batch, 3)

        # 3. 전문가 로짓 동적 가중합
        g1 = gate_weights[:, 0:1]
        g2 = gate_weights[:, 1:2]
        g3 = gate_weights[:, 2:3]

        fused_l16 = g1 * l16_front + g2 * l16_rear + g3 * l16_kin
        fused_l4 = g1 * l4_front + g2 * l4_rear + g3 * l4_kin

        return fused_l16, fused_l4, gate_weights


class MoEConsistencyLoss(nn.Module):
    """
    MoE 메타러너 전용 멀티태스크 + 게이트 밸런싱 정규화 손실
    """
    def __init__(
        self,
        weight_ce: float = 0.55,
        weight_bce: float = 0.25,
        weight_consistency: float = 0.10,
        weight_gate_balance: float = 0.10,
        class_weights_16: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = weight_ce
        self.w_bce = weight_bce
        self.w_cons = weight_consistency
        self.w_gate = weight_gate_balance
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_16)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(
        self,
        logits_16: torch.Tensor,
        logits_4: torch.Tensor,
        targets_16: torch.Tensor,
        targets_4: torch.Tensor,
        gate_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. CE Loss
        loss_ce = self.ce_loss(logits_16, targets_16)

        # 2. BCE Loss
        loss_bce = self.bce_loss(logits_4, targets_4)

        # 3. Consistency Loss
        p16 = torch.softmax(logits_16, dim=-1)
        p4_from_16 = torch.matmul(p16, self.M_matrix)
        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4_from_16, p4_from_bce)

        # 4. Gate Balance Regularizer (KL Divergence to Uniform to prevent gate collapse)
        mean_gates = torch.mean(gate_weights, dim=0) # (3,)
        uniform = torch.full_like(mean_gates, 1.0 / 3.0)
        loss_gate = F.kl_div(mean_gates.log(), uniform, reduction='sum')

        total = self.w_ce * loss_ce + self.w_bce * loss_bce + self.w_cons * loss_cons + self.w_gate * loss_gate

        return total, {
            'loss_total': float(total.item()),
            'loss_ce': float(loss_ce.item()),
            'loss_bce': float(loss_bce.item()),
            'loss_cons': float(loss_cons.item()),
            'loss_gate': float(loss_gate.item())
        }


class FalseCleanAsymmetricLoss(nn.Module):
    """
    vidio-health-trainer v10.0
    Reverse-ASL: 결함 샘플을 Clean으로 오판하는 False Clean에 비대칭 강력 패널티

    y=1 (진짜 Clean):  -alpha * (1-p)^gamma_pos * log(p)          쉬운 Clean 기여도 축소
    y=0 (진짜 Defect): -lambda_fc * p^gamma_neg * log(1-p)        결함->Clean 오판 강력 패널티
    """
    def __init__(
        self,
        lambda_fc: float = 3.0,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        alpha: float = 1.0,
        eps: float = 1e-7
    ):
        super().__init__()
        self.lambda_fc = lambda_fc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.eps = eps

    def forward(self, clean_logit: torch.Tensor, y_clean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clean_logit: (B,) - 16-state 로짓 중 state=0(Clean)의 로짓 값
            y_clean    : (B,) - 실제 Clean 여부 (0 or 1, float)
        """
        p = torch.sigmoid(clean_logit).clamp(self.eps, 1.0 - self.eps)
        y = y_clean.float()
        loss_pos = -self.alpha * y * (1.0 - p).pow(self.gamma_pos) * torch.log(p)
        loss_neg = -self.lambda_fc * (1.0 - y) * p.pow(self.gamma_neg) * torch.log(1.0 - p)
        return (loss_pos + loss_neg).mean()


class MoEConsistencyLossV10(nn.Module):
    """
    vidio-health-trainer v10.0 학습 손실
    기존 MoEConsistencyLoss + FalseCleanAsymmetricLoss 추가

    L_total = w_ce   * L_CE(16상태 가중 CE)
            + w_clean * L_FalseClean(Reverse-ASL)    <- 신규: False Clean 비대칭 패널티
            + w_bce   * L_BCE(c1~c4 개별)
            + w_cons  * L_Consistency
            + w_gate  * L_GateBalance
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
        self.fc_loss = FalseCleanAsymmetricLoss(
            lambda_fc=lambda_fc, gamma_pos=gamma_pos, gamma_neg=gamma_neg
        )
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
        # 1. 16-state CE Loss
        loss_ce = self.ce_loss(logits_16, targets_16)

        # 2. False-Clean Asymmetric Loss (Reverse-ASL)
        # logits_16[:, 0] = state=0(Clean)의 raw 로짓
        loss_fc = self.fc_loss(logits_16[:, 0], y_clean)

        # 3. 4-Way Marginal BCE
        loss_bce = self.bce_loss(logits_4, targets_4)

        # 4. Consistency Loss
        p16 = torch.softmax(logits_16, dim=-1)
        p4_from_16 = torch.matmul(p16, self.M_matrix)
        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4_from_16, p4_from_bce)

        # 5. Gate Balance Regularizer
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


# ==============================================================================
# [v10.1 신규] Exact 16-State Structured Energy Head & MoE Network
# ==============================================================================

class Structured16Head(nn.Module):
    """
    vidio-health-trainer v10.1
    Exact 16-State Energy-Based Structured Head
    - Unary Energy (4D): c1, c2, c3, c4 개별 결함 log-odds
    - Clean Energy (1D): 완전 정상 (State 0) baseline 에너지
    - Pairwise Interaction (6D): 6개 결함 쌍 상호작용 (v_kl)
    - Triple Kinetic Chain (1D): c1(척추) x c3(무릎) x c4(발) 운동사슬 상호작용 (r)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.clean_energy = nn.Linear(in_dim, 1)
        self.unary = nn.Linear(in_dim, 4)
        self.pair  = nn.Linear(in_dim, 6)
        self.triple = nn.Linear(in_dim, 1)
        self.register_buffer('states', torch.from_numpy(STATE_TO_DEFECTS_MAP))
        self.pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.unary(h)    # (B, 4)
        v = self.pair(h)     # (B, 6)
        r = self.triple(h)   # (B, 1)
        c0 = self.clean_energy(h) # (B, 1)

        # 1. Unary energy: u @ S^T -> (B, 16)
        energy = torch.matmul(u, self.states.t()).clone()

        # State 0 (Clean) gets clean_energy
        energy[:, 0:1] = energy[:, 0:1] + c0

        # 2. Pairwise interactions
        for e, (i, j) in enumerate(self.pairs):
            pair_mask = self.states[:, i] * self.states[:, j]
            energy = energy + v[:, e:e+1] * pair_mask.unsqueeze(0)

        # 3. Triple kinetic chain: c1(0) * c3(2) * c4(3)
        triple_mask = self.states[:, 0] * self.states[:, 2] * self.states[:, 3]
        energy = energy + r * triple_mask.unsqueeze(0)

        return energy, u


class StructuredExpertSubNet(nn.Module):
    """v10.1 구조화 에너지 헤드 장착 전문가 서브 네트워크"""
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
        self.energy_head = Structured16Head(in_dim=hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.energy_head(h)


class JointMeta16WayMoENetV10_1(nn.Module):
    """
    vidio-health-trainer v10.1
    View-Conditioned Mixture-of-Experts (MoE) + Exact 16-State Structured Energy Head
    - 3대 전문 서브 네트워크 (Front, Rear, Kinetic Chain) 모두 Structured16Head 탑재
    - 게이팅 네트워크가 에너지 로짓을 동적 가중합하여 최종 물리적 결합 에너지 생성
    """
    def __init__(self, in_features: int = 39, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        self.expert_front = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_rear = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_kinetic = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

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


class MoEConsistencyLossV10_1(nn.Module):
    """
    vidio-health-trainer v10.1 학습 손실
    - Softmax 기반 Reverse-ASL (Softmax Clean Probability P(Clean) 사용)
    - 16-State Weighted CE Loss
    - 4-Way Marginal BCE Loss
    - Consistency Loss
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
        # 1. 16-state CE Loss
        loss_ce = self.ce_loss(logits_16, targets_16)

        # 2. Softmax 기반 Reverse-ASL
        p16 = torch.softmax(logits_16, dim=-1)
        p_clean = p16[:, 0].clamp(1e-7, 1.0 - 1e-7)
        y = y_clean.float()
        loss_pos = -y * (1.0 - p_clean).pow(self.gamma_pos) * torch.log(p_clean)
        loss_neg = -self.lambda_fc * (1.0 - y) * p_clean.pow(self.gamma_neg) * torch.log(1.0 - p_clean)
        loss_fc = (loss_pos + loss_neg).mean()

        # 3. 4-Way Marginal BCE
        loss_bce = self.bce_loss(logits_4, targets_4)

        # 4. Consistency Loss
        p4_from_16 = torch.matmul(p16, self.M_matrix)
        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4_from_16, p4_from_bce)

        # 5. Gate Balance Regularizer
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
