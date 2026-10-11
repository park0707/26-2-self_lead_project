# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: joint_meta_16way_net.py
Purpose: 바벨 스쿼트 16-Way Joint Head (Label Powerset Softmax) + 일관성 손실(Consistency Loss) 듀얼 헤드 신경망
         (4개 세부 결함 c1~c4의 2^4 = 16개 결합 상태 및 완전 정상 s=0000 최적화 메타 모델)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

# 16개 결합 상태와 4개 이진 라벨(c1, c2, c3, c4) 간의 불리언 매핑 행렬 (16, 4)
# State 0:  0000 (완전 정상 Clean)
# State 1:  0001 (c4 결함)
# ...
# State 15: 1111 (c1, c2, c3, c4 전 결함)
STATE_TO_DEFECTS_MAP = np.zeros((16, 4), dtype=np.float32)
for s in range(16):
    STATE_TO_DEFECTS_MAP[s, 0] = (s & 8) >> 3  # c1
    STATE_TO_DEFECTS_MAP[s, 1] = (s & 4) >> 2  # c2
    STATE_TO_DEFECTS_MAP[s, 2] = (s & 2) >> 1  # c3
    STATE_TO_DEFECTS_MAP[s, 3] = (s & 1)       # c4


def defect_tuple_to_idx(c1: int, c2: int, c3: int, c4: int) -> int:
    """4개 이진 결함 플래그를 16개 상태 인덱스(0~15)로 변환"""
    return (int(c1) << 3) | (int(c2) << 2) | (int(c3) << 1) | int(c4)


def idx_to_defect_tuple(s: int) -> Tuple[int, int, int, int]:
    """16개 상태 인덱스(0~15)를 4개 이진 결함 (c1, c2, c3, c4) 튜플로 변환"""
    return (int(STATE_TO_DEFECTS_MAP[s, 0]),
            int(STATE_TO_DEFECTS_MAP[s, 1]),
            int(STATE_TO_DEFECTS_MAP[s, 2]),
            int(STATE_TO_DEFECTS_MAP[s, 3]))


class JointMeta16WayNet(nn.Module):
    """
    18D 메타 입력 특성 기반 16-Way Joint Softmax & 4-Way Marginal Sigmoid 듀얼 헤드 메타러너
    입력(18D):
      - 4대 서브모델 결함 확률 [p_c1, p_c2, p_c3, p_c4] (4D)
      - 불확실성 마진 [|p_c1 - 0.5|, ..., |p_c4 - 0.5|] (4D)
      - 공식 5대 시점 원핫 [시점 A, 시점 B, 시점 C, 시점 D, 시점 E] (5D)
      - Layer 3 물리/시간 안정성 피처 [toe_lift, wobble, ydelta, jerk, f0_rate] (5D)
    """

    def __init__(self, in_features: int = 18, hidden_dim: int = 64, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        # 공유 표현 인코더 (Shared Latent Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU()
        )

        # Head A: 16-Way Joint Head (2^4 결합 상태 로짓)
        self.head_joint_16 = nn.Linear(hidden_dim // 2, 16)

        # Head B: 4-Way Marginal Auxiliary Head (c1~c4 개별 라벨 보조 로짓)
        self.head_marginal_4 = nn.Linear(hidden_dim // 2, 4)

        # 상태 매핑 텐서 등록 (버퍼)
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, 18) 메타 특성 텐서
        Returns:
            logits_16: (Batch, 16) 결합 상태 로짓
            logits_4:  (Batch, 4)  개별 결함 보조 로짓
        """
        latent = self.encoder(x)
        logits_16 = self.head_joint_16(latent)
        logits_4 = self.head_marginal_4(latent)
        return logits_16, logits_4

    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """소프트맥스 및 시그모이드 확률 산출"""
        self.eval()
        with torch.no_grad():
            l16, l4 = self.forward(x)
            p16 = F.softmax(l16, dim=-1)
            p4 = torch.sigmoid(l4)
        return p16, p4


class MultiTaskConsistencyLoss(nn.Module):
    """
    16-Way Cross-Entropy + 4-Way Marginal BCE + Consistency Loss 결합 손실 함수
    """

    def __init__(
        self,
        weight_ce: float = 0.60,
        weight_bce: float = 0.30,
        weight_consistency: float = 0.10,
        class_weights_16: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = weight_ce
        self.w_bce = weight_bce
        self.w_cons = weight_consistency

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_16)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(
        self,
        logits_16: torch.Tensor,
        logits_4: torch.Tensor,
        target_16: torch.Tensor,
        target_4: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits_16: (Batch, 16)
            logits_4:  (Batch, 4)
            target_16: (Batch,) 정수 클래스 (0~15)
            target_4:  (Batch, 4) 실수형 라벨 (0.0 or 1.0)
        """
        loss_ce = self.ce_loss(logits_16, target_16)
        loss_bce = self.bce_loss(logits_4, target_4)

        # 일관성 손실: 16-way 확률에서 유도된 주변확률과 4-way sigmoid 확률 간의 BCE
        p_joint_16 = F.softmax(logits_16, dim=-1) # (B, 16)
        p_marginal_from_16 = torch.matmul(p_joint_16, self.M_matrix) # (B, 4)
        p_marginal_from_16 = torch.clamp(p_marginal_from_16, 1e-6, 1.0 - 1e-6)

        p_indep_4 = torch.sigmoid(logits_4) # (B, 4)

        loss_consistency = F.binary_cross_entropy(p_marginal_from_16, p_indep_4.detach())

        total_loss = self.w_ce * loss_ce + self.w_bce * loss_bce + self.w_cons * loss_consistency

        loss_dict = {
            'total_loss': float(total_loss.item()),
            'loss_ce': float(loss_ce.item()),
            'loss_bce': float(loss_bce.item()),
            'loss_consistency': float(loss_consistency.item())
        }
        return total_loss, loss_dict


def decode_hybrid(
    prob_16: np.ndarray,
    prob_4: np.ndarray,
    eta: float = 0.50,
    tau_clean: float = 0.40
) -> Tuple[int, Tuple[int, int, int, int], bool, float]:
    """
    하이브리드 디코딩 및 안전 게이트 적용 최종 상태 판정 함수
    
    score(s) = log(p_set(s)) + eta * sum_k [s_k * log(p_k) + (1 - s_k) * log(1 - p_k)]
    
    Returns:
        best_state: 16개 상태 인덱스 (0~15)
        defects_tuple: (c1, c2, c3, c4) 4개 결함 불리언 정수 (0 or 1)
        is_clean: 완전 정상 여부 (True if State 0)
        confidence: 선정된 상태의 신뢰도 점수
    """
    eps = 1e-7
    p16 = np.clip(prob_16, eps, 1.0 - eps)
    p4 = np.clip(prob_4, eps, 1.0 - eps)

    scores = np.zeros(16, dtype=np.float32)
    log_p16 = np.log(p16)
    log_p4 = np.log(p4)
    log_1_p4 = np.log(1.0 - p4)

    for s in range(16):
        m_s = STATE_TO_DEFECTS_MAP[s] # (4,)
        log_indep_s = np.sum(m_s * log_p4 + (1.0 - m_s) * log_1_p4)
        scores[s] = log_p16[s] + eta * log_indep_s

    # 1위 상태 선정
    best_s = int(np.argmax(scores))

    # Clean 안전 게이트: 1위가 State 0 (0000)이더라도 p16[0] < tau_clean이면 결함으로 전이
    if best_s == 0:
        if prob_16[0] < tau_clean:
            # 0000을 제외한 상태 중 2위 선택
            scores_no_clean = np.copy(scores)
            scores_no_clean[0] = -1e9
            best_s = int(np.argmax(scores_no_clean))

    defects = idx_to_defect_tuple(best_s)
    is_clean = (best_s == 0)
    conf = float(p16[best_s])

    return best_s, defects, is_clean, conf
