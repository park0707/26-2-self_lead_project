# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for extreme class imbalance (1:17.0).
    - gamma_pos: 0.0 (소수 클래스 정상 자세는 온전히 학습)
    - gamma_neg: 4.0 (다수 클래스 오류 자세의 그래디언트 강한 감쇠)
    - clip: 0.05 (쉬운 오류 샘플의 손실을 완전히 0으로 차단)
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        
        # Positive (정상: 1)
        pos_probs = probs
        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1 - pos_probs) ** self.gamma_pos)

        # Negative (오류: 0)
        neg_probs = 1.0 - probs
        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)
        neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))
        if self.gamma_neg > 0:
            neg_loss = neg_loss * ((1 - neg_probs) ** self.gamma_neg)

        loss = - (pos_loss + neg_loss)
        return loss.mean()

class BinaryFocalLoss(nn.Module):
    """
    Standard Binary Focal Loss
    FL = -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss
        return loss.mean()
