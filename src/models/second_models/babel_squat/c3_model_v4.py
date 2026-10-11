# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c3. 발과 무릎의 방향 일치 (Knee-Toe Alignment / Knee Valgus)
Model Version: v4.0 (c3_model_v4.py)
Status: ★ 최종 확정 프로덕션 모델 (Conv-BiGRU-Attention Deep Temporal Net 17D)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 16채널 무릎-발 동역학 시계열 텐서 (N, 64, 16) 또는 단일 시퀀스 (64, 16)
   - 16채널 구성:
     * ch0, ch1: 좌/우 내측 외반 오차각 (X-Z 바닥 수평 평면 투영)
     * ch2: 최대 외반 오차각 max(err_l, err_r)
     * ch3: 좌우 비대칭도 |err_l - err_r|
     * ch4, ch5: 무릎 거리, 발목 거리
     * ch6, ch7: 무릎/발목 비, 기립 대비 무릎 폭 비율
     * ch8, ch9: 3D 좌/우 외반각 (대퇴골 vs 경골 3차원 내전)
     * ch10: 운동 위상 정규화 높이 (0.0 기립 ~ 1.0 최하사점)
     * ch11: 무릎 수직 속도 (하강/정지/상승 위상 판별)
     * ch12, ch13: 좌/우 외반 각속도 (급격한 내측 붕괴 감지)
     * ch14, ch15: 기립 대비 좌/우 외반 변위 증가량
2. 딥러닝 피처 추출 및 추론 아키텍처:
   - 1D-CNN (2-layer, filters=32, kernel=3, BatchNorm, ReLU, Dropout=0.1): 로컬 프레임 간 무릎 흔들림/붕괴 감지
   - Bi-GRU (hidden_dim=32, bidirectional=True): 하강->바닥->상승 전 구간 양방향 맥락 학습
   - Temporal Self-Attention: 운동 중 외반 피크 시점(하강기, 바닥, 상승기 스티킹 포인트)에 자율 가중치 부여
   - Projection Layer: 64D 어텐션 풀링 벡터를 16D 단일 잠재 피처로 압축
   - Classification Head: 16D -> 2 클래스 (소프트맥스 사후확률)
3. 5-Fold StratifiedGroupKFold 실측 성능:
   - Precision = 93.48% (v3.0 대비 +2.51%p 향상)
   - Recall = 91.82% (v3.0 대비 +0.95%p 향상)
   - F1-Score = 92.64% (v3.0 대비 +1.72%p 수직 상승)
   - ROC-AUC = 97.39% (v3.0 대비 +0.58%p 향상)
   - 기존 v3.0 미인식 결함 87건 중 50건(57.5%) 완벽 구제
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, 32)
        self.u_query = nn.Linear(32, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, hidden_dim)
        v = torch.tanh(self.w_proj(h))
        scores = self.u_query(v).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class C3TemporalEmbeddingNet(nn.Module):
    """
    운동 전 구간 64프레임의 무릎-발 정렬 궤적을 16차원 단일 요약 임베딩 벡터로 압축
    """
    def __init__(self, in_channels: int = 16, conv_dim: int = 32, gru_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(gru_dim * 2)
        self.projector = nn.Sequential(
            nn.Linear(gru_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 64, 16) -> permute to (B, 16, 64) for Conv1D
        x_c = x.permute(0, 2, 1)
        conv_out = self.conv(x_c)
        conv_out = conv_out.permute(0, 2, 1)

        gru_out, _ = self.gru(conv_out)
        context, alpha = self.attention(gru_out)
        embedding = self.projector(context)
        logits = self.head(embedding)
        return logits, embedding, alpha


class BabelSquatC3ModelV4:
    """
    바벨 스쿼트 c3 결함(무릎 외반/정렬 불일치) 확정 프로덕션 모델 v4.0
    5-Fold 딥러닝 앙상블 추론 엔진
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        device: str = 'cpu'
    ):
        self.decision_threshold = decision_threshold
        self.device = torch.device(device)
        self.models: List[C3TemporalEmbeddingNet] = []
        self.norm_means: List[np.ndarray] = []
        self.norm_stds: List[np.ndarray] = []
        self.is_fitted = False

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c3_model_v4_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        입력: (N, 64, 16) 또는 (64, 16)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.models) == 0:
            raise RuntimeError("C3 v4 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]
            single = True

        N = X_seq.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for model, mean, std in zip(self.models, self.norm_means, self.norm_stds):
            model.eval()
            X_norm = (X_seq - mean) / (std + 1e-6)
            tensor_x = torch.from_numpy(X_norm).float().to(self.device)
            with torch.no_grad():
                logits, _, _ = model(tensor_x)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                ensemble_probs += probs / len(self.models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_seq)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def extract_embedding(self, X_seq: np.ndarray) -> np.ndarray:
        """16차원 단일 요약 임베딩 벡터 추출"""
        if not self.is_fitted or len(self.models) == 0:
            raise RuntimeError("C3 v4 모델 가중치가 로드되지 않았습니다.")
        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]

        embeddings = np.zeros((X_seq.shape[0], 16), dtype=np.float32)
        for model, mean, std in zip(self.models, self.norm_means, self.norm_stds):
            model.eval()
            X_norm = (X_seq - mean) / (std + 1e-6)
            tensor_x = torch.from_numpy(X_norm).float().to(self.device)
            with torch.no_grad():
                _, emb, _ = model(tensor_x)
                embeddings += emb.cpu().numpy() / len(self.models)
        return embeddings

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state_dicts = [m.state_dict() for m in self.models]
        payload = {
            'state_dicts': state_dicts,
            'norm_means': self.norm_means,
            'norm_stds': self.norm_stds,
            'decision_threshold': self.decision_threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c3 v4.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        state_dicts = payload['state_dicts']
        self.norm_means = payload['norm_means']
        self.norm_stds = payload['norm_stds']
        self.decision_threshold = payload.get('decision_threshold', 0.50)

        self.models = []
        for sd in state_dicts:
            m = C3TemporalEmbeddingNet(in_channels=16, conv_dim=32, gru_dim=32, embed_dim=16).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)

        self.is_fitted = True
        return self


BabelSquatC3Model = BabelSquatC3ModelV4
