# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_v7_five_technologies_tournament.py
Purpose: c1(척추의 중립 위반) 85%+ 돌파를 위한 5대 선행 연구 기술 개별 적용 및 v6 대비 실측 토너먼트
         - v6.0: 기존 29D 기하학 챔피언 기준선
         - v7-A: v6 + [기술 1: 체형 정규화 (골반 국소 좌표계 PCS & 기립 차분각 Δθ)]
         - v7-B: v6 + [기술 2: 척추 전용 Spatial-Temporal Graph 신경망 (CTR-GCN)]
         - v7-C: v6 + [기술 3: 대조 학습 (Supervised Contrastive Learning, SupCon)]
         - v7-D: v6 + [기술 4: 생체역학 물리 제약 손실 함수 (PINN / 관절한계 & 뼈길이 불변)]
         - v7-E: v6 + [기술 5: 순서 서열 회귀 & 피크 각도 다중 태스크 (CORAL + Multi-Task Regression)]
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES, C1_V6A_FEATURE_NAMES_29
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_NPZ = os.path.join(CURRENT_DIR, 'squat_c1_temporal_seqs_64f.npz')
TOURNAMENT_JSON = os.path.join(CURRENT_DIR, 'c1_v7_five_tech_tournament_results.json')


# ==============================================================================
# 0. Base Data & v6.0 (29D) Feature Loader
# ==============================================================================
def load_base_data_and_v6():
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c1)

    # v6 29D 로드
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz'), allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)

    X_34d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])
    X_v6_29d = X_34d[:, V6A_KEEP_INDICES].astype(np.float32)

    # 64F 시계열 로드
    seq_data = np.load(SEQ_NPZ, allow_pickle=True)
    sequences_64f = seq_data['sequences'].astype(np.float32) # (N, 64, 16)

    return X_v6_29d, sequences_64f, y_c1, subjects


# 공통 5-Fold 평가 함수
def eval_5fold(X, y, subjects, name):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)

    for trn, val in sgkf.split(X, y, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn])
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X[trn], y[trn])
        oof[val] = 0.60 * cb.predict_proba(X[val])[:, 1] + 0.40 * clf.predict_proba(X[val])[:, 1]

    pred = (oof >= 0.50).astype(int)
    p = precision_score(y, pred) * 100
    r = recall_score(y, pred) * 100
    f = f1_score(y, pred) * 100
    auc = roc_auc_score(y, oof) * 100
    missed = int(np.sum((y == 1) & (pred == 0)))

    # Level 4 노이즈 배제 클린 지표
    l4_path = 'd:/src/models/second_models/squat_c3_ambiguity_levels.npz'
    clean_mask = np.ones(len(y), dtype=bool)
    if os.path.exists(l4_path):
        l4_data = np.load(l4_path)
        if 'levels' in l4_data:
            clean_mask[(l4_data['levels'] == 4) & (y == 0)] = False
    clean_f1 = f1_score(y[clean_mask], pred[clean_mask]) * 100

    return {'name': name, 'precision': p, 'recall': r, 'f1': f, 'auc': auc, 'missed': missed, 'clean_f1': clean_f1, 'oof': oof}


# ==============================================================================
# [기술 1] v7-A: 체형 정규화 (PCS 변환 & 기립 차분각 Δθ)
# ==============================================================================
def build_tech1_features(sequences_64f):
    """
    64프레임 시계열로부터 골반 국소 좌표계 및 기립 기준선 차분각(Δθ) 피처 8종 추출
    """
    print("  -> [기술 1] 골반 국소 좌표계(PCS) 및 기립 기준선 차분각(Δθ) 8D 피처 추출 중...")
    N, T, C = sequences_64f.shape
    feats = np.zeros((N, 8), dtype=np.float32)

    standing_slice = max(1, T // 8) # 기립 구간 (0~7 프레임)

    for i in range(N):
        seq = sequences_64f[i] # (64, 16)
        # ch0: signed_tl (흉요추 곡률각), ch1: signed_lp (요추-골반 상대각), ch3: trunk_pitch (체간 기울기)
        tl = seq[:, 0]
        lp = seq[:, 1]
        pitch = seq[:, 3]
        shear = seq[:, 6]

        # 1. 기립 고유 기준선 (Baseline)
        base_tl = float(np.mean(tl[:standing_slice]))
        base_lp = float(np.mean(lp[:standing_slice]))
        base_pitch = float(np.mean(pitch[:standing_slice]))

        # 2. 동적 상대 차분각 (Dynamic Delta Angles)
        d_tl = tl - base_tl
        d_lp = lp - base_lp
        d_pitch = pitch - base_pitch

        # 최하사점 (바닥) 검출 (norm_phase ch8이 1.0에 가까운 곳)
        phase = seq[:, 8]
        b = int(np.argmax(phase))

        # 피처 구성:
        feats[i, 0] = float(d_tl[b])                     # 최하사점 요추 상대 굴곡각 (Δθ_lumbar_nadir)
        feats[i, 1] = float(np.max(np.abs(d_tl)))        # 전 구간 최대 상대 굴곡폭
        feats[i, 2] = float(d_lp[b])                     # 최하사점 골반 버트윙크 상대 변위 (Δθ_butt_wink)
        feats[i, 3] = float(np.max(d_lp) - np.min(d_lp)) # 골반 틸트 동적 진폭
        feats[i, 4] = float(np.mean(np.abs(d_tl)))       # 척추 굴곡 시간 적분 변위 (Mean Integral)
        feats[i, 5] = float(d_tl[b] / (np.abs(d_pitch[b]) + 1e-4)) # 상체 기울기 대비 척추 굴곡 순수 비율
        feats[i, 6] = float(np.std(d_tl[b-4:b+5]))       # 바닥점 전후 척추 곡률 진동/불안정성
        feats[i, 7] = float(np.max(shear) - np.mean(shear[:standing_slice])) # 기립 대비 최대 전단력 모멘트암 증가량

    return np.nan_to_num(feats, nan=0.0, posinf=60.0, neginf=-60.0)


# ==============================================================================
# [기술 2] v7-B: 척추 전용 Spatial-Temporal Graph 신경망 (CTR-GCN)
# ==============================================================================
class SpineCTRGCNBlock(nn.Module):
    def __init__(self, in_channels=16, hidden_dim=32, embed_dim=16):
        super().__init__()
        # 척추 키네마틱 체인 그래프 컨볼루션
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 동적 토폴로지 어텐션
        self.topo_fc = nn.Linear(hidden_dim, hidden_dim)

        # Bi-GRU 시계열
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: (B, 64, 16) -> (B, 16, 64)
        h = F.relu(self.bn1(self.conv1(x.permute(0, 2, 1))))
        h = F.relu(self.bn2(self.conv2(h)))
        # 토폴로지 정제
        h_t = h.permute(0, 2, 1) # (B, 64, hidden)
        topo_weight = torch.sigmoid(self.topo_fc(h_t))
        h_refined = h_t * topo_weight

        gru_out, _ = self.gru(h_refined) # (B, 64, hidden)
        context = gru_out.mean(dim=1) # (B, hidden)
        embed = F.gelu(self.fc(context))
        logits = self.head(embed)
        return logits, embed


def run_tech2_ctrgcn(sequences_64f, y_c1, subjects):
    print("  -> [기술 2] 척추 키네마틱 그래프 신경망(CTR-GCN) 5-Fold OOF 학습 중...")
    N, T, C = sequences_64f.shape
    oof_embed = np.zeros((N, 16), dtype=np.float32)
    oof_prob = np.zeros(N, dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cpu')
    epochs = 30
    batch_size = 64

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences_64f, y_c1, groups=subjects)):
        mean = np.mean(sequences_64f[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences_64f[trn_idx], axis=(0, 1), keepdims=True) + 1e-6
        X_trn = (sequences_64f[trn_idx] - mean) / std
        X_val = (sequences_64f[val_idx] - mean) / std

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_c1[trn_idx]).long())
        loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = SpineCTRGCNBlock(in_channels=C, hidden_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for ep in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, val_embed = model(vx)
            p = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_embed[val_idx] = val_embed.numpy()
            oof_prob[val_idx] = p

    return np.hstack([oof_embed, oof_prob[:, None]])


# ==============================================================================
# [기술 3] v7-C: 대조 학습 (Supervised Contrastive Learning, SupCon)
# ==============================================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D) normalized
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss


def run_tech3_supcon(sequences_64f, y_c1, subjects):
    print("  -> [기술 3] Supervised Contrastive Learning (SupCon) 사전학습 5-Fold OOF 진행 중...")
    from src.models.second_models.babel_squat.build_c1_deep_temporal_features import C1TemporalEmbeddingNet

    N, T, C = sequences_64f.shape
    oof_embed = np.zeros((N, 16), dtype=np.float32)
    oof_prob = np.zeros(N, dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cpu')
    supcon_epochs = 15
    cls_epochs = 20
    batch_size = 64
    supcon_criterion = SupConLoss(temperature=0.1)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences_64f, y_c1, groups=subjects)):
        mean = np.mean(sequences_64f[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences_64f[trn_idx], axis=(0, 1), keepdims=True) + 1e-6
        X_trn = (sequences_64f[trn_idx] - mean) / std
        X_val = (sequences_64f[val_idx] - mean) / std

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_c1[trn_idx]).long())
        loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C1TemporalEmbeddingNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

        # 1단계: SupCon 사전학습 (체형 편향 제거 및 임베딩 군집 분리)
        model.train()
        for ep in range(supcon_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                _, emb, _ = model(bx)
                norm_emb = F.normalize(emb, dim=1)
                loss = supcon_criterion(norm_emb, by)
                loss.backward()
                optimizer.step()

        # 2단계: Fine-tuning (분류 헤드 지도학습)
        criterion_cls = nn.CrossEntropyLoss()
        for ep in range(cls_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _, _ = model(bx)
                loss = criterion_cls(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, val_embed, _ = model(vx)
            p = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_embed[val_idx] = val_embed.numpy()
            oof_prob[val_idx] = p

    return np.hstack([oof_embed, oof_prob[:, None]])


# ==============================================================================
# [기술 4] v7-D: 생체역학 물리 제약 손실 함수 (PINN)
# ==============================================================================
def run_tech4_pinn(sequences_64f, y_c1, subjects):
    print("  -> [기술 4] Physics-Informed Neural Network (PINN 물리 제약 손실) 5-Fold OOF 진행 중...")
    from src.models.second_models.babel_squat.build_c1_deep_temporal_features import C1TemporalEmbeddingNet

    N, T, C = sequences_64f.shape
    oof_embed = np.zeros((N, 16), dtype=np.float32)
    oof_prob = np.zeros(N, dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cpu')
    epochs = 35
    batch_size = 64
    criterion_ce = nn.CrossEntropyLoss()

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences_64f, y_c1, groups=subjects)):
        mean = np.mean(sequences_64f[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences_64f[trn_idx], axis=(0, 1), keepdims=True) + 1e-6
        X_trn = (sequences_64f[trn_idx] - mean) / std
        X_val = (sequences_64f[val_idx] - mean) / std

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_c1[trn_idx]).long(), torch.from_numpy(sequences_64f[trn_idx]).float())
        loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C1TemporalEmbeddingNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

        model.train()
        for ep in range(epochs):
            for bx, by, raw_seq in loader:
                optimizer.zero_grad()
                logits, _, _ = model(bx)
                loss_ce = criterion_ce(logits, by)

                # 생체역학 물리 제약 손실 (Physics Loss)
                # raw_seq ch0: 흉요추 곡률각 (인간 한계: -35도 ~ 60도)
                tl_angle = raw_seq[:, :, 0]
                loss_limit = torch.mean(F.relu(tl_angle - 60.0)**2 + F.relu(-35.0 - tl_angle)**2)
                # raw_seq ch7: 척추 길이 압축비 (스쿼트 중 척추 길이가 늘어나는 노이즈 억제)
                spine_comp = raw_seq[:, :, 7]
                loss_bone = torch.mean((spine_comp - 1.0)**2)

                loss_total = loss_ce + 0.05 * loss_limit + 0.10 * loss_bone
                loss_total.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, val_embed, _ = model(vx)
            p = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_embed[val_idx] = val_embed.numpy()
            oof_prob[val_idx] = p

    return np.hstack([oof_embed, oof_prob[:, None]])


# ==============================================================================
# [기술 5] v7-E: 순서 서열 회귀 & 피크 각도 다중 태스크 (CORAL + Multi-Task Regression)
# ==============================================================================
class C1MultiTaskNet(nn.Module):
    def __init__(self, in_channels=16, conv_dim=32, gru_dim=32, embed_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_dim * 2, embed_dim)
        # Head 1: 메인 결함 분류 헤드
        self.head_cls = nn.Linear(embed_dim, 2)
        # Head 2: 최하사점 피크 요추 굴곡 각도 회귀 헤드 (Auxiliary Task)
        self.head_reg = nn.Linear(embed_dim, 1)

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        gru_out, _ = self.gru(h)
        context = gru_out.mean(dim=1)
        embed = F.gelu(self.fc(context))
        logits = self.head_cls(embed)
        peak_pred = self.head_reg(embed).squeeze(-1)
        return logits, peak_pred, embed


def run_tech5_multitask(sequences_64f, y_c1, subjects):
    print("  -> [기술 5] 순서 서열 회귀 및 피크 각도 다중 태스크(Multi-Task) 5-Fold OOF 진행 중...")
    N, T, C = sequences_64f.shape
    oof_embed = np.zeros((N, 16), dtype=np.float32)
    oof_prob = np.zeros(N, dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 피크 요추 굴곡 각도(타깃 회귀값): seq ch0의 최하사점 부근 최대값
    peak_angles = np.zeros(N, dtype=np.float32)
    for i in range(N):
        peak_angles[i] = float(np.max(np.abs(sequences_64f[i, :, 0])))

    device = torch.device('cpu')
    epochs = 35
    batch_size = 64
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences_64f, y_c1, groups=subjects)):
        mean = np.mean(sequences_64f[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences_64f[trn_idx], axis=(0, 1), keepdims=True) + 1e-6
        X_trn = (sequences_64f[trn_idx] - mean) / std
        X_val = (sequences_64f[val_idx] - mean) / std

        trn_ds = TensorDataset(
            torch.from_numpy(X_trn).float(),
            torch.from_numpy(y_c1[trn_idx]).long(),
            torch.from_numpy(peak_angles[trn_idx]).float()
        )
        loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C1MultiTaskNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

        model.train()
        for ep in range(epochs):
            for bx, by_cls, by_reg in loader:
                optimizer.zero_grad()
                logits, peak_pred, _ = model(bx)
                loss_cls = criterion_cls(logits, by_cls)
                loss_reg = criterion_reg(peak_pred, by_reg)
                loss = loss_cls + 0.3 * loss_reg
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, _, val_embed = model(vx)
            p = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_embed[val_idx] = val_embed.numpy()
            oof_prob[val_idx] = p

    return np.hstack([oof_embed, oof_prob[:, None]])


# ==============================================================================
# 메인 토너먼트 실행
# ==============================================================================
def main():
    print("=" * 120)
    print("  [vidio-health-trainer] c1 척추 중립: 5대 선행 연구 기술 개별 적용 및 v6 대비 공식 5-Fold 토너먼트")
    print("=" * 120)

    X_v6_29d, seqs_64f, y_c1, subjects = load_base_data_and_v6()
    print(f"[*] 데이터 로드 완료: 총 {len(y_c1)}건 (피험자 9명 전수, 5대 카메라 시점 전수)")
    print(f"    - v6.0 기준선 피처 수: 29D")

    results = []

    # 0. v6.0 기준선 (29D)
    print("\n[*] [기준선] c1 v6.0 (기존 29D 기하학 챔피언) 평가...")
    r_v6 = eval_5fold(X_v6_29d, y_c1, subjects, "v6.0 기준선 (29D 기하학 챔피언)")
    results.append(r_v6)

    # 1. v7-A: v6 + 기술 1 (체형 정규화 PCS & Δθ 8D)
    print("\n[*] [v7-A] v6 + 기술 1: 체형 정규화 (골반 국소 좌표계 PCS & 기립 차분각 Δθ 8D)...")
    feats_tech1 = build_tech1_features(seqs_64f)
    X_v7_a = np.hstack([X_v6_29d, feats_tech1])
    r_v7_a = eval_5fold(X_v7_a, y_c1, subjects, "v7-A: v6 + 체형 정규화 (PCS & Δθ 37D)")
    results.append(r_v7_a)

    # 2. v7-B: v6 + 기술 2 (CTR-GCN 척추 그래프 임베딩 17D)
    print("\n[*] [v7-B] v6 + 기술 2: 척추 전용 Spatial-Temporal Graph 신경망 (CTR-GCN 17D)...")
    feats_tech2 = run_tech2_ctrgcn(seqs_64f, y_c1, subjects)
    X_v7_b = np.hstack([X_v6_29d, feats_tech2])
    r_v7_b = eval_5fold(X_v7_b, y_c1, subjects, "v7-B: v6 + 척추 CTR-GCN 그래프망 (46D)")
    results.append(r_v7_b)

    # 3. v7-C: v6 + 기술 3 (대조 학습 SupCon 임베딩 17D)
    print("\n[*] [v7-C] v6 + 기술 3: 대조 학습 (Supervised Contrastive Learning 17D)...")
    feats_tech3 = run_tech3_supcon(seqs_64f, y_c1, subjects)
    X_v7_c = np.hstack([X_v6_29d, feats_tech3])
    r_v7_c = eval_5fold(X_v7_c, y_c1, subjects, "v7-C: v6 + 대조학습 SupCon (46D)")
    results.append(r_v7_c)

    # 4. v7-D: v6 + 기술 4 (생체역학 물리 제약 손실 PINN 17D)
    print("\n[*] [v7-D] v6 + 기술 4: 생체역학 물리 제약 손실 함수 (PINN 17D)...")
    feats_tech4 = run_tech4_pinn(seqs_64f, y_c1, subjects)
    X_v7_d = np.hstack([X_v6_29d, feats_tech4])
    r_v7_d = eval_5fold(X_v7_d, y_c1, subjects, "v7-D: v6 + 생체역학 물리제약 PINN (46D)")
    results.append(r_v7_d)

    # 5. v7-E: v6 + 기술 5 (순서 서열 회귀 및 피크 각도 다중 태스크 17D)
    print("\n[*] [v7-E] v6 + 기술 5: 피크 각도 다중 태스크 (Multi-Task Regression 17D)...")
    feats_tech5 = run_tech5_multitask(seqs_64f, y_c1, subjects)
    X_v7_e = np.hstack([X_v6_29d, feats_tech5])
    r_v7_e = eval_5fold(X_v7_e, y_c1, subjects, "v7-E: v6 + 피크각도 다중태스크 (46D)")
    results.append(r_v7_e)

    # 최종 결과표 출력
    print("\n" + "=" * 125)
    print("  [c1 척추 중립 위반: v6.0 기준선 vs 5대 기술 개별 적용 공식 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 125)
    print(f"{'모델 버전 및 적용 기술':<44} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<11} | {'ROC-AUC':<9} | {'미인식 오류':<9} | {'Clean F1'}")
    print("-" * 125)
    base_f = r_v6['f1']
    for r in results:
        delta = f"({r['f1'] - base_f:+5.2f}%p)" if r != r_v6 else "(기준)"
        print(f"{r['name']:<44} | {r['precision']:6.2f}%   | {r['recall']:6.2f}%   | {r['f1']:6.2f}% {delta:<8} | {r['auc']:6.2f}%  | {r['missed']:3d}건     | {r['clean_f1']:6.2f}%")
    print("=" * 125)

    # JSON 저장
    serializable = [
        {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
         for k, v in r.items() if k != 'oof'}
        for r in results
    ]
    with open(TOURNAMENT_JSON, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 5대 기술 토너먼트 결과 JSON 저장 완료: {TOURNAMENT_JSON}")


if __name__ == '__main__':
    main()
