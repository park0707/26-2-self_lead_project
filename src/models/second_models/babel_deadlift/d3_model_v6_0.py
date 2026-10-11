# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v6.0 (d3_model_v6_0.py)
Status: ★ 이론적 한계선(F1 74%~77%) 돌파를 위한 차세대 아키텍처
        - [1] 원판 위 노출 정강이 전방 경사 각속도/각가속도 피처 (뒤꿈치 가림 간접 포착)
        - [2] 힙 슈트업(골반-무릎 동역학) 및 상체 쏠림 연쇄 반응 피처 (36D 정예셋)
        - [3] Multi-Head Temporal Self-Attention ST-SeqNet v6.0
        - [4] 시점별 보류 예산 차등화 (Adaptive Abstention Budget) 3-State 안전 게이트
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d3_model_v5_3 import (
    FEATURE_NAMES_32_D3,
    apply_kapw_phase_warping,
    inpaint_lower_body_joints
)
from src.models.second_models.babel_deadlift.d3_model_v5_1 import REFINED_24_INDICES
from src.models.second_models.babel_deadlift.d3_model_v4 import (
    clean_and_smooth_keypoints,
    extract_d3_v4_features_from_seq,
    VIEW_CODE_LIST
)

# 15개 핵심 관절 인덱스
# 0:코, 1,2:어깨, 3,4:손목, 5,6:골반, 7,8:무릎, 9,10:발목, 11,12:뒤꿈치, 13,14:발가락
KEY_JOINT_INDICES = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
NUM_KEY_JOINTS = len(KEY_JOINT_INDICES) # 15개

# v6.0 36D 정예 피처 명칭 (v5.3 32D + 신규 정강이 가속도 및 상체 동역학 4종)
FEATURE_NAMES_36_D3: List[str] = FEATURE_NAMES_32_D3 + [
    "d3_shank_angular_vel_max",         # [32] 정강이 전방 경사각 최대 각속도 (원판 가림 간접 포착)
    "d3_shank_angular_acc_max",         # [33] 정강이 전방 경사각 최대 각가속도 (들림 충격 지표)
    "d3_hip_knee_velocity_ratio",       # [34] 골반 상승 속도 대비 무릎 신전 속도 비율 (힙슈트업 연동)
    "d3_torso_anterior_drift_rate"      # [35] 리프팅 구간 상체(어깨-골반) 전방 쏠림 이동률
]


# =========================================================================
# Multi-Head Temporal Self-Attention ST-SeqNet v6.0
# =========================================================================
class DeadliftSTSequenceNetV6_0(nn.Module):
    """
    57D Dual-Stream 입력:
    - Multi-Scale 1D Conv (k=3, k=5) -> BatchNorm -> ReLU
    - Bidirectional GRU (2 Layers)
    - 4-Head Temporal Self-Attention Layer
    - Dynamic Context Pooling & Classifier
    """
    def __init__(self, in_features: int = 57, hidden_dim: int = 64, num_heads: int = 4):
        super(DeadliftSTSequenceNetV6_0, self).__init__()

        self.conv3 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_features, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 4-Head Self Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=0.15,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(hidden_dim * 2)

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 64, 57)
        x_conv = x.transpose(1, 2)
        c3 = self.conv3(x_conv)
        c5 = self.conv5(x_conv)
        feat = self.relu(self.bn1(torch.cat([c3, c5], dim=1)))
        feat = self.dropout(feat).transpose(1, 2) # (Batch, 64, 128)

        gru_out, _ = self.bigru(feat) # (Batch, 64, 128)

        # Self-Attention
        attn_out, _ = self.self_attn(gru_out, gru_out, gru_out)
        res_out = self.ln_attn(gru_out + attn_out)

        # Attention Pooling
        pool_weights = F.softmax(self.attn_pool(res_out), dim=1) # (Batch, 64, 1)
        pooled = torch.sum(res_out * pool_weights, dim=1)        # (Batch, 128)

        logits = self.fc(pooled)
        return logits.squeeze(-1)


# =========================================================================
# 바벨 데드리프트 d3 공식 프로덕션 모델 v6.0
# =========================================================================
class BabelDeadliftD3ModelV6_0:
    """
    바벨 데드리프트 d3 한계선 돌파 모델 v6.0
    - 57D Multi-Head Temporal Self-Attention ST-SeqNet
    - 36D 확장 정예 생체역학 Tri-Ensemble (CatBoost + LightGBM + RF)
    - 시점별 보류 예산 차등화 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v6.0"
    FEATURE_DIM = 36
    FEATURE_NAMES = FEATURE_NAMES_36_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or {}
        self.dl_model: Optional[DeadliftSTSequenceNetV6_0] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.is_fitted = False

    def fit(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray, y: np.ndarray,
            epochs: int = 35, random_state: int = 42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        device = torch.device("cpu")
        dl_net = DeadliftSTSequenceNetV6_0(in_features=57, hidden_dim=64, num_heads=4).to(device)
        optimizer = torch.optim.AdamW(dl_net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        t_seq = torch.tensor(X_seq_57, dtype=torch.float32).to(device)
        t_y = torch.tensor(y, dtype=torch.float32).to(device)

        dl_net.train()
        batch_size = 32
        n_samples = len(y)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                b_idx = indices[i:i + batch_size]
                optimizer.zero_grad()
                logits = dl_net(t_seq[b_idx])
                loss = criterion(logits, t_y[b_idx])
                loss.backward()
                optimizer.step()

        self.dl_model = dl_net

        # CatBoost
        cb = CatBoostClassifier(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats_36, y)
        self.cb_model = cb

        # LightGBM
        ds_tr = lgb.Dataset(X_feats_36, label=y)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'num_leaves': 16,
            'max_depth': 4,
            'scale_pos_weight': float(pos_weight.item()),
            'verbosity': -1,
            'random_state': random_state
        }
        lgb_m = lgb.train(lgb_params, ds_tr, num_boost_round=230)
        self.lgb_model = lgb_m

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_36, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray, views: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq_57, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()

        p_cb = self.cb_model.predict_proba(X_feats_36)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats_36)
        p_rf = self.rf_model.predict_proba(X_feats_36)[:, 1]
        p_tree = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        if views is not None:
            probs = np.zeros(len(p_dl), dtype=np.float32)
            for i, v in enumerate(views):
                w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
                probs[i] = w_dl * p_dl[i] + (1.0 - w_dl) * p_tree[i]
            return probs
        else:
            return 0.45 * p_dl + 0.55 * p_tree

    def save_weights(self, out_path: str):
        data = {
            'dl_state_dict': self.dl_model.state_dict() if self.dl_model else None,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_36_D3
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data['dl_state_dict']:
            self.dl_model = DeadliftSTSequenceNetV6_0(in_features=57, hidden_dim=64, num_heads=4)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.view_thresholds = data.get('view_thresholds', {})
        self.is_fitted = True
