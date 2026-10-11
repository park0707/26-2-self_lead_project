# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v5.0 (d3_model_v5.py)
Status: ★ 솔루션 3: 64프레임 시공간 시계열 딥러닝(ST-SeqNet) + 40D SOTA 생체역학 하이브리드 앙상블
        - [제약 1] 판단 보류율 최대 10% 이하 (커버리지 >= 90.0% 엄격 보장)
        - [제약 2] F1-Score 최우선 단독 최적화
        - [제약 4] 뒤꿈치 들림 연동 전신 전방 쏠림(Forward Lean & Barbell Drift) 동역학 반영
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
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix

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

from src.models.second_models.babel_deadlift.d3_model_v4 import (
    extract_d3_v4_features_from_seq,
    FEATURE_NAMES_40_D3,
    VIEW_CODE_LIST
)

# 핵심 관절 인덱스 (14개: 코, 양어깨, 양손목, 양골반, 양무릎, 양발목, 양뒤꿈치, 양발가락)
KEY_JOINT_INDICES = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
NUM_KEY_JOINTS = len(KEY_JOINT_INDICES) # 15개


# =========================================================================
# 1. 64프레임 시공간 시계열 딥러닝 신경망 (ST-SequenceNet)
# =========================================================================
class DeadliftSTSequenceNet(nn.Module):
    """
    64프레임 x 15개 핵심 관절 시공간 딥러닝 신경망
    - 다중 수용영역 1D Temporal Convolution (k=3, k=5)
    - Bidirectional GRU (양방향 시간 의존성 모델링)
    - Attention Temporal Pooling
    - F1 극대화를 위한 Focal/Weighted BCE 헤드
    """
    def __init__(self, in_features: int = NUM_KEY_JOINTS * 3, hidden_dim: int = 64):
        super(DeadliftSTSequenceNet, self).__init__()
        
        # 1. 시공간 컨볼루션 블록 (Multi-Scale Temporal Conv)
        self.conv1_3 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(in_features, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        # 2. 양방향 GRU (시간적 흐름 및 들림 위상 학습)
        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 3. Temporal Attention Pooling
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 4. 분류 헤드
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 64, in_features)
        x_conv = x.transpose(1, 2) # (Batch, in_features, 64)
        c3 = self.conv1_3(x_conv)
        c5 = self.conv1_5(x_conv)
        feat_conv = self.relu(self.bn1(torch.cat([c3, c5], dim=1))) # (Batch, hidden*2, 64)
        feat_conv = self.dropout(feat_conv).transpose(1, 2) # (Batch, 64, hidden*2)
        
        gru_out, _ = self.bigru(feat_conv) # (Batch, 64, hidden*2)
        
        # Attention Pooling
        attn_weights = F.softmax(self.attn(gru_out), dim=1) # (Batch, 64, 1)
        pooled = torch.sum(gru_out * attn_weights, dim=1) # (Batch, hidden*2)
        
        logits = self.fc(pooled) # (Batch, 1)
        return logits.squeeze(-1)


# =========================================================================
# 2. v5 하이브리드 아키텍처 (ST-SeqNet + 40D Tri-Ensemble)
# =========================================================================
class BabelDeadliftD3ModelV5:
    """
    바벨 데드리프트 d3 공식 프로덕션 모델 v5.0 (하이브리드 SOTA)
    - 64프레임 시공간 관절 시계열 딥러닝 신경망 (PyTorch ST-SeqNet)
    - 40D SOTA 생체역학 피처 Tri-Ensemble (CatBoost + LightGBM + RF)
    - 커버리지 >= 90.0% 엄격 준수 F1 최우선 시점별 적응형 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v5.0"
    
    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or {}
        self.dl_model: Optional[DeadliftSTSequenceNet] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.alpha_dl: float = 0.35 # 딥러닝 35% + 트리 65%
        self.is_fitted = False
        
    def fit(self, X_seq: np.ndarray, X_feats: np.ndarray, y: np.ndarray, epochs: int = 40, random_state: int = 42):
        # 1. PyTorch ST-SeqNet 학습
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        device = torch.device("cpu")
        dl_net = DeadliftSTSequenceNet(in_features=NUM_KEY_JOINTS * 3, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(dl_net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        t_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
        t_y = torch.tensor(y, dtype=torch.float32).to(device)
        
        dl_net.train()
        batch_size = 32
        n_samples = len(y)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                b_idx = indices[i:i+batch_size]
                optimizer.zero_grad()
                logits = dl_net(t_seq[b_idx])
                loss = criterion(logits, t_y[b_idx])
                loss.backward()
                optimizer.step()
                
        self.dl_model = dl_net
        
        # 2. CatBoost 학습 (F1 극대화 가중치)
        cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats, y)
        self.cb_model = cb
        
        # 3. LightGBM 학습
        pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))
        ds_tr = lgb.Dataset(X_feats, label=y)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'scale_pos_weight': pos_w,
            'verbosity': -1,
            'random_state': random_state
        }
        lgb_m = lgb.train(lgb_params, ds_tr, num_boost_round=220)
        self.lgb_model = lgb_m
        
        # 4. Random Forest 학습
        rf = RandomForestClassifier(n_estimators=180, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats, y)
        self.rf_model = rf
        
        self.is_fitted = True
        
    def predict_proba(self, X_seq: np.ndarray, X_feats: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        # 1. 딥러닝 확률
        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()
            
        # 2. 트리 앙상블 확률
        p_cb = self.cb_model.predict_proba(X_feats)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats)
        p_rf = self.rf_model.predict_proba(X_feats)[:, 1]
        p_tree = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf
        
        # 3. 하이브리드 블렌딩
        return self.alpha_dl * p_dl + (1.0 - self.alpha_dl) * p_tree
        
    def save_weights(self, out_path: str):
        data = {
            'dl_state_dict': self.dl_model.state_dict() if self.dl_model else None,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'alpha_dl': self.alpha_dl,
            'view_thresholds': self.view_thresholds
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data['dl_state_dict']:
            self.dl_model = DeadliftSTSequenceNet(in_features=NUM_KEY_JOINTS * 3, hidden_dim=64)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.alpha_dl = data.get('alpha_dl', 0.35)
        self.view_thresholds = data.get('view_thresholds', {})
        self.is_fitted = True
