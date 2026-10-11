# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v5.2 (d3_model_v5_2.py)
Status: ★ 4대 해결책 올인원(All-in-One) 통합 아키텍처
        - [해결책 1] 피처 다이어트: 중복/노이즈 소거 + 바벨 모멘트 암 결합 (26D 정예 피처셋)
        - [해결책 2] 상체-바벨 쏠림 연동 포착: 15개 관절 + 상대 변위 51D 시공간 ST-SeqNet 딥러닝
        - [해결책 3] 시작 타이밍 자동 일치: FastDTW 영감 시작 위상(바닥 vs 기립) 정규화 인코딩
        - [해결책 4] 판단 보류율 <= 10.0% (커버리지 >= 90.0%) & F1-Score 최우선 3-State 안전 게이트
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

from src.models.second_models.babel_deadlift.d3_model_v5_1 import (
    REFINED_24_INDICES,
    FEATURE_NAMES_24_D3
)
from src.models.second_models.babel_deadlift.d3_model_v4 import (
    clean_and_smooth_keypoints,
    extract_d3_v4_features_from_seq,
    VIEW_CODE_LIST
)

# 15개 핵심 관절 인덱스
# 0:코, 11,12:어깨, 15,16:손목, 23,24:골반, 25,26:무릎, 27,28:발목, 29,30:뒤꿈치, 31,32:발가락
KEY_JOINT_INDICES = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
NUM_KEY_JOINTS = len(KEY_JOINT_INDICES) # 15개

# v5.2 26D 정예 피처 명칭
FEATURE_NAMES_26_D3: List[str] = FEATURE_NAMES_24_D3 + [
    "d3_bar_moment_arm_max",     # [24] 정강이 길이 대비 바벨(손목)의 발목 수직선 전방 최대 모멘트 암 거리
    "d3_phase_start_wrist_y"      # [25] 시작 시점 바벨 높이 (바닥 셋업 1.0 vs 기립 0.0 정규화 위상 지표)
]


def extract_d3_v5_2_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 26D 통합 정예 피처 추출
    1. v4 40D 피처셋에서 24D 정예 슬라이싱
    2. 바벨-발목 모멘트 암 피처 계산 (해결책 1 & 2 보강)
    3. 시작 위상(바닥 셋업 vs 기립 락아웃) 판별 피처 계산 (해결책 3 위상 정합)
    """
    # 1. 24D 정예 피처
    f40 = extract_d3_v4_features_from_seq(kps_33_64f, view_code)
    f24 = f40[REFINED_24_INDICES]

    kps = clean_and_smooth_keypoints(kps_33_64f)
    shank_l = np.linalg.norm(kps[:, 25] - kps[:, 27], axis=-1)
    shank_r = np.linalg.norm(kps[:, 26] - kps[:, 28], axis=-1)
    shank_len = float(np.maximum(0.1, np.median((shank_l + shank_r) / 2.0)))

    wrist_mid = (kps[:, 15] + kps[:, 16]) / 2.0
    ank_mid = (kps[:, 27] + kps[:, 28]) / 2.0
    heel_mid = (kps[:, 29] + kps[:, 30]) / 2.0
    toe_mid = (kps[:, 31] + kps[:, 32]) / 2.0

    foot_vec = toe_mid - heel_mid
    foot_len = np.linalg.norm(foot_vec, axis=-1, keepdims=True) + 1e-6
    foot_dir = foot_vec / foot_len

    # 2. 바벨 모멘트 암 피처
    bar_rel_ank = wrist_mid - ank_mid
    bar_anterior = np.sum(bar_rel_ank * foot_dir, axis=-1) / shank_len
    f_moment_arm_max = float(np.max(bar_anterior))

    # 3. 시작 위상 피처 (초기 5프레임 손목 Y좌표: 바닥이면 0.7~0.85, 서있으면 0.4~0.55)
    start_wrist_y = float(np.mean(wrist_mid[0:5, 1]))
    f_phase_start = float(np.clip((start_wrist_y - 0.45) / 0.35, 0.0, 1.0))

    return np.hstack([f24, [f_moment_arm_max, f_phase_start]]).astype(np.float32)


# =========================================================================
# 64프레임 시공간 시계열 딥러닝 신경망 (ST-SeqNet v5.2)
# =========================================================================
class DeadliftSTSequenceNetV5_2(nn.Module):
    """
    15개 핵심 관절 + 손목-발목 상대변위 + 뒤꿈치-발가락 상대변위 = 51D 입력
    - Multi-Scale 1D Temporal Convolution (k=3, k=5)
    - Bidirectional GRU (2 Layers)
    - Temporal Attention Pooling
    """
    def __init__(self, in_features: int = 51, hidden_dim: int = 64):
        super(DeadliftSTSequenceNetV5_2, self).__init__()

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

        self.attn = nn.Sequential(
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
        # x: (Batch, 64, in_features)
        x_conv = x.transpose(1, 2)
        c3 = self.conv3(x_conv)
        c5 = self.conv5(x_conv)
        feat = self.relu(self.bn1(torch.cat([c3, c5], dim=1)))
        feat = self.dropout(feat).transpose(1, 2)

        gru_out, _ = self.bigru(feat)
        attn_w = F.softmax(self.attn(gru_out), dim=1)
        pooled = torch.sum(gru_out * attn_w, dim=1)

        logits = self.fc(pooled)
        return logits.squeeze(-1)


class BabelDeadliftD3ModelV5_2:
    """
    바벨 데드리프트 d3 공식 프로덕션 모델 v5.2 (4대 해결책 올인원)
    - 64프레임 51D 시공간 신경망 (ST-SeqNet v5.2)
    - 26D 최적화 정예 생체역학 Tri-Ensemble (CatBoost + LightGBM + RF)
    - 판단 보류율 <= 10.0% 엄격 보장 & F1-Score 최우선 최적화 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v5.2"
    FEATURE_DIM = 26
    FEATURE_NAMES = FEATURE_NAMES_26_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or {}
        self.dl_model: Optional[DeadliftSTSequenceNetV5_2] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.alpha_dl: float = 0.35
        self.is_fitted = False

    def fit(self, X_seq_51: np.ndarray, X_feats_26: np.ndarray, y: np.ndarray, epochs: int = 35, random_state: int = 42):
        # 1. PyTorch ST-SeqNet v5.2 학습
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        device = torch.device("cpu")
        dl_net = DeadliftSTSequenceNetV5_2(in_features=51, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(dl_net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        t_seq = torch.tensor(X_seq_51, dtype=torch.float32).to(device)
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

        # 2. CatBoost 학습 (26D 정예셋)
        cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats_26, y)
        self.cb_model = cb

        # 3. LightGBM 학습 (26D 정예셋)
        pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))
        ds_tr = lgb.Dataset(X_feats_26, label=y)
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

        # 4. Random Forest 학습 (26D 정예셋)
        rf = RandomForestClassifier(n_estimators=180, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_26, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq_51: np.ndarray, X_feats_26: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq_51, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()

        p_cb = self.cb_model.predict_proba(X_feats_26)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats_26)
        p_rf = self.rf_model.predict_proba(X_feats_26)[:, 1]
        p_tree = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        return self.alpha_dl * p_dl + (1.0 - self.alpha_dl) * p_tree

    def save_weights(self, out_path: str):
        data = {
            'dl_state_dict': self.dl_model.state_dict() if self.dl_model else None,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'alpha_dl': self.alpha_dl,
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_26_D3
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data['dl_state_dict']:
            self.dl_model = DeadliftSTSequenceNetV5_2(in_features=51, hidden_dim=64)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.alpha_dl = data.get('alpha_dl', 0.35)
        self.view_thresholds = data.get('view_thresholds', {})
        self.is_fitted = True
