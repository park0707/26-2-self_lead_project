# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v5.1 (d3_model_v5_1.py)
Status: ★ 피처 전수 감사(Audit) 기반 중복 4쌍 및 노이즈 8대 피처 소거 (40D -> 24D 정예화)
        - [경량화] 완전 동일 중복(r=1.0) 4쌍 및 고상관 피처 제거
        - [노이즈 제거] 피험자 개인차 유발 발목 최소각, 발날 롤(Roll) 등 8대 저해 피처 소거
        - [하이브리드 앙상블] 64프레임 ST-SeqNet(PyTorch) + 24D 정예 생체역학 Tri-Ensemble
        - [제약 준수] 판단 보류율 <= 10.0% (커버리지 >= 90.0%) & F1-Score 최우선 최적화
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

from src.models.second_models.babel_deadlift.d3_model_v5 import (
    DeadliftSTSequenceNet,
    KEY_JOINT_INDICES,
    NUM_KEY_JOINTS
)
from src.models.second_models.babel_deadlift.d3_model_v4 import (
    extract_d3_v4_features_from_seq,
    FEATURE_NAMES_40_D3,
    VIEW_CODE_LIST
)

# 40D 중 보존할 24개 정예 피처 인덱스 및 명칭 매핑
REFINED_24_INDICES = [2, 3, 5, 6, 7, 9, 11, 15, 18, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39]

FEATURE_NAMES_24_D3: List[str] = [FEATURE_NAMES_40_D3[i] for i in REFINED_24_INDICES]


def extract_d3_v5_1_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 24D 최적화 정예 피처 추출
    1. v4 40D 피처셋 추출
    2. 중복/노이즈 16개 피처 슬라이싱 소거 -> 24D 정예 피처 반환
    """
    f40 = extract_d3_v4_features_from_seq(kps_33_64f, view_code)
    return f40[REFINED_24_INDICES]


class BabelDeadliftD3ModelV5_1:
    """
    바벨 데드리프트 d3 공식 프로덕션 모델 v5.1 (피처 최적화 정예 하이브리드)
    - 64프레임 시공간 관절 시계열 신경망 (PyTorch ST-SeqNet)
    - 24D 핵심 정예 생체역학 피처 Tri-Ensemble (CatBoost + LightGBM + RF)
    - 커버리지 >= 90.0% 엄격 준수 F1 최우선 시점별 적응형 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v5.1"
    FEATURE_DIM = 24
    FEATURE_NAMES = FEATURE_NAMES_24_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or {}
        self.dl_model: Optional[DeadliftSTSequenceNet] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.alpha_dl: float = 0.35
        self.is_fitted = False

    def fit(self, X_seq: np.ndarray, X_feats_24: np.ndarray, y: np.ndarray, epochs: int = 35, random_state: int = 42):
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

        # 2. CatBoost 학습 (24D 정예셋)
        cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats_24, y)
        self.cb_model = cb

        # 3. LightGBM 학습 (24D 정예셋)
        pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))
        ds_tr = lgb.Dataset(X_feats_24, label=y)
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

        # 4. Random Forest 학습 (24D 정예셋)
        rf = RandomForestClassifier(n_estimators=180, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_24, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq: np.ndarray, X_feats_24: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # 1. 딥러닝 확률
        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()

        # 2. 트리 앙상블 확률 (24D)
        p_cb = self.cb_model.predict_proba(X_feats_24)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats_24)
        p_rf = self.rf_model.predict_proba(X_feats_24)[:, 1]
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
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_24_D3
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
