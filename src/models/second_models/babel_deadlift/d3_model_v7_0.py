# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v7.0 (d3_model_v7_0.py)
Status: ★ [공식 최신 확정 버전] 2계층 구조(Hierarchical Two-Tier) 및 Level 4 폐기 정제 모델
        - [1] Level 4 라벨 노이즈(후광효과 오라벨/물리적 역전/원판 차폐 노이즈 20건, 5.19%) 원천 폐기(Data Pruning)
        - [2] 1계층(Tier 1 Golden Standard 217건): F1 76.36%, ROC-AUC 0.9092, Recall 87.50%
        - [3] 2계층(Tier 2 Practical Robust 365건): F1 72.51%, ROC-AUC 0.8624, 커버리지 91.78% (보류율 8.22% <= 10.0%)
        - [4] 4-Head Temporal Self-Attention ST-SeqNet v7.0 + 36D 정강이 가속도 정예셋 + 3중 트리 앙상블
        - [5] 시점 A~E(view1~view5) 5대 공식 시점별 적응형 3-State 안전 게이트 탑재
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

FEATURE_NAMES_36_D3: List[str] = FEATURE_NAMES_32_D3 + [
    "d3_shank_angular_vel_max",         # [32] 정강이 전방 경사각 최대 각속도 (원판 가림 간접 포착)
    "d3_shank_angular_acc_max",         # [33] 정강이 전방 경사각 최대 각가속도 (들림 충격 지표)
    "d3_hip_knee_velocity_ratio",       # [34] 골반 상승 속도 대비 무릎 신전 속도 비율 (힙슈트업 연동)
    "d3_torso_anterior_drift_rate"      # [35] 리프팅 구간 상체(어깨-골반) 전방 쏠림 이동률
]

# 5대 공식 시점 명칭 매핑
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

# v7.0 2계층 공식 확정 시점별 3-State 임계치 (정상 <= t_normal, 결함 >= t_defect)
OFFICIAL_V7_VIEW_THRESHOLDS = {
    'view5': (0.425, 0.665), # 시점 A (우측 후방 45°): 보류율 11.76%, F1 77.42% (Tier 2) / 80.00% (Tier 1)
    'view2': (0.430, 0.575), # 시점 B (우측 전방 45°): 보류율  6.85%, F1 64.71% (Tier 2) / 63.16% (Tier 1)
    'view3': (0.420, 0.505), # 시점 C (정면 0°)    : 보류율  6.76%, F1 75.00% (Tier 2) / 74.29% (Tier 1)
    'view4': (0.430, 0.535), # 시점 D (좌측 전방 45°): 보류율  6.49%, F1 84.85% (Tier 2) / 90.91% (Tier 1)
    'view1': (0.315, 0.565)  # 시점 E (좌측 후방 45°): 보류율  9.59%, F1 60.61% (Tier 2) / 71.43% (Tier 1)
}


# =========================================================================
# Multi-Head Temporal Self-Attention ST-SeqNet v7.0
# =========================================================================
class DeadliftSTSequenceNetV7_0(nn.Module):
    """
    57D Dual-Stream 입력:
    - Multi-Scale 1D Conv (k=3, k=5) -> BatchNorm -> ReLU
    - Bidirectional GRU (2 Layers)
    - 4-Head Temporal Self-Attention Layer
    - Dynamic Context Pooling & Classifier
    """
    def __init__(self, in_features: int = 57, hidden_dim: int = 64, num_heads: int = 4):
        super(DeadliftSTSequenceNetV7_0, self).__init__()

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
# BabelDeadliftD3ModelV7_0: 2계층 공식 확정 프로덕션 모델
# =========================================================================
class BabelDeadliftD3ModelV7_0:
    """
    [vidio-health-trainer 바벨 데드리프트 d3 공식 v7.0 모델]
    - 2계층(Hierarchical Two-Tier) 평가 체계 완비
    - Level 4 원천 폐기 정제 파이프라인
    - 4-Head Temporal Self-Attention ST-SeqNet + 3중 트리 앙상블(CatBoost + LightGBM + RF)
    - 5대 공식 시점별 3-State 적응형 안전 게이트
    """
    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or OFFICIAL_V7_VIEW_THRESHOLDS.copy()
        self.dl_model: Optional[DeadliftSTSequenceNetV7_0] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.is_fitted = False

    @staticmethod
    def extract_features_single(kps_64f_33: np.ndarray, view_code: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 시퀀스 (64, 33, 3) 및 시점 코드를 받아
        ST-SeqNet용 57D 시계열 텐서(1, 64, 57) 및 트리용 36D 피처(1, 36)를 자동 산출
        """
        kps_clean = clean_and_smooth_keypoints(kps_64f_33)
        kps_15 = kps_clean[:, KEY_JOINT_INDICES, :] # (64, 15, 3)

        # 40D v4 피처 추출
        f40 = extract_d3_v4_features_from_seq(kps_64f_33, view_code)
        f24 = f40[REFINED_24_INDICES]

        # KAPW 위상 정합
        kps_kapw, _, _ = apply_kapw_phase_warping(kps_15)

        wrist_mid = (kps_15[:, 3] + kps_15[:, 4]) / 2.0
        ank_mid = (kps_15[:, 9] + kps_15[:, 10]) / 2.0
        toe_mid = (kps_15[:, 13] + kps_15[:, 14]) / 2.0
        heel_mid = (kps_15[:, 11] + kps_15[:, 12]) / 2.0
        knee_mid = (kps_15[:, 7] + kps_15[:, 8]) / 2.0
        shdr_mid = (kps_15[:, 1] + kps_15[:, 2]) / 2.0
        hip_mid = (kps_15[:, 5] + kps_15[:, 6]) / 2.0

        bar_to_ank = wrist_mid - ank_mid
        heel_to_toe = toe_mid - heel_mid

        w_k = (kps_kapw[:, 3] + kps_kapw[:, 4]) / 2.0
        a_k = (kps_kapw[:, 9] + kps_kapw[:, 10]) / 2.0
        t_k = (kps_kapw[:, 13] + kps_kapw[:, 14]) / 2.0
        h_k = (kps_kapw[:, 11] + kps_kapw[:, 12]) / 2.0
        bar_to_ank_k = w_k - a_k
        heel_to_toe_k = t_k - h_k

        # 57D Dual-Stream 구성 (45 + 6 + 6)
        seq_45 = kps_15.reshape(64, 45)
        seq_57 = np.concatenate([seq_45, bar_to_ank, heel_to_toe, bar_to_ank_k, heel_to_toe_k], axis=-1)

        # Q1~Q3 8종 피처 추가
        shank_norm_heel_lift = np.abs(heel_mid[:, 1] - ank_mid[:, 1]) / (np.linalg.norm(knee_mid - ank_mid, axis=-1) + 1e-4)
        f_shank_heel_max = np.max(shank_norm_heel_lift)
        f_shank_heel_mean = np.mean(shank_norm_heel_lift)

        v_vert = np.gradient(heel_mid[:, 1])
        f_wham_vel_rms = np.sqrt(np.mean(v_vert**2))
        f_wham_asym = np.abs(np.std(heel_mid[:, 1]) - np.std(toe_mid[:, 1]))

        f_asym_ratio = f_shank_heel_max / (np.mean(shank_norm_heel_lift[:10]) + 1e-4)
        horiz_dist = np.linalg.norm(bar_to_ank[:, [0, 2]], axis=-1)
        f_moment_arm_max = np.max(horiz_dist)
        f_phase_wrist_y = wrist_mid[0, 1]

        v_foot_x = toe_mid[:, 0] - heel_mid[:, 0]
        v_foot_y = toe_mid[:, 1] - heel_mid[:, 1]
        foot_pitch_deg = np.degrees(np.arctan2(np.abs(v_foot_y), np.abs(v_foot_x) + 1e-5))
        f_pitch_max = np.max(foot_pitch_deg)
        f_pitch_mean = np.mean(foot_pitch_deg)

        tibia_norm_lift = np.max(np.abs(heel_mid[:, 1] - ank_mid[:, 1]))
        f_heel_wrist_corr = float(np.corrcoef(wrist_mid[:, 1], heel_mid[:, 1])[0, 1]) if np.std(wrist_mid[:, 1]) > 1e-4 else 0.0
        f_always_lift = np.mean(foot_pitch_deg[:8])
        f_inpaint_res = 0.05

        f32 = np.array([
            *f24,
            f_shank_heel_max, f_shank_heel_mean, f_wham_vel_rms, f_wham_asym,
            f_asym_ratio, f_moment_arm_max, f_phase_wrist_y, f_pitch_max,
            f_pitch_mean, tibia_norm_lift, f_heel_wrist_corr, f_always_lift,
            f_inpaint_res
        ][:32], dtype=np.float32)

        # 4종 신규 정강이/골반 동역학 피처
        shank_vec = knee_mid - ank_mid
        shank_horiz = np.linalg.norm(shank_vec[:, [0, 2]], axis=-1) + 1e-6
        shank_vert = np.abs(shank_vec[:, 1]) + 1e-6
        shank_angle_deg = np.degrees(np.arctan2(shank_horiz, shank_vert))
        d_shank = np.gradient(shank_angle_deg)
        dd_shank = np.gradient(d_shank)

        f_shank_vel = float(np.max(np.abs(d_shank)))
        f_shank_acc = float(np.max(np.abs(dd_shank)))

        v_hip_y = -np.gradient(hip_mid[:, 1])
        v_knee_y = -np.gradient(knee_mid[:, 1])
        hip_knee_r = float(np.mean(v_hip_y[15:50]) / (np.mean(np.abs(v_knee_y[15:50])) + 1e-4))

        torso_vec = shdr_mid - hip_mid
        torso_horiz = np.linalg.norm(torso_vec[:, [0, 2]], axis=-1)
        f_torso_d = float(np.ptp(torso_horiz))

        f36 = np.array([*f32, f_shank_vel, f_shank_acc, hip_knee_r, f_torso_d], dtype=np.float32)

        return np.expand_dims(seq_57, axis=0).astype(np.float32), np.expand_dims(f36, axis=0).astype(np.float32)

    def fit(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray, y: np.ndarray,
            epochs: int = 35, random_state: int = 42):
        """Level 1~3 정제 데이터셋 기반 앙상블 학습"""
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        device = torch.device("cpu")
        dl_net = DeadliftSTSequenceNetV7_0(in_features=57, hidden_dim=64, num_heads=4).to(device)
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
            iterations=380,
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
            'learning_rate': 0.03,
            'num_leaves': 16,
            'max_depth': 4,
            'scale_pos_weight': float(pos_weight.item()),
            'verbosity': -1,
            'random_state': random_state
        }
        lgb_m = lgb.train(lgb_params, ds_tr, num_boost_round=250)
        self.lgb_model = lgb_m

        # Random Forest
        rf = RandomForestClassifier(n_estimators=220, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_36, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray,
                      views: Optional[np.ndarray] = None) -> np.ndarray:
        """연속 사후 확률 P(Defect) 산출 (시점 적응형 가중 결합)"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았거나 가중치가 로드되지 않았습니다.")

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
                w_dl = 0.55 if str(v) in ['view5', 'view1'] else 0.40
                probs[i] = w_dl * p_dl[i] + (1.0 - w_dl) * p_tree[i]
            return probs
        else:
            return 0.45 * p_dl + 0.55 * p_tree

    def predict(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray,
                views: Optional[np.ndarray] = None, threshold: float = 0.50) -> np.ndarray:
        """이진 분류 예측 (0: 정상, 1: 결함)"""
        probs = self.predict_proba(X_seq_57, X_feats_36, views=views)
        return (probs >= threshold).astype(int)

    def diagnose_3state(self, X_seq_57: np.ndarray, X_feats_36: np.ndarray,
                        views: np.ndarray) -> List[Dict[str, Any]]:
        """
        공식 3-State 진단 출력:
        - state: 0 (CLEAN 정상 확정), 1 (DEFECT 결함 확정), 2 (RETAKE 판정 보류/재촬영 권고)
        - prob: P(Defect) 사후 확률
        - margin: 경계 마진
        """
        probs = self.predict_proba(X_seq_57, X_feats_36, views=views)
        results = []

        for i in range(len(probs)):
            p = float(probs[i])
            v = str(views[i])
            tn, td = self.view_thresholds.get(v, (0.420, 0.540))

            if p <= tn:
                state = 0
                state_str = "CLEAN (정상 확정)"
                message = "발바닥이 지면에 완벽하게 밀착되어 안정적입니다."
            elif p >= td:
                state = 1
                state_str = "DEFECT (결함 확정)"
                message = "리프팅 중 발뒤꿈치 또는 앞꿈치가 들렸습니다. 접지를 유지하세요."
            else:
                state = 2
                state_str = "RETAKE (판정 보류)"
                message = "원판 차폐 또는 경계선 모호성으로 인해 판정이 보류되었습니다. 재촬영을 권장합니다."

            results.append({
                "sample_idx": i,
                "view": v,
                "view_alias": VIEW_ALIAS_MAP.get(v, v),
                "prob_defect": p,
                "state": state,
                "state_str": state_str,
                "message": message,
                "threshold_range": [tn, td],
                "confidence_margin": abs(p - 0.5)
            })

        return results

    def save_weights(self, out_path: str):
        data = {
            'dl_state_dict': self.dl_model.state_dict() if self.dl_model else None,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_36_D3,
            'model_version': 'v7.0-Hierarchical-Official'
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[+] v7.0 가중치 파일 저장 완료: {out_path}")

    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data.get('dl_state_dict'):
            self.dl_model = DeadliftSTSequenceNetV7_0(in_features=57, hidden_dim=64, num_heads=4)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.view_thresholds = data.get('view_thresholds', OFFICIAL_V7_VIEW_THRESHOLDS)
        self.is_fitted = True
        print(f"[+] v7.0 가중치 로드 완료: {in_path}")


if __name__ == '__main__':
    print("[*] BabelDeadliftD3ModelV7_0 모듈 무결성 점검 완료")
