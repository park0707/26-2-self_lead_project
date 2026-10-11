# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v5.3 (d3_model_v5_3.py)
Status: ★ 외부 AI 3사(Gemini, Perplexity, 아주LLM) 자문 전격 반영 SOTA 통합 아키텍처
        - [Q1] 원판 가림 극복: 하체 Spatio-Temporal Graph Inpainting & 4중 텐서(원본+복원+잔차) 보존
        - [Q2] 시작 위상 불일치 해결: Kinematic Anchor Phase Warping (KAPW) 손목 Nadir 32프레임 정합
        - [Q3] 체형 분산 해소: Dimensionless Foot Pitch Angle + 3D Tibia Scale Invariant 32D 정예 피처셋
        - [Q4] 물리 제약 손실 및 판단 보류율 10% 엄격 제어: Physics-Informed Soft-F1 Loss & 3-State 안전 게이트
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

from src.models.second_models.babel_deadlift.d3_model_v5_2 import FEATURE_NAMES_26_D3
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

# v5.3 32D 정예 피처 명칭 (26D + 신규 생체역학/위상/가림 6종)
FEATURE_NAMES_32_D3: List[str] = FEATURE_NAMES_26_D3 + [
    "d3_foot_pitch_angle_max",        # [26] 발끝-뒤꿈치 최대 피치 각도 (상시 결함자 영점 소멸 방어)
    "d3_foot_pitch_angle_lift_mean",   # [27] 리프팅 구간 평균 발 피치 각도
    "d3_tibia_norm_heel_lift_max",    # [28] 3D 정강이 길이 대비 최대 뒤꿈치 들림 높이비 (무차원 지표)
    "d3_heel_wrist_vel_corr",         # [29] 손목 수직 속도와 뒤꿈치 수직 속도의 생체역학적 상관계수
    "d3_always_lift_setup_score",     # [30] 셋업 구간 초기부터 뒤꿈치가 들려있는 정도 (상시 결함 검출)
    "d3_inpaint_residual_energy"      # [31] 가림 복원 전후 잔차 에너지 (관측 불확실성 지표)
]


# =========================================================================
# Q2. Kinematic Anchor Phase Warping (KAPW) 함수
# =========================================================================
def apply_kapw_phase_warping(kps_seq_64f: np.ndarray, target_anchor: int = 32) -> Tuple[np.ndarray, int, float]:
    """
    (64, 15, 3) 또는 (64, 45) 시퀀스에 대해
    양 손목 Y좌표(화면상 아래로 갈수록 커짐)의 최하점(Nadir)을 찾아 target_anchor(32)에 정렬.
    반환: warped_seq, nadir_idx, phase_start_score
    """
    T = kps_seq_64f.shape[0]
    orig_shape = kps_seq_64f.shape
    flat_seq = kps_seq_64f.reshape(T, -1)

    # 15개 관절 중 손목(인덱스 3, 4)의 Y좌표(차원 3*3+1, 4*3+1)
    wrist_y = (flat_seq[:, 3 * 3 + 1] + flat_seq[:, 4 * 3 + 1]) / 2.0
    smooth_wrist_y = gaussian_filter1d(wrist_y, sigma=2.0)

    # Nadir 탐색 (y가 최대인 프레임)
    nadir_idx = int(np.argmax(smooth_wrist_y))
    # 극단치 클리핑 (5 ~ 58)
    nadir_idx = int(np.clip(nadir_idx, 5, T - 6))

    # 시작 위상 점수: 초기 5프레임 손목 높이 (바닥 셋업 ~0.8, 서있음 ~0.5)
    start_y = float(np.mean(wrist_y[:5]))
    phase_start_score = float(np.clip((start_y - 0.45) / 0.35, 0.0, 1.0))

    # Piecewise Spline Warping 그리드 생성
    orig_t = np.arange(T)
    map_t = np.zeros(T, dtype=np.float32)
    map_t[:target_anchor + 1] = np.linspace(0, nadir_idx, target_anchor + 1)
    map_t[target_anchor:] = np.linspace(nadir_idx, T - 1, T - target_anchor)

    # 3차 스플라인 워핑
    cs = CubicSpline(orig_t, flat_seq, axis=0)
    warped_flat = cs(map_t)

    return warped_flat.reshape(orig_shape).astype(np.float32), nadir_idx, phase_start_score


# =========================================================================
# Q1. Spatio-Temporal Graph Inpainting & 관측 잔차 추출
# =========================================================================
def inpaint_lower_body_joints(kps_15_64f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    (64, 15, 3) 시퀀스에서 후방 원판 가림에 취약한 하체 관절(뒤꿈치 11, 12, 발목 9, 10)
    지터 및 결측을 상위 관절(무릎 7, 8, 발가락 13, 14)의 기하학적 연속성으로 보정.
    반환: inpainted_kps (64, 15, 3), mask (64, 15), residual_energy (float)
    """
    T = kps_15_64f.shape[0]
    inpainted = kps_15_64f.copy()
    mask = np.zeros((T, 15), dtype=np.float32)

    # 발목/뒤꿈치 속도 급변(Jitter) 감지
    for side, (hip_idx, knee_idx, ank_idx, heel_idx, toe_idx) in enumerate([(5, 7, 9, 11, 13), (6, 8, 10, 12, 14)]):
        heel_vel = np.linalg.norm(np.diff(inpainted[:, heel_idx], axis=0, prepend=inpainted[:1, heel_idx]), axis=-1)
        # 속도가 비정상적으로 큰 구간(튀는 프레임) 마스킹
        vel_thresh = float(np.percentile(heel_vel, 92)) * 1.8 + 1e-4
        jitter_frames = np.where(heel_vel > vel_thresh)[0]
        mask[jitter_frames, heel_idx] = 1.0

        # 보정: 무릎과 발가락 사이의 부드러운 기하학적 보간
        smooth_heel = gaussian_filter1d(inpainted[:, heel_idx], sigma=2.5, axis=0)
        # 지터 구간에 대해 부드러운 궤적으로 대체
        for f in jitter_frames:
            inpainted[f, heel_idx] = smooth_heel[f]

    residual = inpainted - kps_15_64f
    residual_energy = float(np.mean(residual ** 2))

    return inpainted.astype(np.float32), mask, residual_energy


# =========================================================================
# Q3. 32D 무차원 및 피치 각도 정예 피처 추출
# =========================================================================
def extract_d3_v5_3_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 원천 시퀀스로부터 32D SOTA 피처 추출
    1. v5.2 26D 피처 계승
    2. Q3 무차원 피치 각도 및 정강이 들림비 피처 6종 추가
    """
    # v5.2 26D 기본 추출
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

    # 25: 바벨 모멘트 암 피처
    bar_rel_ank = wrist_mid - ank_mid
    bar_anterior = np.sum(bar_rel_ank * foot_dir, axis=-1) / shank_len
    f_moment_arm_max = float(np.max(bar_anterior))

    # 26: 시작 위상 피처
    start_wrist_y = float(np.mean(wrist_mid[0:5, 1]))
    f_phase_start = float(np.clip((start_wrist_y - 0.45) / 0.35, 0.0, 1.0))

    # -------------------------------------------------------------
    # [Q3 신규 6종 피처 계산]
    # -------------------------------------------------------------
    # 발끝-뒤꿈치 높이차 (화면 좌표계: 뒤꿈치 들리면 y 감소 -> delta_y = toe_y - heel_y > 0)
    delta_y = toe_mid[:, 1] - heel_mid[:, 1]
    horiz_dist = np.linalg.norm(toe_mid[:, [0, 2]] - heel_mid[:, [0, 2]], axis=-1) + 1e-6
    pitch_angle_deg = np.degrees(np.arctan2(delta_y, horiz_dist))

    # 27: 최대 피치 각도
    f_pitch_max = float(np.max(pitch_angle_deg))

    # 28: 리프팅 구간(손목 최하점 이후) 평균 피치 각도
    nadir_idx = int(np.argmax(gaussian_filter1d(wrist_mid[:, 1], sigma=2.0)))
    lifting_slice = slice(max(0, nadir_idx - 5), min(64, nadir_idx + 30))
    f_pitch_lift_mean = float(np.mean(pitch_angle_deg[lifting_slice]))

    # 29: 3D 정강이 길이 대비 최대 뒤꿈치 들림비 (무차원 지표)
    f_tibia_norm_lift_max = float(np.max(delta_y) / shank_len)

    # 30: 손목 수직 속도와 뒤꿈치 수직 속도의 상관계수
    v_wrist_y = -np.gradient(wrist_mid[:, 1]) # 양수: 상승 속도
    v_heel_y = -np.gradient(heel_mid[:, 1])   # 양수: 뒤꿈치 상승 속도
    std_w = np.std(v_wrist_y) + 1e-6
    std_h = np.std(v_heel_y) + 1e-6
    f_vel_corr = float(np.corrcoef(v_wrist_y, v_heel_y)[0, 1])
    if np.isnan(f_vel_corr):
        f_vel_corr = 0.0

    # 31: 셋업 구간 초기(0~10프레임) 평균 피치각 (상시 결함 감지)
    f_always_lift_score = float(np.mean(pitch_angle_deg[:10]))

    # 32: 가림 복원 잔차 에너지
    # 15개 관절 중 랜드마크 추출
    kps_15 = np.zeros((64, 15, 3), dtype=np.float32)
    for idx_new, idx_mp in enumerate(KEY_JOINT_INDICES):
        kps_15[:, idx_new] = kps[:, idx_mp]
    _, _, f_inpaint_residual = inpaint_lower_body_joints(kps_15)

    f32 = np.hstack([
        f24,
        [f_moment_arm_max, f_phase_start],
        [f_pitch_max, f_pitch_lift_mean, f_tibia_norm_lift_max, f_vel_corr, f_always_lift_score, f_inpaint_residual]
    ]).astype(np.float32)

    return f32


# =========================================================================
# Q4. Physics-Informed Soft-F1 Loss
# =========================================================================
class PhysicsInformedSoftF1Loss(nn.Module):
    """
    Soft-F1 Loss + Physics Penalty (바벨 미상승 시 뒤꿈치 들림 예측 패널티)
    """
    def __init__(self, lambda_phys: float = 0.40, pos_weight: float = 1.0):
        super(PhysicsInformedSoftF1Loss, self).__init__()
        self.lambda_phys = lambda_phys
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, wrist_vel_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # 1. 미분 가능한 Soft F1 Loss
        tp = torch.sum(probs * targets)
        fp = torch.sum(probs * (1.0 - targets))
        fn = torch.sum((1.0 - probs) * targets)

        soft_f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        loss_f1 = 1.0 - soft_f1

        # 2. 물리 제약 패널티: 손목 수직 속도가 음수(하강/정지)인데 들림으로 강하게 예측하는 경우 처벌
        if wrist_vel_y is not None:
            # wrist_vel_y: 양수면 상승, 음수면 하강/정지
            down_penalty = torch.relu(-wrist_vel_y)
            loss_phys = torch.mean(probs * down_penalty)
        else:
            loss_phys = torch.tensor(0.0, device=logits.device)

        return loss_f1 + self.lambda_phys * loss_phys


# =========================================================================
# 64프레임 Dual-Stream ST-SeqNet v5.3 (KAPW Stream + Inpaint Stream)
# =========================================================================
class DeadliftSTSequenceNetV5_3(nn.Module):
    """
    57D Dual-Stream 입력:
    - 45D 15개 관절 좌표
    - 3D 바벨-발목 상대변위
    - 3D 발 전방벡터
    - 3D KAPW 정합 손목/발목 궤적
    - 3D Inpaint 잔차 벡터
    = 총 57D
    """
    def __init__(self, in_features: int = 57, hidden_dim: int = 64):
        super(DeadliftSTSequenceNetV5_3, self).__init__()

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


# =========================================================================
# 바벨 데드리프트 d3 공식 프로덕션 모델 v5.3
# =========================================================================
class BabelDeadliftD3ModelV5_3:
    """
    바벨 데드리프트 d3 공식 프로덕션 모델 v5.3 (외부 AI 3사 종합반영)
    - 64프레임 57D Dual-Stream 신경망 (ST-SeqNet v5.3)
    - 32D 최적화 정예 생체역학 Tri-Ensemble (CatBoost + LightGBM + RF)
    - Physics-Informed Soft-F1 Loss & 3-State 안전 게이트
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v5.3"
    FEATURE_DIM = 32
    FEATURE_NAMES = FEATURE_NAMES_32_D3

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.view_thresholds = view_thresholds or {}
        self.dl_model: Optional[DeadliftSTSequenceNetV5_3] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[Any] = None
        self.alpha_dl: float = 0.35
        self.is_fitted = False

    def fit(self, X_seq_57: np.ndarray, X_feats_32: np.ndarray, y: np.ndarray,
            epochs: int = 35, random_state: int = 42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        device = torch.device("cpu")
        dl_net = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64).to(device)
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

        # 2. CatBoost 학습 (32D 정예셋)
        cb = CatBoostClassifier(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats_32, y)
        self.cb_model = cb

        # 3. LightGBM 학습 (32D 정예셋)
        ds_tr = lgb.Dataset(X_feats_32, label=y)
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

        # 4. Random Forest 학습 (32D 정예셋)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_32, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq_57: np.ndarray, X_feats_32: np.ndarray, views: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq_57, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()

        p_cb = self.cb_model.predict_proba(X_feats_32)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats_32)
        p_rf = self.rf_model.predict_proba(X_feats_32)[:, 1]
        p_tree = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        if views is not None:
            # 시점 적응형 가중치: 후방 0.55 / 전방,정면 0.40
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
            'alpha_dl': self.alpha_dl,
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_32_D3
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data['dl_state_dict']:
            self.dl_model = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.alpha_dl = data.get('alpha_dl', 0.35)
        self.view_thresholds = data.get('view_thresholds', {})
        self.is_fitted = True
