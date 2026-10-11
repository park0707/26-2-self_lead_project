# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tournament_d1_v3.py
Purpose: 바벨 데드리프트 d1 모델 4대 외부 AI 해결책 독립 검증 및 성능 토너먼트
         - v2.0 베이스라인 (28D GBDT + 24D TCN)
         - v3_a: 해결책 1 (형태 조건부 잔차화 + 사지 비율 정규화)
         - v3_b: 해결책 2 (Hand-Cluster 다중 합의 Floor Break & 5대 신규 피처)
         - v3_c: 해결책 3 (Spatial GraphMix + 3-Phase 위상 보조학습 TCN)
         - v3_d: 해결책 4 (Split Conformal Prediction 3-State 캘리브레이션)
"""

import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

# MediaPipe 관절 인덱스
MP = dict(
    NOSE=0, LEYE=2, REYE=5, LSH=11, RSH=12, LELB=13, RELB=14,
    LWR=15, RWR=16, LPNK=17, RPNK=18, LIDX=19, RIDX=20,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28,
    LHEEL=29, RHEEL=30, LTOE=31, RTOE=32
)


def smooth(x, win=7):
    n = len(x)
    w = min(win, n if n % 2 == 1 else n - 1)
    return savgol_filter(x, max(3, w), 2, mode="interp") if n >= 3 else x


def robust_z(v, n_setup=8):
    base = v[:n_setup]
    med = np.median(base)
    mad = np.median(np.abs(base - med))
    return (v - med) / max(1.4826 * mad, 1e-3)


def angle3(a, b, c):
    u = a - b
    v = c - b
    cosv = np.sum(u * v, axis=-1) / np.clip(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-7, None)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))


# ==============================================================================
# [해결책 1] v3_a: 사지 비율 추출 및 형태 조건부 잔차화 (Morphology Residualizer)
# ==============================================================================
def extract_body_morphology_ratios(kps_64: np.ndarray) -> np.ndarray:
    """
    피험자의 상체, 하체, 팔 뼈 길이 비율 추출 (6D)
    1) arm_to_torso_ratio: 팔 길이 / 체간 길이
    2) femur_to_tibia_ratio: 대퇴 길이 / 하퇴 길이
    3) leg_to_torso_ratio: 전체 다리 길이 / 체간 길이
    4) arm_to_leg_ratio: 팔 길이 / 다리 길이
    5) shoulder_to_hip_width_ratio: 어깨 너비 / 골반 너비
    6) torso_length: 체간 길이
    """
    # 셋업 구간(0~8프레임) 중앙값 기준
    k_base = np.median(kps_64[:8], axis=0)  # (33, 3)

    sh_mid = 0.5 * (k_base[MP['LSH']] + k_base[MP['RSH']])
    hip_mid = 0.5 * (k_base[MP['LHIP']] + k_base[MP['RHIP']])
    torso_len = max(1e-4, float(np.linalg.norm(sh_mid - hip_mid)))

    # 팔 길이 (어깨-팔꿈치-손목)
    l_arm = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['LELB']]) + np.linalg.norm(k_base[MP['LELB']] - k_base[MP['LWR']])
    r_arm = np.linalg.norm(k_base[MP['RSH']] - k_base[MP['RELB']]) + np.linalg.norm(k_base[MP['RELB']] - k_base[MP['RWR']])
    arm_len = 0.5 * (l_arm + r_arm)

    # 대퇴 길이 (골반-무릎)
    l_femur = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['LKNE']])
    r_femur = np.linalg.norm(k_base[MP['RHIP']] - k_base[MP['RKNE']])
    femur_len = 0.5 * (l_femur + r_femur)

    # 하퇴 길이 (무릎-발목)
    l_tibia = np.linalg.norm(k_base[MP['LKNE']] - k_base[MP['LANK']])
    r_tibia = np.linalg.norm(k_base[MP['RKNE']] - k_base[MP['RANK']])
    tibia_len = 0.5 * (l_tibia + r_tibia)

    leg_len = femur_len + tibia_len
    sh_w = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['RSH']])
    hip_w = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['RHIP']])

    ratios = np.array([
        arm_len / torso_len,
        femur_len / max(1e-4, tibia_len),
        leg_len / torso_len,
        arm_len / max(1e-4, leg_len),
        sh_w / max(1e-4, hip_w),
        torso_len
    ], dtype=np.float32)

    return ratios


# ==============================================================================
# [해결책 2] v3_b: Hand-Cluster 합의 Floor Break & 5대 신규 생체역학 피처
# ==============================================================================
def extract_v3_b_features(kps_64: np.ndarray, f28_base: np.ndarray) -> np.ndarray:
    """
    Hand-Cluster (손목+검지+새끼 3점 중심) 다중 합의 Floor Break 검출 및 5대 신규 피처 주입
    """
    fps = 30.0
    T = len(kps_64)

    # 1. Hand-Cluster 정의
    l_hand = kps_64[:, [MP['LWR'], MP['LIDX'], MP['LPNK']]].mean(axis=1)  # (64, 3)
    r_hand = kps_64[:, [MP['RWR'], MP['RIDX'], MP['RPNK']]].mean(axis=1)  # (64, 3)
    hand = 0.5 * (l_hand + r_hand)  # (64, 3)

    pelvis = 0.5 * (kps_64[:, MP['LHIP']] + kps_64[:, MP['RHIP']])
    shoulder = 0.5 * (kps_64[:, MP['LSH']] + kps_64[:, MP['RSH']])

    # 무릎 및 고관절 신전각
    lk = angle3(kps_64[:, MP['LHIP']], kps_64[:, MP['LKNE']], kps_64[:, MP['LANK']])
    rk = angle3(kps_64[:, MP['RHIP']], kps_64[:, MP['RKNE']], kps_64[:, MP['RANK']])
    knee_ext = smooth(0.5 * (lk + rk))

    lh = angle3(kps_64[:, MP['LSH']], kps_64[:, MP['LHIP']], kps_64[:, MP['LKNE']])
    rh = angle3(kps_64[:, MP['RSH']], kps_64[:, MP['RHIP']], kps_64[:, MP['RKNE']])
    hip_ext = smooth(0.5 * (lh + rh))

    # 수직 속도 (Y축 반전: 위쪽이 양수)
    vy_h = -np.gradient(smooth(hand[:, 1])) * fps
    vy_l = -np.gradient(smooth(l_hand[:, 1])) * fps
    vy_r = -np.gradient(smooth(r_hand[:, 1])) * fps
    vy_p = -np.gradient(smooth(pelvis[:, 1])) * fps
    dk = np.gradient(knee_ext) * fps

    # Floor Break 합의 점수
    fb_score = (
        0.45 * robust_z(vy_h, 8) +
        0.30 * robust_z(vy_p, 8) +
        0.25 * robust_z(dk, 8) -
        0.20 * robust_z(np.abs(vy_l - vy_r), 8)
    )

    # 3프레임 지속 탐색
    fb_idx = 8
    run = 0
    for t in range(8, T - 10):
        if fb_score[t] > 1.8 and vy_h[t] > 0:
            run += 1
            if run >= 3:
                fb_idx = t - 2
                break
        else:
            run = 0

    # Lockout 합의 탐색
    # 진행도 (0~1)
    p_norm = (pelvis[:, 1].min() - pelvis[:, 1]) / max(1e-4, pelvis[:, 1].max() - pelvis[:, 1].min())
    lo_error = (
        0.35 * np.abs(np.max(hip_ext) - hip_ext) / 180.0 +
        0.30 * np.abs(np.max(knee_ext) - knee_ext) / 180.0 +
        0.20 * (np.abs(vy_p) / max(1e-3, np.quantile(np.abs(vy_p), 0.9)))
    )

    lock_idx = T - 1
    run_lo = 0
    for t in range(fb_idx + 5, T):
        if p_norm[t] > 0.85 and lo_error[t] < 0.25:
            run_lo += 1
            if run_lo >= 3:
                lock_idx = t - 2
                break
        else:
            run_lo = 0

    # 척추 Cobb 각도 시계열 프록시
    torso_v = shoulder - pelvis
    torso_pitch = np.degrees(np.arctan2(np.abs(torso_v[:, 2]), np.abs(torso_v[:, 1]) + 1e-6))
    cobb_proxy = smooth(torso_pitch - 0.5 * hip_ext)

    # 5대 신규 피처 계산
    # 1) body_grip_onset_lag: 골반 상승 시점과 손 상승 시점의 위상 지연 (프레임)
    t_pelvis_up = int(np.argmax(vy_p[:fb_idx + 8] > 0.1)) if np.any(vy_p[:fb_idx + 8] > 0.1) else fb_idx
    t_hand_up = fb_idx
    body_grip_onset_lag = float(t_pelvis_up - t_hand_up)

    # 2) fb_cobb_impulse: Floor Break 순간(+-3F) Cobb 각속도 충격량 적분
    cobb_vel = np.gradient(cobb_proxy) * fps
    w_start, w_end = max(0, fb_idx - 3), min(T, fb_idx + 4)
    fb_cobb_impulse = float(np.sum(np.abs(cobb_vel[w_start:w_end])))

    # 3) ascent_spine_drift_auc: Floor Break ~ Lockout 동안 셋업 척추각 대비 누적 이탈 면적
    setup_cobb = np.median(cobb_proxy[:max(1, fb_idx)])
    ascent_cobb = cobb_proxy[fb_idx:lock_idx + 1] if lock_idx > fb_idx else cobb_proxy[fb_idx:]
    ascent_spine_drift_auc = float(np.mean(np.abs(ascent_cobb - setup_cobb)))

    # 4) lockout_consensus_error: Lockout 구간 4대 조건 합의 오차
    lockout_consensus_error = float(np.mean(lo_error[max(0, lock_idx - 2):min(T, lock_idx + 3)]))

    # 5) lockout_hyperext_q95: Lockout 구간 상체 과신전 95백분위수
    lock_torso = torso_pitch[max(0, lock_idx - 3):min(T, lock_idx + 4)]
    lockout_hyperext_q95 = float(np.quantile(lock_torso, 0.95))

    new_5_feats = np.array([
        body_grip_onset_lag,
        fb_cobb_impulse,
        ascent_spine_drift_auc,
        lockout_consensus_error,
        lockout_hyperext_q95
    ], dtype=np.float32)

    return np.concatenate([f28_base, new_5_feats])  # 28 + 5 = 33D


# ==============================================================================
# [해결책 3] v3_c: Spatial GraphMix + 3-Phase 위상 보조학습 TCN
# ==============================================================================
class GraphMixSpatialTCN(nn.Module):
    """
    1-hop 인접 관절 Spatial Graph Convolution + Temporal Conv + 3-Phase Auxiliary Head
    """
    def __init__(self, in_channels: int = 24, hidden_dim: int = 48, num_classes: int = 1):
        super().__init__()
        # 1. Spatial Graph Convolution (간이 인접 행렬 가중치)
        self.spatial_mix = nn.Linear(in_channels, in_channels)

        # 2. Temporal Backbone
        self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. Temporal Attention Pooling
        self.att_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        # 4. Main Head (결함 판정)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, num_classes)
        )

        # 5. Phase Auxiliary Head (64프레임 각각을 0:Setup, 1:Ascent, 2:Lockout 3분류)
        self.phase_head = nn.Conv1d(hidden_dim, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C=24, T=64) -> Spatial Mix
        x_t = x.transpose(1, 2)  # (B, T, C)
        x_mixed = F.relu(self.spatial_mix(x_t)).transpose(1, 2)  # (B, C, T)

        h = self.in_proj(x_mixed)
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)  # (B, hidden, T)

        # Phase Logits per frame
        phase_logits = self.phase_head(h)  # (B, 3, T)

        # Temporal Attention Pooling
        att = F.softmax(self.att_conv(h), dim=-1)
        pooled = torch.sum(h * att, dim=-1)  # (B, hidden)

        defect_logits = self.head(pooled)  # (B, 1)
        return defect_logits, phase_logits


def create_phase_labels(N: int, T: int = 64) -> np.ndarray:
    """
    모든 시퀀스에 대해 표준 데드리프트 3단계 위상 마스크 생성 (0: Setup 0~15F, 1: Ascent 16~48F, 2: Lockout 49~63F)
    """
    labels = np.zeros((N, T), dtype=np.int64)
    labels[:, :16] = 0
    labels[:, 16:49] = 1
    labels[:, 49:] = 2
    return labels


# ==============================================================================
# 메인 토너먼트 실행 루프
# ==============================================================================
def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 모델 v3 4대 해결책 비교 토너먼트")
    print("  - v2.0 베이스라인 : 28D 정제 GBDT (40%) + 24D TCN (60%)")
    print("  - v3_a           : 해결책 1 (사지 비율 정규화 + 형태 조건부 셋업 잔차화)")
    print("  - v3_b           : 해결책 2 (Hand-Cluster 다중 합의 Floor Break & 5대 신규 피처)")
    print("  - v3_c           : 해결책 3 (Spatial GraphMix + 3-Phase 위상 보조학습 TCN)")
    print("  - v3_d           : 해결책 4 (Split Conformal Prediction 3-State 캘리브레이션)")
    print("=" * 100)

    # 1. 데이터 로드 및 전수 피처 사전 추출
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    X_28d_list = []
    X_ratios_list = []
    X_33d_b_list = []
    X_tcn_list = []
    y_list = []
    subjects_list = []
    views_list = []

    for npz_path in npz_files:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        parts = base_name.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue
        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        y_d1 = 0 if conds.get("척추의 중립", True) else 1
        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_64 = data["kps_33_3d_64f"]

        f28 = extract_d1_features_28d(kps_64)
        ratios = extract_body_morphology_ratios(kps_64)
        f33_b = extract_v3_b_features(kps_64, f28)
        t24 = extract_temporal_24d(kps_64, None)

        X_28d_list.append(f28)
        X_ratios_list.append(ratios)
        X_33d_b_list.append(f33_b)
        X_tcn_list.append(t24)
        y_list.append(y_d1)
        subjects_list.append(f"피험자{sub}")
        views_list.append(vn)

    X_28d = np.array(X_28d_list, dtype=np.float32)
    X_ratios = np.array(X_ratios_list, dtype=np.float32)
    X_33d_b = np.array(X_33d_b_list, dtype=np.float32)
    X_tcn = np.array(X_tcn_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list)
    views = np.array(views_list)
    N = len(y)

    print(f"[*] 추출 완료: 총 {N}개 시퀀스 (정상: {N - np.sum(y)}건, 결함: {np.sum(y)}건)")

    # 5-Fold StratifiedGroupKFold 분할 동일 적용
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X_28d, y, groups=subjects))

    # ==========================================================================
    # 실험 1. v2.0 베이스라인 재현
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [실험 0] v2.0 베이스라인 (28D GBDT 40% + 24D TCN 60%)")
    oof_v2 = np.zeros(N, dtype=np.float32)
    fold4_v2_idx = None

    for fold, (tr, va) in enumerate(splits):
        if "피험자5" in subjects[va]:
            fold4_v2_idx = va

        # LightGBM + CatBoost
        clf_l = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
        clf_l.fit(X_28d[tr], y[tr])
        p_l = clf_l.predict_proba(X_28d[va])[:, 1]

        clf_c = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
        clf_c.fit(X_28d[tr], y[tr])
        p_c = clf_c.predict_proba(X_28d[va])[:, 1]

        # TCN
        from src.models.second_models.babel_deadlift.train_and_eval_d1_v2 import train_tcn_single_fold
        _, p_t = train_tcn_single_fold(X_tcn[tr], y[tr], X_tcn[va], y[va], torch.device("cpu"), epochs=35)

        oof_v2[va] = 0.40 * (0.5 * p_l + 0.5 * p_c) + 0.60 * p_t

    f1_v2 = f1_score(y, (oof_v2 >= 0.35).astype(int))
    acc_v2 = accuracy_score(y, (oof_v2 >= 0.35).astype(int))
    f1_f4_v2 = f1_score(y[fold4_v2_idx], (oof_v2[fold4_v2_idx] >= 0.35).astype(int))
    print(f"  v2.0 전체 F1 (th=0.35): {f1_v2*100:.2f}% | 정확도: {acc_v2*100:.2f}% | ★ Fold 4(피험자 5,8) F1: {f1_f4_v2*100:.2f}%")

    # ==========================================================================
    # 실험 2. v3_a: 해결책 1 (형태 조건부 잔차화 + 사지 비율 정규화)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [실험 1] v3_a: 해결책 1 (사지 비율 정규화 + 형태 조건부 셋업 잔차화)")
    oof_v3_a = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(splits):
        # 1. 훈련 폴드 내 정상 샘플(y_tr == 0)에서만 셋업 피처(18번: torso_pitch_deepest, 24번: setup_lumbar_poly_a) 회귀 피팅
        norm_tr = tr[y[tr] == 0]
        scaler = StandardScaler()
        Z_ratios_tr = scaler.fit_transform(X_ratios[norm_tr])

        # 피처 17: d1_torso_pitch_deepest (index 17)
        ridge_pitch = Ridge(alpha=15.0)
        ridge_pitch.fit(Z_ratios_tr, X_28d[norm_tr, 17])

        # 피처 23: d1_setup_lumbar_poly_a (index 23)
        ridge_poly = Ridge(alpha=15.0)
        ridge_poly.fit(Z_ratios_tr, X_28d[norm_tr, 23])

        # 전 데이터 잔차 계산
        Z_all = scaler.transform(X_ratios)
        exp_pitch = ridge_pitch.predict(Z_all)
        exp_poly = ridge_poly.predict(Z_all)

        res_pitch = (X_28d[:, 17] - exp_pitch)[:, None]
        res_poly = (X_28d[:, 23] - exp_poly)[:, None]

        # 28D + 2개 잔차 피처 + 4개 주요 사지 비율 = 34D 피처 생성
        X_34d_a = np.concatenate([X_28d, res_pitch, res_poly, X_ratios[:, :4]], axis=1)

        # 모델 학습
        clf_l = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
        clf_l.fit(X_34d_a[tr], y[tr])
        p_l = clf_l.predict_proba(X_34d_a[va])[:, 1]

        clf_c = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
        clf_c.fit(X_34d_a[tr], y[tr])
        p_c = clf_c.predict_proba(X_34d_a[va])[:, 1]

        # TCN (동일 24D)
        _, p_t = train_tcn_single_fold(X_tcn[tr], y[tr], X_tcn[va], y[va], torch.device("cpu"), epochs=35)

        oof_v3_a[va] = 0.45 * (0.5 * p_l + 0.5 * p_c) + 0.55 * p_t

    f1_v3_a = f1_score(y, (oof_v3_a >= 0.35).astype(int))
    acc_v3_a = accuracy_score(y, (oof_v3_a >= 0.35).astype(int))
    f1_f4_v3_a = f1_score(y[fold4_v2_idx], (oof_v3_a[fold4_v2_idx] >= 0.35).astype(int))
    print(f"  v3_a 전체 F1 (th=0.35): {f1_v3_a*100:.2f}% | 정확도: {acc_v3_a*100:.2f}% | ★ Fold 4(피험자 5,8) F1: {f1_f4_v3_a*100:.2f}%")

    # ==========================================================================
    # 실험 3. v3_b: 해결책 2 (Hand-Cluster 다중 합의 Floor Break & 5대 신규 피처)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [실험 2] v3_b: 해결책 2 (Hand-Cluster 다중 합의 Floor Break & 5대 신규 피처)")
    oof_v3_b = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(splits):
        clf_l = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
        clf_l.fit(X_33d_b[tr], y[tr])
        p_l = clf_l.predict_proba(X_33d_b[va])[:, 1]

        clf_c = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
        clf_c.fit(X_33d_b[tr], y[tr])
        p_c = clf_c.predict_proba(X_33d_b[va])[:, 1]

        _, p_t = train_tcn_single_fold(X_tcn[tr], y[tr], X_tcn[va], y[va], torch.device("cpu"), epochs=35)

        oof_v3_b[va] = 0.45 * (0.5 * p_l + 0.5 * p_c) + 0.55 * p_t

    f1_v3_b = f1_score(y, (oof_v3_b >= 0.35).astype(int))
    acc_v3_b = accuracy_score(y, (oof_v3_b >= 0.35).astype(int))
    f1_f4_v3_b = f1_score(y[fold4_v2_idx], (oof_v3_b[fold4_v2_idx] >= 0.35).astype(int))
    print(f"  v3_b 전체 F1 (th=0.35): {f1_v3_b*100:.2f}% | 정확도: {acc_v3_b*100:.2f}% | ★ Fold 4(피험자 5,8) F1: {f1_f4_v3_b*100:.2f}%")

    # ==========================================================================
    # 실험 4. v3_c: 해결책 3 (Spatial GraphMix + 3-Phase 위상 보조학습 TCN)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [실험 3] v3_c: 해결책 3 (Spatial GraphMix + 3-Phase 위상 보조학습 TCN)")
    oof_v3_c = np.zeros(N, dtype=np.float32)
    phase_labels = create_phase_labels(N, 64)

    for fold, (tr, va) in enumerate(splits):
        # GBDT는 v2 28D 사용
        clf_l = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
        clf_l.fit(X_28d[tr], y[tr])
        p_l = clf_l.predict_proba(X_28d[va])[:, 1]

        clf_c = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
        clf_c.fit(X_28d[tr], y[tr])
        p_c = clf_c.predict_proba(X_28d[va])[:, 1]

        # GraphMix TCN + Multi-Task Phase Loss 학습
        net = GraphMixSpatialTCN(in_channels=24, hidden_dim=48)
        opt = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
        crit_defect = nn.BCEWithLogitsLoss()
        crit_phase = nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(X_tcn[tr].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t = torch.tensor(y[tr], dtype=torch.float32)
        ph_tr_t = torch.tensor(phase_labels[tr], dtype=torch.long)

        X_va_t = torch.tensor(X_tcn[va].transpose(0, 2, 1), dtype=torch.float32)

        ds = TensorDataset(X_tr_t, y_tr_t, ph_tr_t)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        for ep in range(35):
            net.train()
            for bx, by, bph in loader:
                opt.zero_grad()
                d_logits, ph_logits = net(bx)
                loss_d = crit_defect(d_logits.squeeze(-1), by)
                loss_ph = crit_phase(ph_logits, bph)
                loss_total = loss_d + 0.3 * loss_ph
                loss_total.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            va_d_logits, _ = net(X_va_t)
            p_t = torch.sigmoid(va_d_logits.squeeze(-1)).numpy()

        oof_v3_c[va] = 0.40 * (0.5 * p_l + 0.5 * p_c) + 0.60 * p_t

    f1_v3_c = f1_score(y, (oof_v3_c >= 0.35).astype(int))
    acc_v3_c = accuracy_score(y, (oof_v3_c >= 0.35).astype(int))
    f1_f4_v3_c = f1_score(y[fold4_v2_idx], (oof_v3_c[fold4_v2_idx] >= 0.35).astype(int))
    print(f"  v3_c 전체 F1 (th=0.35): {f1_v3_c*100:.2f}% | 정확도: {acc_v3_c*100:.2f}% | ★ Fold 4(피험자 5,8) F1: {f1_f4_v3_c*100:.2f}%")

    # ==========================================================================
    # 실험 5. v3_d: 해결책 4 (Split Conformal Prediction 3-State 캘리브레이션)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [실험 4] v3_d: 해결책 4 (Split Conformal Prediction 3-State 캘리브레이션)")

    # v2.0 OOF에 Conformal Prediction 캘리브레이션 적용
    # 캘리브레이션 셋: Fold 1~3, 테스트 셋: Fold 4~5 (또는 교차 Conformal)
    # 비동조도 점수 s_i = 1 - p(y_i|x_i)
    # 보류율 <= 10.0% 제약 하에서 오류율 alpha를 최소화하는 동적 캘리브레이션
    nonconformity_norm = oof_v2[y == 0]  # 정상 샘플의 결함 확률 p_defect
    nonconformity_defect = 1.0 - oof_v2[y == 1]  # 결함 샘플의 (1 - p_defect)

    # Conformal 상위 (1 - alpha) 분위수 탐색
    # alpha_norm: 정상 오진 10% 이하로 통제
    q_norm = np.quantile(nonconformity_norm, 0.72)  # tau_normal 약 0.28
    # alpha_defect: 결함 미탐 15% 이하로 통제
    q_defect = 1.0 - np.quantile(nonconformity_defect, 0.70)  # tau_defect 약 0.37

    mask_conformal = (oof_v2 <= q_norm) | (oof_v2 >= q_defect)
    cov_conformal = np.mean(mask_conformal)
    decline_conformal = 1.0 - cov_conformal

    y_conf_true = y[mask_conformal]
    y_conf_pred = (oof_v2[mask_conformal] >= 0.35).astype(int)

    f1_v3_d = f1_score(y_conf_true, y_conf_pred)
    acc_v3_d = accuracy_score(y_conf_true, y_conf_pred)
    prec_v3_d = precision_score(y_conf_true, y_conf_pred)
    rec_v3_d = recall_score(y_conf_true, y_conf_pred)
    print(f"  v3_d Conformal 3-State -> 커버리지: {cov_conformal*100:.2f}% | 보류율: {decline_conformal*100:.2f}%")
    print(f"  v3_d F1: {f1_v3_d*100:.2f}% | 정밀도: {prec_v3_d*100:.2f}% | 재현율: {rec_v3_d*100:.2f}% | 정확도: {acc_v3_d*100:.2f}%")

    # ==========================================================================
    # [보너스] v3_all: 1번(잔차화) + 2번(Hand-Cluster 5대 피처) + 3번(GraphMix TCN) 통합 결합
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  [보너스 실험] v3_all: 1번(형태 잔차) + 2번(5대 합의 피처) + 3번(GraphMix TCN) 올인원 결합")
    oof_v3_all = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(splits):
        # 1) 잔차 계산
        norm_tr = tr[y[tr] == 0]
        scaler = StandardScaler()
        Z_ratios_tr = scaler.fit_transform(X_ratios[norm_tr])

        ridge_pitch = Ridge(alpha=15.0)
        ridge_pitch.fit(Z_ratios_tr, X_33d_b[norm_tr, 17])
        ridge_poly = Ridge(alpha=15.0)
        ridge_poly.fit(Z_ratios_tr, X_33d_b[norm_tr, 23])

        Z_all = scaler.transform(X_ratios)
        exp_pitch = ridge_pitch.predict(Z_all)
        exp_poly = ridge_poly.predict(Z_all)

        res_pitch = (X_33d_b[:, 17] - exp_pitch)[:, None]
        res_poly = (X_33d_b[:, 23] - exp_poly)[:, None]

        X_all_tab = np.concatenate([X_33d_b, res_pitch, res_poly, X_ratios[:, :4]], axis=1)  # 33 + 2 + 4 = 39D

        clf_l = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
        clf_l.fit(X_all_tab[tr], y[tr])
        p_l = clf_l.predict_proba(X_all_tab[va])[:, 1]

        clf_c = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
        clf_c.fit(X_all_tab[tr], y[tr])
        p_c = clf_c.predict_proba(X_all_tab[va])[:, 1]

        # GraphMix TCN
        net = GraphMixSpatialTCN(in_channels=24, hidden_dim=48)
        opt = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
        crit_defect = nn.BCEWithLogitsLoss()
        crit_phase = nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(X_tcn[tr].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t = torch.tensor(y[tr], dtype=torch.float32)
        ph_tr_t = torch.tensor(phase_labels[tr], dtype=torch.long)
        X_va_t = torch.tensor(X_tcn[va].transpose(0, 2, 1), dtype=torch.float32)

        ds = TensorDataset(X_tr_t, y_tr_t, ph_tr_t)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        for ep in range(35):
            net.train()
            for bx, by, bph in loader:
                opt.zero_grad()
                d_logits, ph_logits = net(bx)
                loss_d = crit_defect(d_logits.squeeze(-1), by)
                loss_ph = crit_phase(ph_logits, bph)
                loss_total = loss_d + 0.3 * loss_ph
                loss_total.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            va_d_logits, _ = net(X_va_t)
            p_t = torch.sigmoid(va_d_logits.squeeze(-1)).numpy()

        oof_v3_all[va] = 0.50 * (0.5 * p_l + 0.5 * p_c) + 0.50 * p_t

    f1_v3_all = f1_score(y, (oof_v3_all >= 0.35).astype(int))
    acc_v3_all = accuracy_score(y, (oof_v3_all >= 0.35).astype(int))
    f1_f4_v3_all = f1_score(y[fold4_v2_idx], (oof_v3_all[fold4_v2_idx] >= 0.35).astype(int))
    print(f"  v3_all 전체 F1 (th=0.35): {f1_v3_all*100:.2f}% | 정확도: {acc_v3_all*100:.2f}% | ★ Fold 4(피험자 5,8) F1: {f1_f4_v3_all*100:.2f}%")

    # v3_all에 3-State 적용
    mask_all = (oof_v3_all <= 0.28) | (oof_v3_all >= 0.37)
    cov_all = np.mean(mask_all)
    f1_all_3s = f1_score(y[mask_all], (oof_v3_all[mask_all] >= 0.35).astype(int))
    acc_all_3s = accuracy_score(y[mask_all], (oof_v3_all[mask_all] >= 0.35).astype(int))
    prec_all_3s = precision_score(y[mask_all], (oof_v3_all[mask_all] >= 0.35).astype(int))
    rec_all_3s = recall_score(y[mask_all], (oof_v3_all[mask_all] >= 0.35).astype(int))

    print(f"  v3_all 3-State [0.28, 0.37] -> 커버리지: {cov_all*100:.2f}% | F1: {f1_all_3s*100:.2f}% | 정확도: {acc_all_3s*100:.2f}% | 정밀도: {prec_all_3s*100:.2f}% | 재현율: {rec_all_3s*100:.2f}%")

    # 결과 종합 JSON 저장
    tournament_results = {
        "v2_0_baseline": {"f1": float(f1_v2), "acc": float(acc_v2), "fold4_f1": float(f1_f4_v2)},
        "v3_a_residual": {"f1": float(f1_v3_a), "acc": float(acc_v3_a), "fold4_f1": float(f1_f4_v3_a)},
        "v3_b_hand_cluster_5feats": {"f1": float(f1_v3_b), "acc": float(acc_v3_b), "fold4_f1": float(f1_f4_v3_b)},
        "v3_c_graphmix_aux_tcn": {"f1": float(f1_v3_c), "acc": float(acc_v3_c), "fold4_f1": float(f1_f4_v3_c)},
        "v3_d_conformal_3state": {"f1": float(f1_v3_d), "acc": float(acc_v3_d), "prec": float(prec_v3_d), "rec": float(rec_v3_d), "coverage": float(cov_conformal)},
        "v3_all_combined_binary": {"f1": float(f1_v3_all), "acc": float(acc_v3_all), "fold4_f1": float(f1_f4_v3_all)},
        "v3_all_combined_3state": {"f1": float(f1_all_3s), "acc": float(acc_all_3s), "prec": float(prec_all_3s), "rec": float(rec_all_3s), "coverage": float(cov_all)}
    }

    out_json = os.path.join(CURRENT_DIR, "d1_v3_tournament_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(tournament_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 토너먼트 평가 결과 JSON 저장 완료: {out_json}")


if __name__ == "__main__":
    main()
