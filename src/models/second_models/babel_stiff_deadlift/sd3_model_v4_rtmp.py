# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect)
Version: v4.0 RTMPose Optimized (sd3_model_v4_rtmp.py)
Purpose: RTMPose-m (Halpe26) 2D 고정밀 관절 및 바벨 궤적에 최적화된
         전략 4 Multi-View Dynamic Meta Fusion (36D GBDT + Dilated-TCN 2-Stream)
         정면(view3) 블라인드 스팟 극복 및 3-State 선택적 진단 인터페이스
Rule: AGENTS.md 규정 및 공식 5대 시점 체계(시점 A~E), 순수 비전 64프레임 입력 준수
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "d:/"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "sd3_model_v4_rtmp_weights.pkl")

# Halpe26 키포인트 인덱스
HALPE = dict(
    NOSE=0, LEYE=1, REYE=2, LEAR=3, REAR=4,
    LSH=5, RSH=6, LELB=7, RELB=8, LWR=9, RWR=10,
    LHIP=11, RHIP=12, LKNE=13, RKNE=14, LANK=15, RANK=16,
    HEAD=17, NECK=18, HIP=19,
    LBIG_TOE=20, RBIG_TOE=21, LHEEL=24, RHEEL=25
)

# MediaPipe 33 키포인트 인덱스 (호환용)
MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

VIEW_CODE_MAP = {
    'view5': 0.0,
    'view2': 1.0,
    'view3': 2.0,
    'view4': 3.0,
    'view1': 4.0
}

FEATURE_NAMES_36_SD3_RTMP: List[str] = [
    # Group 1: 2D 바벨-하지 수평 이격 거리 동역학 (8D)
    'stiff_d1_bar_shank_dist_max_2d',            # [0] 정강이 구간 바벨-정강이 최대 2D 수평 이격거리
    'stiff_d1_bar_shank_dist_mean_2d',           # [1] 정강이 구간 바벨-정강이 평균 2D 이격거리
    'stiff_d1_bar_knee_dist_pass_2d',            # [2] 무릎 높이 통과 순간 바벨-무릎 2D 거리
    'stiff_d1_bar_thigh_dist_mean_2d',           # [3] 허벅지 구간 바벨-허벅지 평균 2D 이격거리
    'stiff_d1_bar_thigh_dist_max_2d',            # [4] 허벅지 구간 바벨-허벅지 최대 2D 이격거리
    'stiff_d1_bar_midfoot_init_offset_2d',       # [5] 셋업 시작점 바벨-미드풋 2D 정렬 오차
    'stiff_d1_bar_midfoot_max_offset_2d',        # [6] 전 구간 바벨-미드풋 최대 2D 수평 이격거리
    'stiff_d1_bar_leg_contact_deficit_ratio_2d', # [7] 하지 밀착 역치 초과 프레임 비율

    # Group 2: 2D 바벨 수직 궤적 선형성 및 흔들림 (4D)
    'stiff_d1_bar_drift_horizontal_std_2d',      # [8] 바벨 수평 표류 표준편차
    'stiff_d1_bar_path_curvature_2d',            # [9] 바벨 2D 궤적 길이 / 수직 변위 비율
    'stiff_d1_bar_lateral_wobble_std_2d',        # [10] 바벨 좌우 흔들림 표준편차
    'stiff_d1_bar_ascent_velocity_smoothness_2d',# [11] 바벨 수직 상승 속도 부드러움 (저크 억제)

    # Group 3: 2D 상체 모멘트 암 & 어깨-손목 각도 (3D)
    'stiff_d1_shoulder_wrist_angle_max_deg_2d',  # [12] 어깨-손목 벡터 최대 전방 경사각 (deg)
    'stiff_d1_shoulder_wrist_angle_mean_deg_2d', # [13] 리프팅 구간 어깨-손목 평균 전방 경사각 (deg)
    'stiff_d1_hip_bar_moment_arm_at_knee_2d',    # [14] 무릎 높이 골반-바벨 수평 모멘트 암

    # Group 4: 구간별 위상 동역학 (5D)
    'stiff_d1_p1_shank_clearance_rate_2d',       # [15] Phase 1 바닥~무릎 바벨 이탈률
    'stiff_d1_p2_thigh_drag_contact_rate_2d',    # [16] Phase 2 무릎~락아웃 허벅지 밀착 유지율
    'stiff_d1_eccentric_bar_drift_max_2d',       # [17] 편심성 하강 구간 바벨 최대 전방 이탈량
    'stiff_d1_peak_drift_timing_norm_2d',        # [18] 최대 이탈 발생 정규화 시점
    'stiff_d1_knee_pass_timing_norm_2d',         # [19] 무릎 통과 발생 정규화 시점

    # Group 5: 멀티뷰 투영 & 왜곡 보정 (6D)
    'stiff_d1_sagittal_drift_2d',                # [20] 체간 로컬 보정 수평 표류량
    'stiff_d1_coronal_overlap_deficit_2d',       # [21] 정면 뷰(view3) 정강이-바벨 수직 투영 결손도
    'stiff_d1_bilateral_wrist_asym_2d',          # [22] 좌우 손목 수평 비대칭도
    'stiff_d1_view_code',                        # [23] 카메라 시점 범주형 코드 (0~4)
    'stiff_d1_is_frontal',                       # [24] 정면 뷰 지시자 (view3=1.0)
    'stiff_d1_is_rear',                          # [25] 후방 뷰 지시자 (view1, view5=1.0)

    # Group 6: RTMPose 전용 정밀 기하학 (2D 최적화) (6D)
    'stiff_d1_gate_is_oblique',                  # [26] 사각/측면 뷰 지시자 (view2, 4, 1, 5=1.0)
    'stiff_d1_ke_angle_sek_deg_2d',              # [27] 최하사점 어깨-팔꿈치-무릎 2D 사잇각 (deg)
    'stiff_d1_ke_width_ratio_nadir_2d',          # [28] 최하사점 무릎/팔꿈치 너비비율
    'stiff_d1_ke_dist_nadir_2d',                 # [29] 최하사점 무릎-팔꿈치 2D 유클리드 거리
    'stiff_d1_frontal_dwell_ratio',              # [30] 정면 뷰 특화 최하사점 부근 감속 체류비율
    'stiff_d1_ke_horizontal_offset_2d',          # [31] 최하사점 무릎-팔꿈치 2D 수평 오프셋

    # Group 7: RTMPose 특화 하지-바벨 접촉 랜드마크 (4D)
    'stiff_d1_toe_bar_horiz_offset_min',         # [32] 발끝(BigToe)-바벨 최소 수평 거리
    'stiff_d1_torso_scaled_bar_dist_max',        # [33] 체간 길이 정규화 바벨 이탈 최대치
    'stiff_d1_wrist_vert_accel_jerk',            # [34] 손목 수직 가속도 저크도
    'stiff_d1_frontal_arm_compression_ratio'     # [35] 정면 투영 팔 길이 압축률 (전방 뻗음 지표)
]


def extract_sd3_rtmp_features(kps_seq: np.ndarray, view_name: str = 'view2') -> np.ndarray:
    """
    RTMPose Halpe26 (또는 호환 MP33) 시퀀스로부터 sd3 전용 36D 생체역학 피처 추출
    """
    T = kps_seq.shape[0]
    dt = 3.0 / max(1, T - 1)

    if kps_seq.shape[1] == 26:
        l_sh = kps_seq[:, HALPE['LSH'], :2]
        r_sh = kps_seq[:, HALPE['RSH'], :2]
        l_elb = kps_seq[:, HALPE['LELB'], :2]
        r_elb = kps_seq[:, HALPE['RELB'], :2]
        l_wr = kps_seq[:, HALPE['LWR'], :2]
        r_wr = kps_seq[:, HALPE['RWR'], :2]
        l_hip = kps_seq[:, HALPE['LHIP'], :2]
        r_hip = kps_seq[:, HALPE['RHIP'], :2]
        l_kne = kps_seq[:, HALPE['LKNE'], :2]
        r_kne = kps_seq[:, HALPE['RKNE'], :2]
        l_ank = kps_seq[:, HALPE['LANK'], :2]
        r_ank = kps_seq[:, HALPE['RANK'], :2]
        l_toe = kps_seq[:, HALPE['LBIG_TOE'], :2]
        r_toe = kps_seq[:, HALPE['RBIG_TOE'], :2]
        neck = kps_seq[:, HALPE['NECK'], :2]
        pelvis = kps_seq[:, HALPE['HIP'], :2]
    else:
        l_sh = kps_seq[:, MP['LSH'], :2]
        r_sh = kps_seq[:, MP['RSH'], :2]
        l_elb = kps_seq[:, MP['LELB'], :2]
        r_elb = kps_seq[:, MP['RELB'], :2]
        l_wr = kps_seq[:, MP['LWR'], :2]
        r_wr = kps_seq[:, MP['RWR'], :2]
        l_hip = kps_seq[:, MP['LHIP'], :2]
        r_hip = kps_seq[:, MP['RHIP'], :2]
        l_kne = kps_seq[:, MP['LKNE'], :2]
        r_kne = kps_seq[:, MP['RKNE'], :2]
        l_ank = kps_seq[:, MP['LANK'], :2]
        r_ank = kps_seq[:, MP['RANK'], :2]
        l_toe = l_ank.copy()
        r_toe = r_ank.copy()
        neck = (l_sh + r_sh) / 2.0
        pelvis = (l_hip + r_hip) / 2.0

    mid_sh = (l_sh + r_sh) / 2.0
    mid_elb = (l_elb + r_elb) / 2.0
    mid_wr = (l_wr + r_wr) / 2.0
    mid_hip = pelvis if np.linalg.norm(pelvis) > 1e-4 else (l_hip + r_hip) / 2.0
    mid_kne = (l_kne + r_kne) / 2.0
    mid_ank = (l_ank + r_ank) / 2.0
    mid_toe = (l_toe + r_toe) / 2.0
    mid_foot = (mid_ank + mid_toe) / 2.0

    torso_len = np.linalg.norm(neck - mid_hip, axis=-1)
    mean_torso = float(np.mean(torso_len)) if np.mean(torso_len) > 1e-4 else 1.0

    bar_x = mid_wr[:, 0]
    bar_y = mid_wr[:, 1]
    bar_y_s = gaussian_filter1d(bar_y, sigma=1.2)
    bottom_idx = int(np.argmax(bar_y_s))

    # 1. 바벨-정강이/무릎/허벅지 2D 수평 이격거리
    shank_x = (mid_kne[:, 0] + mid_ank[:, 0]) / 2.0
    shank_dist = np.abs(bar_x - shank_x) / mean_torso
    knee_dist = np.abs(bar_x - mid_kne[:, 0]) / mean_torso
    thigh_x = (mid_hip[:, 0] + mid_kne[:, 0]) / 2.0
    thigh_dist = np.abs(bar_x - thigh_x) / mean_torso

    f0_bar_shank_max = float(np.max(shank_dist))
    f1_bar_shank_mean = float(np.mean(shank_dist))

    # 무릎 높이 통과 순간
    knee_diff = np.abs(bar_y - mid_kne[:, 1])
    pass_k_idx = int(np.argmin(knee_diff))
    f2_bar_knee_pass = float(knee_dist[pass_k_idx])

    f3_bar_thigh_mean = float(np.mean(thigh_dist))
    f4_bar_thigh_max = float(np.max(thigh_dist))

    # 미드풋 대비 오프셋
    midfoot_dist = np.abs(bar_x - mid_foot[:, 0]) / mean_torso
    f5_bar_midfoot_init = float(midfoot_dist[0])
    f6_bar_midfoot_max = float(np.max(midfoot_dist))

    f7_contact_deficit_ratio = float(np.mean(shank_dist > 0.12))

    # 2. 바벨 수직 궤적 선형성 및 흔들림
    f8_bar_drift_std = float(np.std(bar_x))
    dx = np.diff(bar_x)
    dy = np.diff(bar_y)
    path_len = np.sum(np.sqrt(dx * dx + dy * dy))
    vert_disp = max(1e-4, float(np.ptp(bar_y_s)))
    f9_curvature = float(path_len / vert_disp)

    f10_wobble_std = float(np.std(l_wr[:, 0] - r_wr[:, 0]))

    bar_vy = np.gradient(bar_y_s, dt)
    bar_ay = np.gradient(bar_vy, dt)
    f11_vel_smoothness = float(np.std(bar_ay))

    # 3. 어깨-손목 벡터 전방 경사각 & 모멘트 암
    arm_dx = bar_x - mid_sh[:, 0]
    arm_dy = bar_y - mid_sh[:, 1]
    arm_pitch = np.degrees(np.arctan2(np.abs(arm_dx), np.maximum(1e-4, arm_dy)))
    f12_sh_wr_angle_max = float(np.max(arm_pitch))
    f13_sh_wr_angle_mean = float(np.mean(arm_pitch))

    hip_bar_moment = np.abs(bar_x - mid_hip[:, 0]) / mean_torso
    f14_moment_arm_knee = float(hip_bar_moment[pass_k_idx])

    # 4. 구간별 위상 동역학
    f15_p1_shank_clearance = float(np.mean(shank_dist[:pass_k_idx+1] > 0.10)) if pass_k_idx > 0 else 0.0
    f16_p2_thigh_drag = float(np.mean(thigh_dist[pass_k_idx:] < 0.15)) if pass_k_idx < T else 1.0
    f17_eccentric_drift_max = float(np.max(shank_dist[:bottom_idx+1])) if bottom_idx > 0 else f0_bar_shank_max
    f18_peak_timing = float(np.argmax(shank_dist) / max(1, T - 1))
    f19_knee_pass_timing = float(pass_k_idx / max(1, T - 1))

    # 5. 멀티뷰 투영 & 왜곡 보정
    f20_sagittal_drift = float(np.std(mid_wr[:, 0] - mid_hip[:, 0]) / mean_torso)
    f21_coronal_overlap = float(np.mean(np.abs(mid_wr[:, 0] - mid_kne[:, 0]) < 0.08))
    f22_bilateral_asym = float(np.mean(np.abs(l_wr[:, 1] - r_wr[:, 1])))

    v_code = float(VIEW_CODE_MAP.get(view_name, 1.0))
    is_front = 1.0 if view_name == 'view3' else 0.0
    is_rear = 1.0 if view_name in ['view1', 'view5'] else 0.0
    is_oblique = 1.0 if view_name in ['view2', 'view4', 'view1', 'view5'] else 0.0

    # 6. RTMPose 전용 무릎-팔꿈치 기하학
    v_es = mid_sh - mid_elb
    v_ek = mid_kne - mid_elb
    dot_ek = np.sum(v_es * v_ek, axis=-1)
    norm_ek = np.linalg.norm(v_es, axis=-1) * np.linalg.norm(v_ek, axis=-1) + 1e-7
    sek_angle = np.degrees(np.arccos(np.clip(dot_ek / norm_ek, -1.0, 1.0)))
    f27_sek_nadir = float(sek_angle[bottom_idx])

    kne_w = np.abs(l_kne[:, 0] - r_kne[:, 0]) + 1e-6
    elb_w = np.abs(l_elb[:, 0] - r_elb[:, 0]) + 1e-6
    f28_width_ratio = float((kne_w / elb_w)[bottom_idx])

    ke_dist_2d = np.linalg.norm(mid_kne - mid_elb, axis=-1) / mean_torso
    f29_ke_dist = float(ke_dist_2d[bottom_idx])

    # 정면 특화 최하사점 감속 체류 비율
    dwell_mask = np.abs(bar_y_s - bar_y_s[bottom_idx]) < (0.05 * vert_disp)
    f30_dwell_ratio = float(np.mean(dwell_mask))

    f31_ke_horiz_offset = float(np.abs(mid_kne[bottom_idx, 0] - mid_elb[bottom_idx, 0]) / mean_torso)

    # 7. RTMPose 특화 정밀 랜드마크 피처
    toe_bar_dist = np.abs(bar_x - mid_toe[:, 0]) / mean_torso
    f32_toe_bar_min = float(np.min(toe_bar_dist))

    f33_torso_scaled_max = float(np.max(np.abs(bar_x - mid_hip[:, 0]) / mean_torso))
    f34_wrist_jerk = float(np.std(np.gradient(bar_ay, dt)))

    # 정면 뷰 팔 2D 투영 압축률: 리프팅 중 최소 투영 길이 / 셋업 투영 길이
    arm_proj_len = np.linalg.norm(mid_sh - mid_wr, axis=-1)
    f35_arm_compression = float(np.min(arm_proj_len) / max(1e-4, arm_proj_len[0]))

    feats = np.array([
        f0_bar_shank_max, f1_bar_shank_mean, f2_bar_knee_pass, f3_bar_thigh_mean, f4_bar_thigh_max,
        f5_bar_midfoot_init, f6_bar_midfoot_max, f7_contact_deficit_ratio,
        f8_bar_drift_std, f9_curvature, f10_wobble_std, f11_vel_smoothness,
        f12_sh_wr_angle_max, f13_sh_wr_angle_mean, f14_moment_arm_knee,
        f15_p1_shank_clearance, f16_p2_thigh_drag, f17_eccentric_drift_max,
        f18_peak_timing, f19_knee_pass_timing,
        f20_sagittal_drift, f21_coronal_overlap, f22_bilateral_asym,
        v_code, is_front, is_rear,
        is_oblique, f27_sek_nadir, f28_width_ratio, f29_ke_dist,
        f30_dwell_ratio, f31_ke_horiz_offset,
        f32_toe_bar_min, f33_torso_scaled_max, f34_wrist_jerk, f35_arm_compression
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_sd3_rtmp_temporal_36d(kps_seq: np.ndarray) -> np.ndarray:
    """(64, 36) RTMPose 12개 주요 관절 시계열 텐서 추출"""
    T = kps_seq.shape[0]
    if kps_seq.shape[1] == 26:
        # Halpe26: 12 key joints x 3 channels (x, y, conf) = 36D
        joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        temp = kps_seq[:, joints, :3].reshape(T, 36)
    else:
        # MP33
        joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        temp = kps_seq[:, joints, :3].reshape(T, 36)
    return np.nan_to_num(temp.astype(np.float32), nan=0.0)


class DilatedTCN(nn.Module):
    def __init__(self, in_channels: int = 36, hidden_dim: int = 32):
        super(DilatedTCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(res1))) + res1
        res3 = self.relu(self.bn3(self.conv3(res2))) + res2
        out = self.pool(res3).squeeze(-1)
        logits = self.fc(out).squeeze(-1)
        return logits


class SD3RTMPClassifier:
    """
    바벨 스티프 데드리프트 3번 오류(sd3: 바벨 궤적과 몸 밀착 위반) RTMPose 전용 공식 v4 분류기
    - 전략 4: Multi-View Dynamic Meta Fusion (36D GBDT Tabular + Dilated-TCN 2-Stream)
    - 3-State 선택적 진단 (CLEAN, DEFECT, RETAKE)
    - 정면(view3) 블라인드 스팟 극복을 위한 시점 적응형 가중 결합 및 임계치 탑재
    """

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd3_rtmp 가중치 번들 부재: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            bundle = pickle.load(fp)

        self.model_name = bundle.get('model_name', 'sd3_model_v4_rtmp')
        self.defect_code = 'sd3'
        self.defect_name = 'sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Proximity)'
        self.feature_names = bundle.get('feature_names', FEATURE_NAMES_36_SD3_RTMP)

        self.lgb_models = bundle['lgb_models']
        self.cb_models = bundle['cb_models']
        self.tcn_weights = bundle['tcn_weights']

        self.view_fusion_weights = bundle.get('view_fusion_weights', {
            'view3': {'gbdt': 0.70, 'tcn': 0.30},
            'view2': {'gbdt': 0.85, 'tcn': 0.15},
            'view4': {'gbdt': 0.85, 'tcn': 0.15},
            'view1': {'gbdt': 0.75, 'tcn': 0.25},
            'view5': {'gbdt': 0.75, 'tcn': 0.25}
        })

        self.view_thresholds = bundle.get('view_thresholds', {
            'view5': (0.37, 0.49),
            'view2': (0.38, 0.49),
            'view3': (0.35, 0.50),
            'view4': (0.38, 0.49),
            'view1': (0.37, 0.49)
        })

        self.binary_threshold = bundle.get('binary_threshold', 0.50)

        # PyTorch TCN 모델 복원
        self.device = torch.device("cpu")
        self.tcn_models = []
        for state in self.tcn_weights:
            m = DilatedTCN(in_channels=36, hidden_dim=32).to(self.device)
            m.load_state_dict(state)
            m.eval()
            self.tcn_models.append(m)

    def extract_features(self, kps_seq: np.ndarray, view_name: str = 'view2') -> Dict[str, np.ndarray]:
        f36 = extract_sd3_rtmp_features(kps_seq, view_name=view_name)
        t36 = extract_sd3_rtmp_temporal_36d(kps_seq)
        return {'features_36d': f36, 'temporal_36d': t36}

    def predict_proba(self, kps_seq: np.ndarray, view_name: str = 'view2') -> float:
        extracted = self.extract_features(kps_seq, view_name=view_name)
        f36 = extracted['features_36d'].reshape(1, -1)
        t36 = extracted['temporal_36d']

        # 1. GBDT 앙상블 (LightGBM 55% + CatBoost 45%)
        lgb_p = float(np.mean([m.predict_proba(f36)[0, 1] for m in self.lgb_models]))
        cb_p = float(np.mean([m.predict_proba(f36)[0, 1] for m in self.cb_models]))
        p_gbdt = 0.55 * lgb_p + 0.45 * cb_p

        # 2. TCN 앙상블
        tcn_x = torch.tensor(t36.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
        tcn_probs = []
        with torch.no_grad():
            for m in self.tcn_models:
                p = torch.sigmoid(m(tcn_x)).item()
                tcn_probs.append(p)
        p_tcn = float(np.mean(tcn_probs))

        # 3. 시점 적응형 Dynamic Meta Fusion
        fw = self.view_fusion_weights.get(view_name, {'gbdt': 0.80, 'tcn': 0.20})
        p_defect = fw['gbdt'] * p_gbdt + fw['tcn'] * p_tcn
        return float(np.clip(p_defect, 0.0, 1.0))

    def diagnose(self, kps_seq: np.ndarray, view_name: str = 'view2') -> Dict[str, Any]:
        p_def = self.predict_proba(kps_seq, view_name=view_name)
        extracted = self.extract_features(kps_seq, view_name=view_name)
        f36 = extracted['features_36d']

        tau_c, tau_d = self.view_thresholds.get(view_name, (0.37, 0.49))

        reasons = []
        if f36[0] > 0.20:
            reasons.append("정강이 구간 바벨 과도 이격")
        if f36[4] > 0.25:
            reasons.append("허벅지 구간 바벨 밀착 실패")
        if f36[12] > 18.0:
            reasons.append("어깨-손목 전방 과도 경사 (광배근 텐션 부족)")

        if p_def < tau_c:
            state = 0
            state_label = "CLEAN"
            feedback = "바벨이 허벅지와 정강이를 따라 수직으로 완벽하게 밀착되어 이동하고 있습니다!"
        elif p_def >= tau_d:
            state = 1
            state_label = "DEFECT"
            if len(reasons) > 0:
                feedback = f"바벨 이격 결함 감지: {', '.join(reasons)}. 광배근에 힘을 주어 바벨을 몸쪽으로 끌어당기며 동작하세요."
            else:
                feedback = "바벨이 다리에서 멀어져 허리에 과도한 부하가 걸리고 있습니다. 바벨을 정강이와 허벅지에 스치듯 밀착시키세요."
        else:
            state = 2
            state_label = "RETAKE"
            feedback = "바벨 궤적이 카메라 앵글상 모호하게 관측되었습니다. 대각선 45도 또는 측면에서 바벨 경로가 잘 보이도록 1회 다시 수행해 주세요."

        return {
            'state': state,
            'state_label': state_label,
            'status': state_label,
            'probability': round(p_def, 4),
            'binary_prediction': int(p_def >= self.binary_threshold),
            'reasons': reasons,
            'feedback': feedback
        }
