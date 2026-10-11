import numpy as np
from typing import Tuple

from lunge_c2_features_v6 import extract_lunge_c2_features_v6

def detect_lead_leg_2d(kps_seq_2d: np.ndarray) -> Tuple[bool, int]:
    l_knee_y = kps_seq_2d[:, 25, 1]
    r_knee_y = kps_seq_2d[:, 26, 1]
    l_max_y_idx = int(np.argmax(l_knee_y))
    r_max_y_idx = int(np.argmax(r_knee_y))
    
    if l_knee_y[l_max_y_idx] >= r_knee_y[r_max_y_idx]:
        is_left_lead = True
        bottom_idx = l_max_y_idx
    else:
        is_left_lead = False
        bottom_idx = r_max_y_idx

    if abs(l_knee_y[l_max_y_idx] - r_knee_y[r_max_y_idx]) < 0.05:
        is_left_lead = kps_seq_2d[bottom_idx, 27, 1] > kps_seq_2d[bottom_idx, 28, 1]

    return is_left_lead, bottom_idx

def smooth_sequence(seq, window_size=3):
    pad_width = window_size // 2
    padded = np.pad(seq, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
    smoothed = np.zeros_like(seq)
    for i in range(seq.shape[0]):
        smoothed[i] = np.mean(padded[i:i+window_size], axis=0)
    return smoothed

def calc_z_offset(parent_2d_seq, child_2d_seq, sign_z=1.0):
    v_2d = child_2d_seq - parent_2d_seq
    l_2d = np.linalg.norm(v_2d, axis=1)
    L_true = np.percentile(l_2d, 95)
    z_mag = np.sqrt(np.clip(L_true**2 - l_2d**2, 0, None))
    return z_mag * sign_z

def reconstruct_full_skeleton_3d(kps_seq_2d: np.ndarray) -> np.ndarray:
    """
    Reconstructs 33-point 3D skeleton from 2D using bone length constancy
    """
    T = kps_seq_2d.shape[0]
    kps_2d_smooth = smooth_sequence(kps_seq_2d, window_size=3)
    kps_3d = np.zeros((T, 33, 3), dtype=np.float32)
    kps_3d[:, :, :2] = kps_2d_smooth

    is_left_lead, _ = detect_lead_leg_2d(kps_2d_smooth)
    lead_h, lead_k, lead_a, lead_t, lead_heel = (23, 25, 27, 31, 29) if is_left_lead else (24, 26, 28, 32, 30)
    trail_h, trail_k, trail_a, trail_t, trail_heel = (24, 26, 28, 32, 30) if is_left_lead else (23, 25, 27, 31, 29)

    # 1. Hips (Root)
    mid_hip = (kps_2d_smooth[:, 23] + kps_2d_smooth[:, 24]) * 0.5
    # Keep hips at Z=0
    kps_3d[:, 23, 2] = 0.0
    kps_3d[:, 24, 2] = 0.0

    # 2. Shoulders (from mid-hip)
    mid_sh = (kps_2d_smooth[:, 11] + kps_2d_smooth[:, 12]) * 0.5
    z_spine = calc_z_offset(mid_hip, mid_sh, sign_z=1.0)
    kps_3d[:, 11, 2] = z_spine
    kps_3d[:, 12, 2] = z_spine

    # 3. Lead Leg
    # Thigh: points forward (+Z)
    z_lead_thigh = calc_z_offset(kps_2d_smooth[:, lead_h], kps_2d_smooth[:, lead_k], sign_z=1.0)
    kps_3d[:, lead_k, 2] = kps_3d[:, lead_h, 2] + z_lead_thigh
    
    # Shin: Knee to Ankle points backward (-Z) because knee is over toe
    z_lead_shin = calc_z_offset(kps_2d_smooth[:, lead_k], kps_2d_smooth[:, lead_a], sign_z=-1.0)
    kps_3d[:, lead_a, 2] = kps_3d[:, lead_k, 2] + z_lead_shin
    
    # Foot: Ankle to Toe points forward (+Z)
    z_lead_foot = calc_z_offset(kps_2d_smooth[:, lead_a], kps_2d_smooth[:, lead_t], sign_z=1.0)
    kps_3d[:, lead_t, 2] = kps_3d[:, lead_a, 2] + z_lead_foot
    
    # Heel: Ankle to Heel points backward (-Z)
    z_lead_heel = calc_z_offset(kps_2d_smooth[:, lead_a], kps_2d_smooth[:, lead_heel], sign_z=-1.0)
    kps_3d[:, lead_heel, 2] = kps_3d[:, lead_a, 2] + z_lead_heel

    # 4. Trail Leg
    # Thigh: points backward (-Z)
    z_trail_thigh = calc_z_offset(kps_2d_smooth[:, trail_h], kps_2d_smooth[:, trail_k], sign_z=-1.0)
    kps_3d[:, trail_k, 2] = kps_3d[:, trail_h, 2] + z_trail_thigh
    
    # Shin: Knee to Ankle points backward (-Z)
    z_trail_shin = calc_z_offset(kps_2d_smooth[:, trail_k], kps_2d_smooth[:, trail_a], sign_z=-1.0)
    kps_3d[:, trail_a, 2] = kps_3d[:, trail_k, 2] + z_trail_shin
    
    # Foot: Ankle to Toe (+Z), Ankle to Heel (-Z)
    z_trail_foot = calc_z_offset(kps_2d_smooth[:, trail_a], kps_2d_smooth[:, trail_t], sign_z=1.0)
    kps_3d[:, trail_t, 2] = kps_3d[:, trail_a, 2] + z_trail_foot
    
    z_trail_heel = calc_z_offset(kps_2d_smooth[:, trail_a], kps_2d_smooth[:, trail_heel], sign_z=-1.0)
    kps_3d[:, trail_heel, 2] = kps_3d[:, trail_a, 2] + z_trail_heel

    return kps_3d

def extract_lunge_c2_features_rtmp_4(kps_seq_raw: np.ndarray) -> np.ndarray:
    """
    RTMP 2D 데이터를 입력받아 가상 3D 골격을 복원한 후, 
    기존의 최고 성능(SOTA) 피처 추출기(v6)를 그대로 통과시킵니다.
    """
    kps_2d = kps_seq_raw[:, :, :2]
    kps_3d_clean = reconstruct_full_skeleton_3d(kps_2d)
    
    # 기존 v6 피처 추출기 재활용!
    feats = extract_lunge_c2_features_v6(kps_3d_clean)
    return feats

def extract_lunge_c2_features_rtmp_4_batch(X_seqs: np.ndarray) -> np.ndarray:
    N = len(X_seqs)
    feats = np.zeros((N, 18), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_rtmp_4(X_seqs[i])
    return feats
