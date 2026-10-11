# -*- coding: utf-8 -*-
"""
Test script: Applying cl2_final (cl2_hybrid_2) 67D feature pipeline & architecture to Barbell Lunge (bl2)
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    safe_savgol_filter_kps, build_pacs_basis, compute_cross_lunge_geometric_alignment,
    vector_angle_3d, FEATURE_NAMES_51D
)
from src.models.second_models.cross_lunge.cl2_hybrid_1 import (
    RTM_UNIQUE_FEATURE_NAMES_29D,
    CROSS_MODAL_FEATURE_NAMES_6D,
    CL2_HYBRID_1_FEATURE_NAMES,
    normalize_2d_vec,
    reconstruct_3d_vector
)
from src.models.second_models.cross_lunge.cl2_hybrid_2 import (
    CL2_HYBRID_2_FEATURE_NAMES,
    HYBRID_1_TO_2_INDICES
)
from src.models.second_models.babel_lunge.lunge_c2_features_v4 import detect_lead_leg_v4

# Load bl2 metadata
c = np.load(r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v4_cache.npz", allow_pickle=True)
clip_names = c["clip_names"]
y = c["y"]
views = c["views"]
subjects = c["subjects"]

mp_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
rtmp_dir = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\04_바벨_런지"

cache_bl2_cl2_hybrid = r"D:\src\models\second_models\babel_lunge\bl2_cl2_hybrid_67d_cache.npz"

def extract_bl2_with_cl2_pipeline(mp33_seq, rtm26_seq, view_code="view3"):
    """Adapts cl2_hybrid_1 feature extractor to barbell lunge (using detect_lead_leg_v4)."""
    mp_clean = safe_savgol_filter_kps(mp33_seq[:, :, :3], window=7, poly=2)
    T, V, C = mp_clean.shape
    
    # 1. Lead leg for MP (Barbell lunge)
    is_left_lead_mp, t_bot_mp = detect_lead_leg_v4(mp_clean)
    t_bot_mp = max(5, min(T - 5, t_bot_mp))
    
    hip_idx = 23 if is_left_lead_mp else 24
    knee_idx = 25 if is_left_lead_mp else 26
    ank_idx = 27 if is_left_lead_mp else 28
    heel_idx = 29 if is_left_lead_mp else 30
    toe_idx = 31 if is_left_lead_mp else 32
    
    kps_stand = np.mean(mp_clean[:4], axis=0)
    kps_bot = mp_clean[t_bot_mp]
    
    R_pacs_bot, _ = build_pacs_basis(kps_bot)
    R_pacs_stand, _ = build_pacs_basis(kps_stand)
    
    # --- MP 51D features ---
    v_thigh_bot = kps_bot[knee_idx] - kps_bot[hip_idx]
    v_thigh_pacs = R_pacs_bot @ v_thigh_bot
    v_shank_bot = kps_bot[ank_idx] - kps_bot[knee_idx]
    v_shank_pacs = R_pacs_bot @ v_shank_bot
    v_foot_bot = kps_bot[toe_idx] - kps_bot[heel_idx]
    v_foot_pacs = R_pacs_bot @ v_foot_bot
    
    pacs_trunk_knee_azimuth_error = float(np.degrees(np.arctan2(abs(v_thigh_pacs[0]), max(abs(v_thigh_pacs[2]), 1e-4))))
    u_fwd_pacs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pacs_thigh_heading_to_trunk_angle = vector_angle_3d(v_thigh_pacs, u_fwd_pacs)
    
    # pacs_knee_ankle_coronal_tilt
    v_shank_coronal = np.array([v_shank_pacs[0], v_shank_pacs[1], 0.0], dtype=np.float32)
    pacs_knee_ankle_coronal_tilt = vector_angle_3d(v_shank_coronal, np.array([0.0, -1.0, 0.0], dtype=np.float32))
    
    pacs_lead_hip_rotation_proxy = float(np.degrees(np.arctan2(v_thigh_pacs[0], max(abs(v_thigh_pacs[1]), 1e-4))))
    
    mech_axis = kps_bot[ank_idx] - kps_bot[hip_idx]
    norm_mech = np.linalg.norm(mech_axis) + 1e-6
    u_mech = mech_axis / norm_mech
    proj_len = np.dot(kps_bot[knee_idx] - kps_bot[hip_idx], u_mech)
    proj_pt = kps_bot[hip_idx] + proj_len * u_mech
    pacs_dkcd_medial_deviation = float(np.linalg.norm(kps_bot[knee_idx] - proj_pt) / norm_mech)
    
    pacs_fppa_3d_collapse = 0.0
    pacs_lead_foot_yaw_angle = float(np.degrees(np.arctan2(v_foot_pacs[0], max(abs(v_foot_pacs[2]), 1e-4))))
    pacs_heel_toe_skew_ratio = float((kps_bot[toe_idx, 0] - kps_bot[heel_idx, 0]) / (abs(kps_bot[toe_idx, 2] - kps_bot[heel_idx, 2]) + 1e-4))
    
    # Shank torsion
    n_femur_knee = np.cross(kps_bot[hip_idx] - kps_bot[knee_idx], kps_bot[ank_idx] - kps_bot[knee_idx])
    norm_fn = np.linalg.norm(n_femur_knee) + 1e-6
    u_fn = n_femur_knee / norm_fn
    v_ank_heel = kps_bot[heel_idx] - kps_bot[ank_idx]
    shank_torsion_plane_angle = vector_angle_3d(u_fn, v_ank_heel)
    
    v_knee_disp = kps_bot[knee_idx] - kps_stand[knee_idx]
    knee_trajectory_to_ankle_vector_angle = vector_angle_3d(v_knee_disp, kps_bot[ank_idx] - kps_stand[ank_idx])
    
    knee_pts_desc = mp_clean[:t_bot_mp, knee_idx, :]
    if len(knee_pts_desc) > 3:
        p_start, p_end = knee_pts_desc[0], knee_pts_desc[-1]
        line_vec = p_end - p_start
        norm_lv = np.linalg.norm(line_vec) + 1e-6
        u_lv = line_vec / norm_lv
        dists = [np.linalg.norm(np.cross(pt - p_start, u_lv)) for pt in knee_pts_desc]
        knee_trajectory_linearity_rmse = float(np.sqrt(np.mean(np.square(dists))))
    else:
        knee_trajectory_linearity_rmse = 0.0
        
    rear_heel_ankle_knee_alignment = vector_angle_3d(kps_bot[heel_idx] - kps_bot[ank_idx], kps_bot[ank_idx] - kps_bot[knee_idx])
    lead_knee_mediolateral_sway_variance = float(np.var(mp_clean[:t_bot_mp, knee_idx, 0]))
    
    opp_knee_idx = 24 if is_left_lead_mp else 23
    opp_ank_idx = 28 if is_left_lead_mp else 27
    d_knee = np.linalg.norm(kps_bot[knee_idx] - kps_bot[opp_knee_idx])
    d_ank = np.linalg.norm(kps_bot[ank_idx] - kps_bot[opp_ank_idx]) + 1e-6
    inter_knee_to_ankle_span_ratio = float(d_knee / d_ank)
    
    d_shank_sag = abs(kps_bot[knee_idx, 2] - kps_stand[knee_idx, 2])
    d_shank_cor = abs(kps_bot[knee_idx, 0] - kps_stand[knee_idx, 0]) + 1e-6
    lead_shank_sagittal_progression_ratio = float(d_shank_sag / d_shank_cor)
    
    lead_foot_heel_contact_stability = float(np.std(mp_clean[t_bot_mp-3:t_bot_mp+4, heel_idx, 1]))
    
    # Delta foot yaw
    v_foot_std = kps_stand[toe_idx] - kps_stand[heel_idx]
    v_foot_std_pacs = R_pacs_stand @ v_foot_std
    yaw_std = float(np.degrees(np.arctan2(v_foot_std_pacs[0], max(abs(v_foot_std_pacs[2]), 1e-4))))
    delta_foot_yaw_standing_to_bottom = float(pacs_lead_foot_yaw_angle - yaw_std)
    
    knee_v_ml = np.diff(mp_clean[:t_bot_mp, knee_idx, 0])
    eccentric_knee_inward_velocity_peak = float(np.max(abs(knee_v_ml))) if len(knee_v_ml) > 0 else 0.0
    
    # 3D Q-angle
    q_std = vector_angle_3d(kps_stand[knee_idx] - kps_stand[hip_idx], kps_stand[ank_idx] - kps_stand[knee_idx])
    q_bot = vector_angle_3d(kps_bot[knee_idx] - kps_bot[hip_idx], kps_bot[ank_idx] - kps_bot[knee_idx])
    delta_q_angle_standing_to_bottom = float(q_bot - q_std)
    
    bottom_phase_alignment_variance = float(np.var([
        vector_angle_3d(mp_clean[t, knee_idx] - mp_clean[t, hip_idx], mp_clean[t, toe_idx] - mp_clean[t, heel_idx])
        for t in range(max(0, t_bot_mp-3), min(T, t_bot_mp+4))
    ]))
    
    if t_bot_mp < T - 4:
        knee_up = mp_clean[t_bot_mp:, knee_idx, :]
        acc = np.diff(knee_up, n=2, axis=0)
        concentric_knee_rebound_jerk = float(np.mean(np.linalg.norm(acc, axis=1))) if len(acc) > 0 else 0.0
    else:
        concentric_knee_rebound_jerk = 0.0
        
    v_pelvis_seq = mp_clean[:, 23, :2] - mp_clean[:, 24, :2]
    v_sh_seq = mp_clean[:, 11, :2] - mp_clean[:, 12, :2]
    ang_torsion = [abs(np.degrees(np.arctan2(v_pelvis_seq[t, 1], v_pelvis_seq[t, 0]) - np.arctan2(v_sh_seq[t, 1], v_sh_seq[t, 0]))) for t in range(t_bot_mp)]
    trunk_pelvis_yaw_torsion_mean = float(np.mean(ang_torsion)) if len(ang_torsion) > 0 else 0.0
    
    pelvic_yaws = [np.degrees(np.arctan2(v_pelvis_seq[t, 1], v_pelvis_seq[t, 0])) for t in range(T)]
    pelvic_rotation_dynamic_range = float(np.ptp(pelvic_yaws))
    
    fppa_pre_bottom_slope = 0.0
    foot_axis_temporal_stability = float(np.std([np.degrees(np.arctan2(mp_clean[t, toe_idx, 0] - mp_clean[t, heel_idx, 0], mp_clean[t, toe_idx, 2] - mp_clean[t, heel_idx, 2])) for t in range(T)]))
    
    # View one-hot
    v1 = 1.0 if view_code == 'view1' else 0.0
    v2 = 1.0 if view_code == 'view2' else 0.0
    v3 = 1.0 if view_code == 'view3' else 0.0
    v4 = 1.0 if view_code == 'view4' else 0.0
    v5 = 1.0 if view_code == 'view5' else 0.0
    
    # Geometric alignment
    dot_tk, dot_kf, dot_tf, ang_tf, ang_tk, ang_kf = compute_cross_lunge_geometric_alignment(
        kps_bot, hip_idx, knee_idx, heel_idx, toe_idx
    )
    _, _, dot_tf_std, _, _, _ = compute_cross_lunge_geometric_alignment(
        kps_stand, hip_idx, knee_idx, heel_idx, toe_idx
    )
    delta_tf = float(dot_tf - dot_tf_std)
    
    optical_axis_foreshortening = float(np.linalg.norm(kps_bot[knee_idx, :2] - kps_bot[hip_idx, :2]) / (np.linalg.norm(kps_bot[knee_idx] - kps_bot[hip_idx]) + 1e-6))
    shank_optical_collinearity = float(vector_angle_3d(kps_bot[ank_idx] - kps_bot[knee_idx], np.array([0.0, 0.0, 1.0])))
    
    # sl2_v7 temporal features
    t30 = int(t_bot_mp * 0.3)
    t60 = int(t_bot_mp * 0.6)
    d_tk_30, _, d_tf_30, _, _, _ = compute_cross_lunge_geometric_alignment(mp_clean[t30], hip_idx, knee_idx, heel_idx, toe_idx)
    _, _, d_tf_60, _, _, _ = compute_cross_lunge_geometric_alignment(mp_clean[t60], hip_idx, knee_idx, heel_idx, toe_idx)
    
    dots_desc = [compute_cross_lunge_geometric_alignment(mp_clean[t], hip_idx, knee_idx, heel_idx, toe_idx)[2] for t in range(t_bot_mp)]
    temporal_descent_median = float(np.median(dots_desc)) if len(dots_desc) > 0 else 0.0
    temporal_descent_std = float(np.std(dots_desc)) if len(dots_desc) > 0 else 0.0
    
    ray_proj_heel_toe_collinearity = 0.0
    ray_proj_transverse_deviation = float(abs(kps_bot[toe_idx, 0] - kps_bot[heel_idx, 0]))
    ray_proj_occlusion_likelihood_score = 0.0
    kinematic_shank_torsion_angle = 0.0
    kinematic_reconstructed_foot_yaw = 0.0
    kinematic_torso_shank_alignment = float(np.dot(v_thigh_pacs[:2] / (np.linalg.norm(v_thigh_pacs[:2]) + 1e-6), v_shank_pacs[:2] / (np.linalg.norm(v_shank_pacs[:2]) + 1e-6)))
    kinematic_foot_vs_observed_deviation = 0.0
    
    f_mp_51d = [
        pacs_trunk_knee_azimuth_error, pacs_thigh_heading_to_trunk_angle, pacs_knee_ankle_coronal_tilt,
        pacs_lead_hip_rotation_proxy, pacs_dkcd_medial_deviation, pacs_fppa_3d_collapse,
        pacs_lead_foot_yaw_angle, pacs_heel_toe_skew_ratio,
        shank_torsion_plane_angle, knee_trajectory_to_ankle_vector_angle, knee_trajectory_linearity_rmse,
        rear_heel_ankle_knee_alignment, lead_knee_mediolateral_sway_variance, inter_knee_to_ankle_span_ratio,
        lead_shank_sagittal_progression_ratio, lead_foot_heel_contact_stability,
        delta_foot_yaw_standing_to_bottom, eccentric_knee_inward_velocity_peak, delta_q_angle_standing_to_bottom,
        bottom_phase_alignment_variance, concentric_knee_rebound_jerk, trunk_pelvis_yaw_torsion_mean,
        pelvic_rotation_dynamic_range, fppa_pre_bottom_slope, foot_axis_temporal_stability,
        v1, v2, v3, v4, v5,
        dot_tk, dot_kf, dot_tf, ang_tf, ang_tk, ang_kf, delta_tf,
        optical_axis_foreshortening, shank_optical_collinearity,
        d_tf_30, d_tk_30, d_tf_60, temporal_descent_median, temporal_descent_std,
        ray_proj_heel_toe_collinearity, ray_proj_transverse_deviation, ray_proj_occlusion_likelihood_score,
        kinematic_shank_torsion_angle, kinematic_reconstructed_foot_yaw, kinematic_torso_shank_alignment,
        kinematic_foot_vs_observed_deviation
    ]
    
    # --- RTMPose 29D features ---
    kps_2d = rtm26_seq[:, :, :2]
    conf = rtm26_seq[:, :, 2] if rtm26_seq.shape[-1] >= 3 else np.ones((T, 26))
    
    # Lead leg for RTMPose
    l_knee_y = kps_2d[:, 13, 1]
    r_knee_y = kps_2d[:, 14, 1]
    t_l = int(np.argmax(l_knee_y))
    t_r = int(np.argmax(r_knee_y))
    if is_left_lead_mp:
        hip_l, knee_l, ank_l = 11, 13, 15
        toe_big_l, toe_sml_l, heel_l = 20, 22, 24
    else:
        hip_l, knee_l, ank_l = 12, 14, 16
        toe_big_l, toe_sml_l, heel_l = 21, 23, 25
        
    t_bot_rtm = max(4, min(T - 5, t_bot_mp))
    win = slice(max(0, t_bot_rtm - 3), min(T, t_bot_rtm + 4))
    k_std = np.mean(kps_2d[:5], axis=0)
    k_bot = np.mean(kps_2d[win], axis=0)
    
    pelvis_bot = k_bot[19] if conf[t_bot_rtm, 19] > 0.3 else (k_bot[11] + k_bot[12]) * 0.5
    neck_bot = k_bot[18] if conf[t_bot_rtm, 18] > 0.3 else (k_bot[5] + k_bot[6]) * 0.5
    v_trunk_rtm = normalize_2d_vec(neck_bot - pelvis_bot)
    trunk_len_rtm = np.linalg.norm(neck_bot - pelvis_bot) + 1e-6
    
    v_thigh_rtm = normalize_2d_vec(k_bot[knee_l] - k_bot[hip_l])
    v_shank_rtm = normalize_2d_vec(k_bot[ank_l] - k_bot[knee_l])
    mid_toe_bot = (k_bot[toe_big_l] + k_bot[toe_sml_l]) * 0.5
    v_foot_rtm = normalize_2d_vec(mid_toe_bot - k_bot[heel_l])
    
    cp_trunk_thigh = float(v_trunk_rtm[0]*v_thigh_rtm[1] - v_trunk_rtm[1]*v_thigh_rtm[0])
    dp_trunk_thigh = float(v_trunk_rtm[0]*v_thigh_rtm[0] + v_trunk_rtm[1]*v_thigh_rtm[1])
    cp_thigh_foot = float(v_thigh_rtm[0]*v_foot_rtm[1] - v_thigh_rtm[1]*v_foot_rtm[0])
    dp_thigh_foot = float(v_thigh_rtm[0]*v_foot_rtm[0] + v_thigh_rtm[1]*v_foot_rtm[1])
    cp_trunk_foot = float(v_trunk_rtm[0]*v_foot_rtm[1] - v_trunk_rtm[1]*v_foot_rtm[0])
    dp_trunk_foot = float(v_trunk_rtm[0]*v_foot_rtm[0] + v_trunk_rtm[1]*v_foot_rtm[1])
    cp_shank_foot = float(v_shank_rtm[0]*v_foot_rtm[1] - v_shank_rtm[1]*v_foot_rtm[0])
    dp_shank_foot = float(v_shank_rtm[0]*v_foot_rtm[0] + v_shank_rtm[1]*v_foot_rtm[1])
    cp_thigh_shank = float(v_thigh_rtm[0]*v_shank_rtm[1] - v_thigh_rtm[1]*v_shank_rtm[0])
    dp_thigh_shank = float(v_thigh_rtm[0]*v_shank_rtm[0] + v_thigh_rtm[1]*v_shank_rtm[1])
    
    toe_heel_x_diff = float((mid_toe_bot[0] - k_bot[heel_l][0]) / trunk_len_rtm)
    toe_heel_y_diff = float((mid_toe_bot[1] - k_bot[heel_l][1]) / trunk_len_rtm)
    foot_width = float(np.linalg.norm(k_bot[toe_sml_l] - k_bot[toe_big_l]) / trunk_len_rtm)
    ankle_knee_x_offset = float((k_bot[ank_l][0] - k_bot[knee_l][0]) / trunk_len_rtm)
    knee_toe_x_offset = float((mid_toe_bot[0] - k_bot[knee_l][0]) / trunk_len_rtm)
    
    v_foot_std = normalize_2d_vec((k_std[toe_big_l] + k_std[toe_sml_l]) * 0.5 - k_std[heel_l])
    delta_foot_dp = float(np.dot(v_foot_rtm, v_foot_std))
    delta_foot_cp = float(v_foot_rtm[0]*v_foot_std[1] - v_foot_rtm[1]*v_foot_std[0])
    
    sh_diff_y = float((k_bot[5][1] - k_bot[6][1]) / trunk_len_rtm)
    sh_diff_x = float(abs(k_bot[5][0] - k_bot[6][0]) / trunk_len_rtm)
    hip_diff_y = float((k_bot[11][1] - k_bot[12][1]) / trunk_len_rtm)
    hip_diff_x = float(abs(k_bot[11][0] - k_bot[12][0]) / trunk_len_rtm)
    torso_coronal_angle = float(np.degrees(np.arctan2(k_bot[5][1] - k_bot[6][1], k_bot[5][0] - k_bot[6][0])))
    
    head_yaw_asym = 0.0
    nose_to_ear_l = 0.0
    nose_to_ear_r = 0.0
    
    v_thigh_seq = kps_2d[:, knee_l] - kps_2d[:, hip_l]
    v_shank_seq = kps_2d[:, ank_l] - kps_2d[:, knee_l]
    v_foot_seq = kps_2d[:, toe_big_l] - kps_2d[:, heel_l]
    u3_thigh = reconstruct_3d_vector(v_thigh_seq)
    u3_shank = reconstruct_3d_vector(v_shank_seq)
    u3_foot = reconstruct_3d_vector(v_foot_seq)
    
    dp3_thigh_shank = float(np.dot(u3_thigh[t_bot_rtm], u3_shank[t_bot_rtm]))
    dp3_thigh_foot = float(np.dot(u3_thigh[t_bot_rtm], u3_foot[t_bot_rtm]))
    dp3_shank_foot = float(np.dot(u3_shank[t_bot_rtm], u3_foot[t_bot_rtm]))
    
    f_rtm_29d = [
        cp_trunk_thigh, dp_trunk_thigh,
        cp_thigh_foot, dp_thigh_foot,
        cp_trunk_foot, dp_trunk_foot,
        cp_shank_foot, dp_shank_foot,
        cp_thigh_shank, dp_thigh_shank,
        toe_heel_x_diff, toe_heel_y_diff,
        foot_width, ankle_knee_x_offset, knee_toe_x_offset,
        delta_foot_dp, delta_foot_cp,
        sh_diff_y, sh_diff_x, hip_diff_y, hip_diff_x, torso_coronal_angle,
        head_yaw_asym, nose_to_ear_l, nose_to_ear_r,
        dp3_thigh_shank, dp3_thigh_foot, dp3_shank_foot,
        float(is_left_lead_mp)
    ]
    
    # --- Cross-Modal 6D ---
    sh_mid_mp = (mp_clean[t_bot_mp, 11, :2] + mp_clean[t_bot_mp, 12, :2]) * 0.5
    hip_mid_mp = (mp_clean[t_bot_mp, 23, :2] + mp_clean[t_bot_mp, 24, :2]) * 0.5
    v_trunk_mp_2d = normalize_2d_vec(sh_mid_mp - hip_mid_mp)
    v_thigh_mp_2d = normalize_2d_vec(mp_clean[t_bot_mp, knee_idx, :2] - mp_clean[t_bot_mp, hip_idx, :2])
    
    cross_mp_trunk_rtm_foot_dp = float(np.dot(v_trunk_mp_2d, v_foot_rtm))
    cross_mp_trunk_rtm_foot_cp = float(v_trunk_mp_2d[0]*v_foot_rtm[1] - v_trunk_mp_2d[1]*v_foot_rtm[0])
    cross_mp_thigh_rtm_foot_dp = float(np.dot(v_thigh_mp_2d, v_foot_rtm))
    cross_mp_thigh_rtm_foot_cp = float(v_thigh_mp_2d[0]*v_foot_rtm[1] - v_thigh_mp_2d[1]*v_foot_rtm[0])
    cross_mp_rtm_knee_x_diff = float(abs(mp_clean[t_bot_mp, knee_idx, 0] - k_bot[knee_l, 0]))
    cross_mp_rtm_lead_leg_agree = 1.0
    
    f_cross_6d = [
        cross_mp_trunk_rtm_foot_dp, cross_mp_trunk_rtm_foot_cp,
        cross_mp_thigh_rtm_foot_dp, cross_mp_thigh_rtm_foot_cp,
        cross_mp_rtm_knee_x_diff, cross_mp_rtm_lead_leg_agree
    ]
    
    f_86d = np.array(f_mp_51d + f_rtm_29d + f_cross_6d, dtype=np.float32)
    f_67d = f_86d[HYBRID_1_TO_2_INDICES]
    return np.nan_to_num(f_67d, nan=0.0)

print("[*] Checking 67D extraction cache...")
if os.path.exists(cache_bl2_cl2_hybrid):
    print("[+] Loading from cache:", cache_bl2_cl2_hybrid)
    c_hyb = np.load(cache_bl2_cl2_hybrid, allow_pickle=True)
    X_67d = c_hyb["X"]
else:
    print(f"[*] Extracting cl2_final 67D features for {len(clip_names)} Barbell Lunge clips...")
    feats = []
    for name, v_code in zip(clip_names, views):
        p_mp = os.path.join(mp_dir, name + ".npz")
        p_rtm = os.path.join(rtmp_dir, name + ".npz")
        d_mp = np.load(p_mp)
        d_rtm = np.load(p_rtm)
        f_vec = extract_bl2_with_cl2_pipeline(d_mp["kps_33_3d_64f"], d_rtm["kps_26_64f"], view_code=v_code)
        feats.append(f_vec)
    X_67d = np.array(feats, dtype=np.float32)
    np.savez_compressed(cache_bl2_cl2_hybrid, X=X_67d, y=y, views=views, subjects=subjects, clip_names=clip_names)
    print(f"[+] Saved 67D features to cache: shape={X_67d.shape}")

print(f"[+] Dataset Ready: {len(X_67d)} clips, {X_67d.shape[1]}D features")

# --- Training cl2_final Architecture on bl2 ---
print("\n[*] Starting 5-Fold SGKF Training with cl2_final Architecture (CatBoost + XGBoost + LightGBM)...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_cb = np.zeros(len(y))
oof_preds_xgb = np.zeros(len(y))
oof_preds_lgb = np.zeros(len(y))

n_pos = np.sum(y == 1)
n_neg = np.sum(y == 0)
base_scale = float(n_neg) / float(n_pos)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_67d, y, groups=subjects), 1):
    X_tr, y_tr = X_67d[train_idx], y[train_idx]
    X_va, y_va = X_67d[val_idx], y[val_idx]
    
    cb = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.03, l2_leaf_reg=4.0, auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=-1)
    cb.fit(X_tr, y_tr)
    oof_preds_cb[val_idx] = cb.predict_proba(X_va)[:, 1]
    
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    params_xgb = {'max_depth': 4, 'eta': 0.03, 'objective': 'binary:logistic', 'scale_pos_weight': base_scale * 1.2, 'eval_metric': 'logloss', 'seed': 42+fold}
    bst = xgb.train(params_xgb, dtr, num_boost_round=180)
    oof_preds_xgb[val_idx] = bst.predict(dva)
    
    clf_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=4, num_leaves=13, learning_rate=0.03, scale_pos_weight=base_scale * 1.2, random_state=42+fold, verbose=-1, n_jobs=-1)
    clf_lgb.fit(X_tr, y_tr)
    oof_preds_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

# Weights from cl2_final: [0.8862, 0.0, 0.1138]
oof_ens = 0.8862 * oof_preds_cb + 0.1138 * oof_preds_xgb
overall_auc = roc_auc_score(y, oof_ens)
print(f"[+] cl2_final Ensemble OOF ROC-AUC: {overall_auc*100:.2f}%")

# Apply cl2_final Per-View Adaptive Optimization
print("\n" + "=" * 80)
print("  [cl2_final 방식: Barbell Lunge 시점별 3-State 적응형 캘리브레이션 결과]")
print("=" * 80)

view_alias = {
    "view3": "시점 C (정면 0°)",
    "view2": "시점 B (우측 전방 45°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)",
    "view5": "시점 A (우측 후방 45°)"
}

mask_3s_overall = np.zeros(len(y), dtype=bool)
preds_3s_overall = np.zeros(len(y), dtype=int)

for v in ["view3", "view2", "view4", "view1", "view5"]:
    vm = (views == v)
    y_v = y[vm]
    p_v = oof_ens[vm]
    auc_v = roc_auc_score(y_v, p_v)
    
    # 2-state optimal threshold
    best_th = 0.5
    best_f1_2s = 0.0
    for th in np.arange(0.20, 0.70, 0.005):
        s = f1_score(y_v, (p_v >= th).astype(int), zero_division=0)
        if s > best_f1_2s:
            best_f1_2s = s
            best_th = th
            
    # 3-state optimal threshold
    best_3s_f1 = best_f1_2s
    best_cfg = (best_th - 0.001, best_th + 0.001, 0.0, 1.0)
    for tau_c in np.arange(max(0.15, best_th - 0.15), best_th, 0.005):
        for tau_d in np.arange(best_th, min(0.85, best_th + 0.15), 0.005):
            if tau_c >= tau_d: continue
            m = (p_v <= tau_c) | (p_v >= tau_d)
            unc = 1.0 - np.mean(m)
            if unc > 0.10: continue
            s = f1_score(y_v[m], (p_v[m] >= tau_d).astype(int), zero_division=0)
            c_f1 = f1_score(1 - y_v[m], (p_v[m] < tau_c).astype(int), zero_division=0)
            if c_f1 >= 0.50 and s > best_3s_f1:
                best_3s_f1 = s
                best_cfg = (float(tau_c), float(tau_d), float(unc), float(np.mean(m)))
                
    m_v = vm & ((oof_ens <= best_cfg[0]) | (oof_ens >= best_cfg[1]))
    mask_3s_overall |= m_v
    preds_3s_overall[m_v] = (oof_ens[m_v] >= best_cfg[1]).astype(int)
    
    print(f"[*] {view_alias[v]:<25} | 2-State F1: {best_f1_2s*100:.2f}% | 3-State F1: {best_3s_f1*100:.2f}% | tau_c={best_cfg[0]:.3f}, tau_d={best_cfg[1]:.3f} | Retake: {best_cfg[2]*100:.1f}% | AUC: {auc_v:.4f}")

overall_3s_f1 = f1_score(y[mask_3s_overall], preds_3s_overall[mask_3s_overall], zero_division=0)
overall_3s_clean = f1_score(1 - y[mask_3s_overall], 1 - preds_3s_overall[mask_3s_overall], zero_division=0)
overall_3s_acc = accuracy_score(y[mask_3s_overall], preds_3s_overall[mask_3s_overall])
overall_3s_cov = np.mean(mask_3s_overall)

print(f"\n[+] cl2_final 이식 통합 3-State 결과:")
print(f"    Selective Defect F1 : {overall_3s_f1*100:.2f}%")
print(f"    Selective Clean F1  : {overall_3s_clean*100:.2f}%")
print(f"    판정 정확도 (Acc)    : {overall_3s_acc*100:.2f}%")
print(f"    판정률 (Coverage)    : {overall_3s_cov*100:.2f}% (보류율: {(1-overall_3s_cov)*100:.2f}%)")
