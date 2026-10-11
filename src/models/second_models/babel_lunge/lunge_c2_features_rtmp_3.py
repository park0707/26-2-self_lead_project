import numpy as np

def detect_lead_leg_2d(kps_seq_2d: np.ndarray):
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

def normalize_2d(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return v / norm

def smooth_sequence(seq, window_size=3):
    pad_width = window_size // 2
    padded = np.pad(seq, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
    smoothed = np.zeros_like(seq)
    for i in range(seq.shape[0]):
        smoothed[i] = np.mean(padded[i:i+window_size], axis=0)
    return smoothed

def reconstruct_3d_vector(v_2d_seq, sign_z=1.0):
    l_2d = np.linalg.norm(v_2d_seq, axis=1)
    L_true = np.percentile(l_2d, 95)
    z_mag = np.sqrt(np.clip(L_true**2 - l_2d**2, 0, None))
    z = z_mag * sign_z
    v_3d = np.column_stack([v_2d_seq, z])
    norms = np.linalg.norm(v_3d, axis=1, keepdims=True) + 1e-6
    return v_3d / norms

def extract_lunge_c2_features_rtmp_3(kps_seq_3d: np.ndarray) -> np.ndarray:
    kps_2d = kps_seq_3d[:, :, :2]
    kps_2d_smooth = smooth_sequence(kps_2d, window_size=3)
    T = kps_2d.shape[0]
    
    is_left_lead, t_bot = detect_lead_leg_2d(kps_2d_smooth)
    
    bot_start = max(0, t_bot - 4)
    bot_end = min(T, t_bot + 5)
    
    if is_left_lead:
        hip_lead, knee_lead, ank_lead, toe_lead = 23, 25, 27, 31
        hip_trail, knee_trail, ank_trail, toe_trail = 24, 26, 28, 32
    else:
        hip_lead, knee_lead, ank_lead, toe_lead = 24, 26, 28, 32
        hip_trail, knee_trail, ank_trail, toe_trail = 23, 25, 27, 31

    # --- Pure 2D Features (from RTMP_2) ---
    kps_bot = np.mean(kps_2d_smooth[bot_start:bot_end], axis=0)
    kps_std = np.mean(kps_2d_smooth[:5], axis=0)
    
    mid_sh_bot = (kps_bot[11] + kps_bot[12]) * 0.5
    mid_hip_bot = (kps_bot[23] + kps_bot[24]) * 0.5
    v_trunk = normalize_2d(mid_sh_bot - mid_hip_bot)
    
    v_lead_thigh = normalize_2d(kps_bot[knee_lead] - kps_bot[hip_lead])
    v_lead_shin = normalize_2d(kps_bot[ank_lead] - kps_bot[knee_lead])
    v_lead_foot = normalize_2d(kps_bot[toe_lead] - kps_bot[ank_lead])
    
    v_trail_thigh = normalize_2d(kps_bot[knee_trail] - kps_bot[hip_trail])
    v_trail_shin = normalize_2d(kps_bot[ank_trail] - kps_bot[knee_trail])
    v_trail_foot = normalize_2d(kps_bot[toe_trail] - kps_bot[ank_trail])
    
    v_knee_toe = normalize_2d(kps_bot[knee_lead] - kps_bot[toe_lead])
    trunk_len = np.linalg.norm(mid_sh_bot - mid_hip_bot) + 1e-6
    dist_knee_toe = np.linalg.norm(kps_bot[knee_lead] - kps_bot[toe_lead]) / trunk_len
    
    v_stance = normalize_2d(kps_bot[ank_lead] - kps_bot[ank_trail])
    dist_stance = np.linalg.norm(kps_bot[ank_lead] - kps_bot[ank_trail]) / trunk_len
    
    v_disp_knee = normalize_2d(kps_bot[knee_lead] - kps_std[knee_lead])
    v_disp_hip = normalize_2d(mid_hip_bot - ((kps_std[23] + kps_std[24]) * 0.5))
    
    cp_lead_knee = v_lead_thigh[0]*v_lead_shin[1] - v_lead_thigh[1]*v_lead_shin[0]
    cp_trail_knee = v_trail_thigh[0]*v_trail_shin[1] - v_trail_thigh[1]*v_trail_shin[0]
    dp_lead_knee = v_lead_thigh[0]*v_lead_shin[0] + v_lead_thigh[1]*v_lead_shin[1]
    dp_trail_knee = v_trail_thigh[0]*v_trail_shin[0] + v_trail_thigh[1]*v_trail_shin[1]
    
    dp_trunk_y = v_trunk[1]
    cp_trunk_y = v_trunk[0]

    # --- New 3D Reconstructed Features (RTMP_3) ---
    v_thigh_seq = kps_2d_smooth[:, knee_lead] - kps_2d_smooth[:, hip_lead]
    v_foot_seq = kps_2d_smooth[:, toe_lead] - kps_2d_smooth[:, ank_lead]
    
    U_thigh = reconstruct_3d_vector(v_thigh_seq, sign_z=1.0)
    U_foot = reconstruct_3d_vector(v_foot_seq, sign_z=1.0)
    
    U_thigh_bot = np.mean(U_thigh[bot_start:bot_end], axis=0)
    U_thigh_bot /= np.linalg.norm(U_thigh_bot) + 1e-6
    
    U_foot_bot = np.mean(U_foot[bot_start:bot_end], axis=0)
    U_foot_bot /= np.linalg.norm(U_foot_bot) + 1e-6
    
    dot_tf = np.clip(np.dot(U_thigh_bot, U_foot_bot), -1.0, 1.0)
    angle_thigh_foot_3d = float(np.degrees(np.arccos(dot_tf)))
    
    yaw_thigh = float(np.degrees(np.arctan2(U_thigh_bot[0], U_thigh_bot[2] + 1e-6)))
    yaw_foot = float(np.degrees(np.arctan2(U_foot_bot[0], U_foot_bot[2] + 1e-6)))
    yaw_diff = yaw_thigh - yaw_foot
    abs_yaw_diff = abs(yaw_diff)
    
    z_mag_thigh_bot = float(U_thigh_bot[2])
    z_mag_foot_bot = float(U_foot_bot[2])

    features = [
        v_trunk[0], v_trunk[1],
        v_lead_thigh[0], v_lead_thigh[1],
        v_lead_shin[0], v_lead_shin[1],
        v_lead_foot[0], v_lead_foot[1],
        v_trail_thigh[0], v_trail_thigh[1],
        v_trail_shin[0], v_trail_shin[1],
        v_trail_foot[0], v_trail_foot[1],
        v_knee_toe[0], v_knee_toe[1],
        dist_knee_toe,
        v_stance[0], v_stance[1],
        dist_stance,
        v_disp_knee[0], v_disp_knee[1],
        v_disp_hip[0], v_disp_hip[1],
        cp_lead_knee, dp_lead_knee,
        cp_trail_knee, dp_trail_knee,
        dp_trunk_y, cp_trunk_y,
        
        # New 7 features
        angle_thigh_foot_3d,
        yaw_thigh,
        yaw_foot,
        yaw_diff,
        abs_yaw_diff,
        z_mag_thigh_bot,
        z_mag_foot_bot
    ]
    
    feats = np.array(features, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats

def extract_lunge_c2_features_rtmp_3_batch(X_seqs: np.ndarray) -> np.ndarray:
    N = len(X_seqs)
    feats = np.zeros((N, 37), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_rtmp_3(X_seqs[i])
    return feats

FEATURE_NAMES_RTMP_3 = [
    "v_trunk_x", "v_trunk_y",
    "v_lead_thigh_x", "v_lead_thigh_y",
    "v_lead_shin_x", "v_lead_shin_y",
    "v_lead_foot_x", "v_lead_foot_y",
    "v_trail_thigh_x", "v_trail_thigh_y",
    "v_trail_shin_x", "v_trail_shin_y",
    "v_trail_foot_x", "v_trail_foot_y",
    "v_knee_toe_x", "v_knee_toe_y",
    "dist_knee_toe",
    "v_stance_x", "v_stance_y",
    "dist_stance",
    "v_disp_knee_x", "v_disp_knee_y",
    "v_disp_hip_x", "v_disp_hip_y",
    "cp_lead_knee", "dp_lead_knee",
    "cp_trail_knee", "dp_trail_knee",
    "dp_trunk_y", "cp_trunk_y",
    "angle_thigh_foot_3d",
    "yaw_thigh",
    "yaw_foot",
    "yaw_diff",
    "abs_yaw_diff",
    "z_mag_thigh_bot",
    "z_mag_foot_bot"
]
