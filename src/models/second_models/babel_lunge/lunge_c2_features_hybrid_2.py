import numpy as np
from lunge_c2_features_v4 import detect_lead_leg_v4
from lunge_c2_features_rtmp_4 import detect_lead_leg_2d
from lunge_c2_features_hybrid import extract_hybrid_features

def extract_frontal_biomechanics_2d(kps_2d: np.ndarray, is_left_lead: bool, t_bot: int) -> np.ndarray:
    """
    Extracts 5 advanced DKV/In-toeing features for Frontal View.
    kps_2d: (64, 33, 2) sequence.
    """
    T = kps_2d.shape[0]
    bot_start = max(0, t_bot - 4)
    bot_end = min(T, t_bot + 5)
    
    side_modifier = -1.0 if is_left_lead else 1.0
    
    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32

    # Pelvis Width at bottom
    pelvis_w = np.linalg.norm(kps_2d[t_bot, 23] - kps_2d[t_bot, 24]) + 1e-6
    
    # 1. NPKD (Normalized Perpendicular Knee Deviation) at bottom window
    npkds = []
    for t in range(bot_start, bot_end):
        H = kps_2d[t, hip_idx]
        K = kps_2d[t, knee_idx]
        A = kps_2d[t, ank_idx]
        
        dev = (A[0] - H[0])*(H[1] - K[1]) - (H[0] - K[0])*(A[1] - H[1])
        dist_HA = np.linalg.norm(H - A) + 1e-6
        npkd = (dev * side_modifier) / (dist_HA * pelvis_w)
        npkds.append(npkd)
    npkd_bottom = np.mean(npkds)
    
    # 2 & 3. MKTO (Medial Knee-to-Toe Horizontal Offset)
    mkto_seq = (kps_2d[:, knee_idx, 0] - kps_2d[:, toe_idx, 0]) * side_modifier / pelvis_w
    mkto_bottom = np.mean(mkto_seq[bot_start:bot_end])
    
    # MKTO velocity during eccentric phase (t=0 to t_bot)
    if t_bot > 2:
        mkto_diff = np.diff(mkto_seq[:t_bot])
        mkto_velocity = np.max(mkto_diff) if len(mkto_diff) > 0 else 0.0
    else:
        mkto_velocity = 0.0
        
    # 4 & 5. NTMD (Normalized Toe-to-Heel Deviation)
    ntmd_seq = []
    for t in range(T):
        toe = kps_2d[t, toe_idx]
        heel = kps_2d[t, heel_idx]
        dist_ht = np.linalg.norm(heel - toe) + 1e-6
        ntmd = (toe[0] - heel[0]) * side_modifier / dist_ht
        ntmd_seq.append(ntmd)
    ntmd_seq = np.array(ntmd_seq)
    
    ntmd_bottom = np.mean(ntmd_seq[bot_start:bot_end])
    ntmd_standing = np.mean(ntmd_seq[:5])
    
    return np.array([npkd_bottom, mkto_bottom, mkto_velocity, ntmd_bottom, ntmd_standing], dtype=np.float32)

def extract_hybrid_2_features(mp_kps_3d: np.ndarray, rtmp_kps_2d: np.ndarray) -> np.ndarray:
    """
    Extracts 46D Hybrid Features: 
    - 36D from hybrid_1 (18D MP + 18D RTMP)
    - 5D MP Frontal Biomechanics
    - 5D RTMP Frontal Biomechanics
    """
    # 36D Base Features
    base_36 = extract_hybrid_features(mp_kps_3d, rtmp_kps_2d)
    
    # MediaPipe Frontal (5D)
    mp_2d = mp_kps_3d[:, :, :2]
    is_left_lead_mp, t_bot_mp = detect_lead_leg_v4(mp_kps_3d)
    mp_frontal = extract_frontal_biomechanics_2d(mp_2d, is_left_lead_mp, t_bot_mp)
    
    # RTMP Frontal (5D)
    from lunge_c2_features_hybrid import map_halpe26_to_mp33_2d
    rtmp_mp_format = map_halpe26_to_mp33_2d(rtmp_kps_2d)
    is_left_lead_rtmp, t_bot_rtmp = detect_lead_leg_2d(rtmp_mp_format)
    rtmp_frontal = extract_frontal_biomechanics_2d(rtmp_mp_format, is_left_lead_rtmp, t_bot_rtmp)
    
    hybrid_46 = np.concatenate([base_36, mp_frontal, rtmp_frontal])
    hybrid_46 = np.nan_to_num(hybrid_46, nan=0.0, posinf=0.0, neginf=0.0)
    return hybrid_46

def extract_hybrid_2_features_batch(mp_seqs: np.ndarray, rtmp_seqs: np.ndarray) -> np.ndarray:
    N = len(mp_seqs)
    feats = np.zeros((N, 46), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_hybrid_2_features(mp_seqs[i], rtmp_seqs[i])
    return feats

FEATURE_NAMES_FRONTAL = [
    "npkd_bottom",
    "mkto_bottom",
    "mkto_velocity",
    "ntmd_bottom",
    "ntmd_standing"
]
