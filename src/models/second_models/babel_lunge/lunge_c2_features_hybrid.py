import numpy as np
from lunge_c2_features_rtmp_4 import reconstruct_full_skeleton_3d
from lunge_c2_features_v6 import extract_lunge_c2_features_v6

def map_halpe26_to_mp33_2d(halpe_kps: np.ndarray) -> np.ndarray:
    """
    halpe_kps: (64, 26, 2)
    Returns: mp_kps: (64, 33, 2)
    """
    T = halpe_kps.shape[0]
    mp_kps = np.zeros((T, 33, 2), dtype=np.float32)
    halpe_kps = halpe_kps[:, :, :2]
    
    mp_kps[:, 23] = halpe_kps[:, 11] # LHip
    mp_kps[:, 24] = halpe_kps[:, 12] # RHip
    mp_kps[:, 25] = halpe_kps[:, 13] # LKnee
    mp_kps[:, 26] = halpe_kps[:, 14] # RKnee
    mp_kps[:, 27] = halpe_kps[:, 15] # LAnkle
    mp_kps[:, 28] = halpe_kps[:, 16] # RAnkle
    mp_kps[:, 29] = halpe_kps[:, 24] # LHeel
    mp_kps[:, 30] = halpe_kps[:, 25] # RHeel
    mp_kps[:, 31] = halpe_kps[:, 20] # LToe (BigToe)
    mp_kps[:, 32] = halpe_kps[:, 21] # RToe (BigToe)
    mp_kps[:, 11] = halpe_kps[:, 5]  # LShoulder
    mp_kps[:, 12] = halpe_kps[:, 6]  # RShoulder
    
    return mp_kps

def extract_hybrid_features(mp_kps_3d: np.ndarray, rtmp_kps_2d: np.ndarray) -> np.ndarray:
    """
    Extracts 36D Hybrid Features: 18D from MediaPipe + 18D from RTMP
    mp_kps_3d: (64, 33, 3) from MediaPipe
    rtmp_kps_2d: (64, 26, 2) from RTMPose
    """
    # 1. MediaPipe 18D Features
    mp_feats = extract_lunge_c2_features_v6(mp_kps_3d)
    
    # 2. RTMP 18D Features
    rtmp_mp_format = map_halpe26_to_mp33_2d(rtmp_kps_2d)
    rtmp_3d_clean = reconstruct_full_skeleton_3d(rtmp_mp_format)
    rtmp_feats = extract_lunge_c2_features_v6(rtmp_3d_clean)
    
    # 3. Concatenate (36D)
    hybrid_feats = np.concatenate([mp_feats, rtmp_feats])
    return hybrid_feats
