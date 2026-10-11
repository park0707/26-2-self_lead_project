# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    safe_savgol_filter_kps, build_pacs_basis, compute_cross_lunge_geometric_alignment,
    vector_angle_3d, FEATURE_NAMES_51D
)
from src.models.second_models.babel_lunge.lunge_c2_features_v4 import detect_lead_leg_v4

# Load babel lunge dataset
c = np.load(r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v4_cache.npz", allow_pickle=True)
clip_names = c["clip_names"]
y = c["y"]
views = c["views"]
subjects = c["subjects"]

mp_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"

# Test 1 clip
p0 = os.path.join(mp_dir, clip_names[0] + ".npz")
d0 = np.load(p0)
kps0 = d0["kps_33_3d_64f"]
print(f"[+] Loaded clip {clip_names[0]}: shape={kps0.shape}")

# Let's inspect cl2's Group E geometric alignment features on babel lunge!
def extract_bl_cl2_geom(kps_seq):
    kps_clean = safe_savgol_filter_kps(kps_seq[:, :, :3], window=7, poly=2)
    is_left, t_bot = detect_lead_leg_v4(kps_clean)
    t_bot = max(5, min(len(kps_clean) - 5, t_bot))
    
    if is_left:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        
    dot_tk, dot_kf, dot_tf, ang_tf, ang_tk, ang_kf = compute_cross_lunge_geometric_alignment(
        kps_clean[t_bot], hip_idx, knee_idx, heel_idx, toe_idx
    )
    
    # Standing baseline
    _, _, dot_tf_std, _, _, _ = compute_cross_lunge_geometric_alignment(
        np.mean(kps_clean[:4], axis=0), hip_idx, knee_idx, heel_idx, toe_idx
    )
    delta_tf = float(dot_tf - dot_tf_std)
    
    return [dot_tk, dot_kf, dot_tf, ang_tf, ang_tk, ang_kf, delta_tf]

# Extract on first 50 clips to test
feats_sample = []
for name in clip_names[:50]:
    p = os.path.join(mp_dir, name + ".npz")
    d = np.load(p)
    feats_sample.append(extract_bl_cl2_geom(d["kps_33_3d_64f"]))

feats_sample = np.array(feats_sample)
print(f"[+] Extracted 7 geometric alignment features on 50 clips: shape={feats_sample.shape}")
print("[+] Sample feature values:", feats_sample[0])
