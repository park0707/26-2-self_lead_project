import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from lunge_c2_features_v4 import detect_lead_leg_v4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
v4_cache = os.path.join(CURRENT_DIR, "lunge_c2_features_v4_cache.npz")
feat_cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_v6_18d_delta_cache.npz")

c4 = np.load(v4_cache, allow_pickle=True)
clip_names = c4["clip_names"]
y = c4["y"]
c6 = np.load(feat_cache_file, allow_pickle=True)
v6_feats = c6["X"]

features = []
for name in clip_names:
    p = os.path.join(prep_dir, name + ".npz")
    data = np.load(p)
    seq = data["kps_33_3d_64f"]
    if seq.shape[-1] == 4:
        seq = seq[:, :, :3]
        
    is_left_lead, t_bot = detect_lead_leg_v4(seq)
    
    hip_l = 23
    hip_r = 24
    heel_idx = 29 if is_left_lead else 30
    toe_idx = 31 if is_left_lead else 32
    
    # Body lateral vector (Left hip to right hip)
    v_lat = np.array([seq[t_bot, hip_r, 0] - seq[t_bot, hip_l, 0], 
                      seq[t_bot, hip_r, 2] - seq[t_bot, hip_l, 2]])
    norm_lat = np.linalg.norm(v_lat) + 1e-6
    u_lat = v_lat / norm_lat
    
    # Body forward vector (orthogonal to lateral)
    u_fwd = np.array([-u_lat[1], u_lat[0]])
    
    # Foot vector
    v_foot = np.array([seq[t_bot, toe_idx, 0] - seq[t_bot, heel_idx, 0],
                       seq[t_bot, toe_idx, 2] - seq[t_bot, heel_idx, 2]])
    norm_foot = np.linalg.norm(v_foot) + 1e-6
    u_foot = v_foot / norm_foot
    
    dot = np.clip(np.dot(u_foot, u_fwd), -1.0, 1.0)
    angle = np.arccos(dot) * 180.0 / np.pi
    angle_from_line = min(angle, 180.0 - angle)
    features.append(angle_from_line)

features = np.array(features)
auc = roc_auc_score(y, features)
if auc < 0.5:
    auc = 1 - auc
print(f"Absolute Foot Heading Angle Feature AUC: {auc:.4f}")

v6_auc = roc_auc_score(y, v6_feats[:, 16])
v6_auc = max(v6_auc, 1 - v6_auc)
print(f"v6 Feature 17 (Knee vs Foot) AUC: {v6_auc:.4f}")

import json
with open("result.json", "w") as f:
    json.dump({"new_auc": auc, "v6_auc": v6_auc}, f)
