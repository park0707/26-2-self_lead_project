# -*- coding: utf-8 -*-
import os
import numpy as np
from lunge_c2_features_v4 import detect_lead_leg_v4

prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
c = np.load("lunge_c2_features_v4_cache.npz", allow_pickle=True)

for name in c["clip_names"][:10]:
    p = os.path.join(prep_dir, name + ".npz")
    d = np.load(p)
    kps = d["kps_33_3d_64f"][:, :, :3]
    is_left, t_bot = detect_lead_leg_v4(kps)
    knee_idx = 25 if is_left else 26
    toe_idx = 31 if is_left else 32
    heel_idx = 29 if is_left else 30
    hip_idx = 23 if is_left else 24
    ank_idx = 27 if is_left else 28

    v_foot = kps[t_bot, toe_idx, [0, 2]] - kps[t_bot, heel_idx, [0, 2]]
    v_disp = kps[t_bot, knee_idx, [0, 2]] - kps[0, knee_idx, [0, 2]]
    v_thigh = kps[t_bot, knee_idx, [0, 2]] - kps[t_bot, hip_idx, [0, 2]]
    v_shank = kps[t_bot, ank_idx, [0, 2]] - kps[t_bot, knee_idx, [0, 2]]

    u_f = v_foot / (np.linalg.norm(v_foot) + 1e-6)
    u_d = v_disp / (np.linalg.norm(v_disp) + 1e-6)
    u_t = v_thigh / (np.linalg.norm(v_thigh) + 1e-6)
    u_s = v_shank / (np.linalg.norm(v_shank) + 1e-6)

    ang_disp = np.degrees(np.arccos(np.clip(np.dot(u_f, u_d), -1, 1)))
    ang_thigh = np.degrees(np.arccos(np.clip(np.dot(u_f, u_t), -1, 1)))
    ang_shank = np.degrees(np.arccos(np.clip(np.dot(u_f, u_s), -1, 1)))

    side = "L" if is_left else "R"
    print(f"{name[-15:]} (Lead {side}): Disp={ang_disp:5.1f}°, Thigh-Foot={ang_thigh:5.1f}°, Shank-Foot={ang_shank:5.1f}° | v_disp_xz={v_disp}")
