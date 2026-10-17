# -*- coding: utf-8 -*-
import glob
import numpy as np

files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트\*.npz")[:10]
print(f"Loaded {len(files)} AI-Hub files.")

ratios = []
for f in files:
    data = np.load(f)
    kps = data["kps_33_3d_64f"]
    # 프레임 0 기립 상태
    sh_w = np.linalg.norm(kps[0, 11] - kps[0, 12])
    thigh = (np.linalg.norm(kps[0, 23] - kps[0, 25]) + np.linalg.norm(kps[0, 24] - kps[0, 26])) / 2.0
    torso = np.linalg.norm((kps[0, 11] + kps[0, 12])/2.0 - (kps[0, 23] + kps[0, 24])/2.0)
    ratios.append((sh_w, thigh, torso, sh_w / thigh, torso / thigh))

sh_w_m = np.mean([r[0] for r in ratios])
thigh_m = np.mean([r[1] for r in ratios])
torso_m = np.mean([r[2] for r in ratios])
ratio_st_m = np.mean([r[3] for r in ratios])
ratio_tt_m = np.mean([r[4] for r in ratios])

print(f"AI-Hub Average Body Metrics:")
print(f"  - Shoulder Width : {sh_w_m:.4f} m (약 {sh_w_m*100:.1f} cm)")
print(f"  - Thigh Length   : {thigh_m:.4f} m (약 {thigh_m*100:.1f} cm)")
print(f"  - Torso Length   : {torso_m:.4f} m (약 {torso_m*100:.1f} cm)")
print(f"  - Shoulder / Thigh Ratio : {ratio_st_m:.4f}")
print(f"  - Torso / Thigh Ratio    : {ratio_tt_m:.4f}")
