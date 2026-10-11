import numpy as np
import os
import pickle
import sys

def get_angle(A, B, C):
    ba = A - B
    bc = C - B
    cosine = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

path = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
with open(path, 'rb') as f:
    c = pickle.load(f)
df = c['df']

# The directory where RTMPose files actually live on this OS:
RTM_DIR = r'D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트'

y_clean = []
for i, row in df.iterrows():
    sid = os.path.splitext(os.path.basename(row['path']))[0]
    p = os.path.join(RTM_DIR, sid + '.npz')
    if os.path.exists(p):
        kps = np.load(p)['kps_26_64f']
        rk = get_angle(kps[:, 12, :2], kps[:, 14, :2], kps[:, 16, :2])
        lk = get_angle(kps[:, 11, :2], kps[:, 13, :2], kps[:, 15, :2])
        min_k = min(np.min(rk), np.min(lk))
        if min_k <= 155.0:
            y_clean.append(1)
        else:
            y_clean.append(0)
    else:
        print(f"Missing: {p}")
        y_clean.append(0)

y_clean = np.array(y_clean, dtype=np.int32)
np.save(r'D:\src\models\second_models\babel_stiff_deadlift\y_clean.npy', y_clean)
print(f"Saved y_clean.npy with shape {y_clean.shape}, defects: {y_clean.sum()}")
