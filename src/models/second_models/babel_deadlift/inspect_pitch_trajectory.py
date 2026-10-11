# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_V4_1_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

spine_cache = np.load(SPINEPOSE_V4_1_CACHE, allow_pickle=True)
spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

# 피험자 1 (정상 / 결함), 피험자 5 (정상 / 결함), 피험자 7 (FP 케이스) 몇 개 시퀀스 조사
test_sids = [
    # 피험자 7 FP 케이스
    "[오류]_D03_피험자7_Key286_view5",
    "[정상]_D03_피험자7_Key281_view5",
    # 피험자 5 FN 케이스
    "[오류]_D03_피험자5_Key287_view5",
    "[정상]_D03_피험자5_Key281_view5",
    # 피험자 2 TP 케이스
    "[오류]_D03_피험자2_Key282_view2",
    "[정상]_D03_피험자2_Key281_view2",
]

MP = {'LSH': 11, 'RSH': 12, 'LHIP': 23, 'RHIP': 24, 'LKNE': 25, 'RKNE': 26, 'LANK': 27, 'RANK': 28, 'LWR': 15, 'RWR': 16}

for sid in test_sids:
    npz_p = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
    if not os.path.exists(npz_p):
        continue
    kps = np.load(npz_p)['kps_33_3d_64f']
    mid_sh = (kps[:, MP['LSH']] + kps[:, MP['RSH']]) / 2.0
    mid_hip = (kps[:, MP['LHIP']] + kps[:, MP['RHIP']]) / 2.0
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))

    mid_wrist = (kps[:, MP['LWR']] + kps[:, MP['RWR']]) / 2.0
    wrist_y = mid_wrist[:, 1]

    s9 = spine_dict.get(sid, None)
    
    print(f"=== {sid} ===")
    print(f"  torso_pitch: start(F0)={torso_pitch[0]:.1f}, mid(F32)={torso_pitch[32]:.1f}, end(F63)={torso_pitch[63]:.1f}, min={np.min(torso_pitch):.1f}, max={np.max(torso_pitch):.1f}")
    print(f"  wrist_y: start={wrist_y[0]:.3f}, mid={wrist_y[32]:.3f}, end={wrist_y[63]:.3f}, min={np.min(wrist_y):.3f}, max={np.max(wrist_y):.3f}")
    if s9 is not None:
        # C7, T8, L3, S1 y 좌표
        c7_y = s9[:, 6, 1]
        s1_y = s9[:, 0, 1]
        print(f"  spine: C7_y(F32)={c7_y[32]:.3f}, S1_y(F32)={s1_y[32]:.3f} (C7-S1 dist={np.mean(np.linalg.norm(s9[:, 6] - s9[:, 0], axis=-1)):.3f})")
