import os
import sys
import glob
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_rtmp_3 import (
    extract_lunge_c5_features_rtmp_3_vector,
    LUNGE_C5_RTMP_3_FEATURE_NAMES
)

RAW_RTMP_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\04_바벨_런지"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\lunge_c5_rtmp_3_features.npz"

def build_rtmp_3_cache():
    print("Building RTMP 3 cache for c5")
    npz_files = sorted(glob.glob(os.path.join(RAW_RTMP_DIR, "*.npz")))
    
    X_list, y_list, subj_list, view_list, key_list, file_list = [], [], [], [], [], []
    
    for nf in npz_files:
        try:
            data = np.load(nf, allow_pickle=True)
            if 'kps_26_64f' in data:
                kps_64f = data['kps_26_64f']
            elif 'keypoints_64f' in data:
                kps_64f = data['keypoints_64f']
            else:
                continue
                
            if 'label' in data:
                target_label = int(data['label'])
            else:
                continue
                
            subj = int(data.get('subject', 1))
            vn = str(data.get('view', 'view1'))
            fname = os.path.basename(nf)
            
            if kps_64f.shape[1] == 26:
                mp33 = np.zeros((64, 33, 3), dtype=np.float32)
                mp33[:, 0] = kps_64f[:, 0]
                mp33[:, 7] = kps_64f[:, 3]
                mp33[:, 8] = kps_64f[:, 4]
                mp33[:, 11] = kps_64f[:, 5]
                mp33[:, 12] = kps_64f[:, 6]
                mp33[:, 13] = kps_64f[:, 7]
                mp33[:, 14] = kps_64f[:, 8]
                mp33[:, 15] = kps_64f[:, 9]
                mp33[:, 16] = kps_64f[:, 10]
                mp33[:, 23] = kps_64f[:, 11]
                mp33[:, 24] = kps_64f[:, 12]
                mp33[:, 25] = kps_64f[:, 13]
                mp33[:, 26] = kps_64f[:, 14]
                mp33[:, 27] = kps_64f[:, 15]
                mp33[:, 28] = kps_64f[:, 16]
                mp33[:, 31] = kps_64f[:, 20]
                mp33[:, 32] = kps_64f[:, 21]
                kps_64f = mp33
                
            feat_vec = extract_lunge_c5_features_rtmp_3_vector(kps_64f)
            if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
                feat_vec = np.nan_to_num(feat_vec, nan=0.0)
                
            X_list.append(feat_vec)
            y_list.append(target_label)
            subj_list.append(subj)
            view_list.append(vn)
            key_list.append("329")
            file_list.append(fname)
        except Exception as e:
            print(f"Error on {nf}: {e}")
            continue
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    
    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=X_arr, y=y_arr,
        subjects=np.array(subj_list, dtype=np.int32),
        views=np.array(view_list, dtype=object),
        keys=np.array(key_list, dtype=object),
        filenames=np.array(file_list, dtype=object),
        feature_names=np.array(LUNGE_C5_RTMP_3_FEATURE_NAMES, dtype=object)
    )
    print(f"[+] Saved cache to {OUT_NPZ_PATH}. Total samples: {len(X_arr)}")

if __name__ == '__main__':
    build_rtmp_3_cache()
