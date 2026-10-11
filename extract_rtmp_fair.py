# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import time
import numpy as np
import concurrent.futures
from tqdm import tqdm

import sys
sys.path.insert(0, 'd:/')
sys.path.insert(0, r'D:\src\models\second_models\babel_lunge')

from src.features.lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
from src.models.second_models.babel_lunge.lunge_c2_features_v6 import extract_lunge_c2_features_v6
from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_14d_features_v4
from src.features.lunge_c4_features_v3 import extract_lunge_c4_features_v3
from src.features.lunge_c5_features_v5 import extract_lunge_c5_v5_feature_vector

RTMP_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\04_바벨_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_NPZ_PATH = r"D:\rtmp_fair_dataset.npz"

def load_label_map():
    label_map = {}
    json_files = glob.glob(os.path.join(LABELS_JSON_DIR, "*.json"))
    for jf in json_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            tinfo = data.get("type_info", {})
            conds = tinfo.get("conditions", [])
            
            y_c1 = 1 if len(conds) > 0 and conds[0].get("value") is False else 0
            y_c2 = 1 if len(conds) > 1 and conds[1].get("value") is False else 0
            y_c3 = 1 if len(conds) > 2 and conds[2].get("value") is False else 0
            y_c4 = 1 if len(conds) > 3 and conds[3].get("value") is False else 0
            y_c5 = 1 if len(conds) > 4 and conds[4].get("value") is False else 0
            
            label_map[base_name] = {
                "y_c1": y_c1, "y_c2": y_c2, "y_c3": y_c3, "y_c4": y_c4, "y_c5": y_c5
            }
        except Exception:
            pass
    return label_map

def init_worker():
    pass

def process_file_fair(f):
    fname = os.path.basename(f)
    # e.g., [결함]_D03_여1_Key330_view1.npz -> [결함]_D03_여1_Key330
    base_name = fname.split("_view")[0]
    parts = fname.replace('.npz', '').split('_')
    subj = int("".join(c for c in parts[2] if c.isdigit()))
    view = parts[-1]
    
    try:
        d = np.load(f)
        kps26 = d['kps_26_64f'] # (64, 26, 3)
        # map to MP 33
        kps33 = np.zeros((64, 33, 3), dtype=np.float32)
        kps33[:, 0] = kps26[:, 0]
        kps33[:, 2] = kps26[:, 1]
        kps33[:, 5] = kps26[:, 2]
        kps33[:, 7] = kps26[:, 3]
        kps33[:, 8] = kps26[:, 4]
        kps33[:, 11] = kps26[:, 5]
        kps33[:, 12] = kps26[:, 6]
        kps33[:, 13] = kps26[:, 7]
        kps33[:, 14] = kps26[:, 8]
        kps33[:, 15] = kps26[:, 9]
        kps33[:, 16] = kps26[:, 10]
        kps33[:, 23] = kps26[:, 11]
        kps33[:, 24] = kps26[:, 12]
        kps33[:, 25] = kps26[:, 13]
        kps33[:, 26] = kps26[:, 14]
        kps33[:, 27] = kps26[:, 15]
        kps33[:, 28] = kps26[:, 16]
        kps33[:, 29] = kps26[:, 24] # LEFT_HEEL
        kps33[:, 30] = kps26[:, 25] # RIGHT_HEEL
        kps33[:, 31] = kps26[:, 20] # LEFT_FOOT_INDEX
        kps33[:, 32] = kps26[:, 21] # RIGHT_FOOT_INDEX
        
        # [RTMP Optimization] Inject fake Z depth to restore 3D cross-product math for Torso Pitch / Yaw
        # Left joints get Z = -0.1, Right joints get Z = 0.1
        left_mp_indices = [1,2,3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        right_mp_indices = [4,5,6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        kps33[:, left_mp_indices, 2] = -0.1
        kps33[:, right_mp_indices, 2] = 0.1
        
        feat_c1 = extract_lunge_c1_feature_vector_v4(kps33)
        feat_c2 = extract_lunge_c2_features_v6(kps33)
        feat_c4_dict = extract_lunge_c4_features_v3(kps33); feat_c4 = np.array(list(feat_c4_dict.values())) if isinstance(feat_c4_dict, dict) else feat_c4_dict
        feat_c5 = extract_lunge_c5_v5_feature_vector(kps33)
        
        # map to 15 keypoints for c3
        kps15 = np.zeros((64, 15, 2), dtype=np.float32)
        kps15[:, 0] = (kps26[:, 11, :2] + kps26[:, 12, :2]) / 2.0
        kps15[:, 1] = (kps26[:, 5, :2] + kps26[:, 6, :2]) / 2.0
        kps15[:, 8] = kps26[:, 11, :2]
        kps15[:, 9] = kps26[:, 12, :2]
        kps15[:, 10] = kps26[:, 13, :2]
        kps15[:, 11] = kps26[:, 14, :2]
        kps15[:, 12] = kps26[:, 15, :2]
        kps15[:, 13] = kps26[:, 16, :2]
        
        feat_c3 = extract_lunge_bl3_14d_features_v4(kps15)
        
        return (fname, base_name, feat_c1, feat_c2, feat_c3, feat_c4, feat_c5, subj, view)
    except Exception as e:
        return None

def main():
    label_map = load_label_map()
    print(f"Loaded {len(label_map)} labels from JSON.", flush=True)
    
    files = glob.glob(os.path.join(RTMP_DIR, "*.npz"))
    print(f"Found {len(files)} RTMP files.", flush=True)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10, initializer=init_worker) as executor:
        for res in tqdm(executor.map(process_file_fair, files), total=len(files)):
            if res is not None:
                fname, base_name, fc1, fc2, fc3, fc4, fc5, subj, view = res
                if base_name in label_map:
                    lab = label_map[base_name]
                    results.append({
                        'fname': fname,
                        'subj': subj,
                        'view': view,
                        'fc1': fc1, 'fc2': fc2, 'fc3': fc3, 'fc4': fc4, 'fc5': fc5,
                        'y_c1': lab['y_c1'], 'y_c2': lab['y_c2'], 'y_c3': lab['y_c3'], 
                        'y_c4': lab['y_c4'], 'y_c5': lab['y_c5']
                    })
    
    print(f"Extracted {len(results)} valid samples with labels.")
    
    fnames = np.array([r['fname'] for r in results])
    subjects = np.array([r['subj'] for r in results])
    views = np.array([r['view'] for r in results])
    
    X_c1 = np.array([r['fc1'] for r in results])
    X_c2 = np.array([r['fc2'] for r in results])
    X_c3 = np.array([r['fc3'] for r in results])
    X_c4 = np.array([r['fc4'] for r in results])
    X_c5 = np.array([r['fc5'] for r in results])
    
    y_c1 = np.array([r['y_c1'] for r in results])
    y_c2 = np.array([r['y_c2'] for r in results])
    y_c3 = np.array([r['y_c3'] for r in results])
    y_c4 = np.array([r['y_c4'] for r in results])
    y_c5 = np.array([r['y_c5'] for r in results])
    
    np.savez(OUT_NPZ_PATH, 
             fnames=fnames, subjects=subjects, views=views,
             X_c1=X_c1, X_c2=X_c2, X_c3=X_c3, X_c4=X_c4, X_c5=X_c5,
             y_c1=y_c1, y_c2=y_c2, y_c3=y_c3, y_c4=y_c4, y_c5=y_c5)
    
    print(f"Saved to {OUT_NPZ_PATH}")

if __name__ == '__main__':
    main()
