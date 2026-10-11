import os
import glob
import numpy as np
import pickle
import sys
import json
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, 'd:/')
sys.path.insert(0, r'D:\src\models\second_models\babel_lunge')

from src.features.lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
from src.models.second_models.babel_lunge.lunge_c2_features_v6 import extract_lunge_c2_features_v6

def get_c4(kps):
    from src.features.lunge_c4_features_v3 import extract_lunge_c4_features_v3, extract_lunge_c4_feature_vector_v3
    try:
        return extract_lunge_c4_feature_vector_v3(kps)
    except:
        f = extract_lunge_c4_features_v3(kps)
        if isinstance(f, dict): return np.array(list(f.values()))
        return f

def get_c5(kps):
    from src.features.lunge_c5_features_v5 import extract_lunge_c5_features_v5, extract_lunge_c5_feature_vector_v5
    try:
        return extract_lunge_c5_feature_vector_v5(kps)
    except:
        f = extract_lunge_c5_features_v5(kps)
        if isinstance(f, dict): return np.array(list(f.values()))
        return f

HALPE_TO_MP = {
    0: 0,
    5: 11, 6: 12,
    7: 13, 8: 14,
    9: 15, 10: 16,
    11: 23, 12: 24,
    13: 25, 14: 26,
    15: 27, 16: 28,
    20: 31, 21: 32,
    24: 29, 25: 30
}

def map_halpe26_to_mp33(kps26):
    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    for halpe_idx, mp_idx in HALPE_TO_MP.items():
        kps33[:, mp_idx, 0] = kps26[:, halpe_idx, 0]
        kps33[:, mp_idx, 1] = kps26[:, halpe_idx, 1]
        kps33[:, mp_idx, 2] = 0.0
    return kps33

rtmp_dir = r"D:/Data/전처리_데이터/RTMPose_Halpe26_64F/04_바벨_런지"
files = glob.glob(os.path.join(rtmp_dir, "*.npz"))

print(f"Found {len(files)} RTMP files.")
if len(files) == 0:
    sys.exit(1)

X_c1, X_c2, X_c4, X_c5 = [], [], [], []
y_labels = []
views = []
subjects = []

count = 0
for f in files:
    fname = os.path.basename(f)
    if "[정상]" in fname:
        y = 0
    else:
        y = 1
        
    parts = fname.replace('.npz', '').split('_')
    subj = parts[2]
    view = parts[-1]
    
    d = np.load(f)
    kps26 = d['kps_26_64f']
    kps33 = map_halpe26_to_mp33(kps26)
    
    try:
        f1 = extract_lunge_c1_feature_vector_v4(kps33)
        f2 = extract_lunge_c2_features_v6(kps33)
        f4 = get_c4(kps33)
        f5 = get_c5(kps33)
        
        X_c1.append(f1)
        X_c2.append(f2)
        X_c4.append(f4)
        X_c5.append(f5)
        y_labels.append(y)
        views.append(view)
        subjects.append(subj)
    except Exception as e:
        pass
    
    count += 1
    if count % 100 == 0:
        print(f"Processed {count}/{len(files)}")

X_c1 = np.array(X_c1)
X_c2 = np.array(X_c2)
X_c4 = np.array(X_c4)
X_c5 = np.array(X_c5)
y_labels = np.array(y_labels)
views = np.array(views)

print("RTMP Extraction done:", X_c1.shape)
np.savez("d:/rtmp_features.npz", X_c1=X_c1, X_c2=X_c2, X_c4=X_c4, X_c5=X_c5, y=y_labels, views=views, subjects=subjects)
