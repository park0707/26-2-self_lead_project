import os
import glob
import numpy as np
import pickle
import sys

sys.path.insert(0, 'd:/')
sys.path.insert(0, r'D:\src\models\second_models\babel_lunge')

HALPE_TO_MP = {
    0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    20: 31, 21: 32, 24: 29, 25: 30
}

def process_file(f):
    from src.features.lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
    from src.models.second_models.babel_lunge.lunge_c2_features_v6 import extract_lunge_c2_features_v6
    from src.features.lunge_c4_features_v3 import extract_lunge_c4_features_v3
    from src.features.lunge_c5_features_v5 import extract_lunge_c5_v5_feature_vector

    fname = os.path.basename(f)
    y = 0 if "[정상]" in fname else 1
    parts = fname.replace('.npz', '').split('_')
    subj = parts[2]
    view = parts[-1]
    
    d = np.load(f)
    kps26 = d['kps_26_64f']
    
    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    for halpe_idx, mp_idx in HALPE_TO_MP.items():
        kps33[:, mp_idx, 0] = kps26[:, halpe_idx, 0]
        kps33[:, mp_idx, 1] = kps26[:, halpe_idx, 1]
        kps33[:, mp_idx, 2] = 0.0

    try:
        f1 = extract_lunge_c1_feature_vector_v4(kps33)
    except: f1 = np.zeros(30)
    
    try:
        f2 = extract_lunge_c2_features_v6(kps33)
    except: f2 = np.zeros(18)
    
    try:
        f4_dict = extract_lunge_c4_features_v3(kps33)
        if isinstance(f4_dict, dict): f4 = np.array(list(f4_dict.values()))
        else: f4 = f4_dict
    except: f4 = np.zeros(24)
            
    try:
        f5 = extract_lunge_c5_v5_feature_vector(kps33)
    except: f5 = np.zeros(36)

    return f1, f2, f4, f5, y, view, subj

if __name__ == '__main__':
    rtmp_dir = r"D:/Data/전처리_데이터/RTMPose_Halpe26_64F/04_바벨_런지"
    files = glob.glob(os.path.join(rtmp_dir, "*.npz"))
    
    X_c1, X_c2, X_c4, X_c5, y_labels, views, subjects = [], [], [], [], [], [], []
    
    for i, f in enumerate(files):
        try:
            r = process_file(f)
            X_c1.append(r[0])
            X_c2.append(r[1])
            X_c4.append(r[2])
            X_c5.append(r[3])
            y_labels.append(r[4])
            views.append(r[5])
            subjects.append(r[6])
        except Exception as e:
            pass
            
    np.savez("d:/rtmp_features.npz", 
             X_c1=np.array(X_c1), X_c2=np.array(X_c2), X_c4=np.array(X_c4), X_c5=np.array(X_c5), 
             y=np.array(y_labels), views=np.array(views), subjects=np.array(subjects))
    print("Done extracting.")
