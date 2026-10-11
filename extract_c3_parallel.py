# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, 'd:/')
sys.path.insert(0, r'D:\src\models\second_models\babel_lunge')

def init_worker():
    global extract_c3
    from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_14d_features_v4
    extract_c3 = extract_lunge_bl3_14d_features_v4

def process_file_c3(f):
    fname = os.path.basename(f)
    y = 0 if "[정상]" in fname else 1
    parts = fname.replace('.npz', '').split('_')
    subj = int("".join(c for c in parts[2] if c.isdigit()))
    
    d = np.load(f)
    kps26 = d['kps_26_64f'] # (64, 26, 3)
    
    kps15 = np.zeros((64, 15, 2), dtype=np.float32)
    kps15[:, 8] = kps26[:, 11, :2] # L Hip
    kps15[:, 9] = kps26[:, 12, :2] # R Hip
    kps15[:, 10] = kps26[:, 13, :2] # L Knee
    kps15[:, 11] = kps26[:, 14, :2] # R Knee
    kps15[:, 12] = kps26[:, 15, :2] # L Ankle
    kps15[:, 13] = kps26[:, 16, :2] # R Ankle
    kps15[:, 0] = (kps15[:, 8] + kps15[:, 9]) / 2.0 # Pelvis
    kps15[:, 1] = (kps26[:, 5, :2] + kps26[:, 6, :2]) / 2.0 # Torso
    
    try:
        f3 = extract_c3(kps15)
    except:
        f3 = np.zeros(14)
        
    return f3, y, subj

if __name__ == '__main__':
    rtmp_dir = r"D:/Data/전처리_데이터/RTMPose_Halpe26_64F/04_바벨_런지"
    files = glob.glob(os.path.join(rtmp_dir, "*.npz"))
    
    X_c3, y_labels, groups = [], [], []
    
    with ProcessPoolExecutor(max_workers=10, initializer=init_worker) as executor:
        futures = {executor.submit(process_file_c3, f): f for f in files}
        for future in as_completed(futures):
            f3, y, g = future.result()
            X_c3.append(f3)
            y_labels.append(y)
            groups.append(g)
            
    X_c3 = np.array(X_c3)
    y_labels = np.array(y_labels)
    groups = np.array(groups)
    
    valid = np.sum(np.abs(X_c3), axis=1) > 0
    X_v = X_c3[valid]
    y_v = y_labels[valid]
    g_v = groups[valid]
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, aucs = [], []
    for train_idx, val_idx in sgkf.split(X_v, y_v, g_v):
        X_train, y_train = X_v[train_idx], y_v[train_idx]
        X_val, y_val = X_v[val_idx], y_v[val_idx]
        
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            aucs.append(roc_auc_score(y_val, probs))
        f1s.append(f1_score(y_val, preds))
        
    f1_mean = np.mean(f1s)
    auc_mean = np.mean(aucs) if aucs else 0.0
    
    # Update JSON
    try:
        with open("d:/rtmp_train_results.json", 'r', encoding='utf-8') as f:
            res = json.load(f)
    except:
        res = {}
        
    res['bl3_v4'] = {'f1': f1_mean, 'auc': auc_mean}
    
    with open("d:/rtmp_train_results.json", "w", encoding='utf-8') as f:
        json.dump(res, f, indent=2)
