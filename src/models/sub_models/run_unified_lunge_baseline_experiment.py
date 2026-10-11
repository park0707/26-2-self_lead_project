# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
try:
    from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_18d_features_v4 as feature_extractor
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

DIRS = {
    '04': r"D:\Data\MediaPipe\전처리\04_바벨_런지",
    '05': r"D:\Data\MediaPipe\전처리\05_스텝_포워드_다이나믹_런지",
    '06': r"D:\Data\MediaPipe\전처리\06_스텝_백워드_다이나믹_런지"
}

def load_data():
    X_list, y_list, group_list, src_list = [], [], [], []
    for src_id, dpath in DIRS.items():
        npz_files = sorted(glob.glob(os.path.join(dpath, "*.npz")))
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                kps = data['keypoints_2d_norm']
                conds = data['conditions']
                try:
                    meta = json.loads(str(data['metadata'].item()))
                except:
                    meta = json.loads(str(data['metadata']))
                
                is_defect = 1 if not conds[2] else 0
                sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
                
                f_vec = feature_extractor(kps)
                X_list.append(f_vec)
                y_list.append(is_defect)
                group_list.append(sub_id)
                src_list.append(src_id)
            except Exception as e:
                # print(f"Error on {f}: {e}")
                continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(group_list, dtype=np.int32), np.array(src_list)

def main():
    print("[*] Extracting 18D features from all Lunge variants...")
    X, y, groups, sources = load_data()
    print(f"[+] Extraction complete. Total sequences: {len(X)}")
    print(f"    - 04 (Barbell Lunge): {np.sum(sources == '04')}")
    print(f"    - 05 (Forward Lunge): {np.sum(sources == '05')}")
    print(f"    - 06 (Backward Lunge): {np.sum(sources == '06')}")
    
    if len(X) == 0:
        return
        
    mask_04 = sources == '04'
    X_04 = X[mask_04]
    y_04 = y[mask_04]
    groups_04 = groups[mask_04]
    
    mask_dyn = (sources == '05') | (sources == '06')
    X_dyn = X[mask_dyn]
    y_dyn = y[mask_dyn]
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X_04, y_04, groups_04))
    tr_idx, te_idx = splits[0]
    
    X_04_train, y_04_train = X_04[tr_idx], y_04[tr_idx]
    X_04_test, y_04_test = X_04[te_idx], y_04[te_idx]
    
    print("\n[*] [Baseline] Training on Barbell Lunge (04) ONLY...")
    cb_baseline = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_baseline.fit(X_04_train, y_04_train)
    
    preds_base = cb_baseline.predict(X_04_test)
    base_f1 = f1_score(y_04_test, preds_base)
    base_acc = accuracy_score(y_04_test, preds_base)
    print(f"    -> Baseline F1: {base_f1:.4f} | Acc: {base_acc:.4f}")
    
    print("\n[*] [Unified] Training on Barbell Lunge Train (04) + All Dynamic Lunge (05, 06)...")
    X_uni_train = np.vstack([X_04_train, X_dyn])
    y_uni_train = np.hstack([y_04_train, y_dyn])
    
    cb_unified = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_unified.fit(X_uni_train, y_uni_train)
    
    preds_uni = cb_unified.predict(X_04_test)
    uni_f1 = f1_score(y_04_test, preds_uni)
    uni_acc = accuracy_score(y_04_test, preds_uni)
    print(f"    -> Unified F1: {uni_f1:.4f} | Acc: {uni_acc:.4f}")
    
    print("\n[*] Conclusion:")
    print(f"    - F1 Score change: {base_f1:.4f} -> {uni_f1:.4f} ({uni_f1 - base_f1:+.4f})")
    print(f"    - Accuracy change: {base_acc:.4f} -> {uni_acc:.4f} ({uni_acc - base_acc:+.4f})")
    
if __name__ == "__main__":
    main()
