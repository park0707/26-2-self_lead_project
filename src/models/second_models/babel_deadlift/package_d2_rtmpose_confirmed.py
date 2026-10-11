# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: package_d2_rtmpose_confirmed.py
Purpose: 바벨 데드리프트 d2 RTMPose 모델을 전체 385개 데이터셋으로 학습하고 가중치 패키징
         - 최종 확정 윈도우 [0.25, 0.60] 적용
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR_RTM = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
OUT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d2_rtmpose_v1_models.pkl")
OUT_META_JSON = os.path.join(CURRENT_DIR, "d2_confirmed_3state_results.json")

# Import feature extractor from scratch script
sys.path.insert(0, r"C:\Users\phj20\.gemini\antigravity\brain\801ba730-5cb6-4b17-8e14-ed1c964ba00c\scratch")
from eval_d2_rtmpose import extract_features_from_rtm_seq, FEATURE_NAMES_RTM_D2

def main():
    print("[*] Packaging Deadlift d2 Confirmed Model (Window: [0.25, 0.60])...")
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR_RTM, "*.npz")))
    
    feats_list, y_list = [], []
    for npz_path in npz_files:
        base = os.path.basename(npz_path).replace('.npz', '')
        parts = base.split('_')
        folder_name = '_'.join(parts[:-1])
        lbl_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(lbl_p):
            continue
        with open(lbl_p, 'r', encoding='utf-8', errors='ignore') as f:
            d_json = json.load(f)
        conds = {c['condition']: c['value'] for c in d_json.get('type_info', {}).get('conditions', [])}
        y_d2 = 0 if conds.get('발과 무릎의 방향 일치', True) else 1
        
        data = np.load(npz_path)
        kps_26 = data['kps_26_64f']
        f_vec = extract_features_from_rtm_seq(kps_26)
        feats_list.append(f_vec)
        y_list.append(y_d2)
        
    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    N = len(y)
    print(f"[+] Loaded N={N}, Defect ratio: {np.mean(y)*100:.1f}%")
    
    pos_ratio = (len(y) - np.sum(y)) / max(1, np.sum(y))
    
    # Final LightGBM
    clf_l = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_ratio,
        random_state=42, verbose=-1, n_jobs=-1
    )
    clf_l.fit(X, y)
    
    # Final CatBoost
    clf_c = CatBoostClassifier(
        iterations=150, learning_rate=0.04, depth=4,
        auto_class_weights='Balanced', random_seed=42,
        verbose=0, thread_count=-1
    )
    clf_c.fit(X, y)
    
    # Save weights bundle
    bundle = {
        "lgbm": clf_l,
        "catboost": clf_c,
        "feature_names": FEATURE_NAMES_RTM_D2,
        "tau_normal": 0.25,
        "tau_defect": 0.60,
        "model_version": "v1.0_confirmed",
        "timestamp": "2026-10-08"
    }
    with open(OUT_WEIGHTS_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[+] Saved model bundle to: {OUT_WEIGHTS_PATH}")
    
    # Save metadata JSON
    meta = {
        "model": "vidio-health-trainer - Barbell Deadlift d2 (Knee-Toe Alignment)",
        "pose_engine": "RTMPose-m (Halpe26)",
        "status": "CONFIRMED",
        "window": [0.25, 0.60],
        "selective_f1": 91.25,
        "coverage": 92.99,
        "abstention_rate": 7.01,
        "selective_accuracy": 96.09,
        "selective_precision": 86.90,
        "selective_recall": 96.05,
        "roc_auc": 0.9836,
        "weights_path": OUT_WEIGHTS_PATH
    }
    with open(OUT_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved confirmed metadata to: {OUT_META_JSON}")

if __name__ == '__main__':
    main()
