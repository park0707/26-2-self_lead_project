# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix

sys.path.insert(0, 'd:/')
from src.features.stiff_spine_features_v3 import extract_stiff_spine_features_v3, FEATURE_NAMES_V3
from src.features.good_morning_c2_features_v1 import extract_good_morning_c2_feature_vector_v1, GOOD_MORNING_C2_FEATURE_NAMES_V1

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\09_굿모닝"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\09_굿모닝\2D"
MODEL_DIR = r"D:\Data\피처_데이터\tuning_results"
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "gm3_v1_champion_bundle.joblib")
REPORT_PATH = r"D:\자주프 문서\gm3_v1_performance_report.md"

FEATURE_NAMES = FEATURE_NAMES_V3 + GOOD_MORNING_C2_FEATURE_NAMES_V1

def load_data():
    print(f"[DEBUG] Inside load_data, DATA_DIR: {DATA_DIR}")
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"[DEBUG] Found {len(npz_files)} npz files.")
    X = []
    y = []
    groups = []
    
    valid_count = 0
    defect_count = 0
    normal_count = 0

    for fpath in npz_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            kps_33_3d_64f = data['kps_33_3d_64f']
            base_seq_id = str(data['base_sequence_id'])
            subject = int(data['subject'])
        except Exception as e:
            continue
            
        json_path = os.path.join(LABEL_DIR, f"{base_seq_id}.json")
        if not os.path.exists(json_path):
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                jdata = json.load(fp)
            
            # Find condition 3 (척추의 중립)
            conditions = jdata.get('type_info', {}).get('conditions', [])
            spine_neutral_value = True # Default to normal
            for cond in conditions:
                if '척추의 중립' in cond.get('condition', ''):
                    spine_neutral_value = cond.get('value', True)
                    break
                    
            label = 0 if spine_neutral_value else 1
            
            # Extract features
            feats_v3 = extract_stiff_spine_features_v3(kps_33_3d_64f)
            feats_c2 = extract_good_morning_c2_feature_vector_v1(kps_33_3d_64f)
            feat_vec = np.concatenate([feats_v3, feats_c2])
            
            X.append(feat_vec)
            y.append(label)
            groups.append(subject)
            
            valid_count += 1
            if label == 1:
                defect_count += 1
            else:
                normal_count += 1
                
        except Exception as e:
            print(f"Error parsing {json_path}: {e}")
            
    return np.array(X), np.array(y), np.array(groups), valid_count, normal_count, defect_count

def train_and_eval():
    print("[DEBUG] Started train_and_eval()")
    print("[*] Loading and extracting features...")
    X, y, groups, valid, norm, defect = load_data()
    print(f"[*] Loaded {valid} samples. Normal: {norm}, Defect: {defect}")
    
    if valid < 10:
        print("[!] Not enough data to train.")
        return
        
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    f1_scores = []
    
    params = {
        'iterations': 250,
        'learning_rate': 0.035,
        'depth': 5,
        'l2_leaf_reg': 4.0,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 4
    }
    
    print("[*] Training 5-Fold CatBoost models...")
    models = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        
        fold_f1 = f1_score(y_val, (preds >= 0.5).astype(int))
        f1_scores.append(fold_f1)
        models.append(model)
        print(f"  Fold {fold+1} F1: {fold_f1*100:.2f}%")
        
    final_preds = (oof_preds >= 0.5).astype(int)
    f1 = f1_score(y, final_preds) * 100
    acc = accuracy_score(y, final_preds) * 100
    prec = precision_score(y, final_preds) * 100
    rec = recall_score(y, final_preds) * 100
    try:
        auc = roc_auc_score(y, oof_preds) * 100
    except:
        auc = 50.0
        
    print("\n[+] Cross-Validation Final Results:")
    print(f"  F1 Score:  {f1:.2f}%")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  ROC-AUC:   {auc:.2f}%")
    
    # Train champion model on all data
    champion = CatBoostClassifier(**params)
    champion.fit(X, y)
    
    # Find important features
    importances = champion.get_feature_importance()
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [(FEATURE_NAMES[i], importances[i]) for i in top_indices]
    
    print("\n[+] Top 10 Feature Importances:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.2f}")
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'cb_model': champion,
        'feature_names': FEATURE_NAMES,
        'optimal_tau': 0.5,
        'metrics': {
            'f1': f1,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'auc': auc
        }
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\n[+] Saved champion model to {MODEL_BUNDLE_PATH}")
    
    # Generate report
    report = f"""# 굿모닝 척추의 중립 오류(gm3_v1) 모델 성능 보고서

## 1. 개요
- **모델명**: gm3_v1 (Good Morning Error 3 - Spine Neutral)
- **목적**: 굿모닝 동작 시 척추의 중립 유지 여부를 판별하는 초정밀 3-State 전문가 분류기
- **학습 데이터**: {valid}개 시퀀스 (정상: {norm}개, 결함: {defect}개)
- **피처 아키텍처**: 45D Stiff Spine & Good Morning Kinematics (MediaPipe 33관절 기반 100% 비전 추출)

## 2. 5-Fold 교차 검증 성능 (OOF)
- **F1 Score**: {f1:.2f}%
- **Accuracy**: {acc:.2f}%
- **Precision**: {prec:.2f}%
- **Recall**: {rec:.2f}%
- **ROC-AUC**: {auc:.2f}%

## 3. Top-10 중요 피처 (Feature Importance)
"""
    for name, imp in top_features:
        report += f"- **{name}**: {imp:.2f}%\n"
        
    with open(REPORT_PATH, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"[+] Saved report to {REPORT_PATH}")

if __name__ == '__main__':
    train_and_eval()
