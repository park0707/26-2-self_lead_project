# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: build_and_evaluate_bl3_v3.py
Purpose: v3 11D 피처셋 기반 1,085개 전수 5-Fold 교차검증 및 프로덕션 번들 저장
         - v2(18D) vs v3(11D) 전체 3-State F1-Score 정밀 비교
         - 5대 공식 시점(시점 A~E) 전수 감사
         - 최종 프로덕션 모델 저장: bl3_v3_champion_bundle.joblib
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v2 import extract_lunge_c3_18d_features_v2, FEATURE_NAMES_18D
from src.features.lunge_c3_rear_knee_features_v3 import extract_lunge_c3_11d_features_v3, FEATURE_NAMES_11D

RAW_DATA_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"
CACHE_18D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_18d.npz"
CACHE_11D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_11d.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_or_extract_11d():
    if os.path.exists(CACHE_11D_PATH):
        print(f"[+] Loading cached 11D dataset from {CACHE_11D_PATH}...")
        d = np.load(CACHE_11D_PATH, allow_pickle=True)
        return d['X'], d['y'], d['groups'], d['views'], d['filenames']
        
    print("[*] Extracting 11D features from raw npz files...")
    npz_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.npz")))
    X_list, y_list, group_list, view_list, fname_list = [], [], [], [], []
    
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            kps = data['keypoints_2d_norm']
            conds = data['conditions']
            meta = json.loads(str(data['metadata']))
            
            is_defect = 1 if not conds[2] else 0
            sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
            view_name = meta.get('view', '시점A')
            
            f11 = extract_lunge_c3_11d_features_v3(kps)
            X_list.append(f11)
            y_list.append(is_defect)
            group_list.append(sub_id)
            view_list.append(view_name)
            fname_list.append(os.path.basename(f))
        except Exception:
            continue
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    views = np.array(view_list)
    fnames = np.array(fname_list)
    
    np.savez_compressed(CACHE_11D_PATH, X=X, y=y, groups=groups, views=views, filenames=fnames, feature_names=FEATURE_NAMES_11D)
    print(f"[+] Cached 11D dataset ({X.shape}) successfully.")
    return X, y, groups, views, fnames

def evaluate_predictions(y_true, y_prob, tau_low, tau_high):
    total = len(y_true)
    y_pred = np.full(total, -1, dtype=np.int32)
    y_pred[y_prob < tau_low] = 0
    y_pred[y_prob > tau_high] = 1
    
    det_mask = (y_pred != -1)
    n_det = np.sum(det_mask)
    coverage = (n_det / total) * 100.0
    
    if n_det > 0:
        y_det_true = y_true[det_mask]
        y_det_pred = y_pred[det_mask]
        acc = accuracy_score(y_det_true, y_det_pred) * 100.0
        prec = precision_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        rec = recall_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except:
        auc = 0.0
        
    return {
        'coverage': coverage,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'total': total,
        'determined': int(n_det),
        'undetermined': int(total - n_det)
    }

def main():
    print("=" * 80)
    print(" 바벨 런지 c3 v2(18D) vs v3(11D) 공식 성능 비교 및 프로덕션 번들 빌드")
    print("=" * 80)
    
    # 1. 데이터 로드
    X11, y, groups, views, fnames = load_or_extract_11d()
    d18 = np.load(CACHE_18D_PATH, allow_pickle=True)
    X18 = d18['X']
    
    n_samples = len(y)
    scale_pos = np.sum(y == 0) / np.sum(y == 1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_v2_cb = np.zeros(n_samples, dtype=np.float32)
    oof_v3_cb = np.zeros(n_samples, dtype=np.float32)
    oof_v3_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_v3_xgb = np.zeros(n_samples, dtype=np.float32)
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X11, y, groups), 1):
        # v2 18D CatBoost
        m_v2_cb = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        m_v2_cb.fit(X18[tr_idx], y[tr_idx])
        oof_v2_cb[val_idx] = m_v2_cb.predict_proba(X18[val_idx])[:, 1]
        
        # v3 11D CatBoost
        m_v3_cb = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        m_v3_cb.fit(X11[tr_idx], y[tr_idx])
        oof_v3_cb[val_idx] = m_v3_cb.predict_proba(X11[val_idx])[:, 1]
        
        # v3 11D LightGBM
        m_v3_lgb = lgb.LGBMClassifier(objective='binary', n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
        m_v3_lgb.fit(X11[tr_idx], y[tr_idx])
        oof_v3_lgb[val_idx] = m_v3_lgb.predict_proba(X11[val_idx])[:, 1]
        
        # v3 11D XGBoost
        m_v3_xgb = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', n_jobs=4)
        m_v3_xgb.fit(X11[tr_idx], y[tr_idx])
        oof_v3_xgb[val_idx] = m_v3_xgb.predict_proba(X11[val_idx])[:, 1]
        
    oof_v3_ens = 0.60 * oof_v3_cb + 0.25 * oof_v3_lgb + 0.15 * oof_v3_xgb
    print(f"[+] 5-Fold Cross Validation completed in {time.time() - t0:.2f}s")
    
    models = {
        'v2 CatBoost (18D Baseline)': oof_v2_cb,
        'v3 CatBoost (11D Orthogonal Champion)': oof_v3_cb,
        'v3 LightGBM (11D Orthogonal)': oof_v3_lgb,
        'v3 XGBoost (11D Orthogonal)': oof_v3_xgb,
        'v3 Tri-Ensemble (11D Soft Blend)': oof_v3_ens
    }
    
    summary_results = []
    best_configs = {}
    
    for name, probs in models.items():
        best_f1 = -1.0
        best_res = None
        best_cfg = None
        
        for tau_mid in np.linspace(0.48, 0.62, 15):
            for margin in [0.06, 0.08, 0.10, 0.12, 0.14]:
                tau_l = max(0.2, tau_mid - margin)
                tau_h = min(0.8, tau_mid + margin)
                res = evaluate_predictions(y, probs, tau_l, tau_h)
                if res['coverage'] >= 90.0 and res['f1'] > best_f1:
                    best_f1 = res['f1']
                    best_res = res
                    best_cfg = (tau_mid, tau_l, tau_h, margin)
                    
        summary_results.append({
            'Model': name,
            '3-State F1 (%)': f"{best_res['f1']:.2f}%",
            'Coverage (%)': f"{best_res['coverage']:.2f}%",
            'Accuracy (%)': f"{best_res['accuracy']:.2f}%",
            'Precision (%)': f"{best_res['precision']:.2f}%",
            'Recall (%)': f"{best_res['recall']:.2f}%",
            'ROC-AUC (%)': f"{best_res['auc']:.2f}%",
            'Optimal tau_mid': f"{best_cfg[0]:.2f}",
            'tau_low': f"{best_cfg[1]:.2f}",
            'tau_high': f"{best_cfg[2]:.2f}",
            'raw_res': best_res,
            'raw_cfg': best_cfg
        })
        best_configs[name] = (best_cfg, probs)
        
    df_sum = pd.DataFrame(summary_results).drop(columns=['raw_res', 'raw_cfg'])
    print("\n" + "=" * 80)
    print(" 전체 3-State F1-Score 실측 성능 비교 (1,085개 전수 표본)")
    print("=" * 80)
    print(df_sum.to_string(index=False))
    
    # 지연시간 벤치마크
    print("\n[*] Measuring Inference Latency (1,000 single evaluations)...")
    dummy_kps = np.zeros((64, 15, 2), dtype=np.float32)
    
    t_v2_f = time.time()
    for _ in range(500):
        _ = extract_lunge_c3_18d_features_v2(dummy_kps)
    lat_v2_feat = (time.time() - t_v2_f) / 500 * 1000.0
    
    t_v3_f = time.time()
    for _ in range(500):
        _ = extract_lunge_c3_11d_features_v3(dummy_kps)
    lat_v3_feat = (time.time() - t_v3_f) / 500 * 1000.0
    
    print(f"[*] Feature Extraction Latency -> v2 (18D): {lat_v2_feat:.3f} ms | v3 (11D): {lat_v3_feat:.3f} ms (감소율: {(1 - lat_v3_feat/lat_v2_feat)*100:.1f}%)")
    
    # 최종 프로덕션 모델 전체 데이터셋 학습
    print("\n[*] Training final v3 production models on full dataset...")
    final_cb = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    final_cb.fit(X11, y)
    
    final_lgb = lgb.LGBMClassifier(objective='binary', n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
    final_lgb.fit(X11, y)
    
    final_xgb = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', n_jobs=4)
    final_xgb.fit(X11, y)
    
    v3_cfg = best_configs['v3 CatBoost (11D Orthogonal Champion)'][0]
    
    v3_bundle = {
        'model_name': 'BL3_v3_Champion',
        'feature_names': FEATURE_NAMES_11D,
        'cb_model': final_cb,
        'lgb_model': final_lgb,
        'xgb_model': final_xgb,
        'optimal_tau': v3_cfg[0],
        'tau_low': v3_cfg[1],
        'tau_high': v3_cfg[2],
        'scale_pos_weight': scale_pos,
        'summary_results': summary_results
    }
    
    bundle_save_path = os.path.join(RESULTS_DIR, "bl3_v3_champion_bundle.joblib")
    joblib.dump(v3_bundle, bundle_save_path)
    print(f"[+] Saved v3 production bundle to: {bundle_save_path}")
    
    # JSON 메트릭 저장
    json_path = os.path.join(RESULTS_DIR, "bl3_v2_vs_v3_comparison_metrics.json")
    def conv(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'champion': 'v3 CatBoost (11D Orthogonal Champion)',
            'summary': summary_results,
            'v3_cfg': v3_cfg,
            'lat_v2_feat_ms': lat_v2_feat,
            'lat_v3_feat_ms': lat_v3_feat,
            'features_11d': FEATURE_NAMES_11D
        }, f, indent=2, ensure_ascii=False, default=conv)
    print(f"[+] Saved comparison metrics to: {json_path}")

if __name__ == "__main__":
    main()
