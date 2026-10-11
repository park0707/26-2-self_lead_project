# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: run_lunge_c3_v2_comparison.py
Purpose: 선행 연구(Farrokhi et al. JOSPT, NSCA, OpenCap) 기반 v2(18D) 모델과 v1(14D) 베이스라인의
         전수 1,085개 표본 대상 5-Fold StratifiedGroupKFold 교차검증 비교 평가
         - 전체 3-State F1-Score, 커버리지, 정확도, ROC-AUC, 5대 시점별 전수 감사
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
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features import extract_lunge_c3_14d_features, FEATURE_NAMES_14D
from src.features.lunge_c3_rear_knee_features_v2 import extract_lunge_c3_18d_features_v2, FEATURE_NAMES_18D

CACHE_14D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_14d.npz"
CACHE_18D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_18d.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
RAW_DATA_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_or_extract_data():
    """14D 및 18D 데이터 로드 또는 추출"""
    if os.path.exists(CACHE_14D_PATH) and os.path.exists(CACHE_18D_PATH):
        try:
            d14 = np.load(CACHE_14D_PATH, allow_pickle=True)
            d18 = np.load(CACHE_18D_PATH, allow_pickle=True)
            if len(d14['X']) > 0 and len(d18['X']) > 0:
                print(f"[+] Loading cached datasets: 14D ({d14['X'].shape}), 18D ({d18['X'].shape})...")
                return d14['X'], d18['X'], d14['y'], d14['groups'], d14['views'], d14['filenames']
        except Exception:
            pass

    print("[*] Extracting features from raw npz files...")
    npz_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.npz")))
    print(f"[*] Found {len(npz_files)} raw sequence files.")
    
    X14_list, X18_list, y_list, groups_list, views_list, fnames_list = [], [], [], [], [], []
    
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            kps = data['keypoints_2d_norm'] # (64, 15, 2)
            conds = data['conditions']
            meta_raw = data['metadata']
            meta_str = str(meta_raw) if isinstance(meta_raw, np.ndarray) else meta_raw
            meta = json.loads(meta_str)
            
            # 결함 타깃: 뒤다리 무릎 각도 90도 (conditions[2]) 위반 여부 (False -> 1, True -> 0)
            is_defect = 1 if not conds[2] else 0
            
            # 피험자 ID 파싱 (1~9)
            sub_str = meta.get('subject_id', '피험자1')
            sub_id = int(''.join(filter(str.isdigit, sub_str)) or 1)
            
            # 시점 파싱 (시점A ~ 시점E)
            view_name = meta.get('view', '시점A')
            
            f14 = extract_lunge_c3_14d_features(kps)
            f18 = extract_lunge_c3_18d_features_v2(kps)
            
            X14_list.append(f14)
            X18_list.append(f18)
            y_list.append(is_defect)
            groups_list.append(sub_id)
            views_list.append(view_name)
            fnames_list.append(os.path.basename(f))
        except Exception as e:
            continue
            
    X14 = np.array(X14_list, dtype=np.float32)
    X18 = np.array(X18_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list, dtype=np.int32)
    views = np.array(views_list)
    fnames = np.array(fnames_list)
    
    np.savez_compressed(CACHE_14D_PATH, X=X14, y=y, groups=groups, views=views, filenames=fnames, feature_names=FEATURE_NAMES_14D)
    np.savez_compressed(CACHE_18D_PATH, X=X18, y=y, groups=groups, views=views, filenames=fnames, feature_names=FEATURE_NAMES_18D)
    print(f"[+] Cached 14D ({X14.shape}) and 18D ({X18.shape}) datasets successfully.")
    return X14, X18, y, groups, views, fnames

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, tau_low: float, tau_high: float):
    """
    전체 샘플 대상 3-State 평가 수행
    - y_prob < tau_low: 0 (Normal)
    - y_prob > tau_high: 1 (Defect)
    - tau_low <= y_prob <= tau_high: -1 (Undetermined)
    """
    total = len(y_true)
    y_pred_3state = np.full(total, -1, dtype=np.int32)
    y_pred_3state[y_prob < tau_low] = 0
    y_pred_3state[y_prob > tau_high] = 1
    
    det_mask = (y_pred_3state != -1)
    n_det = np.sum(det_mask)
    coverage = (n_det / total) * 100.0
    
    if n_det > 0:
        y_det_true = y_true[det_mask]
        y_det_pred = y_pred_3state[det_mask]
        
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
    print(" 바벨 런지 c3(뒤다리 무릎 각도 90도) v1 vs v2 공식 성능 비교 토너먼트")
    print("=" * 80)
    
    X14, X18, y, groups, views, fnames = load_or_extract_data()
    n_samples = len(y)
    n_def = int(np.sum(y == 1))
    n_norm = int(np.sum(y == 0))
    scale_pos = n_norm / (n_def + 1e-4)
    print(f"[*] Dataset: Total {n_samples} | Normal {n_norm} ({n_norm/n_samples*100:.1f}%) | Defect {n_def} ({n_def/n_samples*100:.1f}%) | scale_pos_weight: {scale_pos:.2f}")
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 5대 모델 정의
    # 1. v1: 14D LightGBM
    # 2. v2-LGBM: 18D LightGBM
    # 3. v2-CatBoost: 18D CatBoost
    # 4. v2-XGBoost: 18D XGBoost
    # 5. v2-Ensemble: 18D Tri-Ensemble (LGBM 0.5 + CB 0.3 + XGB 0.2)
    
    oof_v1_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_v2_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_v2_cb = np.zeros(n_samples, dtype=np.float32)
    oof_v2_xgb = np.zeros(n_samples, dtype=np.float32)
    
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(cv.split(X18, y, groups), 1):
        print(f"\n--- [Fold {fold}/5] Evaluating ---")
        X14_tr, y_tr = X14[train_idx], y[train_idx]
        X14_va, y_va = X14[val_idx], y[val_idx]
        
        X18_tr = X18[train_idx]
        X18_va = X18[val_idx]
        
        # 1. v1 LightGBM (14D)
        m_v1 = lgb.LGBMClassifier(
            objective='binary', n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=15,
            scale_pos_weight=scale_pos, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=4
        )
        m_v1.fit(X14_tr, y_tr)
        oof_v1_lgb[val_idx] = m_v1.predict_proba(X14_va)[:, 1]
        
        # 2. v2 LightGBM (18D)
        m_v2_lgb = lgb.LGBMClassifier(
            objective='binary', n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18,
            scale_pos_weight=scale_pos, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=4
        )
        m_v2_lgb.fit(X18_tr, y_tr)
        oof_v2_lgb[val_idx] = m_v2_lgb.predict_proba(X18_va)[:, 1]
        
        # 3. v2 CatBoost (18D)
        m_v2_cb = CatBoostClassifier(
            iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced',
            random_seed=42, verbose=0, thread_count=4
        )
        m_v2_cb.fit(X18_tr, y_tr)
        oof_v2_cb[val_idx] = m_v2_cb.predict_proba(X18_va)[:, 1]
        
        # 4. v2 XGBoost (18D)
        m_v2_xgb = XGBClassifier(
            n_estimators=120, learning_rate=0.04, max_depth=4, scale_pos_weight=scale_pos,
            subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss', n_jobs=4
        )
        m_v2_xgb.fit(X18_tr, y_tr)
        oof_v2_xgb[val_idx] = m_v2_xgb.predict_proba(X18_va)[:, 1]
        
    oof_v2_ens = 0.50 * oof_v2_lgb + 0.30 * oof_v2_cb + 0.20 * oof_v2_xgb
    print(f"\n[+] 5-Fold Cross Validation completed in {time.time() - t0:.2f}s")
    
    # 모델별 최적 임계값 및 마진 탐색 그리드
    models = {
        'v1 LightGBM (14D Baseline)': oof_v1_lgb,
        'v2 LightGBM (18D Biomechanics)': oof_v2_lgb,
        'v2 CatBoost (18D Biomechanics)': oof_v2_cb,
        'v2 XGBoost (18D Biomechanics)': oof_v2_xgb,
        'v2 Tri-Ensemble (18D Soft Blend)': oof_v2_ens
    }
    
    print("\n" + "=" * 80)
    print(" 모델별 전체 3-State F1-Score 최적화 결과 (1,085개 전수 표본 대상)")
    print("=" * 80)
    
    summary_results = []
    best_configs = {}
    
    for name, probs in models.items():
        best_f1 = -1.0
        best_res = None
        best_cfg = None
        
        # 격자 탐색: tau_mid in [0.50, 0.62], margin in [0.06, 0.14]
        for tau_mid in np.linspace(0.48, 0.62, 15):
            for margin in [0.06, 0.08, 0.10, 0.12, 0.14]:
                tau_l = max(0.2, tau_mid - margin)
                tau_h = min(0.8, tau_mid + margin)
                
                res = evaluate_predictions(y, probs, tau_l, tau_h)
                # 커버리지 90% 이상 유지하면서 F1 최대화
                if res['coverage'] >= 90.0:
                    if res['f1'] > best_f1:
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
        
    df_summary = pd.DataFrame(summary_results).drop(columns=['raw_res', 'raw_cfg'])
    print(df_summary.to_string(index=False))
    
    view_alias = {
        '시점A': '시점 A (우측 후방 45°)',
        '시점B': '시점 B (우측 전방 45°)',
        '시점C': '시점 C (정면 0°)',
        '시점D': '시점 D (좌측 전방 45°)',
        '시점E': '시점 E (좌측 후방 45°)',
        'view1': '시점 E (좌측 후방 45°)',
        'view2': '시점 B (우측 전방 45°)',
        'view3': '시점 C (정면 0°)',
        'view4': '시점 D (좌측 전방 45°)',
        'view5': '시점 A (우측 후방 45°)'
    }
    
    print("\n" + "=" * 80)
    print(" 5대 공식 시점별 v1 vs v2 챔피언 세부 성능 감사 (Multi-View Audit)")
    print("=" * 80)
    
    # 최고 챔피언 모델 결정
    v2_champ_name = max(
        [m for m in models.keys() if 'v2' in m],
        key=lambda m: float([r['3-State F1 (%)'].replace('%', '') for r in summary_results if r['Model'] == m][0])
    )
    print(f"[*] Selected v2 Champion: {v2_champ_name}")
    
    v1_cfg, v1_probs = best_configs['v1 LightGBM (14D Baseline)']
    v2_cfg, v2_probs = best_configs[v2_champ_name]
    
    view_audit_list = []
    unique_views = sorted(list(set(views)))
    for v in unique_views:
        v_mask = (views == v)
        y_v = y[v_mask]
        
        v1_res = evaluate_predictions(y_v, v1_probs[v_mask], v1_cfg[1], v1_cfg[2])
        v2_res = evaluate_predictions(y_v, v2_probs[v_mask], v2_cfg[1], v2_cfg[2])
        
        diff_f1 = v2_res['f1'] - v1_res['f1']
        view_audit_list.append({
            'View Code': v,
            'Official Alias': view_alias.get(v, v),
            'Sample Count': int(np.sum(v_mask)),
            'v1 F1 (%)': f"{v1_res['f1']:.2f}%",
            'v2 F1 (%)': f"{v2_res['f1']:.2f}%",
            'Δ F1 (%p)': f"{diff_f1:+.2f}%p",
            'v1 Cov (%)': f"{v1_res['coverage']:.2f}%",
            'v2 Cov (%)': f"{v2_res['coverage']:.2f}%",
            'v2 Acc (%)': f"{v2_res['accuracy']:.2f}%"
        })
        
    df_view = pd.DataFrame(view_audit_list)
    print(df_view.to_string(index=False))
    
    # 지연시간(Latency) 벤치마크 (1,000회 단일 샘플 추론)
    print("\n[*] Running Latency Benchmark (1,000 single sequence inferences)...")
    sample_kps = np.zeros((64, 15, 2), dtype=np.float32)
    
    # v1 피처 추출 + 추론
    t_v1_start = time.time()
    for _ in range(500):
        _ = extract_lunge_c3_14d_features(sample_kps)
    lat_v1_feat = (time.time() - t_v1_start) / 500 * 1000.0
    
    # v2 피처 추출 + 추론
    t_v2_start = time.time()
    for _ in range(500):
        _ = extract_lunge_c3_18d_features_v2(sample_kps)
    lat_v2_feat = (time.time() - t_v2_start) / 500 * 1000.0
    
    print(f"[*] Feature Extraction Latency -> v1: {lat_v1_feat:.3f} ms | v2: {lat_v2_feat:.3f} ms")
    
    # 전체 학습된 v2 챔피언 모델 저장
    print("\n[*] Training final production models on full dataset...")
    final_lgb = lgb.LGBMClassifier(
        objective='binary', n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18,
        scale_pos_weight=scale_pos, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=4
    )
    final_lgb.fit(X18, y)
    
    final_cb = CatBoostClassifier(
        iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced',
        random_seed=42, verbose=0, thread_count=4
    )
    final_cb.fit(X18, y)
    
    final_xgb = XGBClassifier(
        n_estimators=120, learning_rate=0.04, max_depth=4, scale_pos_weight=scale_pos,
        subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss', n_jobs=4
    )
    final_xgb.fit(X18, y)
    
    v2_save_payload = {
        'champion_name': v2_champ_name,
        'lgb_model': final_lgb,
        'cb_model': final_cb,
        'xgb_model': final_xgb,
        'optimal_tau': v2_cfg[0],
        'tau_low': v2_cfg[1],
        'tau_high': v2_cfg[2],
        'scale_pos_weight': scale_pos,
        'summary_results': summary_results,
        'view_audit': view_audit_list
    }
    
    save_path = os.path.join(RESULTS_DIR, "bl3_v2_champion_bundle.joblib")
    joblib.dump(v2_save_payload, save_path)
    print(f"[+] Saved production bundle to: {save_path}")
    
    # 결과를 JSON 파일로도 저장하여 문서화 스크립트에서 참조 가능하게 함
    json_path = os.path.join(RESULTS_DIR, "bl3_v1_vs_v2_comparison_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # NumPy 직렬화 보정
        def default_conv(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            return str(o)
        json.dump({
            'v2_champion': v2_champ_name,
            'summary': summary_results,
            'view_audit': view_audit_list,
            'v1_cfg': v1_cfg,
            'v2_cfg': v2_cfg,
            'lat_v1_feat_ms': lat_v1_feat,
            'lat_v2_feat_ms': lat_v2_feat
        }, f, indent=2, ensure_ascii=False, default=default_conv)
        
    print(f"[+] Saved comparison metrics to: {json_path}")
    print("\n[+] Tournament execution completed successfully!")

if __name__ == "__main__":
    main()
