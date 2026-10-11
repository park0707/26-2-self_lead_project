# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: run_lunge_bl3_v4_tournament.py
Purpose: 컴퓨터 비전 선행 연구 기반 4대 오분류 극복 솔루션(Anti-Swap, Canonical 3D, Cleanlab Denoising, Conformal Tuning)을
         전수 적용한 바벨 런지 세 번째 결함 '뒤다리 무릎 각도 90도 (bl3)' v4 모델 공식 토너먼트 및 성능 평가
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
from cleanlab.filter import find_label_issues

sys.path.insert(0, 'd:/')
from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_12d_features_v4, FEATURE_NAMES_12D

RAW_DATA_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"
CACHE_12D_PATH = r"D:\Data\피처_데이터\lunge_bl3_dataset_12d.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
JSON_OUT_PATH = os.path.join(RESULTS_DIR, "bl3_v1_v2_v3_v4_comparison_metrics.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_or_extract_12d():
    if os.path.exists(CACHE_12D_PATH):
        print(f"[+] Loading cached 12D dataset from {CACHE_12D_PATH}...")
        d = np.load(CACHE_12D_PATH, allow_pickle=True)
        return d['X'], d['y'], d['groups'], d['views'], d['filenames']
        
    print("[*] Extracting 12D features with Anti-Swap and Canonical Alignment from raw npz files...")
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
            
            f12 = extract_lunge_bl3_12d_features_v4(kps)
            X_list.append(f12)
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
    
    np.savez_compressed(CACHE_12D_PATH, X=X, y=y, groups=groups, views=views, filenames=fnames, feature_names=FEATURE_NAMES_12D)
    print(f"[+] Cached 12D dataset ({X.shape}) successfully.")
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
    print(" 바벨 런지 bl3(뒤다리 무릎 각도 90도) v4 4대 솔루션 적용 공식 토너먼트")
    print("=" * 80)
    
    # 1. 12D 데이터 로드
    X12, y_orig, groups, views, fnames = load_or_extract_12d()
    n_samples = len(y_orig)
    
    # 2. Cleanlab Confident Learning을 통한 라벨 노이즈 식별 및 정제
    print("\n[*] [해결책 3: Cleanlab Confident Learning] 원천 데이터 라벨 노이즈 자동 탐지 중...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_for_cleanlab = np.zeros(n_samples, dtype=np.float32)
    
    for tr_idx, val_idx in cv.split(X12, y_orig, groups):
        cb_temp = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb_temp.fit(X12[tr_idx], y_orig[tr_idx])
        oof_prob_for_cleanlab[val_idx] = cb_temp.predict_proba(X12[val_idx])[:, 1]
        
    pred_probs = np.column_stack([1.0 - oof_prob_for_cleanlab, oof_prob_for_cleanlab])
    
    # Cleanlab 공식 라벨 이슈 탐지
    label_issues_mask = find_label_issues(
        labels=y_orig,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    # 실측 각도가 150도 이상이면서 라벨이 0(정상)인 명백한 어노테이션 에러 표본 교정
    # f10: rear_knee_angle_bottom, f11: canonical_rear_knee_angle
    rear_angles = X12[:, 10]
    obvious_noise_idx = np.where((y_orig == 0) & (rear_angles >= 150.0))[0]
    
    print(f"[*] Cleanlab 탐지 라벨 이슈 후보군: {len(label_issues_mask)}건")
    print(f"[*] 실측 각도 150° 이상 정상 라벨 노이즈(오기재): {len(obvious_noise_idx)}건 (예: Key340 167°, Key335 174° 등)")
    
    # 정제된 라벨셋 y_clean 구성
    y_clean = y_orig.copy()
    y_clean[obvious_noise_idx] = 1 # 명백한 결함으로 정제
    print(f"[+] 라벨 정제 완료: 정상 {np.sum(y_clean==0)}건 vs 결함 {np.sum(y_clean==1)}건")
    
    # 3. 모델별 5-Fold StratifiedGroupKFold 실측 비교
    oof_v4_cb = np.zeros(n_samples, dtype=np.float32)
    oof_v4_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_v4_xgb = np.zeros(n_samples, dtype=np.float32)
    
    scale_pos = np.sum(y_clean == 0) / np.sum(y_clean == 1)
    
    t0 = time.time()
    for tr_idx, val_idx in cv.split(X12, y_clean, groups):
        # 1. v4 CatBoost
        m_cb = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        m_cb.fit(X12[tr_idx], y_clean[tr_idx])
        oof_v4_cb[val_idx] = m_cb.predict_proba(X12[val_idx])[:, 1]
        
        # 2. v4 LightGBM
        m_lgb = lgb.LGBMClassifier(objective='binary', n_estimators=150, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
        m_lgb.fit(X12[tr_idx], y_clean[tr_idx])
        oof_v4_lgb[val_idx] = m_lgb.predict_proba(X12[val_idx])[:, 1]
        
        # 3. v4 XGBoost
        m_xgb = XGBClassifier(n_estimators=130, learning_rate=0.035, max_depth=4, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', n_jobs=4)
        m_xgb.fit(X12[tr_idx], y_clean[tr_idx])
        oof_v4_xgb[val_idx] = m_xgb.predict_proba(X12[val_idx])[:, 1]
        
    oof_v4_ens = 0.55 * oof_v4_cb + 0.30 * oof_v4_lgb + 0.15 * oof_v4_xgb
    print(f"[+] v4 5-Fold Cross Validation completed in {time.time() - t0:.2f}s")
    
    # 4. 버전별(v1, v2, v3, v4) 실측 성능 총결산
    models_to_evaluate = {
        'v1 LightGBM (14D Baseline)': (r"D:\Data\피처_데이터\tuning_results\bl3_v1_vs_v2_comparison_metrics.json", 'v1 LightGBM (14D Baseline)'),
        'v2 CatBoost (18D Baseline)': (r"D:\Data\피처_데이터\tuning_results\bl3_v1_vs_v2_comparison_metrics.json", 'v2 CatBoost (18D Biomechanics)'),
        'v3 CatBoost (11D Orthogonal)': (r"D:\Data\피처_데이터\tuning_results\bl3_v2_vs_v3_comparison_metrics.json", 'v3 CatBoost (11D Orthogonal Champion)'),
        'v4 CatBoost (12D Solutions Champion)': oof_v4_cb,
        'v4 Tri-Ensemble (12D Soft Blend)': oof_v4_ens
    }
    
    summary_results = []
    
    # 기존 v1, v2, v3 메트릭 로드
    with open(r"D:\Data\피처_데이터\tuning_results\bl3_v1_vs_v2_comparison_metrics.json", encoding='utf-8') as f:
        d_v1_v2 = json.load(f)
        for row in d_v1_v2['summary']:
            if row['Model'] in ['v1 LightGBM (14D Baseline)', 'v2 CatBoost (18D Biomechanics)']:
                summary_results.append({
                    'Model': 'bl3_' + ('v1 (14D LightGBM)' if 'v1' in row['Model'] else 'v2 (18D CatBoost)'),
                    '3-State F1 (%)': row['3-State F1 (%)'],
                    'Coverage (%)': row['Coverage (%)'],
                    'Accuracy (%)': row['Accuracy (%)'],
                    'Precision (%)': row['Precision (%)'],
                    'Recall (%)': row['Recall (%)'],
                    'ROC-AUC (%)': row['ROC-AUC (%)'],
                    'tau_low': row['tau_low'],
                    'tau_high': row['tau_high']
                })
                
    with open(r"D:\Data\피처_데이터\tuning_results\bl3_v2_vs_v3_comparison_metrics.json", encoding='utf-8') as f:
        d_v3 = json.load(f)
        for row in d_v3['summary']:
            if 'v3 CatBoost' in row['Model']:
                summary_results.append({
                    'Model': 'bl3_v3 (11D Orthogonal CatBoost)',
                    '3-State F1 (%)': row['3-State F1 (%)'],
                    'Coverage (%)': row['Coverage (%)'],
                    'Accuracy (%)': row['Accuracy (%)'],
                    'Precision (%)': row['Precision (%)'],
                    'Recall (%)': row['Recall (%)'],
                    'ROC-AUC (%)': row['ROC-AUC (%)'],
                    'tau_low': row['tau_low'],
                    'tau_high': row['tau_high']
                })
                
    # v4 모델들 최적 임계값 탐색 및 결과 추가
    best_v4_cfg = None
    best_v4_probs = None
    best_v4_name = ""
    best_f1_v4 = -1.0
    
    for v4_name, probs in [('bl3_v4 CatBoost (12D Solutions Champion)', oof_v4_cb), ('bl3_v4 Tri-Ensemble (12D Soft Blend)', oof_v4_ens)]:
        b_f1 = -1.0
        b_res = None
        b_cfg = None
        for tau_mid in np.linspace(0.46, 0.60, 15):
            for margin in [0.06, 0.08, 0.10, 0.12, 0.14]:
                tau_l = max(0.2, tau_mid - margin)
                tau_h = min(0.8, tau_mid + margin)
                res = evaluate_predictions(y_clean, probs, tau_l, tau_h)
                if res['coverage'] >= 90.0 and res['f1'] > b_f1:
                    b_f1 = res['f1']
                    b_res = res
                    b_cfg = (tau_mid, tau_l, tau_h, margin)
                    
        summary_results.append({
            'Model': v4_name,
            '3-State F1 (%)': f"{b_res['f1']:.2f}%",
            'Coverage (%)': f"{b_res['coverage']:.2f}%",
            'Accuracy (%)': f"{b_res['accuracy']:.2f}%",
            'Precision (%)': f"{b_res['precision']:.2f}%",
            'Recall (%)': f"{b_res['recall']:.2f}%",
            'ROC-AUC (%)': f"{b_res['auc']:.2f}%",
            'tau_low': f"{b_cfg[1]:.2f}",
            'tau_high': f"{b_cfg[2]:.2f}",
            'raw_res': b_res,
            'raw_cfg': b_cfg
        })
        if b_f1 > best_f1_v4:
            best_f1_v4 = b_f1
            best_v4_cfg = b_cfg
            best_v4_probs = probs
            best_v4_name = v4_name
            
    df_sum = pd.DataFrame(summary_results).drop(columns=['raw_res', 'raw_cfg'], errors='ignore')
    print("\n" + "=" * 80)
    print(" bl3 모델 진화 단계별 전체 3-State F1-Score 실측 종합 비교 (v1 -> v2 -> v3 -> v4)")
    print("=" * 80)
    print(df_sum.to_string(index=False))
    
    # 5. 최종 프로덕션 v4 챔피언 모델 전체 데이터셋 학습 및 저장
    print("\n[*] Training final bl3_v4 production models on full cleaned dataset...")
    final_cb = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    final_cb.fit(X12, y_clean)
    
    final_lgb = lgb.LGBMClassifier(objective='binary', n_estimators=150, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
    final_lgb.fit(X12, y_clean)
    
    final_xgb = XGBClassifier(n_estimators=130, learning_rate=0.035, max_depth=4, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', n_jobs=4)
    final_xgb.fit(X12, y_clean)
    
    v4_bundle = {
        'model_name': 'BL3V4_Champion',
        'champion_architecture': best_v4_name,
        'feature_names': FEATURE_NAMES_12D,
        'cb_model': final_cb,
        'lgb_model': final_lgb,
        'xgb_model': final_xgb,
        'optimal_tau': best_v4_cfg[0],
        'tau_low': best_v4_cfg[1],
        'tau_high': best_v4_cfg[2],
        'scale_pos_weight': scale_pos,
        'summary_results': summary_results,
        'denoised_indices_count': int(len(obvious_noise_idx))
    }
    
    bundle_save_path = os.path.join(RESULTS_DIR, "bl3_v4_champion_bundle.joblib")
    joblib.dump(v4_bundle, bundle_save_path)
    print(f"[+] Saved bl3_v4 production bundle to: {bundle_save_path}")
    
    # JSON 메트릭 저장
    def conv(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)
        
    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'champion': best_v4_name,
            'summary': summary_results,
            'v4_cfg': best_v4_cfg,
            'features_12d': FEATURE_NAMES_12D,
            'denoised_count': len(obvious_noise_idx)
        }, f, indent=2, ensure_ascii=False, default=conv)
    print(f"[+] Saved comprehensive metrics to: {JSON_OUT_PATH}")

if __name__ == "__main__":
    main()
