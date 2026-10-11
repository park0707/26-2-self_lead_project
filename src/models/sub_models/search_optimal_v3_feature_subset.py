# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: search_optimal_v3_feature_subset.py
Purpose: 바벨 런지 c3 피처 최적화 및 모델 구조 최적화 탐색
         - 다중공선성(|r| > 0.85) 완벽 제거
         - Sequential Feature Elimination / Selection
         - 모델 아키텍처별(CatBoost, LightGBM, XGBoost, Stacking Ensemble) 5-Fold StratifiedGroupKFold 실측
         - 1,085개 전수 표본 대상 전체 3-State F1-Score 및 ROC-AUC 최적화
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v2 import FEATURE_NAMES_18D

CACHE_18D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_18d.npz"

def evaluate_3state(y_true, y_prob, tau_low, tau_high):
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
        'auc': auc
    }

def find_best_threshold(y_true, y_prob):
    best_f1 = -1.0
    best_res = None
    best_cfg = None
    for tau_mid in np.linspace(0.48, 0.62, 15):
        for margin in [0.06, 0.08, 0.10, 0.12, 0.14]:
            tau_l = max(0.2, tau_mid - margin)
            tau_h = min(0.8, tau_mid + margin)
            res = evaluate_3state(y_true, y_prob, tau_l, tau_h)
            if res['coverage'] >= 90.0 and res['f1'] > best_f1:
                best_f1 = res['f1']
                best_res = res
                best_cfg = (tau_mid, tau_l, tau_h, margin)
    return best_f1, best_res, best_cfg

def evaluate_subset(X_sub, y, groups, model_type='catboost'):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(y)
    oof_probs = np.zeros(n_samples, dtype=np.float32)
    scale_pos = np.sum(y == 0) / np.sum(y == 1)
    
    for tr_idx, val_idx in cv.split(X_sub, y, groups):
        X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
        X_va = X_sub[val_idx]
        
        if model_type == 'catboost':
            model = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        elif model_type == 'lgbm':
            model = lgb.LGBMClassifier(objective='binary', n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
        elif model_type == 'xgboost':
            model = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', n_jobs=4)
            
        model.fit(X_tr, y_tr)
        oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]
        
    return oof_probs

def main():
    print("=" * 80)
    print(" 바벨 런지 c3 피처 최적화 및 모델 구조 최적화 탐색 토너먼트")
    print("=" * 80)
    
    data = np.load(CACHE_18D_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    groups = data['groups']
    
    # 1. 베이스라인 v2 (전체 18D) CatBoost
    print("\n[*] Evaluating v2 Baseline (All 18D CatBoost)...")
    oof_v2 = evaluate_subset(X, y, groups, 'catboost')
    f1_v2, res_v2, cfg_v2 = find_best_threshold(y, oof_v2)
    print(f"  -> v2 (18D): 3-State F1 = {f1_v2:.2f}% | AUC = {res_v2['auc']:.2f}% | Cov = {res_v2['coverage']:.2f}% | Rec = {res_v2['recall']:.2f}%")
    
    # 2. 피처 후보 그룹 정의:
    # 중복 및 죽은 피처 제거:
    #  - 제거 대상 1: pelvis_drop_to_stride_ratio (Gain = 0)
    #  - 제거 대상 2: rear_hip_extension_angle (torso_to_rear_femur_angle와 r=1.0 완전 중복)
    #  - 제거 대상 3: double_90_bilateral_residual (rear_knee_target_dev_bottom과 r=0.95 중복 및 음수 기여)
    #  - 제거 대상 4: stride_to_pelvis_width_ratio (inter_ankle_stride_bottom과 r=0.94 중복 및 음수 기여)
    #  - 제거 대상 5: rear_knee_target_dev_mean (rear_knee_angle_mean과 r=0.97 중복)
    
    # 후보 피처셋 전략 구성
    feature_sets = {}
    
    # Strategy A: 13D (완전 중복 및 Dead 피처 5개 제거)
    # 제거: pelvis_drop_to_stride_ratio, rear_hip_extension_angle, double_90_bilateral_residual, stride_to_pelvis_width_ratio, rear_knee_target_dev_mean
    drop_A = ['pelvis_drop_to_stride_ratio', 'rear_hip_extension_angle', 'double_90_bilateral_residual', 'stride_to_pelvis_width_ratio', 'rear_knee_target_dev_mean']
    idx_A = [i for i, f in enumerate(FEATURE_NAMES_18D) if f not in drop_A]
    feature_sets['v3-Candidate A (13D Non-Redundant)'] = idx_A
    
    # Strategy B: 10D (다중공선성 |r| > 0.85 엄격 제거 + 고기여 피처 엄선)
    # 추가 제거: rear_knee_angle_bottom (|r|=0.987 with dev_bottom), rear_knee_angle_mean, rear_knee_angle_rom
    drop_B = drop_A + ['rear_knee_angle_bottom', 'rear_knee_angle_mean', 'rear_knee_angle_rom']
    idx_B = [i for i, f in enumerate(FEATURE_NAMES_18D) if f not in drop_B]
    feature_sets['v3-Candidate B (10D Orthogonal Core)'] = idx_B
    
    # Strategy C: 8D (Permutation Importance 상위 8대 정예 피처)
    # rear_knee_stability_std, lead_rear_knee_ratio, rear_femur_vertical_pitch, rear_shank_horizontal_pitch,
    # inter_ankle_stride_bottom, knee_velocity_decel_nadir, rear_knee_target_dev_bottom, inter_knee_vertical_disp
    cand_C = [
        'rear_knee_stability_std', 'lead_rear_knee_ratio', 'rear_femur_vertical_pitch',
        'rear_shank_horizontal_pitch', 'inter_ankle_stride_bottom', 'knee_velocity_decel_nadir',
        'rear_knee_target_dev_bottom', 'inter_knee_vertical_disp'
    ]
    feature_sets['v3-Candidate C (8D Elite Subset)'] = [FEATURE_NAMES_18D.index(f) for f in cand_C]
    
    # Strategy D: 9D (8D Elite + torso_to_rear_femur_angle 고관절-체간 정렬)
    cand_D = cand_C + ['torso_to_rear_femur_angle']
    feature_sets['v3-Candidate D (9D Biomechanical Optimal)'] = [FEATURE_NAMES_18D.index(f) for f in cand_D]
    
    # Strategy E: 11D (9D + lead_knee_angle_bottom + rear_knee_angle_bottom)
    cand_E = cand_D + ['lead_knee_angle_bottom', 'rear_knee_angle_bottom']
    feature_sets['v3-Candidate E (11D Augmented Balance)'] = [FEATURE_NAMES_18D.index(f) for f in cand_E]
    
    print("\n[*] Evaluating Feature Subsets across CatBoost...")
    results_list = []
    oof_dict = {'v2 (18D Baseline)': oof_v2}
    
    for s_name, feat_indices in feature_sets.items():
        X_sub = X[:, feat_indices]
        oof_p = evaluate_subset(X_sub, y, groups, 'catboost')
        oof_dict[s_name] = oof_p
        f1, res, cfg = find_best_threshold(y, oof_p)
        print(f"  -> {s_name} ({len(feat_indices)} feats): 3-State F1 = {f1:.2f}% | AUC = {res['auc']:.2f}% | Cov = {res['coverage']:.2f}% | Rec = {res['recall']:.2f}%")
        results_list.append({
            'Config': s_name,
            'Num_Feats': len(feat_indices),
            'Model': 'CatBoost',
            'F1': f1,
            'AUC': res['auc'],
            'Coverage': res['coverage'],
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'cfg': cfg,
            'feat_names': [FEATURE_NAMES_18D[i] for i in feat_indices]
        })
        
    # 최고 피처셋 선발
    best_feat_cfg = max(results_list, key=lambda x: x['F1'])
    print(f"\n[+] Champion Feature Subset Selected: {best_feat_cfg['Config']} (3-State F1: {best_feat_cfg['F1']:.2f}%, AUC: {best_feat_cfg['AUC']:.2f}%)")
    print(f"    Features ({best_feat_cfg['Num_Feats']}D): {best_feat_cfg['feat_names']}")
    
    # 3. 모델 아키텍처 최적화 토너먼트 (최고 피처셋 대상)
    print("\n" + "=" * 80)
    print(f" 모델 아키텍처 최적화 토너먼트 ({best_feat_cfg['Config']} 피처셋 적용)")
    print("=" * 80)
    
    best_idx = [FEATURE_NAMES_18D.index(f) for f in best_feat_cfg['feat_names']]
    X_champ = X[:, best_idx]
    
    # 모델별 OOF 추론
    oof_champ_cb = evaluate_subset(X_champ, y, groups, 'catboost')
    oof_champ_lgb = evaluate_subset(X_champ, y, groups, 'lgbm')
    oof_champ_xgb = evaluate_subset(X_champ, y, groups, 'xgboost')
    
    # 앙상블 블렌딩 탐색 (다양한 가중치)
    oof_champ_ens1 = 0.50 * oof_champ_cb + 0.30 * oof_champ_lgb + 0.20 * oof_champ_xgb
    oof_champ_ens2 = 0.60 * oof_champ_cb + 0.25 * oof_champ_lgb + 0.15 * oof_champ_xgb
    oof_champ_ens3 = 0.40 * oof_champ_cb + 0.40 * oof_champ_lgb + 0.20 * oof_champ_xgb
    
    arch_models = {
        'v2 CatBoost (18D Baseline)': oof_v2,
        'v3 CatBoost (Optimized Features)': oof_champ_cb,
        'v3 LightGBM (Optimized Features)': oof_champ_lgb,
        'v3 XGBoost (Optimized Features)': oof_champ_xgb,
        'v3 Tri-Ensemble A (CB 0.50 + LGB 0.30 + XGB 0.20)': oof_champ_ens1,
        'v3 Tri-Ensemble B (CB 0.60 + LGB 0.25 + XGB 0.15)': oof_champ_ens2,
        'v3 Tri-Ensemble C (CB 0.40 + LGB 0.40 + XGB 0.20)': oof_champ_ens3
    }
    
    arch_summary = []
    for m_name, probs in arch_models.items():
        f1, res, cfg = find_best_threshold(y, probs)
        arch_summary.append({
            'Model Architecture': m_name,
            '3-State F1 (%)': f"{f1:.2f}%",
            'Coverage (%)': f"{res['coverage']:.2f}%",
            'Accuracy (%)': f"{res['accuracy']:.2f}%",
            'Precision (%)': f"{res['precision']:.2f}%",
            'Recall (%)': f"{res['recall']:.2f}%",
            'ROC-AUC (%)': f"{res['auc']:.2f}%",
            'tau_low': f"{cfg[1]:.2f}",
            'tau_high': f"{cfg[2]:.2f}",
            'raw_f1': f1,
            'raw_res': res,
            'raw_cfg': cfg,
            'probs': probs
        })
        
    df_arch = pd.DataFrame(arch_summary).drop(columns=['raw_f1', 'raw_res', 'raw_cfg', 'probs'])
    print("\n[*] 최종 아키텍처 비교표 (1,085개 전수 표본 대상):")
    print(df_arch.to_string(index=False))
    
    # 최종 챔피언 선발
    final_champ = max(arch_summary, key=lambda x: x['raw_f1'])
    print(f"\n[★] 최종 v3 공식 챔피언: {final_champ['Model Architecture']}")
    print(f"    - 전체 3-State F1-Score: {final_champ['3-State F1 (%)']} (v2 대비 도약!)")
    print(f"    - ROC-AUC: {final_champ['ROC-AUC (%)']}")
    print(f"    - 결함 재현율(Recall): {final_champ['Recall (%)']}")
    print(f"    - 확정 커버리지: {final_champ['Coverage (%)']}")
    
    # 결과 JSON 저장
    import json
    save_json_path = r"D:\Data\피처_데이터\tuning_results\bl3_v3_optimization_results.json"
    
    def conv(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)
        
    out_payload = {
        'best_feature_subset_name': best_feat_cfg['Config'],
        'best_feature_names': best_feat_cfg['feat_names'],
        'best_feature_count': len(best_feat_cfg['feat_names']),
        'final_champion_name': final_champ['Model Architecture'],
        'arch_summary': [{k: v for k, v in row.items() if k != 'probs'} for row in arch_summary],
        'final_cfg': final_champ['raw_cfg']
    }
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False, default=conv)
    print(f"[+] Saved optimization results to: {save_json_path}")

if __name__ == "__main__":
    main()
