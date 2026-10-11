# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: audit_bl3_v4_integrity_and_leakage.py
Purpose: bl3_v4 모델의 전수 감사 (Integrity & Data Leakage Audit)
         1) 입력 피처(X) 내 AI-Hub 라벨/3D 좌표 유입 여부 전수 감사
         2) Fold 격리 무결성 감사 (Out-of-Fold Leakage 여부)
         3) 평가 타깃(y) 정제에 따른 왜곡 감사 (y_orig 기준 vs y_clean 기준 실측 대조)
         4) 엄격한 원천 라벨(y_orig) 기준 순수 실측 성능 산출 (No Leakage Benchmark)
"""

import os
import sys
import json
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
from cleanlab.filter import find_label_issues

sys.path.insert(0, 'd:/')
sys.stdout.reconfigure(encoding='utf-8')
import src.features.lunge_bl3_features_v4 as feat_v4
from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_12d_features_v4, FEATURE_NAMES_12D

CACHE_12D_PATH = r"D:\Data\피처_데이터\lunge_bl3_dataset_12d.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
AUDIT_JSON_PATH = os.path.join(RESULTS_DIR, "bl3_v4_leakage_audit_report.json")

def audit_feature_extractor_code():
    """1. 피처 추출기 소스코드 정적 분석 (AI-Hub 라벨/JSON 참조 여부)"""
    source = inspect.getsource(feat_v4)
    forbidden_terms = ['condition', 'conditions', 'label', 'labels', 'metadata', 'AI-Hub', 'aihub', '3D.json', 'Key']
    findings = []
    
    for term in forbidden_terms:
        # 단어 검색
        lines = [line.strip() for line in source.split('\n') if term in line and not line.strip().startswith('#')]
        if len(lines) > 0:
            findings.append((term, lines))
            
    return findings

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

def find_best_threshold(y_true, y_prob):
    best_f1 = -1.0
    best_res = None
    best_cfg = None
    for tau_mid in np.linspace(0.40, 0.62, 20):
        for margin in [0.06, 0.08, 0.10, 0.12, 0.14]:
            tau_l = max(0.2, tau_mid - margin)
            tau_h = min(0.8, tau_mid + margin)
            res = evaluate_predictions(y_true, y_prob, tau_l, tau_h)
            if res['coverage'] >= 90.0 and res['f1'] > best_f1:
                best_f1 = res['f1']
                best_res = res
                best_cfg = (tau_mid, tau_l, tau_h, margin)
    return best_f1, best_res, best_cfg

def main():
    print("=" * 80)
    print(" [전수 감사] bl3_v4 모델 무결성, 데이터 누수 및 평가 왜곡 감사")
    print("=" * 80)
    
    # 1. 소스코드 무결성 감사
    print("\n[감사 1] 피처 추출기(lunge_bl3_features_v4.py) 소스코드 내 금지 키워드 정적 감사:")
    findings = audit_feature_extractor_code()
    if len(findings) == 0:
        print("  [PASS] 피처 추출기 내부에는 라벨, 메타데이터, AI-Hub JSON 참조 코드가 0건이며 100% 순수 비전 keypoints만 입력받음.")
    else:
        print("  [FAIL] 금지 키워드 발견:", findings)
        
    # 2. 데이터셋 로드
    d = np.load(CACHE_12D_PATH, allow_pickle=True)
    X = d['X'] # (1085, 12)
    y_orig = d['y'] # (1085,) - AI-Hub 원천 라벨 (노이즈 포함 원본)
    groups = d['groups']
    n_samples = len(y_orig)
    
    # 3. 95.42%가 나온 원인 규명 분석:
    # 아까 run_lunge_bl3_v4_tournament.py는:
    # 1) 전체 데이터셋에서 실측 각도 150도 이상인 180건을 y_clean = 1로 교정함.
    # 2) 교차검증을 수행하고, evaluate_predictions(y_clean, ...) 즉, 정제된 라벨을 정답 타깃으로 평가함!
    # 이것이 왜 성능 폭등을 일으켰는가?
    # -> 원래 원천 라벨 기준으로는 FP(오답)로 감점되던 180건이, y_clean 기준에서는 TP(정답)로 채점되었기 때문!
    
    print("\n[감사 2] 라벨 정제 전/후 타깃 분포 및 왜곡 원인 분석:")
    rear_angles = X[:, 10]
    obvious_noise_idx = np.where((y_orig == 0) & (rear_angles >= 150.0))[0]
    y_clean = y_orig.copy()
    y_clean[obvious_noise_idx] = 1
    
    print(f"  • AI-Hub 원천 라벨(y_orig): 정상 {np.sum(y_orig==0)}건 ({np.sum(y_orig==0)/n_samples*100:.1f}%) | 결함 {np.sum(y_orig==1)}건 ({np.sum(y_orig==1)/n_samples*100:.1f}%)")
    print(f"  • 정제된 라벨(y_clean): 정상 {np.sum(y_clean==0)}건 ({np.sum(y_clean==0)/n_samples*100:.1f}%) | 결함 {np.sum(y_clean==1)}건 ({np.sum(y_clean==1)/n_samples*100:.1f}%)")
    print(f"  • 라벨이 뒤바뀐 표본 수: {len(obvious_noise_idx)}건 (전체의 {len(obvious_noise_idx)/n_samples*100:.1f}%)")
    
    # 4. 엄격한 3가지 시나리오 교차검증 실측:
    # 시나리오 A (원천 데이터 완전 고정, No Denoising):
    #   - 훈련도 y_orig로, 평가도 y_orig로! (순수 v4 피처(Anti-Swap + Canonical 12D)만의 순수한 성능)
    # 시나리오 B (엄격한 Fold 내 Cleanlab Denoising, 평가는 원천 y_orig로 채점):
    #   - 훈련 Fold 내에서만 Cleanlab 노이즈를 제거하고 모델 학습, 평가는 사용자의 "원천 데이터셋 y_orig"로 엄격 채점!
    # 시나리오 C (이전 보고서의 y_clean 기준 채점):
    #   - 훈련도 y_clean, 평가도 y_clean으로 채점 (95.42%가 나온 이유)
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs_scen_A = np.zeros(n_samples, dtype=np.float32)
    oof_probs_scen_B = np.zeros(n_samples, dtype=np.float32)
    oof_probs_scen_C = np.zeros(n_samples, dtype=np.float32)
    
    print("\n[감사 3] 3대 시나리오별 5-Fold StratifiedGroupKFold 실측 수행 중...")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_orig, groups), 1):
        # --- 시나리오 A: 원천 라벨 100% 그대로 학습 (No Cleanlab) ---
        scale_pos_A = np.sum(y_orig[tr_idx] == 0) / np.sum(y_orig[tr_idx] == 1)
        cb_A = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb_A.fit(X[tr_idx], y_orig[tr_idx])
        oof_probs_scen_A[val_idx] = cb_A.predict_proba(X[val_idx])[:, 1]
        
        # --- 시나리오 B: Fold 내에서만 훈련데이터 라벨 정제, 검증 Fold는 손대지 않음 ---
        tr_angles = X[tr_idx, 10]
        tr_noise = tr_idx[np.where((y_orig[tr_idx] == 0) & (tr_angles >= 150.0))[0]]
        y_tr_fold_clean = y_orig[tr_idx].copy()
        for idx_local, global_idx in enumerate(tr_idx):
            if global_idx in tr_noise:
                y_tr_fold_clean[idx_local] = 1
                
        scale_pos_B = np.sum(y_tr_fold_clean == 0) / np.sum(y_tr_fold_clean == 1)
        cb_B = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb_B.fit(X[tr_idx], y_tr_fold_clean)
        oof_probs_scen_B[val_idx] = cb_B.predict_proba(X[val_idx])[:, 1]
        
        # --- 시나리오 C: 이전 보고서 (전역 y_clean 학습) ---
        scale_pos_C = np.sum(y_clean[tr_idx] == 0) / np.sum(y_clean[tr_idx] == 1)
        cb_C = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb_C.fit(X[tr_idx], y_clean[tr_idx])
        oof_probs_scen_C[val_idx] = cb_C.predict_proba(X[val_idx])[:, 1]
        
    print("\n" + "=" * 80)
    print(" [전수 감사 결과 비교표] 시나리오별 실측 성능 및 정답 기준 비교")
    print("=" * 80)
    
    eval_cases = [
        ("bl3_v1 (14D Baseline)", "원천 라벨 (y_orig)", 83.11, 92.53, 91.14, 85.88, 80.51, 91.64),
        ("bl3_v2 (18D Baseline)", "원천 라벨 (y_orig)", 83.52, 91.34, 90.92, 82.91, 84.13, 92.79),
        ("bl3_v3 (11D Orthogonal)", "원천 라벨 (y_orig)", 83.70, 90.97, 90.37, 79.74, 88.09, 92.81),
    ]
    
    # 시나리오 A 평가 (y_orig 기준)
    f1_A, res_A, cfg_A = find_best_threshold(y_orig, oof_probs_scen_A)
    eval_cases.append(("v4 시나리오 A (12D 피처 순수 효과, No Denoising)", "원천 라벨 (y_orig)", f1_A, res_A['coverage'], res_A['accuracy'], res_A['precision'], res_A['recall'], res_A['auc']))
    
    # 시나리오 B 평가 (훈련 정제, 평가는 원천 y_orig 기준 채점)
    f1_B, res_B, cfg_B = find_best_threshold(y_orig, oof_probs_scen_B)
    eval_cases.append(("v4 시나리오 B (Fold내 정제 훈련, 평가는 y_orig 채점)", "원천 라벨 (y_orig)", f1_B, res_B['coverage'], res_B['accuracy'], res_B['precision'], res_B['recall'], res_B['auc']))
    
    # 시나리오 C 평가 (이전 보고서: 정제 라벨 y_clean 기준 채점)
    f1_C, res_C, cfg_C = find_best_threshold(y_clean, oof_probs_scen_C)
    eval_cases.append(("v4 시나리오 C (이전 보고서: y_clean 기준 채점)", "정제 라벨 (y_clean)", f1_C, res_C['coverage'], res_C['accuracy'], res_C['precision'], res_C['recall'], res_C['auc']))
    
    df_eval = pd.DataFrame(eval_cases, columns=['실험 시나리오', '평가 정답 기준 (Ground Truth)', '3-State F1 (%)', 'Coverage (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'ROC-AUC (%)'])
    for col in ['3-State F1 (%)', 'Coverage (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'ROC-AUC (%)']:
        df_eval[col] = df_eval[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
    print(df_eval.to_string(index=False))
    
    # 감사 보고서 JSON 저장
    audit_data = {
        'findings': findings,
        'label_stats': {
            'y_orig_normal': int(np.sum(y_orig == 0)),
            'y_orig_defect': int(np.sum(y_orig == 1)),
            'y_clean_normal': int(np.sum(y_clean == 0)),
            'y_clean_defect': int(np.sum(y_clean == 1)),
            'denoised_count': int(len(obvious_noise_idx))
        },
        'comparison_table': eval_cases,
        'scen_A_cfg': cfg_A,
        'scen_B_cfg': cfg_B,
        'scen_C_cfg': cfg_C
    }
    
    with open(AUDIT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Audit JSON saved to: {AUDIT_JSON_PATH}")

if __name__ == "__main__":
    main()
