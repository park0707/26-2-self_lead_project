# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: test_c1_v2_in_integrated.py
Purpose: c1 v2 (3D 척추 피처 결합) 도입에 따른 바벨 스쿼트 통합 정상/오류 진단 엔진 성능 변화 실측
- 비교 대상:
  1) c1 v1.0 Legacy (22D)
  2) c1 v2-A Full (30D)
  3) c1 v2-C Cost-Sensitive (30D, FP 가중치 1.7배)
- 측정 지표:
  * 클린 정상(125건) 오탐 건수 (Clean FP)
  * 정상 판별 특이도 (Specificity)
  * 오류 탐지 민감도 (Sensitivity/Recall)
  * 종합 F1-Score & ROC-AUC
  * 소프트 위험도 융합(Soft Risk-Fusion, threshold=0.79) 기준
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, Any

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v1 import BabelSquatC1ModelV1

C2_INDICES = [0, 1, 2, 3, 5, 8, 9, 10, 11, 14, 15, 18, 19]
C3_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 23]
C4_INDICES = [2, 3, 4, 14, 15, 16, 18, 20, 23]

def get_oof_probs(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sgkf, sample_weight: np.ndarray = None) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        clf = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42 + fold, verbose=False)
        clf.fit(X[train_idx], y[train_idx], sample_weight=sw_train)
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return oof

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] c1 v2 도입에 따른 바벨 스쿼트 통합 정상/오류 진단 실측 벤치마크")
    print("=" * 110)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)
    y_normal = ds['y_normal']
    subjects = ds['subjects']
    N_total = len(Y_defects)
    y_overall_true = (np.sum(Y_defects, axis=1) > 0).astype(int) # 0: 완전정상(125건), 1: 결함포함(1781건)
    clean_mask = (y_overall_true == 0)

    # 2. 피처 로드
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)

    # c1 피처들
    feats_c1_22d = BabelSquatC1ModelV1.extract_features(X_b_64, use_pruned=True)
    spine_3d = np.load('d:/src/models/second_models/babel_squat/squat_c1_spine_feats.npz', allow_pickle=True)['spine_feats'].astype(np.float32)
    feats_c1_30d = np.concatenate([feats_c1_22d, spine_3d], axis=1)

    # c2 피처 (13D)
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names_c2 = list(h_data['feature_names'])
    elite_idx_c2 = [i for i, n in enumerate(feat_names_c2) if n != 'c2_gaze_hysteresis_rmse']
    X_18d_c2 = X_19d[:, elite_idx_c2]
    g_data_c2 = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats_c2 = g_data_c2['gated_feats'].astype(np.float32)[:, :2]
    feats_c2_13d = np.concatenate([X_18d_c2, gated_feats_c2], axis=1)[:, C2_INDICES]

    # c3 피처 (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    foot_data_c3 = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    feats_c3_20d = np.concatenate([v1_data_c3['feats'].astype(np.float32), foot_data_c3['foot_feats'].astype(np.float32)], axis=1)[:, C3_INDICES]

    # c4 피처 (9D)
    v1_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    foot_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    feats_c4_9d = np.concatenate([v1_data_c4['feats'].astype(np.float32), foot_data_c4['foot_feats'].astype(np.float32)], axis=1)[:, C4_INDICES]

    # 3. c2, c3, c4 고정 OOF 계산
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    print("[*] c2, c3, c4 OOF 확률 계산 중...")
    oof_c2 = get_oof_probs(feats_c2_13d, Y_defects[:, 1], subjects, sgkf)
    oof_c3 = get_oof_probs(feats_c3_20d, Y_defects[:, 2], subjects, sgkf)
    oof_c4 = get_oof_probs(feats_c4_9d, Y_defects[:, 3], subjects, sgkf)

    # 4. c1 후보군별 OOF 계산
    c1_candidates = {
        'c1_v1.0_Legacy_22D': {
            'X': feats_c1_22d,
            'sw': None,
            'desc': '레거시 22D'
        },
        'c1_v2-A_Full_30D': {
            'X': feats_c1_30d,
            'sw': None,
            'desc': '3D 척추 8D 결합 30D'
        },
        'c1_v2-C_CostSensitive_30D': {
            'X': feats_c1_30d,
            'sw': np.where(Y_defects[:, 0] == 0, 1.7, 1.0),
            'desc': '30D 결합 + 정상 FP 억제 가중치 1.7배'
        }
    }

    # Soft Risk-Fusion 파라미터
    risk_weights = {'c1': 1.15, 'c2': 0.90, 'c3': 1.10, 'c4': 1.00}
    threshold = 0.79

    results = {}
    print("\n" + "=" * 90)
    print("  [통합 엔진 Soft Risk-Fusion (threshold=0.79)] 실측 비교")
    print("=" * 90)

    for cname, cinfo in c1_candidates.items():
        t0 = time.time()
        print(f"[*] c1 모델 평가 중: {cname} ({cinfo['desc']}) ...")
        oof_c1 = get_oof_probs(cinfo['X'], Y_defects[:, 0], subjects, sgkf, sample_weight=cinfo['sw'])

        # c1 자체 성능
        c1_pred = (oof_c1 >= 0.50).astype(int)
        c1_f1 = f1_score(Y_defects[:, 0], c1_pred) * 100
        c1_auc = roc_auc_score(Y_defects[:, 0], oof_c1) * 100
        c1_clean_fp = int(np.sum((c1_pred == 1) & clean_mask))

        # Soft Risk-Fusion
        # P_safe = (1 - w1*p1)*(1 - w2*p2)*(1 - w3*p3)*(1 - w4*p4)
        p1_eff = np.clip(oof_c1 * risk_weights['c1'], 0.0, 1.0)
        p2_eff = np.clip(oof_c2 * risk_weights['c2'], 0.0, 1.0)
        p3_eff = np.clip(oof_c3 * risk_weights['c3'], 0.0, 1.0)
        p4_eff = np.clip(oof_c4 * risk_weights['c4'], 0.0, 1.0)

        p_safe = (1.0 - p1_eff) * (1.0 - p2_eff) * (1.0 - p3_eff) * (1.0 - p4_eff)
        integrated_defect_score = 1.0 - p_safe
        integrated_pred = (integrated_defect_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_overall_true, integrated_pred).ravel()
        spec = (tn / (tn + fp)) * 100.0
        sens = (tp / (tp + fn)) * 100.0
        prec = (tp / (tp + fp)) * 100.0
        f1 = f1_score(y_overall_true, integrated_pred) * 100.0
        auc = roc_auc_score(y_overall_true, integrated_defect_score) * 100.0

        clean_fps = int(np.sum((integrated_pred == 1) & clean_mask))
        clean_total = int(np.sum(clean_mask))
        clean_acc = ((clean_total - clean_fps) / clean_total) * 100.0

        results[cname] = {
            'c1_f1': c1_f1,
            'c1_auc': c1_auc,
            'c1_clean_fp': c1_clean_fp,
            'spec': spec,
            'sens': sens,
            'prec': prec,
            'f1': f1,
            'auc': auc,
            'clean_fps': clean_fps,
            'clean_total': clean_total,
            'clean_acc': clean_acc,
            'elapsed': time.time() - t0
        }

        print(f"    -> [c1 단독] F1: {c1_f1:.2f}%, AUC: {c1_auc:.2f}%, Clean FP: {c1_clean_fp}건")
        print(f"    -> [통합 엔진] 종합 F1: {f1:.2f}%, AUC: {auc:.2f}% | Specificity: {spec:.2f}%, Clean FP: {clean_fps}건 (정상인식률 {clean_acc:.1f}%) [{results[cname]['elapsed']:.1f}초]")

    print("\n" + "=" * 110)
    print("  ★ [c1 교체에 따른 바벨 스쿼트 통합 정상/오류 진단 성능 최종 비교]")
    print("=" * 110)
    print(f"{'c1 모델 버전':<28s} | {'c1 F1':<8s} | {'c1 Clean FP':<12s} | {'통합 F1':<8s} | {'통합 AUC':<9s} | {'통합 Specificity':<16s} | {'통합 Clean FP'}")
    print("-" * 110)
    for cname, r in results.items():
        print(f"{cname:<28s} | {r['c1_f1']:6.2f}% | {r['c1_clean_fp']:2d}건 ({r['c1_clean_fp']/125*100:4.1f}%) | {r['f1']:6.2f}% | {r['auc']:7.2f}% | {r['spec']:6.2f}% ({r['clean_acc']:4.1f}% 인식) | {r['clean_fps']}건 / 125건")
    print("=" * 110)

    # 결과 JSON 저장
    out_json = 'd:/src/models/second_models/babel_squat/c1_v2_integrated_impact.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] 결과 JSON 저장 완료: {out_json}")

if __name__ == '__main__':
    main()
