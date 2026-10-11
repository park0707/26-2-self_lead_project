# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_pruning_ablation.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)
Purpose: 
  LOFO 감사 및 상관계수 결과를 바탕으로 단계별 피처 다이어트(Feature Pruning)를 수행하고,
  Tiny TCN과의 Meta-Fusion 및 5대 시점별 F1 변화를 전수 실측 검증
"""

import os
import sys
import json
import glob
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
from cleanlab.filter import find_label_issues

sys.path.insert(0, 'D:/')
from src.models.second_models.babel_deadlift.d2_model_v5_final import (
    extract_d2_v5_final_features, FEATURE_NAMES_25_D2, VIEW_ALIAS_MAP
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_PATH = r"D:\src\models\second_models\babel_deadlift\weights\d2_v5_final_models.pkl"
OUT_DIR = r"D:\src\models\second_models\babel_deadlift\weights"

def run_ablation():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 피처 다이어트 & 최적화 실측 실험 (Feature Pruning Ablation)")
    print("  - 목적: 중복 고상관 피처(|r|>=0.85) 및 LOFO 저해 피처 제거를 통한 3-State F1 극대화")
    print("  - 데이터셋: AI-Hub 385개 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 검증 프로토콜: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리, 데이터 누수 0%)")
    print("=" * 115)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_list, y_list, subs, views = [], [], [], []

    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        v_code = base.split("_")[-1]
        views.append(v_code)
        subj = "Subj_Unknown"
        for part in base.split("_"):
            if "피험자" in part:
                subj = part
                break
        subs.append(subj)

        folder = "_".join(base.split("_")[:-1])
        json_path = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        with open(json_path, 'r', encoding='utf-8') as fp:
            conds = {c['condition']: c['value'] for c in json.load(fp).get('type_info', {}).get('conditions', [])}
        y_list.append(0 if conds.get('발과 무릎의 방향 일치', True) else 1)

        kps = np.load(p)['kps_33_3d_64f']
        f25, _, _ = extract_d2_v5_final_features(kps)
        X_list.append(f25)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs)
    views = np.array(views)

    # Load Tiny TCN OOF from v5.0
    with open(r'D:\src\models\second_models\babel_deadlift\weights\d2_v5_unified_models.pkl', 'rb') as fp:
        tcn_data = pickle.load(fp)
    oof_tcn = tcn_data['oof_tcn_probs']

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)

    # 4단계 피처 서브셋 정의
    # Subset 1: 베이스라인 25D 전수
    # Subset 2: 초고상관 중복 5개 제거 (20D)
    #   제거: can_kawr_p50 (r=1.0 with w_p50), can_kawr_p25, world_kawr_p25, can_kawr_p10, world_valgus_abs_max
    # Subset 3: LOFO 상위 저해 노이즈 6개 추가 제거 (14D)
    #   제거: ankle_internal_rotation_mismatch, arm_knee_min_distance, phase_delta_ka_pull_drop,
    #        persistent_fraction, bilateral_asymmetry_max, valgus_accel_snap_peak
    # Subset 4: 초정예 핵심 9D 피처 (Elite 9D)
    #   선정: world_kawr_min, world_kawr_p10, world_kawr_p50, can_kawr_min, can_kawr_mean,
    #        can_valgus_abs_max, phase1_first_pull_ka_min, foot_dev_pull_p90, dev_foot_left_p90

    subsets = {
        "Config 1 (초기 25D 전수)": list(range(25)),
        "Config 2 (고상관 중복 제거 20D)": [
            i for i in range(25) if FEATURE_NAMES_25_D2[i] not in [
                'can_kawr_p50', 'can_kawr_p25', 'world_kawr_p25', 'can_kawr_p10', 'world_valgus_abs_max'
            ]
        ],
        "Config 3 (노이즈/저해 피처 제거 14D)": [
            i for i in range(25) if FEATURE_NAMES_25_D2[i] not in [
                'can_kawr_p50', 'can_kawr_p25', 'world_kawr_p25', 'can_kawr_p10', 'world_valgus_abs_max',
                'ankle_internal_rotation_mismatch', 'arm_knee_min_distance', 'phase_delta_ka_pull_drop',
                'persistent_fraction', 'bilateral_asymmetry_max', 'valgus_accel_snap_peak'
            ]
        ],
        "Config 4 (초정예 10D 최적화)": [
            FEATURE_NAMES_25_D2.index(name) for name in [
                'world_kawr_min', 'world_kawr_p10', 'world_kawr_p50', 'can_kawr_min', 'can_kawr_mean',
                'can_valgus_abs_max', 'phase1_first_pull_ka_min', 'foot_dev_pull_p90',
                'dev_foot_left_p90', 'dev_foot_right_p90'
            ]
        ]
    }

    results = {}

    for cfg_name, feat_indices in subsets.items():
        print(f"\n[*] 평가 중: {cfg_name} (피처 수: {len(feat_indices)}개)")
        oof_gbdt = np.zeros(len(y), dtype=np.float32)
        X_sub = X[:, feat_indices]

        for tr_idx, val_idx in sgkf.split(X_sub, y, groups=subs):
            X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
            X_val, y_val = X_sub[val_idx], y[val_idx]

            # Cleanlab within training fold (top 30)
            init_clf = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.05, max_depth=3, scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1)
            init_clf.fit(X_tr, y_tr)
            probs_tr = init_clf.predict_proba(X_tr)
            issues = find_label_issues(labels=y_tr, pred_probs=probs_tr, return_indices_ranked_by='self_confidence', n_jobs=1)
            c_mask = np.ones(len(y_tr), dtype=bool)
            if len(issues) > 0:
                c_mask[issues[:min(30, len(issues))]] = False

            X_tr_clean, y_tr_clean = X_tr[c_mask], y_tr[c_mask]

            clf_lgb = lgb.LGBMClassifier(
                n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
                subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
                scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1
            )
            clf_lgb.fit(X_tr_clean, y_tr_clean)
            p_l = clf_lgb.predict_proba(X_val)[:, 1]

            clf_cat = CatBoostClassifier(
                iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1,
                train_dir=r"C:\Temp\catboost_info"
            )
            clf_cat.fit(X_tr_clean, y_tr_clean)
            p_c = clf_cat.predict_proba(X_val)[:, 1]
            oof_gbdt[val_idx] = 0.5 * p_l + 0.5 * p_c

        # Meta-Fusion with TCN (0.80 GBDT + 0.20 TCN)
        oof_final = 0.80 * oof_gbdt + 0.20 * oof_tcn
        auc = roc_auc_score(y, oof_final)

        # 3-State Search
        best = None
        for tn in np.arange(0.20, 0.42, 0.005):
            for td in np.arange(tn + 0.005, 0.50, 0.005):
                dec = (oof_final <= tn) | (oof_final >= td)
                cov = np.mean(dec) * 100
                if cov < 90.0: continue
                yy = y[dec]
                pred = (oof_final[dec] >= td).astype(int)
                f1 = f1_score(yy, pred, zero_division=0) * 100
                prec = precision_score(yy, pred, zero_division=0) * 100
                rec = recall_score(yy, pred, zero_division=0) * 100
                acc = accuracy_score(yy, pred) * 100
                cand = {'win': [round(float(tn),3), round(float(td),3)], 'f1': f1, 'prec': prec, 'rec': rec, 'acc': acc, 'cov': cov, 'dec': 100-cov}
                if best is None or cand['f1'] > best['f1']:
                    best = cand

        tn, td = best['win']
        dec_mask = (oof_final <= tn) | (oof_final >= td)
        cm = confusion_matrix(y[dec_mask], (oof_final[dec_mask] >= td).astype(int))

        # View breakdown
        v_res = {}
        for code, name in VIEW_ALIAS_MAP.items():
            v_idx = (views == code)
            v_dec = dec_mask[v_idx]
            cov_v = np.mean(v_dec) * 100
            y_v = y[v_idx][v_dec]
            pred_v = (oof_final[v_idx][v_dec] >= td).astype(int)
            v_res[code] = {
                'f1': f1_score(y_v, pred_v, zero_division=0) * 100,
                'prec': precision_score(y_v, pred_v, zero_division=0) * 100,
                'rec': recall_score(y_v, pred_v, zero_division=0) * 100,
                'cov': cov_v
            }

        results[cfg_name] = {
            'n_features': len(feat_indices),
            'features': [FEATURE_NAMES_25_D2[i] for i in feat_indices],
            'auc': auc,
            '3state': best,
            'cm': {'tp': int(cm[1, 1]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tn': int(cm[0, 0])},
            'views': v_res,
            'oof_final': oof_final.tolist()
        }

        print(f"  --> ROC-AUC: {auc:.4f} | 3-State F1: {best['f1']:.2f}% | Prec: {best['prec']:.2f}% | Rec: {best['rec']:.2f}% | Acc: {best['acc']:.2f}% | FP: {cm[0, 1]}건 | Win: {best['win']}")

    # Print Comparison Table
    print("\n" + "=" * 115)
    print("  [피처 다이어트 단계별 종합 성능 대조표 (Ablation Summary)]")
    print("=" * 115)
    print(f"{'구성 명칭':<28} | {'피처수':<5} | {'ROC-AUC':<8} | {'3-State F1':<10} | {'정밀도':<8} | {'재현율':<8} | {'정확도':<8} | {'FP수':<5}")
    print("-" * 115)
    for name, r in results.items():
        s = r['3state']
        cm = r['cm']
        print(f"{name:<28} | {r['n_features']:4d}개 | {r['auc']:8.4f} | {s['f1']:8.2f}% | {s['prec']:6.2f}% | {s['rec']:6.2f}% | {s['acc']:6.2f}% | {cm['fp']:3d}건")
    print("=" * 115)

    out_file = os.path.join(OUT_DIR, "d2_pruning_ablation_results.json")
    with open(out_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] 피처 다이어트 실측 결과 JSON 저장 완료: {out_file}")

if __name__ == "__main__":
    run_ablation()
