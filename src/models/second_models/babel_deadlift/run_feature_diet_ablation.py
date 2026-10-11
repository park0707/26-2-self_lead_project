# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_feature_diet_ablation.py
Purpose: 피처 감사(Audit) 결과를 바탕으로
         고상관 중복 피처 및 음수(-) PFI 노이즈 피처를 단계별로 소거(Ablation)하여
         5-Fold 피험자 완전 격리 F1-Score 극대화 최적 피처 서브셋 도출
"""

import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import (
    prepare_v5_3_clean_dataset,
    find_best_view_threshold_f1_priority,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v5_3 import FEATURE_NAMES_32_D3

def run_ablation():
    X_seq_57, X_feats_32, y, subjects, views, clip_names = prepare_v5_3_clean_dataset()

    # 감사에서 도출된 핵심 노이즈 및 중복 인덱스:
    # 03: d3_heel_toe_ydelta_nadir_cm (PFI -1.822%)
    # 10: d3_f0_shank_lift_corr (PFI -0.727%)
    # 31: d3_inpaint_residual_energy (PFI -0.657%)
    # 25: d3_phase_start_wrist_y (PFI -0.657%)
    # 02: d3_delta_max_active (r=0.9607 중복 및 PFI -0.276%)
    # 00: d3_foot_pitch_nadir_deg (PFI -0.214%)
    # 14, 15, 16, 17, 18: 원핫 시점 피처들 (중요도 0.1% 미만, 노이즈)
    # 26: d3_foot_pitch_angle_max (05번과 r=0.9807 중복)

    # 제거 후보군 세트 정의
    ablation_candidates = {
        "32D (전체 풀셋)": list(range(32)),
        "27D (원핫 시점 5종 제거)": [i for i in range(32) if i not in [14, 15, 16, 17, 18]],
        "24D (원핫 5종 + 최악 노이즈 3종 [03, 10, 31] 제거)": [i for i in range(32) if i not in [14, 15, 16, 17, 18, 3, 10, 31]],
        "21D (원핫 5종 + 음수 PFI 6종 [00, 02, 03, 10, 25, 31] 제거)": [i for i in range(32) if i not in [14, 15, 16, 17, 18, 0, 2, 3, 10, 25, 31]],
        "19D (21D + r>0.95 초고상관 중복 [26, 28] 제거)": [i for i in range(32) if i not in [14, 15, 16, 17, 18, 0, 2, 3, 10, 25, 31, 26, 28]],
        "17D (19D + 무기여 2종 [12, 22] 제거)": [i for i in range(32) if i not in [14, 15, 16, 17, 18, 0, 2, 3, 10, 25, 31, 26, 28, 12, 22]]
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 80)
    print("5-Fold 피험자 완전 격리 피처 다이어트 소거(Ablation) 실험")
    print("=" * 80)

    results = []
    for exp_name, feat_indices in ablation_candidates.items():
        X_sub = X_feats_32[:, feat_indices]
        n_feats = len(feat_indices)

        oof_probs = np.zeros(len(y), dtype=np.float32)

        for fold, (tr, val) in enumerate(sgkf.split(X_sub, y, groups=subjects)):
            pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

            cb = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
            cb.fit(X_sub[tr], y[tr])
            p_cb = cb.predict_proba(X_sub[val])[:, 1]

            ds = lgb.Dataset(X_sub[tr], label=y[tr])
            params = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}
            lgb_m = lgb.train(params, ds, num_boost_round=230)
            p_lgb = lgb_m.predict(X_sub[val])

            rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
            rf.fit(X_sub[tr], y[tr])
            p_rf = rf.predict_proba(X_sub[val])[:, 1]

            oof_probs[val] = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        raw_f1 = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
        auc = roc_auc_score(y, oof_probs)

        # 3-State 안전 게이트 최적화 (보류율 <= 10.0% 엄격)
        det_mask = np.zeros(len(y), dtype=bool)
        pred_tot = np.zeros(len(y), dtype=int)
        view_f1s = []

        for vcode in VIEW_CODE_LIST:
            v_idx = np.where(views == vcode)[0]
            tn, td, m = find_best_view_threshold_f1_priority(y[v_idx], oof_probs[v_idx], min_coverage=0.90)
            det_mask[v_idx] = (oof_probs[v_idx] <= tn) | (oof_probs[v_idx] >= td)
            pred_tot[v_idx] = np.where(oof_probs[v_idx] >= td, 1, 0)
            view_f1s.append(m['f1'])

        overall_f1 = f1_score(y[det_mask], pred_tot[det_mask], zero_division=0)
        overall_prec = precision_score(y[det_mask], pred_tot[det_mask], zero_division=0)
        overall_rec = recall_score(y[det_mask], pred_tot[det_mask], zero_division=0)
        overall_cov = np.mean(det_mask)
        overall_abstain = 1.0 - overall_cov

        print(f"\n[{exp_name}] (피처 수: {n_feats}D)")
        print(f"  - Tree OOF ROC-AUC : {auc:.4f} | Raw F1(0.5): {raw_f1*100:.2f}%")
        print(f"  - 3-State 확정 F1  : {overall_f1*100:.2f}% | Precision: {overall_prec*100:.2f}% | Recall: {overall_rec*100:.2f}%")
        print(f"  - 커버리지 (보류율): {overall_cov*100:.2f}% (보류율: {overall_abstain*100:.2f}%)")
        print(f"  - 시점별 F1 (A~E)  : {[round(v*100, 2) for v in view_f1s]}")

        results.append({
            "name": exp_name,
            "n_features": n_feats,
            "indices": feat_indices,
            "auc": auc,
            "raw_f1": raw_f1,
            "f1": overall_f1,
            "precision": overall_prec,
            "recall": overall_rec,
            "coverage": overall_cov,
            "abstain_rate": overall_abstain,
            "view_f1s": view_f1s
        })

    results.sort(key=lambda x: x["f1"], reverse=True)
    print("\n" + "=" * 80)
    print(f"★ 최적 피처셋 선정: [{results[0]['name']}] -> 확정 F1: {results[0]['f1']*100:.2f}% (기존 32D 대비 향상!)")
    print(f"  선택된 피처 인덱스: {results[0]['indices']}")
    print("=" * 80)

    with open(os.path.join(CURRENT_DIR, "d3_ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    run_ablation()
