# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: experiment_trunk_compensation_noise.py
Purpose: 상체 보상 작용(Trunk Compensation) 피처가 Condition 2(몸통 발 무릎 방향 일치)
         식별에 실제 유효 신호인지 vs 노이즈인지 엄밀한 통계 및 머신러닝 교차검증 실험 수행
"""

import os
import sys
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = r"D:\src\models\second_models\babel_lunge"
CACHE_PATH = os.path.join(CURRENT_DIR, "lunge_c2_features_v4_cache.npz")

from lunge_c2_features_v4 import FEATURE_NAMES_20D

def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 상체 보상 작용 피처의 노이즈 여부 검증 실험")
    print("=" * 80)

    data = np.load(CACHE_PATH, allow_pickle=True)
    X_full = data["X"][:, :19]  # 19D
    y = data["y"]
    subjects = data["subjects"]
    views = data["views"]
    feat_names = FEATURE_NAMES_20D[:19]

    # 1. 상체 관련 피처 식별
    trunk_feats = [
        "trunk_canting_rom",                    # 12. 전체 동작 중 상체 측방 기울기 가동범위
        "trunk_counterbalance_score",           # 13. 선행 발 반대 방향으로 상체가 보상 기울어짐
        "decoupled_trunk_lateral_lean_bottom",  # 18. PACS 기반 척추-골반 상대 순수 관상면 롤
        "trunk_pelvis_yaw_torsion",             # 11. 흉곽 평면 법선 vs 골반 축 법선 수평면 비틀림
        "trunk_foot_angle_deviation"            # 16. 체간 전방 벡터 vs 선행 발 장축 입체 편차각
    ]
    
    trunk_indices = [feat_names.index(fn) for fn in trunk_feats]
    lower_body_indices = [i for i in range(19) if i not in trunk_indices]
    
    print("\n[1] 상체 보상 피처 vs 하지 정렬 피처 통계적 상관관계 및 유의성 분석")
    print(f"{'Feature Name':<38} | {'Corr (r)':<10} | {'p-value':<12} | {'Normal Mean':<12} | {'Defect Mean':<12} | {'Classification'}")
    print("-" * 105)

    for i, fn in enumerate(feat_names):
        r, p = stats.pointbiserialr(y, X_full[:, i])
        m_norm = np.mean(X_full[y == 0, i])
        m_def = np.mean(X_full[y == 1, i])
        cat = "[TRUNK COMP]" if fn in trunk_feats else "[LOWER ALIGN]"
        print(f"{fn:<38} | {r:>10.4f} | {p:>12.4e} | {m_norm:>12.4f} | {m_def:>12.4f} | {cat}")

    # 2. 피처 서브셋 정의 및 비교 실험
    subsets = {
        "1. Full 19D (상체 보상 포함)": list(range(19)),
        "2. No Pure Trunk Lean (canting + counterbalance 제거, 17D)": [
            i for i, fn in enumerate(feat_names) if fn not in ["trunk_canting_rom", "trunk_counterbalance_score"]
        ],
        "3. No Decoupled Trunk Lean (decoupled_lean 제거, 18D)": [
            i for i, fn in enumerate(feat_names) if fn != "decoupled_trunk_lateral_lean_bottom"
        ],
        "4. No Trunk Lateral at all (상체 횡방향 3종 전면 제거, 16D)": [
            i for i, fn in enumerate(feat_names) if fn not in ["trunk_canting_rom", "trunk_counterbalance_score", "decoupled_trunk_lateral_lean_bottom"]
        ],
        "5. Pure Lower-Body (상체 5종 전면 제거, 14D)": lower_body_indices
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cost_weight = (len(y) - np.sum(y)) / np.sum(y) * 1.35

    print("\n[2] 5-Fold 피험자 격리 StratifiedGroupKFold 교차검증 비교 (LGBM + CatBoost 앙상블)")
    print(f"{'Subset Name':<55} | {'AUC':<8} | {'Binary F1':<10} | {'Prec':<8} | {'Recall':<8}")
    print("-" * 95)

    results_dict = {}

    for name, indices in subsets.items():
        X_sub = X_full[:, indices]
        oof_ens = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects), 1):
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_val = X_sub[val_idx]

            lgb_clf = lgb.LGBMClassifier(
                n_estimators=140, max_depth=4, num_leaves=14, learning_rate=0.038,
                min_child_samples=18, subsample=0.8, colsample_bytree=0.75,
                reg_alpha=0.15, reg_lambda=1.2, scale_pos_weight=cost_weight,
                random_state=42 + fold, n_jobs=-1, verbose=-1
            )
            lgb_clf.fit(X_tr, y_tr)
            p_lgb = lgb_clf.predict_proba(X_val)[:, 1]

            cat_clf = CatBoostClassifier(
                iterations=160, depth=4, learning_rate=0.038, l2_leaf_reg=3.5,
                auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=-1
            )
            cat_clf.fit(X_tr, y_tr)
            p_cat = cat_clf.predict_proba(X_val)[:, 1]

            oof_ens[val_idx] = 0.50 * p_lgb + 0.50 * p_cat

        auc = roc_auc_score(y, oof_ens)
        pred_bin = (oof_ens >= 0.5).astype(int)
        f1 = f1_score(y, pred_bin, zero_division=0)
        prec = precision_score(y, pred_bin, zero_division=0)
        rec = recall_score(y, pred_bin, zero_division=0)

        results_dict[name] = {
            "oof": oof_ens,
            "auc": auc,
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "indices": indices
        }

        print(f"{name:<55} | {auc*100:>6.2f}% | {f1*100:>8.2f}% | {prec*100:>6.2f}% | {rec*100:>6.2f}%")

    # 3. 3-State F1 (보류율 <= 10.0%) 최적화 비교
    print("\n[3] 3-State F1-Score 최적화 비교 (판정 보류율 <= 10.0% 엄격 제약)")
    print(f"{'Subset Name':<55} | {'Coverage':<9} | {'Retake':<8} | {'Defect F1':<10} | {'Clean F1':<9} | {'Macro F1':<9}")
    print("-" * 115)

    prob_vals_n = np.linspace(0.45, 0.54, 91)
    prob_vals_d = np.linspace(0.50, 0.60, 101)

    for name, data_res in results_dict.items():
        oof_probs = data_res["oof"]
        best_cand = None
        best_score = -1

        for tn in prob_vals_n:
            for td in prob_vals_d:
                if td <= tn: continue
                dec = (oof_probs <= tn) | (oof_probs >= td)
                cov = float(np.mean(dec))
                rej = 1.0 - cov
                if rej <= 0.1001:
                    y_dec = y[dec]
                    p_dec = (oof_probs[dec] >= td).astype(int)
                    d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                    c_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
                    acc = accuracy_score(y_dec, p_dec)
                    if c_f1 >= 0.73 and acc >= 0.63:
                        score = 0.60 * d_f1 + 0.40 * c_f1
                        if score > best_score:
                            best_score = score
                            best_cand = (d_f1, c_f1, acc, cov, rej, tn, td)

        if best_cand:
            d_f1, c_f1, acc, cov, rej, tn, td = best_cand
            macro = (d_f1 + c_f1) / 2
            print(f"{name:<55} | {cov*100:>7.2f}% | {rej*100:>6.2f}% | {d_f1*100:>8.2f}% | {c_f1*100:>7.2f}% | {macro*100:>7.2f}%")
        else:
            print(f"{name:<55} | No combination found meeting constraints")

    # 4. 시점별 AUC 변화 감사 (상체 왜곡이 심한 45도 사각 시점 확인)
    print("\n[4] 카메라 시점별 ROC-AUC 감사 (45도 사각 시점 vs 정면)")
    print(f"{'시점':<15} | {'1. Full 19D':<14} | {'2. No Canting/Counter':<22} | {'4. No Lateral 16D':<18} | {'5. Pure Lower 14D':<18}")
    print("-" * 95)

    view_codes = ["view5", "view2", "view3", "view4", "view1"]
    view_names = ["시점 A (우측 후방 45°)", "시점 B (우측 전방 45°)", "시점 C (정면 0°)", "시점 D (좌측 전방 45°)", "시점 E (좌측 후방 45°)"]

    for vc, vn in zip(view_codes, view_names):
        v_mask = (views == vc)
        auc1 = roc_auc_score(y[v_mask], results_dict["1. Full 19D (상체 보상 포함)"]["oof"][v_mask])
        auc2 = roc_auc_score(y[v_mask], results_dict["2. No Pure Trunk Lean (canting + counterbalance 제거, 17D)"]["oof"][v_mask])
        auc4 = roc_auc_score(y[v_mask], results_dict["4. No Trunk Lateral at all (상체 횡방향 3종 전면 제거, 16D)"]["oof"][v_mask])
        auc5 = roc_auc_score(y[v_mask], results_dict["5. Pure Lower-Body (상체 5종 전면 제거, 14D)"]["oof"][v_mask])
        print(f"{vn:<15} | {auc1*100:>12.2f}% | {auc2*100:>20.2f}% | {auc4*100:>16.2f}% | {auc5*100:>16.2f}%")

if __name__ == "__main__":
    main()
