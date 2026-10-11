# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: optimize_feature_pruning_backward.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)
Purpose:
  v5.0 25D 피처셋에서 LOFO 감사 결과를 기반으로
  순차적 후진 제거(Sequential Backward Elimination)를 수행하여
  F1/AUC를 극대화하는 '최적 정예 피처 서브셋(Elite Minimal Subset)' 탐색
"""

import os
import sys
import json
import glob
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'D:/')
from src.models.second_models.babel_deadlift.d2_model_v5_final import (
    extract_d2_v5_final_features, FEATURE_NAMES_25_D2
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_DIR = r"D:\src\models\second_models\babel_deadlift\weights"

def run_backward_pruning():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 순차적 후진 제거 최적화 (Greedy Backward Elimination)")
    print("  - 목적: F1 저해 피처 및 중복 고상관 피처 제거를 통한 일반화 F1/AUC 극대화")
    print("  - 데이터셋: AI-Hub 385개 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 검증 프로토콜: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리)")
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

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)

    from cleanlab.filter import find_label_issues

    def evaluate_subset(feat_indices):
        oof_p = np.zeros(len(y), dtype=np.float32)
        X_sub = X[:, feat_indices]
        for tr_idx, val_idx in sgkf.split(X_sub, y, groups=subs):
            X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
            X_val = X_sub[val_idx]

            # Cleanlab within training fold (top 30 label noise pruning)
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
            oof_p[val_idx] = 0.5 * p_l + 0.5 * p_c

        auc = roc_auc_score(y, oof_p)
        best_f1 = 0.0
        best_prec = 0.0
        best_rec = 0.0
        best_win = None
        for tn in np.arange(0.20, 0.42, 0.005):
            for td in np.arange(tn + 0.005, 0.50, 0.005):
                dec = (oof_p <= tn) | (oof_p >= td)
                if np.mean(dec) < 0.90: continue
                yy = y[dec]
                pred = (oof_p[dec] >= td).astype(int)
                f1 = f1_score(yy, pred, zero_division=0) * 100
                if f1 > best_f1:
                    best_f1 = f1
                    best_prec = precision_score(yy, pred, zero_division=0) * 100
                    best_rec = recall_score(yy, pred, zero_division=0) * 100
                    best_win = [round(float(tn), 3), round(float(td), 3)]
        return auc, best_f1, best_prec, best_rec, best_win, oof_p

    current_indices = list(range(25))
    init_auc, init_f1, init_p, init_r, init_win, _ = evaluate_subset(current_indices)
    print(f"\n[*] 초기 25D 전체 성능: AUC={init_auc:.4f} | 3-State F1={init_f1:.2f}% (Prec={init_p:.2f}%, Rec={init_r:.2f}%)")

    history = [{
        'round': 0,
        'n_features': len(current_indices),
        'indices': list(current_indices),
        'names': [FEATURE_NAMES_25_D2[i] for i in current_indices],
        'auc': init_auc,
        'f1': init_f1,
        'prec': init_p,
        'rec': init_r,
        'win': init_win
    }]

    round_num = 1
    # 순차적 후진 제거 (최소 6개 피처가 남을 때까지)
    while len(current_indices) > 6:
        print(f"\n--- [Round {round_num}] 현재 피처 수: {len(current_indices)}개 ---")
        best_candidate = None
        best_cand_idx = None
        best_metric = -1.0

        for cand_idx in current_indices:
            trial_indices = [i for i in current_indices if i != cand_idx]
            auc_t, f1_t, prec_t, rec_t, win_t, _ = evaluate_subset(trial_indices)
            # F1-Score를 최우선으로 하되 AUC를 보조 지표로 종합 평가
            composite_score = f1_t + 20.0 * auc_t
            if composite_score > best_metric:
                best_metric = composite_score
                best_candidate = (auc_t, f1_t, prec_t, rec_t, win_t)
                best_cand_idx = cand_idx

        drop_name = FEATURE_NAMES_25_D2[best_cand_idx]
        current_indices.remove(best_cand_idx)
        b_auc, b_f1, b_prec, b_rec, b_win = best_candidate

        history.append({
            'round': round_num,
            'dropped_feature': drop_name,
            'n_features': len(current_indices),
            'indices': list(current_indices),
            'names': [FEATURE_NAMES_25_D2[i] for i in current_indices],
            'auc': b_auc,
            'f1': b_f1,
            'prec': b_prec,
            'rec': b_rec,
            'win': b_win
        })

        f1_delta = b_f1 - history[-2]['f1']
        auc_delta = b_auc - history[-2]['auc']
        print(f"  [제거 완료] '{drop_name}' 제거됨 -> 남은 피처 {len(current_indices)}개")
        print(f"             AUC={b_auc:.4f} ({auc_delta:+.4f}) | F1={b_f1:.2f}% ({f1_delta:+.2f}%p) | Prec={b_prec:.2f}%")

        round_num += 1

    # 최적 서브셋 탐색 (F1 최대치 기준)
    best_step = max(history, key=lambda x: (x['f1'], x['auc']))
    print("\n" + "=" * 115)
    print(f"  [★ 후진 제거 최적화 완료! 최고의 정예 서브셋 선정 결과]")
    print("=" * 115)
    print(f"  • 최적 피처 수      : {best_step['n_features']}개 (25개 중 {25 - best_step['n_features']}개 저해/중복 피처 제거 완료!)")
    print(f"  • 최적 ROC-AUC      : {best_step['auc']:.4f} (초기 25D 대비 {best_step['auc'] - init_auc:+.4f})")
    print(f"  • 최적 3-State F1   : {best_step['f1']:.2f}% (초기 25D 대비 {best_step['f1'] - init_f1:+.2f}%p🔺)")
    print(f"  • 최적 정밀도 (Prec): {best_step['prec']:.2f}%")
    print(f"  • 최적 채택 윈도우  : {best_step['win']}")
    print("\n  [최종 확정된 정예 피처 명단]:")
    for idx, name in enumerate(best_step['names'], 1):
        print(f"    {idx:2d}. {name}")

    out_file = os.path.join(OUT_DIR, "d2_backward_pruning_results.json")
    with open(out_file, 'w', encoding='utf-8') as fp:
        json.dump({
            'initial': history[0],
            'best_step': best_step,
            'history': history
        }, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] 최적화 결과 JSON 저장 완료: {out_file}")

if __name__ == "__main__":
    run_backward_pruning()
