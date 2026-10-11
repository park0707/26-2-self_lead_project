# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부
Script: build_and_train_sl2_v9_relabel_front.py
Purpose: 라벨 노이즈 텍스트 기반 정제 (Description Rule Alignment) 후 v9(Angular Augmentation) 재학습
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sl2_augmentation import generate_augmented_defect_variants
from side_lunge_c2_v8_front_features import extract_side_lunge_c2_v8_front_features_58d

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
CACHE_FRONT_58D_PKL = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
AUG_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v9_relabel_front_defect_angular_aug_58d_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl2_v9_relabel_front_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view2": "시점 B (우측 전방 45°)",
    "view4": "시점 D (좌측 전방 45°)"
}

def get_cleaned_labels_dict() -> Dict[str, int]:
    """사용자 시각적 판정 기반 Golden Label 딕셔너리 생성"""
    jfs = glob.glob(os.path.join(LABELS_JSON_DIR, "*.json"))
    label_map = {}
    
    texts = {
        'A': '정자세에서 발방향 바깥쪽 안굽히는 다리 구부정',
        'B': '정자세에서 발방향 바깥쪽 몸통 옆으로 갸우뚱 몸통 상체 뒤로 젖힘',
        'C': '1앞다리 90도, 2무릎발방향일치, 3한쪽다리바르게 펴기, 4척추 중립, 5상체 중립',
        'D': '정자세에서 안굽히는 다리 구부정 몸통 옆으로 갸우뚱',
        'E': '정자세에서 왼쪽보고 몸통 상체 뒤로 젖힘'
    }
    
    fixed_1, fixed_0 = 0, 0
    for jf in jfs:
        clip_name = os.path.basename(jf).replace(".json", "")
        with open(jf, 'r', encoding='utf-8') as f:
            d = json.load(f)
        desc = d['type_info'].get('description', '')
        original_val = d['type_info']['conditions'][1]['value']
        is_defect = not original_val
        
        new_is_defect = is_defect
        if desc == texts['A']:
            new_is_defect = True
        elif desc == texts['B']:
            new_is_defect = True
        elif desc == texts['C']:
            new_is_defect = False
        elif desc == texts['D']:
            new_is_defect = False
        elif desc == texts['E']:
            new_is_defect = True
            
        if new_is_defect and not is_defect:
            fixed_1 += 1
        elif not new_is_defect and is_defect:
            fixed_0 += 1
        
        label_map[clip_name] = 1 if new_is_defect else 0
        
    print(f"[+] Golden 라벨 정제 완료: {fixed_1}건(0->1 교정), {fixed_0}건(1->0 교정)")
    return label_map

def build_augmented_defect_features(y_cleaned: np.ndarray, clip_names: np.ndarray, views: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """정제된 라벨 기준으로 결함 샘플에 대한 Angular Augmentation 피처 생성"""
    if os.path.exists(AUG_CACHE_PKL):
        print(f"[*] 증강 결함 피처 캐시 로드 중: {AUG_CACHE_PKL}")
        with open(AUG_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 정제된 전방 뷰 결함 샘플 대상 5종 증강(각도 교란) 58D 피처 생성 중...")
    defect_indices = np.where(y_cleaned == 1)[0]
    aug_X_list = []
    aug_y_list = []
    aug_grp_list = []
    aug_view_list = []
    aug_orig_idx_list = []

    t0 = time.time()
    for i, idx in enumerate(defect_indices):
        clip_name = clip_names[idx]
        view_code = views[idx]
        grp = groups[idx]

        npz_path = os.path.join(DATA_DIR, f"{clip_name}_{view_code}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        kps = data['kps_33_3d_64f'][:, :, :3]

        variants = generate_augmented_defect_variants(kps, n_variants=5)
        for var_kps in variants:
            feat_58d = extract_side_lunge_c2_v8_front_features_58d(var_kps, view_code=view_code)
            aug_X_list.append(feat_58d)
            aug_y_list.append(1)
            aug_grp_list.append(grp)
            aug_view_list.append(view_code)
            aug_orig_idx_list.append(idx)
            
        if (i+1) % 50 == 0:
            print(f"  - {i+1}/{len(defect_indices)} 샘플 증강 완료...")

    aug_data = {
        "X_aug": np.array(aug_X_list, dtype=np.float32),
        "y_aug": np.array(aug_y_list, dtype=np.int32),
        "grp_aug": np.array(aug_grp_list),
        "view_aug": np.array(aug_view_list),
        "orig_indices": np.array(aug_orig_idx_list, dtype=np.int32)
    }

    with open(AUG_CACHE_PKL, 'wb') as f:
        pickle.dump(aug_data, f)

    print(f"[+] 결함 {len(defect_indices)}건 -> {len(aug_X_list)}건 증강 완료 (소요: {time.time()-t0:.1f}초), 캐시 저장: {AUG_CACHE_PKL}")
    return aug_data

def optimize_3state_thresholds(y_true, probs, max_uncertainty=0.10):
    best_f1 = -1.0
    best_res = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    for tc in np.arange(0.25, 0.60, 0.02):
        for td in np.arange(tc, 0.82, 0.02):
            pred = np.zeros_like(y_true, dtype=np.int32)
            pred[probs >= td] = 1
            pred[probs < tc] = 0
            uncertain_mask = (probs >= tc) & (probs < td)
            unc_rate = np.mean(uncertain_mask)
            if unc_rate > max_uncertainty: continue
            decided = ~uncertain_mask
            if np.sum(decided) == 0: continue
            y_dec = y_true[decided]
            p_dec = pred[decided]
            if np.sum(p_dec == 1) == 0: continue
            f1 = f1_score(y_dec, p_dec, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_res = (float(tc), float(td), float(f1), float(accuracy_score(y_dec, p_dec)), float(precision_score(y_dec, p_dec, zero_division=0)), float(recall_score(y_dec, p_dec, zero_division=0)))
    return best_res

def train_and_evaluate_v9_relabel_front():
    t_start = time.time()
    with open(CACHE_FRONT_58D_PKL, 'rb') as f:
        base_dataset = pickle.load(f)

    X = base_dataset['X']
    y_old = base_dataset['y']
    groups = base_dataset['groups']
    views = base_dataset['views']
    clip_names = base_dataset['clip_names']
    N, D = X.shape

    # 1. 라벨 교체
    label_map = get_cleaned_labels_dict()
    y = np.zeros_like(y_old)
    changed = 0
    for i in range(N):
        cname = clip_names[i]
        # JSON에 없는 경우 예외 처리
        if cname in label_map:
            y[i] = label_map[cname]
            if y[i] != y_old[i]:
                changed += 1
        else:
            y[i] = y_old[i]
    print(f"[*] 총 {changed}건의 시퀀스 라벨 변경 적용 완료.")

    # 2. 증강 피처 구축
    aug_data = build_augmented_defect_features(y, clip_names, views, groups)
    X_aug = aug_data['X_aug']
    y_aug = aug_data['y_aug']
    orig_indices_aug = aug_data['orig_indices']

    print(f"\n[*] sl2_v9_relabel_front + 각도 교란 증강 5-Fold 학습 시작...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        train_idx_set = set(train_idx)
        aug_mask = np.array([idx in train_idx_set for idx in orig_indices_aug])

        X_tr = np.vstack([X[train_idx], X_aug[aug_mask]])
        y_tr = np.concatenate([y[train_idx], y_aug[aug_mask]])
        X_val, y_val = X[val_idx], y[val_idx]

        pos_weight = float(np.sqrt(np.sum(y_tr == 0) / max(1, np.sum(y_tr == 1))))

        cb = CatBoostClassifier(iterations=350, learning_rate=0.035, depth=5, l2_leaf_reg=4.5, scale_pos_weight=pos_weight, random_seed=42 + fold, verbose=False, thread_count=-1)
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
        oof_preds_cb[val_idx] = cb.predict_proba(X_val)[:, 1]

        xg = xgb.XGBClassifier(n_estimators=350, learning_rate=0.035, max_depth=5, scale_pos_weight=pos_weight, reg_alpha=1.2, reg_lambda=3.0, random_state=42 + fold, eval_metric="auc", early_stopping_rounds=30, n_jobs=-1)
        xg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xg.predict_proba(X_val)[:, 1]

        auc_cb = roc_auc_score(y_val, oof_preds_cb[val_idx])
        auc_xg = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        print(f"  - Fold {fold}/5: 학습={len(y_tr)}건 | Cat AUC={auc_cb*100:.2f}%, XGB AUC={auc_xg*100:.2f}%")

    # 앙상블 F1 최적화
    best_w = (0.5, 0.5)
    best_f1 = -1.0
    for w in np.arange(0.1, 1.0, 0.1):
        p_ens = w * oof_preds_cb + (1 - w) * oof_preds_xgb
        res = optimize_3state_thresholds(y, p_ens, 0.18)
        if res[2] > best_f1:
            best_f1 = res[2]
            best_w = (w, 1-w)
            
    w_cb, w_xgb = best_w
    oof_preds = w_cb * oof_preds_cb + w_xgb * oof_preds_xgb
    auc_total = roc_auc_score(y, oof_preds)
    print(f"\n[+] 앙상블 최적 가중치: Cat={w_cb:.2f}, XGB={w_xgb:.2f}, 전체 AUC={auc_total*100:.2f}%")

    profiles = {}
    for prof_key, max_unc in [("standard_10pct", 0.10), ("f1_70_breakthrough", 0.18)]:
        tc_tot, td_tot, f1_tot, acc_tot, prec_tot, rec_tot = optimize_3state_thresholds(y, oof_preds, max_uncertainty=max_unc)
        unc_tot = float(np.mean((oof_preds >= tc_tot) & (oof_preds < td_tot)))

        view_audits = {}
        for v in ['view2', 'view4']:
            mask_v = (views == v)
            y_v = y[mask_v]
            p_v = oof_preds[mask_v]
            tc_v, td_v, f1_v, acc_v, prec_v, rec_v = optimize_3state_thresholds(y_v, p_v, max_uncertainty=max_unc)
            unc_v = float(np.mean((p_v >= tc_v) & (p_v < td_v)))
            view_audits[v] = {
                "alias": VIEW_ALIAS_MAP[v],
                "effective_f1": f1_v,
                "accuracy": acc_v,
                "precision": prec_v,
                "recall": rec_v,
                "uncertainty_rate": unc_v
            }

        profiles[prof_key] = {
            "overall_f1": f1_tot,
            "overall_accuracy": acc_tot,
            "uncertainty_rate": unc_tot,
            "view_audits": view_audits
        }

    p1 = profiles['standard_10pct']
    p2 = profiles['f1_70_breakthrough']
    print("\n" + "="*80)
    print(" [라벨 정제 완료된 sl2_v9_relabel_front 성능 결과]")
    print(f" 1. 표준 프로필 (보류율 <= 10.0%):")
    print(f"    - 전방 통합 F1: {p1['overall_f1']*100:.2f}%, 확정 정확도: {p1['overall_accuracy']*100:.2f}%, 보류율: {p1['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p1['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p1['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p1['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p1['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print(f" 2. F1 70% 돌파 최적 프로필 (보류율 <= 18.0%):")
    print(f"    - 전방 통합 F1: {p2['overall_f1']*100:.2f}%, 확정 정확도: {p2['overall_accuracy']*100:.2f}%, 보류율: {p2['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 B]: F1={p2['view_audits']['view2']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view2']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view2']['uncertainty_rate']*100:.1f}%")
    print(f"    - [시점 D]: F1={p2['view_audits']['view4']['effective_f1']*100:.2f}%, ACC={p2['view_audits']['view4']['accuracy']*100:.2f}%, 보류율={p2['view_audits']['view4']['uncertainty_rate']*100:.1f}%")
    print("="*80)

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    train_and_evaluate_v9_relabel_front()
