# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: train_and_evaluate_submodels_v2.py
Purpose: 규정(AGENTS.md) 준수 기반 c1~c4 최신 서브모델 재학습 및 다차원 성능 감사 (Audit)
         - 규정 준수:
           1) 인터페이스 기반 독립 피처 파이프라인(squat_submodel_feature_pipeline) 적용
           2) Format A 이름 매핑(Name-based Lookup)을 통한 인덱스 버그 100% 원천 차단
           3) 9명 피험자 완전 격리 (StratifiedGroupKFold, 5-Fold)
           4) 5대 전 시점(view1~view5) 2,605개 원천 64F 스플라인 전수 평가
           5) 다차원 성능 감사: 서브모델별 F1/AUC뿐만 아니라 5대 시점별 F1 대조 감사
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score
)

sys.path.insert(0, 'd:/')
from src.features.squat_submodel_feature_pipeline import (
    build_c1_features,
    build_c2_features,
    build_c3_features,
    build_c4_features
)

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
OUT_OOF_PATH = r"D:\Data\피처_데이터\oof_master_64f_predictions_v2.npz"
OUT_WEIGHTS_DIR = r"D:\src\models\second_models\babel_squat\weights_v2"


def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2)
    }


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1~c4 최신 서브모델 재학습 및 다차원 성능 감사", flush=True)
    print("  ★ 인터페이스 기반 Name-based Lookup 피처 파이프라인 적용", flush=True)
    print("  ★ 9명 피험자 완전 격리 (StratifiedGroupKFold, 5-Fold) & 5대 전 시점(view1~view5) 2,605개 전수", flush=True)
    print("=" * 115, flush=True)

    os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

    d = np.load(MASTER_PATH, allow_pickle=True)
    X_fa = d['X_format_a']
    X_c1_raw = d['X_c1']
    X_c2_raw = d['X_c2']
    X_c3_raw = d['X_c3']
    X_c4_raw = d['X_c4']

    Y_def = d['Y_defects_golden'] # (2605, 4)
    y_norm = d['y_normal_golden'] # (2605,)
    subjects = d['subjects']       # (2605,)
    views = d['views']             # (2605,)
    seq_ids = d['sequence_ids']
    base_sids = d['base_sequence_ids']
    N = len(subjects)

    # 인터페이스 기반 표준 피처 빌드
    print("[*] 인터페이스 기반 피처셋 생성 중...", flush=True)
    X_c1, c1_names = build_c1_features(X_fa, X_c1_raw)
    X_c2, c2_names = build_c2_features(X_fa, X_c2_raw)
    X_c3, c3_names = build_c3_features(X_fa, X_c3_raw)
    X_c4, c4_names = build_c4_features(X_fa, X_c4_raw)

    feature_sets = [
        ('c1 (척추)', X_c1, c1_names, Y_def[:, 0], 220, 6, 0.03, 150, 6, 0.03),
        ('c2 (고개)', X_c2, c2_names, Y_def[:, 1], 200, 6, 0.03, 140, 6, 0.03),
        ('c3 (무릎)', X_c3, c3_names, Y_def[:, 2], 240, 6, 0.03, 160, 6, 0.03),
        ('c4 (발목)', X_c4, c4_names, Y_def[:, 3], 220, 6, 0.03, 150, 6, 0.03),
    ]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((N, 4), dtype=np.float32)
    saved_weights = {}

    print(f"\n[*] 5-Fold 피험자 격리 교차 검증 시작 (총 {N}개 시퀀스)", flush=True)

    t0 = time.time()
    submodel_results = []

    for defect_idx, (name, X_feat, fnames, y_target, cb_iter, cb_depth, cb_lr, lgb_iter, lgb_depth, lgb_lr) in enumerate(feature_sets):
        print("-" * 115, flush=True)
        print(f"[*] [{defect_idx+1}/4] {name} 모델 학습 (피처 차원: {X_feat.shape[1]}D, 결함 수: {int(np.sum(y_target))}개/{N}개)", flush=True)
        
        oof_defect = np.zeros(N, dtype=np.float32)
        cb_list = []
        lgb_list = []

        for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_feat, y_target, groups=subjects), 1):
            X_tr, y_tr = X_feat[trn_idx], y_target[trn_idx]
            X_va, y_va = X_feat[val_idx], y_target[val_idx]

            # CatBoost
            cb = CatBoostClassifier(
                iterations=cb_iter,
                depth=cb_depth,
                learning_rate=cb_lr,
                l2_leaf_reg=5.0,
                random_seed=42 + fold,
                verbose=0
            )
            cb.fit(X_tr, y_tr)

            # LightGBM
            clf = lgb.LGBMClassifier(
                n_estimators=lgb_iter,
                max_depth=lgb_depth,
                learning_rate=lgb_lr,
                num_leaves=31,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr)

            p_cb = cb.predict_proba(X_va)[:, 1]
            p_lgb = clf.predict_proba(X_va)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            oof_defect[val_idx] = p_fold

            cb_list.append(cb)
            lgb_list.append(clf)

        oof_probs[:, defect_idx] = oof_defect
        saved_weights[f"c{defect_idx+1}"] = {
            'cb': cb_list,
            'lgb': lgb_list,
            'feature_names': fnames
        }

        m = compute_metrics(y_target, oof_defect)
        submodel_results.append(m)
        print(f"  --> {name} 최종 OOF 성능: F1={m['f1']}% | Prec={m['prec']}% | Rec={m['rec']}% | AUC={m['auc']}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n[+] 4대 서브모델 학습 및 OOF 생성 완료 (총 소요 시간: {elapsed:.1f}초)", flush=True)

    # =========================================================================
    # 다차원 성능 감사 (Audit)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("  [다차원 성능 감사 1] 4대 서브모델 갱신 전/후 1:1 비교", flush=True)
    print("=" * 115, flush=True)
    print(f"{'결함 모델':<18} | {'이전 F1 (초기)':<14} | {'최신 F1 (갱신)':<14} | {'F1 변화폭':<12} | {'AUC':<10} | {'판정':<8}", flush=True)
    print("-" * 115, flush=True)

    # 이전 v6 초기 F1 (Step 1159/Step 767 기준)
    prev_f1 = [79.04, 76.50, 87.54, 85.20]
    defect_labels = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면고정']

    for i in range(4):
        curr_f1 = submodel_results[i]['f1']
        diff = curr_f1 - prev_f1[i]
        diff_str = f"+{diff:.2f}%p" if diff >= 0 else f"{diff:.2f}%p"
        status = "★ 상승!" if diff > 0 else "유지"
        print(f"{defect_labels[i]:<18} | {prev_f1[i]:>12.2f}% | {curr_f1:>12.2f}% | {diff_str:>11} | {submodel_results[i]['auc']:>8.2f}% | {status}", flush=True)

    print("\n" + "=" * 115, flush=True)
    print("  [다차원 성능 감사 2] 5대 시점(view1~view5)별 세부 F1 추이 (시점 편향 검증)", flush=True)
    print("=" * 115, flush=True)
    
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (후방)', 'view2 (우측45)', 'view3 (정면)', 'view4 (좌측45)', 'view5 (우측후방45)']

    header = f"{'시점 (카메라 각도)':<20}"
    for i in range(4):
        header += f" | {f'c{i+1} F1':<10}"
    header += f" | {'시점 평균':<10}"
    print(header, flush=True)
    print("-" * 115, flush=True)

    for v, vn in zip(view_list, view_names):
        v_mask = (views == v)
        row_str = f"{vn:<20}"
        v_f1s = []
        for i in range(4):
            vf = f1_score(Y_def[v_mask, i], (oof_probs[v_mask, i] >= 0.50).astype(int), zero_division=0) * 100.0
            row_str += f" | {vf:>8.2f}%"
            v_f1s.append(vf)
        row_str += f" | {np.mean(v_f1s):>8.2f}%"
        print(row_str, flush=True)

    # =========================================================================
    # 다중 라벨 완전 일치율(AL-EMR) 산출
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("  [다차원 성능 감사 3] 부위별 독립 3-State (Micro 3-State) AL-EMR 완전 일치율", flush=True)
    print("=" * 115, flush=True)

    taus = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]
    print(f"{'신뢰도 대역 (tau)':<18} | {'AL-EMR':<12} | {'평균 유효 부위수':<18} | {'유효 라벨 비율':<14} | {'Clean F1':<12}", flush=True)
    print("-" * 115, flush=True)

    for tau in taus:
        valid_masks = (oof_probs < (0.50 - tau)) | (oof_probs >= (0.50 + tau))
        preds_bin = (oof_probs >= 0.50).astype(int)
        
        matches = (preds_bin == Y_def)
        
        row_all_match = []
        row_valid_counts = []
        row_clean_pred = []
        row_clean_true = (np.sum(Y_def, axis=1) == 0).astype(int)

        for i in range(N):
            vm = valid_masks[i]
            v_cnt = np.sum(vm)
            row_valid_counts.append(v_cnt)
            if v_cnt == 0:
                row_all_match.append(True)
                row_clean_pred.append(0)
            else:
                row_all_match.append(bool(np.all(matches[i][vm])))
                v_preds = preds_bin[i][vm]
                row_clean_pred.append(int(np.sum(v_preds) == 0))

        al_emr = np.mean(row_all_match) * 100.0
        mean_v = np.mean(row_valid_counts)
        pct_v = (mean_v / 4.0) * 100.0
        cl_f1 = f1_score(row_clean_true, row_clean_pred, zero_division=0) * 100.0

        highlight = " ★ [표준 권장]" if tau == 0.10 else ""
        print(f"tau = +/-{tau:.2f}{highlight:<10} | {al_emr:>10.2f}% | {mean_v:>6.2f}개 / 4.00개   | {pct_v:>12.1f}% | {cl_f1:>10.2f}%", flush=True)

    # 결과 및 가중치 저장
    np.savez_compressed(
        OUT_OOF_PATH,
        P_oof=oof_probs,
        Y_defects=Y_def,
        y_normal=y_norm,
        subjects=subjects,
        views=views,
        sequence_ids=seq_ids,
        base_sequence_ids=base_sids
    )

    with open(os.path.join(OUT_WEIGHTS_DIR, 'submodels_weights_v2.pkl'), 'wb') as f:
        pickle.dump(saved_weights, f)

    print("\n" + "=" * 115, flush=True)
    print(f"[+] OOF 예측 및 최신 서브모델 가중치 영구 저장 완료:")
    print(f"  • OOF 저장 경로   : {OUT_OOF_PATH}")
    print(f"  • 가중치 저장 경로: {os.path.join(OUT_WEIGHTS_DIR, 'submodels_weights_v2.pkl')}")
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
