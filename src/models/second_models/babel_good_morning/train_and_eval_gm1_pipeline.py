# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 굿모닝 (09_굿모닝)
Defect: 무릎 구부린 채 고정 위반 (gm1)
Script: train_and_eval_gm1_pipeline.py
Purpose:
  1. 스티프 데드리프트 sd1의 피처와 구조를 계승
  2. v6 파이프라인 (정규 3D 변환, KADR, Excess Flexion) 피처 추출
  3. 피험자 1~9 전수 StratifiedGroupKFold 5-Fold 교차검증 훈련
  4. 공식 5대 시점(시점 A ~ 시점 E) 성능 보고
"""

import os
import sys
import json
import pickle
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    FEATURE_NAMES_19D
)

CACHE_PATH = os.path.join(CURRENT_DIR, "gm1_dataset_cache.npz")
JSON_PATH = os.path.join(CURRENT_DIR, "gm1_benchmark_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "weights", "gm1_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# GM1 정예 피처 (sd1 v7에서 context_sd3_prob 제외)
GM1_FEATURE_INDICES = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15, 17]
GM1_FEATURE_NAMES = [FEATURE_NAMES_19D[i] for i in GM1_FEATURE_INDICES]

def run_pipeline():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 굿모닝 gm1 훈련 및 벤치마크 평가 (sd1 계승)", flush=True)
    print("=" * 100, flush=True)

    # 1. 데이터 로드
    data = np.load(CACHE_PATH, allow_pickle=True)
    train_labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    sequence_ids = data['sequence_ids']
    X_seqs = data['X']

    N_valid = len(train_labels)

    # 2. sd1 계승 피처 추출
    print("[*] sd1 계승 3D 생체역학 피처 추출 중...", flush=True)
    feats_19d_all = np.zeros((N_valid, 19), dtype=np.float32)
    for i in range(N_valid):
        # sd3 보조 분류 확률은 0.0으로 고정 (굿모닝에 해당 없음)
        feats_19d_all[i] = extract_knee_invariance_features_v6(X_seqs[i], p_sd3=0.0)

    # 정예 피처 슬라이싱
    meta_features = feats_19d_all[:, GM1_FEATURE_INDICES]
    
    print(f"[*] 피처 추출 완료. 샘플 수={N_valid}, 정상={np.sum(train_labels==0)}, 결함={np.sum(train_labels==1)}")

    # 4. 피험자 격리 5-Fold 교차검증 (CatBoost 정예 모델)
    print("\n" + "-" * 80, flush=True)
    print("  [피험자 격리 5-Fold 교차검증 (StratifiedGroupKFold, 9 Subjects)]", flush=True)
    print("-" * 80, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_proba = np.zeros(N_valid, dtype=np.float32)
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(np.arange(N_valid), train_labels, subject_ids)):
        X_tr = meta_features[tr_idx]
        y_tr = train_labels[tr_idx]
        X_val = meta_features[val_idx]
        y_val = train_labels[val_idx]

        cb = CatBoostClassifier(
            iterations=200,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=5,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        )
        cb.fit(X_tr, y_tr)
        fold_models.append(cb)

        val_preds = cb.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds) * 100.0
        print(f"  [Fold {fold+1}/5] Train: {len(tr_idx)}개, Val: {len(val_idx)}개 (Val AUC: {fold_auc:.2f}%)", flush=True)

    # 5. 2-State 전수 평가
    preds_2s = (oof_proba >= 0.50).astype(int)
    auc_l13 = roc_auc_score(train_labels, oof_proba)
    f1_macro_l13 = f1_score(train_labels, preds_2s, average='macro', zero_division=0)
    f1_clean_l13 = f1_score(train_labels, preds_2s, pos_label=0, zero_division=0)
    f1_defect_l13 = f1_score(train_labels, preds_2s, pos_label=1, zero_division=0)
    tn_l13, fp_l13, fn_l13, tp_l13 = confusion_matrix(train_labels, preds_2s).ravel()
    spec_l13 = tn_l13 / (tn_l13 + fp_l13)
    acc_l13 = accuracy_score(train_labels, preds_2s)
    
    print("\n[★ 2-State 진단 성능]:")
    print(f"  - 전체 ROC-AUC:        {auc_l13*100:.2f}%")
    print(f"  - 결함 F1-score:       {f1_defect_l13*100:.2f}%")
    print(f"  - 정상 F1:             {f1_clean_l13*100:.2f}%")
    print(f"  - Macro F1:            {f1_macro_l13*100:.2f}%")
    print(f"  - Clean Specificity:   {spec_l13*100:.2f}%")

    # 8. 공식 5대 시점별 성적
    print("\n" + "-" * 95, flush=True)
    print("  [공식 5대 시점별 결함 F1]", flush=True)
    print("-" * 95, flush=True)
    view_results = {}
    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (view_ids == vn)
        y_v = train_labels[v_mask]
        st_v = preds_2s[v_mask]
        if len(y_v) > 0:
            v_f1_d = float(f1_score(y_v, st_v, pos_label=1, zero_division=0) * 100.0)
            v_f1_m = float(f1_score(y_v, st_v, average='macro', zero_division=0) * 100.0)
            view_results[vn] = {
                'alias': v_alias,
                'count': int(np.sum(v_mask)),
                'defect_f1': v_f1_d,
                'macro_f1': v_f1_m
            }
            print(f"  * {v_alias:25s}: 결함 F1={v_f1_d:5.2f}% | Macro F1={v_f1_m:5.2f}% (N={np.sum(v_mask)})", flush=True)
        else:
            print(f"  * {v_alias:25s}: 해당 샘플 없음", flush=True)

    # 9. 피처 중요도 분석
    feature_importances = np.mean([m.get_feature_importance() for m in fold_models], axis=0)
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features = {GM1_FEATURE_NAMES[idx]: float(feature_importances[idx]) for idx in sorted_idx}

    print("\n" + "-" * 95, flush=True)
    print("  [gm1 피처 중요도 순위]", flush=True)
    print("-" * 95, flush=True)
    for rank, (fname, imp) in enumerate(top_features.items(), 1):
        print(f"  {rank:2d}. {fname:28s}: {imp:6.2f}", flush=True)

    # 10. JSON 저장
    benchmark_data = {
        'exercise': '09_굿모닝',
        'defect': '무릎 구부린 채 고정 위반 (gm1)',
        'metrics': {
            'macro_f1': float(f1_macro_l13),
            'clean_f1': float(f1_clean_l13),
            'defect_f1': float(f1_defect_l13),
            'clean_specificity': float(spec_l13),
            'roc_auc': float(auc_l13),
            'accuracy': float(acc_l13)
        },
        'view_results': view_results,
        'top_features': top_features
    }

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] gm1 벤치마크 JSON 저장 완료: {JSON_PATH}", flush=True)

    # 11. 가중치 번들 저장
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle = {
        'version': 'gm1_v1',
        'fold_models': fold_models,
        'feature_indices': GM1_FEATURE_INDICES,
        'feature_names': GM1_FEATURE_NAMES
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] gm1 공식 가중치 번들 배포 완료: {BUNDLE_SAVE_PATH}", flush=True)

if __name__ == '__main__':
    run_pipeline()
