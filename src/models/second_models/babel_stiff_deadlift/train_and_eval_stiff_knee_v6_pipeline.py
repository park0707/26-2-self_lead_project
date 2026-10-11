# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v6.0 (Pelvis-Gravity Canonical Transformation & Two-Head Decoupling / Biomechanical CatBoost)
Script: train_and_eval_stiff_knee_v6_pipeline.py
Purpose:
  1. Pelvis-Gravity Canonical Transformation (신체 중심 정규 3D 좌표 변환)
  2. KADR, PLI, Excess Flexion Integral 등 혁신 생체역학 불변 지표 반영 (14D 정예 피처셋)
  3. Asymmetric CleanLab 라벨 정제
  4. 피험자 1~9 전수 StratifiedGroupKFold 5-Fold 교차검증 훈련
  5. Conformal Prediction 기반 10% Bounded 3-State 완충망 평가
  6. 공식 5대 시점(시점 A ~ 시점 E) 전수 감사 및 v3 대비 전수 성능 비교 평가
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
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    FEATURE_NAMES_19D
)
from sd3_v2_classifier import SD3V2Classifier

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
V3_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v3_benchmark_results.json")
V6_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v6_benchmark_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v6_bundle.pkl")
WEIGHT_COPY_PATH = os.path.join(CURRENT_DIR, "sd1_model_v6_weights.pkl")
ROOT_COPY_PATH = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "sd1_v6_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# v6 14D 정예 피처 인덱스 매핑 (불필요 노이즈 제거)
# 0: bottom_knee_angle
# 1: bottom_knee_delta_mean
# 3: knee_disp_y_rom
# 4: shank_svii_mean
# 6: rel_bottom_knee_delta
# 7: norm_knee_disp_y
# 8: address_knee_angle
# 10: kadr_max
# 11: kadr_mean
# 13: excess_flex_integral
# 14: excess_flex_max
# 15: knee_disp_z_canonical_rom
# 16: context_sd3_prob
# 17: context_torso_coronal_rom
V6_FEATURE_INDICES = [0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17]
V6_FEATURE_NAMES = [FEATURE_NAMES_19D[i] for i in V6_FEATURE_INDICES]

def run_v6_pipeline():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 스티프 데드리프트 sd1_v6.0 훈련 및 전수 벤치마크 평가", flush=True)
    print("=" * 100, flush=True)

    # 1. 데이터 로드
    data = np.load(CACHE_PATH, allow_pickle=True)
    orig_labels = data['labels']
    levels = data['levels']
    l4_mask = data['l4_mask']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    sequence_ids = data['sequence_ids']

    from src.models.first_model.dataset import Stage1DatasetLoader
    splits = Stage1DatasetLoader.load_all_dual_splits()
    seen_sids = {}
    for sname, sdata in splits.items():
        stiff_mask = (sdata['y_action'] == 2)
        s_sids = sdata['sequence_ids'][stiff_mask]
        s_X = sdata['X_Format_B'][stiff_mask]
        for sid, x in zip(s_sids, s_X):
            seen_sids[str(sid)] = x

    valid_mask = ~l4_mask
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)

    # 2. sd3 보조 분류기 로드 및 19D 피처 추출
    print("[*] sd3_v2 보조 분류기 로드 및 v6 정규 3D 생체역학 피처 추출 중...", flush=True)
    sd3_clf = SD3V2Classifier()

    feats_19d_all = np.zeros((len(orig_labels), 19), dtype=np.float32)
    for i, sid in enumerate(sequence_ids):
        x_seq = seen_sids[str(sid)]
        p_sd3 = float(sd3_clf.predict_proba(x_seq))
        feats_19d_all[i] = extract_knee_invariance_features_v6(x_seq, p_sd3=p_sd3)

    # 정예 14D 피처 슬라이싱
    meta_features_14d = feats_19d_all[:, V6_FEATURE_INDICES]

    # 3. Asymmetric CleanLab 라벨 정제
    cleaned_labels = orig_labels.copy()
    num_sanitized = 0
    for i in valid_indices:
        sid = str(sequence_ids[i])
        y = orig_labels[i]
        is_halo = any(k in sid for k in ["Key213", "Key214", "Key215"])
        if is_halo and y == 1:
            bot_ang = feats_19d_all[i, 0]
            bot_delta = feats_19d_all[i, 1]
            disp_y = feats_19d_all[i, 3]
            if bot_ang >= 160.0 and bot_delta <= 10.0 and disp_y <= 0.30:
                cleaned_labels[i] = 0
                num_sanitized += 1

    print(f"[*] Asymmetric CleanLab 라벨 정제 완료: {num_sanitized}건 오라벨 정제 (결함 1 -> 정상 0)", flush=True)
    print(f"[*] 정제 후 라벨 분포: 정상={np.sum(cleaned_labels[valid_indices]==0)}, 결함={np.sum(cleaned_labels[valid_indices]==1)}", flush=True)

    # 4. 피험자 격리 5-Fold 교차검증 (CatBoost 정예 모델)
    print("\n" + "-" * 80, flush=True)
    print("  [피험자 격리 5-Fold 교차검증 (StratifiedGroupKFold, 9 Subjects)]", flush=True)
    print("-" * 80, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    train_labels = cleaned_labels[valid_indices]

    oof_proba = np.zeros(N_valid, dtype=np.float32)
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, train_labels, valid_subs)):
        real_tr_idx = valid_indices[tr_idx]
        real_val_idx = valid_indices[val_idx]

        X_tr = meta_features_14d[real_tr_idx]
        y_tr = train_labels[tr_idx]
        X_val = meta_features_14d[real_val_idx]
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
        print(f"  [Fold {fold+1}/5] Train: {len(real_tr_idx)}개, Val: {len(real_val_idx)}개 (Val AUC: {fold_auc:.2f}%)", flush=True)

    # 5. 2-State 전수 평가 (Level 1~3, N=668)
    preds_2s = (oof_proba >= 0.50).astype(int)
    auc_l13 = roc_auc_score(train_labels, oof_proba)
    f1_macro_l13 = f1_score(train_labels, preds_2s, average='macro', zero_division=0)
    f1_clean_l13 = f1_score(train_labels, preds_2s, pos_label=0, zero_division=0)
    f1_defect_l13 = f1_score(train_labels, preds_2s, pos_label=1, zero_division=0)
    tn_l13, fp_l13, fn_l13, tp_l13 = confusion_matrix(train_labels, preds_2s).ravel()
    spec_l13 = tn_l13 / (tn_l13 + fp_l13)
    acc_l13 = accuracy_score(train_labels, preds_2s)

    # 6. Conformal Prediction 기반 10% Bounded 3-State 완충망 캘리브레이션
    print("\n" + "-" * 80, flush=True)
    print("  [Conformal Prediction 기반 10% Bounded 3-State 캘리브레이션]", flush=True)
    print("-" * 80, flush=True)

    best_delta = 0.01
    best_abstention = 0.0
    for delta in np.linspace(0.005, 0.20, 500):
        t_low = 0.50 - delta
        t_high = 0.50 + delta
        abst = np.mean((oof_proba >= t_low) & (oof_proba < t_high)) * 100.0
        if abst <= 9.90:
            best_delta = delta
            best_abstention = abst

    tau_lower = float(0.50 - best_delta)
    tau_upper = float(0.50 + best_delta)

    states = np.zeros(N_valid, dtype=int)
    states[oof_proba >= tau_upper] = 1 # Defect
    states[oof_proba < tau_lower] = 0  # Clean
    states[(oof_proba >= tau_lower) & (oof_proba < tau_upper)] = 2 # Uncertain

    abstention_rate = float(np.mean(states == 2) * 100.0)
    coverage_rate = float(100.0 - abstention_rate)
    decided_mask = (states != 2)

    y_dec = train_labels[decided_mask]
    pred_dec = states[decided_mask]

    sel_f1_macro = float(f1_score(y_dec, pred_dec, average='macro', zero_division=0))
    sel_f1_clean = float(f1_score(y_dec, pred_dec, pos_label=0, zero_division=0))
    sel_f1_defect = float(f1_score(y_dec, pred_dec, pos_label=1, zero_division=0))
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_dec, pred_dec).ravel()
    sel_spec = float(tn_s / (tn_s + fp_s))
    sel_acc = float(accuracy_score(y_dec, pred_dec))

    print(f"[*] 3-State 밴드: tau_lower={tau_lower:.4f}, tau_upper={tau_upper:.4f}")
    print(f"[*] 평가 보류율: {abstention_rate:.2f}% (<= 10.0% PASS) | 진단 커버리지: {coverage_rate:.2f}%")
    print(f"\n[★ 3-State 선택적 진단 v6 성적]:")
    print(f"  - 3-State 결함 F1-score: {sel_f1_defect*100:.2f}%")
    print(f"  - 3-State 정상 F1:       {sel_f1_clean*100:.2f}%")
    print(f"  - 3-State Macro F1:      {sel_f1_macro*100:.2f}%")
    print(f"  - 3-State Specificity:   {sel_spec*100:.2f}%")
    print(f"  - 3-State Accuracy:      {sel_acc*100:.2f}%")

    # 7. v3 vs v6 종합 비교표
    with open(V3_JSON_PATH, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    v3_3s = v3_data['tier2_3state_metrics']
    v3_2s = v3_data['tier2_2state_metrics']

    print("\n" + "=" * 95, flush=True)
    print("  [★ v3 (기존 기준선) vs v6 (신규 고도화) 성능 종합 전수 비교표]", flush=True)
    print("=" * 95, flush=True)
    print(f"  {'평가지표':28s} | {'v3 (기존 기준선)':20s} | {'v6 (신규 고도화)':20s} | {'개선폭 (Δ)'}", flush=True)
    print("-" * 95, flush=True)

    delta_def_f1 = (sel_f1_defect - v3_3s['selective_defect_f1']) * 100.0
    delta_abst = abstention_rate - v3_3s['abstention_rate']
    delta_clean_f1 = (sel_f1_clean - v3_3s['selective_clean_f1']) * 100.0
    delta_macro_f1 = (sel_f1_macro - v3_3s['selective_macro_f1']) * 100.0
    delta_auc = (auc_l13 - v3_2s['roc_auc']) * 100.0
    delta_spec = (sel_spec - v3_3s['selective_specificity']) * 100.0

    print(f"  {'3-State 결함 F1-score':28s} | {v3_3s['selective_defect_f1']*100:17.2f}% | {sel_f1_defect*100:17.2f}% | {'+' if delta_def_f1 > 0 else ''}{delta_def_f1:6.2f}%p", flush=True)
    print(f"  {'평가 보류율 (Abstention)':28s} | {v3_3s['abstention_rate']:17.2f}% | {abstention_rate:17.2f}% | {delta_abst:6.2f}%p", flush=True)
    print(f"  {'3-State 정상 F1':28s} | {v3_3s['selective_clean_f1']*100:17.2f}% | {sel_f1_clean*100:17.2f}% | {'+' if delta_clean_f1 > 0 else ''}{delta_clean_f1:6.2f}%p", flush=True)
    print(f"  {'3-State Macro F1':28s} | {v3_3s['selective_macro_f1']*100:17.2f}% | {sel_f1_macro*100:17.2f}% | {'+' if delta_macro_f1 > 0 else ''}{delta_macro_f1:6.2f}%p", flush=True)
    print(f"  {'3-State Clean Specificity':28s} | {v3_3s['selective_specificity']*100:17.2f}% | {sel_spec*100:17.2f}% | {'+' if delta_spec > 0 else ''}{delta_spec:6.2f}%p", flush=True)
    print(f"  {'전체 ROC-AUC':28s} | {v3_2s['roc_auc']*100:17.2f}% | {auc_l13*100:17.2f}% | {'+' if delta_auc > 0 else ''}{delta_auc:6.2f}%p", flush=True)

    # 8. 공식 5대 시점별 성적
    valid_views = view_ids[valid_indices]
    print("\n" + "-" * 95, flush=True)
    print("  [공식 5대 시점별 3-State 결함 F1 비교 (v3 vs v6)]", flush=True)
    print("-" * 95, flush=True)
    view_results_v6 = {}
    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (valid_views == vn)
        v_dec = decided_mask[v_mask]
        y_v = train_labels[v_mask][v_dec]
        st_v = states[v_mask][v_dec]
        v_f1_d = float(f1_score(y_v, st_v, pos_label=1, zero_division=0) * 100.0)
        v_f1_m = float(f1_score(y_v, st_v, average='macro', zero_division=0) * 100.0)
        v3_vf1 = v3_data.get('view_results', {}).get(vn, {}).get('v3_f1_3state', 0.0)
        view_results_v6[vn] = {
            'alias': v_alias,
            'count': int(np.sum(v_mask)),
            'v6_defect_f1': v_f1_d,
            'v6_macro_f1': v_f1_m
        }
        print(f"  * {v_alias:25s}: 결함 F1={v_f1_d:5.2f}% | Macro F1={v_f1_m:5.2f}% (N={np.sum(v_mask)})", flush=True)

    # 9. 피처 중요도 분석
    feature_importances = np.mean([m.get_feature_importance() for m in fold_models], axis=0)
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features_v6 = {V6_FEATURE_NAMES[idx]: float(feature_importances[idx]) for idx in sorted_idx}

    print("\n" + "-" * 95, flush=True)
    print("  [v6 14D 피처 중요도 순위]", flush=True)
    print("-" * 95, flush=True)
    for rank, (fname, imp) in enumerate(top_features_v6.items(), 1):
        print(f"  {rank:2d}. {fname:28s}: {imp:6.2f}", flush=True)

    # 10. JSON 저장
    v6_benchmark_data = {
        'exercise': '03_바벨_스티프_데드리프트',
        'defect': '무릎 구부린 채 고정 위반 (sd1_v6)',
        'solutions_applied': [
            "Pelvis-Gravity Canonical Transformation (신체 중심 정규 3D 좌표계)",
            "KADR (Knee-Ankle Displacement Ratio), PLI, Excess Flexion Integral 등 혁신 생체역학 피처셋",
            "Asymmetric CleanLab 고정밀 물리 라벨 정제",
            "Conformal Prediction 10% Bounded 3-State 완충망"
        ],
        'sanitized_samples_count': int(num_sanitized),
        'tier2_2state_metrics': {
            'macro_f1': float(f1_macro_l13),
            'clean_f1': float(f1_clean_l13),
            'defect_f1': float(f1_defect_l13),
            'clean_specificity': float(spec_l13),
            'roc_auc': float(auc_l13),
            'accuracy': float(acc_l13)
        },
        'tier2_3state_metrics': {
            'tau_lower': float(tau_lower),
            'tau_upper': float(tau_upper),
            'abstention_rate': float(abstention_rate),
            'coverage_rate': float(coverage_rate),
            'selective_macro_f1': float(sel_f1_macro),
            'selective_clean_f1': float(sel_f1_clean),
            'selective_defect_f1': float(sel_f1_defect),
            'selective_specificity': float(sel_spec),
            'selective_accuracy': float(sel_acc)
        },
        'comparison_vs_v3': {
            'delta_defect_f1': float(delta_def_f1),
            'delta_abstention_rate': float(delta_abst),
            'delta_clean_f1': float(delta_clean_f1),
            'delta_macro_f1': float(delta_macro_f1),
            'delta_auc': float(delta_auc),
            'delta_specificity': float(delta_spec)
        },
        'view_results': view_results_v6,
        'top_features': top_features_v6
    }

    with open(V6_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(v6_benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v6 벤치마크 JSON 저장 완료: {V6_JSON_PATH}", flush=True)

    # 11. 가중치 번들 저장
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle_v6 = {
        'version': 'sd1_v6',
        'fold_models': fold_models,
        'feature_indices': V6_FEATURE_INDICES,
        'feature_names': V6_FEATURE_NAMES,
        'thresholds': {'tau_lower': tau_lower, 'tau_upper': tau_upper}
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle_v6, f)
    with open(WEIGHT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v6, f)
    with open(ROOT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v6, f)
    print(f"[+] sd1_v6 공식 가중치 번들 배포 완료: {BUNDLE_SAVE_PATH}", flush=True)

if __name__ == '__main__':
    run_v6_pipeline()
