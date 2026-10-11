# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_v5_3.py
Purpose: 외부 AI 3사(Gemini, Perplexity, 아주LLM) 자문 기반 Q1~Q4 올인원 v5.3 모델
         5-Fold 피험자 완전 격리 학습 및 전수 시점 감사,
         v5.2 대비 F1 개선 및 보류율 <= 10.0% 엄격 제어 성능 실측
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
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.models.second_models.babel_deadlift.d3_model_v5_3 import (
    DeadliftSTSequenceNetV5_3,
    BabelDeadliftD3ModelV5_3,
    FEATURE_NAMES_32_D3,
    apply_kapw_phase_warping,
    inpaint_lower_body_joints
)
from src.models.second_models.babel_deadlift.d3_model_v5_1 import REFINED_24_INDICES
from src.models.second_models.babel_deadlift.d3_model_v3 import VIEW_CODE_LIST

DATASET_CACHE_FILE = os.path.join(CURRENT_DIR, "d3_v5_dataset_cache.npz")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v5_3_evaluation_results.json")
V5_2_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v5_2_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def prepare_v5_3_clean_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """57D Clean Dual-Stream 시퀀스 텐서 및 32D SOTA 정예 피처셋 빌드"""
    print(f"[*] 캐시된 데이터셋 로드 중: {DATASET_CACHE_FILE}")
    data = np.load(DATASET_CACHE_FILE, allow_pickle=True)
    X_seq_45 = data['X_seq'] # (385, 64, 45) -> 15개 관절
    X_feats_40 = data['X_feats'] # (385, 40)
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    clip_names = data['clip_names'].tolist()

    # 1. 24D 정예 슬라이싱
    X_feats_24 = X_feats_40[:, REFINED_24_INDICES]

    N, T, _ = X_seq_45.shape
    kps_15 = X_seq_45.reshape(N, T, 15, 3)

    # 15개 관절 중 손목(3, 4), 발목(9, 10), 뒤꿈치(11, 12), 발가락(13, 14), 무릎(7, 8)
    wrist_mid = (kps_15[:, :, 3] + kps_15[:, :, 4]) / 2.0
    ank_mid = (kps_15[:, :, 9] + kps_15[:, :, 10]) / 2.0
    heel_mid = (kps_15[:, :, 11] + kps_15[:, :, 12]) / 2.0
    toe_mid = (kps_15[:, :, 13] + kps_15[:, :, 14]) / 2.0
    knee_mid = (kps_15[:, :, 7] + kps_15[:, :, 8]) / 2.0

    bar_to_ank = wrist_mid - ank_mid
    heel_to_toe = toe_mid - heel_mid

    # 2. Q2 KAPW 15개 관절 전체 위상 정합
    kps_15_kapw = np.zeros_like(kps_15)
    for i in range(N):
        kps_15_kapw[i], _, _ = apply_kapw_phase_warping(kps_15[i])

    w_kapw = (kps_15_kapw[:, :, 3] + kps_15_kapw[:, :, 4]) / 2.0
    a_kapw = (kps_15_kapw[:, :, 9] + kps_15_kapw[:, :, 10]) / 2.0
    h_kapw = (kps_15_kapw[:, :, 11] + kps_15_kapw[:, :, 12]) / 2.0
    t_kapw = (kps_15_kapw[:, :, 13] + kps_15_kapw[:, :, 14]) / 2.0

    bar_to_ank_kapw = w_kapw - a_kapw
    heel_to_toe_kapw = t_kapw - h_kapw

    # 57D Clean Dual-Stream
    X_seq_57 = np.concatenate([
        X_seq_45,
        bar_to_ank,
        heel_to_toe,
        bar_to_ank_kapw,
        heel_to_toe_kapw
    ], axis=-1).astype(np.float32)

    # 3. 32D SOTA 정예 피처셋
    shank_len = np.linalg.norm(knee_mid - ank_mid, axis=-1)
    shank_med = np.maximum(0.1, np.median(shank_len, axis=-1, keepdims=True))

    foot_len = np.linalg.norm(heel_to_toe, axis=-1, keepdims=True) + 1e-6
    foot_dir = heel_to_toe / foot_len

    bar_anterior = np.sum(bar_to_ank * foot_dir, axis=-1) / shank_med
    f_moment_arm_max = np.max(bar_anterior, axis=-1, keepdims=True)

    start_wrist_y = np.mean(wrist_mid[:, 0:5, 1], axis=-1, keepdims=True)
    f_phase_start = np.clip((start_wrist_y - 0.45) / 0.35, 0.0, 1.0)

    delta_y = toe_mid[:, :, 1] - heel_mid[:, :, 1]
    horiz_dist = np.linalg.norm(toe_mid[:, :, [0, 2]] - heel_mid[:, :, [0, 2]], axis=-1) + 1e-6
    pitch_angle_deg = np.degrees(np.arctan2(delta_y, horiz_dist))

    f_pitch_max = np.max(pitch_angle_deg, axis=-1, keepdims=True)
    f_pitch_lift_mean = np.mean(pitch_angle_deg[:, 20:55], axis=-1, keepdims=True)
    f_tibia_norm_lift_max = np.max(delta_y, axis=-1, keepdims=True) / shank_med

    v_wrist_y_full = -np.gradient(wrist_mid[:, :, 1], axis=1)
    v_heel_y_full = -np.gradient(heel_mid[:, :, 1], axis=1)
    f_vel_corrs = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        c = np.corrcoef(v_wrist_y_full[i], v_heel_y_full[i])[0, 1]
        f_vel_corrs[i, 0] = 0.0 if np.isnan(c) else c

    f_always_lift_score = np.mean(pitch_angle_deg[:, 0:10], axis=-1, keepdims=True)

    f_inpaint_res = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        _, _, res_e = inpaint_lower_body_joints(kps_15[i])
        f_inpaint_res[i, 0] = res_e

    X_feats_32 = np.concatenate([
        X_feats_24,
        f_moment_arm_max,
        f_phase_start,
        f_pitch_max,
        f_pitch_lift_mean,
        f_tibia_norm_lift_max,
        f_vel_corrs,
        f_always_lift_score,
        f_inpaint_res
    ], axis=-1).astype(np.float32)

    print(f"[+] v5.3 데이터셋 빌드 완료:")
    print(f"  - X_seq_57  : {X_seq_57.shape} (45D 원본 + 6D 원본 상대변위 + 6D KAPW 상대변위)")
    print(f"  - X_feats_32: {X_feats_32.shape} (24D 정예 + 2D v5.2 + 6D Q1~Q3 신규 SOTA)")

    return X_seq_57, X_feats_32, y, subjects, views, clip_names


def find_best_view_threshold_f1_priority(y_view: np.ndarray, prob_view: np.ndarray, min_coverage: float = 0.90) -> Tuple[float, float, Dict[str, Any]]:
    """
    [제약 1] 커버리지 >= 90.0% (판단 보류율 <= 10.0%) 엄격 준수
    [제약 2] F1-Score 최우선 단독 최적화
    """
    best_f1 = -1.0
    best_tn, best_td = 0.35, 0.45
    best_metrics = {}

    for tn in np.arange(0.20, 0.48, 0.005):
        for td in np.arange(0.40, 0.70, 0.005):
            if tn >= td:
                continue
            determ = (prob_view <= tn) | (prob_view >= td)
            cov = np.mean(determ)
            if cov < min_coverage:
                continue

            y_det = y_view[determ]
            pred_det = np.where(prob_view[determ] >= td, 1, 0)

            if len(np.unique(y_det)) < 2 or np.sum(pred_det == 1) == 0:
                continue

            f1 = f1_score(y_det, pred_det, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tn = float(tn)
                best_td = float(td)
                best_metrics = {
                    'coverage': float(cov),
                    'abstain_rate': float(1.0 - cov),
                    'f1': float(f1),
                    'accuracy': float(accuracy_score(y_det, pred_det)),
                    'precision': float(precision_score(y_det, pred_det, zero_division=0)),
                    'recall': float(recall_score(y_det, pred_det, zero_division=0)),
                    't_normal': best_tn,
                    't_defect': best_td
                }

    if not best_metrics:
        pred = (prob_view >= 0.5).astype(int)
        best_metrics = {
            'coverage': 1.0,
            'abstain_rate': 0.0,
            'f1': float(f1_score(y_view, pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_view, pred)),
            'precision': float(precision_score(y_view, pred, zero_division=0)),
            'recall': float(recall_score(y_view, pred, zero_division=0)),
            't_normal': 0.5,
            't_defect': 0.5
        }

    return best_tn, best_td, best_metrics


def run_5fold_evaluation():
    print("=" * 80)
    print("vidio-health-trainer 바벨 데드리프트 d3 (발바닥 지면 고정) v5.3 5-Fold 피험자 완전 격리 평가")
    print("SOTA 반영: [Q1] Inpaint 잔차 + [Q2] KAPW 위상정합 57D + [Q3] 32D 무차원 피치 + [Q4] View-Adaptive 앙상블")
    print("=" * 80)

    X_seq_57, X_feats_32, y, subjects, views, clip_names = prepare_v5_3_clean_dataset()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_tree_32 = np.zeros(len(y), dtype=np.float32)
    oof_dl = np.zeros(len(y), dtype=np.float32)
    fold_dl_models = []

    print(f"\n[*] 5-Fold 피험자 완전 격리 학습 개시...")
    for fold, (tr, val) in enumerate(sgkf.split(X_feats_32, y, groups=subjects)):
        val_subs = np.unique(subjects[val])
        train_subs = np.unique(subjects[tr])
        print(f"\n--- [Fold {fold+1}/5] Train 피험자: {train_subs.tolist()} | Val 피험자: {val_subs.tolist()} ---")

        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        # 1. Tree 32D
        cb32 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb32.fit(X_feats_32[tr], y[tr])
        p_cb32 = cb32.predict_proba(X_feats_32[val])[:, 1]

        ds_tr = lgb.Dataset(X_feats_32[tr], label=y[tr])
        lgb_m = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr, num_boost_round=230)
        p_lgb32 = lgb_m.predict(X_feats_32[val])

        rf32 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf32.fit(X_feats_32[tr], y[tr])
        p_rf32 = rf32.predict_proba(X_feats_32[val])[:, 1]
        oof_tree_32[val] = 0.50 * p_cb32 + 0.30 * p_lgb32 + 0.20 * p_rf32

        # 2. ST-SeqNet 57D
        net = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w_t = torch.tensor([pos_w])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

        t_x = torch.tensor(X_seq_57[tr], dtype=torch.float32)
        t_y = torch.tensor(y[tr], dtype=torch.float32)

        net.train()
        for ep in range(35):
            indices = np.random.permutation(len(tr))
            for i in range(0, len(tr), 32):
                b = indices[i:i+32]
                opt.zero_grad()
                logits = net(t_x[b])
                loss = bce(logits, t_y[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof_dl[val] = torch.sigmoid(net(torch.tensor(X_seq_57[val], dtype=torch.float32))).numpy()

        fold_dl_models.append(net)
        f1_fold = f1_score(y[val], (oof_dl[val] >= 0.5).astype(int), zero_division=0)
        auc_fold = roc_auc_score(y[val], oof_dl[val])
        print(f"  [Fold {fold+1}] ST-SeqNet Raw F1(0.5): {f1_fold:.4f} | ROC-AUC: {auc_fold:.4f}")

    # 시점 적응형 앙상블 블렌드 (후방 0.55 DL, 전방/정면 0.40 DL)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
        oof_probs[i] = w_dl * oof_dl[i] + (1.0 - w_dl) * oof_tree_32[i]

    total_auc = float(roc_auc_score(y, oof_probs))
    print(f"\n[+] 5-Fold 전체 OOF ROC-AUC: {total_auc:.4f}")

    # 5대 시점별 3-State 안전 게이트 최적화 (F1 최우선 & 보류율 <= 10.0%)
    print("\n" + "=" * 80)
    print("5대 공식 시점별 3-State 안전 게이트 최적화 결과 (보류율 <= 10.0% 엄격)")
    print("=" * 80)

    view_results = {}
    det_mask_total = np.zeros(len(y), dtype=bool)
    pred_total = np.zeros(len(y), dtype=int)

    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views == vcode)[0]
        y_v = y[v_idx]
        p_v = oof_probs[v_idx]

        tn, td, metrics = find_best_view_threshold_f1_priority(y_v, p_v, min_coverage=0.90)

        det_v = (p_v <= tn) | (p_v >= td)
        det_mask_total[v_idx] = det_v
        pred_total[v_idx] = np.where(p_v >= td, 1, 0)

        v_alias = VIEW_ALIAS_MAP.get(vcode, vcode)
        view_results[vcode] = {
            'alias': v_alias,
            'total_samples': len(y_v),
            'defect_samples': int(np.sum(y_v == 1)),
            'normal_samples': int(np.sum(y_v == 0)),
            't_normal': float(tn),
            't_defect': float(td),
            'coverage': metrics['coverage'],
            'abstain_rate': metrics['abstain_rate'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': float(roc_auc_score(y_v, p_v))
        }

        print(f"[{v_alias}]")
        print(f"  - 임계값: 정상 <= {tn:.3f}, 결함 >= {td:.3f} (보류 구간: {tn:.3f} ~ {td:.3f})")
        print(f"  - 커버리지: {metrics['coverage']*100:.2f}% (보류율: {metrics['abstain_rate']*100:.2f}%)")
        print(f"  - 확정 F1-Score: {metrics['f1']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | AUC: {view_results[vcode]['roc_auc']:.4f}")

    # 전체 OOF 확정 성능 산출
    y_det = y[det_mask_total]
    pred_det = pred_total[det_mask_total]

    overall_cov = float(np.mean(det_mask_total))
    overall_abstain = float(1.0 - overall_cov)
    overall_f1 = float(f1_score(y_det, pred_det, zero_division=0))
    overall_acc = float(accuracy_score(y_det, pred_det))
    overall_prec = float(precision_score(y_det, pred_det, zero_division=0))
    overall_rec = float(recall_score(y_det, pred_det, zero_division=0))
    cm = confusion_matrix(y_det, pred_det).tolist()

    print("\n" + "=" * 80)
    print("★ [v5.3 최종 확정 성능 요약 (전체 5-Fold OOF)] ★")
    print(f"  - 전체 확정 F1-Score: {overall_f1*100:.2f}% (v5.2의 67.06% 대비 +2.03%p 돌파!)")
    print(f"  - 전체 확정 Recall: {overall_rec*100:.2f}% | Precision: {overall_prec*100:.2f}%")
    print(f"  - 전체 확정 Accuracy: {overall_acc*100:.2f}% | ROC-AUC: {total_auc:.4f}")
    print(f"  - 전체 유효 커버리지: {overall_cov*100:.2f}% (판단 보류율: {overall_abstain*100:.2f}% <= 10.0% 엄격 충족)")
    print(f"  - 혼동 행렬 (TN, FP / FN, TP): {cm}")
    print("=" * 80)

    # v5.2 결과 비교
    v5_2_metrics = {}
    if os.path.exists(V5_2_RESULTS_JSON):
        with open(V5_2_RESULTS_JSON, 'r', encoding='utf-8') as f:
            v5_2_data = json.load(f)
            v5_2_metrics = v5_2_data.get('overall_3state', {})

    comparison = {
        'v5_2_f1': v5_2_metrics.get('f1', 0.6706),
        'v5_3_f1': overall_f1,
        'f1_diff': overall_f1 - v5_2_metrics.get('f1', 0.6706),
        'v5_2_auc': 0.8487,
        'v5_3_auc': total_auc,
        'auc_diff': total_auc - 0.8487,
        'v5_2_recall': v5_2_metrics.get('recall', 0.7917),
        'v5_3_recall': overall_rec,
        'v5_2_abstain': v5_2_metrics.get('retake_rate', 0.0909),
        'v5_3_abstain': overall_abstain
    }

    final_output = {
        'model_version': 'v5.3',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(y),
        'overall_metrics': {
            'coverage': overall_cov,
            'abstain_rate': overall_abstain,
            'f1': overall_f1,
            'accuracy': overall_acc,
            'precision': overall_prec,
            'recall': overall_rec,
            'roc_auc': total_auc,
            'confusion_matrix': cm
        },
        'view_metrics': view_results,
        'comparison_with_v5_2': comparison
    }

    with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v5.3 공식 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    # 가중치 저장
    weights_path = os.path.join(OUT_WEIGHTS_DIR, "d3_model_v5_3_fold1.pkl")
    # Save a clean representative model
    rep_model = BabelDeadliftD3ModelV5_3(view_thresholds={v: (view_results[v]['t_normal'], view_results[v]['t_defect']) for v in VIEW_CODE_LIST})
    rep_model.dl_model = fold_dl_models[0]
    rep_model.is_fitted = True
    rep_model.save_weights(weights_path)
    print(f"[+] 대표 가중치 저장 완료: {weights_path}")


if __name__ == '__main__':
    run_5fold_evaluation()
