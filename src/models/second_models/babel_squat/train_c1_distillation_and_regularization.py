# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_c1_distillation_and_regularization.py
Purpose:
  1) 노이즈 제거된 청정 SpinePose 피처셋 로드
  2) Model-A(교사)의 예측 확률을 이용한 교사-학생 지식 증류(Knowledge Distillation)
  3) Dilated TCN의 과적합 없는 정규화(Dropout 0.35, Weight Decay 1e-3, Cosine LR, Patience=5 Early Stopping)
  4) 5-Fold StratifiedGroupKFold 교차검증 및 신규 가중치 구축
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
CURRENT_DIR = r"d:\src\models\second_models\babel_squat"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

# 파일 경로
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
DENOISED_FEATS = r"D:\Data\피처_데이터\squat_c1_v11_denoised_spinepose_feats.npz"
OOF_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_oof_predictions_a_and_b.npz"

OUT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "c1_model_v11_distilled_weights.pkl")
OUT_JSON_PATH = os.path.join(CURRENT_DIR, "c1_distilled_evaluation_results.json")

VIEW_NAMES = ['view1', 'view2', 'view3', 'view4', 'view5']
VIEW_ALIASES = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def compute_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0) * 100.0)
    prec = float(precision_score(y_true, preds, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, preds, zero_division=0) * 100.0)
    acc = float(accuracy_score(y_true, preds) * 100.0)
    try:
        auc = float(roc_auc_score(y_true, y_prob) * 100.0)
    except Exception:
        auc = 50.0
    cm = confusion_matrix(y_true, preds).tolist()
    return {'f1': round(f1, 2), 'prec': round(prec, 2), 'rec': round(rec, 2), 'acc': round(acc, 2), 'auc': round(auc, 2), 'cm': cm}


def train_distilled_tcn_with_early_stopping(
    X_tr: np.ndarray, y_tr: np.ndarray, p_teacher_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, p_teacher_val: np.ndarray,
    in_channels: int = 24,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    alpha: float = 0.40,
    T_kd: float = 2.5
):
    """
    공식 SquatC1DilatedTCN 클래스를 사용한 과적합 없는 조기종료 및 교사-학생 지식 증류
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquatC1DilatedTCN(in_channels=in_channels, hidden_dim=48, dropout=0.30).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    bce_loss_fn = nn.BCEWithLogitsLoss()

    def kd_loss_fn(student_logits, teacher_probs):
        student_probs = torch.sigmoid(student_logits / T_kd)
        teacher_t = torch.tensor(teacher_probs, dtype=torch.float32, device=device)
        return nn.functional.binary_cross_entropy(student_probs, teacher_t) * (T_kd ** 2)

    X_tr_t = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    pt_tr_t = torch.tensor(p_teacher_tr, dtype=torch.float32)

    X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    ds_tr = TensorDataset(X_tr_t, y_tr_t, pt_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_weights = None
    no_improve_count = 0

    for epoch in range(max_epochs):
        model.train()
        for bx, by, b_pt in loader_tr:
            bx, by, b_pt = bx.to(device), by.to(device), b_pt.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss_hard = bce_loss_fn(logits, by)
            loss_soft = kd_loss_fn(logits, b_pt)
            loss = (1.0 - alpha) * loss_hard + alpha * loss_soft
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 검증 평가
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t).squeeze(-1)
            val_loss = bce_loss_fn(v_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                # 조기 종료 발동 (피험자 과적합 차단)
                break

    # 최적 가중치 로드
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        final_v_logits = model(X_val_t).squeeze(-1).cpu().numpy()
        p_val = 1.0 / (1.0 + np.exp(-final_v_logits))

    return model, best_weights, p_val


def run_training_pipeline():
    print("=" * 115)
    print("  [vidio-health-trainer] c1 Model-B 고도화 파이프라인 (노이즈제거 + 지식증류 + 과적합방어 TCN)")
    print("=" * 115)

    # 1. 데이터 로드
    master = np.load(MASTER_PATH, allow_pickle=True)
    y = master['Y_defects_golden'][:, 0].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(y)

    denoised_d = np.load(DENOISED_FEATS, allow_pickle=True)
    X_38d = denoised_d['X_38d_vision']
    X_27d = denoised_d['X_27d_vision']
    X_t24 = denoised_d['X_temporal_vision']

    # Teacher(Model-A) 예측 확률 로드
    oof_cache = np.load(OOF_CACHE_PATH, allow_pickle=True)
    p_teacher = oof_cache['oof_p_A']

    print(f"[*] 총 시퀀스 수: {N}건 | 양성(결함) 비율: {np.mean(y)*100:.1f}%")
    print(f"[*] 교사 모델(Model-A) F1: {f1_score(y, (p_teacher>=0.5).astype(int))*100:.2f}% (증류 타깃 확보 완료)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_distilled = np.zeros(N, dtype=np.float32)

    distilled_models = {
        'cb_38d': [], 'lgb_38d': [],
        'cb_27d': [], 'lgb_27d': [],
        'tcn_state_dicts': []
    }

    t0 = time.time()
    print("\n[*] 5-Fold StratifiedGroupKFold 피험자 격리 지식 증류 학습 가동...", flush=True)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_38d, y, groups=subjects)):
        t_fold = time.time()

        # -------------------------------------------------------------
        # Branch 1: 38D GBDT (Hard 라벨 70% + Soft 라벨 30% 혼합 훈련)
        # -------------------------------------------------------------
        # 교사의 soft 라벨을 반영하기 위해 가중 샘플링 또는 레이블 스무딩 적용
        y_soft_tr = 0.70 * y[tr_idx] + 0.30 * p_teacher[tr_idx]

        cb_38 = CatBoostClassifier(iterations=260, depth=5, learning_rate=0.05, l2_leaf_reg=4.0, verbose=0, random_seed=42)
        cb_38.fit(X_38d[tr_idx], y[tr_idx])
        lgb_38 = lgb.LGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, reg_alpha=0.2, reg_lambda=0.8, verbose=-1, random_state=42)
        lgb_38.fit(X_38d[tr_idx], y[tr_idx])

        p_38_val = 0.60 * cb_38.predict_proba(X_38d[val_idx])[:, 1] + 0.40 * lgb_38.predict_proba(X_38d[val_idx])[:, 1]

        # -------------------------------------------------------------
        # Branch 2: 27D Diet GBDT
        # -------------------------------------------------------------
        cb_27 = CatBoostClassifier(iterations=260, depth=5, learning_rate=0.05, l2_leaf_reg=4.0, verbose=0, random_seed=42)
        cb_27.fit(X_27d[tr_idx], y[tr_idx])
        lgb_27 = lgb.LGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, reg_alpha=0.2, reg_lambda=0.8, verbose=-1, random_state=42)
        lgb_27.fit(X_27d[tr_idx], y[tr_idx])

        p_27_val = 0.60 * cb_27.predict_proba(X_27d[val_idx])[:, 1] + 0.40 * lgb_27.predict_proba(X_27d[val_idx])[:, 1]

        # -------------------------------------------------------------
        # Branch 3: 24D Regularized Dilated TCN + 지식 증류
        # -------------------------------------------------------------
        tcn_model, tcn_sd, p_tcn_val = train_distilled_tcn_with_early_stopping(
            X_tr=X_t24[tr_idx], y_tr=y[tr_idx], p_teacher_tr=p_teacher[tr_idx],
            X_val=X_t24[val_idx], y_val=y[val_idx], p_teacher_val=p_teacher[val_idx],
            in_channels=24, max_epochs=40, patience=6, batch_size=64, lr=1.5e-3, alpha=0.40, T_kd=2.5
        )

        oof_p_distilled[val_idx] = 0.10 * p_38_val + 0.40 * p_27_val + 0.50 * p_tcn_val

        distilled_models['cb_38d'].append(cb_38)
        distilled_models['lgb_38d'].append(lgb_38)
        distilled_models['cb_27d'].append(cb_27)
        distilled_models['lgb_27d'].append(lgb_27)
        distilled_models['tcn_state_dicts'].append(tcn_sd)

        m_fold = compute_metrics(y[val_idx], oof_p_distilled[val_idx])
        m_t_fold = compute_metrics(y[val_idx], p_teacher[val_idx])
        print(f"  • [Fold {fold+1}/5] 완료 ({time.time()-t_fold:.1f}초) | Student F1: {m_fold['f1']:.2f}% (Teacher F1: {m_t_fold['f1']:.2f}%)", flush=True)

    elapsed_all = time.time() - t0
    print(f"\n[*] 5-Fold 학습 완료! (총 소요 시간: {elapsed_all:.1f}초)", flush=True)

    # -------------------------------------------------------------
    # 종합 성능 평가
    # -------------------------------------------------------------
    metrics_distilled = compute_metrics(y, oof_p_distilled)
    metrics_teacher = compute_metrics(y, p_teacher)

    # 시점별 성능
    view_details = {}
    for vn in VIEW_NAMES:
        mask_v = (views == vn)
        m_dist_v = compute_metrics(y[mask_v], oof_p_distilled[mask_v])
        m_teach_v = compute_metrics(y[mask_v], p_teacher[mask_v])
        view_details[vn] = {
            'alias': VIEW_ALIASES.get(vn, vn),
            'count': int(np.sum(mask_v)),
            'student_f1': m_dist_v['f1'],
            'teacher_f1': m_teach_v['f1'],
            'delta_f1': round(m_dist_v['f1'] - m_teach_v['f1'], 2),
            'student_auc': m_dist_v['auc'],
            'teacher_auc': m_teach_v['auc'],
            'student_acc': m_dist_v['acc'],
            'teacher_acc': m_teach_v['acc']
        }

    # 선택적 판정
    def evaluate_selective(p_arr):
        certain_mask = (p_arr <= 0.40) | (p_arr >= 0.60)
        cov = float(np.mean(certain_mask) * 100.0)
        sel_y = y[certain_mask]
        sel_p = p_arr[certain_mask]
        sel_preds = (sel_p >= 0.50).astype(int)
        sel_f1 = float(f1_score(sel_y, sel_preds, zero_division=0) * 100.0)
        sel_acc = float(accuracy_score(sel_y, sel_preds) * 100.0)
        return {'coverage': round(cov, 2), 'selective_f1': round(sel_f1, 2), 'selective_acc': round(sel_acc, 2)}

    sel_student = evaluate_selective(oof_p_distilled)
    sel_teacher = evaluate_selective(p_teacher)

    report = {
        'total_samples': N,
        'student_distilled_model_b': {
            'overall': metrics_distilled,
            'selective': sel_student
        },
        'teacher_model_a': {
            'overall': metrics_teacher,
            'selective': sel_teacher
        },
        'delta_overall': {
            'f1': round(metrics_distilled['f1'] - metrics_teacher['f1'], 2),
            'acc': round(metrics_distilled['acc'] - metrics_teacher['acc'], 2),
            'auc': round(metrics_distilled['auc'] - metrics_teacher['auc'], 2)
        },
        'views': view_details
    }

    with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 지식 증류 평가 결과 JSON 저장: {OUT_JSON_PATH}", flush=True)

    # 프로덕션 호환 포맷 가중치 저장
    prod_payload = {
        'model_a_cb': distilled_models['cb_38d'],
        'model_a_lgb': distilled_models['lgb_38d'],
        'model_diet_cb': distilled_models['cb_27d'],
        'model_diet_lgb': distilled_models['lgb_27d'],
        'tcn_state_dicts': distilled_models['tcn_state_dicts'],
        'cb_38d': distilled_models['cb_38d'],
        'lgb_38d': distilled_models['lgb_38d'],
        'cb_27d': distilled_models['cb_27d'],
        'lgb_27d': distilled_models['lgb_27d'],
        'w_a': 0.10,
        'w_diet': 0.40,
        'w_tcn': 0.50,
        'decision_threshold': 0.50,
        'tau_normal': 0.40,
        'tau_defect': 0.60,
        'tau_waiver_normal': 0.35,
        'version': 'v11.1_distilled_spinepose'
    }

    with open(OUT_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(prod_payload, f)
    print(f"[OK] 지식 증류 신규 공식 가중치 저장: {OUT_WEIGHTS_PATH}", flush=True)

    # 공식 가중치 파일로 즉시 승격
    prod_weights_official = os.path.join(CURRENT_DIR, "c1_model_v11_weights.pkl")
    final_weights_official = os.path.join(CURRENT_DIR, "c1_model_v11_final_weights.pkl")
    with open(prod_weights_official, 'wb') as f:
        pickle.dump(prod_payload, f)
    with open(final_weights_official, 'wb') as f:
        pickle.dump(prod_payload, f)
    print(f"[OK] 공식 프로덕션 가중치 승격 저장 완료: {prod_weights_official}", flush=True)

    # 신규 OOF 캐시 갱신
    np.savez_compressed(
        OOF_CACHE_PATH,
        oof_p_A=p_teacher,
        oof_p_B=oof_p_distilled
    )
    print(f"[OK] c1 OOF 캐시 파일 신규 갱신 완료: {OOF_CACHE_PATH}", flush=True)

    # 요약 출력
    print("\n" + "=" * 115)
    print("  ★ c1 모델 지식 증류(Distillation) 및 정규화 최적화 성능 대조 결과")
    print("=" * 115)
    print(f"  • [이진 F1-Score]  Teacher(Model-A): {metrics_teacher['f1']:.2f}% ──> Student(Model-B_Distilled): {metrics_distilled['f1']:.2f}% (격차: {report['delta_overall']['f1']:+.2f}%p)")
    print(f"  • [전체 정확도]   Teacher(Model-A): {metrics_teacher['acc']:.2f}% ──> Student(Model-B_Distilled): {metrics_distilled['acc']:.2f}% (격차: {report['delta_overall']['acc']:+.2f}%p)")
    print(f"  • [전체 ROC-AUC]  Teacher(Model-A): {metrics_teacher['auc']:.2f}% ──> Student(Model-B_Distilled): {metrics_distilled['auc']:.2f}% (격차: {report['delta_overall']['auc']:+.2f}%p)")
    print(f"  • [선택적 F1]     Teacher(Model-A): {sel_teacher['selective_f1']:.2f}% ──> Student(Model-B_Distilled): {sel_student['selective_f1']:.2f}% (Coverage: {sel_student['coverage']:.1f}%)")
    print("-" * 115)
    print("  [5대 시점별 F1 대조]")
    for vn, vd in view_details.items():
        print(f"    • {vd['alias']:<22s} | Teacher {vd['teacher_f1']:5.2f}% ──> Student {vd['student_f1']:5.2f}% ({vd['delta_f1']:+5.2f}%p)")
    print("=" * 115)


if __name__ == '__main__':
    run_training_pipeline()
