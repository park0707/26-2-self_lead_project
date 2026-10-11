# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: eval_squat_clean_classification_a_vs_b.py
Purpose:
  1. Model-B(SpinePose 순수 비전)를 c1 공식 프로덕션 모델로 승격 저장
  2. 바벨 스쿼트 "정상/오류 분류 모델 (Clean Squat vs Defective Squat)" 성능 평가
  3. 이전 Model-A(AI-Hub JSON 의존) 적용 시 vs Model-B 적용 시 전수 비교 분석
     - 4대 결함 독립 결합 판정
     - 16-way MoE 통합 마스터 엔진(v11.0) 판정
     - 5대 공식 시점(시점 A~시점 E) 세부 지표
"""

import os
import sys
import json
import pickle
import shutil
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
CURRENT_DIR = r"d:\src\models\second_models\babel_squat"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from joint_meta_16way_moe_net import JointMeta16WayMoENetV10_1
from joint_meta_16way_net import decode_hybrid

# -------------------------------------------------------------
# 파일 경로 정의
# -------------------------------------------------------------
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
META_PATH   = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH     = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH= r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH= r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")
SCALER_PATH_V11 = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
WEIGHTS_V11     = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")
OOF_RES_PATH     = os.path.join(CURRENT_DIR, "stgcn_transformer_oof_residuals.npz")

# Model-A 피처 & 가중치
A_FEATS_38D = r"D:\Data\피처_데이터\squat_c1_v11_a_38d_feats.npz"
A_FEATS_27D = r"D:\Data\피처_데이터\squat_c1_v11_a_diet_27d_feats.npz"
A_TENSOR_24D = r"D:\Data\피처_데이터\squat_c1_temporal_24d_64f.npz"
A_WEIGHTS_38 = os.path.join(CURRENT_DIR, "c1_model_v11_a_weights.pkl")
A_WEIGHTS_27 = os.path.join(CURRENT_DIR, "c1_model_v11_a_diet_weights.pkl")
A_WEIGHTS_TCN = os.path.join(CURRENT_DIR, "c1_model_v11_b_weights.pkl")

# Model-B 피처 & 가중치
B_FEATS_ALL = r"D:\Data\피처_데이터\squat_c1_v11_vision_spinepose_feats.npz"
B_WEIGHTS_SRC = os.path.join(CURRENT_DIR, "c1_model_v11_spinepose_vision_weights.pkl")

# 공식 승격 대상 파일
PROD_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "c1_model_v11_weights.pkl")
FINAL_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "c1_model_v11_final_weights.pkl")
BACKUP_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "c1_model_v11_weights_backup_aihub.pkl")

OUT_JSON = os.path.join(CURRENT_DIR, "squat_clean_classification_a_vs_b_results.json")

VIEW_NAMES = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_ALIASES = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def promote_model_b_weights():
    """Model-B 가중치를 프로덕션 c1 표준 가중치로 승격 및 백업 저장"""
    print("[1] Model-B(SpinePose 순수 비전) 공식 c1 가중치 승격 작업 시작...")
    if not os.path.exists(B_WEIGHTS_SRC):
        raise FileNotFoundError(f"Model-B 가중치 원본이 없습니다: {B_WEIGHTS_SRC}")

    with open(B_WEIGHTS_SRC, 'rb') as f:
        b_data = pickle.load(f)

    # 기존 가중치 백업 (아직 백업 파일이 없는 경우)
    if os.path.exists(PROD_WEIGHTS_PATH) and not os.path.exists(BACKUP_WEIGHTS_PATH):
        shutil.copy2(PROD_WEIGHTS_PATH, BACKUP_WEIGHTS_PATH)
        print(f"  [+] 기존 Model-A 가중치 안전 백업 완료: {BACKUP_WEIGHTS_PATH}")

    # 프로덕션 호환 포맷 생성
    prod_payload = {
        'model_a_cb': b_data['cb_38d'],
        'model_a_lgb': b_data['lgb_38d'],
        'model_diet_cb': b_data['cb_27d'],
        'model_diet_lgb': b_data['lgb_27d'],
        'tcn_state_dicts': b_data['tcn_state_dicts'],
        # 자체 키도 함께 포함하여 양방향 호환 유지
        'cb_38d': b_data['cb_38d'],
        'lgb_38d': b_data['lgb_38d'],
        'cb_27d': b_data['cb_27d'],
        'lgb_27d': b_data['lgb_27d'],
        'w_a': 0.10,
        'w_diet': 0.40,
        'w_tcn': 0.50,
        'decision_threshold': 0.50,
        'tau_normal': 0.40,
        'tau_defect': 0.60,
        'tau_waiver_normal': 0.35,
        'version': 'v11.1_spinepose_vision'
    }

    with open(PROD_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(prod_payload, f)
    print(f"  [+] 공식 프로덕션 가중치 승격 저장 완료: {PROD_WEIGHTS_PATH}")

    with open(FINAL_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(prod_payload, f)
    print(f"  [+] 공식 최종 가중치 승격 저장 완료: {FINAL_WEIGHTS_PATH}")

    # 검증: c1_model_v11_1 로드 테스트
    from c1_model_v11_1 import BabelSquatC1ModelV11_1
    test_m = BabelSquatC1ModelV11_1(weights_path=PROD_WEIGHTS_PATH)
    assert test_m.is_fitted, "가중치 로드 검증 실패!"
    assert len(test_m.tcn_models) == 5, f"TCN 폴드 개수 불일치: {len(test_m.tcn_models)}"
    print("  [+] BabelSquatC1ModelV11_1 로드 및 무결성 검증 100% 통과!\n")


def train_tcn_fast(X_tr, y_tr, X_val, in_channels=24, epochs=15, batch_size=64, lr=1.5e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquatC1DilatedTCN(in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32)

    ds_tr = TensorDataset(X_tr_t, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        v_logits = model(X_val_t.to(device)).squeeze(-1).cpu().numpy()
        p_val = 1.0 / (1.0 + np.exp(-v_logits))
    return p_val


def get_oof_predictions(master_data, a_38, a_27, a_t24, b_38, b_27, b_t24):
    """5-Fold StratifiedGroupKFold로 Model-A 및 Model-B의 c1 OOF 확률 산출"""
    print("[2] c1 Model-A vs Model-B OOF 예측 확률 산출 (5-Fold StratifiedGroupKFold)...")
    y_c1 = master_data['Y_defects_golden'][:, 0].astype(int)
    subjects = master_data['subjects']
    N = len(y_c1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p_A = np.zeros(N, dtype=np.float32)
    oof_p_B = np.zeros(N, dtype=np.float32)

    with open(A_WEIGHTS_38, 'rb') as f:
        w_a_38 = pickle.load(f)
    with open(A_WEIGHTS_27, 'rb') as f:
        w_a_27 = pickle.load(f)
    with open(A_WEIGHTS_TCN, 'rb') as f:
        w_a_tcn = pickle.load(f)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(np.zeros(N), y_c1, groups=subjects)):
        # Model-A 추론
        cb_a_38 = w_a_38['cb_models'][fold]
        lgb_a_38 = w_a_38['lgb_models'][fold]
        p_a_38 = 0.60 * cb_a_38.predict_proba(a_38[val_idx])[:, 1] + 0.40 * lgb_a_38.predict_proba(a_38[val_idx])[:, 1]

        cb_a_27 = w_a_27['cb_models'][fold]
        lgb_a_27 = w_a_27['lgb_models'][fold]
        p_a_27 = 0.60 * cb_a_27.predict_proba(a_27[val_idx])[:, 1] + 0.40 * lgb_a_27.predict_proba(a_27[val_idx])[:, 1]

        tcn_a = SquatC1DilatedTCN(in_channels=24)
        tcn_a.load_state_dict(w_a_tcn['tcn_state_dicts'][fold])
        tcn_a.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(a_t24[val_idx].transpose(0, 2, 1), dtype=torch.float32)
            logits_a = tcn_a(x_val_t).squeeze(-1).numpy()
            p_a_tcn = 1.0 / (1.0 + np.exp(-logits_a))

        oof_p_A[val_idx] = 0.10 * p_a_38 + 0.40 * p_a_27 + 0.50 * p_a_tcn

        # Model-B 학습 및 추론
        cb_b_38 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_b_38.fit(b_38[tr_idx], y_c1[tr_idx])
        lgb_b_38 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_b_38.fit(b_38[tr_idx], y_c1[tr_idx])
        p_b_38 = 0.60 * cb_b_38.predict_proba(b_38[val_idx])[:, 1] + 0.40 * lgb_b_38.predict_proba(b_38[val_idx])[:, 1]

        cb_b_27 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_b_27.fit(b_27[tr_idx], y_c1[tr_idx])
        lgb_b_27 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_b_27.fit(b_27[tr_idx], y_c1[tr_idx])
        p_b_27 = 0.60 * cb_b_27.predict_proba(b_27[val_idx])[:, 1] + 0.40 * lgb_b_27.predict_proba(b_27[val_idx])[:, 1]

        p_b_tcn = train_tcn_fast(b_t24[tr_idx], y_c1[tr_idx], b_t24[val_idx], in_channels=24, epochs=15, batch_size=64)

        oof_p_B[val_idx] = 0.10 * p_b_38 + 0.40 * p_b_27 + 0.50 * p_b_tcn
        print(f"  • Fold {fold+1}/5 완료: Model-A F1={f1_score(y_c1[val_idx], (oof_p_A[val_idx]>=0.5).astype(int))*100:.2f}% | Model-B F1={f1_score(y_c1[val_idx], (oof_p_B[val_idx]>=0.5).astype(int))*100:.2f}%")

    print("[+] OOF 확률 생성 완료!\n")
    return oof_p_A, oof_p_B


def calc_classification_metrics(y_true, y_pred, y_prob=None):
    """정상/오류 분류 평가 지표 산출"""
    # y_true == 1: 정상 (Clean), 0: 오류 (Defect)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # tn: 오류를 오류로, fp: 오류를 정상으로(거짓 정상 False Clean), fn: 정상을 오류로, tp: 정상을 정상으로
    
    clean_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0)
    clean_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0)
    clean_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0)
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    
    defect_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0)
    defect_prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0)
    defect_rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0)

    auc = 50.0
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob) * 100.0)
        except Exception:
            auc = 50.0

    return {
        'clean_f1': round(clean_f1, 2),
        'clean_prec': round(clean_prec, 2),
        'clean_rec': round(clean_rec, 2),
        'defect_f1': round(defect_f1, 2),
        'defect_prec': round(defect_prec, 2),
        'defect_rec': round(defect_rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2),
        'false_clean_fp': int(fp),  # 치명적 오진 (오류인데 정상으로 오판)
        'false_defect_fn': int(fn),
        'clean_tp': int(tp),
        'defect_tn': int(tn),
        'total': len(y_true)
    }


def evaluate_moe_v11(probs_defect_in, y_all, true_clean, views, c4_data, q_data, k_data, oof_res):
    """v11.0 MoE 통합 마스터 엔진 공식 파이프라인 추론 및 메트릭 평가"""
    eps = 1e-7
    with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
        t_cfg = json.load(f)
    T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
    view_T_c2 = t_cfg['c2']['view_T']

    p_clip = np.clip(probs_defect_in, eps, 1.0 - eps)
    logits_raw = np.log(p_clip / (1.0 - p_clip))
    p_cal = np.zeros_like(p_clip)
    p_cal[:, 0] = 1.0 / (1.0 + np.exp(-logits_raw[:, 0] / T_c1))
    p_cal[:, 2] = 1.0 / (1.0 + np.exp(-logits_raw[:, 2] / T_c3))
    p_cal[:, 3] = 1.0 / (1.0 + np.exp(-logits_raw[:, 3] / T_c4))
    for v, T_v in view_T_c2.items():
        m = (views == v)
        p_cal[m, 1] = 1.0 / (1.0 + np.exp(-logits_raw[m, 1] / T_v))

    margins_cal = np.abs(p_cal - 0.5)

    X_c4 = c4_data['X_c4_v5_25d']
    wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
    jerk_norm   = np.clip(jerk   / 200.0, 0.0, 2.5)[:, None]
    wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)[:, None]
    ydelta_norm = np.clip(X_c4[:, 6] / 0.020, -1.0, 2.5)[:, None]
    toe_norm    = np.clip(X_c4[:, 1] / 0.020,  0.0, 2.5)[:, None]
    f0_norm     = np.clip(f0_rate / 0.50,       0.0, 2.0)[:, None]
    phys_feats  = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])
    view_map     = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
    view_onehot  = np.eye(5, dtype=np.float32)[[view_map[v] for v in views]]

    X_quality_raw = q_data['X_quality_12d']
    q_bone     = X_quality_raw[:, 0:1]; q_jitter = X_quality_raw[:, 1:2]; q_overall_f = X_quality_raw[:, 2:3]
    bone_instab= np.clip(X_quality_raw[:, 3:4]   / 0.20,          0.0, 2.5)
    jitter_s   = np.clip(X_quality_raw[:, 4:5]   / 0.001,          0.0, 2.5)
    nadir_s    = X_quality_raw[:, 5:6]
    rom_s      = np.clip(X_quality_raw[:, 6:7]   / 0.015,          0.0, 2.5)
    sh_w_s     = np.clip((X_quality_raw[:, 7:8]  - 0.65) / 0.15,  -2.5, 2.5)
    hip_w_s    = np.clip((X_quality_raw[:, 8:9]  - 0.42) / 0.10,  -2.5, 2.5)
    z_knee_s   = np.clip(X_quality_raw[:, 9:10]  / 0.40,           0.0, 2.5)
    z_ank_s    = np.clip(X_quality_raw[:, 10:11] / 0.45,           0.0, 2.5)
    head_vis_s = np.clip((X_quality_raw[:, 11:12]- 0.27) / 0.05,  -2.5, 2.5)
    quality_feats = np.hstack([q_bone, q_jitter, q_overall_f, bone_instab, jitter_s,
                               nadir_s, rom_s, sh_w_s, hip_w_s, z_knee_s, z_ank_s, head_vis_s]).astype(np.float32)

    kinetic_raw = k_data['X_kinetic_9d']
    c_fk   = kinetic_raw[:, 0:1]; c_kt = kinetic_raw[:, 1:2]
    c_tn   = kinetic_raw[:, 2:3]; c_ft = kinetic_raw[:, 3:4]
    d_valg = np.clip(kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
    n_tilt = np.clip(kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
    n_valg = np.clip((kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
    a_heel = np.clip(kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
    p_lag  = np.clip(kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)
    kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

    X_meta = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)

    # v11 모델 로드
    moe_model = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)
    moe_model.load_state_dict(torch.load(WEIGHTS_V11, map_location='cpu', weights_only=True))
    moe_model.eval()

    with torch.no_grad():
        raw_l16, raw_l4, _ = moe_model(torch.from_numpy(X_meta).float())
        z_raw_16 = raw_l16.numpy()
        p4_pred  = torch.sigmoid(raw_l4).numpy()

    # 시점별 16D 벡터 스케일러 적용
    with open(SCALER_PATH_V11, 'r', encoding='utf-8') as f:
        scaler_params = json.load(f)

    oof_dz16 = oof_res['oof_delta_z16']
    is_front = (views == 'view2') | (views == 'view3') | (views == 'view4')
    g_vec = np.where(is_front, 0.08, 0.00)[:, None]

    z_cal = np.zeros_like(z_raw_16)
    for v_name in ['view5', 'view2', 'view3', 'view4', 'view1']:
        m = (views == v_name)
        s_v = np.array(scaler_params[v_name]['scale'], dtype=np.float32)
        b_v = np.array(scaler_params[v_name]['bias'], dtype=np.float32)
        z_cal[m] = z_raw_16[m] * s_v + b_v
    z_fused = z_cal + g_vec * oof_dz16
    exp_z = np.exp(z_fused - np.max(z_fused, axis=-1, keepdims=True))
    p16 = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # 위험도
    H_max = np.log(16.0)
    log_p = np.log(np.clip(p16, 1e-8, 1.0))
    ent = -np.sum(p16 * log_p, axis=-1)
    sp = np.sort(p16, axis=-1)
    margin = sp[:, -1] - sp[:, -2]
    risk = (ent / H_max) * (1.0 - margin)

    # 예측 디코딩
    N = len(views)
    q_overall_vec = q_data['X_quality_12d'][:, 2]
    tau_cln = 0.36
    r_thresh = 0.45
    eta = 0.05

    pred_full = np.zeros(N, dtype=int)
    defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4_pred[i], eta=eta, tau_clean=tau_cln)
        if is_cln:
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        else:
            defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        if pred_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_full[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True; continue
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4_pred[i], eta=eta, tau_clean=tau_cln)
        if risk[i] > r_thresh and not is_cln:
            is_retake[i] = True; continue
        if is_cln:
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    # 1. 전수 강제 판정 지표
    m_force = calc_classification_metrics(true_clean, pred_full, p16[:, 0])
    pred_16_f = defs_full[:, 0]*8 + defs_full[:, 1]*4 + defs_full[:, 2]*2 + defs_full[:, 3]*1
    true_16 = y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1
    m_force['emr'] = round(float(np.mean(pred_16_f == true_16)) * 100.0, 2)

    # 2. 선택적 판정 지표
    m_certain = ~is_retake
    m_sel = calc_classification_metrics(true_clean[m_certain], pred_sel[m_certain], p16[m_certain, 0])
    pred_16_s = defs_sel[m_certain, 0]*8 + defs_sel[m_certain, 1]*4 + defs_sel[m_certain, 2]*2 + defs_sel[m_certain, 3]*1
    m_sel['emr'] = round(float(np.mean(pred_16_s == true_16[m_certain])) * 100.0, 2)
    m_sel['coverage'] = round(float(np.mean(m_certain)) * 100.0, 2)

    # 개별 서브모델 F1
    c_names = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
    submodel_f1 = {
        c_names[k]: round(float(f1_score(y_all[m_certain, k], defs_sel[m_certain, k], zero_division=0) * 100.0), 2)
        for k in range(4)
    }

    # 5대 시점별 세부 분석 (강제 판정 기준)
    view_details = {}
    for vn in VIEW_NAMES:
        vm = (views == vn)
        mf_v = calc_classification_metrics(true_clean[vm], pred_full[vm], p16[vm, 0])
        emr_v = float(np.mean(pred_16_f[vm] == true_16[vm])) * 100.0
        mf_v['emr'] = round(emr_v, 2)
        view_details[vn] = mf_v

    return {
        'force': m_force,
        'selective': m_sel,
        'submodels': submodel_f1,
        'views': view_details
    }


def run_evaluation():
    # 1. Model-B 승격 저장
    promote_model_b_weights()

    # 2. 원천 데이터 로드
    master_d = np.load(MASTER_PATH, allow_pickle=True)
    meta_d   = np.load(META_PATH,   allow_pickle=True)
    c4_d     = np.load(C4_PATH,     allow_pickle=True)
    q_d      = np.load(QUALITY_PATH,allow_pickle=True)
    c2_d     = np.load(C2_FEATS_PATH, allow_pickle=True)
    norm_d   = np.load(NORM_PRED_PATH, allow_pickle=True)
    k_d      = np.load(KINETIC_PATH,   allow_pickle=True)
    oof_res  = np.load(OOF_RES_PATH,   allow_pickle=True)

    views      = meta_d['views']
    y_all      = meta_d['y_all']        # (2605, 4) [c1, c2, c3, c4]
    true_clean = meta_d['true_clean']  # (2605,) 1=Clean, 0=Defect
    N = len(true_clean)

    # 3. OOF 산출 (캐시 활용)
    oof_cache_path = r"D:\Data\피처_데이터\squat_c1_oof_predictions_a_and_b.npz"
    if os.path.exists(oof_cache_path):
        print(f"[2] 저장된 c1 OOF 예측 확률 캐시 로드: {oof_cache_path}")
        c_oof = np.load(oof_cache_path, allow_pickle=True)
        oof_p_A = c_oof['oof_p_A']
        oof_p_B = c_oof['oof_p_B']
    else:
        # c1 피처 로드
        a_38 = np.load(A_FEATS_38D, allow_pickle=True)['X_c1_v11_a']
        a_27 = np.load(A_FEATS_27D, allow_pickle=True)['X_c1_v11_a_diet']
        a_t24 = np.load(A_TENSOR_24D, allow_pickle=True)['X_temporal']

        b_feats = np.load(B_FEATS_ALL, allow_pickle=True)
        b_38 = b_feats['X_38d_vision']
        b_27 = b_feats['X_27d_vision']
        b_t24 = b_feats['X_temporal_vision']

        oof_p_A, oof_p_B = get_oof_predictions(master_d, a_38, a_27, a_t24, b_38, b_27, b_t24)
        np.savez_compressed(oof_cache_path, oof_p_A=oof_p_A, oof_p_B=oof_p_B)
        print(f"[+] c1 OOF 캐시 저장 완료: {oof_cache_path}")

    # 4. 베이스라인 결함 확률 구성 (c2, c3, c4는 공식 챔피언 고정)
    probs_base = meta_d['probs_defect'].copy()
    p_c2_feats = c2_d['X_c2_v7_19d']
    face_valid = (p_c2_feats[:, 14] > 0.5)
    is_rear = (views == 'view1') | (views == 'view5')
    sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
    kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
    eps = 1e-7
    p2_clip = np.clip(probs_base[:, 1], eps, 1.0 - eps)
    logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
    probs_base[:, 1] = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
    probs_base[:, 2] = 0.70 * probs_base[:, 2] + 0.30 * norm_d['p_c3_norm']

    # -----------------------------------------------------------------
    # [평가 체계 1] 4대 결함 독립 결합 판정 (Threshold=0.50)
    # -----------------------------------------------------------------
    # Model-A 적용 시
    probs_indep_A = probs_base.copy()
    probs_indep_A[:, 0] = oof_p_A
    preds_indep_A = (probs_indep_A >= 0.50).astype(int)
    pred_clean_indep_A = (np.sum(preds_indep_A, axis=1) == 0).astype(int)
    prob_clean_indep_A = np.prod(1.0 - probs_indep_A, axis=1)

    m_indep_A = calc_classification_metrics(true_clean, pred_clean_indep_A, prob_clean_indep_A)
    m_indep_A['emr'] = round(float(np.mean(np.all(preds_indep_A == y_all, axis=1)) * 100.0), 2)

    # Model-B 적용 시
    probs_indep_B = probs_base.copy()
    probs_indep_B[:, 0] = oof_p_B
    preds_indep_B = (probs_indep_B >= 0.50).astype(int)
    pred_clean_indep_B = (np.sum(preds_indep_B, axis=1) == 0).astype(int)
    prob_clean_indep_B = np.prod(1.0 - probs_indep_B, axis=1)

    m_indep_B = calc_classification_metrics(true_clean, pred_clean_indep_B, prob_clean_indep_B)
    m_indep_B['emr'] = round(float(np.mean(np.all(preds_indep_B == y_all, axis=1)) * 100.0), 2)

    # -----------------------------------------------------------------
    # [평가 체계 2] 16-way MoE 통합 마스터 엔진 v11.0 적용
    # -----------------------------------------------------------------
    res_moe_A = evaluate_moe_v11(probs_indep_A, y_all, true_clean, views, c4_d, q_d, k_d, oof_res)
    res_moe_B = evaluate_moe_v11(probs_indep_B, y_all, true_clean, views, c4_d, q_d, k_d, oof_res)

    # -----------------------------------------------------------------
    # 5대 시점별 세부 비교 집계
    # -----------------------------------------------------------------
    view_comparison = {}
    for vn in VIEW_NAMES:
        alias = VIEW_ALIASES.get(vn, vn)
        count = int(np.sum(views == vn))
        v_a = res_moe_A['views'][vn]
        v_b = res_moe_B['views'][vn]
        view_comparison[vn] = {
            'alias': alias,
            'count': count,
            'model_a': v_a,
            'model_b': v_b,
            'delta_clean_f1': round(v_b['clean_f1'] - v_a['clean_f1'], 2),
            'delta_clean_prec': round(v_b['clean_prec'] - v_a['clean_prec'], 2),
            'delta_emr': round(v_b['emr'] - v_a['emr'], 2),
            'delta_fp': v_b['false_clean_fp'] - v_a['false_clean_fp']
        }

    # 결과 종합 객체
    final_output = {
        'total_samples': N,
        'true_clean_count': int(np.sum(true_clean == 1)),
        'true_defect_count': int(np.sum(true_clean == 0)),
        'independent_combination': {
            'description': '4대 결함 서브모델 독립 결합 판정 (c1~c4 모두 0일 때 정상)',
            'model_a': m_indep_A,
            'model_b': m_indep_B,
            'delta': {
                'clean_f1': round(m_indep_B['clean_f1'] - m_indep_A['clean_f1'], 2),
                'clean_prec': round(m_indep_B['clean_prec'] - m_indep_A['clean_prec'], 2),
                'clean_rec': round(m_indep_B['clean_rec'] - m_indep_A['clean_rec'], 2),
                'emr': round(m_indep_B['emr'] - m_indep_A['emr'], 2),
                'false_clean_fp': m_indep_B['false_clean_fp'] - m_indep_A['false_clean_fp']
            }
        },
        'integrated_moe_master': {
            'description': '공식 마스터 챔피언 MoE 통합 진단 엔진 v11.0 기반 판정',
            'model_a': {
                'force': res_moe_A['force'],
                'selective': res_moe_A['selective'],
                'submodels': res_moe_A['submodels']
            },
            'model_b': {
                'force': res_moe_B['force'],
                'selective': res_moe_B['selective'],
                'submodels': res_moe_B['submodels']
            },
            'delta_force': {
                'clean_f1': round(res_moe_B['force']['clean_f1'] - res_moe_A['force']['clean_f1'], 2),
                'clean_prec': round(res_moe_B['force']['clean_prec'] - res_moe_A['force']['clean_prec'], 2),
                'clean_rec': round(res_moe_B['force']['clean_rec'] - res_moe_A['force']['clean_rec'], 2),
                'emr': round(res_moe_B['force']['emr'] - res_moe_A['force']['emr'], 2),
                'false_clean_fp': res_moe_B['force']['false_clean_fp'] - res_moe_A['force']['false_clean_fp']
            },
            'delta_selective': {
                'clean_f1': round(res_moe_B['selective']['clean_f1'] - res_moe_A['selective']['clean_f1'], 2),
                'clean_prec': round(res_moe_B['selective']['clean_prec'] - res_moe_A['selective']['clean_prec'], 2),
                'clean_rec': round(res_moe_B['selective']['clean_rec'] - res_moe_A['selective']['clean_rec'], 2),
                'emr': round(res_moe_B['selective']['emr'] - res_moe_A['selective']['emr'], 2),
                'coverage': round(res_moe_B['selective']['coverage'] - res_moe_A['selective']['coverage'], 2),
                'false_clean_fp': res_moe_B['selective']['false_clean_fp'] - res_moe_A['selective']['false_clean_fp']
            }
        },
        'view_comparison': view_comparison
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 정상/오류 분류 비교 평가 결과 JSON 저장 완료: {OUT_JSON}")

    # 콘솔 요약 출력
    print("\n" + "=" * 115)
    print("  ★ 바벨 스쿼트 정상/오류 분류 모델 성능 비교 분석: Model-A(기존 AI-Hub) vs Model-B(신규 SpinePose)")
    print("=" * 115)
    print(f"  • 총 데이터 수: {N}건 (피험자 9명 전수, 5대 전 시점) | 실제 정상(Clean): {final_output['true_clean_count']}건, 실제 결함: {final_output['true_defect_count']}건")
    print("-" * 115)
    print("  [1. 통합 마스터 MoE 엔진 (전수 100% 강제 판정)]")
    fa = res_moe_A['force']
    fb = res_moe_B['force']
    print(f"    - 정상(Clean) F1-Score : Model-A {fa['clean_f1']:.2f}%  ──>  Model-B {fb['clean_f1']:.2f}%  (격차: {fb['clean_f1']-fa['clean_f1']:+.2f}%p)")
    print(f"    - 정상(Clean) 정밀도   : Model-A {fa['clean_prec']:.2f}%  ──>  Model-B {fb['clean_prec']:.2f}%  (격차: {fb['clean_prec']-fa['clean_prec']:+.2f}%p)")
    print(f"    - 정상(Clean) 재현율   : Model-A {fa['clean_rec']:.2f}%  ──>  Model-B {fb['clean_rec']:.2f}%  (격차: {fb['clean_rec']-fa['clean_rec']:+.2f}%p)")
    print(f"    - 거짓 정상(오진, FP)  : Model-A {fa['false_clean_fp']}건  ──>  Model-B {fb['false_clean_fp']}건  (격차: {fb['false_clean_fp']-fa['false_clean_fp']:+d}건)")
    print(f"    - 4대 결함 EMR (완전일치): Model-A {fa['emr']:.2f}%  ──>  Model-B {fb['emr']:.2f}%  (격차: {fb['emr']-fa['emr']:+.2f}%p)")
    print(f"    - 전체 이진 정확도     : Model-A {fa['acc']:.2f}%  ──>  Model-B {fb['acc']:.2f}%  (격차: {fb['acc']-fa['acc']:+.2f}%p)")
    print("-" * 115)
    print("  [2. 통합 마스터 MoE 엔진 (선택적 판정 Selective)]")
    sa = res_moe_A['selective']
    sb = res_moe_B['selective']
    print(f"    - 판정 채택률 (Coverage): Model-A {sa['coverage']:.2f}%  ──>  Model-B {sb['coverage']:.2f}%  (격차: {sb['coverage']-sa['coverage']:+.2f}%p)")
    print(f"    - 정상(Clean) F1-Score : Model-A {sa['clean_f1']:.2f}%  ──>  Model-B {sb['clean_f1']:.2f}%  (격차: {sb['clean_f1']-sa['clean_f1']:+.2f}%p)")
    print(f"    - 정상(Clean) 정밀도   : Model-A {sa['clean_prec']:.2f}%  ──>  Model-B {sb['clean_prec']:.2f}%  (격차: {sb['clean_prec']-sa['clean_prec']:+.2f}%p)")
    print(f"    - 4대 결함 EMR (완전일치): Model-A {sa['emr']:.2f}%  ──>  Model-B {sb['emr']:.2f}%  (격차: {sb['emr']-sa['emr']:+.2f}%p)")
    print("-" * 115)
    print("  [3. 5대 공식 카메라 시점별 정상 F1-Score & EMR 변화]")
    for vn, vc in view_comparison.items():
        ma_v = vc['model_a']
        mb_v = vc['model_b']
        print(f"    • {vc['alias']:<22} | Clean F1: {ma_v['clean_f1']:5.2f}% -> {mb_v['clean_f1']:5.2f}% ({vc['delta_clean_f1']:+5.2f}%p) | EMR: {ma_v['emr']:5.2f}% -> {mb_v['emr']:5.2f}% ({vc['delta_emr']:+5.2f}%p) | False Clean: {ma_v['false_clean_fp']}->{mb_v['false_clean_fp']}건")
    print("=" * 115)


if __name__ == '__main__':
    run_evaluation()
