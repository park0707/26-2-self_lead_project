# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_joint_meta_16way_v8_2.py
Purpose: 해결책 2 [Level 1 + Level 2 계층적 보정] 기반 v8.2 16-Way Joint Head 5-Fold 학습
         - Level 1: 4대 서브모델 로짓 온도 보정 (c1 T=0.456, c2 5-view T=0.89~0.96 등)
         - Level 2: 16-Way 메타 소프트맥스 온도 스케일링 (T_meta) 및 하이브리드 안전 디코딩 최적화
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_net import (
    JointMeta16WayNet,
    MultiTaskConsistencyLoss,
    decode_hybrid,
    STATE_TO_DEFECTS_MAP,
    idx_to_defect_tuple
)

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")
SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_weights.pth")
SAVE_META_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_meta.json")

# 1. 데이터 로드 및 Level 1 확률 보정
print("=" * 95)
print("  [Step 2] Level 1 보정된 확률 벡터 기반 18D 특성 행렬 구성")
print("=" * 95)

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)

probs_defect = data['probs_defect'].copy() # (N, 4)
views = data['views']                      # (N,)
subjects = data['subjects']                # (N,)
y_all = data['y_all']                      # (N, 4)
true_clean = data['true_clean']            # (N,)
N = len(true_clean)

# 온도 파라미터 로드
with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    temp_params = json.load(f)

T_c1 = temp_params['c1']['global_T']
T_c3 = temp_params['c3']['global_T']
T_c4 = temp_params['c4']['global_T']
view_T_c2 = temp_params['c2']['view_T']

eps = 1e-7
p_clip = np.clip(probs_defect, eps, 1.0 - eps)
logits = np.log(p_clip / (1.0 - p_clip))

# Level 1 보정 적용
p_calibrated = np.zeros_like(p_clip)
p_calibrated[:, 0] = 1.0 / (1.0 + np.exp(-logits[:, 0] / T_c1))
p_calibrated[:, 2] = 1.0 / (1.0 + np.exp(-logits[:, 2] / T_c3))
p_calibrated[:, 3] = 1.0 / (1.0 + np.exp(-logits[:, 3] / T_c4))

for v, T_v in view_T_c2.items():
    mask = (views == v)
    p_calibrated[mask, 1] = 1.0 / (1.0 + np.exp(-logits[mask, 1] / T_v))

# 물리 피처 및 18D 결합
X_c4 = c4_data['X_c4_v5_25d']
jerk_norm = np.clip(X_c4[:, 15:16] / 200.0, 0.0, 2.5)
wobble_norm = np.clip(X_c4[:, 3:4] / 0.015, 0.0, 2.5)
ydelta_norm = np.clip(X_c4[:, 6:7] / 0.020, -1.0, 2.5)
toe_norm = np.clip(X_c4[:, 1:2] / 0.020, 0.0, 2.5)
f0_norm = np.clip(X_c4[:, 24:25] / 0.50, 0.0, 2.0)
phys_feats = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])

margins_cal = np.abs(p_calibrated - 0.5)

view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot = np.eye(5, dtype=np.float32)[view_indices]

X_meta_18d = np.hstack([p_calibrated, margins_cal, view_onehot, phys_feats]).astype(np.float32)
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

# 클래스 완화 역빈도 가중치
class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * class_counts)) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)

print(f"  총 표본: {N}건 | 18D 메타 특성 행렬 구축 완료 (보정 확률 + 보정 마진 + 시점 + 물리)")

# 2. 5-Fold StratifiedGroupKFold 피험자 격리 학습
print("\n" + "=" * 95)
print("  [Step 3] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습")
print("=" * 95)

sgkf = StratifiedGroupKFold(n_splits=5)
oof_l16 = np.zeros((N, 16), dtype=np.float32)
oof_l4 = np.zeros((N, 4), dtype=np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta_18d, y_16, groups=subjects)):
    val_subs = np.unique(subjects[val_idx])
    print(f"  --- Fold {fold + 1}/5 (검증 피험자: {val_subs}) ---")

    x_tr = torch.tensor(X_meta_18d[train_idx], device=device)
    y16_tr = torch.tensor(y_16[train_idx], device=device)
    y4_tr = torch.tensor(y_all[train_idx], dtype=torch.float32, device=device)
    x_va = torch.tensor(X_meta_18d[val_idx], device=device)

    model = JointMeta16WayNet(in_features=18, hidden_dim=64, dropout_rate=0.20).to(device)
    loss_fn = MultiTaskConsistencyLoss(
        weight_ce=0.60,
        weight_bce=0.30,
        weight_consistency=0.10,
        class_weights_16=weights_tensor.to(device)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        l16, l4 = model(x_tr)
        loss, _ = loss_fn(l16, l4, y16_tr, y4_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        l16_val, l4_val = model(x_va)
        oof_l16[val_idx] = l16_val.cpu().numpy()
        oof_l4[val_idx] = l4_val.cpu().numpy()

# 3. Level 2 Meta Softmax Temperature (T_meta) 및 디코딩 최적화
print("\n" + "=" * 95)
print("  [Step 4] Level 2 Meta Softmax 온도(T_meta), eta, tau_clean 다차원 그리드 최적화")
print("=" * 95)

best_cfg = {'T_meta': 1.0, 'eta': 0.20, 'tau_clean': 0.32, 'clean_f1': 0.0, 'emr': 0.0, 'pre': 0.0, 'rec': 0.0}

for T_meta in [0.70, 0.80, 0.85, 0.90, 1.00, 1.10, 1.20]:
    # 16-way softmax with T_meta
    p16_scaled = np.exp(oof_l16 / T_meta) / np.sum(np.exp(oof_l16 / T_meta), axis=1, keepdims=True)
    p4_sig = 1.0 / (1.0 + np.exp(-oof_l4))

    for eta in [0.10, 0.20, 0.30]:
        for tau in [0.26, 0.28, 0.30, 0.32, 0.35, 0.38]:
            pred = np.zeros(N, dtype=int)
            pred_defs = np.zeros((N, 4), dtype=int)

            for i in range(N):
                is_rear = views[i] in ['view1', 'view5']
                s, defs, is_cln, _ = decode_hybrid(p16_scaled[i], p4_sig[i], eta=eta, tau_clean=tau)
                if is_cln:
                    pred[i] = 1
                    pred_defs[i] = [0, 0, 0, 0]
                elif is_rear:
                    # Rear Observable Clean
                    cond_3p_safe = (p_calibrated[i, 0] <= 0.40) and (p_calibrated[i, 2] <= 0.40) and (p_calibrated[i, 3] <= 0.40) and (p_calibrated[i, 1] < 0.70)
                    p_cln_3 = (1.0 - p_calibrated[i, 0]) * (1.0 - p_calibrated[i, 2]) * (1.0 - p_calibrated[i, 3])
                    if cond_3p_safe and p_cln_3 >= 0.55 and (p_calibrated[i, 0] < 0.50 and p_calibrated[i, 2] < 0.50 and p_calibrated[i, 3] < 0.50):
                        pred[i] = 1
                        pred_defs[i] = [0, 0, 0, 0]
                    else:
                        pred_defs[i] = defs
                else:
                    pred_defs[i] = defs

            # Layer 3 Temporal Filtering
            for i in range(N):
                if pred[i] == 0:
                    is_smooth = (X_c4[i, 3] <= 0.008) and (X_c4[i, 15] <= 180.0)
                    if max(p_calibrated[i]) <= 0.50 and is_smooth:
                        pred[i] = 1
                        pred_defs[i] = [0, 0, 0, 0]
                if pred[i] == 1:
                    if (X_c4[i, 15] > 220.0 and X_c4[i, 3] > 0.008) or (X_c4[i, 24] > 0.35):
                        pred[i] = 0

            cur_f1 = f1_score(true_clean, pred, zero_division=0) * 100.0
            cur_pre = precision_score(true_clean, pred, zero_division=0) * 100.0
            cur_rec = recall_score(true_clean, pred, zero_division=0) * 100.0

            pred_states = pred_defs[:, 0]*8 + pred_defs[:, 1]*4 + pred_defs[:, 2]*2 + pred_defs[:, 3]*1
            cur_emr = (pred_states == y_16).mean() * 100.0

            if cur_pre >= 55.0 and cur_f1 > best_cfg['clean_f1']:
                best_cfg = {
                    'T_meta': T_meta,
                    'eta': eta,
                    'tau_clean': tau,
                    'clean_f1': cur_f1,
                    'emr': cur_emr,
                    'pre': cur_pre,
                    'rec': cur_rec
                }

print(f"  ★ v8.2 최적 하이퍼파라미터 선정:")
print(f"    - Level 2 Meta Temperature: T_meta = {best_cfg['T_meta']:.2f}")
print(f"    - Hybrid Weight: eta = {best_cfg['eta']:.2f}")
print(f"    - Clean Safety Threshold: tau_clean = {best_cfg['tau_clean']:.2f}")
print(f"    - Clean F1-Score: {best_cfg['clean_f1']:.2f}% (Precision: {best_cfg['pre']:.2f}%, Recall: {best_cfg['rec']:.2f}%)")
print(f"    - 4-Defect Exact Match Ratio (EMR): {best_cfg['emr']:.2f}%")

# 4. 전체 데이터 기반 최종 프로덕션 모델 학습 및 저장
print("\n" + "=" * 95)
print("  [Step 5] 전체 데이터 기반 최종 프로덕션 가중치 학습 및 저장")
print("=" * 95)

prod_model = JointMeta16WayNet(in_features=18, hidden_dim=64, dropout_rate=0.20).to(device)
prod_loss_fn = MultiTaskConsistencyLoss(
    weight_ce=0.60,
    weight_bce=0.30,
    weight_consistency=0.10,
    class_weights_16=weights_tensor.to(device)
).to(device)

prod_opt = optim.AdamW(prod_model.parameters(), lr=0.005, weight_decay=1e-3)
prod_sched = optim.lr_scheduler.CosineAnnealingLR(prod_opt, T_max=120, eta_min=1e-4)

x_full = torch.tensor(X_meta_18d, device=device)
y16_full = torch.tensor(y_16, device=device)
y4_full = torch.tensor(y_all, dtype=torch.float32, device=device)

prod_model.train()
for epoch in range(1, 121):
    prod_opt.zero_grad()
    l16, l4 = prod_model(x_full)
    loss, _ = prod_loss_fn(l16, l4, y16_full, y4_full)
    loss.backward()
    prod_opt.step()
    prod_sched.step()

torch.save(prod_model.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  ★ v8.2 가중치 저장 완료: {SAVE_WEIGHTS_PATH}")

meta_dict_v8_2 = {
    'version': 'v8.2',
    'in_features': 18,
    'hidden_dim': 64,
    'T_meta': float(best_cfg['T_meta']),
    'eta': float(best_cfg['eta']),
    'tau_clean': float(best_cfg['tau_clean']),
    'level1_temperature': temp_params,
    'oof_metrics': {
        'clean_f1': float(best_cfg['clean_f1']),
        'clean_precision': float(best_cfg['pre']),
        'clean_recall': float(best_cfg['rec']),
        'emr': float(best_cfg['emr'])
    }
}
with open(SAVE_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(meta_dict_v8_2, f, indent=4, ensure_ascii=False)
print(f"  ★ v8.2 메타 설정 저장 완료: {SAVE_META_PATH}")
print("=" * 95)
