# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_joint_meta_16way.py
Purpose: 18D 16-Way Joint Meta Head (Label Powerset Softmax + Consistency Loss)
         AI-Hub 9명 피험자 격리 5-Fold StratifiedGroupKFold 교차 검증 학습 및
         최적 하이브리드 디코딩 파라미터(eta, tau_clean) 튜닝, 최종 모델 가중치 저장
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
SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_weights.pth")
SAVE_META_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_meta.json")

# 1. 데이터 로드 및 18D 특성 구성
print("=" * 90)
print("  [Step 1] 18D 메타 특성 행렬 및 16-Way 타깃 구성")
print("=" * 90)

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)

probs_defect = data['probs_defect'] # (N, 4)
views = data['views']               # (N,)
subjects = data['subjects']         # (N,)
y_all = data['y_all']               # (N, 4)
true_clean = data['true_clean']     # (N,)
N = len(true_clean)

X_c4 = c4_data['X_c4_v5_25d']
toe_lift_norm = X_c4[:, 1:2]
wobble = X_c4[:, 3:4]
ydelta = X_c4[:, 6:7]
jerk = X_c4[:, 15:16]
f0_rate = X_c4[:, 24:25]

# 물리 안정성 피처 정규화 (Standardization for Jerk & F0)
jerk_norm = np.clip(jerk / 200.0, 0.0, 2.5)
wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)
ydelta_norm = np.clip(ydelta / 0.020, -1.0, 2.5)
toe_norm = np.clip(toe_lift_norm / 0.020, 0.0, 2.5)
f0_norm = np.clip(f0_rate / 0.50, 0.0, 2.0)

phys_feats = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm]) # (N, 5)
margins = np.abs(probs_defect - 0.5) # (N, 4)

view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot = np.eye(5, dtype=np.float32)[view_indices] # (N, 5)

X_meta_18d = np.hstack([probs_defect, margins, view_onehot, phys_feats]).astype(np.float32)
print(f"  총 표본 수: {N}, 특성 차원: {X_meta_18d.shape[1]}D")

# 16-State 타깃 (0~15)
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

# 클래스별 완화 역빈도 가중치 계산 (Class Weighting with alpha=0.35)
class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * class_counts)) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)
print(f"  State 0 (Clean 135건) 가중치: {smoothed_weights[0]:.3f}, State Max/Min: {smoothed_weights.max():.3f} / {smoothed_weights.min():.3f}")

# 2. 5-Fold StratifiedGroupKFold 피험자 격리 학습
print("\n" + "=" * 90)
print("  [Step 2] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습")
print("=" * 90)

sgkf = StratifiedGroupKFold(n_splits=5)
oof_prob_16 = np.zeros((N, 16), dtype=np.float32)
oof_prob_4 = np.zeros((N, 4), dtype=np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  사용 디바이스: {device}")

torch.manual_seed(42)
np.random.seed(42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta_18d, y_16, groups=subjects)):
    val_subs = np.unique(subjects[val_idx])
    train_subs = np.unique(subjects[train_idx])
    print(f"\n  --- Fold {fold + 1}/5 (검증 피험자: {val_subs}) ---")

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

    best_val_emr = 0.0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        l16, l4 = model(x_tr)
        loss, _ = loss_fn(l16, l4, y16_tr, y4_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # OOF 예측 수집
    model.eval()
    with torch.no_grad():
        l16_val, l4_val = model(x_va)
        p16_val = torch.softmax(l16_val, dim=-1).cpu().numpy()
        p4_val = torch.sigmoid(l4_val).cpu().numpy()

        oof_prob_16[val_idx] = p16_val
        oof_prob_4[val_idx] = p4_val

# 3. 하이브리드 디코딩 파라미터 (eta, tau_clean) 전수 탐색
print("\n" + "=" * 90)
print("  [Step 3] 최적 하이브리드 디코딩 파라미터 (eta, tau_clean) 전수 탐색")
print("=" * 90)

best_cfg = {'eta': 0.5, 'tau_clean': 0.40, 'clean_f1': 0.0, 'emr': 0.0, 'pre': 0.0, 'rec': 0.0}

for eta in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        pred_states = np.zeros(N, dtype=int)
        for i in range(N):
            s, _, _, _ = decode_hybrid(oof_prob_16[i], oof_prob_4[i], eta=eta, tau_clean=tau)
            pred_states[i] = s

        cur_emr = (pred_states == y_16).mean() * 100.0
        cur_clean_pred = (pred_states == 0).astype(int)
        cur_pre = precision_score(true_clean, cur_clean_pred, zero_division=0) * 100.0
        cur_rec = recall_score(true_clean, cur_clean_pred, zero_division=0) * 100.0
        cur_f1 = f1_score(true_clean, cur_clean_pred, zero_division=0) * 100.0

        # 선택 기준: Clean Precision 55% 이상이면서 Clean F1 최대화
        if cur_pre >= 55.0 and cur_f1 > best_cfg['clean_f1']:
            best_cfg = {
                'eta': eta,
                'tau_clean': tau,
                'clean_f1': cur_f1,
                'emr': cur_emr,
                'pre': cur_pre,
                'rec': cur_rec
            }

print(f"  ★ 최적 설정: eta={best_cfg['eta']}, tau_clean={best_cfg['tau_clean']}")
print(f"  ★ Clean F1: {best_cfg['clean_f1']:.2f}% (Precision: {best_cfg['pre']:.2f}%, Recall: {best_cfg['rec']:.2f}%)")
print(f"  ★ 4-Defect EMR: {best_cfg['emr']:.2f}%")

# 최적 설정으로 OOF 전체 예측 생성
final_pred_states = np.zeros(N, dtype=int)
final_pred_defects = np.zeros((N, 4), dtype=int)
for i in range(N):
    s, defs, _, _ = decode_hybrid(oof_prob_16[i], oof_prob_4[i], eta=best_cfg['eta'], tau_clean=best_cfg['tau_clean'])
    final_pred_states[i] = s
    final_pred_defects[i] = defs

final_clean_pred = (final_pred_states == 0).astype(int)
tp = int(np.sum((final_clean_pred == 1) & (true_clean == 1)))
fp = int(np.sum((final_clean_pred == 1) & (true_clean == 0)))

print("\n" + "=" * 90)
print("  [Step 4] 다차원 세부 감사 (Submodel F1 & 5대 시점 재현율)")
print("=" * 90)
print(f"  적중(TP): {tp} / {int(np.sum(true_clean))}, 오인(FP): {fp}건 ({(fp / (N - 135)) * 100:.2f}%)")
print("\n  [4대 세부 결함 개별 F1-Score]")
for k_idx, c_name in enumerate(['c1 (척추 중립)', 'c2 (시선 정면)', 'c3 (무릎 방향)', 'c4 (발바닥 고정)']):
    f = f1_score(y_all[:, k_idx], final_pred_defects[:, k_idx]) * 100.0
    print(f"    - {c_name}: {f:.2f}%")

print("\n  [공식 5대 카메라 시점별 정상 재현율]")
official_views = [
    ('시점 A', 'view5', '우측 후방 (45도)'),
    ('시점 B', 'view2', '우측 전방 (45도)'),
    ('시점 C', 'view3', '정면 (0도)'),
    ('시점 D', 'view4', '좌측 전방 (45도)'),
    ('시점 E', 'view1', '좌측 후방 (45도)'),
]
for v_label, v_code, v_desc in official_views:
    mask = (views == v_code)
    gold_cnt = int(np.sum(true_clean[mask]))
    rec = recall_score(true_clean[mask], final_clean_pred[mask], zero_division=0) * 100.0
    print(f"    - {v_label} [{v_desc}]: {rec:.1f}% (골든 {gold_cnt}건)")

# 5. 전수 데이터 기반 최종 프로덕션 모델 학습 및 저장
print("\n" + "=" * 90)
print("  [Step 5] 전체 데이터 기반 최종 프로덕션 가중치 학습 및 저장")
print("=" * 90)

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

# 모델 저장
torch.save(prod_model.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  ★ 가중치 저장 완료: {SAVE_WEIGHTS_PATH}")

meta_dict = {
    'version': 'v8.1',
    'in_features': 18,
    'hidden_dim': 64,
    'eta': float(best_cfg['eta']),
    'tau_clean': float(best_cfg['tau_clean']),
    'oof_metrics': {
        'clean_f1': float(best_cfg['clean_f1']),
        'clean_precision': float(best_cfg['pre']),
        'clean_recall': float(best_cfg['rec']),
        'emr': float(best_cfg['emr']),
        'tp': tp,
        'fp': fp
    }
}
with open(SAVE_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(meta_dict, f, indent=4, ensure_ascii=False)
print(f"  ★ 메타 설정 저장 완료: {SAVE_META_PATH}")
print("=" * 90)
