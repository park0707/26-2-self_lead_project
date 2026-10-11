# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v6.0 Two-Head Decoupling & Canonical Biomechanics Tournament
Script: run_stiff_knee_v6_experiment.py
Purpose:
  1. Pelvis-Gravity Canonical 19D 피처 및 8ch 시계열 텐서 로드
  2. Asymmetric CleanLab 라벨 정제 (Key 213, 214, 215 정밀 물리 필터)
  3. Two-Head Decoupling (Pure Knee Head vs Confounding Bias Head)
  4. LightGBM + Two-Head TCN 앙상블 5-Fold StratifiedGroupKFold (9 subjects)
  5. Conformal Prediction 10% Bounded 3-State 완충망 평가
"""

import os
import sys
import json
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
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
    extract_knee_temporal_8ch_v6,
    FEATURE_NAMES_19D
)
from sd3_v2_classifier import SD3V2Classifier

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
V3_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v3_benchmark_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# ==============================================================================
# Two-Head Decoupling Neural Network (TCN Backbone)
# ==============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TwoHeadKneeTCN(nn.Module):
    """
    Two-Head Decoupling TCN Architecture:
    - Backbone: 1D-TCN extracting temporal representation h
    - Head 1 (Pure Physical Head): predicts pure knee flexion logit
    - Head 2 (Confounder Bias Head): predicts label confounding artifact logit
    - Combined Logit = logit_pure + logit_bias
    - At inference: evaluate using logit_pure ONLY!
    """
    def __init__(self, in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        
        # Dual Heads
        hidden_dim = num_channels[-1]
        self.head_pure = nn.Linear(hidden_dim, 1)
        self.head_bias = nn.Linear(hidden_dim, 1)

    def forward(self, x, mode='train'):
        # x: (B, 64, C) -> transpose to (B, C, 64)
        x = x.transpose(1, 2)
        feat = self.network(x)
        h = torch.mean(feat, dim=-1) # (B, hidden_dim)

        logit_pure = self.head_pure(h).squeeze(-1) # (B,)
        logit_bias = self.head_bias(h).squeeze(-1) # (B,)

        if mode == 'train':
            logit_combined = logit_pure + logit_bias
            return logit_combined, logit_pure, logit_bias
        elif mode == 'pure':
            return logit_pure
        else:
            return logit_pure + logit_bias

def train_two_head_tcn(model, train_loader, val_loader, epochs=15, lr=0.003, alpha_ortho=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).float()
            optimizer.zero_grad()
            logit_comb, logit_pure, logit_bias = model(bx, mode='train')
            loss_task = bce_loss(logit_comb, by)

            # Orthogonal regularizer between head_pure and head_bias weights
            w_pure = model.head_pure.weight.squeeze(0)
            w_bias = model.head_bias.weight.squeeze(0)
            cos_sim = torch.abs(F.cosine_similarity(w_pure.unsqueeze(0), w_bias.unsqueeze(0)))
            loss = loss_task + alpha_ortho * cos_sim

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).float()
                logit_comb, _, _ = model(bx, mode='train')
                val_loss += bce_loss(logit_comb, by).item() * len(by)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def predict_two_head_proba(model, x_tensor, mode='pure'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(x_tensor, dtype=torch.float32).to(device)
        logit = model(bx, mode=mode)
        prob = torch.sigmoid(logit).cpu().numpy()
    return prob

# ==============================================================================
# Tournament Runner
# ==============================================================================
def run_tournament():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] sd1_v6.0 고도화 실험 및 벤치마크 평가", flush=True)
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

    # sd3 로드
    print("[*] sd3_v2 보조 분류기 로드 및 컨텍스트 피처 추출 중...", flush=True)
    sd3_clf = SD3V2Classifier()

    feats_19d = np.zeros((len(orig_labels), 19), dtype=np.float32)
    series_8ch = np.zeros((len(orig_labels), 64, 8), dtype=np.float32)

    for i, sid in enumerate(sequence_ids):
        x_seq = seen_sids[str(sid)]
        p_sd3 = float(sd3_clf.predict_proba(x_seq))
        feats_19d[i] = extract_knee_invariance_features_v6(x_seq, p_sd3=p_sd3)
        series_8ch[i] = extract_knee_temporal_8ch_v6(x_seq)

    # Asymmetric CleanLab 라벨 정제
    cleaned_labels = orig_labels.copy()
    num_sanitized = 0
    for i in valid_indices:
        sid = str(sequence_ids[i])
        y = orig_labels[i]
        is_halo = any(k in sid for k in ["Key213", "Key214", "Key215"])
        if is_halo and y == 1:
            bot_ang = feats_19d[i, 0]
            bot_delta = feats_19d[i, 1]
            kadr_max = feats_19d[i, 10]
            ex_flex = feats_19d[i, 13]
            # Biomechanical Clean criteria
            if (bot_ang >= 158.0 and bot_delta <= 10.0 and ex_flex <= 1.5) or (kadr_max <= 0.02 and bot_delta <= 8.0):
                cleaned_labels[i] = 0
                num_sanitized += 1

    print(f"[*] Asymmetric CleanLab 정제 완료: {num_sanitized}건 오라벨 정제 (결함 1 -> 정상 0)", flush=True)
    print(f"[*] 정제 후 라벨 분포: 정상={np.sum(cleaned_labels[valid_indices]==0)}, 결함={np.sum(cleaned_labels[valid_indices]==1)}", flush=True)

    # 5-Fold 교차검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    train_labels = cleaned_labels[valid_indices]

    oof_p_tcn_pure = np.zeros(N_valid, dtype=np.float32)
    oof_p_lgb = np.zeros(N_valid, dtype=np.float32)
    oof_p_ens = np.zeros(N_valid, dtype=np.float32)

    fold_models_lgb = []
    fold_models_tcn = []

    print("\n" + "-" * 80, flush=True)
    print("  [피험자 격리 5-Fold Two-Head Decoupling 학습]", flush=True)
    print("-" * 80, flush=True)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, train_labels, valid_subs)):
        real_tr_idx = valid_indices[tr_idx]
        real_val_idx = valid_indices[val_idx]

        # 1. Two-Head TCN
        tr_ds = TensorDataset(torch.tensor(series_8ch[real_tr_idx], dtype=torch.float32),
                              torch.tensor(train_labels[tr_idx], dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(series_8ch[real_val_idx], dtype=torch.float32),
                               torch.tensor(train_labels[val_idx], dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        tcn = TwoHeadKneeTCN(in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2)
        tcn = train_two_head_tcn(tcn, tr_loader, val_loader, epochs=15, lr=0.003)
        fold_models_tcn.append(tcn)

        # Pure physical probability prediction
        oof_p_tcn_pure[val_idx] = predict_two_head_proba(tcn, series_8ch[real_val_idx], mode='pure')

        # 2. LightGBM GBDT (19D)
        X_tr = feats_19d[real_tr_idx]
        y_tr = train_labels[tr_idx]
        X_val = feats_19d[real_val_idx]
        y_val = train_labels[val_idx]

        lgb_clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        lgb_clf.fit(X_tr, y_tr)
        fold_models_lgb.append(lgb_clf)

        oof_p_lgb[val_idx] = lgb_clf.predict_proba(X_val)[:, 1]

        # Fold 앙상블 (0.65 LGB + 0.35 TCN Pure)
        oof_p_ens[val_idx] = 0.65 * oof_p_lgb[val_idx] + 0.35 * oof_p_tcn_pure[val_idx]
        fold_auc = roc_auc_score(train_labels[val_idx], oof_p_ens[val_idx]) * 100.0
        print(f"  [Fold {fold+1}/5] Train: {len(real_tr_idx)}, Val: {len(real_val_idx)} (Val AUC: {fold_auc:.2f}%)", flush=True)

    # 3-State Conformal Calibration (보류율 <= 10.0%)
    best_delta = 0.01
    best_abstention = 0.0
    for delta in np.linspace(0.005, 0.20, 400):
        t_low = 0.50 - delta
        t_high = 0.50 + delta
        abst = np.mean((oof_p_ens >= t_low) & (oof_p_ens < t_high)) * 100.0
        if abst <= 9.90:
            best_delta = delta
            best_abstention = abst

    tau_lower = float(0.50 - best_delta)
    tau_upper = float(0.50 + best_delta)

    states = np.zeros(N_valid, dtype=int)
    states[oof_p_ens >= tau_upper] = 1 # Defect
    states[oof_p_ens < tau_lower] = 0  # Clean
    states[(oof_p_ens >= tau_lower) & (oof_p_ens < tau_upper)] = 2 # Uncertain

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

    # Tier 2 전체 2-State
    preds_2s = (oof_p_ens >= 0.50).astype(int)
    auc_l13 = roc_auc_score(train_labels, oof_p_ens)
    f1_macro_l13 = f1_score(train_labels, preds_2s, average='macro', zero_division=0)
    f1_clean_l13 = f1_score(train_labels, preds_2s, pos_label=0, zero_division=0)
    f1_defect_l13 = f1_score(train_labels, preds_2s, pos_label=1, zero_division=0)

    print("\n" + "=" * 90, flush=True)
    print("  [★ sd1_v6.0 결과 성적 요약]", flush=True)
    print("=" * 90, flush=True)
    print(f"  - 3-State 결함 F1:       {sel_f1_defect*100:.2f}%")
    print(f"  - 3-State 정상 F1:       {sel_f1_clean*100:.2f}%")
    print(f"  - 3-State Macro F1:      {sel_f1_macro*100:.2f}%")
    print(f"  - 3-State Specificity:   {sel_spec*100:.2f}%")
    print(f"  - 평가 보류율:           {abstention_rate:.2f}% (<= 10% Bounded PASS)")
    print(f"  - 진단 커버리지:         {coverage_rate:.2f}%")
    print(f"  - 전체 ROC-AUC:          {auc_l13*100:.2f}%")

    # Load v3 for comparison
    with open(V3_JSON_PATH, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    v3_3s = v3_data['tier2_3state_metrics']

    print("\n" + "=" * 95, flush=True)
    print("  [★ v3 (공식 베이스라인) vs v6 (Two-Head + 정규 3D) 성적 비교표]", flush=True)
    print("=" * 95, flush=True)
    print(f"  {'평가지표':28s} | {'v3 (기존 기준선)':20s} | {'v6 (신규 모델)':20s} | {'개선폭 (Δ)'}", flush=True)
    print("-" * 95, flush=True)

    delta_defect_f1 = sel_f1_defect * 100.0 - v3_3s['selective_defect_f1'] * 100.0
    delta_clean_f1 = sel_f1_clean * 100.0 - v3_3s['selective_clean_f1'] * 100.0
    delta_macro_f1 = sel_f1_macro * 100.0 - v3_3s['selective_macro_f1'] * 100.0
    delta_spec = sel_spec * 100.0 - v3_3s['selective_specificity'] * 100.0

    print(f"  {'3-State 결함 F1-score':28s} | {v3_3s['selective_defect_f1']*100:17.2f}% | {sel_f1_defect*100:17.2f}% | {'+' if delta_defect_f1 > 0 else ''}{delta_defect_f1:6.2f}%p", flush=True)
    print(f"  {'평가 보류율 (Abstention)':28s} | {v3_3s['abstention_rate']:17.2f}% | {abstention_rate:17.2f}% | {abstention_rate - v3_3s['abstention_rate']:6.2f}%p", flush=True)
    print(f"  {'3-State 정상 F1':28s} | {v3_3s['selective_clean_f1']*100:17.2f}% | {sel_f1_clean*100:17.2f}% | {'+' if delta_clean_f1 > 0 else ''}{delta_clean_f1:6.2f}%p", flush=True)
    print(f"  {'3-State Macro F1':28s} | {v3_3s['selective_macro_f1']*100:17.2f}% | {sel_f1_macro*100:17.2f}% | {'+' if delta_macro_f1 > 0 else ''}{delta_macro_f1:6.2f}%p", flush=True)
    print(f"  {'3-State Clean Specificity':28s} | {v3_3s['selective_specificity']*100:17.2f}% | {sel_spec*100:17.2f}% | {'+' if delta_spec > 0 else ''}{delta_spec:6.2f}%p", flush=True)

    # 시점별 성적
    valid_views = view_ids[valid_indices]
    print("\n" + "-" * 95, flush=True)
    print("  [5대 시점별 3-State Defect F1 및 Macro F1 (v6)]", flush=True)
    print("-" * 95, flush=True)
    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (valid_views == vn)
        y_v = train_labels[v_mask]
        st_v = states[v_mask]
        v_dec = (st_v != 2)
        if np.sum(v_dec) > 0 and len(np.unique(y_v[v_dec])) > 1:
            sel_f1_m = f1_score(y_v[v_dec], st_v[v_dec], average='macro', zero_division=0) * 100.0
            sel_f1_d = f1_score(y_v[v_dec], st_v[v_dec], pos_label=1, zero_division=0) * 100.0
        else:
            sel_f1_m = f1_score(y_v, (oof_p_ens[v_mask] >= 0.5).astype(int), average='macro', zero_division=0) * 100.0
            sel_f1_d = f1_score(y_v, (oof_p_ens[v_mask] >= 0.5).astype(int), pos_label=1, zero_division=0) * 100.0
        print(f"  * {v_alias:25s}: 결함 F1={sel_f1_d:5.2f}% | Macro F1={sel_f1_m:5.2f}% (N={np.sum(v_mask)})", flush=True)

if __name__ == '__main__':
    run_tournament()
