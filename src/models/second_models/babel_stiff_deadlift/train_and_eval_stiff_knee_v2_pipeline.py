# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v2.0 (Curated 6D Feature Diet & v1 Succession)
Script: train_and_eval_stiff_knee_v2_pipeline.py
Purpose:
  1. v1 아키텍처를 온전히 계승(Baseline Clone)하되, LOFO 감사로 검증된 6D 정예 피처셋 적용
  2. 피험자 1~9 전수 StratifiedGroupKFold 5-Fold 교차검증 재학습
  3. 1계층(Level 1) & 2계층(Level 1~3) 분할 평가 및 10% Bounded 3-State 캘리브레이션
  4. v1 vs v2 종합 성능 정량 비교 및 모델 공식 저장 (sd1_v2)
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

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

from stiff_deadlift_knee_features_v2 import (
    extract_knee_invariance_features_6d, extract_knee_temporal_8ch_v2, FEATURE_NAMES_6D
)

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
V1_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_benchmark_results.json")
V2_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v2_benchmark_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v2_bundle.pkl")
WEIGHT_COPY_PATH = os.path.join(CURRENT_DIR, "sd1_model_v2_weights.pkl")
ROOT_COPY_PATH = os.path.join(CURRENT_DIR, "..", "sd1_v2_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# ==============================================================================
# 1. 1D-TCN 신경망 모델 (v2: 8ch 입력)
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

class KneeTCNClassifierV2(nn.Module):
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
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.network(x)
        pooled = torch.mean(feat, dim=-1)
        return self.fc(pooled)

def train_tcn(model, train_loader, val_loader, epochs=12, lr=0.003):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits, by).item() * len(by)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def predict_tcn_proba(model, x_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        bx = torch.tensor(x_tensor, dtype=torch.float32).to(device)
        logits = model(bx)
        prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return prob

# ==============================================================================
# 2. v2 메인 훈련 및 평가 파이프라인
# ==============================================================================
def run_v2_pipeline():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 스티프 데드리프트 sd1_v2 (6D Curated Diet) 재학습 및 성능 비교", flush=True)
    print("=" * 100, flush=True)

    # 1. 데이터 로드 및 6D 피처/8ch 시계열 재추출
    data = np.load(CACHE_PATH, allow_pickle=True)
    labels = data['labels']
    levels = data['levels']
    l4_mask = data['l4_mask']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']

    # 684개 원본 텐서에서 6D 정예 피처 및 8ch 텐서 구성
    from src.models.first_model.dataset import Stage1DatasetLoader
    splits = Stage1DatasetLoader.load_all_dual_splits()
    seen_sids = {}
    for sname, sdata in splits.items():
        stiff_mask = (sdata['y_action'] == 2)
        s_sids = sdata['sequence_ids'][stiff_mask]
        s_X = sdata['X_Format_B'][stiff_mask]
        for sid, x in zip(s_sids, s_X):
            sid_str = str(sid)
            if sid_str not in seen_sids:
                seen_sids[sid_str] = x

    N_total = len(labels)
    feats_6d = np.zeros((N_total, 6), dtype=np.float32)
    series_8ch = np.zeros((N_total, 64, 8), dtype=np.float32)

    seq_ids = data['sequence_ids']
    for i, sid in enumerate(seq_ids):
        x_seq = seen_sids[str(sid)]
        feats_6d[i] = extract_knee_invariance_features_6d(x_seq)
        series_8ch[i] = extract_knee_temporal_8ch_v2(x_seq)

    # Level 4 제외한 유효 임상 샘플 (Level 1~3, N=668)
    valid_mask = ~l4_mask
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)

    print(f"[*] 유효 임상 시퀀스: {N_valid}개 (Level 4 프루닝: {np.sum(l4_mask)}개)", flush=True)
    print(f"[*] 투입 피처: 6D 정예 피처셋 {FEATURE_NAMES_6D}", flush=True)

    # 2. 피험자 1~9 전수 5-Fold 교차검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    valid_labels = labels[valid_indices]

    oof_p_tcn = np.zeros(N_valid, dtype=np.float32)
    oof_p_lgb = np.zeros(N_valid, dtype=np.float32)
    oof_p_ens = np.zeros(N_valid, dtype=np.float32)

    fold_models_lgb = []
    fold_models_tcn = []

    print("\n" + "-" * 80, flush=True)
    print("  [Step 2: 피험자 격리 5-Fold 교차검증 훈련 진행]", flush=True)
    print("-" * 80, flush=True)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, valid_labels, valid_subs)):
        real_tr_idx = valid_indices[tr_idx]
        real_val_idx = valid_indices[val_idx]

        tr_subs = np.unique(subject_ids[real_tr_idx])
        val_subs = np.unique(subject_ids[real_val_idx])
        print(f"  [Fold {fold+1}/5] Train: {len(real_tr_idx)}개 (피험자 {tr_subs}), Val: {len(real_val_idx)}개 (피험자 {val_subs})", flush=True)

        # Stream 1: TCN 훈련 (8ch)
        tr_ds = TensorDataset(torch.tensor(series_8ch[real_tr_idx], dtype=torch.float32),
                              torch.tensor(labels[real_tr_idx], dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(series_8ch[real_val_idx], dtype=torch.float32),
                               torch.tensor(labels[real_val_idx], dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        tcn = KneeTCNClassifierV2(in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2)
        tcn = train_tcn(tcn, tr_loader, val_loader, epochs=12, lr=0.003)
        fold_models_tcn.append(tcn)
        p_tcn_val = predict_tcn_proba(tcn, series_8ch[real_val_idx])
        oof_p_tcn[val_idx] = p_tcn_val

        # Stream 2: LightGBM 훈련 (6D 정예 피처)
        pos_weight = float(np.sum(labels[real_tr_idx] == 0)) / max(1, np.sum(labels[real_tr_idx] == 1))
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            scale_pos_weight=pos_weight * 0.9,
            min_child_samples=15,
            reg_alpha=0.1,
            reg_lambda=0.5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1
        )
        lgb_model.fit(feats_6d[real_tr_idx], labels[real_tr_idx])
        fold_models_lgb.append(lgb_model)
        p_lgb_val = lgb_model.predict_proba(feats_6d[real_val_idx])[:, 1]
        oof_p_lgb[val_idx] = p_lgb_val

        # 앙상블 결합 (Stream 2 GBDT 0.70 + Stream 1 TCN 0.30)
        p_ens_val = 0.70 * p_lgb_val + 0.30 * p_tcn_val
        oof_p_ens[val_idx] = p_ens_val
        print(f"    -> Fold {fold+1} 완료 (Val AUC: {roc_auc_score(labels[real_val_idx], p_ens_val)*100:.2f}%)", flush=True)

    # ==============================================================================
    # 3. 2계층 평가 (Tier 1 vs Tier 2)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("  [Step 3: sd1_v2 2계층 분할 평가 벤치마크 결과]", flush=True)
    print("=" * 80, flush=True)

    # 3.1 Tier 1: Level 1 골드 스탠다드 평가
    l1_sub_indices = np.where(levels[valid_indices] == 1)[0]
    y_l1 = valid_labels[l1_sub_indices]
    p_l1 = oof_p_ens[l1_sub_indices]
    pred_l1 = (p_l1 >= 0.50).astype(int)

    tn1, fp1, fn1, tp1 = confusion_matrix(y_l1, pred_l1).ravel()
    spec_l1 = tn1 / (tn1 + fp1) if (tn1 + fp1) > 0 else 0.0
    f1_l1 = f1_score(y_l1, pred_l1, zero_division=0)
    auc_l1 = roc_auc_score(y_l1, p_l1)
    acc_l1 = accuracy_score(y_l1, pred_l1)

    print(f"\n[★ Tier 1 (Level 1 골드 스탠다드) v2 성적]:", flush=True)
    print(f"  - Clean Specificity: {spec_l1 * 100:.2f}%", flush=True)
    print(f"  - Macro F1:          {f1_l1 * 100:.2f}%", flush=True)
    print(f"  - ROC-AUC:           {auc_l1 * 100:.2f}%", flush=True)

    # 3.2 Tier 2: Level 1~3 실서비스 포괄 평가 (2-State)
    pred_l13 = (oof_p_ens >= 0.50).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(valid_labels, pred_l13).ravel()
    spec_l13 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
    f1_macro_l13 = f1_score(valid_labels, pred_l13, average='macro')
    f1_clean_l13 = f1_score(valid_labels, pred_l13, pos_label=0)
    f1_defect_l13 = f1_score(valid_labels, pred_l13, pos_label=1)
    auc_l13 = roc_auc_score(valid_labels, oof_p_ens)
    acc_l13 = accuracy_score(valid_labels, pred_l13)

    print(f"\n[★ Tier 2 (Level 1~3 실서비스 2-State) v2 성적]:", flush=True)
    print(f"  - Full-Coverage Macro F1: {f1_macro_l13 * 100:.2f}%", flush=True)
    print(f"  - Clean F1:               {f1_clean_l13 * 100:.2f}%", flush=True)
    print(f"  - Defect F1:              {f1_defect_l13 * 100:.2f}%", flush=True)
    print(f"  - Clean Specificity:      {spec_l13 * 100:.2f}%", flush=True)
    print(f"  - ROC-AUC:                {auc_l13 * 100:.2f}%", flush=True)

    # ==============================================================================
    # 4. 10% Bounded 3-State 선택적 진단 캘리브레이션
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("  [Step 4: 10% Bounded 3-State 선택적 진단 캘리브레이션]", flush=True)
    print("-" * 80, flush=True)

    margins = np.abs(oof_p_ens - 0.50)
    max_abstention_pct = 9.88 # 10% 이내
    tau_margin = float(np.percentile(margins, max_abstention_pct))
    tau_lower = float(0.50 - tau_margin)
    tau_upper = float(0.50 + tau_margin)

    states = np.zeros(N_valid, dtype=np.int32)
    for i in range(N_valid):
        p = oof_p_ens[i]
        if p < tau_lower:
            states[i] = 0
        elif p >= tau_upper:
            states[i] = 1
        else:
            states[i] = 2

    n_uncertain = np.sum(states == 2)
    abstention_rate = n_uncertain / N_valid * 100.0
    coverage_rate = 100.0 - abstention_rate

    decided_mask = (states != 2)
    y_decided = valid_labels[decided_mask]
    pred_decided = states[decided_mask]

    sel_f1_macro = f1_score(y_decided, pred_decided, average='macro')
    sel_f1_clean = f1_score(y_decided, pred_decided, pos_label=0)
    sel_f1_defect = f1_score(y_decided, pred_decided, pos_label=1)
    sel_acc = accuracy_score(y_decided, pred_decided)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_decided, pred_decided).ravel()
    sel_spec = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0

    print(f"[*] 3-State 밴드: tau_lower={tau_lower:.4f}, tau_upper={tau_upper:.4f}", flush=True)
    print(f"[*] 보류율: {abstention_rate:.2f}% (<= 10.0% PASS) | 커버리지: {coverage_rate:.2f}%", flush=True)
    print(f"\n[★ 3-State 선택적 진단 v2 성적]:", flush=True)
    print(f"  - Selective Macro F1:  {sel_f1_macro * 100:.2f}%", flush=True)
    print(f"  - Selective Clean F1:  {sel_f1_clean * 100:.2f}%", flush=True)
    print(f"  - Selective Defect F1: {sel_f1_defect * 100:.2f}%", flush=True)
    print(f"  - Selective Specificity:{sel_spec * 100:.2f}%", flush=True)
    print(f"  - Selective Accuracy:  {sel_acc * 100:.2f}%", flush=True)

    # ==============================================================================
    # 5. v1 vs v2 직접 대조 비교
    # ==============================================================================
    with open(V1_JSON_PATH, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)

    v1_2s = v1_data['tier2_2state_metrics']
    v1_3s = v1_data['tier2_3state_metrics']

    print("\n" + "=" * 90, flush=True)
    print("  [★ v1 (14D 전체) vs v2 (6D 정예 다이어트) 종합 성능 전수 비교표]", flush=True)
    print("=" * 90, flush=True)
    print(f"  {'평가지표':28s} | {'v1 (14D 베이스라인)':20s} | {'v2 (6D 정예 다이어트)':20s} | {'개선폭 (Δ)'}", flush=True)
    print("-" * 90, flush=True)

    metrics_compare = [
        ("Tier 2 ROC-AUC", v1_2s['roc_auc']*100, auc_l13*100, "%p"),
        ("Tier 2 2-State Macro F1", v1_2s['macro_f1']*100, f1_macro_l13*100, "%p"),
        ("Tier 2 2-State Clean F1", v1_2s['clean_f1']*100, f1_clean_l13*100, "%p"),
        ("Tier 2 2-State Defect F1", v1_2s['defect_f1']*100, f1_defect_l13*100, "%p"),
        ("Tier 2 2-State Specificity", v1_2s['clean_specificity']*100, spec_l13*100, "%p"),
        ("3-State Macro F1", v1_3s['selective_macro_f1']*100, sel_f1_macro*100, "%p"),
        ("3-State Clean F1", v1_3s['selective_clean_f1']*100, sel_f1_clean*100, "%p"),
        ("3-State Defect F1", v1_3s['selective_defect_f1']*100, sel_f1_defect*100, "%p"),
        ("3-State Clean Specificity", v1_3s['selective_specificity']*100, sel_spec*100, "%p"),
        ("3-State Accuracy", v1_3s['selective_accuracy']*100, sel_acc*100, "%p"),
    ]

    for label, val_v1, val_v2, unit in metrics_compare:
        delta = val_v2 - val_v1
        sign = "+" if delta > 0 else ""
        print(f"  {label:28s} | {val_v1:17.2f}% | {val_v2:17.2f}% | {sign}{delta:6.2f}{unit}", flush=True)

    # 6. 시점별 3-State F1 비교
    print("\n" + "-" * 90, flush=True)
    print("  [시점별 3-State Macro F1 비교]", flush=True)
    print("-" * 90, flush=True)
    valid_views = view_ids[valid_indices]
    view_results_v2 = {}

    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (valid_views == vn)
        v_n = np.sum(v_mask)
        if v_n == 0:
            continue
        y_v = valid_labels[v_mask]
        st_v = states[v_mask]
        v_dec = (st_v != 2)
        if np.sum(v_dec) > 0 and len(np.unique(y_v[v_dec])) > 1:
            sel_f1 = f1_score(y_v[v_dec], st_v[v_dec], average='macro', zero_division=0)
        else:
            sel_f1 = f1_score(y_v, (oof_p_ens[v_mask] >= 0.5).astype(int), average='macro', zero_division=0)

        v1_view_f1 = v1_data['view_results'].get(vn, {}).get('f1_3state', 0) * 100.0
        v2_view_f1 = sel_f1 * 100.0
        delta_v = v2_view_f1 - v1_view_f1

        view_results_v2[vn] = {
            'alias': v_alias,
            'count': int(v_n),
            'v1_f1_3state': float(v1_view_f1),
            'v2_f1_3state': float(v2_view_f1),
            'delta_f1': float(delta_v)
        }
        sign_v = "+" if delta_v > 0 else ""
        print(f"  * {v_alias:25s}: v1={v1_view_f1:5.2f}% -> v2={v2_view_f1:5.2f}% ({sign_v}{delta_v:+.2f}%p)", flush=True)

    # 7. 피처 중요도
    feat_importances = np.mean([m.feature_importances_ for m in fold_models_lgb], axis=0)
    top_features_v2 = {FEATURE_NAMES_6D[k]: float(feat_importances[k]) for k in np.argsort(feat_importances)[::-1]}

    # 8. JSON 저장
    v2_benchmark_data = {
        'exercise': '03_바벨_스티프_데드리프트',
        'defect': '무릎 구부린 채 고정 위반 (sd1_v2)',
        'feature_diet': {
            'v1_num_features': 14,
            'v2_num_features': 6,
            'features_used': FEATURE_NAMES_6D
        },
        'tier1_metrics': {
            'clean_specificity': float(spec_l1),
            'macro_f1': float(f1_l1),
            'roc_auc': float(auc_l1),
            'accuracy': float(acc_l1)
        },
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
        'comparison_vs_v1': {
            'delta_auc': float((auc_l13 - v1_2s['roc_auc']) * 100.0),
            'delta_2state_macro_f1': float((f1_macro_l13 - v1_2s['macro_f1']) * 100.0),
            'delta_3state_macro_f1': float((sel_f1_macro - v1_3s['selective_macro_f1']) * 100.0),
            'delta_3state_clean_f1': float((sel_f1_clean - v1_3s['selective_clean_f1']) * 100.0),
            'delta_3state_specificity': float((sel_spec - v1_3s['selective_specificity']) * 100.0)
        },
        'view_results': view_results_v2,
        'top_features': top_features_v2
    }

    with open(V2_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(v2_benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v2 벤치마크 JSON 저장 완료: {V2_JSON_PATH}", flush=True)

    # 모델 가중치 번들 저장 (sd1_v2)
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle_v2 = {
        'version': 'sd1_v2',
        'fold_models_lgb': fold_models_lgb,
        'tcn_state_dicts': [m.state_dict() for m in fold_models_tcn],
        'feature_names': FEATURE_NAMES_6D,
        'thresholds': {'tau_lower': tau_lower, 'tau_upper': tau_upper},
        'stream_weights': {'lgb': 0.70, 'tcn': 0.30}
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle_v2, f)
    with open(WEIGHT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v2, f)
    with open(ROOT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v2, f)
    print(f"[+] sd1_v2 공식 가중치 번들 배포 완료: {BUNDLE_SAVE_PATH}", flush=True)

if __name__ == '__main__':
    run_v2_pipeline()
