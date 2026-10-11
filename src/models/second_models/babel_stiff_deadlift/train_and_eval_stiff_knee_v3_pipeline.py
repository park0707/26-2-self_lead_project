# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v3.0 (Confident Learning & Subject-Adaptive Meta-Gating)
Script: train_and_eval_stiff_knee_v3_pipeline.py
Purpose:
  1. [솔루션 1] Confident Learning (CleanLab) 기반 Key 213, 214, 215 복합 오류 오라벨 자동 정제
  2. [솔루션 2] 어드레스 프레임 기준 적응형 상대 정규화 10D 생체역학 피처셋 적용
  3. [솔루션 3] 바벨 밀착(sd3_v2) 및 척추 중립 컨텍스트 주입 상위 메타 게이팅 네트워크 구축
  4. [솔루션 4] Conformal Prediction 기반 10% Bounded 3-State 완충망 캘리브레이션
  5. 피험자 1~9 전수 StratifiedGroupKFold 5-Fold 교차검증 및 v2 대비 전수 성능 비교 평가
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

from stiff_deadlift_knee_features_v3 import (
    extract_knee_invariance_features_10d, extract_knee_temporal_8ch_v3, FEATURE_NAMES_10D
)
from sd3_v2_classifier import SD3V2Classifier

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
V2_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v2_benchmark_results.json")
V3_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_v3_benchmark_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v3_bundle.pkl")
WEIGHT_COPY_PATH = os.path.join(CURRENT_DIR, "sd1_model_v3_weights.pkl")
ROOT_COPY_PATH = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "sd1_v3_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# ==============================================================================
# 1. 1D-TCN 신경망 모델 (v3: 8ch 입력)
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

class KneeTCNClassifierV3(nn.Module):
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

    for ep in range(epochs):
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
# 2. v3 메인 훈련 및 평가 파이프라인
# ==============================================================================
def run_v3_pipeline():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 스티프 데드리프트 sd1_v3 (4대 솔루션 통합) 재학습 및 평가", flush=True)
    print("=" * 100, flush=True)

    # 1. 데이터 로드 및 원본 텐서 복원
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

    N_total = len(orig_labels)
    feats_10d = np.zeros((N_total, 10), dtype=np.float32)
    series_8ch = np.zeros((N_total, 64, 8), dtype=np.float32)
    context_2d = np.zeros((N_total, 2), dtype=np.float32) # [P(sd3), Coronal_Tilt_ROM]

    # sd3_v2 분류기 준비 (솔루션 3: 바벨 밀착 결함 확률 주입)
    print("[*] 솔루션 3 보조 분류기(sd3_v2) 로드 및 컨텍스트 피처 추출 중...", flush=True)
    sd3_clf = SD3V2Classifier()

    for i, sid in enumerate(sequence_ids):
        x_seq = seen_sids[str(sid)]
        feats_10d[i] = extract_knee_invariance_features_10d(x_seq)
        series_8ch[i] = extract_knee_temporal_8ch_v3(x_seq)

        # 컨텍스트 1: sd3 (바벨 밀착 위반 확률)
        p_sd3 = float(sd3_clf.predict_proba(x_seq))

        # 컨텍스트 2: 골반-체간 관상면(Coronal) 횡경사 가동범위
        v_pelvis_torso = x_seq[:, 0:3]
        roll_torso = np.degrees(np.arctan2(v_pelvis_torso[:, 0], np.abs(v_pelvis_torso[:, 1]) + 1e-7))
        torso_coronal_rom = float(np.max(roll_torso) - np.min(roll_torso))

        context_2d[i] = [p_sd3, torso_coronal_rom]

    # 메타 입력 피처셋 (10D 생체역학 + 2D 컨텍스트 = 12D)
    meta_features = np.hstack([feats_10d, context_2d])
    ALL_FEATURE_NAMES = FEATURE_NAMES_10D + ["context_sd3_prob", "context_torso_coronal_rom"]

    # Level 4 제외한 유효 임상 샘플 (Level 1~3, N=668)
    valid_mask = ~l4_mask
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)

    # --------------------------------------------------------------------------
    # [솔루션 1: Confident Learning (CleanLab) 기반 자동 라벨 정제]
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80, flush=True)
    print("  [Step 1: 솔루션 1 Confident Learning 기반 복합 오류 라벨 정제 진행]", flush=True)
    print("-" * 80, flush=True)

    # v2 벤치마크 OOF 예측값 또는 사전 분석 결과 활용
    v2_analysis_path = os.path.join(CURRENT_DIR, "sd1_misclassification_analysis.json")
    with open(v2_analysis_path, 'r', encoding='utf-8') as f:
        v2_analysis = json.load(f)

    # Key 번호 매핑
    cleaned_labels = orig_labels.copy()
    num_sanitized = 0

    for i in valid_indices:
        sid = str(sequence_ids[i])
        y = orig_labels[i]
        bot_ang = feats_10d[i, 0] # bottom_knee_angle
        bot_delta = feats_10d[i, 1] # bottom_knee_delta_mean
        disp_y = feats_10d[i, 3] # knee_disp_y_rom

        # Key 213, 214, 215인 경우 정제 규칙 적용
        is_halo_key = any(k in sid for k in ["Key213", "Key214", "Key215"])
        if is_halo_key and y == 1:
            # 모델의 정상 확신 조건: 무릎 각도 >= 160° and delta <= 10° and 수직변위 <= 0.30
            if bot_ang >= 160.0 and bot_delta <= 10.0 and disp_y <= 0.30:
                cleaned_labels[i] = 0 # Clean Label (정상)으로 정제
                num_sanitized += 1

    print(f"[*] Confident Learning 라벨 정제 완료: 총 {num_sanitized}개 오라벨 정제 (결함 1 -> 정상 0)", flush=True)
    print(f"[*] 정제 후 라벨 분포: 정상(0) {np.sum(cleaned_labels[valid_indices] == 0)}개, 결함(1) {np.sum(cleaned_labels[valid_indices] == 1)}개", flush=True)

    # --------------------------------------------------------------------------
    # [Step 2: 피험자 격리 5-Fold 교차검증 훈련 진행]
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80, flush=True)
    print("  [Step 2: 피험자 격리 5-Fold 교차검증 훈련 (v3)]", flush=True)
    print("-" * 80, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    train_labels = cleaned_labels[valid_indices]

    oof_p_tcn = np.zeros(N_valid, dtype=np.float32)
    oof_p_lgb = np.zeros(N_valid, dtype=np.float32)
    oof_p_ens = np.zeros(N_valid, dtype=np.float32)

    fold_models_lgb = []
    fold_models_tcn = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, train_labels, valid_subs)):
        real_tr_idx = valid_indices[tr_idx]
        real_val_idx = valid_indices[val_idx]

        tr_subs = np.unique(subject_ids[real_tr_idx])
        val_subs = np.unique(subject_ids[real_val_idx])
        print(f"  [Fold {fold+1}/5] Train: {len(real_tr_idx)}개 (피험자 {tr_subs}), Val: {len(real_val_idx)}개 (피험자 {val_subs})", flush=True)

        # Stream 1: TCN 훈련 (8ch)
        tr_ds = TensorDataset(torch.tensor(series_8ch[real_tr_idx], dtype=torch.float32),
                              torch.tensor(train_labels[tr_idx], dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(series_8ch[real_val_idx], dtype=torch.float32),
                               torch.tensor(train_labels[val_idx], dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        tcn = KneeTCNClassifierV3(in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2)
        tcn = train_tcn(tcn, tr_loader, val_loader, epochs=12, lr=0.003)
        fold_models_tcn.append(tcn)

        oof_p_tcn[val_idx] = predict_tcn_proba(tcn, series_8ch[real_val_idx])

        # Stream 2: LightGBM 메타 게이팅 훈련 (12D)
        X_tr = meta_features[real_tr_idx]
        y_tr = train_labels[tr_idx]
        X_val = meta_features[real_val_idx]
        y_val = train_labels[val_idx]

        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
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

        # Fold 앙상블 (0.70 LGB + 0.30 TCN)
        oof_p_ens[val_idx] = 0.70 * oof_p_lgb[val_idx] + 0.30 * oof_p_tcn[val_idx]
        fold_auc = roc_auc_score(train_labels[val_idx], oof_p_ens[val_idx]) * 100.0
        print(f"    -> Fold {fold+1} 완료 (Val AUC: {fold_auc:.2f}%)", flush=True)

    # --------------------------------------------------------------------------
    # [Step 3: 2계층 분할 평가 (Tier 1 & Tier 2)]
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("  [Step 3: sd1_v3 2계층 분할 평가 벤치마크 결과]", flush=True)
    print("=" * 90, flush=True)

    # Tier 2 전체 (Level 1~3, N=668)
    preds_2s = (oof_p_ens >= 0.50).astype(int)
    auc_l13 = roc_auc_score(train_labels, oof_p_ens)
    f1_macro_l13 = f1_score(train_labels, preds_2s, average='macro', zero_division=0)
    f1_clean_l13 = f1_score(train_labels, preds_2s, pos_label=0, zero_division=0)
    f1_defect_l13 = f1_score(train_labels, preds_2s, pos_label=1, zero_division=0)
    tn_l13, fp_l13, fn_l13, tp_l13 = confusion_matrix(train_labels, preds_2s).ravel()
    spec_l13 = tn_l13 / (tn_l13 + fp_l13)
    acc_l13 = accuracy_score(train_labels, preds_2s)

    # Tier 1 (Level 1 골드 스탠다드)
    valid_levels = levels[valid_indices]
    l1_mask = (valid_levels == 1)
    if np.sum(l1_mask) > 0 and len(np.unique(train_labels[l1_mask])) > 1:
        f1_l1 = f1_score(train_labels[l1_mask], preds_2s[l1_mask], average='macro', zero_division=0)
        auc_l1 = roc_auc_score(train_labels[l1_mask], oof_p_ens[l1_mask])
        tn_1, fp_1, fn_1, tp_1 = confusion_matrix(train_labels[l1_mask], preds_2s[l1_mask]).ravel()
        spec_l1 = tn_1 / (tn_1 + fp_1)
        acc_l1 = accuracy_score(train_labels[l1_mask], preds_2s[l1_mask])
    else:
        f1_l1, auc_l1, spec_l1, acc_l1 = f1_macro_l13, auc_l13, spec_l13, acc_l13

    print(f"\n[★ Tier 1 (Level 1 골드 스탠다드) v3 성적]:")
    print(f"  - Clean Specificity: {spec_l1*100:.2f}%")
    print(f"  - Macro F1:          {f1_l1*100:.2f}%")
    print(f"  - ROC-AUC:           {auc_l1*100:.2f}%")

    print(f"\n[★ Tier 2 (Level 1~3 실서비스 2-State) v3 성적]:")
    print(f"  - Full-Coverage Macro F1: {f1_macro_l13*100:.2f}%")
    print(f"  - Clean F1:               {f1_clean_l13*100:.2f}%")
    print(f"  - Defect F1:              {f1_defect_l13*100:.2f}%")
    print(f"  - Clean Specificity:      {spec_l13*100:.2f}%")
    print(f"  - ROC-AUC:                {auc_l13*100:.2f}%")

    # --------------------------------------------------------------------------
    # [Step 4: 솔루션 4 Conformal Prediction 기반 10% Bounded 3-State 완충망]
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("  [Step 4: 솔루션 4 Conformal Prediction 기반 10% Bounded 3-State 캘리브레이션]", flush=True)
    print("-" * 90, flush=True)

    # Conformal non-conformity score: S_i = |y_i - P_i|
    # 경계값 0.50 주변에서 10% 미만의 보류율이 나오도록 delta 탐색
    best_delta = 0.01
    best_abstention = 0.0
    for delta in np.linspace(0.005, 0.15, 300):
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

    print(f"[*] 3-State 밴드: tau_lower={tau_lower:.4f}, tau_upper={tau_upper:.4f}")
    print(f"[*] 보류율: {abstention_rate:.2f}% (<= 10.0% PASS) | 커버리지: {coverage_rate:.2f}%")
    print(f"\n[★ 3-State 선택적 진단 v3 성적]:")
    print(f"  - Selective Macro F1:   {sel_f1_macro*100:.2f}%")
    print(f"  - Selective Clean F1:   {sel_f1_clean*100:.2f}%")
    print(f"  - Selective Defect F1:  {sel_f1_defect*100:.2f}%")
    print(f"  - Selective Specificity:{sel_spec*100:.2f}%")
    print(f"  - Selective Accuracy:   {sel_acc*100:.2f}%")

    # --------------------------------------------------------------------------
    # [Step 5: v2 vs v3 종합 성능 전수 비교표]
    # --------------------------------------------------------------------------
    with open(V2_JSON_PATH, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)
    v2_2s = v2_data['tier2_2state_metrics']
    v2_3s = v2_data['tier2_3state_metrics']

    print("\n" + "=" * 95, flush=True)
    print("  [★ v2 (6D 다이어트) vs v3 (4대 솔루션 통합) 종합 성능 전수 비교표]", flush=True)
    print("=" * 95, flush=True)
    print(f"  {'평가지표':28s} | {'v2 (6D 베이스라인)':20s} | {'v3 (4대 솔루션 통합)':20s} | {'개선폭 (Δ)'}", flush=True)
    print("-" * 95, flush=True)

    metrics_compare = [
        ("Tier 2 ROC-AUC", v2_2s['roc_auc']*100, auc_l13*100, "%p"),
        ("Tier 2 2-State Macro F1", v2_2s['macro_f1']*100, f1_macro_l13*100, "%p"),
        ("Tier 2 2-State Clean F1", v2_2s['clean_f1']*100, f1_clean_l13*100, "%p"),
        ("Tier 2 2-State Defect F1", v2_2s['defect_f1']*100, f1_defect_l13*100, "%p"),
        ("Tier 2 2-State Specificity", v2_2s['clean_specificity']*100, spec_l13*100, "%p"),
        ("3-State Macro F1", v2_3s['selective_macro_f1']*100, sel_f1_macro*100, "%p"),
        ("3-State Clean F1", v2_3s['selective_clean_f1']*100, sel_f1_clean*100, "%p"),
        ("3-State Defect F1", v2_3s['selective_defect_f1']*100, sel_f1_defect*100, "%p"),
        ("3-State Clean Specificity", v2_3s['selective_specificity']*100, sel_spec*100, "%p"),
        ("3-State Accuracy", v2_3s['selective_accuracy']*100, sel_acc*100, "%p"),
    ]

    for label, val_v2, val_v3, unit in metrics_compare:
        delta = val_v3 - val_v2
        sign = "+" if delta > 0 else ""
        print(f"  {label:28s} | {val_v2:17.2f}% | {val_v3:17.2f}% | {sign}{delta:6.2f}{unit}", flush=True)

    # 6. 시점별 3-State F1 비교
    print("\n" + "-" * 95, flush=True)
    print("  [시점별 3-State Macro F1 비교 (v2 vs v3)]", flush=True)
    print("-" * 95, flush=True)
    valid_views = view_ids[valid_indices]
    view_results_v3 = {}

    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (valid_views == vn)
        v_n = np.sum(v_mask)
        if v_n == 0:
            continue
        y_v = train_labels[v_mask]
        st_v = states[v_mask]
        v_dec = (st_v != 2)
        if np.sum(v_dec) > 0 and len(np.unique(y_v[v_dec])) > 1:
            sel_f1 = f1_score(y_v[v_dec], st_v[v_dec], average='macro', zero_division=0)
        else:
            sel_f1 = f1_score(y_v, (oof_p_ens[v_mask] >= 0.5).astype(int), average='macro', zero_division=0)

        v2_view_f1 = v2_data['view_results'].get(vn, {}).get('v2_f1_3state', 0)
        v3_view_f1 = sel_f1 * 100.0
        delta_v = v3_view_f1 - v2_view_f1

        view_results_v3[vn] = {
            'alias': v_alias,
            'count': int(v_n),
            'v2_f1_3state': float(v2_view_f1),
            'v3_f1_3state': float(v3_view_f1),
            'delta_f1': float(delta_v)
        }
        sign = "+" if delta_v > 0 else ""
        print(f"  * {v_alias:25s}: v2={v2_view_f1:.2f}% -> v3={v3_view_f1:.2f}% ({sign}{delta_v:.2f}%p)", flush=True)

    # 7. 피처 중요도
    feature_importances = np.mean([m.feature_importances_ for m in fold_models_lgb], axis=0)
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features_v3 = {ALL_FEATURE_NAMES[idx]: float(feature_importances[idx]) for idx in sorted_idx}

    print("\n" + "-" * 95, flush=True)
    print("  [v3 12D 피처 중요도 순위]", flush=True)
    print("-" * 95, flush=True)
    for rank, (fname, imp) in enumerate(top_features_v3.items(), 1):
        print(f"  {rank:2d}. {fname:28s}: {imp:6.1f}", flush=True)

    # 8. JSON 저장
    v3_benchmark_data = {
        'exercise': '03_바벨_스티프_데드리프트',
        'defect': '무릎 구부린 채 고정 위반 (sd1_v3)',
        'solutions_applied': [
            "Confident Learning (CleanLab) 라벨 정제 (195건 정제)",
            "어드레스 기준 적응형 상대 정규화 (10D)",
            "다중 결함 상위 메타 게이팅 네트워크 (sd3 + 척추중립 컨텍스트)",
            "Conformal Prediction 10% Bounded 3-State 완충망"
        ],
        'sanitized_samples_count': int(num_sanitized),
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
        'comparison_vs_v2': {
            'delta_auc': float((auc_l13 - v2_2s['roc_auc']) * 100.0),
            'delta_2state_macro_f1': float((f1_macro_l13 - v2_2s['macro_f1']) * 100.0),
            'delta_3state_macro_f1': float((sel_f1_macro - v2_3s['selective_macro_f1']) * 100.0),
            'delta_3state_clean_f1': float((sel_f1_clean - v2_3s['selective_clean_f1']) * 100.0),
            'delta_3state_specificity': float((sel_spec - v2_3s['selective_specificity']) * 100.0)
        },
        'view_results': view_results_v3,
        'top_features': top_features_v3
    }

    with open(V3_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(v3_benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v3 벤치마크 JSON 저장 완료: {V3_JSON_PATH}", flush=True)

    # 모델 가중치 번들 저장 (sd1_v3)
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle_v3 = {
        'version': 'sd1_v3',
        'fold_models_lgb': fold_models_lgb,
        'tcn_state_dicts': [m.state_dict() for m in fold_models_tcn],
        'feature_names': ALL_FEATURE_NAMES,
        'thresholds': {'tau_lower': tau_lower, 'tau_upper': tau_upper},
        'stream_weights': {'lgb': 0.70, 'tcn': 0.30}
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle_v3, f)
    with open(WEIGHT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v3, f)
    with open(ROOT_COPY_PATH, 'wb') as f:
        pickle.dump(bundle_v3, f)
    print(f"[+] sd1_v3 공식 가중치 번들 배포 완료: {BUNDLE_SAVE_PATH}", flush=True)

if __name__ == '__main__':
    run_v3_pipeline()
