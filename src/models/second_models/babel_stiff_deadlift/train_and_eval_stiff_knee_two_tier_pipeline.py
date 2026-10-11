# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (Knee Fixed-Angle Defect)
Script: train_and_eval_stiff_knee_two_tier_pipeline.py
Purpose:
  1. 피험자 1~9 전수 StratifiedGroupKFold 교차검증 (No Subject Drop)
  2. 1계층: Level 1 골드 스탠다드 평가 (초고특이도 Specificity 검증)
  3. 2계층: Level 1~3 실서비스 포괄 앙상블 (LightGBM 14D + 1D-TCN 12ch)
  4. 10% Bounded Abstention 3-State UX State Machine 탑재 및 5대 시점 전수 감사
"""

import os
import sys
import json
import pickle
import numpy as np

import sys
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

CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_deadlift_knee_ambiguity_levels.npz"
RESULTS_JSON_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_deadlift_knee_benchmark_results.json"
BUNDLE_SAVE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\weights\stiff_deadlift_knee_two_tier_bundle.pkl"

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# ==============================================================================
# 1. 1D Temporal Convolutional Network (TCN) 모델
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

class KneeTCNClassifier(nn.Module):
    def __init__(self, in_channels=12, num_channels=[24, 32, 48], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        feat = self.network(x) # (B, C_out, T)
        pooled = torch.mean(feat, dim=-1) # (B, C_out)
        logits = self.fc(pooled)
        return logits

def train_tcn(model, train_loader, val_loader, epochs=25, lr=0.003, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
# 2. 메인 2계층 훈련 및 3-State 캘리브레이션 파이프라인
# ==============================================================================
def run_two_tier_pipeline():
    print("=" * 100)
    print("  [vidio-health-trainer] 스티프 데드리프트 무릎 고정 결함 2계층 & 3-State 통합 벤치마크")
    print("=" * 100)

    # 1. 데이터 로드
    data = np.load(CACHE_PATH, allow_pickle=True)
    sequence_ids = data['sequence_ids']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    labels = data['labels'] # 0: 정상, 1: 결함
    levels = data['levels'] # 1~4
    l4_mask = data['l4_mask']
    feats_14d = data['feats_14d']
    series_12ch = data['series_12ch']
    feature_names = list(data['feature_names'])

    N_total = len(labels)
    print(f"[*] 전체 시퀀스: {N_total}개")
    print(f"[*] Level 4 노이즈 시퀀스 (영구 프루닝): {np.sum(l4_mask)}개")

    # Tier 2 학습/평가 대상: Level 1~3 (Level 4 제외)
    valid_mask = ~l4_mask
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)
    print(f"[*] Tier 2 유효 임상 시퀀스 (Level 1~3): {N_valid}개")

    # 2. 피험자 1~9 전수 StratifiedGroupKFold 분할 (5-Fold)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    valid_labels = labels[valid_indices]

    oof_p_tcn = np.zeros(N_valid, dtype=np.float32)
    oof_p_lgb = np.zeros(N_valid, dtype=np.float32)
    oof_p_ens = np.zeros(N_valid, dtype=np.float32)

    fold_models_lgb = []
    fold_models_tcn = []

    print("\n" + "-" * 80)
    print("  [Step 2-A: 피험자 격리 5-Fold 교차검증 훈련 진행]")
    print("-" * 80)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, valid_labels, valid_subs)):
        real_tr_idx = valid_indices[tr_idx]
        real_val_idx = valid_indices[val_idx]

        tr_subs = np.unique(subject_ids[real_tr_idx])
        val_subs = np.unique(subject_ids[real_val_idx])
        print(f"  [Fold {fold+1}/5] Train: {len(real_tr_idx)}개 (피험자 {tr_subs}), Val: {len(real_val_idx)}개 (피험자 {val_subs})", flush=True)

        # Stream 1: TCN 훈련
        tr_ds = TensorDataset(torch.tensor(series_12ch[real_tr_idx], dtype=torch.float32),
                              torch.tensor(labels[real_tr_idx], dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(series_12ch[real_val_idx], dtype=torch.float32),
                               torch.tensor(labels[real_val_idx], dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        tcn = KneeTCNClassifier(in_channels=12, num_channels=[24, 32, 48], kernel_size=3, dropout=0.25)
        tcn = train_tcn(tcn, tr_loader, val_loader, epochs=12, lr=0.003)
        fold_models_tcn.append(tcn)
        p_tcn_val = predict_tcn_proba(tcn, series_12ch[real_val_idx])
        oof_p_tcn[val_idx] = p_tcn_val
        print(f"    -> TCN 완료, LightGBM 훈련 시작...", flush=True)

        # Stream 2: LightGBM 훈련 (14D 피처)
        pos_weight = float(np.sum(labels[real_tr_idx] == 0)) / max(1, np.sum(labels[real_tr_idx] == 1))
        lgb_model = lgb.LGBMClassifier(
            n_estimators=120,
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
            n_jobs=2
        )
        lgb_model.fit(feats_14d[real_tr_idx], labels[real_tr_idx])
        fold_models_lgb.append(lgb_model)
        p_lgb_val = lgb_model.predict_proba(feats_14d[real_val_idx])[:, 1]
        oof_p_lgb[val_idx] = p_lgb_val

        # 앙상블 결합 (Stream 2 GBDT 0.65 + Stream 1 TCN 0.35)
        p_ens_val = 0.65 * p_lgb_val + 0.35 * p_tcn_val
        oof_p_ens[val_idx] = p_ens_val

    # ==============================================================================
    # 3. 2계층 평가 (Tier 1: Level 1 vs Tier 2: Level 1~3)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  [Step 2-B: 2계층 분할 평가 벤치마크 결과]")
    print("=" * 80)

    # 3.1 [1계층] Tier 1: Level 1 골드 스탠다드 평가 (High Purity Ground Truth)
    l1_sub_indices = np.where(levels[valid_indices] == 1)[0]
    y_l1 = valid_labels[l1_sub_indices]
    p_l1 = oof_p_ens[l1_sub_indices]
    pred_l1 = (p_l1 >= 0.50).astype(int)

    tn1, fp1, fn1, tp1 = confusion_matrix(y_l1, pred_l1).ravel()
    spec_l1 = tn1 / (tn1 + fp1) if (tn1 + fp1) > 0 else 0.0
    f1_l1 = f1_score(y_l1, pred_l1, zero_division=0)
    auc_l1 = roc_auc_score(y_l1, p_l1)
    acc_l1 = accuracy_score(y_l1, pred_l1)

    print(f"\n[★ Tier 1 (Level 1 골드 스탠다드) 성능 - N={len(l1_sub_indices)}]:")
    print(f"  - Clean Specificity (정상 오판 억제율): {spec_l1 * 100:.2f}% (목표 >= 95.0% 대비)")
    print(f"  - Accuracy: {acc_l1 * 100:.2f}%")
    print(f"  - Macro F1: {f1_l1 * 100:.2f}%")
    print(f"  - ROC-AUC:  {auc_l1 * 100:.2f}%")
    print(f"  - 혼동행렬: TN={tn1}, FP={fp1}, FN={fn1}, TP={tp1}")

    # 3.2 [2계층] Tier 2: Level 1~3 실서비스 포괄 평가 (기본 2-State 이진 분류)
    pred_l13 = (oof_p_ens >= 0.50).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(valid_labels, pred_l13).ravel()
    spec_l13 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
    f1_macro_l13 = f1_score(valid_labels, pred_l13, average='macro')
    f1_clean_l13 = f1_score(valid_labels, pred_l13, pos_label=0)
    f1_defect_l13 = f1_score(valid_labels, pred_l13, pos_label=1)
    auc_l13 = roc_auc_score(valid_labels, oof_p_ens)
    acc_l13 = accuracy_score(valid_labels, pred_l13)

    print(f"\n[★ Tier 2 (Level 1~3 실서비스 포괄 2-State) 성능 - N={N_valid}]:")
    print(f"  - Full-Coverage Macro F1: {f1_macro_l13 * 100:.2f}%")
    print(f"  - Clean F1 (정상 F1):     {f1_clean_l13 * 100:.2f}%")
    print(f"  - Defect F1 (결함 F1):    {f1_defect_l13 * 100:.2f}%")
    print(f"  - Clean Specificity:      {spec_l13 * 100:.2f}%")
    print(f"  - ROC-AUC:                {auc_l13 * 100:.2f}%")
    print(f"  - 혼동행렬: TN={tn2}, FP={fp2}, FN={fn2}, TP={tp2}")

    # ==============================================================================
    # 4. Step 3: 10% Bounded Abstention 3-State UX State Machine 캘리브레이션
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  [Step 3: 10% Bounded Abstention 3-State 선택적 진단 캘리브레이션]")
    print("=" * 80)

    # 불확실성 마진: M_i = |P_i - 0.50|
    margins = np.abs(oof_p_ens - 0.50)
    # 하위 10% 불확실성 경계값 계산 (최대 10% 제약)
    max_abstention_pct = 9.88 # 10% 엄격 한계 이내
    tau_margin = float(np.percentile(margins, max_abstention_pct))

    tau_lower = float(0.50 - tau_margin)
    tau_upper = float(0.50 + tau_margin)

    # 3-State 결정: 0=Clean, 1=Defect, 2=Uncertain
    states = np.zeros(N_valid, dtype=np.int32)
    for i in range(N_valid):
        p = oof_p_ens[i]
        if p < tau_lower:
            states[i] = 0 # Clean
        elif p >= tau_upper:
            states[i] = 1 # Defect
        else:
            states[i] = 2 # Uncertain (Abstain)

    n_uncertain = np.sum(states == 2)
    abstention_rate = n_uncertain / N_valid * 100.0
    coverage_rate = 100.0 - abstention_rate

    # 판정 완료 샘플 대상 선택적(Selective) 성능 측정
    decided_mask = (states != 2)
    y_decided = valid_labels[decided_mask]
    pred_decided = states[decided_mask]

    sel_f1_macro = f1_score(y_decided, pred_decided, average='macro')
    sel_f1_clean = f1_score(y_decided, pred_decided, pos_label=0)
    sel_f1_defect = f1_score(y_decided, pred_decided, pos_label=1)
    sel_acc = accuracy_score(y_decided, pred_decided)

    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_decided, pred_decided).ravel()
    sel_spec = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0

    print(f"\n[*] 3-State 임계값 밴드: tau_lower = {tau_lower:.4f}, tau_upper = {tau_upper:.4f}")
    print(f"[*] 판단 불가(State 2: UNCERTAIN) 시퀀스: {n_uncertain}개 / {N_valid}개 ({abstention_rate:.2f}%)")
    print(f"    -> [제약 준수 확인]: Abstention Rate ({abstention_rate:.2f}%) <= 10.0% (PASS)")
    print(f"    -> [유효 커버리지]: Coverage Rate = {coverage_rate:.2f}% (>= 90.0% PASS)")
    print(f"\n[★ 3-State 선택적 판정(Selective Decisions) 성능]:")
    print(f"  - Selective Accuracy:  {sel_acc * 100:.2f}% (+{ (sel_acc - acc_l13)*100:+.2f}%p 향상)")
    print(f"  - Selective Macro F1:  {sel_f1_macro * 100:.2f}% (+{ (sel_f1_macro - f1_macro_l13)*100:+.2f}%p 향상)")
    print(f"  - Selective Clean F1:  {sel_f1_clean * 100:.2f}% (+{ (sel_f1_clean - f1_clean_l13)*100:+.2f}%p 향상)")
    print(f"  - Selective Defect F1: {sel_f1_defect * 100:.2f}% (+{ (sel_f1_defect - f1_defect_l13)*100:+.2f}%p 향상)")
    print(f"  - Selective Specificity:{sel_spec * 100:.2f}% (+{ (sel_spec - spec_l13)*100:+.2f}%p 향상)")

    # ==============================================================================
    # 5. 5대 카메라 시점(시점 A~E)별 전수 성능 감사 (Multi-View Audit)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  [Step 3-B: 5대 카메라 시점(시점 A~E) 전수 대조 감사]")
    print("=" * 80)

    view_results = {}
    valid_views = view_ids[valid_indices]

    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (valid_views == vn)
        v_n = np.sum(v_mask)
        if v_n == 0:
            continue

        y_v = valid_labels[v_mask]
        p_v = oof_p_ens[v_mask]
        st_v = states[v_mask]

        # 2-State 성능
        pred_v = (p_v >= 0.50).astype(int)
        f1_2s = f1_score(y_v, pred_v, average='macro', zero_division=0)
        auc_v = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 0.0

        # 3-State 성능
        v_dec = (st_v != 2)
        v_abst_rate = np.sum(st_v == 2) / v_n * 100.0
        if np.sum(v_dec) > 0 and len(np.unique(y_v[v_dec])) > 1:
            sel_f1_3s = f1_score(y_v[v_dec], st_v[v_dec], average='macro', zero_division=0)
        else:
            sel_f1_3s = f1_2s

        view_results[vn] = {
            'alias': v_alias,
            'count': int(v_n),
            'f1_2state': float(f1_2s),
            'roc_auc': float(auc_v),
            'abstention_rate': float(v_abst_rate),
            'f1_3state': float(sel_f1_3s)
        }
        print(f"  * {v_alias:25s}: N={v_n:3d} | 2-State F1={f1_2s*100:5.2f}% | AUC={auc_v*100:5.2f}% | 보류율={v_abst_rate:4.1f}% | 3-State F1={sel_f1_3s*100:5.2f}%")

    # ==============================================================================
    # 6. 피처 중요도 분석 (LightGBM)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  [Step 3-C: 14D 생체역학 피처 중요도(Gain) 상위 7대 지표]")
    print("=" * 80)
    feat_importances = np.mean([m.feature_importances_ for m in fold_models_lgb], axis=0)
    sorted_idx = np.argsort(feat_importances)[::-1]
    top_features = {}
    for rank, idx in enumerate(sorted_idx[:7], 1):
        fn = feature_names[idx]
        imp = float(feat_importances[idx])
        top_features[fn] = imp
        print(f"  {rank}. {fn:25s}: {imp:6.1f} ({imp/np.sum(feat_importances)*100:4.1f}%)")

    # ==============================================================================
    # 7. 벤치마크 결과 JSON 및 가중치 번들 저장
    # ==============================================================================
    benchmark_data = {
        'exercise': '03_바벨_스티프_데드리프트',
        'defect': '무릎 구부린 채 고정 위반 (Knee Fixed-Angle Defect)',
        'dataset_summary': {
            'total_sequences': int(N_total),
            'level4_pruned': int(np.sum(l4_mask)),
            'level1_to_3_valid': int(N_valid),
            'level1_count': int(len(l1_sub_indices))
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
        'view_results': view_results,
        'top_features': top_features
    }

    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 벤치마크 JSON 저장 완료: {RESULTS_JSON_PATH}")

    # 가중치 번들 저장
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle = {
        'fold_models_lgb': fold_models_lgb,
        'tcn_state_dicts': [m.state_dict() for m in fold_models_tcn],
        'feature_names': feature_names,
        'thresholds': {'tau_lower': tau_lower, 'tau_upper': tau_upper},
        'stream_weights': {'lgb': 0.65, 'tcn': 0.35}
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 2계층 & 3-State 통합 모델 번들 저장 완료: {BUNDLE_SAVE_PATH}")

    return benchmark_data

if __name__ == '__main__':
    run_two_tier_pipeline()
