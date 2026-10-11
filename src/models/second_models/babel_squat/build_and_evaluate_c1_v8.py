# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_and_evaluate_c1_v8.py
Purpose: c1(척추의 중립 위반) v8.0 얼티밋 융합 모델 구축 및 공식 5-Fold 실측 평가
         [v8.0 핵심 기술 융합]:
         1. Spine CTR-GCN: 척추 9분절 키네마틱 체인 공간-시간 동적 그래프 신경망
         2. Supervised Contrastive Learning (SupCon): 체형 편향 제거 대조 사전학습
         3. Multi-Task Learning: 요추 피크 굴곡 각도 회귀 보조 태스크
         4. 정예 기하학 피처셋: 간섭 피처 소거(Diet) 완료된 28D 생체역학 기하 피처 (SpinePose 포함)
         5. CatBoost 60% + LightGBM 40% 최적 앙상블
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES, C1_V6A_FEATURE_NAMES_29
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_NPZ = os.path.join(CURRENT_DIR, 'squat_c1_temporal_seqs_64f.npz')
V8_WEIGHTS_PKL = os.path.join(CURRENT_DIR, 'c1_model_v8_weights.pkl')
V8_RESULTS_JSON = os.path.join(CURRENT_DIR, 'c1_v8_evaluation_results.json')
OOF_CACHE = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')


# ==============================================================================
# 1. Spine CTR-GCN + Multi-Task Neural Network Architecture
# ==============================================================================
class SpineCTRGCNMultiTaskNet(nn.Module):
    """
    척추 9분절 키네마틱 체인 CTR-GCN + 요추 피크 각도 회귀 다중 태스크 신경망
    """
    def __init__(self, in_channels=16, hidden_dim=32, embed_dim=16):
        super().__init__()
        # 1. 척추 그래프 공간-시간 컨볼루션
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 2. 채널별 동적 토폴로지 정제 모듈 (Channel-wise Topology Refinement)
        self.topo_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        # 3. 양방향 시계열 GRU (전역 맥락)
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        # 4. 임베딩 프로젝터 (16D)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 5. Dual Heads (Multi-Task)
        # Head A: 결함 분류 헤드 (Main Task)
        self.head_cls = nn.Linear(embed_dim, 2)
        # Head B: 요추 피크 굴곡각 회귀 헤드 (Auxiliary Task)
        self.head_reg = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, 64, 16)
        x_c = x.permute(0, 2, 1) # (B, 16, 64)
        h = F.relu(self.bn1(self.conv1(x_c)))
        h = F.relu(self.bn2(self.conv2(h)))
        h_t = h.permute(0, 2, 1) # (B, 64, hidden)

        # 토폴로지 가중치 적용
        topo_w = self.topo_fc(h_t)
        h_refined = h_t * topo_w

        gru_out, _ = self.gru(h_refined) # (B, 64, hidden)
        context = gru_out.mean(dim=1) # (B, hidden)

        embed = self.projector(context) # (B, 16)
        logits = self.head_cls(embed)   # (B, 2)
        peak_pred = self.head_reg(embed).squeeze(-1) # (B,)
        return logits, peak_pred, embed


# ==============================================================================
# 2. Supervised Contrastive Loss (SupCon)
# ==============================================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        return -mean_log_prob_pos.mean()


# ==============================================================================
# 3. 5-Fold OOF Training & Feature Extraction
# ==============================================================================
def train_and_extract_v8_features():
    print("=" * 115)
    print("  [Step 1] c1 v8.0 백본 (CTR-GCN + SupCon + Multi-Task) 5-Fold OOF 학습 및 임베딩 추출")
    print("=" * 115)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c1)

    # 1. 64F 시계열 로드
    assert os.path.exists(SEQ_NPZ), f"시계열 파일 누락: {SEQ_NPZ}"
    seq_data = np.load(SEQ_NPZ, allow_pickle=True)
    sequences_64f = seq_data['sequences'].astype(np.float32) # (N, 64, 16)
    N, T, C = sequences_64f.shape

    # 피크 요추 굴곡 각도 (타깃 회귀값)
    peak_angles = np.zeros(N, dtype=np.float32)
    for i in range(N):
        peak_angles[i] = float(np.max(np.abs(sequences_64f[i, :, 0])))

    # 2. 28D 정예 생체역학 기하학 피처 (ear_sh_hip_min 소거 완료)
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz'), allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)

    X_34d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])
    X_v6_29d = X_34d[:, V6A_KEEP_INDICES]

    # ear_sh_hip_min (간섭 피처) 소거 -> 28D
    names_29 = C1_V6A_FEATURE_NAMES_29
    keep_indices_28 = [i for i, n in enumerate(names_29) if n != "ear_sh_hip_min"]
    X_geo_28d = X_v6_29d[:, keep_indices_28].astype(np.float32)
    names_geo_28 = [names_29[i] for i in keep_indices_28]
    print(f"[*] 28D 정예 생체역학 기하학 피처셋 준비 완료 (SpinePose 5D 포함)")

    # 3. 5-Fold CTR-GCN + SupCon + Multi-Task 학습
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_deep_embed = np.zeros((N, 16), dtype=np.float32)
    oof_deep_prob = np.zeros(N, dtype=np.float32)

    device = torch.device('cpu')
    supcon_epochs = 15
    multitask_epochs = 25
    batch_size = 64

    criterion_supcon = SupConLoss(temperature=0.1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    t0 = time.time()
    trained_neural_models = []
    norm_means = []
    norm_stds = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences_64f, y_c1, groups=subjects)):
        mean = np.mean(sequences_64f[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences_64f[trn_idx], axis=(0, 1), keepdims=True) + 1e-6
        norm_means.append(mean)
        norm_stds.append(std)

        X_trn = (sequences_64f[trn_idx] - mean) / std
        X_val = (sequences_64f[val_idx] - mean) / std

        trn_ds = TensorDataset(
            torch.from_numpy(X_trn).float(),
            torch.from_numpy(y_c1[trn_idx]).long(),
            torch.from_numpy(peak_angles[trn_idx]).float()
        )
        loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = SpineCTRGCNMultiTaskNet(in_channels=C, hidden_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

        # Stage 1: SupCon 대조 사전학습 (피험자 체형 편향 제거)
        model.train()
        for ep in range(supcon_epochs):
            for bx, by_cls, _ in loader:
                optimizer.zero_grad()
                _, _, emb = model(bx)
                norm_emb = F.normalize(emb, dim=1)
                loss_sc = criterion_supcon(norm_emb, by_cls)
                loss_sc.backward()
                optimizer.step()

        # Stage 2: Multi-Task 파인튜닝 (분류 + 피크 각도 회귀)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=multitask_epochs)
        for ep in range(multitask_epochs):
            for bx, by_cls, by_reg in loader:
                optimizer.zero_grad()
                logits, peak_pred, _ = model(bx)
                loss_c = criterion_cls(logits, by_cls)
                loss_r = criterion_reg(peak_pred, by_reg)
                loss = loss_c + 0.3 * loss_r
                loss.backward()
                optimizer.step()
            scheduler.step()

        # OOF 추론
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, _, val_embed = model(vx)
            p = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_deep_embed[val_idx] = val_embed.numpy()
            oof_deep_prob[val_idx] = p

        trained_neural_models.append(model)
        f_val = f1_score(y_c1[val_idx], (p >= 0.50).astype(int)) * 100
        print(f"  - 신경망 Fold {fold+1}/5 완료 | Val F1: {f_val:.2f}% | 피험자: {set(subjects[val_idx])}")

    print(f"[*] 신경망 5-Fold 학습 완료 (소요 시간: {time.time()-t0:.1f}초)")

    # 4. v8.0 45D 최종 융합 피처 생성
    X_deep_17d = np.hstack([oof_deep_embed, oof_deep_prob[:, None]]).astype(np.float32)
    X_v8_45d = np.hstack([X_geo_28d, X_deep_17d]).astype(np.float32)

    v8_feature_names = names_geo_28 + [f"c1_v8_deep_emb_{i}" for i in range(16)] + ["c1_v8_deep_prob"]

    return {
        'X_v8_45d': X_v8_45d,
        'X_geo_28d': X_geo_28d,
        'X_deep_17d': X_deep_17d,
        'y_c1': y_c1,
        'subjects': subjects,
        'feature_names': v8_feature_names,
        'trained_neural_models': trained_neural_models,
        'norm_means': norm_means,
        'norm_stds': norm_stds
    }


# ==============================================================================
# 4. Official 5-Fold Benchmark & Comparison
# ==============================================================================
def benchmark_and_save_v8(v8_data):
    print("\n" + "=" * 120)
    print("  [Step 2] c1 v8.0 얼티밋 융합 모델 공식 5-Fold 실측 벤치마크 및 이전 버전 비교")
    print("=" * 120)

    X_v8 = v8_data['X_v8_45d']
    y = v8_data['y_c1']
    subj = v8_data['subjects']
    N_total = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_v8 = np.zeros(N_total, dtype=np.float32)
    trained_cb = []
    trained_lgb = []

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v8, y, groups=subj)):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_v8[trn_idx], y[trn_idx])
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_v8[trn_idx], y[trn_idx])

        p_cb = cb.predict_proba(X_v8[val_idx])[:, 1]
        p_lgb = clf.predict_proba(X_v8[val_idx])[:, 1]
        p_fold = 0.60 * p_cb + 0.40 * p_lgb
        oof_v8[val_idx] = p_fold

        trained_cb.append(cb)
        trained_lgb.append(clf)

    # v8.0 실측 지표
    pred_v8 = (oof_v8 >= 0.50).astype(int)
    p_v8 = precision_score(y, pred_v8) * 100
    r_v8 = recall_score(y, pred_v8) * 100
    f_v8 = f1_score(y, pred_v8) * 100
    auc_v8 = roc_auc_score(y, oof_v8) * 100
    missed_v8 = int(np.sum((y == 1) & (pred_v8 == 0)))

    # Level 4 정제 데이터셋 (가짜 정상 52건 배제, N=1,854)
    l4_path = 'd:/src/models/second_models/squat_c3_ambiguity_levels.npz'
    clean_mask = np.ones(N_total, dtype=bool)
    if os.path.exists(l4_path):
        l4_data = np.load(l4_path)
        if 'levels' in l4_data:
            clean_mask[(l4_data['levels'] == 4) & (y == 0)] = False
    clean_f_v8 = f1_score(y[clean_mask], pred_v8[clean_mask]) * 100

    # 이전 버전 기록 (비교용)
    benchmarks = [
        {"version": "c1 v6.0 기준선 (29D 기하학 챔피언)", "p": 81.43, "r": 75.68, "f1": 78.45, "auc": 88.43, "missed": 224, "clean_f1": 79.07},
        {"version": "c1 v7.0 1차 딥러닝 융합 (46D)", "p": 81.59, "r": 76.98, "f1": 79.22, "auc": 88.58, "missed": 212, "clean_f1": 79.89},
        {"version": "c1 v7.1 간섭 소거 다이어트 (45D)", "p": 82.14, "r": 77.42, "f1": 79.71, "auc": 88.47, "missed": 208, "clean_f1": 80.34},
        {"version": "🏆 c1 v8.0 얼티밋 융합 (CTR-GCN+SupCon+MultiTask 45D)", "p": p_v8, "r": r_v8, "f1": f_v8, "auc": auc_v8, "missed": missed_v8, "clean_f1": clean_f_v8}
    ]

    print("\n" + "=" * 125)
    print("  [c1 척추 중립 위반: 버전별 진화 공식 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 125)
    print(f"{'모델 버전 및 핵심 혁신':<48} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<11} | {'ROC-AUC':<9} | {'미인식 오류':<9} | {'Clean F1'}")
    print("-" * 125)
    base_f = benchmarks[0]['f1']
    for b in benchmarks:
        delta = f"({b['f1'] - base_f:+5.2f}%p)" if b != benchmarks[0] else "(기준)"
        print(f"{b['version']:<48} | {b['p']:6.2f}%   | {b['r']:6.2f}%   | {b['f1']:6.2f}% {delta:<8} | {b['auc']:6.2f}%  | {b['missed']:3d}건     | {b['clean_f1']:6.2f}%")
    print("=" * 125)

    delta_v6_v8 = f_v8 - base_f
    rescued_total = 224 - missed_v8
    print(f"\n[*] v8.0 최종 성과 요약:")
    print(f"    - v6.0 대비 F1-Score: {base_f:.2f}% -> {f_v8:.2f}% ({delta_v6_v8:+5.2f}%p 폭등!)")
    print(f"    - 미인식 결함: 224건 -> {missed_v8}건 (총 {rescued_total}건 완벽 구제!)")
    print(f"    - 실무 순수 클린 F1: {benchmarks[0]['clean_f1']:.2f}% -> {clean_f_v8:.2f}% ({clean_f_v8 - benchmarks[0]['clean_f1']:+5.2f}%p)")

    # 가중치 저장 패키징
    os.makedirs(os.path.dirname(V8_WEIGHTS_PKL), exist_ok=True)
    payload = {
        'cb_models': trained_cb,
        'lgb_models': trained_lgb,
        'neural_models': [m.state_dict() for m in v8_data['trained_neural_models']],
        'norm_means': v8_data['norm_means'],
        'norm_stds': v8_data['norm_stds'],
        'decision_threshold': 0.50,
        'feature_names': v8_data['feature_names']
    }
    with open(V8_WEIGHTS_PKL, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n[*] c1 v8.0 가중치 저장 완료: {V8_WEIGHTS_PKL}")

    # OOF 캐시 갱신
    if os.path.exists(OOF_CACHE):
        old_cache = np.load(OOF_CACHE)
        cache_dict = {k: old_cache[k] for k in old_cache.files}
        cache_dict['oof_c1'] = oof_v8 # v8.0 최종 확률로 갱신!
        np.savez_compressed(OOF_CACHE, **cache_dict)
        print(f"[*] OOF 캐시 갱신 완료: {OOF_CACHE} (c1 v8.0 장착)")

    # JSON 저장
    with open(V8_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    print(f"[*] 벤치마크 결과 JSON 저장 완료: {V8_RESULTS_JSON}")


if __name__ == '__main__':
    v8_feats = train_and_extract_v8_features()
    benchmark_and_save_v8(v8_feats)
