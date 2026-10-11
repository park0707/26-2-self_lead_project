# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: train_and_save_c3_v4.py
Purpose: c3 v4.0 5-Fold 가중치 패키징 저장 및 oof_submodels_cache.npz 갱신
"""

import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v4 import C3TemporalEmbeddingNet, BabelSquatC3ModelV4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_NPZ = os.path.join(CURRENT_DIR, 'squat_c3_temporal_seqs_64f.npz')
WEIGHTS_PKL = os.path.join(CURRENT_DIR, 'c3_model_v4_weights.pkl')
OOF_CACHE = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')


def main():
    print("=" * 100)
    print("  [Step 1] c3 v4.0 5-Fold 딥러닝 모델 학습 및 가중치 저장")
    print("=" * 100)

    # 1. 시계열 및 데이터셋 로드
    assert os.path.exists(SEQ_NPZ), f"시계열 파일 누락: {SEQ_NPZ}"
    seq_data = np.load(SEQ_NPZ, allow_pickle=True)
    sequences = seq_data['sequences'].astype(np.float32) # (N, 64, 16)
    N, T, C = sequences.shape

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    epochs = 35
    batch_size = 64
    lr = 0.003
    device = torch.device('cpu')

    trained_models = []
    norm_means = []
    norm_stds = []
    oof_c3_v4 = np.zeros(N, dtype=np.float32)

    torch.manual_seed(42)
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences, y_c3, groups=subjects)):
        mean = np.mean(sequences[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences[trn_idx], axis=(0, 1), keepdims=True) + 1e-6

        norm_means.append(mean)
        norm_stds.append(std)

        X_trn = (sequences[trn_idx] - mean) / std
        X_val = (sequences[val_idx] - mean) / std
        y_trn = y_c3[trn_idx]
        y_val = y_c3[val_idx]

        neg_count = np.sum(y_trn == 0)
        pos_count = np.sum(y_trn == 1)
        w0 = pos_count / (neg_count + pos_count)
        w1 = neg_count / (neg_count + pos_count)
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C3TemporalEmbeddingNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for epoch in range(epochs):
            for bx, by in trn_loader:
                optimizer.zero_grad()
                logits, _, _ = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, _, _ = model(vx)
            probs = F.softmax(val_logits, dim=-1)[:, 1].numpy()
            oof_c3_v4[val_idx] = probs

        val_pred = (probs >= 0.50).astype(int)
        f_val = f1_score(y_val, val_pred) * 100
        print(f"  - Fold {fold+1}/5 완료 | Val F1: {f_val:.2f}% | 피험자: {set(subjects[val_idx])}")
        trained_models.append(model)

    pred_v4 = (oof_c3_v4 >= 0.50).astype(int)
    p_tot = precision_score(y_c3, pred_v4) * 100
    r_tot = recall_score(y_c3, pred_v4) * 100
    f_tot = f1_score(y_c3, pred_v4) * 100
    auc_tot = roc_auc_score(y_c3, oof_c3_v4) * 100
    missed_tot = int(np.sum((y_c3 == 1) & (pred_v4 == 0)))

    print("\n" + "=" * 100)
    print(f"[*] c3 v4.0 5-Fold 전체 실측 성능:")
    print(f"    - Precision: {p_tot:.2f}% | Recall: {r_tot:.2f}% | F1-Score: {f_tot:.2f}% | ROC-AUC: {auc_tot:.2f}% | 미인식: {missed_tot}건")
    print(f"    - 소요 시간: {time.time()-t0:.1f}초")
    print("=" * 100)

    # 2. 가중치 저장
    os.makedirs(os.path.dirname(WEIGHTS_PKL), exist_ok=True)
    state_dicts = [m.state_dict() for m in trained_models]
    payload = {
        'state_dicts': state_dicts,
        'norm_means': norm_means,
        'norm_stds': norm_stds,
        'decision_threshold': 0.50
    }
    with open(WEIGHTS_PKL, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[*] 가중치 저장 완료: {WEIGHTS_PKL}")

    # 3. oof_submodels_cache.npz 갱신
    if os.path.exists(OOF_CACHE):
        old_cache = np.load(OOF_CACHE)
        cache_dict = {k: old_cache[k] for k in old_cache.files}
        cache_dict['oof_c3_v3'] = cache_dict['oof_c3'] # 기존 v3 백업
        cache_dict['oof_c3'] = oof_c3_v4               # v4로 갱신!
        np.savez_compressed(OOF_CACHE, **cache_dict)
        print(f"[*] OOF 캐시 갱신 완료: {OOF_CACHE} (기존 v3 백업 및 v4 장착)")
    else:
        np.savez_compressed(OOF_CACHE, oof_c3=oof_c3_v4)
        print(f"[*] OOF 캐시 신규 저장: {OOF_CACHE}")


if __name__ == '__main__':
    main()
