# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_c1_savgol_only.py
Purpose: 순수 Zero-Phase Savitzky-Golay (Segment Constraint 배제) 단독 성능 평가
"""

import os
import sys
import time
import json
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq
from src.models.second_models.babel_squat.c1_model_v11_a_diet import extract_c1_v11_a_diet_features_from_seq
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
SAVGOL_ONLY_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_savgol_only_64f.npz"
FEATS_PATH = r"D:\Data\피처_데이터\squat_c1_v11_savgol_only_feats.npz"


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
    return {'f1': round(f1, 2), 'prec': round(prec, 2), 'rec': round(rec, 2), 'acc': round(acc, 2), 'auc': round(auc, 2)}


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
        val_logits = model(X_val_t.to(device)).squeeze(-1).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    return model, val_probs


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] c1 모델: Zero-Phase Savitzky-Golay 단독 (Segment 제약 배제) 평가")
    print("=" * 100)

    if os.path.exists(FEATS_PATH):
        d = np.load(FEATS_PATH, allow_pickle=True)
        X_38d, X_27d, X_t24 = d['X_38d'], d['X_27d'], d['X_t24']
    else:
        spine_sg = np.load(SAVGOL_ONLY_CACHE_PATH, allow_pickle=True)['spine_9_64f_vision']
        master = np.load(MASTER_PATH, allow_pickle=True)
        master_sids = master['sequence_ids']
        N = len(master_sids)

        X_38d = np.zeros((N, 38), dtype=np.float32)
        X_27d = np.zeros((N, 27), dtype=np.float32)
        X_t24 = np.zeros((N, 64, 24), dtype=np.float32)

        t0 = time.time()
        for i, sid in enumerate(master_sids):
            mp_path = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
            mp_d = np.load(mp_path, allow_pickle=True)
            kps_33 = mp_d['kps_33_3d_64f']
            s9 = spine_sg[i]

            X_38d[i] = extract_c1_v11_a_features_from_seq(kps_33, s9)
            X_27d[i] = extract_c1_v11_a_diet_features_from_seq(kps_33, s9)
            X_t24[i] = extract_temporal_24d(kps_33, s9)

        np.savez_compressed(FEATS_PATH, X_38d=X_38d, X_27d=X_27d, X_t24=X_t24, sequence_ids=master_sids)
        print(f"피처셋 추출 완료: {time.time()-t0:.1f}초")

    master = np.load(MASTER_PATH, allow_pickle=True)
    y = master['Y_defects_golden'][:, 0].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_38d, y, groups=subjects), 1):
        cb_38 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_38.fit(X_38d[tr_idx], y[tr_idx])
        lgb_38 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_38.fit(X_38d[tr_idx], y[tr_idx])
        p_38 = 0.60 * cb_38.predict_proba(X_38d[val_idx])[:, 1] + 0.40 * lgb_38.predict_proba(X_38d[val_idx])[:, 1]

        cb_27 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_27.fit(X_27d[tr_idx], y[tr_idx])
        lgb_27 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_27.fit(X_27d[tr_idx], y[tr_idx])
        p_27 = 0.60 * cb_27.predict_proba(X_27d[val_idx])[:, 1] + 0.40 * lgb_27.predict_proba(X_27d[val_idx])[:, 1]

        tcn_model, p_tcn = train_tcn_fast(X_t24[tr_idx], y[tr_idx], X_t24[val_idx], in_channels=24, epochs=15, batch_size=64)

        p_ens = 0.10 * p_38 + 0.40 * p_27 + 0.50 * p_tcn
        oof_probs[val_idx] = p_ens
        m_f = compute_metrics(y[val_idx], p_ens)
        print(f"  Fold {fold}/5: F1={m_f['f1']:.2f}%, AUC={m_f['auc']:.2f}%, Acc={m_f['acc']:.2f}%")

    overall = compute_metrics(y, oof_probs)
    print("=" * 100)
    print(f"  ★ Savitzky-Golay 단독 전체 결과: F1={overall['f1']:.2f}%, AUC={overall['auc']:.2f}%, Acc={overall['acc']:.2f}%, Prec={overall['prec']:.2f}%, Rec={overall['rec']:.2f}%")
    print("=" * 100)

    # 5대 시점별
    for vn, alias in [('view1', '시점 E (좌측 후방)'), ('view2', '시점 B (우측 전방)'), ('view3', '시점 C (정면)'), ('view4', '시점 D (좌측 전방)'), ('view5', '시점 A (우측 후방)')]:
        mask_v = (views == vn)
        m_v = compute_metrics(y[mask_v], oof_probs[mask_v])
        print(f"    - {alias:25s}: F1={m_v['f1']:5.2f}% | AUC={m_v['auc']:5.2f}% | Acc={m_v['acc']:5.2f}%")


if __name__ == '__main__':
    main()
