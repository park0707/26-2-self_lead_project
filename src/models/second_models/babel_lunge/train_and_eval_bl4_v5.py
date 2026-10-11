# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)  /  Defect: c4 (척추의 중립 위반)
Model: bl4_v5  (bl4_v4b 계승 + SpinePose 9분절 척추 피처 31D + 시계열 6ch 추가)
Script: train_and_eval_bl4_v5.py

[조사 기반 개선 기법]
  - Confident Learning (Northcutt et al., JAIR 2021): 학습 폴드 내부 피험자-그룹 교차예측으로
    라벨 오류 의심 샘플을 탐지 -> 제외하지 않고 '소프트 하향 가중'(AGENTS.md 전수 사용 원칙 준수)
  - Symmetric Cross Entropy (Wang et al., ICCV 2019): TCN 손실에 RCE 항 추가
  - 피험자 균형 가중(1/sqrt(n_subject)): 피험자1~3(각 144~206건) 지배 완화 (도메인 일반화)

[공정 비교 프로토콜]
  - v3 원본 스크립트는 TCN 에포크를 '검증 폴드 F1'로 선택(누수) -> 본 스크립트는 모든 모드에서
    고정 에포크(마지막 가중치) 사용. 'ref' 모드가 동일 프로토콜의 v3 재측정치.
  - 라벨/샘플은 평가에서 전혀 제외하지 않음 (원본 라벨 전수 평가).
  - AI-Hub JSON 라벨은 정답 타깃(y)으로만 사용, 입력 피처는 순수 MediaPipe 비전 피처.

Usage: python train_and_eval_bl4_v5.py <mode>   mode in {sp_ref, sp_cl}
"""

import os
import sys
import time
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")

# v3 안정 버전 모델 구조 계승 (Copy-and-Refine)
from train_and_eval_bl4_v3 import MultiScaleSETCN  # noqa: E402

CACHE_V3_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
SP_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_spinepose_64f.npz"
TCN_EPOCHS = 30
SEED = 42


# -------------------------------------------------------------
# 1. 라벨 노이즈 / 피험자 균형 가중치 (학습 폴드 내부에서만 계산 -> 누수 없음)
# -------------------------------------------------------------
def confident_learning_weights(X, y, groups, low_weight=0.25, n_splits=4, seed=0):
    """학습 폴드 내부 그룹 교차예측 확률로 라벨 오류 의심 샘플을 소프트 하향 가중.

    Confident Learning: 클래스별 자기신뢰 임계 t_c = 클래스 c로 라벨된 샘플의 평균 p(c).
    '라벨 c인데 p(c) < t_c 이고 반대 클래스 p >= t_반대' 이면 의심(suspect).
    """
    oof = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                               num_leaves=7, class_weight='balanced',
                               random_state=seed, verbose=-1)
        m.fit(sc.transform(X[tr]), y[tr])
        oof[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
    p_def = oof
    p_clean = 1.0 - oof
    t_def = p_def[y == 1].mean()
    t_clean = p_clean[y == 0].mean()
    suspect = ((y == 1) & (p_def < t_def) & (p_clean >= t_clean)) | \
              ((y == 0) & (p_clean < t_clean) & (p_def >= t_def))
    w = np.ones(len(y))
    w[suspect] = low_weight
    return w, suspect


def subject_balance_weights(groups):
    """피험자별 표본 수 역제곱근 가중 (평균 1로 정규화)."""
    u, cnt = np.unique(groups, return_counts=True)
    inv = {s: 1.0 / np.sqrt(c) for s, c in zip(u, cnt)}
    w = np.array([inv[g] for g in groups], dtype=float)
    return w / w.mean()


# -------------------------------------------------------------
# 2. 노이즈 강건 손실 (Focal + RCE, 샘플 가중)
# -------------------------------------------------------------
class WeightedFocalSCE(nn.Module):
    def __init__(self, alpha=0.45, gamma=1.5, beta=0.0, rce_a=4.0):
        super().__init__()
        self.alpha, self.gamma, self.beta, self.rce_a = alpha, gamma, beta, rce_a

    def forward(self, logits, targets, weights):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * ce
        loss = focal
        if self.beta > 0:
            loss = loss + self.beta * self.rce_a * (1.0 - pt)  # RCE: -sum p log(q), q=onehot clamp
        return (loss * weights).sum() / weights.sum()


def load_sp_features(d):
    """SpinePose 31D 피처 + 6ch 시계열을 v3 캐시 순서에 맞춰 생성 (이름 기반 인터페이스 사용)."""
    from src.features.lunge_c4_spinepose_features import (
        extract_lunge_c4_spinepose_features, extract_lunge_c4_spinepose_temporal,
        SPINEPOSE_FEATURE_NAMES)
    sp = np.load(SP_CACHE_PATH, allow_pickle=True)
    assert list(sp['clip_ids']) == [str(c) for c in d['clip_ids']], "SpinePose 캐시 순서 불일치"
    valid = sp['valid']
    assert valid.all(), f"SpinePose 누락 {int((~valid).sum())}건"
    F = np.stack([extract_lunge_c4_spinepose_features(sp['spine3d'][i], sp['spine2d'][i]) for i in range(len(valid))])
    Tt = np.stack([extract_lunge_c4_spinepose_temporal(sp['spine3d'][i]) for i in range(len(valid))])
    assert np.isfinite(F).all() and np.isfinite(Tt).all(), "SpinePose 피처 NaN/Inf"
    return F, Tt, SPINEPOSE_FEATURE_NAMES


def run(mode: str):
    # mode: sp_ref (SpinePose 추가, 강건학습 없음) / sp_cl (SpinePose + CL + SCE = v4b 레시피)
    use_cl = mode in ('sp_cl',)
    use_sb = False
    beta = 0.3 if use_cl else 0.0

    d = np.load(CACHE_V3_PATH, allow_pickle=True)
    F_sp, T_sp, sp_names = load_sp_features(d)
    X = np.hstack([d['X_24d'], F_sp]); T = np.concatenate([d['X_temp_14d'], T_sp], axis=2)
    y = d['y']; subj = d['subjects']
    N = len(y)
    device = torch.device('cpu')
    print(f"[mode={mode}] N={N}, X={X.shape}, T={T.shape}, CL={use_cl}, SCE_beta={beta}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {k: np.zeros(N, dtype=np.float32) for k in ('tcn', 'lgb', 'xgb', 'cat')}
    models = {'tcn': [], 'lgb': [], 'xgb': [], 'cat': [], 'scaler': []}
    n_suspect = []
    t0 = time.time()

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=subj), 1):
        w = np.ones(len(tr))
        if use_cl:
            w_cl, sus = confident_learning_weights(X[tr], y[tr], subj[tr], seed=SEED + fold)
            w = w * w_cl
            n_suspect.append(int(sus.sum()))
        if use_sb:
            w = w * subject_balance_weights(subj[tr])
        w = w / w.mean()

        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        models['scaler'].append(sc)
        ytr = y[tr]

        # --- TCN: 고정 에포크, 마지막 가중치 (검증 누수 제거) ---
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        tcn = MultiScaleSETCN(in_features=T.shape[2], num_classes=2).to(device)
        crit = WeightedFocalSCE(beta=beta)
        opt = torch.optim.AdamW(tcn.parameters(), lr=1e-3, weight_decay=1e-4)
        ds = TensorDataset(torch.from_numpy(T[tr]).float(), torch.from_numpy(ytr).long(),
                           torch.from_numpy(w).float())
        ld = DataLoader(ds, batch_size=32, shuffle=True)
        for _ in range(TCN_EPOCHS):
            tcn.train()
            for bx, by, bw in ld:
                opt.zero_grad()
                crit(tcn(bx), by, bw).backward()
                opt.step()
        tcn.eval()
        with torch.no_grad():
            oof['tcn'][va] = F.softmax(tcn(torch.from_numpy(T[va]).float()), -1)[:, 1].numpy()
        models['tcn'].append(tcn.cpu())

        # --- GBDT 3종 (v3 하이퍼파라미터 그대로, sample_weight만 추가) ---
        m = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=15,
                               subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=0.3,
                               class_weight='balanced', random_state=42 + fold, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=w)
        oof['lgb'][va] = m.predict_proba(Xva)[:, 1]; models['lgb'].append(m)

        pos_w = float(np.sum(ytr == 0)) / float(max(1, np.sum(ytr == 1)))
        m = xgb.XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, subsample=0.85,
                              colsample_bytree=0.80, gamma=0.15, scale_pos_weight=pos_w,
                              eval_metric='logloss', random_state=42 + fold)
        m.fit(Xtr, ytr, sample_weight=w)
        oof['xgb'][va] = m.predict_proba(Xva)[:, 1]; models['xgb'].append(m)

        m = CatBoostClassifier(iterations=120, learning_rate=0.04, depth=4, l2_leaf_reg=3.0,
                               auto_class_weights='Balanced', verbose=0, random_seed=42 + fold)
        m.fit(Xtr, ytr, sample_weight=w)
        oof['cat'][va] = m.predict_proba(Xva)[:, 1]; models['cat'].append(m)
        print(f"  fold {fold} done ({time.time()-t0:.0f}s)")

    print(f"[mode={mode}] suspect per fold: {n_suspect}")
    out = os.path.join(CURRENT_DIR, f"bl4_v5_oof_{mode}.npz")
    np.savez(out, **oof)
    with open(os.path.join(CURRENT_DIR, f"bl4_v5_models_{mode}.pkl"), 'wb') as f:
        pickle.dump(models, f)
    print(f"[+] saved {out}")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'ref')

