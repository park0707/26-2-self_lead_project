# -*- coding: utf-8 -*-
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

print("[*] Quick speed test starting...", flush=True)
master = np.load(r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz", allow_pickle=True)
rtm = np.load(r"D:\Data\피처_데이터\Master_Squat_RTMPose26_64F.npz", allow_pickle=True)

X = rtm['X_c3'].astype(np.float32)
y = master['Y_defects_golden'][:, 2]
groups = master['subjects']

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
trn_idx, val_idx = next(sgkf.split(X, y, groups=groups))

t0 = time.time()
cb = CatBoostClassifier(iterations=120, depth=5, learning_rate=0.05, thread_count=1, allow_writing_files=False, verbose=False)
cb.fit(X[trn_idx], y[trn_idx])
p_cb = cb.predict_proba(X[val_idx])[:, 1]
t_cb = time.time() - t0

t0 = time.time()
clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, num_leaves=20, n_jobs=1, verbose=-1)
clf.fit(X[trn_idx], y[trn_idx])
p_lgb = clf.predict_proba(X[val_idx])[:, 1]
t_lgb = time.time() - t0

p_ens = 0.5 * p_cb + 0.5 * p_lgb
f1 = f1_score(y[val_idx], (p_ens >= 0.5).astype(int)) * 100
auc = roc_auc_score(y[val_idx], p_ens) * 100

print(f"[+] CB time: {t_cb:.2f}s | LGB time: {t_lgb:.2f}s | 1 Fold Val F1: {f1:.2f}%, AUC: {auc:.2f}%", flush=True)
