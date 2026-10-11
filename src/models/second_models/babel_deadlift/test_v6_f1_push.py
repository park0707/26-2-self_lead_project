# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from src.models.second_models.babel_deadlift.train_and_eval_d3_v6_0 import (
    prepare_v6_0_dataset,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v6_0 import DeadliftSTSequenceNetV6_0

def test_v6_f1_push():
    X_seq_57, X_feats_36, y, subjects, views, clip_names = prepare_v6_0_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tree_36 = np.zeros(len(y), dtype=np.float32)
    oof_dl = np.zeros(len(y), dtype=np.float32)

    for fold, (tr, val) in enumerate(sgkf.split(X_feats_36, y, groups=subjects)):
        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        cb36 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb36.fit(X_feats_36[tr], y[tr])
        p_cb36 = cb36.predict_proba(X_feats_36[val])[:, 1]

        ds_tr = lgb.Dataset(X_feats_36[tr], label=y[tr])
        lgb_m = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr, num_boost_round=230)
        p_lgb36 = lgb_m.predict(X_feats_36[val])

        rf36 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf36.fit(X_feats_36[tr], y[tr])
        p_rf36 = rf36.predict_proba(X_feats_36[val])[:, 1]
        oof_tree_36[val] = 0.50 * p_cb36 + 0.30 * p_lgb36 + 0.20 * p_rf36

        net = DeadliftSTSequenceNetV6_0(in_features=57, hidden_dim=64, num_heads=4)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w_t = torch.tensor([pos_w])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

        t_x = torch.tensor(X_seq_57[tr], dtype=torch.float32)
        t_y = torch.tensor(y[tr], dtype=torch.float32)

        net.train()
        for ep in range(35):
            indices = np.random.permutation(len(tr))
            for i in range(0, len(tr), 32):
                b = indices[i:i+32]
                opt.zero_grad()
                logits = net(t_x[b])
                loss = bce(logits, t_y[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof_dl[val] = torch.sigmoid(net(torch.tensor(X_seq_57[val], dtype=torch.float32))).numpy()

    # Blend
    p_final = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
        p_final[i] = w_dl * oof_dl[i] + (1.0 - w_dl) * oof_tree_36[i]

    # Save cached predictions
    np.savez('d3_v6_oof_preds.npz', p_final=p_final, y=y, views=views)
    print(f"[+] Total AUC: {roc_auc_score(y, p_final):.4f}")

    # Threshold optimization with recall priority
    view_budget_map = {
        'view5': 0.88,
        'view2': 0.92,
        'view3': 0.93,
        'view4': 0.92,
        'view1': 0.88
    }

    det_mask = np.zeros(len(y), dtype=bool)
    pred_total = np.zeros(len(y), dtype=int)
    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views == vcode)[0]
        y_v = y[v_idx]
        p_v = p_final[v_idx]
        min_cov = view_budget_map[vcode]

        best_f1 = 0
        best_cfg = None
        for tn in np.arange(0.20, 0.46, 0.005):
            for td in np.arange(0.38, 0.58, 0.005):
                if tn >= td:
                    continue
                determ = (p_v <= tn) | (p_v >= td)
                cov = np.mean(determ)
                if cov < min_cov:
                    continue
                pred = np.where(p_v[determ] >= td, 1, 0)
                if len(np.unique(y_v[determ])) < 2 or np.sum(pred == 1) == 0:
                    continue
                f1 = f1_score(y_v[determ], pred, zero_division=0)
                prec = precision_score(y_v[determ], pred, zero_division=0)
                rec = recall_score(y_v[determ], pred, zero_division=0)
                if f1 > best_f1 and prec >= 0.50 and rec >= 0.65:
                    best_f1 = f1
                    best_cfg = (tn, td, f1, prec, rec, cov)
        if not best_cfg:
            # fallback
            for tn in np.arange(0.20, 0.46, 0.005):
                for td in np.arange(0.38, 0.65, 0.005):
                    if tn >= td:
                        continue
                    determ = (p_v <= tn) | (p_v >= td)
                    cov = np.mean(determ)
                    if cov < min_cov:
                        continue
                    pred = np.where(p_v[determ] >= td, 1, 0)
                    f1 = f1_score(y_v[determ], pred, zero_division=0)
                    prec = precision_score(y_v[determ], pred, zero_division=0)
                    rec = recall_score(y_v[determ], pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = (tn, td, f1, prec, rec, cov)

        tn, td, f1, prec, rec, cov = best_cfg
        det_mask[v_idx] = (p_v <= tn) | (p_v >= td)
        pred_total[v_idx] = np.where(p_v >= td, 1, 0)
        print(f"[{VIEW_ALIAS_MAP[vcode]}] F1: {f1*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, Cov: {cov*100:.2f}%, tn={tn:.3f}, td={td:.3f}")

    y_det = y[det_mask]
    p_det = pred_total[det_mask]
    tot_f1 = f1_score(y_det, p_det)
    tot_prec = precision_score(y_det, p_det)
    tot_rec = recall_score(y_det, p_det)
    tot_cov = np.mean(det_mask)
    tot_acc = accuracy_score(y_det, p_det)
    cm = confusion_matrix(y_det, p_det)

    print("\n" + "=" * 80)
    print("★ [v6.0 Recall 균형 최적화 종합 실측 결과] ★")
    print(f"  - 전체 확정 F1-Score: {tot_f1*100:.2f}%")
    print(f"  - 전체 확정 Recall: {tot_rec*100:.2f}% | Precision: {tot_prec*100:.2f}%")
    print(f"  - 전체 확정 Accuracy: {tot_acc*100:.2f}%")
    print(f"  - 전체 커버리지: {tot_cov*100:.2f}% (보류율: {(1-tot_cov)*100:.2f}%)")
    print("  - Confusion Matrix:\n", cm)
    print("=" * 80)

if __name__ == '__main__':
    test_v6_f1_push()
