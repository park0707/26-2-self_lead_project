import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_2 import prepare_v5_2_dataset
from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import (
    prepare_v5_3_dataset,
    find_best_view_threshold_f1_priority,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v5_3 import (
    DeadliftSTSequenceNetV5_3,
    apply_kapw_phase_warping
)

def run():
    X_seq_51, X_feats_26, y, subjects, views, _ = prepare_v5_2_dataset()
    _, X_feats_32, _, _, _, _, _ = prepare_v5_3_dataset()

    N, T, _ = X_seq_51.shape
    data = np.load(r'd:\src\models\second_models\babel_deadlift\d3_v5_dataset_cache.npz', allow_pickle=True)
    kps_15 = data['X_seq'].reshape(N, T, 15, 3)

    # 올바른 시간축 정합 KAPW 57D
    kps_15_kapw = np.zeros_like(kps_15)
    for i in range(N):
        kps_15_kapw[i], _, _ = apply_kapw_phase_warping(kps_15[i])

    w_kapw = (kps_15_kapw[:, :, 3] + kps_15_kapw[:, :, 4]) / 2.0
    a_kapw = (kps_15_kapw[:, :, 9] + kps_15_kapw[:, :, 10]) / 2.0
    h_kapw = (kps_15_kapw[:, :, 11] + kps_15_kapw[:, :, 12]) / 2.0
    t_kapw = (kps_15_kapw[:, :, 13] + kps_15_kapw[:, :, 14]) / 2.0

    bar_to_ank_kapw = w_kapw - a_kapw
    heel_to_toe_kapw = t_kapw - h_kapw
    X_seq_57 = np.concatenate([X_seq_51, bar_to_ank_kapw, heel_to_toe_kapw], axis=-1).astype(np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tree_32 = np.zeros(len(y), dtype=np.float32)
    oof_tree_26 = np.zeros(len(y), dtype=np.float32)
    oof_dl = np.zeros(len(y), dtype=np.float32)

    print("[*] 5-Fold OOF 학습 진행 중...")
    for fold, (tr, val) in enumerate(sgkf.split(X_feats_32, y, groups=subjects)):
        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        # 1. Tree on 32D
        cb32 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb32.fit(X_feats_32[tr], y[tr])
        p_cb32 = cb32.predict_proba(X_feats_32[val])[:, 1]

        ds_tr = lgb.Dataset(X_feats_32[tr], label=y[tr])
        lgb_m = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr, num_boost_round=230)
        p_lgb32 = lgb_m.predict(X_feats_32[val])

        rf32 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf32.fit(X_feats_32[tr], y[tr])
        p_rf32 = rf32.predict_proba(X_feats_32[val])[:, 1]
        oof_tree_32[val] = 0.50 * p_cb32 + 0.30 * p_lgb32 + 0.20 * p_rf32

        # 2. Tree on 26D
        cb26 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb26.fit(X_feats_26[tr], y[tr])
        p_cb26 = cb26.predict_proba(X_feats_26[val])[:, 1]

        ds_tr26 = lgb.Dataset(X_feats_26[tr], label=y[tr])
        lgb_m26 = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr26, num_boost_round=230)
        p_lgb26 = lgb_m26.predict(X_feats_26[val])

        rf26 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf26.fit(X_feats_26[tr], y[tr])
        p_rf26 = rf26.predict_proba(X_feats_26[val])[:, 1]
        oof_tree_26[val] = 0.50 * p_cb26 + 0.30 * p_lgb26 + 0.20 * p_rf26

        # 3. ST-SeqNet 57D
        net = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64)
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

    print("\n--- [개별 모델 5-Fold OOF ROC-AUC & Raw F1] ---")
    for name, p in [('Tree 26D', oof_tree_26), ('Tree 32D', oof_tree_32), ('ST-SeqNet 57D', oof_dl)]:
        auc = roc_auc_score(y, p)
        raw_f1 = f1_score(y, (p >= 0.5).astype(int))
        print(f"  {name:15s}: ROC-AUC = {auc:.4f} | Raw F1(0.5) = {raw_f1*100:.2f}%")

    # 다양한 앙상블 블렌드 테스트
    print("\n--- [앙상블 블렌드 3-State 안전 게이트 최적화 (보류율 <= 10.0%)] ---")
    candidates = [
        ("Tree 26D 100%", oof_tree_26),
        ("Tree 32D 100%", oof_tree_32),
        ("ST-SeqNet 30% + Tree 26D 70%", 0.30 * oof_dl + 0.70 * oof_tree_26),
        ("ST-SeqNet 35% + Tree 26D 65%", 0.35 * oof_dl + 0.65 * oof_tree_26),
        ("ST-SeqNet 30% + Tree 32D 70%", 0.30 * oof_dl + 0.70 * oof_tree_32),
        ("ST-SeqNet 35% + Tree 32D 65%", 0.35 * oof_dl + 0.65 * oof_tree_32),
        ("ST-SeqNet 25% + Tree 32D 50% + Tree 26D 25%", 0.25 * oof_dl + 0.50 * oof_tree_32 + 0.25 * oof_tree_26),
    ]

    for cname, p_blend in candidates:
        det_mask = np.zeros(len(y), dtype=bool)
        pred_tot = np.zeros(len(y), dtype=int)
        view_f1s = []
        for vcode in VIEW_CODE_LIST:
            v_idx = np.where(views == vcode)[0]
            tn, td, m = find_best_view_threshold_f1_priority(y[v_idx], p_blend[v_idx], min_coverage=0.90)
            det_mask[v_idx] = (p_blend[v_idx] <= tn) | (p_blend[v_idx] >= td)
            pred_tot[v_idx] = np.where(p_blend[v_idx] >= td, 1, 0)
            view_f1s.append(m['f1'])

        overall_f1 = f1_score(y[det_mask], pred_tot[det_mask])
        overall_cov = np.mean(det_mask)
        overall_auc = roc_auc_score(y, p_blend)
        print(f"[{cname}]")
        print(f"  -> 확정 F1: {overall_f1*100:.2f}% | 커버리지: {overall_cov*100:.2f}% (보류율: {(1-overall_cov)*100:.2f}%) | AUC: {overall_auc:.4f}")
        print(f"  -> 시점별 F1(A~E): {[round(v*100, 2) for v in view_f1s]}")

if __name__ == '__main__':
    run()
