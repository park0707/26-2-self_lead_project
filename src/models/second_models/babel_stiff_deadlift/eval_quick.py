import os
import numpy as np
import pickle
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
import sys

CURRENT_DIR = r"D:\src\models\second_models\babel_stiff_deadlift"
sys.path.insert(0, r"D:\src")
from features.stiff_spine_feature_extractor import extract_stiff_temporal_24d
from models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

def evaluate():
    NPZ_AMBIGUITY_PATH = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")
    SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
    RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
    BUNDLE_V3_PATH = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")

    data = np.load(NPZ_AMBIGUITY_PATH, allow_pickle=True)
    seq_ids = data['sequence_ids']
    labels = data['labels']
    subjects = data['subject_ids']
    X_30d = data['features_30d']
    N = len(seq_ids)

    # 2. 24D 시계열 텐서 로드
    sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    sids = list(sc['sequence_ids'])
    s_vision = sc['spine_9_64f_vision']
    spine_map = {sids[i]: s_vision[i] for i in range(len(sids))}

    X_temp_24d = np.zeros((N, 64, 24), dtype=np.float32)
    for i, sid in enumerate(seq_ids):
        fpath = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        raw = np.load(fpath, allow_pickle=True)
        kps = raw['kps_33_3d_64f'] if 'kps_33_3d_64f' in raw else raw['kps_33_2d_64f']
        s_pts = spine_map.get(sid, None)
        X_temp_24d[i] = extract_stiff_temporal_24d(kps, s_pts)

    with open(BUNDLE_V3_PATH, 'rb') as f:
        bundle = pickle.load(f)

    scalers = bundle['scalers']
    tcn_weights = bundle['tcn_weights']
    lgb_models = bundle['lgb_models']
    cat_models = bundle['cat_models']
    ens_w = bundle['ensemble_weights']
    thres = bundle['thresholds']

    sgkf = StratifiedGroupKFold(n_splits=5)
    
    oof_tcn = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cat = np.zeros(N, dtype=np.float32)
    
    device = torch.device("cpu")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_30d, labels, groups=subjects)):
        scaler = scalers[fold]
        X_va_scaled = scaler.transform(X_30d[val_idx])

        # TCN
        tcn_x_va = torch.tensor(X_temp_24d[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(device)
        tcn_model.load_state_dict(tcn_weights[fold])
        tcn_model.eval()
        with torch.no_grad():
            out = tcn_model(tcn_x_va)
            p = torch.sigmoid(out).squeeze(-1).cpu().numpy()
            oof_tcn[val_idx] = p

        # LGB
        lgb_clf = lgb_models[fold]
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_va_scaled)[:, 1]

        # CAT
        cat_clf = cat_models[fold]
        oof_cat[val_idx] = cat_clf.predict_proba(X_va_scaled)[:, 1]

    oof_ens = ens_w['tcn'] * oof_tcn + ens_w['lgb'] * oof_lgb + ens_w['cat'] * oof_cat

    # 3-State Evaluation
    state_preds_all = np.full(N, -1, dtype=int)
    state_preds_all[oof_ens <= thres['tau_clean']] = 0
    state_preds_all[oof_ens >= thres['tau_defect']] = 1
    
    det_mask_all = (state_preds_all != -1)
    cov_all = np.mean(det_mask_all)
    f1_det_all = f1_score(labels[det_mask_all], state_preds_all[det_mask_all], zero_division=0)
    acc_det_all = accuracy_score(labels[det_mask_all], state_preds_all[det_mask_all])

    print("=== Total Baseline (Level 1~4) 3-State Performance ===")
    print(f"Coverage: {cov_all*100:.2f}%")
    print(f"Abstain Rate: {(1.0 - cov_all)*100:.2f}%")
    print(f"F1-Score: {f1_det_all*100:.2f}%")
    print(f"Accuracy: {acc_det_all*100:.2f}%\n")

    views = data['view_ids']
    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    VIEW_ALIAS_MAP = {
        'view5': '시점 A (우측 후방 45°)',
        'view2': '시점 B (우측 전방 45°)',
        'view3': '시점 C (정면 0°)',
        'view4': '시점 D (좌측 전방 45°)',
        'view1': '시점 E (좌측 후방 45°)'
    }
    
    print("=== Per-View 3-State Performance ===")
    for vn in view_order:
        mask = (views == vn)
        n_v = np.sum(mask)
        if n_v == 0: continue
        
        y_v = labels[mask]
        p_v = state_preds_all[mask]
        
        det_mask_v = (p_v != -1)
        cov_v = np.mean(det_mask_v)
        
        f1_v = f1_score(y_v[det_mask_v], p_v[det_mask_v], zero_division=0)
        acc_v = accuracy_score(y_v[det_mask_v], p_v[det_mask_v])
        
        vname = VIEW_ALIAS_MAP[vn]
        print(f"{vname}: N={n_v}, F1={f1_v*100:.2f}%, Abstain={(1.0-cov_v)*100:.2f}%")

if __name__ == '__main__':
    evaluate()
