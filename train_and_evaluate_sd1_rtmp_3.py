import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import json

def load_dataset():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    y = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    return seq_ids, subjects, views, y, df['path'].values

def extract_all_features(seq_ids, views, paths):
    import sys
    sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
    from sd1_rtmp_2_features import extract_sd1_rtmp_2_features, FEATURE_NAMES_47D_SD1_RTMP_2
    
    X = np.zeros((len(seq_ids), 47))
    for i, sid in enumerate(seq_ids):
        # Resolve correct base path for the npz
        p = paths[i].replace('MediaPipe_Full33_64F', 'RTMPose_Halpe26_64F')
        if os.path.exists(p):
            kps = np.load(p)['kps_26_64f']
            X[i] = extract_sd1_rtmp_2_features(kps, views[i])
    return X, FEATURE_NAMES_47D_SD1_RTMP_2

def eval_3s(y_true, prob, tc, td):
    m = (prob < tc) | (prob >= td)
    cov = m.mean() * 100.0
    if cov == 0: return 0.0, 0.0
    yt = y_true[m]
    yp = (prob[m] >= td).astype(int)
    if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2: return 0.0, cov
    return f1_score(yt, yp) * 100.0, cov

def main():
    print("[*] Loading dataset and extracting baseline 47D features...")
    seq_ids, subjects, views, y, paths = load_dataset()
    X_full, feat_names = extract_all_features(seq_ids, views, paths)
    
    print("\n[*] Step 1: Feature Importance & Correlation Analysis (Feature Selection)")
    df_X = pd.DataFrame(X_full, columns=feat_names)
    
    # 1. Correlation (Remove >0.95 collinear)
    corr = df_X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
    
    # 2. Importance (LightGBM pre-train)
    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
    clf.fit(X_full, y)
    imp = clf.feature_importances_
    df_imp = pd.DataFrame({'feature': feat_names, 'importance': imp}).sort_values('importance')
    
    # Drop bottom 15 features (least important)
    drop_imp = df_imp['feature'].head(15).tolist()
    
    final_drop = set(drop_corr + drop_imp)
    # Never drop view encodings manually if they are helpful, but we let the algo decide
    
    opt_feats = [f for f in feat_names if f not in final_drop]
    opt_idx = [feat_names.index(f) for f in opt_feats]
    
    X_opt = X_full[:, opt_idx]
    
    print(f"[+] Dropped {len(final_drop)} noisy/redundant features.")
    print(f"    - Correlated: {drop_corr}")
    print(f"    - Low Importance: {drop_imp}")
    print(f"[+] Selected {len(opt_feats)} optimized features.")
    
    print("\n[*] Step 2: Training sd1_rtmp_3 (Optimized Features) on Raw Labels...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y))
    
    for trn_idx, val_idx in sgkf.split(X_opt, y, groups=subjects):
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.025, max_depth=5, num_leaves=20, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf_cb = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, auto_class_weights='Balanced', verbose=0, random_seed=42)
        clf_lgb.fit(X_opt[trn_idx], y[trn_idx])
        clf_cb.fit(X_opt[trn_idx], y[trn_idx])
        oof_prob[val_idx] = (clf_lgb.predict_proba(X_opt[val_idx])[:, 1] + clf_cb.predict_proba(X_opt[val_idx])[:, 1]) / 2.0
        
    print(f"[+] OOF ROC-AUC: {roc_auc_score(y, oof_prob)*100:.2f}%")
    
    print("\n[*] Step 3: Global 3-State Threshold Search (Coverage >= 85%)")
    b_f1, b_tc, b_td, b_cov = 0.0, 0.35, 0.45, 100.0
    for tc in np.linspace(0.20, 0.52, 65):
        for td in np.linspace(tc + 0.02, 0.70, 65):
            f1, cov = eval_3s(y, oof_prob, tc, td)
            if cov < 85.0: continue
            if f1 > b_f1:
                b_f1 = f1; b_tc = tc; b_td = td; b_cov = cov
                
    m = (oof_prob < b_tc) | (oof_prob >= b_td)
    yt = y[m]; yp = (oof_prob[m] >= b_td).astype(int)
    final_f1 = f1_score(yt, yp) * 100.0
    
    print(f"\n==========================================================================================")
    print(f"  [vidio-health-trainer] sd1_rtmp_3 (Feature Optimized) 실측 성적표")
    print(f"==========================================================================================")
    print(f"  - Features:       {len(opt_feats)}D (Reduced from 47D)")
    print(f"  - 3-State F1:     {final_f1:.2f}% (비교: v2.0 47D는 61.75%)")
    print(f"  - Coverage:       {b_cov:.2f}%")
    print(f"==========================================================================================")
    print("완료되었습니다.")

if __name__ == '__main__':
    main()
