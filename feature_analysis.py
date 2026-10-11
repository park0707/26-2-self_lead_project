import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    
    # We will use the original noisy labels for feature selection realism (or confident learning labels)
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    return seq_ids, subjects, views, y_raw

def extract_features(seq_ids, views):
    import sys
    sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
    from sd1_rtmp_2_features import extract_sd1_rtmp_2_features, FEATURE_NAMES_47D_SD1_RTMP_2
    
    X = np.zeros((len(seq_ids), 47))
    dirs = [
        r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트",
        r"D:\Data\시니어_피트니스\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    ]
    for i, sid in enumerate(seq_ids):
        for d in dirs:
            p = os.path.join(d, sid + '.npz')
            if os.path.exists(p):
                kps = np.load(p)['kps_26_64f']
                X[i] = extract_sd1_rtmp_2_features(kps, views[i])
                break
    return X, FEATURE_NAMES_47D_SD1_RTMP_2

def main():
    print("Loading data and extracting features...")
    seq_ids, subjects, views, y = load_data()
    X, feat_names = extract_features(seq_ids, views)
    
    df_X = pd.DataFrame(X, columns=feat_names)
    
    print("\n[1] Correlation Analysis (Multicollinearity Check)")
    corr = df_X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.90
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    print(f"Highly correlated features to drop (>0.90): {len(to_drop_corr)}")
    for f in to_drop_corr:
        # Find what it's correlated with
        corrs_with = upper.index[upper[f] > 0.90].tolist()
        print(f"  - {f} (correlated with {corrs_with})")

    # [2] Feature Importance using LightGBM (with 5-Fold CV)
    print("\n[2] Feature Importance Analysis")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    feature_importances = np.zeros(len(feat_names))
    oof_preds = np.zeros(len(y))
    
    for trn_idx, val_idx in sgkf.split(X, y, groups=subjects):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, 
                                 class_weight='balanced', random_state=42, verbose=-1)
        clf.fit(X[trn_idx], y[trn_idx])
        feature_importances += clf.feature_importances_ / 5.0
        oof_preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        
    baseline_auc = roc_auc_score(y, oof_preds)
    print(f"Baseline OOF AUC (47 features): {baseline_auc:.4f}")
    
    df_imp = pd.DataFrame({'feature': feat_names, 'importance': feature_importances})
    df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Most Important Features:")
    for i in range(10):
        print(f"  {i+1}. {df_imp['feature'].iloc[i]} ({df_imp['importance'].iloc[i]:.1f})")
        
    zero_imp = df_imp[df_imp['importance'] == 0]['feature'].tolist()
    print(f"\nZero Importance Features (pure noise): {len(zero_imp)}")
    for f in zero_imp: print(f"  - {f}")
    
    # Identify bottom 20% features (excluding view codes and one-hots which are contextual)
    bottom_k = int(47 * 0.2)
    low_imp_candidates = df_imp['feature'].iloc[-bottom_k:].tolist()
    to_drop_imp = [f for f in low_imp_candidates if not f.startswith('v_oh_') and f != 'view_code']
    print(f"\nLow Importance Features to Drop: {to_drop_imp}")

    # Final optimized feature list
    drop_set = set(to_drop_corr + zero_imp + to_drop_imp)
    optimized_features = [f for f in feat_names if f not in drop_set]
    print(f"\n[3] Optimized Feature Set: {len(optimized_features)} features (Reduced from 47)")
    print(optimized_features)

if __name__ == '__main__':
    main()
