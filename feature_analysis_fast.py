import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
import warnings
warnings.filterwarnings('ignore')

f_out = open(r'D:\feat_out.txt', 'w', encoding='utf-8')
def p(s):
    f_out.write(s + '\n')
    f_out.flush()
    print(s, flush=True)

try:
    p("Loading data...")
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(pth))[0] for pth in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    
    y = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    
    p("Extracting features...")
    import sys
    sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
    from sd1_rtmp_2_features import extract_sd1_rtmp_2_features, FEATURE_NAMES_47D_SD1_RTMP_2
    
    X = np.zeros((len(seq_ids), 47))
    RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    # also try another dir if missing
    RTM_DIR2 = r"D:\Data\시니어_피트니스\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    
    found_count = 0
    for i, sid in enumerate(seq_ids):
        pth = os.path.join(RTM_DIR, sid + '.npz')
        if not os.path.exists(pth):
            pth = os.path.join(RTM_DIR2, sid + '.npz')
        if os.path.exists(pth):
            kps = np.load(pth)['kps_26_64f']
            X[i] = extract_sd1_rtmp_2_features(kps, views[i])
            found_count += 1
            
    p(f"Found {found_count}/{len(seq_ids)} npz files.")
    
    df_X = pd.DataFrame(X, columns=FEATURE_NAMES_47D_SD1_RTMP_2)
    
    p("\n[1] Correlation Check")
    corr = df_X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    p(f"Correlated features to drop: {len(to_drop_corr)}")
    for f in to_drop_corr:
        corrs = upper.index[upper[f] > 0.90].tolist()
        p(f"  {f} (correlates with {corrs})")
        
    p("\n[2] Feature Importance")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fi = np.zeros(len(FEATURE_NAMES_47D_SD1_RTMP_2))
    
    for trn_idx, val_idx in sgkf.split(X, y, groups=subjects):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, 
                                 class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf.fit(X[trn_idx], y[trn_idx])
        fi += clf.feature_importances_ / 5.0
        
    df_imp = pd.DataFrame({'feature': FEATURE_NAMES_47D_SD1_RTMP_2, 'importance': fi})
    df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    
    p("\nTop 10:")
    for i in range(10): p(f"  {df_imp['feature'].iloc[i]}: {df_imp['importance'].iloc[i]:.1f}")
    
    zero_imp = df_imp[df_imp['importance'] == 0]['feature'].tolist()
    p(f"\nZero Importance: {len(zero_imp)}")
    for f in zero_imp: p(f"  - {f}")
    
    bottom_20 = int(47 * 0.2)
    to_drop_imp = [f for f in df_imp['feature'].iloc[-bottom_20:].tolist() if not f.startswith('v_oh_') and f != 'view_code']
    p(f"\nLow Importance: {to_drop_imp}")
    
    drop_set = set(to_drop_corr + zero_imp + to_drop_imp)
    opt = [f for f in FEATURE_NAMES_47D_SD1_RTMP_2 if f not in drop_set]
    
    p(f"\nOptimized count: {len(opt)}")
    p(f"Optimized list: {opt}")

except Exception as e:
    p(f"ERROR: {str(e)}")
finally:
    f_out.close()
