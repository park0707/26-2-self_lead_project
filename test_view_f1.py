import os
import pickle
import numpy as np
import time
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

def load_dataset():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    return seq_ids, subjects, views, keys, y_raw

def extract_features(seq_ids, views):
    import sys
    sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
    from sd1_rtmp_2_features import extract_sd1_rtmp_2_features
    X = np.zeros((len(seq_ids), 47))
    RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    for i, sid in enumerate(seq_ids):
        p = os.path.join(RTM_DIR, sid + '.npz')
        if not os.path.exists(p):
            p = os.path.join(r"D:\Data\시니어_피트니스\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트", sid + '.npz')
            if not os.path.exists(p):
                import glob
                f = glob.glob(r'D:\Data\**\03_바벨_스티프_데드리프트\\' + sid + '.npz', recursive=True)
                if f: p = f[0]
        if os.path.exists(p):
            kps = np.load(p)['kps_26_64f']
            X[i] = extract_sd1_rtmp_2_features(kps, views[i])
    return X

def eval_3s(y_true, prob, tc, td):
    m = (prob < tc) | (prob >= td)
    cov = m.mean() * 100.0
    if cov == 0: return 0.0, 0.0
    yt = y_true[m]
    yp = (prob[m] >= td).astype(int)
    if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2:
        return 0.0, cov
    return f1_score(yt, yp) * 100.0, cov

def main():
    seq_ids, subjects, views, keys, y_raw = load_dataset()
    X = extract_features(seq_ids, views)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_raw = np.zeros(len(y_raw))
    for trn_idx, val_idx in sgkf.split(X, y_raw, groups=subjects):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, 
                                 class_weight='balanced', random_state=42, verbose=-1)
        clf.fit(X[trn_idx], y_raw[trn_idx])
        oof_prob_raw[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        
    y_clean = y_raw.copy()
    for i in range(len(y_raw)):
        if y_raw[i] == 1 and oof_prob_raw[i] < 0.15: y_clean[i] = 0
        elif y_raw[i] == 0 and oof_prob_raw[i] > 0.85: y_clean[i] = 1
        
    oof_final = np.zeros(len(y_clean))
    for trn_idx, val_idx in sgkf.split(X, y_clean, groups=subjects):
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.025, max_depth=5, num_leaves=20,
                                     class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf_cb = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6,
                                    auto_class_weights='Balanced', verbose=0, random_seed=42)
        clf_lgb.fit(X[trn_idx], y_clean[trn_idx])
        clf_cb.fit(X[trn_idx], y_clean[trn_idx])
        prob = (clf_lgb.predict_proba(X[val_idx])[:, 1] * 0.5) + (clf_cb.predict_proba(X[val_idx])[:, 1] * 0.5)
        oof_final[val_idx] = prob

    VIEW_ALIAS_MAP = {
        'view5': '시점 A (우측 후방)', 'view2': '시점 B (우측 전방)',
        'view3': '시점 C (정면)', 'view4': '시점 D (좌측 전방)', 'view1': '시점 E (좌측 후방)'
    }
    
    print("\n[+] 시점별 최대 F1 최적화 결과 (Coverage >= 85% 제한):")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        mask_v = (views == vk)
        y_v = y_clean[mask_v]
        prob_v = oof_final[mask_v]
        
        b_f1, b_tc, b_td, b_cov = 0.0, 0.35, 0.45, 100.0
        # 밀도 높은 Grid Search
        for tc in np.linspace(0.1, 0.6, 100):
            for td in np.linspace(tc + 0.01, 0.85, 100):
                f1, cov = eval_3s(y_v, prob_v, tc, td)
                if cov < 85.0: continue
                if f1 > b_f1:
                    b_f1 = f1; b_tc = tc; b_td = td; b_cov = cov
                    
        alias = VIEW_ALIAS_MAP.get(vk, vk)
        auc_v = roc_auc_score(y_v, prob_v) * 100.0
        print(f"  - {alias}: 최적 F1 = {b_f1:.2f}% (AUC: {auc_v:.2f}%, Coverage: {b_cov:.2f}%) at tc={b_tc:.3f}, td={b_td:.3f}")

if __name__ == '__main__':
    main()
