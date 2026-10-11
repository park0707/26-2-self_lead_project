import numpy as np
import pickle
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def eval_3s(y_true, prob, tc, td):
    m = (prob < tc) | (prob >= td)
    cov = m.mean() * 100.0
    if cov == 0: return 0.0, 0.0
    yt = y_true[m]
    yp = (prob[m] >= td).astype(int)
    if len(yt) == 0: return 0.0, cov
    f1 = f1_score(yt, yp) * 100.0
    return f1, cov

def main():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    subjects = df['subj'].values

    npz = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\stiff_sd1_rtmp_clean_features.npz')
    X = npz['X_tab']
    X = StandardScaler().fit_transform(X)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aum_oof = np.zeros(len(y_raw))

    for trn_idx, val_idx in sgkf.split(X, y_raw, groups=subjects):
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
        clf.fit(X[trn_idx], y_raw[trn_idx])
        probs = clf.predict_proba(X[val_idx])
        for j, v_i in enumerate(val_idx):
            l = y_raw[v_i]
            aum_oof[v_i] = probs[j, l] - probs[j, 1-l]

    drop_threshold = np.percentile(aum_oof, (339 / len(y_raw)) * 100)
    mask_keep = ~(aum_oof <= drop_threshold)

    X_sub = X[mask_keep]
    y_sub = y_raw[mask_keep]
    subj_sub = subjects[mask_keep]

    oof_prob = np.zeros(len(y_sub))
    for trn_idx, val_idx in sgkf.split(X_sub, y_sub, groups=subj_sub):
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
        clf2.fit(X_sub[trn_idx], y_sub[trn_idx])
        oof_prob[val_idx] = clf2.predict_proba(X_sub[val_idx])[:, 1]

    b_f1, b_cov = 0.0, 100.0
    for tc in np.linspace(0.20, 0.50, 31):
        for td in np.linspace(tc + 0.03, 0.65, 31):
            f1, cov = eval_3s(y_sub, oof_prob, tc, td)
            if cov >= 85.0 and f1 > b_f1:
                b_f1, b_cov = f1, cov

    with open('d:/rtmp_f1_result.txt', 'w') as f:
        f.write(f"{b_f1:.2f}\n{b_cov:.2f}")

if __name__ == '__main__':
    main()
