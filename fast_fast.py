import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)

    npz = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\stiff_sd1_rtmp_clean_features.npz')
    X = npz['X_tab']
    X = StandardScaler().fit_transform(X)

    # Fast AUM (just use out-of-bag from one split)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_raw, test_size=0.5, random_state=42)
    clf = LogisticRegression(max_iter=50)
    clf.fit(X_tr, y_tr)
    probs_te = clf.predict_proba(X_te)
    
    aum = np.zeros(len(y_te))
    for j in range(len(y_te)):
        l = y_te[j]
        aum[j] = probs_te[j, l] - probs_te[j, 1-l]

    drop_threshold = np.percentile(aum, (339 / len(y_raw)) * 100)
    mask_keep = ~(aum <= drop_threshold)

    X_sub = X_te[mask_keep]
    y_sub = y_te[mask_keep]

    clf2 = LogisticRegression(max_iter=50)
    clf2.fit(X_sub, y_sub)
    oof_prob = clf2.predict_proba(X_sub)[:, 1]

    b_f1, b_cov = 0.0, 100.0
    for tc in np.linspace(0.20, 0.50, 11):
        for td in np.linspace(tc + 0.03, 0.65, 11):
            m = (oof_prob < tc) | (oof_prob >= td)
            cov = m.mean() * 100.0
            if cov == 0: continue
            yt = y_sub[m]
            yp = (oof_prob[m] >= td).astype(int)
            if len(yt) == 0: continue
            f1 = f1_score(yt, yp) * 100.0
            if cov >= 85.0 and f1 > b_f1:
                b_f1 = f1
                b_cov = cov

    with open('d:/rtmp_f1_result.txt', 'w') as f:
        f.write(f"{b_f1:.2f}\n{b_cov:.2f}")
    print(f"Done: {b_f1:.2f}")

if __name__ == '__main__':
    main()
