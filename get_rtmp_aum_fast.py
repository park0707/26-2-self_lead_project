import numpy as np
import pickle
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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
    print("Loading data...")
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    subjects = df['subj'].values

    # Load RTMP features
    npz = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\stiff_sd1_rtmp_clean_features.npz')
    X = npz['X_tab']
    
    print("Generating AUM Mask...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    aum_oof = np.zeros(len(y_raw))

    for trn_idx, val_idx in sgkf.split(X_scaled, y_raw, groups=subjects):
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=True, random_state=42)
        fold_aum = np.zeros(len(val_idx))
        for epoch in range(25):
            mlp.partial_fit(X_scaled[trn_idx], y_raw[trn_idx], classes=np.array([0, 1]))
            probs = mlp.predict_proba(X_scaled[val_idx])
            for j, v_i in enumerate(val_idx):
                l = y_raw[v_i]
                margin = probs[j, l] - probs[j, 1-l]
                fold_aum[j] += margin
        aum_oof[val_idx] = fold_aum

    drop_threshold = np.percentile(aum_oof, (339 / len(y_raw)) * 100)
    mask_keep = ~(aum_oof <= drop_threshold)

    X_sub = X[mask_keep]
    y_sub = y_raw[mask_keep]
    subj_sub = subjects[mask_keep]

    print("Training on RTMP AUM subset...")
    oof_prob = np.zeros(len(y_sub))
    for trn_idx, val_idx in sgkf.split(X_sub, y_sub, groups=subj_sub):
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.025, max_depth=5, num_leaves=20, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf_cb = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, auto_class_weights='Balanced', verbose=0, random_seed=42)
        clf_lgb.fit(X_sub[trn_idx], y_sub[trn_idx])
        clf_cb.fit(X_sub[trn_idx], y_sub[trn_idx])
        oof_prob[val_idx] = (clf_lgb.predict_proba(X_sub[val_idx])[:, 1] + clf_cb.predict_proba(X_sub[val_idx])[:, 1]) / 2.0

    b_f1, b_cov = 0.0, 100.0
    for tc in np.linspace(0.20, 0.52, 33):
        for td in np.linspace(tc + 0.02, 0.70, 33):
            f1, cov = eval_3s(y_sub, oof_prob, tc, td)
            if cov >= 85.0 and f1 > b_f1:
                b_f1, b_cov = f1, cov

    print(f"\n[Overall RTMP SD1 - AUM] F1: {b_f1:.2f}% | Cov: {b_cov:.2f}%")
    
    with open('d:/rtmp_f1_result.txt', 'w') as f:
        f.write(f"{b_f1:.2f}\n{b_cov:.2f}")

if __name__ == '__main__':
    main()
