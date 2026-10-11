import numpy as np
import json
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

d = np.load("d:/rtmp_features.npz")
X_c1 = d['X_c1']
X_c2 = d['X_c2']
X_c4 = d['X_c4']
X_c5 = d['X_c5']
y = d['y']
subjects = d['subjects']
views = d['views']

# Extract numeric subject IDs
groups = np.array([int("".join(c for c in str(s) if c.isdigit())) for s in subjects])

def train_and_eval(X, y, groups):
    # filter out failed extractions (all zeros)
    valid = np.sum(np.abs(X), axis=1) > 0
    X_v = X[valid]
    y_v = y[valid]
    g_v = groups[valid]
    
    if len(X_v) == 0:
        return 0.0, 0.0
        
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    aucs = []
    for train_idx, val_idx in sgkf.split(X_v, y_v, g_v):
        X_train, y_train = X_v[train_idx], y_v[train_idx]
        X_val, y_val = X_v[val_idx], y_v[val_idx]
        
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        if len(np.unique(y_val)) > 1:
            aucs.append(roc_auc_score(y_val, probs))
        f1s.append(f1_score(y_val, preds))
        
    return np.mean(f1s), np.mean(aucs) if aucs else 0.0

print("Training bl1 (c1)...")
f1_1, auc_1 = train_and_eval(X_c1, y, groups)
print("Training bl2 (c2)...")
f1_2, auc_2 = train_and_eval(X_c2, y, groups)
print("Training bl4 (c4)...")
f1_4, auc_4 = train_and_eval(X_c4, y, groups)
print("Training bl5 (c5)...")
f1_5, auc_5 = train_and_eval(X_c5, y, groups)

res = {
    'bl1_v4': {'f1': f1_1, 'auc': auc_1},
    'bl2_v6': {'f1': f1_2, 'auc': auc_2},
    'bl4_v6_flip': {'f1': f1_4, 'auc': auc_4},
    'bl5_v5': {'f1': f1_5, 'auc': auc_5}
}

with open("d:/rtmp_train_results.json", "w", encoding='utf-8') as f:
    json.dump(res, f, indent=2)
print("Done training.")
