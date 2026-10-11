import os
import pickle
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

# Load RTMP features
d = np.load("d:/rtmp_features.npz")
X_c1 = d['X_c1']
X_c2 = d['X_c2']
X_c4 = d['X_c4']
X_c5 = d['X_c5']
y = d['y']
views = d['views']
subjects = d['subjects']

results = {}

# Evaluate bl1_v4
try:
    with open(r"D:\src\models\second_models\babel_lunge\bl1_v4_bundle.pkl", "rb") as f:
        bl1_b = pickle.load(f)
    preds1 = []
    for model in bl1_b['models']:
        preds1.append(model.predict_proba(X_c1)[:, 1])
    p1 = np.mean(preds1, axis=0)
    p1_cls = (p1 > 0.5).astype(int)
    results['bl1_v4'] = {
        'f1': f1_score(y, p1_cls),
        'auc': roc_auc_score(y, p1)
    }
except Exception as e:
    results['bl1_v4'] = str(e)

# Evaluate bl2_v6
try:
    with open(r"D:\src\models\second_models\babel_lunge\weights\bl2_v6_bundle.pkl", "rb") as f:
        bl2_b = pickle.load(f)
    if 'models' in bl2_b:
        preds2 = []
        for model in bl2_b['models']:
            preds2.append(model.predict_proba(X_c2)[:, 1])
        p2 = np.mean(preds2, axis=0)
        p2_cls = (p2 > 0.5).astype(int)
        results['bl2_v6'] = {
            'f1': f1_score(y, p2_cls),
            'auc': roc_auc_score(y, p2)
        }
except Exception as e:
    results['bl2_v6'] = str(e)

# Evaluate bl4_v6_flip
try:
    with open(r"D:\src\models\second_models\babel_lunge\bl4_v6_flip_models_cl.pkl", "rb") as f:
        bl4_b = pickle.load(f)
    preds4 = []
    # If it's just a list of models:
    models4 = bl4_b if isinstance(bl4_b, list) else bl4_b.get('models', [])
    for model in models4:
        preds4.append(model.predict_proba(X_c4)[:, 1])
    p4 = np.mean(preds4, axis=0)
    p4_cls = (p4 > 0.5).astype(int)
    results['bl4_v6_flip'] = {
        'f1': f1_score(y, p4_cls),
        'auc': roc_auc_score(y, p4)
    }
except Exception as e:
    results['bl4_v6_flip'] = str(e)

# Evaluate bl5_v5
try:
    with open(r"D:\src\models\second_models\babel_lunge\weights\bl5_v5_bundle.pkl", "rb") as f:
        bl5_b = pickle.load(f)
    preds5 = []
    models5 = bl5_b['models'] if 'models' in bl5_b else bl5_b
    for model in models5:
        preds5.append(model.predict_proba(X_c5)[:, 1])
    p5 = np.mean(preds5, axis=0)
    p5_cls = (p5 > 0.5).astype(int)
    results['bl5_v5'] = {
        'f1': f1_score(y, p5_cls),
        'auc': roc_auc_score(y, p5)
    }
except Exception as e:
    results['bl5_v5'] = str(e)

with open("d:/rtmp_eval_results.json", "w", encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
