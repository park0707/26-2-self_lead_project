import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pickle

CURRENT_DIR = r"D:\src\models\second_models\babel_stiff_deadlift"
NPZ_AMBIGUITY_PATH = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")
BUNDLE_V3_PATH = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")

def eval_3state_full():
    data = np.load(NPZ_AMBIGUITY_PATH, allow_pickle=True)
    labels = data['labels']
    
    with open(BUNDLE_V3_PATH, 'rb') as f:
        bundle = pickle.load(f)
        
    # Wait, bundle doesn't have oof_ens. We need oof_ens to evaluate the exact same threshold.
    # We can approximate by inferencing with the models on all data, but it won't be strict OOF.
    # Or I can just patch train_and_eval_sd2_v3_twotier.py to save oof_ens or just run the 3-state part.
    pass
    
if __name__ == '__main__':
    eval_3state_full()
