import os
import glob
import json
import numpy as np

m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
sq_mask = (m['y_action'] == 0)
base_sids = set(m['base_sequence_ids'][sq_mask])
json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

found = 0
missing_files = 0
for bsid in base_sids:
    jp = os.path.join(json_dir, f'{bsid}_3D.json')
    if os.path.exists(jp):
        found += 1
    else:
        missing_files += 1

print(f"Base sequence count: {len(base_sids)}, Found JSON: {found}, Missing JSON: {missing_files}")
