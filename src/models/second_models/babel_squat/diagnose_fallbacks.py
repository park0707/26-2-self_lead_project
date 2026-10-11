import os
import glob
import json
import numpy as np

m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
sq_mask = (m['y_action'] == 0)
sids = m['sequence_ids'][sq_mask]
bsids = m['base_sequence_ids'][sq_mask]

split_dirs = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')
print("Split dirs:", split_dirs)
json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

no_npz = 0
no_json = 0
short_f = 0
key_err = 0

split_dir = split_dirs[0]

for sid, bsid in zip(sids, bsids):
    npz_p = os.path.join(split_dir, f'{sid}.npz')
    if not os.path.exists(npz_p):
        no_npz += 1
        continue
    jp = os.path.join(json_dir, f'{bsid}_3D.json')
    if not os.path.exists(jp):
        no_json += 1
        continue
    d = np.load(npz_p, allow_pickle=True)
    sf, ef = int(d['orig_start_frame']), int(d['orig_end_frame'])
    try:
        j = json.load(open(jp, encoding='utf-8'))
        frs = j['frames'][sf:ef+1]
        if len(frs) < 4:
            short_f += 1
    except Exception as e:
        key_err += 1

print(f"Total: {len(sids)}, no_npz: {no_npz}, no_json: {no_json}, short_f: {short_f}, key_err: {key_err}")
