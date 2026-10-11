import os
import glob
import json
import numpy as np

split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
sq_mask = (m['y_action'] == 0)
sids = m['sequence_ids'][sq_mask]
bsids = m['base_sequence_ids'][sq_mask]

for sid, bsid in zip(sids, bsids):
    npz_p = os.path.join(split_dir, f'{sid}.npz')
    jp = os.path.join(json_dir, f'{bsid}_3D.json')
    if not os.path.exists(jp) or not os.path.exists(npz_p):
        continue
    d = np.load(npz_p, allow_pickle=True)
    sf, ef = int(d['orig_start_frame']), int(d['orig_end_frame'])
    j = json.load(open(jp, encoding='utf-8'))
    frs = j['frames'][sf:ef+1]
    if len(frs) < 4:
        print(f"sid: {sid}, sf: {sf}, ef: {ef}, len(frs): {len(frs)}, total_frames: {len(j['frames'])}")
        # 만약 sf와 ef가 너무 작으면(예: 3프레임), j['frames'] 전체에서 어떤 프레임들인지 보자
        break
