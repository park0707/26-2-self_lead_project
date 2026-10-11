import os
import sys
import numpy as np
import re

sys.path.insert(0, 'd:/')
from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.second_models.unified_stage2_dataset import UnifiedStage2DatasetLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, "gm1_dataset_cache.npz")

def get_base(sid):
    return re.sub(r'_view\d+$', '', sid)

def main():
    print("[*] GM1 데이터셋 추출 시작...")
    UnifiedStage2DatasetLoader._init_master_and_cache()
    cache = UnifiedStage2DatasetLoader._cache_data
    splits = Stage1DatasetLoader.load_all_dual_splits()

    labels_list = []
    subject_ids_list = []
    view_ids_list = []
    sequence_ids_list = []
    X_list = []

    for sname, sdata in splits.items():
        mask = (sdata['y_action'] == 7)  # Good Morning
        for sid, sub, x in zip(sdata['sequence_ids'][mask], sdata['subjects'][mask], sdata['X_Format_B'][mask]):
            sid_str = str(sid)
            base = get_base(sid_str)
            
            # Label
            if base in cache:
                cond_vals = cache[base].get('cond_vals', [])
                y = cond_vals[0] if len(cond_vals) > 0 else 0
            else:
                y = 0
                
            # View ID
            match = re.search(r'_(view\d+)$', sid_str)
            view_id = match.group(1) if match else "view3"
            
            labels_list.append(y)
            subject_ids_list.append(sub)
            view_ids_list.append(view_id)
            sequence_ids_list.append(sid_str)
            X_list.append(x)

    labels = np.array(labels_list, dtype=np.int32)
    levels = np.zeros_like(labels) # No ambiguity levels defined yet
    l4_mask = np.zeros_like(labels, dtype=bool) # No l4
    subject_ids = np.array(subject_ids_list, dtype=str)
    view_ids = np.array(view_ids_list, dtype=str)
    sequence_ids = np.array(sequence_ids_list, dtype=str)
    X = np.array(X_list, dtype=np.float32)

    np.savez_compressed(
        CACHE_PATH,
        labels=labels,
        levels=levels,
        l4_mask=l4_mask,
        subject_ids=subject_ids,
        view_ids=view_ids,
        sequence_ids=sequence_ids,
        X=X
    )
    print(f"[+] 저장 완료: {CACHE_PATH} (샘플 수: {len(labels)})")

if __name__ == '__main__':
    main()
