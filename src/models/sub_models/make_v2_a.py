import os
import sys

v2_path = r'D:\src\models\sub_models\train_and_eval_gm3_v2.py'
with open(v2_path, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('gm3_v2_champion_bundle.joblib', 'gm3_v2_a_champion_bundle.joblib')
code = code.replace('gm3_v2_performance_report.md', 'gm3_v2_a_performance_report.md')
code = code.replace('gm3_v2', 'gm3_v2_a')
code = code.replace('3-State)', '3-State + SpinePose)')

old_load = '''def load_data():
    print(f"[DEBUG] Inside load_data, DATA_DIR: {DATA_DIR}")
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"[DEBUG] Found {len(npz_files)} npz files.")'''

new_load = '''SPINE_DIR = r"D:\Data\피처_데이터\good_morning_spinepose_denoised_64f.npz"

def load_data():
    print(f"[DEBUG] Inside load_data, DATA_DIR: {DATA_DIR}")
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"[DEBUG] Found {len(npz_files)} npz files.")
    
    try:
        import numpy as np
        spine_data = np.load(SPINE_DIR, allow_pickle=True)
        spine_arr = spine_data['spine_9_64f_vision']
        seq_ids = spine_data['sequence_ids']
        spine_dict = {seq_id: spine_arr[i] for i, seq_id in enumerate(seq_ids)}
    except Exception as e:
        print(f"[!] Error loading SpinePose data: {e}")
        return np.array([]), np.array([]), np.array([]), 0, 0, 0'''

code = code.replace(old_load, new_load)

old_match = '''        except Exception as e:
            continue
            
        json_path = os.path.join(LABEL_DIR, f"{base_seq_id}.json")'''

new_match = '''        except Exception as e:
            continue
            
        fname = os.path.basename(fpath).replace('.npz', '')
        if fname not in spine_dict: continue
        spine_9 = spine_dict[fname]
        if spine_9 is None or spine_9.shape != (64, 9, 3): continue
            
        json_path = os.path.join(LABEL_DIR, f"{base_seq_id}.json")'''

code = code.replace(old_match, new_match)

old_feat = '''            feats_v3 = extract_stiff_spine_features_v3(kps_33_3d_64f)'''
new_feat = '''            feats_v3 = extract_stiff_spine_features_v3(kps_33_3d_64f, spine_9_64f=spine_9)'''

code = code.replace(old_feat, new_feat)

with open(r'D:\src\models\sub_models\train_and_eval_gm3_v2_a.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Done')
