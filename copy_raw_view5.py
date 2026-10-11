import os
import shutil
import pickle
import numpy as np
import glob

def main():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    
    # Extract unique base names (without view) to group the 1035 sequences
    # A seq_id looks like: [원]_D02_1_Key210_view1
    # We want to identify the exact 405 noise cases from the dataset (which might be any view)
    # Actually, the noise is physical - if the actor didn't bend their knee, they didn't bend it in ANY view.
    # So we can just take the 405 seq_ids, replace their view with 'view5', and use that!
    
    seq_ids = np.array([os.path.splitext(os.path.basename(p))[0] for p in df['path']])
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    y_clean = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\y_clean.npy')

    # Find the indices of noise and clean cases
    # We will pick unique base movements to avoid copying the same movement 5 times
    base_movements = np.array([s.rsplit('_view', 1)[0] for s in seq_ids])
    
    fp_bases = list(set(base_movements[(y_raw == 1) & (y_clean == 0)]))
    fn_bases = list(set(base_movements[(y_raw == 0) & (y_clean == 1)]))
    tp_bases = list(set(base_movements[(y_raw == 1) & (y_clean == 1)]))[:5]
    tn_bases = list(set(base_movements[(y_raw == 0) & (y_clean == 0)]))[:5]
    
    out_dir = r"D:\Data\temp"
    # Clean temp dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    cat_dirs = {
        'fp': os.path.join(out_dir, "1_라벨결함_실제정상_오류"),
        'fn': os.path.join(out_dir, "2_라벨정상_실제결함_오류"),
        'tp': os.path.join(out_dir, "3_비교_확실한결함"),
        'tn': os.path.join(out_dir, "4_비교_확실한정상")
    }
    for d in cat_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    RAW_BASE = r"D:\Data\01_raw_data\images\03_바벨_스티프_데드리프트"
    # It might be nested, e.g., images\03_바벨_스티프_데드리프트\Day01\...
    
    def copy_view5_frames(bases, cat_key):
        count = 0
        for base in bases:
            view5_sid = base + "_view5"
            # Fast recursive search using os.walk in the raw images dir
            found_dir = None
            for root, dirs, files in os.walk(r"D:\Data\01_raw_data"):
                if view5_sid in dirs:
                    found_dir = os.path.join(root, view5_sid)
                    break
            
            if found_dir is None:
                # Try fallback, maybe the folder name is slightly different
                for root, dirs, files in os.walk(r"D:\Data"):
                    if view5_sid in dirs and 'images' in root:
                        found_dir = os.path.join(root, view5_sid)
                        break
                        
            if found_dir:
                frames = sorted(glob.glob(os.path.join(found_dir, "*.jpg")))
                if not frames:
                    frames = sorted(glob.glob(os.path.join(found_dir, "*.png")))
                    
                if frames:
                    # Select exactly 16 frames evenly distributed
                    if len(frames) > 16:
                        idx = np.round(np.linspace(0, len(frames) - 1, 16)).astype(int)
                        frames = [frames[i] for i in idx]
                    
                    target_dir = os.path.join(cat_dirs[cat_key], view5_sid)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    for i, f in enumerate(frames):
                        shutil.copy2(f, os.path.join(target_dir, f"frame_{i:02d}.jpg"))
                    count += 1
            else:
                pass # print(f"Could not find raw images for {view5_sid}")
        print(f"[{cat_key}] Copied {count} sequences.")

    print("Copying 16 frames for each View A (view5) sequence...")
    copy_view5_frames(fp_bases, 'fp')
    copy_view5_frames(fn_bases, 'fn')
    copy_view5_frames(tp_bases, 'tp')
    copy_view5_frames(tn_bases, 'tn')
    print("Done!")

if __name__ == '__main__':
    main()
