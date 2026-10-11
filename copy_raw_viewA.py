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
    
    # seq_id: e.g., '[원]_D02_남1_Key210_view1'
    seq_ids = np.array([os.path.splitext(os.path.basename(p))[0] for p in df['path']])
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    y_clean = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\y_clean.npy')

    # Get base movements (e.g. '[원]_D02_남1_Key210')
    base_movements = np.array([s.rsplit('_view', 1)[0] for s in seq_ids])
    
    fp_bases = list(set(base_movements[(y_raw == 1) & (y_clean == 0)]))
    fn_bases = list(set(base_movements[(y_raw == 0) & (y_clean == 1)]))
    tp_bases = list(set(base_movements[(y_raw == 1) & (y_clean == 1)]))[:5]
    tn_bases = list(set(base_movements[(y_raw == 0) & (y_clean == 0)]))[:5]
    
    out_dir = r"D:\Data\temp"
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
    
    def copy_viewA_frames(bases, cat_key):
        count = 0
        for base in bases:
            folder_path = os.path.join(RAW_BASE, base)
            if not os.path.exists(folder_path):
                continue
                
            # Look for 시점A images
            search_pattern = os.path.join(folder_path, "*_시점A_*.jpg")
            frames = sorted(glob.glob(search_pattern))
            
            if not frames:
                # Some datasets might use PNG or different casing
                search_pattern = os.path.join(folder_path, "*_시점A_*.png")
                frames = sorted(glob.glob(search_pattern))
                
            if frames:
                # Select exactly 16 frames evenly if there are more
                if len(frames) > 16:
                    idx = np.round(np.linspace(0, len(frames) - 1, 16)).astype(int)
                    frames = [frames[i] for i in idx]
                
                target_dir = os.path.join(cat_dirs[cat_key], base)
                os.makedirs(target_dir, exist_ok=True)
                
                for i, f in enumerate(frames):
                    shutil.copy2(f, os.path.join(target_dir, f"frame_{i+1:02d}.jpg"))
                count += 1
        print(f"[{cat_key}] Copied {count} sequences.")

    print("Copying 16 raw frames for each 시점A sequence...")
    copy_viewA_frames(fp_bases, 'fp')
    copy_viewA_frames(fn_bases, 'fn')
    copy_viewA_frames(tp_bases, 'tp')
    copy_viewA_frames(tn_bases, 'tn')
    print("Done!")

if __name__ == '__main__':
    main()
