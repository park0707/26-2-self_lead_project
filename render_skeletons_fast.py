import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    views = df['view'].values
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    return df, np.array(seq_ids), views, y_raw

def main():
    print("Loading data...")
    df, seq_ids, views, y_raw = load_dataset()
    y_clean = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\y_clean.npy')

    # Select ONE side view (View 4: 좌측 전방 45도 - 측면)
    mask_side = (views == 'view4')
    
    noise_fp = (y_raw == 1) & (y_clean == 0) & mask_side # AIHub 1, Phys 0
    noise_fn = (y_raw == 0) & (y_clean == 1) & mask_side # AIHub 0, Phys 1
    clean_tp = (y_raw == 1) & (y_clean == 1) & mask_side # AIHub 1, Phys 1
    clean_tn = (y_raw == 0) & (y_clean == 0) & mask_side # AIHub 0, Phys 0
    
    idx_fp = np.where(noise_fp)[0]
    idx_fn = np.where(noise_fn)[0]
    idx_tp = np.where(clean_tp)[0][:5] # just 5 for comparison
    idx_tn = np.where(clean_tn)[0][:5] # just 5 for comparison
    
    out_dir = r"D:\Data\temp"
    os.makedirs(out_dir, exist_ok=True)
    
    # Empty temp dir first
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            try:
                os.remove(os.path.join(out_dir, f))
            except: pass
            
    # Hardcoded dir (Windows path mojibake could happen, let's just use os.listdir if we want, but D:\Data\... is fine if encoded as utf-8 and run properly)
    dir1 = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    dir2 = r"D:\Data\시니어_피트니스\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
    
    # Check what directories exist:
    search_dirs = []
    if os.path.exists(dir1): search_dirs.append(dir1)
    if os.path.exists(dir2): search_dirs.append(dir2)
    # also add a glob to find exactly the dir if there's mojibake
    import glob
    for d in glob.glob(r"D:\Data\*\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"):
        search_dirs.append(d)
        
    connections = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
        (11, 23), (12, 24), (23, 24)
    ]
    
    def render_and_save(indices, prefix):
        count = 0
        for i in indices:
            sid = seq_ids[i]
            p = None
            for d in search_dirs:
                temp_p = os.path.join(d, sid + '.npz')
                if os.path.exists(temp_p):
                    p = temp_p
                    break
            
            if p is not None and os.path.exists(p):
                kps = np.load(p)['kps_26_64f']
                frame = kps[32] # Middle frame
                
                plt.figure(figsize=(4, 6))
                plt.gca().invert_yaxis()
                plt.title(f"{prefix}\n{sid}", fontsize=8)
                
                for (u, v) in connections:
                    if u < len(frame) and v < len(frame):
                        plt.plot([frame[u, 0], frame[v, 0]], [frame[u, 1], frame[v, 1]], 'b-', alpha=0.5)
                
                # Highlight knees
                plt.plot([frame[11, 0], frame[13, 0]], [frame[11, 1], frame[13, 1]], 'r-', linewidth=3)
                plt.plot([frame[13, 0], frame[15, 0]], [frame[13, 1], frame[15, 1]], 'r-', linewidth=3)
                plt.plot([frame[12, 0], frame[14, 0]], [frame[12, 1], frame[14, 1]], 'r-', linewidth=3)
                plt.plot([frame[14, 0], frame[16, 0]], [frame[14, 1], frame[16, 1]], 'r-', linewidth=3)
                
                out_path = os.path.join(out_dir, f"{prefix}_{sid}.jpg")
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                count += 1
        print(f"Rendered {count} images for {prefix}.")

    print("Rendering images to D:\\Data\\temp ...")
    render_and_save(idx_fp, "1_Noise_AIHub결함_실제정상") # prefixed with numbers to sort nicely
    render_and_save(idx_fn, "2_Noise_AIHub정상_실제결함")
    render_and_save(idx_tn, "3_비교용_확실한정상")
    render_and_save(idx_tp, "4_비교용_확실한결함")
    print("Done! All images saved to D:\\Data\\temp")

if __name__ == '__main__':
    main()
