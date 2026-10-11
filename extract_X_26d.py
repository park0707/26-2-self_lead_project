import os
import sys
import pickle
import numpy as np

def main():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    views = df['view'].values
    
    sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
    from sd1_rtmp_2_features import extract_sd1_rtmp_2_features, FEATURE_NAMES_47D_SD1_RTMP_2
    
    # Selected 26 optimized features
    opt_feats = ['knee_angle_bottom_deg', 'knee_hip_rom_ratio', 'hip_knee_velocity_corr', 'mcrp', 'vcrp', 
                 'knee_angular_accel_max', 'knee_accel_sign_changes', 'excess_flexion_integral', 'kadr_max', 
                 'kadr_rom', 'shank_svii', 'norm_knee_disp_y', 'norm_knee_disp_x', 'pli', 'hyperextension_ratio', 
                 'bilateral_knee_asym', 'barbell_knee_clearance_nadir', 'kawr_nadir', 'kawr_rom', 
                 'ankle_dorsiflexion_bottom', 'ankle_dorsiflexion_rom', 'barbell_to_pelvis_drop_ratio', 
                 'torso_pitch_bottom', 'view_code', 'v_oh_2', 'v_oh_4']
                 
    opt_idx = [FEATURE_NAMES_47D_SD1_RTMP_2.index(f) for f in opt_feats]
    
    X_26d = np.zeros((len(seq_ids), 26))
    paths = df['path'].values
    count = 0
    
    for i, sid in enumerate(seq_ids):
        p = paths[i].replace('MediaPipe_Full33_64F', 'RTMPose_Halpe26_64F')
        if os.path.exists(p):
            kps = np.load(p)['kps_26_64f']
            x_47 = extract_sd1_rtmp_2_features(kps, views[i])
            X_26d[i] = x_47[opt_idx]
            count += 1
            
    print(f"Extracted {count} valid sequences to X_26d.npy")
    np.save(r'D:\src\models\second_models\babel_stiff_deadlift\X_26d.npy', X_26d)

if __name__ == '__main__':
    main()
