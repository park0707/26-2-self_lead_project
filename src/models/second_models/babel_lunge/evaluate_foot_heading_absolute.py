import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# Load cached data
c = np.load("lunge_c2_features_v6_18d_delta_cache.npz", allow_pickle=True)
y = c["y"]
names = c["clip_names"]

# We need the raw sequence data to extract the new feature.
# Let's load the raw features. We'll use the feature extractor from v4 which has detect_lead_leg
from lunge_c2_features_v4 import detect_lead_leg_v4

def evaluate_absolute_foot_heading():
    # Load raw dataset (assuming it is available at a standard location used previously)
    # The cache 'babel_lunge_c2_dataset.npz' typically contains the raw sequences
    try:
        raw_data = np.load("babel_lunge_c2_dataset.npz", allow_pickle=True)
        X_seqs = raw_data["X_seqs"]
        y_labels = raw_data["y"]
    except Exception as e:
        print("Could not load raw dataset:", e)
        return

    features = []
    
    for i, seq in enumerate(X_seqs):
        if seq.shape[-1] == 4:
            seq = seq[:, :, :3]
            
        is_left_lead, t_bot = detect_lead_leg_v4(seq)
        
        hip_l = 23
        hip_r = 24
        
        heel_idx = 29 if is_left_lead else 30
        toe_idx = 31 if is_left_lead else 32
        
        # At t_bot (bottom of the lunge)
        # Calculate body's lateral vector on XZ plane from left hip to right hip
        v_lateral = np.array([seq[t_bot, hip_r, 0] - seq[t_bot, hip_l, 0], 
                              seq[t_bot, hip_r, 2] - seq[t_bot, hip_l, 2]])
        
        norm_lat = np.linalg.norm(v_lateral) + 1e-6
        u_lateral = v_lateral / norm_lat
        
        # Forward vector is orthogonal to lateral. 
        # In MediaPipe, if right is +X and down is +Y, then forward is +Z or -Z depending on camera.
        # Let's just use cross product in 2D: (x, y) -> (-y, x)
        # If lateral is pointing right, forward points "into" the screen or "out" of it.
        # Let's just define the body forward vector.
        u_forward = np.array([-u_lateral[1], u_lateral[0]])
        
        # Foot vector
        v_foot = np.array([seq[t_bot, toe_idx, 0] - seq[t_bot, heel_idx, 0],
                           seq[t_bot, toe_idx, 2] - seq[t_bot, heel_idx, 2]])
        
        norm_foot = np.linalg.norm(v_foot) + 1e-6
        u_foot = v_foot / norm_foot
        
        # Angle between foot and true forward
        dot = np.clip(np.dot(u_foot, u_forward), -1.0, 1.0)
        angle_diff = np.arccos(dot) * 180.0 / np.pi
        
        # Because we might have defined forward backwards depending on the lead leg or facing,
        # we can take the absolute angle from the forward line (so angle_diff or 180 - angle_diff)
        angle_from_line = min(angle_diff, 180.0 - angle_diff)
        
        features.append(angle_from_line)
        
    features = np.array(features)
    
    # Calculate AUC
    # If the foot is NOT pointing straight ahead, it's a defect. So higher angle -> defect (y=1)
    auc = roc_auc_score(y_labels, features)
    
    # Check if negative correlation
    if auc < 0.5:
        auc = 1 - auc
        
    print(f"Absolute Foot Heading Angle Feature AUC: {auc:.4f}")
    
    # Let's compare with the v6 feature (knee displacement vs foot heading)
    from lunge_c2_features_v6 import FEATURE_NAMES_18D
    v6_feats = c["X"]
    v6_auc = roc_auc_score(y, v6_feats[:, 16])
    v6_auc = max(v6_auc, 1 - v6_auc)
    print(f"v6 Feature 17 (Knee vs Foot) AUC: {v6_auc:.4f}")

if __name__ == "__main__":
    evaluate_absolute_foot_heading()
