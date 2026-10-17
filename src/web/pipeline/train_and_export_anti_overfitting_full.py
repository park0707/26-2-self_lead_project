# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/train_and_export_anti_overfitting_full.py
Purpose: 전체 8대 운동 33종 세부 결함 모델 100% Full-Dataset 정규화 재학습 및
         프로덕션 서빙용 단일 마스터 가중치 번들(anti_overfitting_master_bundle.pkl) 빌드 파이프라인
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = "D:/"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
WEIGHTS_DIR = os.path.join(WEB_ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

for p in [PROJECT_ROOT, SRC_ROOT, WEB_ROOT, r"D:\src\models\second_models\babel_deadlift", r"D:\src\models\second_models\cross_lunge"]:
    if p not in sys.path:
        sys.path.insert(0, p)

FEATS_DIR = r"D:\Data\피처_데이터"
OUTPUT_BUNDLE = os.path.join(WEIGHTS_DIR, "anti_overfitting_master_bundle.pkl")
OUTPUT_METADATA_JSON = os.path.join(WEIGHTS_DIR, "anti_overfitting_master_metadata.json")

print("=" * 100)
print(" [vidio-health-trainer] 33종 세부 결함 모델 100% Full-Dataset 가중치 생성 및 마스터 번들 빌드")
print("=" * 100)

def train_full_model(X_raw: np.ndarray, y_target: np.ndarray, diet_k: int, name: str, code: str):
    """100% 전수 데이터 적합 및 최적 임계값 도출 함수"""
    X_clean = np.nan_to_num(X_raw, nan=0.0)
    y = np.array(y_target, dtype=np.int32)
    
    # 1. Feature Diet (SelectKBest)
    k_dim = min(diet_k, X_clean.shape[1])
    selector = SelectKBest(f_classif, k=k_dim)
    selector.fit(X_clean, y)
    selected_indices = selector.get_support(indices=True)
    X_diet = X_clean[:, selected_indices]
    
    # 2. Gaussian Jitter Noise Injection for Generalization
    np.random.seed(42)
    noise = np.random.normal(0, 0.02 * np.std(X_diet, axis=0, keepdims=True), size=X_diet.shape)
    X_aug = X_diet + noise
    
    # 3. Regularized Classifiers
    cb = CatBoostClassifier(
        iterations=150,
        depth=3,
        learning_rate=0.035,
        l2_leaf_reg=12.0,
        subsample=0.80,
        rsm=0.80,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False,
        thread_count=1
    )
    cb.fit(X_aug, y)
    
    lgbm = lgb.LGBMClassifier(
        n_estimators=110,
        max_depth=3,
        num_leaves=8,
        min_child_samples=30,
        learning_rate=0.035,
        reg_alpha=2.0,
        reg_lambda=10.0,
        colsample_bytree=0.75,
        subsample=0.80,
        subsample_freq=1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=1
    )
    lgbm.fit(X_aug, y)
    
    # 4. Predict probabilities on clean dataset
    p_cb = cb.predict_proba(X_diet)[:, 1]
    p_lgb = lgbm.predict_proba(X_diet)[:, 1]
    p_ens = 0.50 * p_cb + 0.50 * p_lgb
    
    # 5. Optimal Threshold Calibration (F1-Score Search)
    best_t = 0.50
    best_f1 = 0.0
    for t_cand in np.arange(0.30, 0.72, 0.02):
        f1_cand = f1_score(y, (p_ens >= t_cand).astype(int), zero_division=0)
        if f1_cand > best_f1:
            best_f1 = f1_cand
            best_t = float(t_cand)
            
    # Margin for 3-state decisions
    tau_defect = float(min(0.68, max(0.48, best_t + 0.05)))
    tau_normal = float(max(0.30, min(0.48, best_t - 0.05)))
    
    # Metrics
    final_pred = (p_ens >= best_t).astype(int)
    f1 = float(f1_score(y, final_pred, zero_division=0) * 100)
    acc = float(accuracy_score(y, final_pred) * 100)
    auc = float(roc_auc_score(y, p_ens) * 100) if len(np.unique(y)) > 1 else 100.0
    
    model_entry = {
        'code': code,
        'name': name,
        'raw_dim': int(X_clean.shape[1]),
        'diet_dim': int(len(selected_indices)),
        'selected_indices': selected_indices.tolist(),
        'cb_model': cb,
        'lgb_model': lgbm,
        'thresholds': {
            'tau_opt': round(best_t, 3),
            'tau_defect': round(tau_defect, 3),
            'tau_normal': round(tau_normal, 3)
        },
        'metrics': {
            'f1': round(f1, 2),
            'accuracy': round(acc, 2),
            'auc': round(auc, 2),
            'samples': int(len(y)),
            'positives': int(np.sum(y))
        }
    }
    
    print(f"  [{code.upper():<3}] {name[:22]:<22} | N={len(y):>5} | {X_clean.shape[1]:>2}D->{len(selected_indices):>2}D | F1={f1:5.1f}% | AUC={auc:5.1f}% | Tau=[{tau_normal:.2f}, {tau_defect:.2f}]")
    return model_entry

master_bundle = {}
metadata_summary = {}

# ==============================================================================
# 01. 바벨 스쿼트 (c1 ~ c4)
# ==============================================================================
print("\n[1/8] 01. 바벨 스쿼트 (c1~c4) Full-Fit...")
master_mp_sq = np.load(os.path.join(FEATS_DIR, "Master_Squat_MediaPipe33_64F.npz"), allow_pickle=True)
master_rtm_sq = np.load(os.path.join(FEATS_DIR, "Master_Squat_RTMPose26_64F.npz"), allow_pickle=True)
y_sq = master_mp_sq['Y_defects_golden']

sq_configs = [
    ('c1', '척추의 중립 위반 (Spine Neutrality)', master_rtm_sq['X_c1'], y_sq[:, 0], 14),
    ('c2', '고개 정면 및 시선 (Head/Gaze Orientation)', master_rtm_sq['X_c2'], y_sq[:, 1], 10),
    ('c3', '무릎-발 방향 일치 (Knee-Toe Alignment)', master_rtm_sq['X_c3'], y_sq[:, 2], 10),
    ('c4', '발바닥 지면 고정 (Foot Grounding)', master_rtm_sq['X_c4'], y_sq[:, 3], 12)
]
for code, name, X, y, k in sq_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 02. 바벨 데드리프트 (d1 ~ d5)
# ==============================================================================
print("\n[2/8] 02. 바벨 데드리프트 (d1~d5) Full-Fit...")
SCRATCH_DIR = r"C:\Users\phj20\.gemini\antigravity\brain\84ac3b12-59f7-489a-ac16-0eaf3a15d6fe\scratch"
dl_cache = os.path.join(SCRATCH_DIR, "dl_feats_cache.npz")
cached_dl = np.load(dl_cache)
meta_dl = np.load(r"D:\src\models\second_models\babel_deadlift\deadlift_5models_oof_cache.npz", allow_pickle=True)
y_dl = meta_dl['y']

dl_configs = [
    ('d1', '척추의 중립 위반 (Spine Neutrality)', cached_dl['d1'], y_dl[:, 0], 12),
    ('d2', '무릎-발목 정렬 모임/외반 (Knee Medial Collapse)', cached_dl['d2'], y_dl[:, 1], 8),
    ('d3', '발바닥 접지 및 힙 슈트업 (Foot Ground Fixation)', cached_dl['d3'], y_dl[:, 2], 10),
    ('d4', '무릎-골반 동시 신전 불일치 (Knee-Hip Desynchronization)', cached_dl['d4'], y_dl[:, 3], 10),
    ('d5', '바벨 궤적 신체 밀착 실패 (Barbell Path Proximity)', cached_dl['d5'], y_dl[:, 4], 16)
]
for code, name, X, y, k in dl_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 03. 바벨 스티프 데드리프트 (sd1 ~ sd3)
# ==============================================================================
print("\n[3/8] 03. 바벨 스티프 데드리프트 (sd1~sd3) Full-Fit...")
base_sd = r"D:\src\models\second_models\babel_stiff_deadlift"
with open(os.path.join(base_sd, "stiff_d1_v3_dataset_cache.pkl"), "rb") as f:
    df_sd = pickle.load(f)['df']
npz_sd = np.load(os.path.join(base_sd, "stiff_rtmpose_features_cache.npz"))

keys_sd = df_sd['key'].values
y_sd1 = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys_sd], dtype=np.int32)
y_sd2 = np.array([1 if k in ['Key211', 'Key212', 'Key213', 'Key214', 'Key215', 'Key216'] else 0 for k in keys_sd], dtype=np.int32)
y_sd3 = np.array([1 if k in ['Key210', 'Key213', 'Key214'] else 0 for k in keys_sd], dtype=np.int32)

sd_configs = [
    ('sd1', '무릎 구부린 채 고정 위반 (Knee Fixation Violation)', npz_sd['X_sd1'], y_sd1, 14),
    ('sd2', '척추의 중립 위반 (Spine Neutrality)', npz_sd['X_sd2'], y_sd2, 14),
    ('sd3', '바벨 궤적 신체 밀착 실패 (Barbell Path Proximity)', npz_sd['X_sd3'], y_sd3, 16)
]
for code, name, X, y, k in sd_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 04. 바벨 런지 (bl1 ~ bl5)
# ==============================================================================
print("\n[4/8] 04. 바벨 런지 (bl1~bl5) Full-Fit...")
bl1_data = pickle.load(open(os.path.join(FEATS_DIR, "bl1_v4_dataset_cache.pkl"), "rb"))
bl2_data = np.load(r"D:\src\models\second_models\babel_lunge\lunge_c2_features_hybrid_2_cache.npz")
bl3_data = np.load(os.path.join(FEATS_DIR, "lunge_c3_dataset_14d.npz"))
bl4_data = np.load(os.path.join(FEATS_DIR, "lunge_c4_dataset_cache_v4.npz"))
bl5_data = np.load(os.path.join(FEATS_DIR, "lunge_c5_hybrid_1_79d_features.npz"))

bl_configs = [
    ('bl1', '앞다리 무릎 각도 90도 (Lead Knee 90°)', bl1_data['X'], bl1_data['y'], 14),
    ('bl2', '몸통 발 무릎 방향 일치 (Torso-Foot-Knee Alignment)', bl2_data['X'], bl2_data['y'], 16),
    ('bl3', '뒤다리 무릎 각도 90도 (Rear Knee 90°)', bl3_data['X'], bl3_data['y'], 10),
    ('bl4', '척추의 중립 위반 (Spine Neutrality)', bl4_data['X_30d'], bl4_data['y'], 14),
    ('bl5', '상체 과도 숙임/젖힘 (Torso Pitch/Lean)', bl5_data['X'], bl5_data['y'], 24)
]
for code, name, X, y, k in bl_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 05. 다이나믹 런지 (dl1 ~ dl5)
# ==============================================================================
print("\n[5/8] 05. 다이나믹 런지 (dl1~dl5) Full-Fit...")
dl1_data = np.load(os.path.join(FEATS_DIR, "dl1_rtmp_1_dataset_cache.npz"))
dl2_data = np.load(r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz")
dl3_data = np.load(os.path.join(FEATS_DIR, "dynamic_lunge_dl3_rtmp_27d_cache.npz"))
dl4_data = np.load(os.path.join(FEATS_DIR, "dynamic_lunge_c4_dataset_cache.npz"))
dl5_data = np.load(os.path.join(FEATS_DIR, "dynamic_lunge_c5_rtmp_dataset.npz"))

dl_configs = [
    ('dl1', '앞다리 무릎 각도 90도 (Lead Knee 90°)', dl1_data['X'], dl1_data['y'], 14),
    ('dl2', '몸통 발 무릎 방향 일치 (Torso-Foot-Knee Alignment)', dl2_data['X'], dl2_data['y'], 12),
    ('dl3', '뒤다리 무릎 각도 90도 (Rear Knee 90°)', dl3_data['X'], dl3_data['y'], 12),
    ('dl4', '척추의 중립 위반 (Spine Neutrality)', dl4_data['X_24d'], dl4_data['y'], 12),
    ('dl5', '상체 과도 숙임/젖힘 (Torso Pitch/Lean)', dl5_data['X_hybrid_79d'], dl5_data['y'], 24)
]
for code, name, X, y, k in dl_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 06. 사이드 런지 (sl1 ~ sl5)
# ==============================================================================
print("\n[6/8] 06. 사이드 런지 (sl1~sl5) Full-Fit...")
sl_data = np.load(os.path.join(FEATS_DIR, "side_lunge_rtmp_features.npz"))

sl_configs = [
    ('sl1', '측면 굴곡 무릎 발끝 초과 (Lead Knee Over Toe)', sl_data['X_c1'], sl_data['y_c1'], 20),
    ('sl2', '신전 다리 무릎 굽힘 (Trail Leg Knee Extension)', sl_data['X_c2'], sl_data['y_c2'], 20),
    ('sl3', '골반 측면 회전 및 비대칭 (Pelvis Asymmetry)', sl_data['X_c3'], sl_data['y_c3'], 14),
    ('sl4', '상체 과도 측면 붕괴 (Torso Lateral Collapse)', sl_data['X_c4'], sl_data['y_c4'], 12),
    ('sl5', '접지발 안쪽 들림 (Foot Inversion/Eversion)', sl_data['X_c5'], sl_data['y_c5'], 18)
]
for code, name, X, y, k in sl_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 07. 크로스 런지 (cl1 ~ cl3)
# ==============================================================================
print("\n[7/8] 07. 크로스 런지 (cl1~cl3) Full-Fit...")
cl_opt = np.load(os.path.join(FEATS_DIR, "cross_lunge_rtmp_optimized_features_cache.npz"))
from run_cross_lunge_rtmp_optimized_benchmark import parse_labels_and_metadata
y1_cl, y2_cl, y3_cl, _, _ = parse_labels_and_metadata(cl_opt['seq_ids'])

cl_configs = [
    ('cl1', '앞다리 무릎 각도 90도 (Lead Knee 90°)', cl_opt['X_c1'], np.array(y1_cl, dtype=np.int32), 24),
    ('cl2', '앞다리 무릎 안쪽 무너짐 (Lead Knee Valgus Collapse)', cl_opt['X_c2'], np.array(y2_cl, dtype=np.int32), 24),
    ('cl3', '교차 다리 궤적 이탈 (Cross Alignment Deficit)', cl_opt['X_c3'], np.array(y3_cl, dtype=np.int32), 18)
]
for code, name, X, y, k in cl_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# ==============================================================================
# 08. 바벨 굿모닝 (gm1 ~ gm3)
# ==============================================================================
print("\n[8/8] 08. 바벨 굿모닝 (gm1~gm3) Full-Fit...")
gm_data = np.load(os.path.join(FEATS_DIR, "good_morning_rtmpose_features_cache.npz"))

gm_configs = [
    ('gm1', '무릎 구부린 채 고정 위반 (Knee Fixed Angle Violation)', gm_data['X_c1'], gm_data['y_c1'], 8),
    ('gm2', '척추의 중립 위반 (Spine Neutrality)', gm_data['X_c2'], gm_data['y_c2'], 10),
    ('gm3', '고관절 힌지 깊이 미흡 (Hip Hinge Depth Deficit)', gm_data['X_c3'], gm_data['y_c3'], 16)
]
for code, name, X, y, k in gm_configs:
    entry = train_full_model(X, y, k, name, code)
    master_bundle[code] = entry
    metadata_summary[code] = {k: v for k, v in entry.items() if k not in ['cb_model', 'lgb_model']}

# Save Master Bundle
with open(OUTPUT_BUNDLE, "wb") as f:
    pickle.dump(master_bundle, f)
print(f"\n[+] 33개 전 모델 마스터 번들 저장 완료: {OUTPUT_BUNDLE} ({os.path.getsize(OUTPUT_BUNDLE)/1024/1024:.2f} MB)")

with open(OUTPUT_METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
print(f"[+] 마스터 메타데이터 JSON 저장 완료: {OUTPUT_METADATA_JSON}")
