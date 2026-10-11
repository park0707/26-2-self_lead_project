# -*- coding: utf-8 -*-
import os, sys, pickle, numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

CURRENT_DIR = r"D:\src\models\second_models\babel_stiff_deadlift"
PROJECT_ROOT = "d:/"
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from sd2_v3_classifier import SD2V3Classifier

# 1. NPZ 데이터 로드
npz_path = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")
data = np.load(npz_path, allow_pickle=True)
seq_ids = data['sequence_ids']
y_true = data['labels']
N = len(y_true)

# 2. 원천 데이터 로드 및 v3 일괄 추론
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"

sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
sids = list(sc['sequence_ids'])
s_vision = sc['spine_9_64f_vision']
spine_map = {sids[i]: s_vision[i] for i in range(len(sids))}

clf = SD2V3Classifier()
probs = np.zeros(N, dtype=np.float32)

print("[*] 1,035개 전수 시퀀스에 대해 sd2_v3 추론 중...")
for i, sid in enumerate(seq_ids):
    fpath = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
    raw = np.load(fpath, allow_pickle=True)
    kps = raw['kps_33_3d_64f'] if 'kps_33_3d_64f' in raw else raw['kps_33_2d_64f']
    s_pts = spine_map.get(sid, None)
    probs[i] = clf.predict_proba(kps, s_pts)

# 현재 모델 번들에 저장된 공식 임계치 기준 (tau_clean=0.39, tau_def=0.59)
t_clean = clf.tau_clean
t_def = clf.tau_defect

# 3-State 판정: 0: CLEAN, 1: DEFECT, 2: RETAKE (보류)
preds_3state = np.full(N, 2, dtype=int)
preds_3state[probs < t_clean] = 0
preds_3state[probs >= t_def] = 1

determined_mask = (preds_3state != 2)
cov = np.mean(determined_mask)
abstain_rate = 1.0 - cov

det_y = y_true[determined_mask]
det_pred = preds_3state[determined_mask]

f1_3s = f1_score(det_y, det_pred, zero_division=0)
acc_3s = accuracy_score(det_y, det_pred)
prec_3s = precision_score(det_y, det_pred, zero_division=0)
rec_3s = recall_score(det_y, det_pred, zero_division=0)

# 전체 이진 기준 (0.50 기준)
bin_pred = (probs >= 0.50).astype(int)
f1_bin = f1_score(y_true, bin_pred, zero_division=0)
acc_bin = accuracy_score(y_true, bin_pred)

print("\n" + "=" * 80)
print(f"  [sd2_v3 전체 1,035개 전수 단일 통합 3-State 평가 결과]")
print("=" * 80)
print(f"  - 3-State 확정 결함 F1-Score : {f1_3s*100:.2f}%")
print(f"  - 보류율 (Abstain Rate)       : {abstain_rate*100:.2f}% (총 {np.sum(~determined_mask)}개 보류)")
print(f"  - 판정 커버리지 (Coverage)    : {cov*100:.2f}% (총 {np.sum(determined_mask)}개 확정)")
print(f"  - 3-State 확정 정확도 (Acc)   : {acc_3s*100:.2f}%")
print(f"  - 3-State 확정 정밀도 (Prec)  : {prec_3s*100:.2f}%")
print(f"  - 3-State 확정 재현율 (Rec)   : {rec_3s*100:.2f}%")
print(f"  - 이진 판정(미보류) F1-Score   : {f1_bin*100:.2f}% (전체 정확도: {acc_bin*100:.2f}%)")
print(f"  - 적용 임계치                 : Clean < {t_clean:.2f}, Defect >= {t_def:.2f}")
