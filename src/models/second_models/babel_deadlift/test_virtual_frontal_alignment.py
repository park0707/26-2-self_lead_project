# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_virtual_frontal_alignment.py
Purpose: 3D 가상 정면 표준화(Virtual Frontal Canonical Alignment)를 적용한 피처 추출 및 시점 불변성 검증
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.ndimage import gaussian_filter1d

sys.stdout.reconfigure(encoding='utf-8')

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))


def transform_to_canonical_frontal(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    모든 프레임에서 인체 중심 가상 정면(Virtual Frontal) 직교 좌표계로 정렬
    - 원점: 골반 중심 (Mid-Hip)
    - X축: 우골반 -> 좌골반 (인체 정면 관상면 수평 방향)
    - Y축: 골반 중심 -> 어깨 중심 (체간 수직 방향)
    - Z축: X와 Y의 외적 (인체 전방 방향)
    """
    T = kps_33_64f.shape[0]
    out = np.zeros_like(kps_33_64f)

    for t in range(T):
        pts = kps_33_64f[t]  # (33, 3)
        l_hip, r_hip = pts[23], pts[24]
        mid_hip = (l_hip + r_hip) / 2.0

        l_sh, r_sh = pts[11], pts[12]
        mid_sh = (l_sh + r_sh) / 2.0

        # 평행이동: mid_hip이 원점
        centered = pts - mid_hip

        v_torso = mid_sh - mid_hip
        y_hat = v_torso / (np.linalg.norm(v_torso) + 1e-7)

        v_hip = l_hip - r_hip
        z_raw = np.cross(y_hat, v_hip)
        z_hat = z_raw / (np.linalg.norm(z_raw) + 1e-7)

        x_hat = np.cross(y_hat, z_hat)
        x_hat = x_hat / (np.linalg.norm(x_hat) + 1e-7)

        R = np.column_stack([x_hat, y_hat, z_hat])  # (3, 3)
        out[t] = np.dot(centered, R)

    return out


def extract_canonical_7d_features(kps_33_64f: np.ndarray) -> np.ndarray:
    """가상 정면 좌표계에서 7D 피처 추출"""
    c_kps = transform_to_canonical_frontal(kps_33_64f)
    T = c_kps.shape[0]
    dt = 3.0 / max(1, T - 1)

    l_hip = c_kps[:, 23, :]
    r_hip = c_kps[:, 24, :]
    l_knee = c_kps[:, 25, :]
    r_knee = c_kps[:, 26, :]
    l_ank = c_kps[:, 27, :]
    r_ank = c_kps[:, 28, :]

    # 가상 정면에서 X좌표(0번 축)는 완벽한 좌우 관상면 위치!
    knee_w = np.abs(l_knee[:, 0] - r_knee[:, 0])
    ank_w = np.abs(l_ank[:, 0] - r_ank[:, 0])
    hip_w = np.abs(l_hip[:, 0] - r_hip[:, 0])
    mean_hip_w = float(np.mean(hip_w) + 1e-6)

    ka_ratio = knee_w / (ank_w + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)
    kp_ratio = knee_w / (hip_w + 1e-6)
    kp_ratio_s = gaussian_filter1d(kp_ratio, sigma=1.0)

    standing_slice = max(2, T // 8)
    kp_base = float(np.mean(kp_ratio_s[:standing_slice]))

    f1_kp_ratio_min = float(np.min(kp_ratio_s))
    f2_ka_deficit_integ = float(np.sum(np.maximum(0.0, 0.90 - ka_ratio_s)))

    # 관상면 X축 투영 외반 변위 (cm)
    vl = (l_knee[:, 0] - l_hip[:, 0]) * 100.0
    vr = (r_hip[:, 0] - r_knee[:, 0]) * 100.0
    valg_sum_abs = np.abs(vl) + np.abs(vr)
    valg_sum_abs_s = gaussian_filter1d(valg_sum_abs, sigma=1.0)
    f3_valg_sum_max = float(np.max(valg_sum_abs_s))

    valg_norm = valg_sum_abs_s / (mean_hip_w * 100.0 + 1e-6)
    f4_valg_norm_max = float(np.max(valg_norm))

    # 정강이 관상면 기울기
    shank_l = l_knee - l_ank
    shank_r = r_knee - r_ank
    tilt_l = shank_l[:, 0] / (np.linalg.norm(shank_l, axis=1) + 1e-7)
    tilt_r = -shank_r[:, 0] / (np.linalg.norm(shank_r, axis=1) + 1e-7)
    tilt_sum = np.maximum(0.0, -tilt_l) + np.maximum(0.0, -tilt_r)
    f5_shank_tilt_p90 = float(np.percentile(tilt_sum, 90))

    d_valg = np.gradient(valg_norm, dt)
    d2_valg = np.gradient(d_valg, dt)
    f6_valg_accel_peak = float(np.max(np.abs(d2_valg)))

    f7_kp_drop_max = float(max(0.0, kp_base - np.min(kp_ratio_s)))

    return np.array([
        f1_kp_ratio_min, f2_ka_deficit_integ, f3_valg_sum_max,
        f4_valg_norm_max, f5_shank_tilt_p90, f6_valg_accel_peak, f7_kp_drop_max
    ], dtype=np.float32)


def main():
    print("[*] 3D 가상 정면 표준화 피처 추출 중...")
    X_list = []
    y_list = []
    subs_list = []
    views_list = []

    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        parts = base.split("_")
        vn = parts[-1]
        folder = "_".join(parts[:-1])

        sub = 1
        for i in range(1, 10):
            if f"피험자{i}" in folder:
                sub = i
                break

        with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
            d = json.load(fp)
        conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
        y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

        kps = np.load(p)["kps_33_3d_64f"]
        f_can = extract_canonical_7d_features(kps)

        X_list.append(f_can)
        y_list.append(y_val)
        subs_list.append(sub)
        views_list.append(vn)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs_list, dtype=np.int32)
    views = np.array(views_list)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]

        oof_preds[val_idx] = 0.5 * p_l + 0.5 * p_c

    auc_can = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 90)
    print(f"[*] 3D 가상 정면 표준화 전체 OOF ROC-AUC: {auc_can:.4f}")

    # 시점별 성능 검증
    VIEW_ALIAS_MAP = {
        "view5": "시점 A (우측 후방 45°)",
        "view2": "시점 B (우측 전방 45°)",
        "view3": "시점 C (정면)",
        "view4": "시점 D (좌측 전방 45°)",
        "view1": "시점 E (좌측 후방 45°)"
    }

    best_th = 0.37
    p_opt = (oof_preds >= best_th).astype(int)
    print(f"\n[5대 시점별 세부 AUC 및 F1 (th={best_th:.2f})]:")
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        mask = (views == v_code)
        y_v = y[mask]
        p_v = p_opt[mask]
        prob_v = oof_preds[mask]
        v_auc = roc_auc_score(y_v, prob_v) if len(np.unique(y_v)) > 1 else 0.5
        v_f1 = f1_score(y_v, p_v, zero_division=0) * 100
        print(f"  {v_alias:<22} | AUC: {v_auc:.4f} | F1: {v_f1:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
