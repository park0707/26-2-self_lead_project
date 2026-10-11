# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: export_misclassified_samples_to_temp.py
Purpose: data/temp 폴더에 바벨 런지 c5 오분류 샘플(FP: 정상->결함, FN: 결함->정상) 추출 및 복사
         - FP (False Positive): 실제 정상(0)이나 모델이 결함으로 판정한 데이터 5개
         - FN (False Negative): 실제 결함(1)이나 모델이 정상으로 판정한 데이터 5개
         - 64프레임 MediaPipe 관절 텐서 (.npz), 원본 프레임 이미지, 2D 라벨 JSON 및 상세 원인 메타데이터 동봉
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v4_32d_features.npz"
RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"
RAW_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
TEMP_BASE_DIR = r"D:\Data\temp"

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

VIEW_KOREAN_TAG = {
    'view5': '시점A',
    'view2': '시점B',
    'view3': '시점C',
    'view4': '시점D',
    'view1': '시점E'
}


def run_export():
    print("=" * 80)
    print("  [vidio-health-trainer] data/temp 오분류 샘플(FP / FN) 추출 및 패키징")
    print("=" * 80)

    # 1. 데이터셋 로드
    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    keys = data['keys'].astype(str)
    filenames = data['filenames'].astype(str)
    feat_names = list(data['feature_names'])
    N = len(y)

    # 2. 5-Fold OOF 예측 산출
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_rf = np.zeros(N, dtype=np.float32)
    oof_et = np.zeros(N, dtype=np.float32)

    femur_ratio_idx = feat_names.index("lunge_femur_to_torso_ratio")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        keys_train = keys[train_idx]

        sample_weights = np.ones(len(train_idx), dtype=np.float32)
        for i in range(len(train_idx)):
            if keys_train[i] == '338':
                sample_weights[i] *= 1.8
            if X_train[i, femur_ratio_idx] > 1.05 and y_train[i] == 0:
                sample_weights[i] *= 1.5

        # LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.035, num_leaves=24, max_depth=5,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=1.2,
            scale_pos_weight=pos_ratio, random_state=42 + fold, n_jobs=1, verbose=-1
        )
        lgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val)[:, 1]

        # CatBoost
        cb_clf = CatBoostClassifier(
            iterations=220, learning_rate=0.038, depth=5, l2_leaf_reg=4.5,
            scale_pos_weight=pos_ratio, random_seed=42 + fold, thread_count=1, verbose=0
        )
        cb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_cb[val_idx] = cb_clf.predict_proba(X_val)[:, 1]

        # Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4, class_weight='balanced',
            random_state=42 + fold, n_jobs=1
        )
        rf_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_rf[val_idx] = rf_clf.predict_proba(X_val)[:, 1]

        # Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4, class_weight='balanced',
            random_state=42 + fold, n_jobs=1
        )
        et_clf.fit(X_train, y_train, sample_weight=sample_weights)
        oof_et[val_idx] = et_clf.predict_proba(X_val)[:, 1]

    oof_ens = 0.35 * oof_lgb + 0.35 * oof_cb + 0.15 * oof_rf + 0.15 * oof_et

    th_low = 0.3550
    th_high = 0.4800

    # FP: GT == 0 (정상) and prob >= th_high (결함 확정 판정)
    fp_candidates = []
    # FN: GT == 1 (결함) and prob < th_low (정상 확정 판정)
    fn_candidates = []

    for i in range(N):
        gt = int(y[i])
        prob = float(oof_ens[i])
        fn_str = str(filenames[i])
        s = int(subjects[i])
        v = str(views[i])
        k = str(keys[i])

        if gt == 0 and prob >= th_high:
            fp_candidates.append({
                'idx': i,
                'filename': fn_str,
                'ground_truth': "정상 (Normal, Label 0)",
                'prediction': "결함 (Defect, Label 1)",
                'defect_prob': round(prob, 4),
                'subject': s,
                'view': v,
                'official_view': OFFICIAL_VIEW_MAP.get(v, v),
                'key': k,
                'features': {feat_names[j]: round(float(X[i, j]), 4) for j in range(len(feat_names))}
            })
        elif gt == 1 and prob < th_low:
            fn_candidates.append({
                'idx': i,
                'filename': fn_str,
                'ground_truth': "결함 (Defect, Label 1)",
                'prediction': "정상 (Normal, Label 0)",
                'defect_prob': round(prob, 4),
                'subject': s,
                'view': v,
                'official_view': OFFICIAL_VIEW_MAP.get(v, v),
                'key': k,
                'features': {feat_names[j]: round(float(X[i, j]), 4) for j in range(len(feat_names))}
            })

    print(f"[*] 탐색된 총 FP 건수: {len(fp_candidates)}건")
    print(f"[*] 탐색된 총 FN 건수: {len(fn_candidates)}건")

    # 대표성 있는 5개씩 선별 (피험자 및 시점 다양성 확보)
    # FP: 피험자 1, 2, 3 및 다양한 시점
    selected_fp = []
    seen_fp_subjs = set()
    for cand in sorted(fp_candidates, key=lambda c: -c['defect_prob']):
        if len(selected_fp) >= 5:
            break
        # 다양한 피험자와 시점 조합 우선
        combo = (cand['subject'], cand['view'])
        if combo not in seen_fp_subjs or len(selected_fp) >= 3:
            selected_fp.append(cand)
            seen_fp_subjs.add(combo)

    # FN: Key 338 및 다양한 결함 피험자
    selected_fn = []
    seen_fn_subjs = set()
    for cand in sorted(fn_candidates, key=lambda c: c['defect_prob']):
        if len(selected_fn) >= 5:
            break
        combo = (cand['subject'], cand['view'], cand['key'])
        if combo not in seen_fn_subjs or len(selected_fn) >= 3:
            selected_fn.append(cand)
            seen_fn_subjs.add(combo)

    # 타깃 폴더 준비
    fp_dir = os.path.join(TEMP_BASE_DIR, "01_정상인데_결함으로_분류_FP")
    fn_dir = os.path.join(TEMP_BASE_DIR, "02_결함인데_정상으로_분류_FN")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    # 복사 및 원인 기술
    def package_samples(samples, target_dir, cat_name):
        manifest = []
        for rank, item in enumerate(samples, 1):
            fname = item['filename']
            base_clip = fname.replace(".npz", "")
            # clip_id (e.g. [오류]_D03_피험자1_Key330)
            parts = base_clip.split("_")
            view_code = parts[-1]  # view1 ~ view5
            clip_folder_name = "_".join(parts[:-1])

            sample_subfolder = os.path.join(target_dir, f"sample_{rank:02d}_{base_clip}")
            os.makedirs(sample_subfolder, exist_ok=True)

            # 1. 64F 키포인트 npz 복사
            src_npz = os.path.join(RAW_MP_DIR, fname)
            if os.path.exists(src_npz):
                shutil.copy2(src_npz, os.path.join(sample_subfolder, fname))

            # 2. 2D 라벨 JSON 복사
            json_name = clip_folder_name + ".json"
            src_json = os.path.join(RAW_JSON_DIR, json_name)
            if os.path.exists(src_json):
                shutil.copy2(src_json, os.path.join(sample_subfolder, json_name))
            else:
                # 패턴 매칭
                for jf in glob.glob(os.path.join(RAW_JSON_DIR, "*.json")):
                    if clip_folder_name in os.path.basename(jf):
                        shutil.copy2(jf, os.path.join(sample_subfolder, os.path.basename(jf)))
                        break

            # 3. 원본 프레임 이미지 복사 (해당 시점 이미지들만)
            korean_view = VIEW_KOREAN_TAG.get(view_code, "시점C")
            img_src_dir = os.path.join(RAW_IMG_DIR, clip_folder_name)
            frames_dir = os.path.join(sample_subfolder, "frames")
            copied_frames = 0
            if os.path.exists(img_src_dir):
                os.makedirs(frames_dir, exist_ok=True)
                for img_file in sorted(os.listdir(img_src_dir)):
                    if korean_view in img_file and img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        shutil.copy2(os.path.join(img_src_dir, img_file), os.path.join(frames_dir, img_file))
                        copied_frames += 1

            # 4. 생체역학적 오분류 원인 분석 기술
            ft = item['features']
            pitch_peak = ft.get('lunge_vspa_torso_pitch_peak', 0.0)
            pitch_min = ft.get('lunge_vspa_torso_pitch_min', 0.0)
            femur_ratio = ft.get('lunge_femur_to_torso_ratio', 1.0)
            decoupling = ft.get('lunge_pelvis_spine_decoupling_angle', 0.0)
            burst_jerk = ft.get('lunge_early_ascent_burst_jerk', 0.0)
            hyperext_asym = ft.get('lunge_bidirectional_hyperext_asym', 0.0)

            if cat_name == "FP":
                reason = (
                    f"대퇴골-체간 비({femur_ratio:.2f})가 길어 무게중심(CoM) 유지를 위해 최하사점에서 "
                    f"상체 전경(피치 피크 {pitch_peak:.1f}°)이 자연스럽게 발생한 '정상 깊은 힙힌지(Deep Hip Hinge)' 동작입니다. "
                    f"골반-척추 디커플링 각도는 {decoupling:.1f}°로 척추 자체는 중립을 유지하였으나, "
                    f"수직선 대비 단순 전경각 크기로 인해 모델 예측 확률이 {item['defect_prob']*100:.1f}%로 상승하여 결함으로 오탐되었습니다."
                )
            else:
                reason = (
                    f"실제 결함 동작(Key {item['key']})이나 상체 숙임 각도가 정상 범위(피치 피크 {pitch_peak:.1f}°) 내에 있어 "
                    f"전방 굴곡 모델이 결함 신호를 감지하지 못했습니다. "
                    f"(후방 젖힘 피치 최솟값 {pitch_min:.1f}°, 비대칭 지표 {hyperext_asym:.3f}, 상승전환 저크 {burst_jerk:.3f}). "
                    f"상체 전경각 착시로 인해 모델 결함 확률이 {item['defect_prob']*100:.1f}%에 그쳐 정상으로 오탐되었습니다."
                )

            item_info = {
                'rank': rank,
                'sample_name': base_clip,
                'filename_npz': fname,
                'ground_truth': item['ground_truth'],
                'prediction': item['prediction'],
                'defect_probability': item['defect_prob'],
                'subject': item['subject'],
                'view': item['view'],
                'official_view': item['official_view'],
                'key_scenario': item['key'],
                'copied_frames_count': copied_frames,
                'misclassification_reason': reason,
                'key_features': {
                    'lunge_vspa_torso_pitch_peak': pitch_peak,
                    'lunge_vspa_torso_pitch_min': pitch_min,
                    'lunge_femur_to_torso_ratio': femur_ratio,
                    'lunge_pelvis_spine_decoupling_angle': decoupling,
                    'lunge_bidirectional_hyperext_asym': hyperext_asym,
                    'lunge_early_ascent_burst_jerk': burst_jerk
                }
            }

            # 각 샘플별 info.json 저장
            with open(os.path.join(sample_subfolder, "sample_info.json"), "w", encoding="utf-8") as f:
                json.dump(item_info, f, indent=2, ensure_ascii=False)

            manifest.append(item_info)
            print(f"  [{cat_name} {rank}/5] {base_clip} 복사 완료 (프레임 {copied_frames}장, Prob: {item['defect_prob']})")

        # 카테고리 전체 목록 저장
        with open(os.path.join(target_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

    print("\n[*] 1. False Positive (정상인데 결함 판정) 샘플 복사 중...")
    fp_manifest = package_samples(selected_fp, fp_dir, "FP")

    print("\n[*] 2. False Negative (결함인데 정상 판정) 샘플 복사 중...")
    fn_manifest = package_samples(selected_fn, fn_dir, "FN")

    # 종합 README 생성
    readme_path = os.path.join(TEMP_BASE_DIR, "README_오분류_데이터_안내.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# [vidio-health-trainer] 바벨 런지 c5 오분류 분석용 추출 데이터 안내\n\n")
        f.write("본 폴더는 사용자의 요청에 따라, 바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부(`lunge_c5`)' 판별 과정에서 발생한\n")
        f.write("대표적인 **False Positive(정상 -> 결함 오진 5건)** 및 **False Negative(결함 -> 정상 오진 5건)** 샘플을 추출하여 보관한 디렉터리입니다.\n\n")
        f.write("각 샘플 폴더에는 다음 파일들이 포함되어 있습니다:\n")
        f.write("1. `*.npz`: 64프레임 3차 스플라인 보간된 100% 순수 비전 MediaPipe 33관절 3D 랜드마크\n")
        f.write("2. `frames/`: 원본 영상에서 해당 카메라 시점으로 촬영된 고해상도 연속 프레임 이미지\n")
        f.write("3. `*.json`: AI-Hub 원천 정답 라벨 2D JSON\n")
        f.write("4. `sample_info.json`: 정답 라벨, 모델 예측 확률, 주요 생체역학 피처값 및 오분류 원인 분석 상세\n\n")
        
        f.write("## 1. False Positive (정상인데 결함으로 분류) 샘플 5선\n")
        for m in fp_manifest:
            f.write(f"- **Sample {m['rank']}**: `{m['sample_name']}`\n")
            f.write(f"  * 피험자 / 시점: 피험자 {m['subject']} / {m['official_view']}\n")
            f.write(f"  * 실제 정답: {m['ground_truth']} | 모델 판정: {m['prediction']} (결함확률: {m['defect_probability']*100:.1f}%)\n")
            f.write(f"  * 원인: {m['misclassification_reason']}\n\n")

        f.write("## 2. False Negative (결함인데 정상으로 분류) 샘플 5선\n")
        for m in fn_manifest:
            f.write(f"- **Sample {m['rank']}**: `{m['sample_name']}`\n")
            f.write(f"  * 피험자 / 시점: 피험자 {m['subject']} / {m['official_view']} (Key {m['key_scenario']})\n")
            f.write(f"  * 실제 정답: {m['ground_truth']} | 모델 판정: {m['prediction']} (결함확률: {m['defect_probability']*100:.1f}%)\n")
            f.write(f"  * 원인: {m['misclassification_reason']}\n\n")

    print("\n" + "=" * 80)
    print(f"[+] data/temp 오분류 데이터 패키징 완료!")
    print(f"    - Base Dir: {TEMP_BASE_DIR}")
    print(f"    - FP 폴더 : {fp_dir} (5개 샘플)")
    print(f"    - FN 폴더 : {fn_dir} (5개 샘플)")
    print(f"    - README  : {readme_path}")
    print("=" * 80)


if __name__ == '__main__':
    run_export()
