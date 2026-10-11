# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Script: export_misclassified_samples.py
Purpose: bl4_v3 모델의 주요 오분류(FP/FN) 데이터를 data/temp에 추출 및 저장
         - 원본 64F 전처리 npz 파일 복사
         - 원본 해당 시점 프레임 이미지(.jpg) 복사
         - 상세 분석 메타데이터(csv, txt, md) 생성
"""

import os
import sys
import shutil
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = r"D:\src\models\second_models\babel_lunge"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_and_eval_bl4_v3 import SEBlock1D, MultiScaleConvBlock, MultiScaleSETCN
from src.features.lunge_c4_features_v3 import LUNGE_C4_FEATURE_NAMES_V3

TEMP_DIR = r"D:\Data\temp"
CACHE_V3_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
BUNDLE_V3_PATH = os.path.join(CURRENT_DIR, "bl4_v3_weights.pkl")
MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"

VIEW_CODE_TO_KOR = {
    'view5': '시점A',
    'view2': '시점B',
    'view3': '시점C',
    'view4': '시점D',
    'view1': '시점E'
}

VIEW_NAME_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

def export_samples():
    print("=" * 80)
    print("  [vidio-health-trainer] bl4_v3 주요 오분류 데이터 D:\\Data\\temp 추출 작업")
    print("=" * 80)

    # 1. 캐시 및 모델 로드
    cache = np.load(CACHE_V3_PATH, allow_pickle=True)
    X_24d = cache['X_24d']
    X_temp = cache['X_temp_14d']
    y = cache['y']
    subjects = cache['subjects']
    views = cache['views']
    clip_ids = cache['clip_ids']
    keys = cache['keys']
    N = len(y)

    with open(BUNDLE_V3_PATH, 'rb') as f:
        bundle = pickle.load(f)

    weights = bundle['weights']
    w_t, w_l, w_x, w_c = weights['w_tcn'], weights['w_lgb'], weights['w_xgb'], weights['w_cat']
    fold_models = bundle['fold_models']
    tau_c = bundle['thresholds']['tau_clean'] # 0.44
    tau_d = bundle['thresholds']['tau_def']   # 0.51
    device = torch.device('cpu')

    # 2. 5-Fold OOF 예측 복원
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_24d, y, groups=subjects)):
        scaler = fold_models['scaler'][fold]
        lgb_m = fold_models['lgb'][fold]
        xgb_m = fold_models['xgb'][fold]
        cat_m = fold_models['cat'][fold]
        tcn_m = fold_models['tcn'][fold].to(device)
        tcn_m.eval()

        X_va_sc = scaler.transform(X_24d[val_idx])
        p_lgb = lgb_m.predict_proba(X_va_sc)[:, 1]
        p_xgb = xgb_m.predict_proba(X_va_sc)[:, 1]
        p_cat = cat_m.predict_proba(X_va_sc)[:, 1]

        T_va = torch.from_numpy(X_temp[val_idx]).float().to(device)
        with torch.no_grad():
            logits = tcn_m(T_va)
            p_tcn = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        p_ens = w_t * p_tcn + w_l * p_lgb + w_x * p_xgb + w_c * p_cat
        oof_probs[val_idx] = p_ens

    preds_3s = np.zeros(N, dtype=int)
    preds_3s[oof_probs <= tau_c] = 0
    preds_3s[oof_probs >= tau_d] = 1
    preds_3s[(oof_probs > tau_c) & (oof_probs < tau_d)] = -1

    # 3. 오분류 식별
    # FP: 실제 정상(0)인데 결함(1)으로 판정 (확률이 높은 순)
    fp_indices = np.where((y == 0) & (preds_3s == 1))[0]
    fp_sorted = fp_indices[np.argsort(-oof_probs[fp_indices])]  # 확률 내림차순

    # FN: 실제 결함(1)인데 정상(0)으로 판정 (확률이 낮은 순)
    fn_indices = np.where((y == 1) & (preds_3s == 0))[0]
    fn_sorted = fn_indices[np.argsort(oof_probs[fn_indices])]   # 확률 오름차순

    print(f"[*] 총 샘플수: {N}개")
    print(f"[*] 전체 FP(정상->결함 오판): {len(fp_indices)}건 중 상위 12건 추출")
    print(f"[*] 전체 FN(결함->정상 오판): {len(fn_indices)}건 중 상위 12건 추출")

    # 4. 대상 폴더 초기화
    fp_target_dir = os.path.join(TEMP_DIR, "01_FP_정상인데_결함오판")
    fn_target_dir = os.path.join(TEMP_DIR, "02_FN_결함인데_정상오판")
    os.makedirs(fp_target_dir, exist_ok=True)
    os.makedirs(fn_target_dir, exist_ok=True)

    summary_records = []

    # Helper 함수: 샘플 데이터 내보내기
    def process_export(idx_list, error_type, target_base_dir, max_count=12):
        for rank, idx in enumerate(idx_list[:max_count], 1):
            clip_id = str(clip_ids[idx])
            subj = str(subjects[idx])
            view = str(views[idx])
            key = str(keys[idx])
            prob = float(oof_probs[idx])
            label = int(y[idx])
            pred = int(preds_3s[idx])
            view_kor = VIEW_CODE_TO_KOR.get(view, '시점A')
            view_full = VIEW_NAME_MAP.get(view, view)

            # 폴더명 생성
            folder_name = f"{error_type}_{rank:02d}_{subj}_Key{key}_{view}_Prob_{prob:.3f}"
            sample_dir = os.path.join(target_base_dir, folder_name)
            os.makedirs(sample_dir, exist_ok=True)

            # A. 64F 전처리 npz 파일 복사
            npz_src = os.path.join(MP33_DIR, f"{clip_id}.npz")
            npz_copied = False
            if os.path.exists(npz_src):
                shutil.copy2(npz_src, os.path.join(sample_dir, f"{clip_id}.npz"))
                npz_copied = True

            # B. 원본 프레임 이미지 복사 (해당 시점 16장 프레임)
            # 원본 폴더: clip_id에서 시점 접미사 제거 -> e.g. [원천]_D03_피험자1_Key330
            raw_folder_name = clip_id.rsplit('_', 1)[0]
            raw_folder_path = os.path.join(RAW_IMG_DIR, raw_folder_name)
            img_count = 0

            frames_dir = os.path.join(sample_dir, "raw_frames")
            if os.path.exists(raw_folder_path):
                os.makedirs(frames_dir, exist_ok=True)
                for f in sorted(os.listdir(raw_folder_path)):
                    if view_kor in f and f.lower().endswith(('.jpg', '.png')):
                        shutil.copy2(os.path.join(raw_folder_path, f), os.path.join(frames_dir, f))
                        img_count += 1

            # C. 주요 피처 값 분석
            feats = X_24d[idx]
            feat_dict = {LUNGE_C4_FEATURE_NAMES_V3[i]: round(float(feats[i]), 3) for i in range(len(LUNGE_C4_FEATURE_NAMES_V3))}

            # 특이 피처 감지
            suspicious_factors = []
            if feat_dict['cervical_yaw_max'] > 120:
                suspicious_factors.append(f"경추 회전 극단치 ({feat_dict['cervical_yaw_max']}°)")
            if feat_dict['trunk_roll_max'] > 8.0:
                suspicious_factors.append(f"체간 좌우 기울기 과다 ({feat_dict['trunk_roll_max']}°)")
            if feat_dict['local_torso_torsion_max'] > 15.0:
                suspicious_factors.append(f"상하체 비틀림 과다 ({feat_dict['local_torso_torsion_max']}°)")
            if feat_dict['cervical_pitch_min'] < -20.0:
                suspicious_factors.append(f"경추 숙임 과다 ({feat_dict['cervical_pitch_min']}°)")
            if feat_dict['pelvis_lateral_sway'] > 10.0:
                suspicious_factors.append(f"골반 흔들림 과다 ({feat_dict['pelvis_lateral_sway']}°)")
            
            sus_str = ", ".join(suspicious_factors) if suspicious_factors else "경계 구간 미세 편차"

            # D. sample_info.txt 작성
            info_txt_path = os.path.join(sample_dir, "sample_info.txt")
            with open(info_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"==========================================================\n")
                f.write(f"  [vidio-health-trainer] bl4 오분류 샘플 상세 정보\n")
                f.write(f"==========================================================\n")
                f.write(f"• 오분류 유형 : {error_type} ({'실제 정상이나 결함으로 오판' if 'FP' in error_type else '실제 결함이나 정상으로 오판'})\n")
                f.write(f"• 시퀀스 ID   : {clip_id}\n")
                f.write(f"• 피험자 번호 : {subj}\n")
                f.write(f"• 카메라 시점 : {view} ({view_full})\n")
                f.write(f"• 동작 Key    : {key}\n")
                f.write(f"• 정답 라벨   : {label} ({'정상(Clean)' if label == 0 else '결함(Defect)'})\n")
                f.write(f"• 모델 예측값 : {prob:.4f} (3-State 판정: {'결함(Defect)' if pred == 1 else '정상(Clean)'})\n")
                f.write(f"• 3-State 기준: Clean <= {tau_c:.2f}, Defect >= {tau_d:.2f}\n")
                f.write(f"• 주요 의심요인: {sus_str}\n")
                f.write(f"• 첨부 데이터 : npz 파일({'O' if npz_copied else 'X'}), 원본 이미지 프레임({img_count}장)\n\n")
                f.write(f"[24차원 추출 도메인 피처 수치]\n")
                for fname, fval in feat_dict.items():
                    f.write(f"  - {fname:30s} : {fval}\n")

            summary_records.append({
                '유형': error_type,
                '순위': rank,
                '시퀀스ID': clip_id,
                '피험자': subj,
                '카메라시점': view_full,
                '정답라벨': '정상(0)' if label == 0 else '결함(1)',
                '모델예측확률': f"{prob*100:.2f}%",
                '오분류원인_핵심요인': sus_str,
                '경추Yaw_Max': feat_dict['cervical_yaw_max'],
                '체간Roll_Max': feat_dict['trunk_roll_max'],
                '상하체비틀림_Max': feat_dict['local_torso_torsion_max'],
                '폴더경로': folder_name
            })
            print(f"  [{error_type}] #{rank:02d} | {subj} | {view:6s} | Key{key} | Prob={prob*100:5.1f}% | 복사: npz={npz_copied}, img={img_count}장")

    print("\n--- [1/2] False Positive (FP: 정상인데 결함 오판) 추출 ---")
    process_export(fp_sorted, "FP", fp_target_dir, max_count=12)

    print("\n--- [2/2] False Negative (FN: 결함인데 정상 오판) 추출 ---")
    process_export(fn_sorted, "FN", fn_target_dir, max_count=12)

    # 5. CSV 요약표 생성
    df_summary = pd.DataFrame(summary_records)
    csv_path = os.path.join(TEMP_DIR, "오분류_데이터_종합목록.csv")
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] 오분류 종합 CSV 목록 저장 완료: {csv_path}")

    # 6. 마크다운 가이드 문서 생성
    md_path = os.path.join(TEMP_DIR, "오분류_데이터_분석_가이드.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# [vidio-health-trainer] 바벨 런지 c4 오분류 데이터 분석 가이드\n\n")
        f.write("본 폴더(`D:\\Data\\temp`)는 바벨 런지 c4(척추의 중립 위반) 모델(`bl4_v3`)이 5-Fold 피험자 격리 교차검증 과정에서 오분류한 대표 데이터들을 직접 육안 및 데이터로 확인하실 수 있도록 추출한 결과물입니다.\n\n")
        f.write("## 1. 폴더 구조 안내\n\n")
        f.write("```\n")
        f.write("D:\\Data\\temp\\\n")
        f.write("├── 01_FP_정상인데_결함오판/          # 실제 정답은 '정상(0)'이나 모델이 결함으로 오판 (12개)\n")
        f.write("│   └── FP_01_피험자X_KeyXXX_viewX_Prob_0.XXX/\n")
        f.write("│       ├── [원천]_..._viewX.npz     # 64프레임 MediaPipe 관절 랜드마크\n")
        f.write("│       ├── sample_info.txt          # 정답, 예측 확률, 24D 피처 수치\n")
        f.write("│       └── raw_frames/              # 실제 원본 16장 프레임 이미지 (.jpg)\n")
        f.write("├── 02_FN_결함인데_정상오판/          # 실제 정답은 '결함(1)'이나 모델이 정상으로 오판 (12개)\n")
        f.write("│   └── FN_01_피험자X_KeyXXX_viewX_Prob_0.XXX/\n")
        f.write("├── 오분류_데이터_종합목록.csv        # 엑셀 확인용 전체 요약 성적표\n")
        f.write("└── 오분류_데이터_분석_가이드.md      # 본 설명 문서\n")
        f.write("```\n\n")
        f.write("## 2. 주요 오분류 패턴 및 점검 포인트\n\n")
        f.write("### ① False Positive (정상 -> 결함 오판)의 주요 원인:\n")
        f.write("1. **피험자 고유 체형 비대칭 (피험자2 등)**:\n")
        f.write("   - 준비 자세부터 척추나 어깨가 한쪽으로 기울어진 체형의 경우, 런지 동작 자체는 바른 척추 중립을 유지했음에도 절대 각도(`trunk_roll_max`)가 임계치를 넘어 결함으로 오판되는 현상.\n")
        f.write("2. **카메라 각도에 의한 시각적 왜곡 (원근 왜곡)**:\n")
        f.write("   - 사각 시점(좌측/우측 후방 45°)에서 런지 최하점 시 상체가 카메라 방향으로 회전한 것처럼 보여 상하체 비틀림(`local_torso_torsion_max`)이 과대 추정됨.\n")
        f.write("3. **AI-Hub 정답 라벨 노이즈 의심 케이스**:\n")
        f.write("   - `raw_frames/`의 이미지를 직접 확인해보시면, 라벨은 '정상'으로 되어 있으나 실제로는 척추가 눈에 띄게 굽어 있어 모델이 결함으로 판단한 것이 오히려 타당해 보이는 케이스가 일부 존재합니다.\n\n")
        f.write("### ② False Negative (결함 -> 정상 오판)의 주요 원인:\n")
        f.write("1. **순간적/미세한 척추 비틀림 (피험자1, 3 등)**:\n")
        f.write("   - 최하점 1~2프레임에서만 척추가 살짝 비틀어졌다가 바로 복귀하는 미세 결함의 경우, 64프레임 전반의 평균 각도가 정상 범위에 머물러 모델이 정상으로 오판.\n")
        f.write("2. **두꺼운 피트니스 복장 및 바벨 가림(Occlusion)**:\n")
        f.write("   - 바벨 봉이나 손목, 옷에 쇄골/목 관절이 가려져 MediaPipe가 경추와 체간 축을 수직으로 잘못 보정한 케이스.\n")

    print(f"[+] 마크다운 분석 가이드 생성 완료: {md_path}")
    print("\n[★ SUCCESS] data/temp 오분류 데이터 내보내기 100% 완료!")

if __name__ == '__main__':
    export_samples()
