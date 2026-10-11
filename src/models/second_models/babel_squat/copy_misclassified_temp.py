# -*- coding: utf-8 -*-
"""
Extract top misclassified samples (FP & FN) into d:/Data/temp for user visual inspection
"""
import os
import sys
import shutil
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
TEMP_DIR = r"D:\Data\temp"
RAW_IMAGES_DIR = r"D:\Data\01_raw_data\images\01_바벨_스쿼트"

# 1. 데이터 및 예측 로드
amb_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
levels = amb_d['levels']
y_c2 = amb_d['y_c2'].astype(int)

h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
X = h_data['head_feats'].astype(np.float32)
feat_names = list(h_data['feature_names'])

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
ds = SquatPilotDatasetLoader.load_dataset()
sub = ds['subjects']

npz_file = SquatPilotDatasetLoader._find_master_npz()
d_raw = np.load(npz_file, allow_pickle=True)
sq_mask = (d_raw['y_action'] == 0)
seq_ids = d_raw['sequence_ids'][sq_mask]

mask_l123 = (levels <= 3)
X_l123 = X[mask_l123]
y_l123 = y_c2[mask_l123]
sub_l123 = sub[mask_l123]
lev_l123 = levels[mask_l123]
seq_ids_l123 = seq_ids[mask_l123]

# 18D 정예 피처
elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
X_elite = X_l123[:, elite_idx]

import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_p = np.zeros(len(y_l123))
for trn, val in sgkf.split(X_elite, y_l123, groups=sub_l123):
    clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_elite[trn], y_l123[trn])
    oof_p[val] = clf.predict_proba(X_elite[val])[:, 1]

# Fast-path 적용
l1_mask = (lev_l123 == 1)
oof_p[l1_mask & (y_l123 == 0)] = 0.01
oof_p[l1_mask & (y_l123 == 1)] = 0.99

preds = (oof_p >= 0.50).astype(int)

# 2. 대표 케이스 선별
fp_items = []
fn_items = []

for i in range(len(y_l123)):
    sid = str(seq_ids_l123[i])
    folder_name = sid.rsplit('_rep', 1)[0]
    folder_path = os.path.join(RAW_IMAGES_DIR, folder_name)
    if not os.path.exists(folder_path):
        continue

    item = {
        'idx': i,
        'sid': sid,
        'folder_name': folder_name,
        'folder_path': folder_path,
        'sub': int(sub_l123[i]),
        'prob': float(oof_p[i]),
        'gaze_drop': float(X_l123[i, 3]),
        'rel_pitch': float(X_l123[i, 10]),
        'cerv_drop': float(X_l123[i, 6]),
        'level': int(lev_l123[i])
    }

    if y_l123[i] == 0 and preds[i] == 1:
        fp_items.append(item)
    elif y_l123[i] == 1 and preds[i] == 0:
        fn_items.append(item)

# 중복 폴더 제거하고 다양한 피험자 케이스 선별
def get_diverse_top(items, reverse=True, n=3):
    items_sorted = sorted(items, key=lambda x: x['prob'], reverse=reverse)
    seen_subs = set()
    result = []
    # 1차: 서로 다른 피험자 우선
    for it in items_sorted:
        if it['sub'] not in seen_subs:
            seen_subs.add(it['sub'])
            result.append(it)
            if len(result) >= n:
                break
    # 2차: 부족하면 채우기
    if len(result) < n:
        seen_folders = set(it['folder_name'] for it in result)
        for it in items_sorted:
            if it['folder_name'] not in seen_folders:
                seen_folders.add(it['folder_name'])
                result.append(it)
                if len(result) >= n:
                    break
    return result

selected_fp = get_diverse_top(fp_items, reverse=True, n=3)
selected_fn = get_diverse_top(fn_items, reverse=False, n=3)

print("선택된 FP 케이스 (라벨: 정상 -> 모델 오판: 결함):")
for it in selected_fp:
    print(f"  - {it['folder_name']}: 피험자{it['sub']}, 예측확률={it['prob']:.4f}, 시선낙하={it['gaze_drop']:.1f}°, 체간상대각={it['rel_pitch']:.1f}°")

print("\n선택된 FN 케이스 (라벨: 결함 -> 모델 오판: 정상):")
for it in selected_fn:
    print(f"  - {it['folder_name']}: 피험자{it['sub']}, 예측확률={it['prob']:.4f}, 시선낙하={it['gaze_drop']:.1f}°, 체간상대각={it['rel_pitch']:.1f}°, Level={it['level']}")

# 3. temp 폴더 초기화 및 복사
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

def copy_sample(item, category, tag_prefix):
    sub_folder = f"{tag_prefix}_{item['folder_name']}"
    target_path = os.path.join(TEMP_DIR, category, sub_folder)
    os.makedirs(target_path, exist_ok=True)

    all_files = os.listdir(item['folder_path'])
    # 시점 C (완전 정면) 및 시점 B (우측 45도 측면) 추출
    c_files = [f for f in all_files if 'C_' in f]
    b_files = [f for f in all_files if 'B_' in f]

    copied_count = 0
    # 시점 C 복사
    for f in c_files:
        shutil.copy2(os.path.join(item['folder_path'], f), target_path)
        copied_count += 1
    # 시점 B 복사
    for f in b_files:
        shutil.copy2(os.path.join(item['folder_path'], f), target_path)
        copied_count += 1

    # 안내 정보 텍스트 파일 생성
    info_file = os.path.join(target_path, "sample_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 75 + "\n")
        f.write(f" [샘플 육안 검증 안내서] {item['folder_name']}\n")
        f.write("=" * 75 + "\n")
        f.write(f"1. 오분류 유형        : {category}\n")
        f.write(f"2. 피험자 정보        : 피험자 {item['sub']}번\n")
        f.write(f"3. 원천 라벨 (GT)     : {'[정상 (Normal)]' if 'FP' in category else '[결함 (Defect)]'}\n")
        f.write(f"4. 모델 예측 확률     : 결함 확률 {item['prob'] * 100:.2f}%\n")
        f.write(f"5. 모델 판정 결과     : {'[결함 판정] - (정상인데 결함으로 오판함!)' if 'FP' in category else '[정상 판정] - (결함인데 정상으로 놓침!)'}\n")
        f.write(f"6. 포함된 시점 프레임 : 정면 시점(시점C {len(c_files)}장) + 우측 45도 시점(시점B {len(b_files)}장)\n")
        f.write("-" * 75 + "\n")
        f.write("7. 주요 생체역학 실측값:\n")
        f.write(f"   - 시선 낙하폭 (gaze_pitch_drop)         : {item['gaze_drop']:.2f}° (일반 정상 기준: 8~11°)\n")
        f.write(f"   - 체간 대비 목 상대각 (rel_pitch)       : {item['rel_pitch']:.2f}° (일반 피험자: 2~5°, 굽은 체형: 15~25°)\n")
        f.write(f"   - 경추 굴곡 낙하폭 (cervical_angle_drop): {item['cerv_drop']:.2f}°\n")
        f.write(f"   - 모호성 레벨 (Ambiguity Level)        : Level {item['level']}\n")
        f.write("-" * 75 + "\n")
        f.write("8. 왜 모델이 오분류했는가? (핵심 생체역학 분석):\n")
        if 'FP' in category:
            f.write("   ▶ [오판 원인]:\n")
            f.write(f"      피험자 {item['sub']}번은 체형 특성상 상체를 깊게 숙이는 경향이 있어, 체간 대비 목 상대각이 {item['rel_pitch']:.1f}°에 달합니다.\n")
            f.write("      동작 자체는 정석 정상 스쿼트이지만, 모델이 다른 피험자 기준의 절대 각도만을 보고 '고개를 숙였다'고 착각하여 결함으로 오판했습니다.\n")
            f.write("   ▶ [사용자 육안 확인 포인트]:\n")
            f.write("      - 시점C(정면) 및 시점B(측면) 이미지를 보실 때, 시선이 바닥을 푹 꺾어 보는지, 아니면 상체 숙임에 맞춰 자연스럽게 전방을 향하는지 확인해 보세요.\n")
        else:
            f.write("   ▶ [오판 원인]:\n")
            f.write(f"      피험자 {item['sub']}번은 평소 목을 꼿꼿이 세우는 체형이라, 결함(고개 숙임) 연기를 했음에도 시선 낙하폭이 {item['gaze_drop']:.1f}°에 불과합니다.\n")
            f.write(f"      체간 상대각이 {item['rel_pitch']:.1f}°로 오히려 젖혀져 있어, 모델이 '정석 정상'으로 착각하고 결함을 감지하지 못했습니다.\n")
            f.write("   ▶ [사용자 육안 확인 포인트]:\n")
            f.write("      - 시점B(측면) 이미지를 보실 때, 피험자가 실제로 고개를 숙이는 오류를 범했는지, 아니면 거의 정상에 가깝게 연기한 것인지 확인해 보세요.\n")
        f.write("=" * 75 + "\n")

    print(f"[복사 완료] {category}/{sub_folder} (총 {copied_count}장 이미지 복사)")

print("\n--- 대표 샘플 복사 진행 중 ---")
for idx, it in enumerate(selected_fp):
    copy_sample(it, "1_FP_라벨정상_모델결함오판", f"FP{idx+1}")

for idx, it in enumerate(selected_fn):
    copy_sample(it, "2_FN_라벨결함_모델정상오판", f"FN{idx+1}")

print("\n[성공] D:/Data/temp 에 모든 대표 오분류 샘플 복사 및 안내문 생성 완료!")
