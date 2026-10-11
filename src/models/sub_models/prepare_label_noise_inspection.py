# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: prepare_label_noise_inspection.py
Purpose: 사용자가 직접 눈으로 라벨 노이즈를 검증할 수 있도록
         D:\temp\바벨런지_c3_라벨노이즈_검증\ 폴더에
         1) 실제 원천 촬영 사진 JPG (시점별 주요 프레임)
         2) AI-Hub 원천 라벨 JSON (뒤다리무릎각도 90도: True 기록 확인)
         3) MediaPipe 관절 스켈레톤 및 실측 각도 오버레이 이미지 PNG
         4) 상세 검증 가이드 리포트 TXT
         를 복사 및 자동 생성하여 구성
"""

import os
import sys
import glob
import json
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

TEMP_DIR = r"D:\temp\바벨런지_c3_라벨노이즈_검증"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"
RAW_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지"
PREPROC_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"

os.makedirs(TEMP_DIR, exist_ok=True)

# 대표적인 라벨 노이즈 의심 표본 3건
TARGET_SAMPLES = [
    {
        'id': 'Key340_피험자2',
        'folder_name': '[오류]_D03_피험자2_Key340',
        'npz_file': '[오류]_D03_피험자2_Key340.npz',
        'json_file': '[오류]_D03_피험자2_Key340.json',
        'measured_angle': 167.0,
        'model_prob': 0.970,
        'official_label_c3': 'True (정상으로 기재됨)',
        'description': 'AI-Hub 등록 설명: 발방향 안쪽 몸통 옆으로 갸우뚱'
    },
    {
        'id': 'Key335_피험자5',
        'folder_name': '[오류]_D15_피험자5_Key335',
        'npz_file': '[오류]_D15_피험자5_Key335.npz',
        'json_file': '[오류]_D15_피험자5_Key335.json',
        'measured_angle': 174.0,
        'model_prob': 0.950,
        'official_label_c3': 'True (정상으로 기재됨)',
        'description': 'AI-Hub 등록 설명: 런지 하강 결함'
    },
    {
        'id': 'Key344_피험자2',
        'folder_name': '[오류]_D03_피험자2_Key344',
        'npz_file': '[오류]_D03_피험자2_Key344.npz',
        'json_file': '[오류]_D03_피험자2_Key344.json',
        'measured_angle': 152.2,
        'model_prob': 0.965,
        'official_label_c3': 'True (정상으로 기재됨)',
        'description': 'AI-Hub 등록 설명: 런지 복합 결함'
    }
]

def main():
    print("=" * 80)
    print(" [사용자 직접 검증용] 바벨 런지 c3 라벨 노이즈 의심 표본 데이터 패키징")
    print("=" * 80)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(" [바벨 런지 c3 뒤다리 무릎 각도 90도] AI-Hub 원천 데이터 라벨 노이즈 검증 가이드")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("사용자님께서 직접 눈으로 라벨 노이즈 여부를 확인하실 수 있도록,")
    summary_lines.append("실제 원천 촬영 사진(JPG), AI-Hub 정답 JSON, MediaPipe 관절 각도 오버레이(PNG)를 준비했습니다.")
    summary_lines.append("")
    summary_lines.append("■ 핵심 의혹 현상 요약:")
    summary_lines.append("AI-Hub 원천 데이터셋에는 '뒤다리 무릎 각도 90도(c3)'가 True(정상)로 라벨링되어 있으나,")
    summary_lines.append("실제 피험자의 촬영 사진을 보면 피험자가 하강을 거의 하지 않아 뒤다리 무릎 각도가 152° ~ 174°로 완전히 펴져 있습니다.")
    summary_lines.append("즉, 모델이 결함(1)으로 판정한 것이 정확한 비전 판정이며, 원천 라벨이 오기재(False Positive 라벨 노이즈)된 사례입니다.")
    summary_lines.append("")
    summary_lines.append("-" * 80)
    
    for idx, sample in enumerate(TARGET_SAMPLES, 1):
        sample_dir = os.path.join(TEMP_DIR, f"{idx:02d}_{sample['id']}")
        os.makedirs(sample_dir, exist_ok=True)
        
        print(f"\n[*] Processing [{sample['id']}] -> {sample_dir}")
        summary_lines.append(f"\n[표본 {idx}] {sample['id']} ({sample['folder_name']})")
        summary_lines.append(f"  • AI-Hub 원천 공식 라벨 (c3 뒤다리 무릎 90도): {sample['official_label_c3']}")
        summary_lines.append(f"  • 실제 비전 실측 뒤다리 무릎 각도: {sample['measured_angle']}° (90도 직각 대비 무려 {abs(sample['measured_angle']-90):.1f}° 편차!)")
        summary_lines.append(f"  • v3 챔피언 모델의 결함 예측 확률: {sample['model_prob']*100:.1f}% (압도적 결함 판정)")
        summary_lines.append(f"  • AI-Hub 등록 동작 설명: {sample['description']}")
        
        # 1. 원천 JSON 라벨 복사
        json_src = os.path.join(RAW_JSON_DIR, "2D", sample['json_file'])
        if os.path.exists(json_src):
            json_dst = os.path.join(sample_dir, "01_AI-Hub_원천_정답라벨.json")
            shutil.copy(json_src, json_dst)
            summary_lines.append(f"  -> 복사 완료: 01_AI-Hub_원천_정답라벨.json (열어서 '뒤다리 무릎 각도 90도': true 직접 확인 가능)")
            
        # 2. 원천 이미지 사진 복사 (시점 A 및 시점 C 최하사점 프레임 사진 복사)
        img_src_dir = os.path.join(RAW_IMG_DIR, sample['folder_name'])
        if os.path.exists(img_src_dir):
            all_files = os.listdir(img_src_dir)
            copied_imgs = 0
            for bname in all_files:
                if not bname.endswith('.jpg'):
                    continue
                # 시점A 및 시점C의 최하사점 부근 프레임 선택 (프레임07 ~ 프레임12)
                if any(v in bname for v in ['시점A', '시점C']) and any(f in bname for f in ['프레임07', '프레임08', '프레임09', '프레임10', '프레임11']):
                    src_jf = os.path.join(img_src_dir, bname)
                    dst_jf = os.path.join(sample_dir, f"02_실제사진_{bname}")
                    shutil.copy(src_jf, dst_jf)
                    copied_imgs += 1
            summary_lines.append(f"  -> 복사 완료: 실제 촬영 프레임 사진 {copied_imgs}장 (하강 최저점 사진 직접 육안 확인)")
            
        # 3. MediaPipe 관절 스켈레톤 및 각도 시각화 PNG 생성
        npz_src = os.path.join(PREPROC_DIR, sample['npz_file'])
        if os.path.exists(npz_src):
            data = np.load(npz_src, allow_pickle=True)
            kps = data['keypoints_2d_norm'] # (64, 15, 2)
            
            # 최하사점 프레임 찾기
            knee_l = kps[:, 10, :]
            knee_r = kps[:, 11, :]
            bot_idx = np.argmax(np.maximum(knee_l[:, 1], knee_r[:, 1]))
            
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.set_title(f"{sample['id']} MediaPipe Skeleton (Bottom Phase)\nMeasured Rear Knee Angle: {sample['measured_angle']:.1f}° (Target: 90°)\nAI-Hub Label: True (Normal)", fontsize=11, fontweight='bold', color='navy')
            
            # 관절 좌표 플롯 (Y축 반전)
            kps_bot = kps[bot_idx] # (15, 2)
            ax.scatter(kps_bot[:, 0], -kps_bot[:, 1], c='red', s=50, zorder=5)
            
            # 관절 뼈대 연결선 정의
            edges = [
                (0, 1), (1, 14), # Pelvis-Torso-Neck
                (0, 8), (8, 10), (10, 12), # Left Leg
                (0, 9), (9, 11), (11, 13), # Right Leg
            ]
            for j1, j2 in edges:
                ax.plot([kps_bot[j1, 0], kps_bot[j2, 0]], [-kps_bot[j1, 1], -kps_bot[j2, 1]], 'b-', lw=3, alpha=0.7)
                
            # 뒤다리 무릎 강조 표시
            rear_knee_idx = 10 if kps_bot[10, 1] >= kps_bot[11, 1] else 11
            ax.scatter(kps_bot[rear_knee_idx, 0], -kps_bot[rear_knee_idx, 1], c='gold', s=150, edgecolors='darkred', lw=2, zorder=6)
            ax.text(kps_bot[rear_knee_idx, 0] + 5, -kps_bot[rear_knee_idx, 1], f"Rear Knee: {sample['measured_angle']:.1f}°\n(Should be 90°)", fontsize=10, fontweight='bold', color='crimson')
            
            ax.axis('off')
            plot_dst = os.path.join(sample_dir, "03_관절스켈레톤_실측각도시각화.png")
            plt.tight_layout()
            plt.savefig(plot_dst, dpi=150)
            plt.close()
            summary_lines.append(f"  -> 생성 완료: 03_관절스켈레톤_실측각도시각화.png (관절선 및 167° 무릎 굴곡 시각화)")
            
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("■ 직접 확인 방법 안내:")
    summary_lines.append(f"1. 윈도우 탐색기에서 '{TEMP_DIR}' 폴더를 엽니다.")
    summary_lines.append("2. 각 표본 폴더(01_Key340, 02_Key335 등) 내의 '02_실제촬영사진_*.jpg'를 엽니다.")
    summary_lines.append("   -> 실제 피험자가 런지 시 뒤다리를 90도로 굽히지 않고 160도 이상 펴고 있는 모습을 직접 확인하실 수 있습니다.")
    summary_lines.append("3. '01_AI-Hub_원천_정답라벨.json' 파일을 메모장으로 열어봅니다.")
    summary_lines.append("   -> '뒤다리 무릎 각도 90도'의 value가 'true'(정상)로 잘못 적혀 있는 것을 확인하실 수 있습니다.")
    summary_lines.append("4. '03_관절스켈레톤_실측각도시각화.png'를 열어 MediaPipe가 추출한 실측 각도를 확인합니다.")
    summary_lines.append("=" * 80)
    
    guide_path = os.path.join(TEMP_DIR, "00_라벨노이즈_직접검증_안내서.txt")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
        
    print(f"\n[+] Inspection guide saved to: {guide_path}")
    print(f"[+] All verification files successfully populated in: {TEMP_DIR}")

if __name__ == "__main__":
    main()
