# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c1_vision_benchmark_pipeline.py
Purpose: SpinePose 추출 완료 후 피처 생성 -> 5-Fold 실측 비교 -> Word 보고서 배포까지
         원스톱으로 자동 연계 실행하는 마스터 파이프라인 러너
"""

import os
import sys
import time
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"
FEATS_SCRIPT = os.path.join(CURRENT_DIR, "build_c1_spinepose_vision_features.py")
COMPARE_SCRIPT = os.path.join(CURRENT_DIR, "compare_c1_aihub_vs_spinepose.py")
REPORT_SCRIPT = os.path.join(CURRENT_DIR, "make_c1_aihub_vs_spinepose_report_docx.py")


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 SpinePose 순수 비전 벤치마크 마스터 자동화 러너")
    print("=" * 115)

    # 1. SpinePose 64F 전수 캐시 생성 확인 대기
    print("[*] [1/3] SpinePose 9대 척추 64F 전수 캐시 파일 대기 중...")
    while not os.path.exists(CACHE_PATH):
        time.sleep(10)
    print(f"[OK] SpinePose 캐시 확인 완료: {CACHE_PATH}")

    # 2. 피처 변환 실행
    print("\n[*] [2/3] Model-B 38D, 27D, 24D 피처셋 일괄 생성 착수...")
    ret2 = subprocess.run([sys.executable, FEATS_SCRIPT])
    if ret2.returncode != 0:
        print("[!] 피처 생성 실패!")
        return

    # 3. 5-Fold 비교 학습 및 평가 실행
    print("\n[*] [3/3] Model-A vs Model-B 5-Fold StratifiedGroupKFold 실측 비교 학습 착수...")
    ret3 = subprocess.run([sys.executable, COMPARE_SCRIPT])
    if ret3.returncode != 0:
        print("[!] 비교 학습 실패!")
        return

    # 4. 최종 공식 Word 보고서 생성
    print("\n[*] [완료] 공식 기술 보고서 Word(.docx) 생성 착수...")
    subprocess.run([sys.executable, REPORT_SCRIPT])
    print("\n[★] 전체 파이프라인이 성공적으로 완수되었습니다!")


if __name__ == '__main__':
    main()
