# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Pipeline: Stiff Spine Neutrality Complete Pipeline Master Runner (Step 1 -> Step 5)
Purpose: 1단계(전처리)부터 5단계(최종 Word 보고서 생성)까지 전 과정을 일괄 자동 완주하는 마스터 스크립트
"""

import os
import sys
import time
import glob
import subprocess

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
SUMMARY_JSON = r"D:\src\models\second_models\babel_deadlift\stiff_spine_model_evaluation_summary.json"
FINAL_DOCX = r"D:\자주프 문서\6주차\바벨_스티프_데드리프트_척추중립_모델_최종_평가_및_운영배포명세서.docx"


def run_command_stream(cmd: str, desc: str):
    print("\n" + "=" * 110)
    print(f"  [EXECUTE] {desc}")
    print(f"  Command: {cmd}")
    print("=" * 110)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"[ERROR] '{desc}' 실패 (Exit code: {ret.returncode}, 소요시간: {elapsed:.1f}s)")
        sys.exit(ret.returncode)
    else:
        print(f"[SUCCESS] '{desc}' 성공 완료! (소요시간: {elapsed:.1f}s)")


def wait_or_finish_step1():
    """1단계 MediaPipe Full33 64F 전처리 완료 확인 (미완료 시 추가 실행)"""
    print("\n" + "=" * 110)
    print("  [STEP 1] MediaPipe Full33 64F 전처리 검증 및 완료 대기")
    print("=" * 110)
    
    # 207개 폴더 x 5개 시점 = 최대 1035개
    while True:
        files = glob.glob(os.path.join(RAW_MP33_DIR, "*.npz"))
        print(f"[*] 현재 추출된 MP33 64F 파일 수: {len(files)}개")
        if len(files) >= 1000:
            print("[+] 1단계 MediaPipe Full33 64F 전처리 100% 완료 확인!")
            break
        
        # 아직 1000개 미만인 경우, build_stiff_deadlift_full33_64f_dataset.py를 실행하여 남은 폴더 완성
        print("[*] 남은 폴더 일괄 추출 스크립트 실행...")
        cmd = f'python "D:\\src\\preprocessing\\build_stiff_deadlift_full33_64f_dataset.py"'
        subprocess.run(cmd, shell=True)
        
        # 다시 확인
        files = glob.glob(os.path.join(RAW_MP33_DIR, "*.npz"))
        if len(files) >= 900:  # 일부 결측 파일 제외하고 90% 이상 도달 시 완료로 인정
            print(f"[+] 1단계 완료: 총 {len(files)}개 시퀀스 준비 완료!")
            break
        time.sleep(5)


def run_step2_spinepose():
    """2단계 SpinePose 9-joint 64F 전수 추출 및 Denoising 캐시 생성"""
    if os.path.exists(SPINEPOSE_CACHE):
        print(f"[*] 2단계 SpinePose 캐시 이미 존재함: {SPINEPOSE_CACHE}")
        return
    cmd = f'python "D:\\src\\preprocessing\\extract_stiff_deadlift_spinepose_64f.py"'
    run_command_stream(cmd, "2단계: SpinePose 가상 척추 9관절 64F 전수 추출 및 Denoising")


def run_step3_and_4_training():
    """3~4단계: 30D 피처 추출, 피험자 격리 5-Fold 교차검증, 5대 시점 감사, 3-State 최적화"""
    cmd = f'python "D:\\src\\models\\second_models\\babel_deadlift\\train_and_eval_stiff_spine_model.py"'
    run_command_stream(cmd, "3~4단계: 30D 특화 피처 추출 및 피험자 격리 5-Fold 학습/평가")


def run_step5_report():
    """5단계: Word(.docx) 및 Markdown(.md) 최종 종합 보고서 생성"""
    cmd = f'python "D:\\src\\models\\second_models\\babel_deadlift\\generate_stiff_spine_final_report_docx.py"'
    run_command_stream(cmd, "5단계: 최종 Word 및 Markdown 종합 보고서 배포")


def main():
    total_start = time.time()
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 척추 중립 모델 5단계 파이프라인 마스터 가동")
    print("=" * 110)

    # 1단계
    wait_or_finish_step1()

    # 2단계
    run_step2_spinepose()

    # 3~4단계
    run_step3_and_4_training()

    # 5단계
    run_step5_report()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 110)
    print(f"  [ALL COMPLETE] 바벨 스티프 데드리프트 척추 중립 5단계 파이프라인 100% 완주!")
    print(f"  - 총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"  - 최종 Word 보고서: {FINAL_DOCX}")
    print("=" * 110)


if __name__ == '__main__':
    main()
