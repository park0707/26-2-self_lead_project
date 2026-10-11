# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Pipeline: Steps 2 to 5 Execution (SpinePose -> 30D Features -> 5-Fold Model -> Final DOCX)
"""

import os
import sys
import time
import subprocess

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_step(cmd: str, desc: str):
    print("\n" + "=" * 110)
    print(f"  [RUNNING] {desc}")
    print(f"  Command: {cmd}")
    print("=" * 110)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"[FAIL] '{desc}' 실패 (Exit code: {ret.returncode}, 소요: {elapsed:.1f}s)")
        sys.exit(ret.returncode)
    else:
        print(f"[SUCCESS] '{desc}' 완료! (소요: {elapsed:.1f}s)")

def main():
    total_start = time.time()
    print("=" * 110)
    print("  [vidio-health-trainer] 스티프 데드리프트 척추 중립 2~5단계 파이프라인 시작")
    print("=" * 110)

    # 2단계: SpinePose 추출
    run_step('python "D:\\src\\preprocessing\\extract_stiff_deadlift_spinepose_64f.py"', 
             "2단계: SpinePose 가상 척추 9관절 64F 전수 추출 및 Denoising")

    # 3~4단계: 30D 피처 추출 및 5-Fold 학습/평가
    run_step('python "D:\\src\\models\\second_models\\babel_deadlift\\train_and_eval_stiff_spine_model.py"', 
             "3~4단계: 30D 피처 추출, 피험자 격리 5-Fold 학습, 5대 시점 감사, 3-State 최적화")

    # 5단계: 최종 Word 및 Markdown 보고서 생성
    run_step('python "D:\\src\\models\\second_models\\babel_deadlift\\generate_stiff_spine_final_report_docx.py"', 
             "5단계: 공식 최종 Word(.docx) 및 Markdown(.md) 종합 보고서 배포")

    total_time = time.time() - total_start
    print("\n" + "=" * 110)
    print(f"  ★ [ALL STEPS COMPLETE] 2단계 ~ 5단계 전 파이프라인 100% 완주 성공!")
    print(f"  - 총 소요 시간: {total_time/60:.2f}분")
    print("=" * 110)

if __name__ == '__main__':
    main()
