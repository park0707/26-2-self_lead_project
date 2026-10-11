# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/preprocessing/run_auto_resume_batch.py
Purpose: 9종 전 운동 RTMPose 일괄 추출 배치가 30,020건을 100% 완주할 때까지
         메모리 누수나 OS 리소스 회수로 인한 예외 종료 시 자동으로 3초 내 재시작해주는 무중단 워치독
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = r"D:/"
BATCH_SCRIPT = r"D:\src\preprocessing\batch_extract_all_rtmpose_64f.py"
OUT_BASE_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F"
TOTAL_TARGET = 30020

def get_current_total_count() -> int:
    total = 0
    if not os.path.exists(OUT_BASE_DIR):
        return 0
    for d in os.listdir(OUT_BASE_DIR):
        p = os.path.join(OUT_BASE_DIR, d)
        if os.path.isdir(p):
            total += sum(1 for e in os.scandir(p) if e.name.endswith('.npz'))
    return total

def main():
    num_workers = "9"
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        num_workers = sys.argv[1]

    print("=" * 88, flush=True)
    print(f"  [vidio-health-trainer] RTMPose 30,020건 전수 완주 무중단 워치독 가동 (워커: {num_workers}개)", flush=True)
    print("=" * 88, flush=True)

    attempt = 1
    while True:
        curr_cnt = get_current_total_count()
        print(f"\n[*] [Watchdog Attempt #{attempt}] 현재 완료 누적: {curr_cnt:,} / {TOTAL_TARGET:,}건 ({curr_cnt/TOTAL_TARGET*100:.1f}%)", flush=True)

        if curr_cnt >= TOTAL_TARGET:
            print(f"\n[🎉 SUCCESS] 9종 전 운동 {TOTAL_TARGET:,}건 전수 완료가 확인되었습니다! 워치독을 종료합니다.", flush=True)
            break

        print(f"[*] 배치 러너 프로세스 기동 (워커: {num_workers}개)...", flush=True)
        ret = subprocess.run([sys.executable, BATCH_SCRIPT, num_workers], cwd=PROJECT_ROOT)
        
        new_cnt = get_current_total_count()
        print(f"[*] 프로세스 종료 코드: {ret.returncode} (완료 건수: {new_cnt:,}건)", flush=True)

        if new_cnt >= TOTAL_TARGET:
            print(f"\n[🎉 SUCCESS] 9종 전 운동 {TOTAL_TARGET:,}건 전수 완주 달성!", flush=True)
            break

        print("[!] 미완료 상태에서 프로세스가 종료됨을 감지 -> 3초 후 자동 이어받기(Resume) 실행...", flush=True)
        time.sleep(3)
        attempt += 1

if __name__ == '__main__':
    main()
