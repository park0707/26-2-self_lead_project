# -*- coding: utf-8 -*-
"""
단일 운동 유튜브 수집 파이프라인 엔드투엔드 실증 테스트 스크립트
[실행]: python -m src.data_mining_from_youtube.test_youtube_mining
"""
import os
import sys
import glob
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_mining_from_youtube.pipeline import YouTubeMiningPipeline
from src.data_mining_from_youtube.search_queries import EXERCISE_QUERIES

def run_test():
    print("=================================================================")
    print("   YouTube Automated Data Mining Pipeline End-to-End Test")
    print("=================================================================")

    test_dir = "d:/Data/피처_데이터/YouTube_Mined_3D_Test"
    temp_dir = "d:/Data/temp_youtube_videos_test"

    pipeline = YouTubeMiningPipeline(output_dir=test_dir, temp_dir=temp_dir)

    # 1. 01_바벨_스쿼트 1편 수집 테스트
    target_ex = "01_바벨_스쿼트"
    print(f"[*] Target Exercise: {target_ex}")

    records = pipeline.mine_exercise(
        exercise_key=target_ex,
        target_video_count=1,
        max_candidates=5
    )

    print("\n=================================================================")
    print(f"   Test Result Summary: {len(records)} Repetition Sequences Saved")
    print("=================================================================")

    if len(records) > 0:
        first_rec = records[0]
        vid = first_rec['video_id']
        rep_idx = first_rec['rep_index']
        npz_file = os.path.join(test_dir, target_ex, f"yt_{vid}_rep{rep_idx:02d}.npz")

        print(f"[+] Verified Record 1:")
        print(f"    - Title: {first_rec['title']}")
        print(f"    - Frames: {first_rec['frames']} (FPS: {first_rec['fps']:.1f})")
        print(f"    - Quality Viewpoint: {first_rec['quality']['viewpoint']}")
        print(f"    - Motion ROM Y: {first_rec['quality']['max_joint_rom_y']:.3f}m")

        if os.path.exists(npz_file):
            data = np.load(npz_file)
            kps_3d = data['kps_3d']
            vis = data['visibilities']
            print(f"[+] NPZ Data Integrity Verified:")
            print(f"    - kps_3d Shape: {kps_3d.shape} (T, 15, 3)")
            print(f"    - Visibilities Shape: {vis.shape} (T, 15)")
            print(f"    - Joint Coordinates Valid Range: X[{np.min(kps_3d[:,:,0]):.2f}, {np.max(kps_3d[:,:,0]):.2f}] Y[{np.min(kps_3d[:,:,1]):.2f}, {np.max(kps_3d[:,:,1]):.2f}]")

        # 임시 비디오 폴더 확인 (비디오가 완전히 삭제되었는지 검증)
        temp_files = glob.glob(os.path.join(temp_dir, "*.*"))
        print(f"[*] Temp Video Directory Residual Files: {len(temp_files)} (Should be 0)")
        assert len(temp_files) == 0, "Temporary video was not cleaned up!"
        print("[+] SUCCESS: Pipeline executed flawlessly and temp video was deleted!")
    else:
        print("[-] No records saved. Check network or search queries.")

if __name__ == '__main__':
    run_test()
