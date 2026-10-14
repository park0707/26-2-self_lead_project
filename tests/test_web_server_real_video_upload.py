# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\test_web_server_real_video_upload.py
Purpose: 실제 FastAPI 웹 서버 엔드포인트(/api/analyze/video)에 data/일반화 비디오를 
         직접 HTTP POST(멀티파트 폼데이터)로 업로드하여, 실제 서버 단에서도
         비운동 구간 소거 및 5회 분할이 100% 정상 작동하는지 실증 검증
"""

import os
import sys
import time
from fastapi.testclient import TestClient

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.api.main import app

def test_web_server_upload():
    video_path = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
    assert os.path.exists(video_path), f"비디오 파일이 없습니다: {video_path}"

    print("=" * 85)
    print("  ★ [vidio-health-trainer] 실제 웹 서버 엔드포인트(/api/analyze/video) 실증 테스트")
    print(f"  - 업로드 대상: {video_path}")
    print("=" * 85)

    with TestClient(app) as client:
        print("\n[*] 1. 웹 서버(/api/analyze/video)로 실제 영상 멀티파트 업로드 전송 중...")
        t0 = time.time()
        with open(video_path, "rb") as f:
            files = {"video_file": ("barbell_squat_front.mp4", f, "video/mp4")}
            data = {
                "camera_view": "시점 C: 정면"
            }
            response = client.post("/api/analyze/video", files=files, data=data)
            
        elapsed = time.time() - t0
        print(f"    ✔ HTTP 상태 코드: {response.status_code} (소요 시간: {elapsed:.2f}초)")
        assert response.status_code == 200, f"서버 응답 실패: {response.text}"

        res_data = response.json()
        
        print("\n[*] 2. 서버 응답 데이터 검증:")
        act = res_data.get("active_period", {})
        print(f"    - 활성 운동 구간 (active_period):")
        print(f"      * 시작 프레임: F{act.get('start_frame')}")
        print(f"      * 종료 프레임: F{act.get('end_frame')}")
        print(f"      * 활성 시간: {act.get('duration_sec')}초")
        print(f"      * 소거된 비운동 프레임 수: {act.get('filtered_frames_count')} 프레임")

        exercise = res_data.get("exercise", {})
        print(f"\n    - 1단계 AI 종목 판정: {exercise.get('exercise_name')} (확신도: {exercise.get('confidence')*100:.1f}%)")
        print(f"    - 총 인식된 횟수: {res_data.get('total_reps')}회")

        reps = res_data.get("reps", [])
        for r in reps:
            print(f"      ▶ Rep #{r['rep_index']}: F{r['start_frame']} ~ F{r['end_frame']} ({r['duration_sec']}초, 클린: {r['is_clean']})")

        # 검증 단언문 (VIDEO 모드 및 540p 최적화 환경의 유효 범위 검증)
        assert 130 <= act.get("start_frame") <= 160, f"active start out of range: {act}"
        assert 560 <= act.get("end_frame") <= 585, f"active end out of range: {act}"
        assert res_data.get("total_reps") == 5, f"total reps != 5: {res_data.get('total_reps')}"
        assert "스쿼트" in exercise.get("exercise_name"), f"exercise != 스쿼트: {exercise}"

        print("\n" + "=" * 85)
        print("  ★ [SUCCESS] 실제 웹 서버에서도 비운동 소거 및 횟수 분할이 완벽하게 적용됨을 확인!")
        print("=" * 85)

if __name__ == "__main__":
    test_web_server_upload()
