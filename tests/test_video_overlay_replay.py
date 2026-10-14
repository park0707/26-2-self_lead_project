# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_video_overlay_replay.py
Purpose: 비디오 오버레이 스켈레톤 리플레이 (MediaPipe 33 + RTMPose 26) 단위 및 통합 무결성 검증 하네스
         1. RepetitionDetail 스키마 규격 검증 (overlay_mp, overlay_rtmp, start_sec, end_sec)
         2. 64-Frame 불변식 (Length == 64) 및 2D 정규화 좌표계 [0.0, 1.0] 범위 검증
         3. VideoAnalysisPipeline E2E 비디오 분석 및 듀얼 비전 오버레이 출력 정합성 검증
         4. FastAPI /api/analyze/video 엔드포인트 REST JSON 직렬화 검증
"""

import os
import sys
import time
import tempfile
import numpy as np
import cv2
from fastapi.testclient import TestClient

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.core.schemas import RepetitionDetail, VideoAnalysisResponse, DefectPrediction, ExercisePrediction
from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.api.main import app


def test_schema_overlay_fields():
    """1. RepetitionDetail 스키마 신규 오버레이 필드 검증"""
    print("\n" + "=" * 80)
    print("▶ [Test 1] RepetitionDetail 스키마 2D 오버레이 규격 및 불변식 검증")
    print("=" * 80)

    # 가상 64프레임 MediaPipe 33관절 및 RTMPose 26관절 데이터 생성
    dummy_mp = [[[0.5, 0.5] for _ in range(33)] for _ in range(64)]
    dummy_rtmp = [[[0.5, 0.5, 0.95] for _ in range(26)] for _ in range(64)]

    rep = RepetitionDetail(
        rep_index=1,
        start_frame=0,
        end_frame=60,
        start_sec=0.0,
        end_sec=2.0,
        duration_sec=2.0,
        is_clean=True,
        defects=[],
        keypoints_preview=[[[0.0, 0.0, 0.0] for _ in range(15)] for _ in range(64)],
        overlay_mp=dummy_mp,
        overlay_rtmp=dummy_rtmp
    )

    assert rep.overlay_mp is not None, "overlay_mp 누락"
    assert rep.overlay_rtmp is not None, "overlay_rtmp 누락"
    assert len(rep.overlay_mp) == 64, f"overlay_mp 64프레임 불변식 위반: {len(rep.overlay_mp)}"
    assert len(rep.overlay_rtmp) == 64, f"overlay_rtmp 64프레임 불변식 위반: {len(rep.overlay_rtmp)}"
    assert len(rep.overlay_mp[0]) == 33, f"MediaPipe 33관절 규격 위반: {len(rep.overlay_mp[0])}"
    assert len(rep.overlay_rtmp[0]) == 26, f"RTMPose 26관절 규격 위반: {len(rep.overlay_rtmp[0])}"
    assert rep.start_sec == 0.0 and rep.end_sec == 2.0, "반복 시간 바운딩 오류"

    print("  ✔ RepetitionDetail Pydantic 스키마 직렬화 및 64-Frame 불변식 검증 완료")


def test_pipeline_video_overlay_generation():
    """2. VideoAnalysisPipeline 실제 비디오 분석 시 2D 오버레이 좌표 생성 검증"""
    print("\n" + "=" * 80)
    print("▶ [Test 2] VideoAnalysisPipeline 실제 비디오 분석 및 2D 오버레이 좌표 정합성 검증")
    print("=" * 80)

    # 테스트용 가상 운동 영상 생성 (30프레임, 640x480)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_video_path = tf.name

    try:
        w, h = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (w, h))

        for f in range(40):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = (35, 35, 35)

            # 운동 모션 시뮬레이션
            dy = int(np.sin(f / 5.0) * 25.0)
            cv2.circle(img, (w // 2, 120 + dy), 25, (220, 220, 220), -1)  # 머리
            cv2.line(img, (w // 2, 145 + dy), (w // 2, 280 + dy), (0, 255, 0), 6)  # 몸통
            cv2.line(img, (w // 2 - 40, 160 + dy), (w // 2 + 40, 160 + dy), (255, 0, 0), 4)  # 어깨
            cv2.line(img, (w // 2 - 40, 160 + dy), (w // 2 - 50, 240 + dy), (255, 0, 0), 4)
            cv2.line(img, (w // 2 + 40, 160 + dy), (w // 2 + 50, 240 + dy), (255, 0, 0), 4)
            cv2.line(img, (w // 2, 280 + dy), (w // 2 - 35, 370 + dy // 2), (0, 255, 255), 4)  # 다리
            cv2.line(img, (w // 2, 280 + dy), (w // 2 + 35, 370 + dy // 2), (0, 255, 255), 4)
            cv2.line(img, (w // 2 - 35, 370 + dy // 2), (w // 2 - 40, 450), (0, 255, 255), 4)
            cv2.line(img, (w // 2 + 35, 370 + dy // 2), (w // 2 + 40, 450), (0, 255, 255), 4)
            out.write(img)

        out.release()

        # 파이프라인 싱글톤 획득 및 분석 수행
        pipeline = VideoAnalysisPipeline.get_instance()
        res = pipeline.analyze_video(
            video_path=temp_video_path,
            sample_stride=2,
            include_skeleton_preview=True
        )

        assert res.success is True, "비디오 분석 실패"
        assert res.video_info is not None, "video_info 누락"
        assert res.video_info["width"] == 640 and res.video_info["height"] == 480, "비디오 해상도 불일치"
        assert len(res.reps) > 0, "반복 미검출"

        for rep in res.reps:
            print(f"  ✔ Rep #{rep.rep_index}: {rep.start_sec}s ~ {rep.end_sec}s ({rep.duration_sec}초)")
            assert rep.overlay_mp is not None, "overlay_mp 미생성"
            assert rep.overlay_rtmp is not None, "overlay_rtmp 미생성"
            assert len(rep.overlay_mp) == 64, f"overlay_mp 64프레임 불일치: {len(rep.overlay_mp)}"
            assert len(rep.overlay_rtmp) == 64, f"overlay_rtmp 64프레임 불일치: {len(rep.overlay_rtmp)}"

            mp_arr = np.array(rep.overlay_mp)
            rtmp_arr = np.array(rep.overlay_rtmp)

            # 정규화 좌표계 [0.0, 1.0] 범위 검증
            assert np.all(mp_arr >= 0.0) and np.all(mp_arr <= 1.0), "MediaPipe 2D 정규화 좌표 범위 초과"
            assert np.all(rtmp_arr[:, :, :2] >= 0.0) and np.all(rtmp_arr[:, :, :2] <= 1.0), "RTMPose 2D 정규화 좌표 범위 초과"
            print(f"     - MediaPipe 2D shape: {mp_arr.shape} (정규화 범위: [{mp_arr.min():.3f}, {mp_arr.max():.3f}])")
            print(f"     - RTMPose 2D shape: {rtmp_arr.shape} (정규화 범위: [{rtmp_arr[:, :, :2].min():.3f}, {rtmp_arr[:, :, :2].max():.3f}])")

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    print("  ✔ VideoAnalysisPipeline 듀얼 비전 오버레이 좌표 생성 검증 완료")


def test_api_video_upload_endpoint():
    """3. FastAPI /api/analyze/video 엔드포인트 REST JSON 응답 검증"""
    print("\n" + "=" * 80)
    print("▶ [Test 3] FastAPI /api/analyze/video 엔드포인트 E2E REST 검증")
    print("=" * 80)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_video_path = tf.name

    try:
        w, h = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (w, h))

        for f in range(30):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = (30, 30, 30)
            cv2.circle(img, (w // 2, 150), 30, (200, 200, 200), -1)
            cv2.line(img, (w // 2, 180), (w // 2, 320), (0, 255, 0), 4)
            out.write(img)
        out.release()

        with TestClient(app) as client:
            with open(temp_video_path, "rb") as f:
                response = client.post(
                    "/api/analyze/video",
                    files={"video_file": ("test_rep.mp4", f, "video/mp4")},
                    data={"camera_view": "시점 C: 정면", "sample_stride": 2}
                )

            assert response.status_code == 200, f"API 응답 실패: {response.status_code}, {response.text}"
            data = response.json()
            assert data["success"] is True
            assert "video_info" in data
            assert len(data["reps"]) > 0
            rep0 = data["reps"][0]
            assert "overlay_mp" in rep0 and rep0["overlay_mp"] is not None
            assert "overlay_rtmp" in rep0 and rep0["overlay_rtmp"] is not None
            assert len(rep0["overlay_mp"]) == 64
            assert len(rep0["overlay_rtmp"]) == 64
            print(f"  ✔ REST API JSON 응답 정상 확인 (video_info={data['video_info']}, reps={len(data['reps'])}회)")
            print(f"     - Rep #1 overlay_mp 관절 수: {len(rep0['overlay_mp'][0])}, overlay_rtmp 관절 수: {len(rep0['overlay_rtmp'][0])}")

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    print("  ✔ FastAPI REST 엔드포인트 2D 오버레이 직렬화 100% 통과")


def run_all_tests():
    t0 = time.time()
    test_schema_overlay_fields()
    test_pipeline_video_overlay_generation()
    test_api_video_upload_endpoint()
    print("\n" + "=" * 95)
    print(f"  ★ [PASS] 비디오 오버레이 듀얼 스켈레톤 리플레이 하네스 테스트 전원 통과! ({time.time() - t0:.2f}초 소요)")
    print("=" * 95)


if __name__ == "__main__":
    run_all_tests()
