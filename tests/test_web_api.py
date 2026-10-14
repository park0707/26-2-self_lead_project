# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_web_api.py
Purpose: FastAPI 웹 백엔드 서버 및 엔드포인트 전수 E2E 단위/통합 테스트
         - 헬스체크 및 사전 워밍 검증 (/api/health, /api/exercises, /api/views)
         - 실시간 시퀀스 분석 엔드포인트 검증 (/api/analyze/sequence)
         - 비디오 파일 업로드 E2E 파이프라인 검증 (/api/analyze/video)
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

from src.web.api.main import app


def create_dummy_exercise_video(output_path: str, num_frames: int = 40, width: int = 640, height: int = 480):
    """테스트용 가상 비디오 파일 생성 (인간 형상 막대 스켈레톤이 움직이는 영상)"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for f in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # 단색 배경
        img[:] = (30, 30, 30)

        # 주기적 스쿼트 유사 동작 y좌표
        offset_y = int(np.sin(f / 6.0) * 30.0)

        # 머리
        head_pos = (width // 2, 100 + offset_y)
        cv2.circle(img, head_pos, 25, (200, 200, 200), -1)

        # 몸통
        neck_pos = (width // 2, 125 + offset_y)
        pelvis_pos = (width // 2, 260 + offset_y)
        cv2.line(img, neck_pos, pelvis_pos, (0, 255, 0), 6)

        # 어깨 및 팔
        cv2.line(img, (width // 2 - 40, 150 + offset_y), (width // 2 + 40, 150 + offset_y), (255, 0, 0), 4)
        cv2.line(img, (width // 2 - 40, 150 + offset_y), (width // 2 - 50, 220 + offset_y), (255, 0, 0), 4)
        cv2.line(img, (width // 2 + 40, 150 + offset_y), (width // 2 + 50, 220 + offset_y), (255, 0, 0), 4)

        # 다리 (골반 -> 무릎 -> 발목)
        l_knee = (width // 2 - 35, 350 + offset_y // 2)
        r_knee = (width // 2 + 35, 350 + offset_y // 2)
        l_ankle = (width // 2 - 40, 440)
        r_ankle = (width // 2 + 40, 440)

        cv2.line(img, pelvis_pos, l_knee, (0, 150, 255), 5)
        cv2.line(img, pelvis_pos, r_knee, (0, 150, 255), 5)
        cv2.line(img, l_knee, l_ankle, (0, 150, 255), 5)
        cv2.line(img, r_knee, r_ankle, (0, 150, 255), 5)

        out.write(img)

    out.release()


def run_api_tests():
    print("=" * 95)
    print("  ★ [vidio-health-trainer] Step 3: FastAPI 백엔드 웹 서버 E2E 통합 테스트")
    print("=" * 95)

    # TestClient 컨텍스트 (Lifespan 사전 워밍 자동 실행)
    with TestClient(app) as client:
        # 1. 루트 엔드포인트 테스트
        print("\n[*] 1. GET / 엔드포인트 검증:")
        res = client.get("/")
        assert res.status_code == 200, f"Root API failed: {res.text}"
        assert "vidio-health-trainer" in res.text, "Root page HTML missing title"
        print(f"    ✔ 루트 대시보드 웹 UI(HTML) 서빙 확인 (응답 크기: {len(res.text)} bytes)")

        # 1-1. 정적 파일(CSS, JS) 서빙 검증
        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200 and len(res_css.text) > 1000
        res_js = client.get("/static/app.js")
        assert res_js.status_code == 200 and len(res_js.text) > 1000
        print(f"    ✔ 정적 에셋 서빙 확인 (style.css: {len(res_css.text)}B, app.js: {len(res_js.text)}B)")


        # 2. 헬스체크 및 사전 워밍 상태 테스트
        print("\n[*] 2. GET /api/health 엔드포인트 검증:")
        res = client.get("/api/health")
        assert res.status_code == 200, f"Health API failed: {res.text}"
        health_data = res.json()
        print(f"    ✔ Stage1 준비 상태: {health_data['stage1_ready']}")
        print(f"    ✔ Stage2 준비 상태: {health_data['stage2_ready']}")
        print(f"    ✔ 사전 워밍 완료 운동 수: {health_data['warmed_exercises_count']}/8")
        print(f"    ✔ 총 지원 세부 결함 모델: {health_data['total_defects_count']}종")
        assert health_data["stage1_ready"] is True
        assert health_data["stage2_ready"] is True
        assert health_data["total_defects_count"] == 33

        # 3. 운동 목록 및 시점 메타데이터 테스트
        print("\n[*] 3. GET /api/exercises 및 /api/views 검증:")
        res_ex = client.get("/api/exercises")
        assert res_ex.status_code == 200
        ex_data = res_ex.json()
        print(f"    ✔ 등록된 운동 종목 수: {ex_data['count']}개")

        res_v = client.get("/api/views")
        assert res_v.status_code == 200
        v_data = res_v.json()
        print(f"    ✔ 등록된 공식 5대 시점 수: {v_data['count']}개")
        for code, meta in v_data["views"].items():
            print(f"       - {meta.get('code', code)}: {meta['name']} ({meta.get('desc', '')})")


        # 4. 실시간 시퀀스 분석 엔드포인트 테스트
        print("\n[*] 4. POST /api/analyze/sequence (관절 시퀀스 실시간 추론) 검증:")
        np.random.seed(42)
        dummy_kps = (np.random.randn(64, 33, 3) * 0.1).tolist()
        payload = {
            "exercise_id": 0,  # 바벨 스쿼트
            "camera_view": "시점 C: 정면",
            "keypoints": dummy_kps,
            "source_type": "mediapipe_33"
        }
        t0 = time.time()
        res_seq = client.post("/api/analyze/sequence", json=payload)
        elapsed_seq = (time.time() - t0) * 1000.0
        assert res_seq.status_code == 200, f"Sequence analyze failed: {res_seq.text}"
        seq_data = res_seq.json()
        print(f"    ✔ 분석 성공 여부: {seq_data['success']}")
        print(f"    ✔ 운동 종목: {seq_data['exercise']['exercise_name']}")
        print(f"    ✔ 결함 진단 개수: {len(seq_data['defects'])}개")
        print(f"    ✔ API 왕복 지연시간: {elapsed_seq:.2f} ms")

        # 5. 비디오 업로드 E2E 파이프라인 테스트
        print("\n[*] 5. POST /api/analyze/video (비디오 업로드 및 횟수 분할/결함 진단 E2E) 검증:")
        temp_video_path = os.path.join(tempfile.gettempdir(), "test_squat_sample.mp4")
        try:
            create_dummy_exercise_video(temp_video_path, num_frames=30)
            print(f"    - 가상 테스트 비디오 생성 완료 ({temp_video_path}, 30프레임)")

            t_vid_0 = time.time()
            with open(temp_video_path, "rb") as vf:
                files = {"video_file": ("test_squat_sample.mp4", vf, "video/mp4")}
                data = {
                    "exercise_id": 0,  # 바벨 스쿼트
                    "camera_view": "시점 C: 정면",
                    "channel_id": "test-channel-123"
                }
                res_vid = client.post("/api/analyze/video", files=files, data=data)

            elapsed_vid = time.time() - t_vid_0
            assert res_vid.status_code == 200, f"Video analyze failed: {res_vid.text}"
            vid_result = res_vid.json()

            print(f"    ✔ 비디오 E2E 분석 성공! (소요 시간: {elapsed_vid:.2f}초)")
            print(f"    ✔ 작업 ID: {vid_result['job_id']}")
            print(f"    ✔ 식별된 운동: {vid_result['exercise']['exercise_name']}")
            print(f"    ✔ 감지된 총 반복 횟수: {vid_result['total_reps']} Reps")

            for rep in vid_result["reps"]:
                assert "overlay_mp" in rep and rep["overlay_mp"] is not None, "overlay_mp 누락"
                assert "overlay_rtmp" in rep and rep["overlay_rtmp"] is not None, "overlay_rtmp 누락"
                for df in rep["defects"]:
                    print(f"          - [{df['defect_code']}] {df['defect_name']}: {df['state_label']} (확률: {df['probability']*100:.1f}%) | 피드백: \"{df['feedback']}\"")
                    assert isinstance(df["feedback"], str) and len(df["feedback"]) > 0, "피드백 규격 불일치"

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        # 6. WebSocket 실시간 알림 채널 검증
        print("\n[*] 6. WebSocket /api/ws/analyze/{channel_id} 실시간 알림 채널 검증:")
        with client.websocket_connect("/api/ws/analyze/test-client-999") as ws:
            welcome_msg = ws.receive_json()
            print(f"    ✔ 수신된 초기 연결 메시지: {welcome_msg.get('type')} - {welcome_msg.get('message')}")
            assert welcome_msg.get("type") == "connected"
            ws.send_text('{"type": "ping"}')
            pong_msg = ws.receive_json()
            print(f"    ✔ 핑-퐁 응답 수신: {pong_msg.get('type')}")
            assert pong_msg.get("type") == "pong"

    print("\n" + "=" * 95)
    print("  ★ [PASS] Step 3: FastAPI 백엔드 웹 서버 E2E 통합 테스트 100% 통과!")
    print("=" * 95)



if __name__ == "__main__":
    run_api_tests()
