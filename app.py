# -*- coding: utf-8 -*-
"""
vidio-health-trainer Hugging Face Space Entrypoint (Gradio SDK Free Tier)
Purpose: 
  - Docker 유료 결제 없이 Hugging Face Spaces의 100% 무료 티어(2 vCPU, 16GB RAM) 활용
  - Gradio SDK를 통해 기존 FastAPI 웹 서비스(src.web.api.main:app)를 그대로 마운트하여 서빙
  - 루트 URL(/)에서 기존 HTML5/JS 대시보드를 그대로 서빙하고, /gradio 에서 보조 UI 제공
"""

import os
import sys

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.api.main import app
import gradio as gr

# Gradio Companion 인터페이스 (Hugging Face Spaces Gradio SDK 필수 조건 충족)
with gr.Blocks(title="vidio-health-trainer | AI 실시간 운동 자세 분석") as demo:
    gr.Markdown(
        """
        # 🏋️ vidio-health-trainer AI 실시간 운동 자세 분석 플랫폼
        본 서비스는 **FastAPI + MediaPipe Heavy + RTMPose-m (Halpe26)** 기반의 8대 운동 33종 결함 실시간 분석 엔진입니다.
        
        - 🌐 **정식 웹 대시보드 바로가기**: [대시보드 메인 페이지](/static/index.html) (또는 루트 `/`)
        - 📖 **인터랙티브 API 문서**: [FastAPI Swagger Docs](/docs)
        - 🩺 **시스템 상태 점검**: [API Health Check](/api/health)
        """
    )
    with gr.Row():
        gr.Markdown(
            """
            > **안내**: 상단 메뉴 또는 [정식 웹 대시보드](/static/index.html)로 접속하시면 
            > 비디오 업로드, 5대 시점 선택, 횟수별 33종 결함 실시간 판정 및 스켈레톤 리플레이를 
            > 오리지널 UI로 이용하실 수 있습니다.
            """
        )

# 기존 FastAPI 인스턴스에 Gradio를 /gradio 경로로 마운트 (루트 '/'는 기존 웹 UI 유지)
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces 기본 포트 7860 바인딩
    uvicorn.run(app, host="0.0.0.0", port=7860)
