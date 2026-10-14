# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/web/run_server.py
Purpose: 웹 대시보드 및 AI 분석 백엔드 서버 단일 실행 스크립트
         실행 방법: python src/web/run_server.py
"""

import sys
import os
import uvicorn

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')


def main():
    print("=" * 75)
    print("  ★ [vidio-health-trainer] 웹 서비스 서버 시작 중...")
    print("  - 브라우저 접속 주소: http://localhost:8000")
    print("  - API 문서 (Swagger): http://localhost:8000/docs")
    print("=" * 75)
    uvicorn.run("src.web.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
