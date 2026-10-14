# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/api/websocket.py
Purpose: 실시간 비디오 분석 진행률 알림 및 웹캠 키포인트 스트리밍 WebSocket 라우터
"""

import sys
import os
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket")
router = APIRouter(prefix="/api/ws", tags=["WebSocket"])


class ConnectionManager:
    """웹소켓 클라이언트 세션 관리자"""

    def __init__(self):
        # job_id or client_id -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel_id: str):
        await websocket.accept()
        if channel_id not in self.active_connections:
            self.active_connections[channel_id] = []
        self.active_connections[channel_id].append(websocket)
        logger.info(f"WebSocket 클라이언트 연결: channel={channel_id}")

    def disconnect(self, websocket: WebSocket, channel_id: str):
        if channel_id in self.active_connections:
            if websocket in self.active_connections[channel_id]:
                self.active_connections[channel_id].remove(websocket)
            if not self.active_connections[channel_id]:
                del self.active_connections[channel_id]
        logger.info(f"WebSocket 클라이언트 연결 종료: channel={channel_id}")

    async def send_progress(
        self, channel_id: str, stage: str, progress_percent: float, message: str, data: Optional[Dict[str, Any]] = None
    ):
        """특정 채널에 진행률 메시지 비동기 전송"""
        if channel_id not in self.active_connections:
            return

        payload = {
            "type": "progress",
            "channel_id": channel_id,
            "stage": stage,
            "progress_percent": round(progress_percent, 1),
            "message": message,
            "data": data or {}
        }
        dead_sockets = []
        for ws in self.active_connections[channel_id]:
            try:
                await ws.send_json(payload)
            except Exception as e:
                logger.warning(f"메시지 전송 실패 (소켓 해제 예정): {e}")
                dead_sockets.append(ws)

        for dead in dead_sockets:
            self.disconnect(dead, channel_id)


ws_manager = ConnectionManager()


@router.websocket("/analyze/{channel_id}")
async def websocket_analysis_endpoint(websocket: WebSocket, channel_id: str):
    """
    클라이언트가 채널 ID(또는 작업 ID)로 접속하여 실시간 비디오 분석 진행률 수신
    """
    await ws_manager.connect(websocket, channel_id)
    try:
        # 최초 연결 환영 메시지
        await websocket.send_json({
            "type": "connected",
            "channel_id": channel_id,
            "message": "실시간 분석 알림 채널에 성공적으로 연결되었습니다."
        })

        while True:
            # 클라이언트로부터 핑 또는 추가 명령 수신
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel_id)
    except Exception as e:
        logger.error(f"WebSocket 예외 발생: {e}")
        ws_manager.disconnect(websocket, channel_id)
