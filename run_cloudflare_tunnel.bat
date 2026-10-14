@echo off
chcp 65001 > nul
echo ===============================================================================
echo   ★ [vidio-health-trainer] Cloudflare Tunnel 초고속 무료 배포 런처
echo   - 로컬 NVIDIA GeForce MX570 A GPU 가속 (분석 12~15초 실시간)
echo   - 포트포워딩 / 공유기 설정 없이 무료 공인 HTTPS 보안 도메인 즉시 발급
echo ===============================================================================

echo.
echo [1/2] 로컬 FastAPI 웹 서버(포트 8000) 구동 확인 중...
netstat -ano | findstr 8000 | findstr LISTENING > nul
if %errorlevel% neq 0 (
    echo [*] 백엔드 웹 서버가 실행되어 있지 않습니다. 새 창에서 서버를 기동합니다...
    start "vidio-health-trainer Web Server" cmd /k "python -m uvicorn src.web.api.main:app --host 0.0.0.0 --port 8000"
    timeout /t 5 > nul
) else (
    echo [✔] 로컬 FastAPI 웹 서버가 포트 8000에서 정상 가동 중입니다.
)

echo.
echo [2/2] Cloudflare Tunnel(cloudflared) 실행 중...
echo [*] 아래 출력되는 'https://*.trycloudflare.com' 주소가 전 세계에서 접속 가능한 무료 보안 HTTPS 주소입니다!
echo ===============================================================================

REM cloudflared 실행 (없으면 자동 다운로드 안내)
where cloudflared > nul 2>&1
if %errorlevel% neq 0 (
    echo [*] cloudflared가 설치되어 있지 않습니다.
    echo [*] winget을 통해 cloudflared를 설치합니다...
    winget install --id Cloudflare.cloudflared -e --source winget
)

cloudflared tunnel --url http://localhost:8000
pause
