# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/generate_deployment_planning_report.py
Purpose: 8주차 GitHub 연동 무료 자동배포 CI/CD 파이프라인 기획 보고서 생성 (.docx 및 .md)
"""

import os
import sys
from datetime import datetime
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_DIR = r"D:\자주프 문서\8주차"
MD_DIR = r"D:\자주프 문서\8주차\md"
os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

REPORT_NAME = "8주차_GitHub_연동_무료_자동배포_CI_CD_파이프라인_기획보고서"
DOCX_PATH = os.path.join(DOCX_DIR, f"{REPORT_NAME}.docx")
MD_PATH = os.path.join(MD_DIR, f"{REPORT_NAME}.md")

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def add_callout(doc, text, title="핵심 기획 요약 / EXECUTIVE SUMMARY"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    set_cell_background(c, "F0F4F8")
    set_cell_margins(c, top=140, bottom=140, left=200, right=200)
    
    tcPr = c._element.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="005696"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)
    
    p = c.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}\n")
    run_t.bold = True
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor(0, 86, 150)
    
    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(50, 50, 50)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_docx():
    doc = docx.Document()
    
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.85)
        sec.right_margin = Inches(0.85)
        
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    r_t = p_title.add_run("vidio-health-trainer GitHub 연동 무료 자동배포 CI/CD 파이프라인 기획서")
    r_t.bold = True
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(17)
    r_t.font.color.rgb = RGBColor(16, 44, 87)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    r_s = p_sub.add_run("Hugging Face Spaces (16GB RAM 무료) 기반 Docker 컨테이너 및 GitHub Actions 완전 자동 동기화 배포 전략")
    r_s.font.name = "맑은 고딕"
    r_s.font.size = Pt(10.5)
    r_s.font.color.rgb = RGBColor(100, 110, 120)
    
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("공식 프로젝트 명칭", "vidio-health-trainer"),
        ("기획 기준 일자 / 주차", f"{datetime.now().strftime('%Y년 %m월 %d일')} (8주차)"),
        ("권장 배포 플랫폼", "Hugging Face Spaces (2 vCPU, 16GB RAM, Docker Free Tier)"),
        ("CI/CD 자동화 방식", "GitHub Actions 워크플로 (main 브랜치 push 시 자동 트리거 및 무중단 배포)")
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        row = tbl_meta.rows[r_idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.6)
        set_cell_background(c0, "EAEFF5")
        set_cell_background(c1, "F8FAFC")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "맑은 고딕"
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(30, 41, 59)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(v)
        r1.font.name = "맑은 고딕"
        r1.font.size = Pt(9.5)
        if "Hugging Face" in v or "GitHub Actions" in v:
            r1.bold = True
            r1.font.color.rgb = RGBColor(0, 86, 150)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    add_callout(doc,
        "1. AI/비전 웹서비스 배포의 본질적 난제: vidio-health-trainer는 MediaPipe Heavy, RTMPose-m, 33종 결함 모델, 540p 비디오 프레임 버퍼를 구동하므로 최소 1GB~1.5GB 이상의 RAM과 28초 이상의 지속 처리 시간이 필수적입니다. 일반 무료 호스팅(Render, Fly.io, Koyeb: 256~512MB RAM)은 비디오 분석 중 OOM(Out of Memory)으로 서버가 강제 종료되는 치명적 한계가 존재합니다.\n"
        "2. 100% 무료 최적 해법 (Hugging Face Spaces): 신용카드 등록 없이 영구 무료로 2 vCPU, 16GB RAM, 50GB 디스크를 제공하는 유일한 글로벌 AI 호스팅 플랫폼입니다. Dockerfile을 통한 완벽한 환경 격리와 무료 HTTPS 도메인(hf.space)을 기본 제공합니다.\n"
        "3. GitHub 연동 자동화 (Git-Sync CI/CD): 개발자가 로컬에서 git push origin main만 수행하면, GitHub Actions가 트리거되어 Hugging Face Space 리포지토리로 코드를 자동 동기화하고, Space가 Docker 이미지를 자동 재빌드하여 100% 무중단 자동 배포를 완성합니다.\n"
        "4. 대안 하이브리드 플랜: 외장 GPU(NVIDIA GeForce MX570 A)를 활용하여 12~15초대 초고속 추론을 제공하는 Cloudflare Tunnel 기반 로컬 GPU 무료 배포 방안을 병행 제시합니다.",
        "기획 핵심 요약 (Executive Summary)"
    )

    h1 = doc.add_heading("1. 무료 호스팅 플랫폼별 AI/비전 적합성 비교 분석", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    
    tbl_plat = doc.add_table(rows=5, cols=5)
    tbl_plat.alignment = WD_TABLE_ALIGNMENT.CENTER
    plat_headers = ["호스팅 플랫폼", "무료 스펙 (CPU/RAM/Disk)", "GitHub 자동 배포", "비디오/AI 적합성", "종합 평가 및 권장도"]
    plat_rows = [
        ("Hugging Face Spaces\n(Docker Space)", "2 vCPU / 16GB RAM\n50GB Disk (카드 불필요)", "GitHub Actions\nGit-Sync 자동 연동", "탁월함 (16GB RAM으로\nOOM 위험 0%, 타임아웃 없음)", "★ 1순위 강력 추천\n(AI/CV 서비스 최적)"),
        ("Oracle Cloud Always Free\n(OCI Ampere A1)", "4 OCPU / 24GB RAM\n200GB SSD (카드 인증 필요)", "GitHub Actions SSH\nDocker Compose 배포", "탁월함 (완전 독립 VPS,\n24/365 상시 구동)", "★ 2순위 추천\n(카드 인증 가능 시)"),
        ("Cloudflare Tunnel\n+ 로컬 GPU 서버", "로컬 CPU + MX570 GPU\n무제한 스토리지", "GitHub Actions / Webhook\n로컬 데몬 자동 풀", "최고 성능 (외장 GPU로\n12~15초 초고속 분석)", "★ 하이브리드 추천\n(초고속 GPU 가속 시)"),
        ("Render / Koyeb / Vercel\n(일반 웹 PaaS)", "0.1 CPU / 512MB RAM\n(15분 후 슬립)", "GitHub 리포지토리\n기본 Direct 연동", "치명적 부적합 (512MB 한계로\n비디오 분석 시 OOM Kill)", "비추천 (운영 불가)")
    ]
    for c_idx, h in enumerate(plat_headers):
        cell = tbl_plat.rows[0].cells[c_idx]
        set_cell_background(cell, "005696")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, r_data in enumerate(plat_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tbl_plat.rows[r_idx].cells[c_idx]
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.5)
            if "1순위" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(0, 86, 150)
            elif "비추천" in val or "치명적" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(200, 30, 30)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    h2 = doc.add_heading("2. 최우선 플랜: Hugging Face Spaces + GitHub Actions CI/CD 아키텍처", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "본 기획의 핵심은 개발자가 기존에 사용하던 GitHub 리포지토리(origin/main) 작업 흐름을 100% 그대로 유지하면서, "
        "코드 푸시 즉시 16GB RAM 무료 AI 클라우드에 자동으로 빌드 및 배포되는 CI/CD 파이프라인을 구축하는 것입니다.\n"
    )
    
    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.left_indent = Inches(0.2)
    p_arch.add_run("■ End-to-End 자동 배포 프로세스 (4단계):\n").bold = True
    p_arch.add_run(
        "1. [Developer Push]: 개발자가 로컬에서 코드 수정 후 'git push origin main' 실행.\n"
        "2. [GitHub Actions Trigger]: GitHub 리포지토리의 .github/workflows/deploy.yml이 자동으로 기동.\n"
        "3. [Hugging Face Git-Sync]: GitHub Actions 러너가 HF_TOKEN(비밀 키)을 인증하여 변경 사항을 Hugging Face Space Git 리포지토리로 안전하게 푸시.\n"
        "4. [Docker Auto-Build & Launch]: Hugging Face Space가 Dockerfile을 감지하여 컨테이너 이미지를 빌드하고, 포트 7860으로 FastAPI 웹 서버를 자동 실행 (무료 HTTPS URL 즉시 활성화)."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h3 = doc.add_heading("3. 배포 파이프라인 구성 명세 (Configuration Specifications)", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run("자동 배포 시스템 구축을 위해 프로젝트 루트에 추가되는 4대 핵심 설정 파일 명세입니다:\n")
    
    tbl_files = doc.add_table(rows=5, cols=3)
    tbl_files.alignment = WD_TABLE_ALIGNMENT.CENTER
    files_headers = ["파일명", "역할 및 목적", "핵심 설정 내용"]
    files_rows = [
        ("Dockerfile", "Hugging Face Space 컨테이너 규격 정의", "Debian Linux 기반, libgl1/ffmpeg 설치, Python 의존성 설치, uvicorn 포트 7860 바인딩"),
        (".dockerignore", "빌드 이미지 경량화 및 불필요 파일 제외", ".git, venv, 임시 비디오 파일(data/일반화), 캐시 파일 제외"),
        ("requirements.txt", "클라우드 런타임 의존성 패키지 명세", "fastapi, uvicorn, mediapipe, rtmlib, onnxruntime, opencv-python-headless, scikit-learn 등"),
        (".github/workflows/deploy.yml", "GitHub Actions CI/CD 자동화 파이프라인", "main 브랜치 push 감지 시 huggingface_hub Git-Sync 수행")
    ]
    for c_idx, h in enumerate(files_headers):
        cell = tbl_files.rows[0].cells[c_idx]
        set_cell_background(cell, "162C57")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, r_data in enumerate(files_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tbl_files.rows[r_idx].cells[c_idx]
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(40, 40, 40)
            if c_idx == 0:
                run.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    h4 = doc.add_heading("4. 대안 플랜: Cloudflare Tunnel 기반 로컬 GPU 가속 무료 배포", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "클라우드 CPU 환경(28초)보다 더 빠른 연산이 필요하거나, 사용자 PC의 외장 그래픽카드(NVIDIA GeForce MX570 A, 4GB VRAM)를 "
        "활용하여 실시간 수준(12~15초)의 분석 속도를 전 세계 사용자에게 무료로 서비스하고자 할 경우의 하이브리드 아키텍처입니다.\n"
    )
    
    p_cf = doc.add_paragraph()
    p_cf.paragraph_format.left_indent = Inches(0.2)
    p_cf.add_run("■ Cloudflare Tunnel의 3대 무과금 특장점:\n").bold = True
    p_cf.add_run(
        "1. [공인 IP / 포트포워딩 불필요]: 공유기(NAT) 설정 변경이나 고정 IP 비용 없이, 로컬 데몬(cloudflared)이 아웃바운드로 Cloudflare 엣지 망과 암호화 터널을 체결.\n"
        "2. [무료 SSL/HTTPS 자동 발급]: https://*.trycloudflare.com 또는 사용자의 무료 도메인으로 엔드투엔드 보안 인증서 무료 제공.\n"
        "3. [DDoS 방어 및 CDN 가속]: 프론트엔드 정적 에셋(HTML/CSS/JS)을 Cloudflare 글로벌 CDN이 캐싱하여 초고속 전달."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h5 = doc.add_heading("5. 향후 실행 단계 및 추천 액션 아이템", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "사용자 승인 후 즉시 진행할 수 있는 단계별 실행 절차는 다음과 같습니다:\n"
        "1단계: 프로젝트 루트에 Dockerfile, requirements.txt, .dockerignore 파일 생성 및 로컬 컨테이너 무결성 검증.\n"
        "2단계: Hugging Face Spaces 계정 생성 (무료, 이메일 가입) 및 빈 Docker Space 생성.\n"
        "3단계: GitHub 리포지토리의 Settings > Secrets and variables > Actions에 HF_TOKEN(Hugging Face 발급 토큰) 등록.\n"
        "4단계: .github/workflows/deploy.yml 커밋 및 push → CI/CD 자동 배포 가동 및 퍼블릭 URL 접속 확인."
    )
    
    doc.save(DOCX_PATH)
    print(f"[+] DOCX Report generated: {DOCX_PATH}")

def build_md():
    md_content = f"""# vidio-health-trainer GitHub 연동 무료 자동배포 CI/CD 파이프라인 기획서

**작성 일자:** {datetime.now().strftime('%Y-%m-%d')} (8주차)  
**공식 프로젝트 명칭:** `vidio-health-trainer`  
**권장 배포 플랫폼:** Hugging Face Spaces (2 vCPU, 16GB RAM, Docker Free Tier)  
**CI/CD 자동화 방식:** GitHub Actions 워크플로 (main 브랜치 push 시 자동 트리거 및 무중단 배포)

---

## 1. 개요 및 무료 호스팅 플랫폼별 AI/비전 적합성 비교

`vidio-health-trainer`는 **MediaPipe Heavy (3D 33관절), RTMPose-m (Halpe26 발 3각 지지면), 1단계 8종 운동 분류기, 2단계 33종 결함 모델**과 540p 비디오 프레임 버퍼를 포함하는 고성능 비전 AI 시스템입니다.  
일반적인 웹사이트와 달리 분석 요청 1건당 **최소 1GB~1.5GB 이상의 RAM과 28초 이상의 지속 연산 시간**을 소모하므로, 무료 호스팅 선정 시 메모리 용량이 성패를 좌우합니다.

### 주요 무료 호스팅 플랫폼 비교

| 호스팅 플랫폼 | 무료 제공 스펙 | GitHub 연동 CI/CD | 비디오/AI 적합성 | 종합 평가 및 권장도 |
| :--- | :---: | :---: | :---: | :--- |
| **Hugging Face Spaces (Docker Space)** | **2 vCPU / 16GB RAM / 50GB Disk** (카드 불필요) | GitHub Actions Git-Sync 완벽 연동 | **탁월함 (16GB RAM으로 OOM 위험 0%, 타임아웃 없음)** | **★ 1순위 강력 추천 (AI/CV 서비스 최적)** |
| **Oracle Cloud Always Free (OCI Ampere A1)** | 4 OCPU / 24GB RAM / 200GB SSD (카드 인증 필요) | GitHub Actions SSH / Docker Compose | **탁월함 (독립 Linux VPS, 24/365 상시 구동)** | **★ 2순위 추천 (카드 인증 가능 시)** |
| **Cloudflare Tunnel + 로컬 GPU 서버** | 로컬 CPU + MX570 GPU / 무제한 스토리지 | GitHub Actions / Webhook 자동 풀 | **최고 성능 (외장 GPU로 12~15초 초고속 분석)** | **★ 하이브리드 추천 (초고속 GPU 가속 시)** |
| **Render / Koyeb / Vercel (일반 웹 PaaS)** | 0.1 CPU / 512MB RAM (15분 후 슬립) | GitHub Direct 연동 | 치명적 부적합 (512MB 한계로 비디오 분석 시 OOM Kill) | **비추천 (운영 불가)** |

---

## 2. 최우선 권장 플랜: Hugging Face Spaces + GitHub Actions CI/CD 아키텍처

```mermaid
flowchart LR
    Dev[개발자 로컬 PC] -->|git push origin main| GitHub[GitHub 원격 리포지토리]
    GitHub -->|Push 이벤트 트리거| GHA[GitHub Actions Runner]
    GHA -->|HF_TOKEN 인증 및 Git-Sync| HF[Hugging Face Spaces Space]
    HF -->|Dockerfile 감지| Build[Docker 컨테이너 자동 빌드]
    Build -->|포트 7860 바인딩| Live[FastAPI 웹 서비스 배포 완료]
    User[전 세계 사용자 브라우저] -->|HTTPS 접속| Live
```

### 동작 원리:
1. 개발자가 로컬에서 기능 개발 또는 버그 수정을 완료하고 `git push origin main`을 수행합니다.
2. GitHub Actions가 즉시 트리거되어 프로젝트의 최신 코드를 읽어옵니다.
3. GitHub Secrets에 저장된 `HF_TOKEN`을 통해 Hugging Face Space Git 리모트로 코드를 자동 동기화합니다.
4. Hugging Face Space가 Dockerfile을 감지하여 의존성 패키지와 가중치를 탑재한 Docker 이미지를 자동 빌드하고, 16GB RAM 클라우드 환경에서 서비스를 기동합니다.
5. 사용자에게는 무료 공인 도메인(`https://<username>-vidio-health-trainer.hf.space`)이 제공됩니다.

---

## 3. 필요 구성 파일 및 배포 명세

배포 자동화를 위해 프로젝트에 추가되는 4대 핵심 구성 파일입니다:

1. **`Dockerfile`**:
   - `python:3.10-slim` 기반 경량 컨테이너
   - OpenCV 동작을 위한 필수 라이브러리(`libgl1-mesa-glx`, `libglib2.0-0`, `ffmpeg`) 설치
   - Hugging Face Spaces의 기본 포트인 **7860** 바인딩 및 uvicorn 실행
2. **`requirements.txt`**:
   - `fastapi`, `uvicorn`, `mediapipe`, `rtmlib`, `onnxruntime`, `opencv-python-headless`, `scikit-learn`, `xgboost`, `lightgbm`, `pydantic` 등
3. **`.dockerignore`**:
   - 불필요한 대용량 임시 비디오(`data/일반화`), 캐시 파일, `.git` 등을 제외하여 Docker 빌드 시간 및 이미지 크기 80% 압축
4. **`.github/workflows/deploy.yml`**:
   - GitHub Actions CI/CD 파이프라인 명세 스크립트

---

## 4. 대안 플랜: Cloudflare Tunnel 기반 로컬 GPU 가속 무료 배포

만약 클라우드 CPU 환경(28초)보다 더 빠른 **12~15초대 실시간 분석 속도**를 서비스하고 싶다면, 사용자 PC의 외장 그래픽카드(**NVIDIA GeForce MX570 A**)를 웹에 무료로 개방하는 Cloudflare Tunnel 방식이 최적입니다:

- **공인 IP나 포트포워딩 불필요**: 로컬에서 `cloudflared tunnel run` 명령 하나로 안전한 암호화 터널 생성
- **무료 SSL/HTTPS 자동 제공**: `https://*.trycloudflare.com` 또는 무료 개인 도메인 연동
- **DDoS 방어 및 CDN 정적 에셋 가속**: 전 세계 어디서든 안전하고 빠른 접속 보장
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] MD Report generated: {MD_PATH}")

if __name__ == "__main__":
    build_docx()
    build_md()
