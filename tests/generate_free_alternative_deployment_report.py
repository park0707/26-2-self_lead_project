# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/generate_free_alternative_deployment_report.py
Purpose: 8주차 무과금 클라우드 배포 대안 및 Gradio SDK 우회 기획 보고서 생성 (.docx 및 .md)
"""

import os
import sys
from datetime import datetime
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_DIR = r"D:\자주프 문서\8주차"
MD_DIR = r"D:\자주프 문서\8주차\md"
os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

REPORT_NAME = "8주차_무과금_클라우드_배포_대안_및_Gradio_SDK_우회_기획보고서"
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

def add_callout(doc, text, title="핵심 요약 / EXECUTIVE SUMMARY"):
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
    r_t = p_title.add_run("vidio-health-trainer 무과금 클라우드 배포 대안 및 Gradio SDK 우회 기획서")
    r_t.bold = True
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(17)
    r_t.font.color.rgb = RGBColor(16, 44, 87)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    r_s = p_sub.add_run("Hugging Face Docker 요금제 장벽 해소: Gradio SDK 기반 16GB RAM 100% 무료 배포 및 Cloudflare GPU 터널 대안 비교")
    r_s.font.name = "맑은 고딕"
    r_s.font.size = Pt(10.5)
    r_s.font.color.rgb = RGBColor(100, 110, 120)
    
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("공식 프로젝트 명칭", "vidio-health-trainer"),
        ("기획 기준 일자 / 주차", f"{datetime.now().strftime('%Y년 %m월 %d일')} (8주차)"),
        ("문제 발생 현상", "Hugging Face Spaces의 Docker SDK가 $9/월 PRO 유료 요금제 또는 카드 등록을 요구함"),
        ("해결 및 무과금 대안", "1) Gradio SDK 우회(16GB RAM 100% 무료 유지) | 2) Cloudflare GPU 터널(로컬 외장 GPU 가속)")
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
        if "Gradio SDK 우회" in v or "Cloudflare" in v:
            r1.bold = True
            r1.font.color.rgb = RGBColor(0, 86, 150)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    add_callout(doc,
        "1. Docker SDK 유료화 원인 규명: Hugging Face는 최근 암호화폐 채굴 및 악용 방지를 위해 커스텀 Docker Space를 PRO 요금제($9/월)로 제한하였습니다.\n"
        "2. 대안 1 (Gradio SDK 우회 - 100% 무료 1위): Hugging Face의 Gradio SDK는 신용카드 없이 평생 무료이며, 동일하게 2 vCPU, 16GB RAM을 제공합니다. Gradio는 내부적으로 FastAPI 기반이므로, app.py를 통해 기존 FastAPI 웹 서비스와 HTML UI를 그대로 마운트하여 100% 무료 서빙이 가능합니다.\n"
        "3. 대안 2 (Cloudflare Tunnel - GPU 최고 성능 1위): 클라우드 무료 CPU(28초) 대신, 사용자 PC의 NVIDIA GeForce MX570 A GPU를 활용하여 12~15초대 실시간 분석을 무료 공인 HTTPS 도메인으로 전 세계에 서비스할 수 있습니다.\n"
        "4. 대안 3 (Render.com Free Tier): GitHub 원클릭 자동 배포를 지원하나, 512MB RAM 한계로 인해 비디오 프레임 버퍼 다이어트(360p 리사이즈)가 요구됩니다.",
        "무과금 배포 기획 핵심 요약 (Executive Summary)"
    )

    h1 = doc.add_heading("1. Hugging Face Docker SDK 유료화 분석 및 극복 방안", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "Hugging Face 플랫폼 정책에 따라, 커스텀 Dockerfile을 사용하는 'Docker Space'는 "
        "PRO 구독($9/월) 또는 해외 신용카드 인증을 요구하도록 정책이 강화되었습니다. "
        "그러나 머신러닝 커뮤니티를 위해 제공되는 'Gradio SDK Space'는 신용카드나 과금 없이 100% 무료로 2 vCPU, 16GB RAM 하드웨어를 계속 제공하고 있습니다.\n"
    )

    h2 = doc.add_heading("2. 100% 무과금 배포 3대 대안 상세 비교", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)
    
    tbl_alt = doc.add_table(rows=4, cols=5)
    tbl_alt.alignment = WD_TABLE_ALIGNMENT.CENTER
    alt_headers = ["배포 방식", "비용 및 카드 요구", "하드웨어 스펙", "자동 배포(CI/CD)", "장점 및 주의사항"]
    alt_rows = [
        ("Hugging Face Spaces\n(Gradio SDK 우회)", "100% 영구 무료\n(신용카드 불필요)", "2 vCPU / 16GB RAM\n50GB Disk", "GitHub Actions\nGit-Sync 자동화", "클라우드 16GB RAM 무료 활용,\n기존 FastAPI/HTML UI 100% 보존"),
        ("Cloudflare Tunnel\n+ 로컬 GPU 서버", "100% 영구 무료\n(신용카드 불필요)", "로컬 CPU + MX570 GPU\n16GB RAM / SSD", "GitHub Webhook\n또는 로컬 git-pull", "외장 GPU 가속으로 12~15초 실시간,\n공인 IP/포트포워딩 없이 무료 HTTPS"),
        ("Render.com\n(Web Service Free)", "100% 무료\n(신용카드 불필요)", "0.1 CPU / 512MB RAM\n(15분 후 슬립)", "GitHub 리포지토리\nPush 시 자동 배포", "설정이 가장 간단하나 512MB RAM으로\n영상 처리 중 OOM 발생 주의")
    ]
    for c_idx, h in enumerate(alt_headers):
        cell = tbl_alt.rows[0].cells[c_idx]
        set_cell_background(cell, "005696")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, r_data in enumerate(alt_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tbl_alt.rows[r_idx].cells[c_idx]
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 1 else "F8FAFC")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.5)
            if "100% 영구 무료" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(0, 100, 40)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    h3 = doc.add_heading("3. 대안 1: Gradio SDK를 통한 FastAPI 마운트 및 무료 배포 구현", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "Gradio 라이브러리는 내부가 100% FastAPI로 구현되어 있습니다. "
        "따라서 Hugging Face Space를 생성할 때 SDK를 'Docker' 대신 'Gradio'로 선택하면, "
        "결제 요구창 없이 즉시 무료 16GB RAM Space가 개설되며 다음과 같이 기존 서비스를 100% 서빙할 수 있습니다:\n"
    )
    
    p_code = doc.add_paragraph()
    p_code.paragraph_format.left_indent = Inches(0.2)
    p_code.add_run("■ 구축 완료된 3대 핵심 파일 명세:\n").bold = True
    p_code.add_run(
        "1. [app.py]: gr.mount_gradio_app(app, demo, path='/gradio')를 적용하여, 루트 URL(/)에서는 기존의 세련된 HTML5/JS 운동 분석 UI가 서빙되고 /gradio에서는 보조 화면이 제공되도록 구성.\n"
        "2. [packages.txt]: Debian 시스템 레벨 패키지(ffmpeg, libgl1, libglib2.0-0)를 Hugging Face 빌드 시점에 자동 설치하도록 명시.\n"
        "3. [requirements.txt]: gradio>=4.20.0 및 기존의 딥러닝/머신러닝 패키지를 등록하여 자동 설치."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h4 = doc.add_heading("4. 대안 2: Cloudflare Tunnel 기반 로컬 GPU 초고속 무료 배포", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.add_run(
        "만약 클라우드 CPU의 28초 분석보다 훨씬 빠른 실시간 12~15초 속도로 웹 서비스를 전 세계에 무료 배포하고 싶다면, "
        "사용자 PC의 NVIDIA GeForce MX570 A GPU를 가동하는 Cloudflare Tunnel이 최고의 대안입니다:\n"
    )
    
    p_cf = doc.add_paragraph()
    p_cf.paragraph_format.left_indent = Inches(0.2)
    p_cf.add_run("■ Cloudflare Tunnel의 무과금 특장점 및 원클릭 런처:\n").bold = True
    p_cf.add_run(
        "1. [비용 0원 / 신용카드 불필요]: Cloudflare의 글로벌 터널 네트워크를 영구 무료로 이용.\n"
        "2. [공인 IP 및 공유기 포트포워딩 불필요]: 방화벽이나 공유기 설정 없이 아웃바운드 보안 터널 체결.\n"
        "3. [무료 공인 HTTPS 도메인 즉시 발급]: https://*.trycloudflare.com 보안 URL 즉시 생성.\n"
        "4. [run_cloudflare_tunnel.bat 제공]: 더블 클릭 한 번으로 로컬 서버 기동 및 Cloudflare 터널 연결 자동화."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.save(DOCX_PATH)
    print(f"[+] DOCX Report generated: {DOCX_PATH}")

def build_md():
    md_content = f"""# vidio-health-trainer 무과금 클라우드 배포 대안 및 Gradio SDK 우회 기획서

**작성 일자:** {datetime.now().strftime('%Y-%m-%d')} (8주차)  
**공식 프로젝트 명칭:** `vidio-health-trainer`  
**핵심 문제:** Hugging Face Space의 Docker SDK가 $9/월 PRO 결제 또는 카드 등록을 요구함  
**해결 대안:** 1) Gradio SDK 우회(16GB RAM 100% 무료 유지) | 2) Cloudflare GPU 터널(로컬 외장 GPU 가속)

---

## 1. Docker SDK 유료화 원인 규명 및 극복 방안

Hugging Face는 최근 암호화폐 채굴 및 인프라 악용을 방지하기 위해 **커스텀 Dockerfile을 사용하는 Docker Space에 한해 PRO($9/월) 구독을 요구**하도록 정책을 변경하였습니다.

하지만, 머신러닝 오픈소스 커뮤니티를 위해 제공되는 **`Gradio SDK` Space는 신용카드나 결제 없이 100% 평생 무료로 동일하게 2 vCPU, 16GB RAM, 50GB 디스크를 제공**하고 있습니다.

---

## 2. 100% 무과금 배포 3대 대안 비교

| 배포 방식 | 비용 및 카드 요구 | 하드웨어 스펙 | GitHub 자동 배포 (CI/CD) | 장점 및 주의사항 |
| :--- | :---: | :---: | :---: | :--- |
| **Hugging Face Spaces (Gradio SDK 우회)** | **100% 영구 무료 (카드 불필요)** | **2 vCPU / 16GB RAM / 50GB Disk** | **GitHub Actions Git-Sync 연동** | **클라우드 16GB RAM 무료 활용, 기존 FastAPI 및 HTML UI 100% 유지** |
| **Cloudflare Tunnel + 로컬 GPU 서버** | **100% 영구 무료 (카드 불필요)** | **로컬 CPU + MX570 GPU / 16GB RAM** | **GitHub Webhook 또는 git pull** | **외장 GPU 가속으로 12~15초 실시간 분석, 포트포워딩 없이 무료 HTTPS 발급** |
| **Render.com (Web Service Free)** | **100% 무료 (카드 불필요)** | 0.1 CPU / 512MB RAM (15분 후 슬립) | GitHub Direct 연동 | 설정은 간편하나 512MB RAM 한계로 비디오 분석 시 OOM Kill 발생 주의 |

---

## 3. 대안 1 구현: Gradio SDK를 통한 FastAPI 마운트 및 무료 배포

Gradio 라이브러리는 내부 엔진이 **FastAPI**로 개발되어 있습니다.  
따라서 Space 생성 시 SDK를 `Gradio`로 선택하면, 결제창 없이 즉시 무료 16GB Space가 생성되며, `app.py`를 통해 기존 FastAPI 웹 서비스를 그대로 서빙할 수 있습니다.

```mermaid
flowchart LR
    Dev[개발자 로컬 PC] -->|git push origin main| GitHub[GitHub 원격 리포지토리]
    GitHub -->|GitHub Actions| HF[Hugging Face Gradio Space (16GB RAM 무료)]
    HF -->|packages.txt| Sys[ffmpeg / libgl1 시스템 패키지 자동 설치]
    HF -->|requirements.txt| Py[Python 의존성 자동 설치]
    HF -->|app.py| FastAPI[FastAPI 앱 마운트 및 포트 7860 서빙]
    FastAPI -->|루트 URL /| UI[기존 HTML5/JS 운동 분석 대시보드 서빙]
    FastAPI -->|/gradio| Companion[Gradio 보조 화면]
```

### 구축 완료 파일:
- **`app.py`**: `gr.mount_gradio_app(app, demo, path="/gradio")`로 기존 FastAPI 인스턴스를 마운트. 루트(`/`)에서는 오리지널 HTML UI 서빙.
- **`packages.txt`**: Debian `apt-get`으로 `ffmpeg`, `libgl1`, `libglib2.0-0` 자동 설치.
- **`requirements.txt`**: `gradio>=4.20.0` 추가.
- **`README.md`**: Hugging Face Space 메타데이터(sdk: gradio, app_file: app.py) 헤더 등록.

---

## 4. 대안 2 구현: Cloudflare Tunnel 기반 로컬 GPU 초고속 무료 배포

클라우드 무료 CPU(28초)보다 훨씬 빠른 **12~15초대 실시간 분석**을 원할 경우, 사용자 PC의 외장 그래픽카드(**NVIDIA GeForce MX570 A**)를 웹에 무료로 개방하는 Cloudflare Tunnel 방식이 제공됩니다:

- **원클릭 실행 스크립트**: `run_cloudflare_tunnel.bat` 더블 클릭만으로 로컬 서버 및 보안 터널 자동 실행
- **공인 IP/포트포워딩 0원**: 공유기 설정 없이 전 세계 접속 가능한 `https://*.trycloudflare.com` 보안 주소 발급
- **DDoS 방어 및 CDN 무상 제공**: Cloudflare 글로벌 엣지 망이 보안 및 정적 파일 캐싱 처리
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] MD Report generated: {MD_PATH}")

if __name__ == "__main__":
    build_docx()
    build_md()
