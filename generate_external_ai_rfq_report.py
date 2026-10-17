# -*- coding: utf-8 -*-
"""
vidio-health-trainer
8주차 Fitness-AQA 데이터셋 실측 분석 및 외부 AI 기술 자문 의뢰서(RFC) 보고서 생성기
Word (.docx) 및 Markdown (.md) 동시 생성
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX_PATH = r"D:\자주프 문서\8주차\8주차_외부AI_기술자문_위탁요청서_및_Fitness-AQA_실측보고서.docx"
MD_PATH = r"D:\자주프 문서\8주차\md\8주차_외부AI_기술자문_위탁요청서_및_Fitness-AQA_실측보고서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    run = h.runs[0]
    if level == 1:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 44, 87)
    elif level == 2:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(41, 75, 107)
    elif level == 3:
        run.font.size = Pt(10.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(70, 90, 120)
    return h

def add_callout(doc, text, title="핵심 요약 (Key Takeaway)", border_color="2B6CB0", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '24')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_color)
    tcBorders.append(left)
    for b_name in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{b_name}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"■ {title}\n")
    r_title.bold = True
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = RGBColor(16, 44, 87)
    
    r_txt = p.add_run(text)
    r_txt.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_cell(cell, text, bold=False, color=None, bg=None, align=WD_ALIGN_PARAGRAPH.LEFT, size=Pt(8.5)):
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.font.size = size
    run.font.name = 'Malgun Gothic'
    if color:
        run.font.color.rgb = color
    if bg:
        set_cell_background(cell, bg)
    set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

def generate_report():
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Malgun Gothic'
    normal_style.font.size = Pt(9.5)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(4)

    # Title Block
    title_p = doc.add_paragraph()
    t_run = title_p.add_run("vidio-health-trainer 기술 문서 및 외부 AI 자문 패키지")
    t_run.font.size = Pt(11)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(70, 90, 120)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(2)
    sub_p.paragraph_format.space_after = Pt(10)
    s_run = sub_p.add_run("Fitness-AQA 실측 검증 보고 및 외부 AI 기술 자문 의뢰서 (RFC)")
    s_run.font.size = Pt(17)
    s_run.font.bold = True
    s_run.font.color.rgb = RGBColor(16, 44, 87)

    # Metadata
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    m_run = meta_p.add_run("문서 번호: 8주차 공식 기술 자문 패키지 (2026-10-16)  |  프로젝트: vidio-health-trainer  |  배포 대상: 외부 AI (ChatGPT / Claude / Gemini / 아주LLM)")
    m_run.font.size = Pt(8.5)
    m_run.font.color.rgb = RGBColor(100, 116, 139)

    add_callout(doc, 
        "1. Fitness-AQA 다운로드 가용성 실측 결과: 1클릭 즉시 다운로드는 불가능하며, 저자(Paritosh Parmar)에게 비상업적 연구 라이선스 동의 폼을 제출하여 승인받는 '신청 승인제(Request-based Access)' 데이터셋임.\n"
        "2. Fitness-AQA 데이터 구조 실측: BackSquat(KIE: 무릎 안쪽 모임, KFE: 무릎 과도 전진, SS: 깊이 미달), BarbellRow(Lumbar: 요추 척추 굽힘, TorsoAngle: 상체각), OverheadPress(Elbow, Knees) 등 본 프로젝트 결함과 1:1 대응되는 최고 품질의 어노테이션 확인.\n"
        "3. 외부 AI 기술 자문 의뢰서(RFC) 수록: 우리 프로젝트의 파이프라인, AI-Hub 학습 환경, 스마트폰 세로 9:16 및 바닥 로우앵글(20~30°) 일반화 실패 원인, 필요한 데이터 스펙을 외부 거대언어모델(ChatGPT, Claude, 아주LLM 등)에 복사·붙여넣기하여 즉시 질의할 수 있는 고품질 표준 질의문 수록.",
        title="핵심 요약 (Executive Summary)")

    # Section 1
    add_header_styled(doc, "1. Fitness-AQA 데이터셋 실측 검증 결과", level=1)
    
    add_header_styled(doc, "1.1 즉시 다운로드 가능 여부 (가용성 실측)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "공식 GitHub 리포지토리(ParitoshParmar/Fitness-AQA) 및 Zenodo 아카이브(레코드 7310289), 저자 커밋 내역, 이슈 트래커를 전수 조사한 실측 결과는 다음과 같습니다:\n\n"
        "• 결론: '1클릭 즉시 다운로드'는 불가능하며, 저자에게 연구용 라이선스 동의 폼(Google Form / 이메일)을 제출하고 다운로드 링크를 승인받아야 하는 '신청 승인제(Request-based Access)' 데이터셋입니다.\n"
        "• 현재 상태: 공식 GitHub 리포지토리는 코드 공개 작업 중(_Work in Progress..._) 상태이며, Zenodo에는 데이터셋 구조도(Hierarchy PNG)와 라이선스 PDF(CC BY-NC 4.0)만 업로드되어 있습니다.\n"
        "• GitHub 이슈 현황: 전 세계 연구자들이 저자에게 '라이선스 신청서를 제출했으나 다운로드 링크 이메일을 기다리고 있다'는 이슈(Issue #22, #23 등)를 등록하고 있는 상태입니다.\n"
        "• 권고: 저자에게 연구용 신청 폼을 제출해 두고 승인을 기다리는 한편, 즉시 다운로드가 가능한 대안 데이터셋(RepCount, UI-PRMD, Fit3D) 및 무촬영 시뮬레이션을 병행하는 것이 가장 현실적입니다."
    )

    add_header_styled(doc, "1.2 데이터 형태 및 세부 어노테이션 구조 실측", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "공식 저장소에서 직접 추출한 계층 다이어그램(exercise_dataset_hierarchy_3.png) 및 데이터로더(dataloader.py) 코드를 분석한 세부 형태는 다음과 같습니다:"
    )

    # Insert hierarchy diagram image if exists
    img_path = r"d:\fitness_aqa_hierarchy.png"
    if os.path.exists(img_path):
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(img_path, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("[그림 1] Fitness-AQA 데이터셋 공식 운동 종목 및 결함(Error) 계층 구조도")
        r_cap.font.size = Pt(8)
        r_cap.font.color.rgb = RGBColor(100, 116, 139)

    # Table of Errors
    headers_fa = ["운동 종목", "결함 코드", "결함 명칭 및 내용", "정상 vs 결함 샘플 수", "본 프로젝트(vidio-health-trainer) 대응 결함"]
    tbl_fa = doc.add_table(rows=8, cols=5)
    tbl_fa.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_fa = [Inches(1.2), Inches(0.9), Inches(1.8), Inches(1.2), Inches(1.4)]

    for j, h_text in enumerate(headers_fa):
        cell = tbl_fa.cell(0, j)
        cell.width = col_w_fa[j]
        format_cell(cell, h_text, bold=True, color=RGBColor(255, 255, 255), bg="102C57", align=WD_ALIGN_PARAGRAPH.CENTER, size=Pt(8))

    fa_rows = [
        ("BackSquat\n(스쿼트)", "KIE", "Knee Inward Error\n(무릎 안쪽 모임, Knee Valgus)", "1,387 : 236", "스쿼트 c3 (무릎 정렬 위반)\n[100% 완벽 일치]"),
        ("BackSquat\n(스쿼트)", "KFE", "Knee Forward Error\n(무릎 발끝 과도 전진)", "515 : 1,108", "스쿼트 무릎 전방 과도 돌출"),
        ("BackSquat\n(스쿼트)", "SS", "Shallow Squat\n(스쿼트 깊이 미달)", "1,543 : 2,068", "스쿼트 유효 가동범위 미달"),
        ("BarbellRow\n(바벨로우)", "Lumbar", "Lumbar Spine Flexion\n(허리/요추 척추 말림)", "5,408 : 1,082", "데드리프트 d1 / 스쿼트 c1\n(척추 중립 위반) [100% 일치]"),
        ("BarbellRow\n(바벨로우)", "TorsoAngle", "Improper Torso Angle\n(상체 각도 과도 직립/숙임)", "5,900 : 590", "상체 체간 각도 이탈"),
        ("OverheadPress", "Elbow", "Elbow Path Deviation\n(팔꿈치 궤적 이탈)", "1,686 : 574", "상체 궤적 오류"),
        ("OverheadPress", "Knees", "Knee Bend (반동 치팅)", "1,486 : 774", "하체 반동 오류")
    ]

    for i, row in enumerate(fa_rows):
        for j, val in enumerate(row):
            cell = tbl_fa.cell(i+1, j)
            cell.width = col_w_fa[j]
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            align = WD_ALIGN_PARAGRAPH.CENTER if j in [0, 1, 3] else WD_ALIGN_PARAGRAPH.LEFT
            format_cell(cell, val, bold=(j in [0, 1]), color=RGBColor(16, 44, 87) if j in [0, 1] else RGBColor(30, 41, 59), bg=bg, align=align, size=Pt(8))

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Section 2
    add_header_styled(doc, "2. 외부 AI 기술 자문 의뢰서 (RFC: Request For Comments)", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "아래 박스의 내용은 ChatGPT o1/Plus, Claude 3.5 Sonnet, Gemini Pro, 아주LLM 등의 외부 최상위 AI 모델에 "
        "그대로 복사하여 붙여넣을 수 있도록 정밀 설계된 공식 기술 자문 의뢰서(RFC) 전문입니다. "
        "우리 프로젝트의 배경, 아키텍처, 발생한 도메인 시프트 문제, 그리고 자문받고자 하는 데이터 스펙이 기술적으로 완벽히 정리되어 있습니다."
    )

    rfc_text = """[기술 자문 의뢰서: 영상 기반 피트니스 자세 평가 AI의 스마트폰 실환경 일반화(Domain Generalization) 극복을 위한 공개 데이터셋 및 엔지니어링 대안 질의]

안녕하세요. 저희는 컴퓨터 비전 기반의 인공지능 웨이트 트레이닝 자세 교정 및 결함 판별 시스템인 「vidio-health-trainer」 프로젝트를 개발 중인 공학 연구팀입니다.

저희가 현재 마주한 중대한 성능 병목(Domain Shift) 현상을 해결하기 위해, 귀하의 깊이 있는 컴퓨터 비전, 생체역학(Biomechanics), 도메인 적응(Domain Adaptation) 전문 지식을 기반으로 한 기술 자문을 요청드립니다.

────────────────────────────────────────────────────────
1. 프로젝트 개요 및 현재 파이프라인
────────────────────────────────────────────────────────
• 타깃 운동 종목: 
  - 바벨 스쿼트(Barbell Squat)
  - 바벨 데드리프트(Barbell Deadlift) 및 스티프 데드리프트
  - 바벨 런지(Barbell Lunge) 및 사이드/다이나믹 런지 등 9종 웨이트 동작
• 데이터 입력 및 특징 추출 파이프라인:
  - 단안 RGB 비디오 입력 -> MediaPipe Pose 3D 랜드마크(33개 관절 좌표) 추출
  - 1회 반복(1 Rep) 자동 분할 -> 64프레임 3차 스플라인(Cubic Spline) 보간
  - 생체역학 기반 도메인 피처 추출: 관절 각도(척추 체간각, 무릎-발목 각, 골반각), 가속도 저크(Jerk), 분절 길이 비율, 통계 피처(Std, Max, Min)
  - 모델 아키텍처: 세부 결함 부위별 독립 이진 분류기(Submodels: c1~c4, d1~d5)
    * 스쿼트 결함: c1(척추 중립 위반), c2(시선 불안정), c3(무릎 안쪽 모임/외반), c4(발바닥 지면 고정 위반)
    * 데드리프트 결함: d1(척추 중립 위반), d3(발바닥 고정/힙슛), d4(무릎-골반 동시 신전 위반), d5(바벨 수직 궤적 이탈)

────────────────────────────────────────────────────────
2. 학습 환경 및 현재 성능 상태
────────────────────────────────────────────────────────
• 학습 데이터셋: AI-Hub 피트니스 자세 이미지 원천 데이터 (크로마키 스튜디오)
  - 피험자: 20~30대 성인 9명 전수
  - 카메라: 5대 동기화 카메라 (시점 A~E: 정면, 좌우 45도, 좌우 후방 45도)
  - 촬영 환경: 1920x1080 (16:9 가로 화면), 삼각대를 사람 허리/가슴 높이(1.0~1.2m)에 완벽한 수평(Pitch 0도)으로 거치
• 학습 결과: AI-Hub 내부 피험자 격리 교차검증(StratifiedGroupKFold) 기준 F1-Score 0.85 ~ 0.92 달성

────────────────────────────────────────────────────────
3. 당면 과제: 실서비스 스마트폰 촬영 시 일반화 성능 급락 원인
────────────────────────────────────────────────────────
실제 일반 헬스장에서 일반인이 스마트폰으로 촬영한 72개 세트 클립(총 256회 1회 반복)에 대해 추론을 수행한 결과, 정상 동작임에도 결함으로 오탐(False Positive)되는 비율이 급증하였습니다. 정밀 프로파일링 결과 4대 근본 원인을 규명하였습니다:

① 카메라 종횡비(Aspect Ratio) 3.16배 불일치에 따른 기하 각도 왜곡:
   - 학습: 16:9 가로 (1.78) vs 실제: 9:16 세로 (0.56) -> 종횡비 차이 3.16배
   - MediaPipe 3D 좌표가 세로 화면에서 가로축으로 약 15.8% 비등방 압축(Anisotropic Compression)되어 무릎 각도, 체간각이 왜곡됨.
② 바닥 거치 로우 앵글(Pitch 20~30도 상향 틸트)에 의한 2D 투영 착시:
   - 실제 사용자는 스마트폰을 바닥이나 낮은 탁자에 두고 위를 올려다보며 촬영.
   - 무릎이 굽혀질 때 원근 투영(Perspective Projection) 상 발뒤꿈치가 발가락보다 위로 올라가 보여, 발바닥을 바닥에 완벽히 붙였음에도 c4(발뒤꿈치 들림) 결함 확률이 85~90%로 치솟음.
③ 생체역학적 동반 결함(Co-occurrence)과 단일 의도 라벨 간의 괴리:
   - 전신 다관절 운동 특성상 척추를 숙이면 시선과 바벨이 자연스럽게 연쇄 이동함.
④ 개발자 1인의 수작업 촬영의 한계:
   - 다양한 체형(BMI), 성별, 옷차림, 헬스장 조명, 5대 각도를 1명이 직접 찍고 라벨링하는 것은 물리적으로 불가능하며 표본 편향이 심함.

────────────────────────────────────────────────────────
4. 귀하(외부 AI)에게 요청드리는 핵심 자문 항목
────────────────────────────────────────────────────────
저희가 직접 영상을 찍지 않고도, 스마트폰 세로 9:16 및 바닥 로우앵글 환경에 대해 모델을 강력하게 일반화시킬 수 있는 방안에 대해 아래 4가지 항목을 구체적으로 답변해 주십시오:

[질문 1] 즉시 다운로드 가능한 공개 피트니스 벤치마크 데이터셋 추천
- 승인 대기나 복잡한 절차 없이, 학술 연구용으로 즉시 다운로드(Direct Download, Kaggle, HuggingFace 등)하여 사용할 수 있는 스쿼트/데드리프트/런지 비디오 또는 2D/3D 키포인트 데이터셋이 존재하는가?
- 해당 데이터셋의 구체적인 URL, 데이터 구조(비디오 mp4, 랜드마크 json/csv), 어노테이션(정상 vs 결함 라벨 여부, 1회 반복 타임스탬프)을 알려주십시오.

[질문 2] 보유 중인 3D 좌표 기반의 가상 카메라 시뮬레이션 (Sim2Real) 구현 방안
- 저희는 이미 AI-Hub의 정확한 3D 월드 관절 좌표(x, y, z)를 보유하고 있습니다.
- 이를 활용하여 스마트폰의 '바닥 거치 로우앵글(Pitch +20~30도)'과 '세로 9:16 종횡비'를 수학적 카메라 투영 행렬 P = K[R|t]로 시뮬레이션하여 2D 키포인트를 대량 합성하고자 합니다.
- 이때 원근 왜곡을 물리적으로 정밀하게 재현할 수 있는 가상 카메라 내부 파라미터(K) 및 외부 파라미터(R, t)의 수학적 설정 권고안과 Python 예제 코드를 제안해 주십시오.

[질문 3] 라벨 없는 유튜브 숏츠(Shorts) 9:16 영상의 비지도 도메인 적응 (UDA) 방법론
- 유튜브에서 `#스쿼트`, `#squatcheck` 등의 세로 9:16 영상을 대량 수집할 수 있습니다.
- 정답 결함 라벨이 전혀 없는 타깃 도메인 영상에 대해, AI-Hub 학습 모델의 특징 공간을 적응시킬 수 있는 가장 실용적인 UDA 기법(예: DANN, AdaBN, Pseudo-labeling, Contrastive Learning 등)은 무엇이며 구체적인 학습 손실 함수 설계는 어떻게 해야 하는가?

[질문 4] 전처리단에서의 제로-데이터 기하학적 불변 정규화 (Zero-Data Invariance)
- 새로운 데이터를 학습시키지 않고, 오직 입력단 좌표 변환만으로 카메라 시점(Pitch 각도, 로우앵글) 및 화면비 왜곡을 무력화할 수 있는 생체역학적 좌표계(예: ISB Grood & Suntay JCS, 바닥면 추정 기반 Virtual Leveling 등) 적용 아이디어가 있는가?

위 4가지 질문에 대해 학술 논문 레퍼런스와 함께 실무적으로 구현 가능한 수준의 구체적 답변을 부탁드립니다."""

    add_callout(doc, rfc_text, title="외부 AI 질의용 공식 텍스트 (RFC Template)", border_color="102C57", bg_color="F8FAFC")

    # Section 3
    add_header_styled(doc, "3. 향후 추진 계획 및 실무 가이드", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "1. 외부 AI 질의 수행: 상기 2절의 공식 질의 텍스트를 ChatGPT o1, Claude 3.5 Sonnet, 아주LLM 등의 인터페이스에 입력하여 고도화된 기술 자문 및 추가 데이터셋 후보를 수집합니다.\n"
        "2. Fitness-AQA 라이선스 폼 제출: Paritosh Parmar 저자에게 비상업적 학술 연구 목적임을 명시하여 정식 데이터 접근 신청서를 접수해 둡니다.\n"
        "3. 3D 가상 카메라 투영기 구현: 외부 자문 결과 및 기존 3D 좌표를 결합하여, 바닥 거치 로우앵글 가상 투영 스크립트를 작성하여 AI-Hub 모델에 스마트폰 시점 내성을 부여합니다."
    )

    os.makedirs(os.path.dirname(DOCX_PATH), exist_ok=True)
    doc.save(DOCX_PATH)
    print(f"[SUCCESS] Word RFC report created: {DOCX_PATH}")

def generate_markdown():
    md_content = """# vidio-health-trainer 기술 문서 및 외부 AI 자문 패키지
## Fitness-AQA 실측 검증 보고 및 외부 AI 기술 자문 의뢰서 (RFC)

- **문서 번호**: 8주차 공식 기술 자문 패키지 (2026-10-16)
- **프로젝트 명**: `vidio-health-trainer`
- **배포 대상**: 외부 AI (ChatGPT / Claude / Gemini / 아주LLM 등)
- **Word 배포 문서**: `D:\\자주프 문서\\8주차\\8주차_외부AI_기술자문_위탁요청서_및_Fitness-AQA_실측보고서.docx`

---

> [!IMPORTANT]
> **핵심 요약 (Executive Summary)**
> 1. **Fitness-AQA 다운로드 가용성 실측 결과**: 1클릭 즉시 다운로드는 불가능하며, 저자(Paritosh Parmar)에게 비상업적 연구 라이선스 동의 폼을 제출하여 승인받는 '신청 승인제(Request-based Access)' 데이터셋임.
> 2. **Fitness-AQA 데이터 구조 실측**: BackSquat(KIE: 무릎 안쪽 모임, KFE: 무릎 과도 전진, SS: 깊이 미달), BarbellRow(Lumbar: 요추 척추 굽힘, TorsoAngle: 상체각), OverheadPress(Elbow, Knees) 등 본 프로젝트 결함과 1:1 대응되는 최고 품질의 어노테이션 확인.
> 3. **외부 AI 기술 자문 의뢰서(RFC) 수록**: 우리 프로젝트의 파이프라인, AI-Hub 학습 환경, 스마트폰 세로 9:16 및 바닥 로우앵글(20~30°) 일반화 실패 원인, 필요한 데이터 스펙을 외부 거대언어모델(ChatGPT, Claude, 아주LLM 등)에 복사·붙여넣기하여 즉시 질의할 수 있는 고품질 표준 질의문 수록.

---

## 1. Fitness-AQA 데이터셋 실측 검증 결과

### 1.1 즉시 다운로드 가능 여부 (가용성 실측)
공식 GitHub 리포지토리(`ParitoshParmar/Fitness-AQA`) 및 Zenodo 아카이브(레코드 7310289), 저자 커밋 내역, 이슈 트래커를 전수 조사한 실측 결과는 다음과 같습니다:

- **결론: '1클릭 즉시 다운로드'는 불가능하며, 저자에게 연구용 라이선스 동의 폼(Google Form / 이메일)을 제출하고 다운로드 링크를 승인받아야 하는 '신청 승인제(Request-based Access)' 데이터셋입니다.**
- **현재 상태**: 공식 GitHub 리포지토리는 코드 공개 작업 중(`_Work in Progress..._`) 상태이며, Zenodo에는 데이터셋 구조도(Hierarchy PNG)와 라이선스 PDF(CC BY-NC 4.0)만 업로드되어 있습니다.
- **GitHub 이슈 현황**: 전 세계 연구자들이 저자에게 "라이선스 신청서를 제출했으나 다운로드 링크 이메일을 기다리고 있다"는 이슈(Issue #22, #23 등)를 등록하고 있는 상태입니다.
- **권고**: 저자에게 연구용 신청 폼을 제출해 두고 승인을 기다리는 한편, 즉시 다운로드가 가능한 대안 데이터셋(RepCount, UI-PRMD, Fit3D) 및 무촬영 시뮬레이션을 병행하는 것이 가장 현실적입니다.

---

### 1.2 데이터 형태 및 세부 어노테이션 구조 실측

공식 저장소에서 직접 추출한 계층 다이어그램(`exercise_dataset_hierarchy_3.png`) 및 데이터로더(`dataloader.py`) 코드를 분석한 세부 형태는 다음과 같습니다:

![Fitness-AQA 계층 구조도](file:///d:/fitness_aqa_hierarchy.png)

| 운동 종목 | 결함 코드 | 결함 명칭 및 내용 | 정상 vs 결함 샘플 수 | 본 프로젝트(`vidio-health-trainer`) 대응 결함 |
| :--- | :---: | :--- | :---: | :--- |
| **BackSquat**<br>(스쿼트) | **KIE** | **Knee Inward Error**<br>(무릎 안쪽 모임, Knee Valgus) | 1,387 : 236 | **스쿼트 $c_3$ (무릎 정렬 위반)<br>[100% 완벽 일치]** |
| **BackSquat**<br>(스쿼트) | **KFE** | **Knee Forward Error**<br>(무릎 발끝 과도 전진) | 515 : 1,108 | 스쿼트 무릎 전방 과도 돌출 |
| **BackSquat**<br>(스쿼트) | **SS** | **Shallow Squat**<br>(스쿼트 깊이 미달) | 1,543 : 2,068 | 스쿼트 유효 가동범위 미달 |
| **BarbellRow**<br>(바벨로우) | **Lumbar** | **Lumbar Spine Flexion**<br>(허리/요추 척추 말림) | 5,408 : 1,082 | **데드리프트 $d_1$ / 스쿼트 $c_1$<br>(척추 중립 위반) [100% 일치]** |
| **BarbellRow**<br>(바벨로우) | **TorsoAngle** | **Improper Torso Angle**<br>(상체 각도 과도 직립/숙임) | 5,900 : 590 | 상체 체간 각도 이탈 |
| **OverheadPress** | **Elbow** | **Elbow Path Deviation**<br>(팔꿈치 궤적 이탈) | 1,686 : 574 | 상체 궤적 오류 |
| **OverheadPress** | **Knees** | **Knee Bend** (반동 치팅) | 1,486 : 774 | 하체 반동 오류 |

---

## 2. 외부 AI 기술 자문 의뢰서 (RFC: Request For Comments) 전문

> [!TIP]
> **외부 AI 질의 방법**: 아래 코드 블록 전체를 복사하여 ChatGPT o1/Plus, Claude 3.5 Sonnet, 아주LLM 등의 프롬프트 창에 그대로 입력하시면 됩니다.

```text
[기술 자문 의뢰서: 영상 기반 피트니스 자세 평가 AI의 스마트폰 실환경 일반화(Domain Generalization) 극복을 위한 공개 데이터셋 및 엔지니어링 대안 질의]

안녕하세요. 저희는 컴퓨터 비전 기반의 인공지능 웨이트 트레이닝 자세 교정 및 결함 판별 시스템인 「vidio-health-trainer」 프로젝트를 개발 중인 공학 연구팀입니다.

저희가 현재 마주한 중대한 성능 병목(Domain Shift) 현상을 해결하기 위해, 귀하의 깊이 있는 컴퓨터 비전, 생체역학(Biomechanics), 도메인 적응(Domain Adaptation) 전문 지식을 기반으로 한 기술 자문을 요청드립니다.

────────────────────────────────────────────────────────
1. 프로젝트 개요 및 현재 파이프라인
────────────────────────────────────────────────────────
• 타깃 운동 종목: 
  - 바벨 스쿼트(Barbell Squat)
  - 바벨 데드리프트(Barbell Deadlift) 및 스티프 데드리프트
  - 바벨 런지(Barbell Lunge) 및 사이드/다이나믹 런지 등 9종 웨이트 동작
• 데이터 입력 및 특징 추출 파이프라인:
  - 단안 RGB 비디오 입력 -> MediaPipe Pose 3D 랜드마크(33개 관절 좌표) 추출
  - 1회 반복(1 Rep) 자동 분할 -> 64프레임 3차 스플라인(Cubic Spline) 보간
  - 생체역학 기반 도메인 피처 추출: 관절 각도(척추 체간각, 무릎-발목 각, 골반각), 가속도 저크(Jerk), 분절 길이 비율, 통계 피처(Std, Max, Min)
  - 모델 아키텍처: 세부 결함 부위별 독립 이진 분류기(Submodels: c1~c4, d1~d5)
    * 스쿼트 결함: c1(척추 중립 위반), c2(시선 불안정), c3(무릎 안쪽 모임/외반), c4(발바닥 지면 고정 위반)
    * 데드리프트 결함: d1(척추 중립 위반), d3(발바닥 고정/힙슛), d4(무릎-골반 동시 신전 위반), d5(바벨 수직 궤적 이탈)

────────────────────────────────────────────────────────
2. 학습 환경 및 현재 성능 상태
────────────────────────────────────────────────────────
• 학습 데이터셋: AI-Hub 피트니스 자세 이미지 원천 데이터 (크로마키 스튜디오)
  - 피험자: 20~30대 성인 9명 전수
  - 카메라: 5대 동기화 카메라 (시점 A~E: 정면, 좌우 45도, 좌우 후방 45도)
  - 촬영 환경: 1920x1080 (16:9 가로 화면), 삼각대를 사람 허리/가슴 높이(1.0~1.2m)에 완벽한 수평(Pitch 0도)으로 거치
• 학습 결과: AI-Hub 내부 피험자 격리 교차검증(StratifiedGroupKFold) 기준 F1-Score 0.85 ~ 0.92 달성

────────────────────────────────────────────────────────
3. 당면 과제: 실서비스 스마트폰 촬영 시 일반화 성능 급락 원인
────────────────────────────────────────────────────────
실제 일반 헬스장에서 일반인이 스마트폰으로 촬영한 72개 세트 클립(총 256회 1회 반복)에 대해 추론을 수행한 결과, 정상 동작임에도 결함으로 오탐(False Positive)되는 비율이 급증하였습니다. 정밀 프로파일링 결과 4대 근본 원인을 규명하였습니다:

① 카메라 종횡비(Aspect Ratio) 3.16배 불일치에 따른 기하 각도 왜곡:
   - 학습: 16:9 가로 (1.78) vs 실제: 9:16 세로 (0.56) -> 종횡비 차이 3.16배
   - MediaPipe 3D 좌표가 세로 화면에서 가로축으로 약 15.8% 비등방 압축(Anisotropic Compression)되어 무릎 각도, 체간각이 왜곡됨.
② 바닥 거치 로우 앵글(Pitch 20~30도 상향 틸트)에 의한 2D 투영 착시:
   - 실제 사용자는 스마트폰을 바닥이나 낮은 탁자에 두고 위를 올려다보며 촬영.
   - 무릎이 굽혀질 때 원근 투영(Perspective Projection) 상 발뒤꿈치가 발가락보다 위로 올라가 보여, 발바닥을 바닥에 완벽히 붙였음에도 c4(발뒤꿈치 들림) 결함 확률이 85~90%로 치솟음.
③ 생체역학적 동반 결함(Co-occurrence)과 단일 의도 라벨 간의 괴리:
   - 전신 다관절 운동 특성상 척추를 숙이면 시선과 바벨이 자연스럽게 연쇄 이동함.
④ 개발자 1인의 수작업 촬영의 한계:
   - 다양한 체형(BMI), 성별, 옷차림, 헬스장 조명, 5대 각도를 1명이 직접 찍고 라벨링하는 것은 물리적으로 불가능하며 표본 편향이 심함.

────────────────────────────────────────────────────────
4. 귀하(외부 AI)에게 요청드리는 핵심 자문 항목
────────────────────────────────────────────────────────
저희가 직접 영상을 찍지 않고도, 스마트폰 세로 9:16 및 바닥 로우앵글 환경에 대해 모델을 강력하게 일반화시킬 수 있는 방안에 대해 아래 4가지 항목을 구체적으로 답변해 주십시오:

[질문 1] 즉시 다운로드 가능한 공개 피트니스 벤치마크 데이터셋 추천
- 승인 대기나 복잡한 절차 없이, 학술 연구용으로 즉시 다운로드(Direct Download, Kaggle, HuggingFace 등)하여 사용할 수 있는 스쿼트/데드리프트/런지 비디오 또는 2D/3D 키포인트 데이터셋이 존재하는가?
- 해당 데이터셋의 구체적인 URL, 데이터 구조(비디오 mp4, 랜드마크 json/csv), 어노테이션(정상 vs 결함 라벨 여부, 1회 반복 타임스탬프)을 알려주십시오.

[질문 2] 보유 중인 3D 좌표 기반의 가상 카메라 시뮬레이션 (Sim2Real) 구현 방안
- 저희는 이미 AI-Hub의 정확한 3D 월드 관절 좌표(x, y, z)를 보유하고 있습니다.
- 이를 활용하여 스마트폰의 '바닥 거치 로우앵글(Pitch +20~30도)'과 '세로 9:16 종횡비'를 수학적 카메라 투영 행렬 P = K[R|t]로 시뮬레이션하여 2D 키포인트를 대량 합성하고자 합니다.
- 이때 원근 왜곡을 물리적으로 정밀하게 재현할 수 있는 가상 카메라 내부 파라미터(K) 및 외부 파라미터(R, t)의 수학적 설정 권고안과 Python 예제 코드를 제안해 주십시오.

[질문 3] 라벨 없는 유튜브 숏츠(Shorts) 9:16 영상의 비지도 도메인 적응 (UDA) 방법론
- 유튜브에서 `#스쿼트`, `#squatcheck` 등의 세로 9:16 영상을 대량 수집할 수 있습니다.
- 정답 결함 라벨이 전혀 없는 타깃 도메인 영상에 대해, AI-Hub 학습 모델의 특징 공간을 적응시킬 수 있는 가장 실용적인 UDA 기법(예: DANN, AdaBN, Pseudo-labeling, Contrastive Learning 등)은 무엇이며 구체적인 학습 손실 함수 설계는 어떻게 해야 하는가?

[질문 4] 전처리단에서의 제로-데이터 기하학적 불변 정규화 (Zero-Data Invariance)
- 새로운 데이터를 학습시키지 않고, 오직 입력단 좌표 변환만으로 카메라 시점(Pitch 각도, 로우앵글) 및 화면비 왜곡을 무력화할 수 있는 생체역학적 좌표계(예: ISB Grood & Suntay JCS, 바닥면 추정 기반 Virtual Leveling 등) 적용 아이디어가 있는가?

위 4가지 질문에 대해 학술 논문 레퍼런스와 함께 실무적으로 구현 가능한 수준의 구체적 답변을 부탁드립니다.
```

---

## 3. 향후 추진 계획

1. **외부 AI 질의 수행**: 상기 2절의 공식 질의 텍스트를 복사하여 ChatGPT o1, Claude 3.5 Sonnet, 아주LLM 등에 질의하여 다각도의 기술 자문 결과를 확보합니다.
2. **Fitness-AQA 라이선스 폼 제출**: 저자(Paritosh Parmar)에게 비상업적 연구 목적 승인 신청서를 발송합니다.
3. **가상 카메라 투영기 구현**: AI-Hub 3D 좌표를 기반으로 바닥 로우앵글 투영 파이썬 스크립트를 구현하여 무촬영 데이터 확장을 달성합니다.
"""
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Markdown RFC report created: {MD_PATH}")

if __name__ == "__main__":
    generate_report()
    generate_markdown()
