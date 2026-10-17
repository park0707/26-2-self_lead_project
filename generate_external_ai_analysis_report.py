# -*- coding: utf-8 -*-
"""
vidio-health-trainer
8주차 외부 AI 2종 기술 자문 결과 심층 종합 분석 및 팩트체크 보고서 생성기
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

DOCX_PATH = r"D:\자주프 문서\8주차\8주차_외부AI_2종_조사결과_교차검증_및_핵심통찰_심층분석보고서.docx"
MD_PATH = r"D:\자주프 문서\8주차\md\8주차_외부AI_2종_조사결과_교차검증_및_핵심통찰_심층분석보고서.md"

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
    t_run = title_p.add_run("vidio-health-trainer 심층 기술 평가 보고서")
    t_run.font.size = Pt(11)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(70, 90, 120)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(2)
    sub_p.paragraph_format.space_after = Pt(10)
    s_run = sub_p.add_run("외부 AI 2종 기술 자문 결과 심층 종합 분석 및 엔지니어링 팩트체크 보고서")
    s_run.font.size = Pt(17)
    s_run.font.bold = True
    s_run.font.color.rgb = RGBColor(16, 44, 87)

    # Metadata
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    m_run = meta_p.add_run("문서 번호: 8주차 공식 기술 감사 보고서 (2026-10-17)  |  프로젝트: vidio-health-trainer  |  분석 대상: 외부 AI 2종(이론 모델 & 코드 실사 모델) 응답 전문")
    m_run.font.size = Pt(8.5)
    m_run.font.color.rgb = RGBColor(100, 116, 139)

    add_callout(doc, 
        "1. 외부 AI 2종의 결정적 합의: 두 AI 모두 '개발자가 직접 헬스장에서 영상을 찍는 방식은 비효율적이며, 가상 카메라 Sim2Real 투영과 도메인 적응(UDA)으로 해결 가능하다'는 점에 100% 일치함.\n"
        "2. AI 1(이론 모델)의 기여: 바닥 로우앵글(Pitch 20~30°)에서 발뒤꿈치(Heel)가 발끝(Toe)보다 광축상 멀어져 화면 상단에 맺히는 '원근 투영 역전(Elevation Illusion)'을 완벽히 수식화하고, 락아웃(Lockout) 차분 피치 및 ISB JCS 기하 변환 제안.\n"
        "3. AI 2(실사 모델)의 날카로운 팩트체크: ① MediaPipe 2D 정규화 좌표와 3D 월드 좌표의 혼용 엄금(월드 좌표에 15.8% 임의 곱셈 금지), ② 3D 강체 회전 불변성의 함정(3D 좌표만 회전하면 각도는 불변이므로 반드시 2D 투영 파이프라인을 거쳐야 함), ③ 64프레임 보간 시 물리적 Jerk(1/T^3) 왜곡 주의, ④ 결함별 관측 가능성 게이트 및 '정상/결함/판정불가' 3-State 도입 촉구.\n"
        "4. 데이터셋 실사 확인: Fitness-AQA·Fit3D는 즉시 다운로드가 불가하지만, Kaggle 'AttentiveSkel-3D'(64, 33, 3 텐서 487개)와 'RepCount'(유튜브 1,041개 영상)는 즉시 연동 가능한 실용적 대안으로 발굴됨.",
        title="핵심 요약 (Executive Summary)")

    # Section 1
    add_header_styled(doc, "1. 외부 AI 2종 자문 결과 총평 및 상호보완적 구도", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "사용자께서 의뢰하신 외부 최상위 AI 자문 결과는 매우 수준 높은 두 가지 관점(이론적 수학 모델링 vs 현실적 엔지니어링 팩트체크)을 제공하여, "
        "우리 프로젝트 vidio-health-trainer가 가야 할 명확한 기술 로드맵을 완성해 주었습니다.\n\n"
        "• 외부 AI 1 (이론/수학 모델 - Google Docs 10페이지): 원근 사영 왜곡 메커니즘을 수학적으로 완벽히 규명하고, 가상 카메라 파라미터(K, R, t), CDAN+InfoNCE 복합 손실함수, ISB Grood & Suntay 좌표계, 락아웃 기준 차분 피치 등 정교한 학술적 프레임워크를 수립했습니다.\n"
        "• 외부 AI 2 (실사/엔지니어링 모델 - 코드 검증 16페이지): 웹 검색 22회, 18개 페이지 정밀 분석, 코드 4회 실행을 거쳐 공개 데이터셋의 실제 라이선스/다운로드 가능 여부를 팩트체크하고, MediaPipe 좌표계의 치명적 혼용 위험, 3D 강체 불변성 함정, 64프레임 보간 시의 Jerk 스케일링 오류 등 실무 구현상의 맹점을 날카롭게 지적했습니다."
    )

    # Table: Comparison of 2 AIs
    headers_comp = ["비교 항목", "외부 AI 1 (이론/수학 중심)", "외부 AI 2 (실사/엔지니어링 중심)", "vidio-health-trainer 종합 채택"]
    tbl_comp = doc.add_table(rows=6, cols=4)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_comp = [Inches(1.2), Inches(2.2), Inches(2.2), Inches(1.6)]

    for j, h_text in enumerate(headers_comp):
        cell = tbl_comp.cell(0, j)
        cell.width = col_w_comp[j]
        format_cell(cell, h_text, bold=True, color=RGBColor(255, 255, 255), bg="102C57", align=WD_ALIGN_PARAGRAPH.CENTER, size=Pt(8))

    comp_rows = [
        ("공개 데이터셋\n가용성 평가", "Fitness-AQA, UI-PRMD, Fit3D 등 추천\n(학술적 가치 중심 기술)", "Fitness-AQA, Fit3D 즉시 다운로드 불가 실사 확인.\nKaggle AttentiveSkel-3D (64,33,3), RepCount 제시", "즉시 다운 가능한 AttentiveSkel-3D 및\nRepCount 우선 활용, 3D 합성 병행"),
        ("화면비(3.16배) 및\n각도 왜곡 해석", "16:9 vs 9:16 종횡비로 인한\n15.8% 각도 왜곡 및 등방 정규화 제안", "MediaPipe 2D 정규화 vs 3D 월드 구분 필수.\n월드 좌표에 임의 15.8% 곱셈 금지 경고", "2D 픽셀 각도에만 (W/H) 보정 적용,\n3D 월드 좌표는 미터 단위 보존"),
        ("c4(발뒤꿈치 들림)\n로우앵글 오탐 해결", "바닥 로우앵글 원근 사영 역전 규명,\n락아웃(Lockout) 기준 차분 피치 제안", "발바닥 최저점 0 강제 정규화 금지,\n락아웃 대비 상대 접지 변위 수식 확인", "락아웃 기립 시점 발바닥 각도 영점화\n(Delta alpha_foot) 공식 전격 채택"),
        ("3D Sim2Real\n가상 카메라 구현", "카메라 뷰포트(Pitch 20~30°, FoV 68~74°),\nPython 투영 시뮬레이터 코드 제시", "3D 강체 불변성 함정 경고!\n(3D 회전만 하면 각도 불변이므로 2D 투영 필수)", "3D -> 가상 2D 투영 -> 2D 피처 추출\n재학습 파이프라인 구성"),
        ("도메인 적응\n(UDA) 및 판정 구조", "CDAN + InfoNCE + 동반결함 그래프\n복합 손실함수 제시", "트리 모델은 CDAN/AdaBN 불가 경고.\nCORAL 및 '정상/결함/판정불가' 3-State 권고", "트리 모델용 CORAL 전처리 및\n3-State(Selective Classification) 채택")
    ]

    for i, row in enumerate(comp_rows):
        for j, val in enumerate(row):
            cell = tbl_comp.cell(i+1, j)
            cell.width = col_w_comp[j]
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            align = WD_ALIGN_PARAGRAPH.CENTER if j == 0 else WD_ALIGN_PARAGRAPH.LEFT
            format_cell(cell, val, bold=(j == 0), color=RGBColor(16, 44, 87) if j == 0 else RGBColor(30, 41, 59), bg=bg, align=align, size=Pt(8))

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Section 2
    add_header_styled(doc, "2. 핵심 발견 1: 발뒤꿈치 들림(c4) 85% 오탐의 기하학적 정체 규명", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "두 외부 AI가 만장일치로 증명해 낸 가장 결정적인 수확은 스마트폰 바닥 거치 촬영 시 발뒤꿈치 접지 결함(c4)이 85~90%로 치솟았던 '원근 사영 역전(Perspective Foreshortening & Elevation Illusion)' 메커니즘입니다."
    )

    p_c4_1 = doc.add_paragraph()
    p_c4_1.add_run("■ 원근 투영 왜곡 메커니즘 (Perspective Illusion):\n").bold = True
    p_c4_1.add_run(
        "1. 기하학적 배치: 스마트폰을 바닥(높이 0.1~0.3m)에 두고 위를 올려다보는 상향 피치(Pitch theta = 20~30도)로 거치하면, 카메라 광축 방향 깊이에서 발뒤꿈치(Heel)가 발가락(Toe)보다 뒤에 있으므로 Z_c(heel) > Z_c(toe)가 성립합니다.\n"
        "2. 핀홀 카메라 투영 공식: 픽셀 수직 좌표 v = f_y * (Y_c / Z_c) + c_y\n"
        "3. 투영 역전: 상향 회전된 카메라 좌표계에서, 뒤꿈치와 발끝이 물리적 지면(Y_w = 0)에 완벽히 붙어 있더라도 더 멀리 있는 뒤꿈치가 화면 상단(더 작은 v 좌표, 즉 높은 위치)에 맺히게 됩니다.\n"
        "4. 오탐 발생: 모델은 사용자가 발바닥을 바닥에 완벽히 밀착하고 있음에도 불구하고, 2D 투영 좌표 상 뒤꿈치가 발끝보다 공중에 붕 떠 있는 것으로 인식하여 c4 결함을 85~90% 확률로 확정해 버렸던 것입니다."
    )

    p_c4_2 = doc.add_paragraph()
    p_c4_2.add_run("■ 완벽한 해결책: 락아웃 기준 차분 피치 (Lockout-Anchored Differential Pitch):\n").bold = True
    p_c4_2.add_run(
        "두 AI 모두 발바닥의 '절대 각도'나 '절대 Y좌표'를 보지 말고, 동작 시작 및 종료 시점의 '기립 락아웃(Lockout, 상위 10% 프레임)' 구간 발바닥 각도를 영점 기준선(Zero-Baseline)으로 잡는 차분 공식을 제안했습니다.\n\n"
        "    Delta alpha_foot(t) = alpha_foot(t) - (1 / |T_lockout|) * sum_{t0 in T_lockout} alpha_foot(t0)\n\n"
        "• 효과: 바닥 로우앵글로 인해 뒤꿈치가 시각적으로 높게 찍히더라도, 기립 시점에도 동일하게 높게 찍혀 있으므로 두 값의 차분을 구하면 카메라 설치 각도로 인한 '고정 투영 편향(Static Perspective Bias)'이 수학적으로 100% 상쇄(Cancellation)됩니다!\n"
        "• 실무 적용: 이 수식 하나만 전처리 파이프라인에 반영해도, 추가적인 모델 재학습 없이 c4 오탐률을 즉시 정상 수준(10% 이하)으로 떨어뜨릴 수 있습니다."
    )

    # Section 3
    add_header_styled(doc, "3. 핵심 발견 2: AI 2의 날카로운 팩트체크와 기술적 주의점 4가지", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "외부 AI 2는 단순한 이론 제시를 넘어, 실무 엔지니어링 관점에서 우리가 범하기 쉬운 4가지 치명적 착각을 정확히 짚어냈습니다."
    )

    p_fc1 = doc.add_paragraph()
    p_fc1.add_run("1. 화면비 3.16배와 15.8% 압축의 진실 (MediaPipe 좌표계 구분):\n").bold = True
    p_fc1.add_run(
        "• 16:9 가로(1.78)와 9:16 세로(0.56)의 종횡비 비율이 3.16배인 것은 맞지만, 이것이 3D 공간의 물리적 압축을 뜻하지는 않습니다.\n"
        "• MediaPipe Pose는 정규화 2D 랜드마크(x_n, y_n in [0, 1])와 월드 랜드마크(x_w, y_w, z_w, 골반 중심 미터 단위)를 엄격히 분리하여 출력합니다.\n"
        "• 경고: 15.8% 보정 계수는 2D 픽셀 정규화 좌표로 각도를 계산할 때만 (W/H) 비율을 곱해 보정해야 하며, 미터 단위로 추정된 3D 월드 좌표(world landmarks)에 임의로 15.8%를 곱하면 오히려 물리적 신체 비율을 파괴하게 됩니다."
    )

    p_fc2 = doc.add_paragraph()
    p_fc2.add_run("2. 3D 강체 회전 불변성(Rigid Invariance)의 함정:\n").bold = True
    p_fc2.add_run(
        "• AI-Hub 3D 좌표를 가상 카메라의 상향 틸트 각도(Pitch 20~30°)로 3D 회전시킨 후, 그 3D 좌표에서 관절각(내적)을 다시 계산하면 각도는 회전 전과 정확히 100% 동일합니다(강체 회전 불변성).\n"
        "• 따라서 3D 좌표만 단순히 회전시켜서는 모델에 아무런 도메인 적응 효과를 줄 수 없습니다.\n"
        "• 올바른 파이프라인: 반드시 [글로벌 3D -> 가상 카메라 투영 행렬 P -> 2D 픽셀 좌표 -> 2D 특징 추출기 재학습] 경로를 거쳐야만 원근 왜곡과 화면비 내성이 모델에 전달됩니다."
    )

    p_fc3 = doc.add_paragraph()
    p_fc3.add_run("3. 64프레임 스플라인 보간 시 물리적 Jerk(저크) 왜곡 규명:\n").bold = True
    p_fc3.add_run(
        "• 운동 반복 구간을 64프레임 위상(tau in [0, 1])으로 정규화할 때, 실제 운동 시간 T(초)와의 미분 관계는 다음과 같습니다:\n\n"
        "    d^3 p / dt^3 = (1 / T^3) * (d^3 p / dtau^3)\n\n"
        "• 만약 64프레임으로 보간한 뒤 실제 시간 T를 버리고 위상 미분만 수행하면, 운동 수행 시간 T의 3제곱(T^3)에 반비례하는 물리적 Jerk 정보가 증발합니다. 즉, 2초 만에 빠르게 수행한 스쿼트와 5초 동안 천천히 수행한 스쿼트의 흔들림이 동일하게 취급되어 통계 피처가 오염됩니다.\n"
        "• 조치: 원본 영상의 실제 timestamp에서 속도/가속도/Jerk를 미분한 뒤 64프레임으로 샘플링하거나, 반복 지속 시간(Duration T)을 독립 피처로 반드시 결합해야 합니다."
    )

    p_fc4 = doc.add_paragraph()
    p_fc4.add_run("4. 관측 불가능 결함과 '정상 / 결함 / 판정불가' 3-State 전환:\n").bold = True
    p_fc4.add_run(
        "• 33개 관절 랜드마크만으로는 측면에서 무릎 외반(c3)을 보거나, 바벨이 없는 스켈레톤에서 바벨 궤적(d5)을 보거나, 요추 분절 곡률이 없는 골격에서 요추 굴곡(c1)을 완벽히 판별할 수 없습니다.\n"
        "• 무리하게 이진 분류(0 또는 1)를 강제하면 오탐이 구조적으로 누적되므로, 카메라 각도상 관측이 불가능한 부위는 과감히 '판정 불가(Uncertain/Not Available)'로 마스킹하는 3-State 아키텍처(Selective Classification)를 적용해야 합니다."
    )

    # Section 4
    add_header_styled(doc, "4. 핵심 발견 3: 즉시 연동 가능한 실용적 대안 데이터셋 발굴", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "AI 2의 정밀 실사를 통해, 신청 승인 대기 없이 즉시 활용 가능한 실질적 대안 데이터셋 2종이 구체적으로 발굴되었습니다."
    )

    p_ds1 = doc.add_paragraph()
    p_ds1.add_run("1. Kaggle 「AttentiveSkel-3D」 (스쿼트·데드리프트·벤치프레스 507개 클립):\n").bold = True
    p_ds1.add_run(
        "• 데이터 구성: 1 Rep 단위 MP4 507개 + 전처리된 유효 텐서 487개 (.npy)\n"
        "• 놀라운 정합성: 텐서 형상이 (64, 33, 3)으로 우리 프로젝트 vidio-health-trainer의 표준 입력 규격(64프레임, 33개 관절, 3차원 좌표)과 100% 완벽히 일치합니다!\n"
        "• 활용 방안: iPhone 14 Pro 촬영 영상이 다수 포함되어 있어, 스마트폰 카메라 환경에 대한 모델의 사전 검증 및 파인튜닝 텐서로 즉시 투입할 수 있습니다."
    )

    p_ds2 = doc.add_paragraph()
    p_ds2.add_run("2. CVPR 2022 「RepCount」 (유튜브 1,041개 실제 운동 비디오):\n").bold = True
    p_ds2.add_run(
        "• 데이터 구성: 전 세계 일반인들이 스마트폰(세로 9:16 다수)으로 촬영한 스쿼트, 데드리프트, 런지 영상 1,041개.\n"
        "• 활용 방안: 프레임 단위의 1회 반복 시작/최저점/종료 타임스탬프가 제공되므로, 우리 웹 파이프라인의 1 Rep 자동 분할 모듈 검증 및 비지도 도메인 적응(UDA)의 타깃 데이터셋으로 즉시 활용 가능합니다."
    )

    # Section 5
    add_header_styled(doc, "5. vidio-health-trainer 최종 실행 로드맵 (Action Plan)", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "두 외부 AI의 자문 결과를 통합하여, 개발자의 시간과 노력을 낭비하지 않는 최적의 실행 순서를 확정합니다."
    )

    p_act = doc.add_paragraph()
    p_act.add_run(
        "■ 최우선 권고 실행 순서 (Ablation Pipeline):\n\n"
        "1단계: 전처리 수식 정밀화 (코드 수정 1일, 데이터 수집 불필요)\n"
        "  - 락아웃 기준 차분 발바닥 각도(Delta alpha_foot) 적용 -> c4 오탐 85% 즉시 소멸\n"
        "  - 2D 정규화 좌표(W/H 보정)와 3D 월드 좌표(미터 보존) 처리 분리\n"
        "  - 64프레임 보간 시 운동 지속시간(T) 보존 스케일링 적용\n\n"
        "2단계: AI-Hub 3D 가상 카메라 투영기 (Sim2Real) 가동 (1일, 비용 0원)\n"
        "  - AI 1과 2가 제공한 가상 카메라 투영 매트릭스 P = K[R|t] 구현\n"
        "  - AI-Hub 3D 좌표를 세로 9:16 및 바닥 로우앵글(Pitch 20~30°)로 투영하여 10만 개 2D 키포인트 합성 재학습\n\n"
        "3단계: 오픈 데이터셋 교차 검증 (1~2일, 무료 다운로드)\n"
        "  - Kaggle AttentiveSkel-3D (64, 33, 3) 텐서 다운로드 및 포맷 검증\n"
        "  - RepCount 9:16 유튜브 세로 영상으로 1 Rep 자동 분할 성능 검증\n\n"
        "4단계: 관측 가능성 게이트 및 3-State (정상/결함/판정불가) 웹 서비스 적용\n"
        "  - 측면 시점에서 무릎 외반 판정 차단, 바벨 미검출 시 d5 판정 보류"
    )

    os.makedirs(os.path.dirname(DOCX_PATH), exist_ok=True)
    doc.save(DOCX_PATH)
    print(f"[SUCCESS] Word report created: {DOCX_PATH}")

def generate_markdown():
    md_content = """# vidio-health-trainer 심층 기술 평가 보고서
## 외부 AI 2종 기술 자문 결과 심층 종합 분석 및 엔지니어링 팩트체크 보고서

- **문서 번호**: 8주차 공식 기술 감사 보고서 (2026-10-17)
- **프로젝트 명**: `vidio-health-trainer`
- **분석 대상**: 외부 AI 2종(이론/수학 모델 & 실사/코드 검증 모델) 응답 전문
- **Word 배포 문서**: `D:\\자주프 문서\\8주차\\8주차_외부AI_2종_조사결과_교차검증_및_핵심통찰_심층분석보고서.docx`

---

> [!IMPORTANT]
> **핵심 요약 (Executive Summary)**
> 1. **외부 AI 2종의 결정적 합의**: 두 AI 모두 "개발자가 직접 헬스장에서 영상을 찍는 방식은 비효율적이며, 가상 카메라 Sim2Real 투영과 도메인 적응(UDA)으로 해결 가능하다"는 점에 100% 일치함.
> 2. **AI 1(이론 모델)의 기여**: 바닥 로우앵글(Pitch 20~30°)에서 발뒤꿈치(Heel)가 발끝(Toe)보다 광축상 멀어져 화면 상단에 맺히는 **'원근 투영 역전(Elevation Illusion)'**을 완벽히 수식화하고, **락아웃(Lockout) 차분 피치** 및 **ISB JCS 기하 변환** 제안.
> 3. **AI 2(실사 모델)의 날카로운 팩트체크**:
>    - ① MediaPipe 2D 정규화 좌표와 3D 월드 좌표의 혼용 엄금(월드 좌표에 15.8% 임의 곱셈 금지)
>    - ② 3D 강체 회전 불변성의 함정(3D 좌표만 회전하면 각도는 불변이므로 반드시 2D 투영 파이프라인을 거쳐야 함)
>    - ③ 64프레임 보간 시 물리적 Jerk($1/T^3$) 왜곡 주의
>    - ④ 결함별 관측 가능성 게이트 및 **'정상 / 결함 / 판정불가' 3-State** 도입 촉구.
> 4. **데이터셋 실사 확인**: Fitness-AQA·Fit3D는 즉시 다운로드가 불가하지만, Kaggle **`AttentiveSkel-3D`**($(64, 33, 3)$ 텐서 487개)와 **`RepCount`**(유튜브 1,041개 영상)는 즉시 연동 가능한 실용적 대안으로 발굴됨.

---

## 1. 외부 AI 2종 자문 결과 총평 및 상호보완적 구도

사용자께서 의뢰하신 외부 최상위 AI 자문 결과는 매우 수준 높은 두 가지 관점(이론적 수학 모델링 vs 현실적 엔지니어링 팩트체크)을 제공하여, 우리 프로젝트 `vidio-health-trainer`가 가야 할 명확한 기술 로드맵을 완성해 주었습니다.

- **외부 AI 1 (이론/수학 모델 - Google Docs 10페이지)**: 원근 사영 왜곡 메커니즘을 수학적으로 완벽히 규명하고, 가상 카메라 파라미터($K, R, t$), CDAN+InfoNCE 복합 손실함수, ISB Grood & Suntay 좌표계, 락아웃 기준 차분 피치 등 정교한 학술적 프레임워크를 수립했습니다.
- **외부 AI 2 (실사/엔지니어링 모델 - 코드 검증 16페이지)**: 웹 검색 22회, 18개 페이지 정밀 분석, 코드 4회 실행을 거쳐 공개 데이터셋의 실제 라이선스/다운로드 가능 여부를 팩트체크하고, MediaPipe 좌표계의 치명적 혼용 위험, 3D 강체 불변성 함정, 64프레임 보간 시의 Jerk 스케일링 오류 등 실무 구현상의 맹점을 날카롭게 지적했습니다.

### 외부 AI 2종 핵심 비교 매트릭스

| 비교 항목 | 외부 AI 1 (이론/수학 중심) | 외부 AI 2 (실사/엔지니어링 중심) | vidio-health-trainer 종합 채택 |
| :--- | :--- | :--- | :--- |
| **공개 데이터셋<br>가용성 평가** | Fitness-AQA, UI-PRMD, Fit3D 등 추천<br>(학술적 가치 중심 기술) | Fitness-AQA, Fit3D 즉시 다운로드 불가 실사 확인.<br>Kaggle AttentiveSkel-3D (64,33,3), RepCount 제시 | **즉시 다운 가능한 AttentiveSkel-3D 및<br>RepCount 우선 활용, 3D 합성 병행** |
| **화면비(3.16배) 및<br>각도 왜곡 해석** | 16:9 vs 9:16 종횡비로 인한<br>15.8% 각도 왜곡 및 등방 정규화 제안 | MediaPipe 2D 정규화 vs 3D 월드 구분 필수.<br>월드 좌표에 임의 15.8% 곱셈 금지 경고 | **2D 픽셀 각도에만 (W/H) 보정 적용,<br>3D 월드 좌표는 미터 단위 보존** |
| **c4(발뒤꿈치 들림)<br>로우앵글 오탐 해결** | 바닥 로우앵글 원근 사영 역전 규명,<br>락아웃(Lockout) 기준 차분 피치 제안 | 발바닥 최저점 0 강제 정규화 금지,<br>락아웃 대비 상대 접지 변위 수식 확인 | **락아웃 기립 시점 발바닥 각도 영점화<br>($\Delta \alpha_{\text{foot}}$) 공식 전격 채택** |
| **3D Sim2Real<br>가상 카메라 구현** | 카메라 뷰포트(Pitch 20~30°, FoV 68~74°),<br>Python 투영 시뮬레이터 코드 제시 | 3D 강체 불변성 함정 경고!<br>(3D 회전만 하면 각도 불변이므로 2D 투영 필수) | **3D $\to$ 가상 2D 투영 $\to$ 2D 피처 추출<br>재학습 파이프라인 구성** |
| **도메인 적응(UDA)<br>및 판정 구조** | CDAN + InfoNCE + 동반결함 그래프<br>복합 손실함수 제시 | 트리 모델은 CDAN/AdaBN 불가 경고.<br>CORAL 및 '정상/결함/판정불가' 3-State 권고 | **트리 모델용 CORAL 전처리 및<br>3-State(Selective Classification) 채택** |

---

## 2. 핵심 발견 1: 발뒤꿈치 들림(c4) 85% 오탐의 기하학적 정체 규명

두 외부 AI가 만장일치로 증명해 낸 가장 결정적인 수확은 스마트폰 바닥 거치 촬영 시 발뒤꿈치 접지 결함($c_4$)이 85~90%로 치솟았던 **'원근 사영 역전(Perspective Foreshortening & Elevation Illusion)'** 메커니즘입니다.

```mermaid
flowchart TD
    A["스마트폰 바닥 거치<br>(높이 0.1~0.3m, Pitch 20~30° 상향 틸트)"] --> B["발뒤꿈치가 발끝보다 카메라 광축 상 더 먼 거리 위치<br>(Z_heel > Z_toe)"]
    B --> C["핀홀 원근 투영 법칙: v = fy * (Y/Z) + cy<br>더 멀리 있는 뒤꿈치가 화면 상단(작은 v값)에 투영"]
    C --> D["2D 투영 착시: 발바닥이 바닥에 붙어 있어도 뒤꿈치가 붕 뜬 것처럼 인식"]
    D --> E["모델의 c4 결함 확률 85~90% 오탐 폭증!"]
```

### 완벽한 해결책: 락아웃 기준 차분 피치 (Lockout-Anchored Differential Pitch)
두 AI 모두 발바닥의 '절대 각도'나 '절대 Y좌표'를 보지 말고, 동작 시작 및 종료 시점의 **'기립 락아웃(Lockout, 상위 10% 프레임)'** 구간 발바닥 각도를 영점 기준선(Zero-Baseline)으로 잡는 차분 공식을 제안했습니다.

$$\Delta \alpha_{\text{foot}}(t) = \alpha_{\text{foot}}(t) - \frac{1}{|T_{\text{lockout}}|} \sum_{t_0 \in T_{\text{lockout}}} \alpha_{\text{foot}}(t_0)$$

- **효과**: 바닥 로우앵글로 인해 뒤꿈치가 시각적으로 높게 찍히더라도, 기립 시점에도 동일하게 높게 찍혀 있으므로 두 값의 차분을 구하면 카메라 설치 각도로 인한 **'고정 투영 편향(Static Perspective Bias)'이 수학적으로 100% 상쇄(Cancellation)**됩니다.
- **실무 적용**: 이 수식 하나만 전처리 파이프라인에 반영해도, 추가적인 모델 재학습 없이 $c_4$ 오탐률을 즉시 정상 수준(10% 이하)으로 떨어뜨릴 수 있습니다.

---

## 3. 핵심 발견 2: AI 2의 날카로운 팩트체크와 기술적 주의점 4가지

외부 AI 2는 단순한 이론 제시를 넘어, 실무 엔지니어링 관점에서 우리가 범하기 쉬운 4가지 치명적 착각을 정확히 짚어냈습니다.

### 1. 화면비 3.16배와 15.8% 압축의 진실 (MediaPipe 좌표계 구분)
- 16:9 가로(1.78)와 9:16 세로(0.56)의 종횡비 비율이 3.16배인 것은 맞지만, 이것이 3D 공간의 물리적 압축을 뜻하지는 않습니다.
- MediaPipe Pose는 **정규화 2D 랜드마크($x_n, y_n \in [0, 1]$)**와 **월드 랜드마크($x_w, y_w, z_w$, 골반 중심 미터 단위)**를 엄격히 분리하여 출력합니다.
- **경고**: 15.8% 보정 계수는 2D 픽셀 정규화 좌표로 각도를 계산할 때만 $(W/H)$ 비율을 곱해 보정해야 하며, 미터 단위로 추정된 3D 월드 좌표(world landmarks)에 임의로 15.8%를 곱하면 오히려 물리적 신체 비율을 파괴하게 됩니다.

### 2. 3D 강체 회전 불변성(Rigid Invariance)의 함정
- AI-Hub 3D 좌표를 가상 카메라의 상향 틸트 각도(Pitch 20~30°)로 3D 회전시킨 후, 그 3D 좌표에서 관절각(내적)을 다시 계산하면 각도는 회전 전과 정확히 100% 동일합니다(강체 회전 불변성).
- 따라서 3D 좌표만 단순히 회전시켜서는 모델에 아무런 도메인 적응 효과를 줄 수 없습니다.
- **올바른 파이프라인**: 반드시 **[글로벌 3D $\to$ 가상 카메라 투영 행렬 $P$ $\to$ 2D 픽셀 좌표 $\to$ 2D 특징 추출기 재학습]** 경로를 거쳐야만 원근 왜곡과 화면비 내성이 모델에 전달됩니다.

### 3. 64프레임 스플라인 보간 시 물리적 Jerk(저크) 왜곡 규명
- 운동 반복 구간을 64프레임 위상($\tau \in [0, 1]$)으로 정규화할 때, 실제 운동 시간 $T$(초)와의 미분 관계는 다음과 같습니다:

$$\frac{d^3 p}{dt^3} = \frac{1}{T^3} \frac{d^3 p}{d\tau^3}$$

- 만약 64프레임으로 보간한 뒤 실제 시간 $T$를 버리고 위상 미분만 수행하면, 운동 수행 시간 $T$의 3제곱($T^3$)에 반비례하는 물리적 Jerk 정보가 증발합니다. 즉, 2초 만에 빠르게 수행한 스쿼트와 5초 동안 천천히 수행한 스쿼트의 흔들림이 동일하게 취급되어 통계 피처가 오염됩니다.
- **조치**: 원본 영상의 실제 timestamp에서 속도/가속도/Jerk를 미분한 뒤 64프레임으로 샘플링하거나, 반복 지속 시간(Duration $T$)을 독립 피처로 반드시 결합해야 합니다.

### 4. 관측 불가능 결함과 '정상 / 결함 / 판정불가' 3-State 전환
- 33개 관절 랜드마크만으로는 측면에서 무릎 외반($c_3$)을 보거나, 바벨이 없는 스켈레톤에서 바벨 궤적($d_5$)을 보거나, 요추 분절 곡률이 없는 골격에서 요추 굴곡($c_1$)을 완벽히 판별할 수 없습니다.
- 무리하게 이진 분류(0 또는 1)를 강제하면 오탐이 구조적으로 누적되므로, 카메라 각도상 관측이 불가능한 부위는 과감히 **'판정 불가(Uncertain/Not Available)'**로 마스킹하는 3-State 아키텍처(Selective Classification)를 적용해야 합니다.

---

## 4. 핵심 발견 3: 즉시 연동 가능한 실용적 대안 데이터셋 발굴

AI 2의 정밀 실사를 통해, 신청 승인 대기 없이 즉시 활용 가능한 실질적 대안 데이터셋 2종이 구체적으로 발굴되었습니다.

1. **Kaggle 「AttentiveSkel-3D」 (스쿼트·데드리프트·벤치프레스 507개 클립)**:
   - **데이터 구성**: 1 Rep 단위 MP4 507개 + 전처리된 유효 텐서 487개 (`.npy`)
   - **놀라운 정합성**: 텐서 형상이 **$(64, 33, 3)$**으로 우리 프로젝트 `vidio-health-trainer`의 표준 입력 규격(64프레임, 33개 관절, 3차원 좌표)과 100% 완벽히 일치합니다!
   - **활용 방안**: iPhone 14 Pro 촬영 영상이 다수 포함되어 있어, 스마트폰 카메라 환경에 대한 모델의 사전 검증 및 파인튜닝 텐서로 즉시 투입할 수 있습니다.
2. **CVPR 2022 「RepCount」 (유튜브 1,041개 실제 운동 비디오)**:
   - **데이터 구성**: 전 세계 일반인들이 스마트폰(세로 9:16 다수)으로 촬영한 스쿼트, 데드리프트, 런지 영상 1,041개.
   - **활용 방안**: 프레임 단위의 1회 반복 시작/최저점/종료 타임스탬프가 제공되므로, 우리 웹 파이프라인의 1 Rep 자동 분할 모듈 검증 및 비지도 도메인 적응(UDA)의 타깃 데이터셋으로 즉시 활용 가능합니다.

---

## 5. vidio-health-trainer 최종 실행 로드맵 (Action Plan)

두 외부 AI의 자문 결과를 통합하여, 개발자의 시간과 노력을 낭비하지 않는 최적의 실행 순서를 확정합니다.

```mermaid
flowchart TD
    S1["1단계: 전처리 수식 정밀화 (즉시 적용)<br>• 락아웃 차분 발바닥 각도 적용 -> c4 오탐 즉시 해결<br>• 2D 정규화 vs 3D 월드 좌표 분리<br>• 64프레임 보간 시 운동 시간 T 보존"] --> S2["2단계: AI-Hub 3D 가상 카메라 투영 (Sim2Real)<br>• P = K[R|t] 투영 행렬 구현 (Pitch 20~30°, 9:16)<br>• 3D -> 2D 투영 -> 2D 피처 추출 재학습"]
    S2 --> S3["3단계: 오픈 데이터셋 교차 검증<br>• Kaggle AttentiveSkel-3D (64, 33, 3 텐서) 연동<br>• RepCount 유튜브 세로 영상으로 1 Rep 분할 검증"]
    S3 --> S4["4단계: 3-State (정상/결함/판정불가) 웹 서비스 적용<br>• 측면 시점 무릎 외반 차단, 바벨 미검출 시 d5 보류<br>• 불필요한 결함 알림 원천 방어"]
```

이 로드맵에 따라 진행하시면, **개발자가 헬스장에서 땀 흘려 영상을 찍는 고통에서 완전히 해방**되면서도 학술적으로 가장 완벽하고 신뢰도 높은 스마트폰 일반화 인공지능 모델을 구축하실 수 있습니다.
"""
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Markdown report created: {MD_PATH}")

if __name__ == "__main__":
    generate_report()
    generate_markdown()
