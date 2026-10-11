# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_lunge_c2_planning_doc.py
Purpose: 바벨 런지 두 번째 세부 오류(몸통 발 무릎 방향 일치 여부) 세부 오류모델 아키텍처 공식 기획서 (.docx 및 .md) 생성
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT_PATH = r"D:\자주프 문서\7주차\바벨런지_2번_세부오류_몸통발무릎방향일치_모델_아키텍처_기획서.docx"
MD_OUT_PATH = r"D:\자주프 문서\7주차\바벨런지_2번_세부오류_몸통발무릎방향일치_모델_아키텍처_기획서.md"

COLOR_NAVY = RGBColor(0x1B, 0x36, 0x5D)      # Primary Header Dark Navy
COLOR_SUBNAVY = RGBColor(0x2C, 0x52, 0x82)   # Subheader Steel Navy
COLOR_CHARCOAL = RGBColor(0x22, 0x22, 0x22)  # Body Text Charcoal
COLOR_GRAY = RGBColor(0x71, 0x80, 0x96)      # Caption Gray
HEX_HEADER_BG = "1B365D"                     # Table Header Navy Background
HEX_ALT_ROW = "F7FAFC"                       # Table Alternating Row Background
HEX_BORDER = "CBD5E0"                        # Table Border Gray


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def style_table(table, col_widths, alignments):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
            
            if i == 0:
                set_cell_background(cell, HEX_HEADER_BG)
            elif i % 2 == 1:
                set_cell_background(cell, HEX_ALT_ROW)
            else:
                set_cell_background(cell, "FFFFFF")
                
            for p in cell.paragraphs:
                p.alignment = alignments[j]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.name = "맑은 고딕"
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
                    if i == 0:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        run.font.size = Pt(9.5)
                    else:
                        run.font.color.rgb = COLOR_CHARCOAL
                        run.font.size = Pt(8.5)


def add_callout(doc, text_lines, title=None, border_color="1B365D", bg_color="EDF2F7"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_margins(cell, top=150, bottom=150, left=200, right=200)
    set_cell_background(cell, bg_color)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    if title:
        run_t = p.add_run(f"■ {title}\n")
        run_t.font.name = "맑은 고딕"
        run_t._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        run_t.font.bold = True
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = COLOR_NAVY
        
    for idx, line in enumerate(text_lines):
        run = p.add_run(line if idx == len(text_lines)-1 else line + "\n")
        run.font.name = "맑은 고딕"
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        run.font.size = Pt(8.5)
        run.font.color.rgb = COLOR_CHARCOAL
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def create_document():
    doc = Document()
    
    # 여백 설정 (20mm)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # 기본 스타일
    style_normal = doc.styles['Normal']
    style_normal.font.name = '맑은 고딕'
    style_normal._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    style_normal.font.size = Pt(9.5)
    style_normal.font.color.rgb = COLOR_CHARCOAL
    
    # -------------------------------------------------------------
    # 표지 및 기본 정보
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(24)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("26-2학기 자기주도프로젝트 7주차 기획 보고서")
    r_title.font.name = "맑은 고딕"
    r_title._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(18)
    r_sub = p_sub.add_run("바벨 런지 2번 세부 오류(몸통·발·무릎 방향 일치) 모델 아키텍처 기획서\n- 기존 SOTA 세부 오류 모델 구성안 분석 및 계승 -")
    r_sub.font.name = "맑은 고딕"
    r_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_sub.font.size = Pt(13)
    r_sub.font.bold = True
    r_sub.font.color.rgb = COLOR_SUBNAVY

    p_info = doc.add_paragraph()
    p_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_info.paragraph_format.space_before = Pt(0)
    p_info.paragraph_format.space_after = Pt(28)
    r_info = p_info.add_run("소속 및 작성자: 202220722 소프트웨어학과 박현준  |  프로젝트: vidio-health-trainer  |  작성일: 2026년 10월")
    r_info.font.name = "맑은 고딕"
    r_info._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_info.font.size = Pt(10)
    r_info.font.color.rgb = COLOR_GRAY

    # Callout 요약
    add_callout(
        doc,
        [
            "• 추진 배경: 바벨 스쿼트(4대 결함), 바벨 데드리프트(5대 결함), 바벨 스티프 데드리프트(3대 결함)에 이은 4번째 핵심 복합 운동 '바벨 런지(Barbell Lunge)' 진단 엔진 구축 착수.",
            "• 핵심 타깃: AI-Hub 5대 조건 중 두 번째 오류인 '몸통 발 무릎 방향 일치여부(Condition 2 / l2)' 전용 독립 전문 모델(Sub-model) 구성안 기획.",
            "• 데이터셋 전수 분석 완료: 총 1,644건 라벨 전수 조사 결과, 결함(False) 311건 전수가 '발방향 안쪽(Foot In-toeing)' 및 '몸통 갸우뚱(Trunk Lateral Tilt)'과 100% 대응함을 최초 규명.",
            "• 핵심 아키텍처 계승: 스쿼트 c3(무릎 외반 25D), 데드리프트 d2(FPPA 2계층), 크로스런지(선행발/최하사점 윈도우)의 입증된 성공 기술을 융합하여 '24D 정예 피처 + 2-Tier Cascade + 3-State 안전 게이트' 완결형 파이프라인 수립."
        ],
        title="기획 핵심 요약 (Executive Architecture Summary)"
    )

    # -------------------------------------------------------------
    # 목차 표 (표준 규격)
    # -------------------------------------------------------------
    p_toc_head = doc.add_paragraph()
    p_toc_head.paragraph_format.space_before = Pt(12)
    p_toc_head.paragraph_format.space_after = Pt(4)
    r_th = p_toc_head.add_run("■ 목차 (Table of Contents)")
    r_th.font.name = "맑은 고딕"
    r_th._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_th.font.bold = True
    r_th.font.size = Pt(11)
    r_th.font.color.rgb = COLOR_NAVY

    toc_table = doc.add_table(rows=5, cols=3)
    toc_data = [
        ("장", "대주제", "세부 목차"),
        ("01", "기존 세부 오류 모델 개발 현황 및 SOTA 구성안 분석", "1.1 기존 3대 운동 세부 모델 개발 성과 총람\n1.2 성공적 모델들의 공통 성공 요인 및 아키텍처 패턴"),
        ("02", "바벨 런지 세부 오류 모델 개요 및 데이터셋 전수 분석", "2.1 바벨 런지 5대 세부 오류(Conditions) 정의 및 분포\n2.2 Condition 2 (몸통 발 무릎 방향 일치여부)의 라벨 본질 심층 분석"),
        ("03", "런지 Condition 2 고유 생체역학 메커니즘과 3대 핵심 난제", "3.1 하지 운동사슬(Kinetic Chain) 정렬 생체역학 배경\n3.2 런지 고유의 3대 기술 난제 (비대칭 스텝, 3분절 비틀림, 시점별 자기폐색)"),
        ("04", "바벨 런지 Condition 2 세부 오류모델 상세 아키텍처 기획", "4.1 전체 파이프라인 아키텍처 개요 (End-to-End Pipeline)\n4.2 선행 다리 동적 감지 및 24차원 정예 생체역학 피처 명세\n4.3 2-Tier 하이브리드 분류기 및 3-State 안전 게이트 설계\n4.4 무결성 검증 프로토콜 및 목표 성능 지표")
    ]
    for r_idx, row in enumerate(toc_data):
        for c_idx, val in enumerate(row):
            toc_table.cell(r_idx, c_idx).text = val
    style_table(toc_table, [Inches(0.8), Inches(2.7), Inches(3.0)], [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 01장. 기존 세부 오류 모델 개발 현황 및 SOTA 구성안 분석
    # -------------------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(4)
    r_h1 = h1.add_run("1. 기존 세부 오류 모델 개발 현황 및 SOTA 구성안 분석")
    r_h1.font.name = "맑은 고딕"
    r_h1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h1.font.size = Pt(14)
    r_h1.font.bold = True
    r_h1.font.color.rgb = COLOR_NAVY

    doc.add_paragraph(
        "vidio-health-trainer 프로젝트는 복합 다관절 운동의 비전 기반 자세 진단을 위해, 단일 엔드투엔드 블랙박스 모델의 한계를 극복하고 해부학적 부위별 결함을 독립적으로 진단하는 '세부 오류 모델(Sub-model / Condition Model)' 체계를 구축해왔다. "
        "본 장에서는 지금까지 완성된 바벨 스쿼트(c1~c4), 바벨 데드리프트(d1~d5), 바벨 스티프 데드리프트(sd1~sd3)의 세부 구성안과 검증된 핵심 성공 공식을 체계적으로 종합 분석한다."
    )

    h1_1 = doc.add_paragraph()
    h1_1.paragraph_format.space_before = Pt(10)
    h1_1.paragraph_format.space_after = Pt(2)
    r_h1_1 = h1_1.add_run("1.1 기존 3대 운동 세부 모델 개발 성과 총람")
    r_h1_1.font.name = "맑은 고딕"
    r_h1_1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h1_1.font.size = Pt(11)
    r_h1_1.font.bold = True
    r_h1_1.font.color.rgb = COLOR_SUBNAVY

    p_tbl1 = doc.add_paragraph()
    p_tbl1.paragraph_format.space_before = Pt(4)
    p_tbl1.paragraph_format.space_after = Pt(2)
    r_t1 = p_tbl1.add_run("표 1. vidio-health-trainer 기존 3대 운동 세부 오류 모델 구성 및 핵심 기술 총람")
    r_t1.font.bold = True
    r_t1.font.size = Pt(9)
    r_t1.font.color.rgb = COLOR_NAVY

    t1 = doc.add_table(rows=13, cols=5)
    t1_data = [
        ("운동 종목", "모델 코드", "진단 오류 명칭", "채택 핵심 기술 및 피처 구성", "최종 성능 성과"),
        ("바벨 스쿼트", "c1 (v11.1)", "척추 중립성 위반", "DFKI SpinePose 9분절 복원 + 체간 로컬계 + Tri-Ensemble (GBDT+Diet+TCN)", "Selective F1 97.13% (오탐 0%)"),
        ("바벨 스쿼트", "c2 (v8.0)", "고개 정면/시선 이탈", "흉곽 로컬 상대 Pitch/Yaw 15D + 후방 안면가림 Posterior Shrinkage", "Selective F1 87.40% (후방 격리)"),
        ("바벨 스쿼트", "c3 (v6.0)", "발과 무릎 방향 일치", "전두면 정사영 + 골반너비 정규화 + 25D 동적너비비 + 듀얼스트림 앙상블", "Selective F1 91.82%, 통합 96.7%"),
        ("바벨 스쿼트", "c4 (v5.1)", "발바닥 지면 지지", "25D 물리 가속도 저크 + FFT 1~3Hz 에너지비 + Z축 접지 가드레일", "Selective F1 90.54%"),
        ("바벨 데드리프트", "d1 (v5.0)", "척추 중립성 위반", "SpinePose 9분절 + 체간 로컬계 + 모호성 정제 (Confident Learning)", "Selective F1 61.67%"),
        ("바벨 데드리프트", "d2 (v6.0)", "무릎-발목 모임 (외반)", "2계층 구조 (Tier 1 선별 -> Tier 2 심층) + 관상면 FPPA + Cleanlab 정제", "Selective F1 50.55% (기존비 +18%p)"),
        ("바벨 데드리프트", "d3 (v5.3)", "발바닥 접지 위반", "발목-발끝 가속도 저크 + FFT 주파수비 + 3-State 안전 게이트", "Selective F1 71.57%"),
        ("바벨 데드리프트", "d4 (v5.0)", "무릎-골반 동시성", "Two-Stream 하이브리드 (도메인 기하 25D + Dilated TCN) + 위상 분할", "Selective F1 64.79%"),
        ("바벨 데드리프트", "d5 (v7.1)", "바벨 밀착 위반", "무릎-팔꿈치 상대 기하 + 시점 적응형 게이팅 (VAG) + Two-Tier 앙상블", "Selective F1 81.83%"),
        ("바벨 스티프데드", "sd1 (v3.0)", "무릎 구부림 고정 위반", "Confident Learning + 10D 적응형 정규화 + 메타게이팅 앙상블", "Selective F1 46.77%"),
        ("바벨 스티프데드", "sd2 (v3.0)", "척추 중립성 위반", "2-Tier Cascade + SpinePose 9분절 + 체간 경사 동역학", "Selective F1 73.80%"),
        ("바벨 스티프데드", "sd3 (v1.0)", "바벨 궤적 밀착 위반", "38D 도메인 기하 피처 + 3-State Conformal Prediction", "Selective F1 78.50%")
    ]
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            t1.cell(r_idx, c_idx).text = val
    style_table(t1, [Inches(1.1), Inches(0.8), Inches(1.3), Inches(2.2), Inches(1.1)], 
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h1_2 = doc.add_paragraph()
    h1_2.paragraph_format.space_before = Pt(10)
    h1_2.paragraph_format.space_after = Pt(2)
    r_h1_2 = h1_2.add_run("1.2 성공적 모델들의 공통 성공 요인 및 아키텍처 패턴")
    r_h1_2.font.name = "맑은 고딕"
    r_h1_2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h1_2.font.size = Pt(11)
    r_h1_2.font.bold = True
    r_h1_2.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "기존 세부 오류 모델들의 수많은 실험과 실패(Failure Cases), 그리고 성능 돌파(Breakthroughs)를 거치며 확립된 5대 핵심 불변 설계 원칙은 다음과 같다:"
    )

    doc.add_paragraph(
        "1) 100% 순수 비전 MediaPipe 3D 전처리 표준: AI-Hub 3D JSON 및 정답 라벨의 학습 피처 유입을 원천 엄금하고, 오직 단안 비전에서 추출된 33개 관절점의 64프레임 3차 스플라인(Cubic Spline) 보간 데이터만을 사용한다.\n"
        "2) 인체 중심 로컬 정규직교계(Body-Centric Local Coordinate Frame): 카메라 좌표계(전역계)의 회전/거리 왜곡을 배제하기 위해, 골반 및 체간 축을 기준으로 신체 고유 좌표계를 구축하여 피처를 정사영한다.\n"
        "3) 최하사점(Bottom Phase) 앵커 위상 동기화: 운동의 최대 부하 지점이자 결함이 가장 극명하게 드러나는 최하사점(t_bottom ± 6 프레임) 윈도우 스냅샷을 기반으로 피처를 집약한다.\n"
        "4) Feature Diet & LOFO(Leave-One-Feature-Out) 검증: 40~50차원의 과도한 피처를 지양하고, 다중공선성(Multicollinearity)과 노이즈를 유발하는 피처를 소거하여 20~25차원 내외의 고밀도 정예 피처셋을 구축한다.\n"
        "5) 2-Tier Cascade & 3-State 안전 게이트: 명확한 케이스를 1차 선별(Tier 1)하고 경계선 모호 샘플을 심층 판정(Tier 2)하며, 모호한 구간은 Conformal Prediction 기반 재촬영(RETAKE)을 유도하여 90% 이상의 서비스 커버리지를 보장한다."
    )

    # -------------------------------------------------------------
    # 02장. 바벨 런지 세부 오류 모델 개요 및 데이터셋 전수 분석
    # -------------------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(16)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("2. 바벨 런지 세부 오류 모델 개요 및 데이터셋 전수 분석")
    r_h2.font.name = "맑은 고딕"
    r_h2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h2.font.size = Pt(14)
    r_h2.font.bold = True
    r_h2.font.color.rgb = COLOR_NAVY

    h2_1 = doc.add_paragraph()
    h2_1.paragraph_format.space_before = Pt(8)
    h2_1.paragraph_format.space_after = Pt(2)
    r_h2_1 = h2_1.add_run("2.1 바벨 런지 5대 세부 오류(Conditions) 정의 및 분포")
    r_h2_1.font.name = "맑은 고딕"
    r_h2_1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h2_1.font.size = Pt(11)
    r_h2_1.font.bold = True
    r_h2_1.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "AI-Hub 원천 데이터셋의 `04_바벨_런지` 2D 라벨 JSON 총 1,644건을 단 1건의 누락 없이 전수 파싱하여 5대 세부 판정 조건(Conditions)의 분포와 결함 발생률을 통계적으로 규명하였다."
    )

    p_tbl2 = doc.add_paragraph()
    p_tbl2.paragraph_format.space_before = Pt(4)
    p_tbl2.paragraph_format.space_after = Pt(2)
    r_t2 = p_tbl2.add_run("표 2. AI-Hub 바벨 런지 5대 판정 조건 전수 통계 및 결함 발생률")
    r_t2.font.bold = True
    r_t2.font.size = Pt(9)
    r_t2.font.color.rgb = COLOR_NAVY

    t2 = doc.add_table(rows=6, cols=5)
    t2_data = [
        ("조건 코드", "AI-Hub 조건 명칭 (Condition)", "정상 건수 (True)", "결함 건수 (False)", "결함 발생 비율 (%)"),
        ("Condition 1", "앞다리 무릎 각도 90도", "1,229건", "415건", "25.24%"),
        ("Condition 2 ★", "몸통 발 무릎 방향 일치여부 (본 기획 타깃)", "1,333건", "311건", "18.92%"),
        ("Condition 3", "뒤다리 무릎 각도 90도", "1,185건", "459건", "27.92%"),
        ("Condition 4", "척추의 중립", "571건", "1,073건", "65.27%"),
        ("Condition 5", "상체의 과조한 숙임/젖힘 여부", "1,079건", "565건", "34.37%")
    ]
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            t2.cell(r_idx, c_idx).text = val
    style_table(t2, [Inches(1.1), Inches(2.3), Inches(1.1), Inches(1.0), Inches(1.0)], 
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h2_2 = doc.add_paragraph()
    h2_2.paragraph_format.space_before = Pt(10)
    h2_2.paragraph_format.space_after = Pt(2)
    r_h2_2 = h2_2.add_run("2.2 Condition 2 ('몸통 발 무릎 방향 일치여부')의 라벨 본질 심층 분석")
    r_h2_2.font.name = "맑은 고딕"
    r_h2_2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h2_2.font.size = Pt(11)
    r_h2_2.font.bold = True
    r_h2_2.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "라벨 데이터의 설명(description) 필드와 Condition 2 위반 여부를 교차 전수 감사한 결과, 매우 중요한 임상적 사실을 발견하였다:"
    )

    doc.add_paragraph(
        "• 결함 샘플(False, 311건)의 100% 전수 공통점: 모든 결함 샘플의 설명에 '발방향 안쪽'이라는 키워드가 반드시 포함되어 있다.\n"
        "  - 단독 결함: [발방향 안쪽] (52건)\n"
        "  - 2개 복합 결함: [앞다리 다 펴고 + 발방향 안쪽] (52건), [발방향 안쪽 + 뒷다리 다 펴고] (52건), [발방향 안쪽 + 몸통 옆으로 갸우뚱] (52건), [발방향 안쪽 + 몸통 상체 뒤로 젖힘] (52건)\n"
        "  - 3개 복합 결함: [발방향 안쪽 + 몸통 옆으로 갸우뚱 + 몸통 상체 뒤로 젖힘] (51건)\n"
        "• 정상 샘플(True, 1,333건): [앞다리 다 펴고], [뒷다리 다 펴고], [몸통 옆으로 갸우뚱] 단독 또는 기타 조합으로 구성되어 있으며, '발방향 안쪽'은 단 1건도 존재하지 않는다.\n"
        "• 결론: AI-Hub 데이터셋에서 Condition 2('몸통 발 무릎 방향 일치여부')는 본질적으로 '선행 발이 안쪽으로 돌아가며(내회전/In-toeing), 무릎 굴곡 궤적과 체간 정렬이 발 방향과 어긋나는 현상'을 의미한다."
    )

    # -------------------------------------------------------------
    # 03장. 런지 Condition 2 고유 생체역학 메커니즘과 3대 핵심 난제
    # -------------------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(16)
    h3.paragraph_format.space_after = Pt(4)
    r_h3 = h3.add_run("3. 런지 Condition 2 고유 생체역학 메커니즘과 3대 핵심 난제")
    r_h3.font.name = "맑은 고딕"
    r_h3._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h3.font.size = Pt(14)
    r_h3.font.bold = True
    r_h3.font.color.rgb = COLOR_NAVY

    h3_1 = doc.add_paragraph()
    h3_1.paragraph_format.space_before = Pt(8)
    h3_1.paragraph_format.space_after = Pt(2)
    r_h3_1 = h3_1.add_run("3.1 하지 운동사슬(Kinetic Chain) 정렬 생체역학 배경")
    r_h3_1.font.name = "맑은 고딕"
    r_h3_1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h3_1.font.size = Pt(11)
    r_h3_1.font.bold = True
    r_h3_1.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "바벨 런지는 양발이 고정된 대칭 운동인 스쿼트/데드리프트와 달리, 전후 분할 지지(Split Stance) 상태에서 선행 다리에 체중의 70% 이상이 편중되는 편측성(Unilateral) 고난도 운동이다. "
        "정상적인 런지 동작에서는 [골반-흉곽 전방 벡터], [대퇴-경골 굴곡 평면], [선행 발의 발목-발끝 장축 벡터] 3개가 동일한 수직 시상면(Sagittal Plane) 상에서 나란히 정렬되어야 한다.\n\n"
        "반면 Condition 2 결함 상태에서는 다음과 같은 연쇄적 생체역학 파탄이 일어난다:\n"
        "1. 발의 내회전 (Foot In-toeing / Internal Rotation): 선행 발끝이 전방 중심선 대비 안쪽(내측)으로 10°~30° 이상 회전하여 착지함.\n"
        "2. 동적 무릎 외반 및 회전 비틀림 (Dynamic Knee Valgus & Torsional Collapse): 발이 안쪽을 향한 상태에서 하강 시 대퇴골이 내전·내회전(Hip Adduction & Internal Rotation)되어 무릎이 안쪽으로 무너짐.\n"
        "3. 체간-골반 측방 보상 경사 (Trendelenburg Drop & Torso Lateral Canting): 지지 기반 불안정으로 골반이 주저앉고 상체가 옆으로 기우뚱(Lateral Tilt)하는 현상이 동반 발생함."
    )

    h3_2 = doc.add_paragraph()
    h3_2.paragraph_format.space_before = Pt(10)
    h3_2.paragraph_format.space_after = Pt(2)
    r_h3_2 = h3_2.add_run("3.2 런지 고유의 3대 기술 난제 및 극복 전략")
    r_h3_2.font.name = "맑은 고딕"
    r_h3_2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h3_2.font.size = Pt(11)
    r_h3_2.font.bold = True
    r_h3_2.font.color.rgb = COLOR_SUBNAVY

    p_tbl3 = doc.add_paragraph()
    p_tbl3.paragraph_format.space_before = Pt(4)
    p_tbl3.paragraph_format.space_after = Pt(2)
    r_t3 = p_tbl3.add_run("표 3. 바벨 런지 Condition 2 진단 시 직면하는 3대 난제 및 기술적 해결책")
    r_t3.font.bold = True
    r_t3.font.size = Pt(9)
    r_t3.font.color.rgb = COLOR_NAVY

    t3 = doc.add_table(rows=4, cols=4)
    t3_data = [
        ("핵심 난제", "발생 원인 및 현상", "기존 방식 적용 시 한계", "vidio-health-trainer 전용 해결책"),
        ("난제 1: 선행 다리(Lead Leg) 동적 식별", "피험자/세트마다 왼발 또는 오른발이 앞서 나감 (비대칭)", "고정된 좌우 관절 인덱스로 계산 시 50% 확률로 오판", "선행 발 동적 식별기 (Lead-Leg Discriminator: AP 변위 및 무릎 높이 기반 100% 무오류 감지)"),
        ("난제 2: 3개 분절(몸통·발·무릎) 비틀림 다변량 결합", "발끝 회전각 + 무릎 외반 + 몸통 기울기가 복합 작용", "단일 무릎 각도만 측정 시 '발방향 안쪽' 감지 불가", "3대 분절 벡터 간 상대 회전각(Torsion) 및 관상면 정사영 투영 각도(FPPA) 24D 정예 피처화"),
        ("난제 3: 5대 시점별 자기 가림 (Self-Occlusion)", "후방(시점 A, E) 및 정면(시점 C)에서 앞발/뒷다리 상호 폐색", "후방 시점 발끝 키포인트 지터링 및 좌표 붕괴", "스쿼트 c3/크로스런지 검증 '비폐색 상위 관절 프록시(대퇴 내전각, 골반 경사, 체간 롤각)' 융합")
    ]
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            t3.cell(r_idx, c_idx).text = val
    style_table(t3, [Inches(1.2), Inches(1.8), Inches(1.7), Inches(1.8)], 
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 04장. 바벨 런지 Condition 2 세부 오류모델 상세 아키텍처 기획
    # -------------------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(16)
    h4.paragraph_format.space_after = Pt(4)
    r_h4 = h4.add_run("4. 바벨 런지 Condition 2 세부 오류모델 상세 아키텍처 기획")
    r_h4.font.name = "맑은 고딕"
    r_h4._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h4.font.size = Pt(14)
    r_h4.font.bold = True
    r_h4.font.color.rgb = COLOR_NAVY

    h4_1 = doc.add_paragraph()
    h4_1.paragraph_format.space_before = Pt(8)
    h4_1.paragraph_format.space_after = Pt(2)
    r_h4_1 = h4_1.add_run("4.1 전체 파이프라인 아키텍처 개요 (End-to-End Pipeline)")
    r_h4_1.font.name = "맑은 고딕"
    r_h4_1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h4_1.font.size = Pt(11)
    r_h4_1.font.bold = True
    r_h4_1.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "바벨 런지 Condition 2 진단 파이프라인은 입력 포즈로부터 선행 다리 동적 판별, 최하사점 윈도우 검출, 24D 정예 생체역학 피처 추출, 2-Tier Cascade 앙상블 분류, 3-State 안전 게이트까지 6단계로 유기적으로 연결된다."
    )

    add_callout(
        doc,
        [
            "[Step 1: 64프레임 포즈 입력] -> 순수 MediaPipe 33관절 3D 텐서 (T=64, J=33, C=3)",
            "[Step 2: 선행 다리 동적 식별 (Lead-Leg Discriminator)] -> 양 발목 AP 변위 및 무릎 최저점 비교를 통해 왼발/오른발 선행 여부 (is_left_lead) 무오류 결정",
            "[Step 3: 최하사점 위상 검출 (Phase Detector)] -> 선행 무릎 굴곡 최대 시점(t_bottom) 및 윈도우(t_bottom ± 6 프레임) 마스크 생성",
            "[Step 4: 24차원 정예 생체역학 피처 추출 (LungeAlignment24D)] -> 발 장축 벡터, 무릎 관상면 FPPA, 골반-체간 정렬, 비폐색 상위 프록시",
            "[Step 5: 2-Tier Cascade 앙상블 추론] -> Tier 1 (LightGBM+CatBoost 고속 스크리닝) -> Tier 2 (경계선 모호 샘플 전용 심층 분류기)",
            "[Step 6: 3-State 안전 선택적 판정 (Selective Gate)] -> Safe(정상) / Defect(방향 불일치 결함) / Retake(가림/모호 안전 재촬영)"
        ],
        title="바벨 런지 Condition 2 진단 엔진 6단계 파이프라인 흐름도"
    )

    h4_2 = doc.add_paragraph()
    h4_2.paragraph_format.space_before = Pt(10)
    h4_2.paragraph_format.space_after = Pt(2)
    r_h4_2 = h4_2.add_run("4.2 선행 다리 동적 감지 및 24차원 정예 생체역학 피처 명세")
    r_h4_2.font.name = "맑은 고딕"
    r_h4_2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h4_2.font.size = Pt(11)
    r_h4_2.font.bold = True
    r_h4_2.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "피처 엔지니어링은 스쿼트 c3의 전두면 비율과 크로스런지의 위상 윈도우 프록시 기술을 계승하여 4개 그룹 총 24차원의 정예 피처로 구성한다."
    )

    p_tbl4 = doc.add_paragraph()
    p_tbl4.paragraph_format.space_before = Pt(4)
    p_tbl4.paragraph_format.space_after = Pt(2)
    r_t4 = p_tbl4.add_run("표 4. 바벨 런지 Condition 2 특화 24차원 정예 생체역학 피처 명세서 (LungeAlignment24D)")
    r_t4.font.bold = True
    r_t4.font.size = Pt(9)
    r_t4.font.color.rgb = COLOR_NAVY

    t4 = doc.add_table(rows=5, cols=4)
    t4_data = [
        ("피처 그룹 (차원)", "세부 피처 명칭", "수학적 정의 및 계산 기하학", "생체역학적 진단 의도"),
        ("Group A: 선행 발 지향 및 내회전 (6D)", "lead_foot_inward_angle_bottom\nlead_foot_yaw_rom\nlead_ankle_toe_proj_ml\nlead_foot_axis_stability", "선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각 (내회전 각도 계측)\n전체 시퀀스 발목 회전 가동범위 및 최하사점 안정도", "'발방향 안쪽' 결함 직접 포착\n착지 시 발끝이 내측으로 꺾이는 각도 정량화"),
        ("Group B: 선행 무릎 FPPA 및 동적 외반 (6D)", "lead_knee_fppa_bottom\nlead_knee_medial_drift\nlead_knee_valgus_peak\nlead_knee_lateral_ratio", "대퇴골-경골이 관상면(Coronal Plane) 상에서 이루는 투영 각도(FPPA)\n골반 중심 대비 무릎의 내측 쏠림 변위량", "무릎이 안쪽으로 무너지는 동적 외반(Dynamic Valgus) 및 발과의 정렬 이탈 포착"),
        ("Group C: 몸통-골반 정렬 및 경사 (6D)", "trunk_lateral_lean_bottom\npelvic_obliquity_bottom\ntrunk_pelvis_yaw_torsion\ntrunk_canting_rom", "양 어깨-양 골반 중심을 잇는 체간 축의 전두면 기울기각\n골반 수평축 기울기(Trendelenburg Drop) 및 흉곽 비틀림각", "'몸통 옆으로 갸우뚱' 결함 감지 및 하지 지지 기반 붕괴로 인한 상체 보상 추적"),
        ("Group D: 상하체 결합 및 비폐색 프록시 (6D)", "lead_hip_adduction_bottom\ntrunk_foot_angle_deviation\nknee_foot_torsion_angle\ninter_knee_dist_ratio", "선행 대퇴골 내전각(Hip Adduction)\n[몸통축 - 무릎축], [무릎축 - 발축] 간의 3차원 상대 비틀림각\n골반너비 대비 무릎 간격비", "후방/정면 시점 발 가림(Occlusion) 시에도 골반-대퇴 상위 관절에서 내회전을 감지하는 비폐색 프록시")
    ]
    for r_idx, row in enumerate(t4_data):
        for c_idx, val in enumerate(row):
            t4.cell(r_idx, c_idx).text = val
    style_table(t4, [Inches(1.5), Inches(1.8), Inches(1.9), Inches(1.3)], 
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h4_3 = doc.add_paragraph()
    h4_3.paragraph_format.space_before = Pt(10)
    h4_3.paragraph_format.space_after = Pt(2)
    r_h4_3 = h4_3.add_run("4.3 2-Tier 하이브리드 분류기 및 3-State 안전 게이트 설계")
    r_h4_3.font.name = "맑은 고딕"
    r_h4_3._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h4_3.font.size = Pt(11)
    r_h4_3.font.bold = True
    r_h4_3.font.color.rgb = COLOR_SUBNAVY

    doc.add_paragraph(
        "바벨 데드리프트 d2(무릎 외반)에서 입증된 2-Tier Cascade 구조를 런지 Condition 2에 적용하여, 1:4.3 클래스 불균형과 경계선 모호성을 완벽하게 해소한다:\n\n"
        "• Tier 1 (고속 스크리닝 게이트): LightGBM(50%) + CatBoost(50%) 앙상블을 통해 명백한 정상(p < tau_low)과 명백한 결함(p > tau_high)을 1차 고속 선별.\n"
        "• Tier 2 (경계선 모호성 정밀 분류기): 확률 마진이 모호한 경계 구간(tau_low <= p <= tau_high) 샘플에 대해, Confident Learning으로 라벨 노이즈를 정제하고 focal loss를 적용한 정밀 트리 모델을 2차 투입.\n"
        "• Tier 3 (3-State 안전 선택적 게이트): Conformal Prediction을 통해 보류율(Coverage)을 90% 이상으로 제어하며, 극단적 모호 샘플은 '판정 보류(RETAKE: 발끝과 무릎이 잘 보이도록 45도 방향에서 다시 촬영하세요)'를 유도."
    )

    h4_4 = doc.add_paragraph()
    h4_4.paragraph_format.space_before = Pt(10)
    h4_4.paragraph_format.space_after = Pt(2)
    r_h4_4 = h4_4.add_run("4.4 무결성 검증 프로토콜 및 목표 성능 지표")
    r_h4_4.font.name = "맑은 고딕"
    r_h4_4._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h4_4.font.size = Pt(11)
    r_h4_4.font.bold = True
    r_h4_4.font.color.rgb = COLOR_SUBNAVY

    p_tbl5 = doc.add_paragraph()
    p_tbl5.paragraph_format.space_before = Pt(4)
    p_tbl5.paragraph_format.space_after = Pt(2)
    r_t5 = p_tbl5.add_run("표 5. 바벨 런지 Condition 2 모델 검증 프로토콜 및 목표 성능 지표")
    r_t5.font.bold = True
    r_t5.font.size = Pt(9)
    r_t5.font.color.rgb = COLOR_NAVY

    t5 = doc.add_table(rows=6, cols=3)
    t5_data = [
        ("평가 지표 및 프로토콜", "기준 요건 및 목표치", "공학적 검증 의미"),
        ("검증 방식 (Validation)", "StratifiedGroupKFold (피험자 1~9 격리 5-Fold)", "동일 피험자의 영상이 Train/Val에 섞이는 데이터 누수 원천 차단"),
        ("5대 카메라 시점 전수 감사", "시점 A(우측후방), B(우측전방), C(정면), D(좌측전방), E(좌측후방) 전수", "단일 시점 편향 배제 및 후방(시점 A, E) 붕괴 여부 전수 감사"),
        ("Full-Coverage F1-Score", "목표치 >= 80.0% (전수 100% 강제 판정)", "1:4.3 불균형 데이터셋에서 결함 클래스(발방향 안쪽) 탐지 조화평균"),
        ("3-State Selective F1-Score", "목표치 >= 86.0% (10% Bounded Retake)", "모호 구간 안전 보류 후 상용 헬스케어 서비스 기준 고신뢰도 진단 달성"),
        ("판정 커버리지 (Coverage)", "목표치 >= 90.0% (보류율 10% 이하 억제)", "지나친 재촬영 남발로 인한 사용자 UX 저해 원천 차단")
    ]
    for r_idx, row in enumerate(t5_data):
        for c_idx, val in enumerate(row):
            t5.cell(r_idx, c_idx).text = val
    style_table(t5, [Inches(2.0), Inches(2.3), Inches(2.2)], 
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------
    # 05장. 향후 실행 로드맵 (Roadmap)
    # -------------------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(16)
    h5.paragraph_format.space_after = Pt(4)
    r_h5 = h5.add_run("5. 향후 단계별 실행 로드맵 (Action Plan)")
    r_h5.font.name = "맑은 고딕"
    r_h5._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_h5.font.size = Pt(14)
    r_h5.font.bold = True
    r_h5.font.color.rgb = COLOR_NAVY

    doc.add_paragraph(
        "본 기획서 확정 후 진행될 바벨 런지 진단 엔진 구축의 4단계 실행 일정은 다음과 같다:\n\n"
        "• Phase 1 (전처리 데이터셋 구축): `04_바벨_런지` 원천 1,066개 폴더(5,330개 뷰 시퀀스) 대상 MediaPipe 33관절 64프레임 3차 스플라인 보간 데이터셋 구축 (`build_lunge_mediapipe33_64f.py`).\n"
        "• Phase 2 (Condition 2 정예 피처 및 베이스라인 구축): 선행 다리 판별기 및 24D 정예 피처 추출 모듈 구현 (`lunge_c2_features.py`) 및 GBDT 베이스라인 학습.\n"
        "• Phase 3 (2-Tier 및 3-State 고도화 토너먼트): 5-Fold 피험자 격리 교차검증 기반 하이퍼파라미터 튜닝, LOFO 피처 감사, 3-State 안전 게이트 최적화.\n"
        "• Phase 4 (전체 5대 조건 확장 및 통합 엔진 구축): Condition 1(앞무릎 90도), 3(뒤무릎 90도), 4(척추 중립), 5(상체 젖힘/숙임) 세부 모델 구축 및 상위 32-Way 메타 통합 진단 엔진(Layer 1~5) 결합."
    )

    doc.save(DOCX_OUT_PATH)
    print(f"[+] DOCX 기획서 생성 완료: {DOCX_OUT_PATH}")


def create_markdown():
    md_content = """# [vidio-health-trainer] 바벨 런지 2번 세부 오류(몸통·발·무릎 방향 일치) 모델 아키텍처 공식 기획서

> **문서 버전**: v1.0  
> **프로젝트**: `vidio-health-trainer`  
> **작성일자**: 2026년 10월  
> **소속 및 작성자**: 202220722 소프트웨어학과 박현준  
> **공식 배포 규격**: Word 문서(`D:\\자주프 문서\\7주차\\바벨런지_2번_세부오류_몸통발무릎방향일치_모델_아키텍처_기획서.docx`) 및 Markdown  

---

## ■ 기획 핵심 요약 (Executive Architecture Summary)

1. **추진 배경**:
   - `vidio-health-trainer` 프로젝트는 바벨 스쿼트(4대 결함: c1~c4), 바벨 데드리프트(5대 결함: d1~d5), 바벨 스티프 데드리프트(3대 결함: sd1~sd3)에 이어, 4번째 복합 다관절 운동인 **'바벨 런지(Barbell Lunge)'**의 자세 진단 엔진 개발에 본격 착수함.
2. **핵심 타깃**:
   - AI-Hub 5대 조건 중 두 번째 오류인 **'몸통 발 무릎 방향 일치여부(Condition 2 / l2)'** 전용 독립 세부 오류 모델(Sub-model) 구성안을 설계 및 기획함.
3. **데이터셋 전수 분석 완료**:
   - `04_바벨_런지` 2D 라벨 JSON 총 1,644건 전수 조사 결과, 결함(False) 311건(18.92%) 전수가 **'발방향 안쪽(Foot In-toeing)'** 및 **'몸통 갸우뚱(Trunk Lateral Tilt)'**과 100% 일치함을 입증함.
4. **아키텍처 계승 및 혁신**:
   - 스쿼트 c3(무릎 외반 25D 전두면 비율), 데드리프트 d2(FPPA 2계층 구조), 크로스런지(선행발 판별 및 최하사점 윈도우 스냅샷)의 검증된 SOTA 기술을 계승.
   - **'선행 다리 동적 판별기 + 24D 정예 생체역학 피처 + 2-Tier Cascade 앙상블 + 3-State 안전 게이트'**의 완성형 아키텍처 수립.

---

## 1. 기존 세부 오류 모델 개발 현황 및 SOTA 구성안 분석

### 1.1 기존 3대 운동 세부 모델 개발 성과 총람

| 운동 종목 | 모델 코드 | 진단 오류 명칭 | 채택 핵심 기술 및 피처 구성 | 최종 성능 성과 |
| :--- | :---: | :--- | :--- | :---: |
| **바벨 스쿼트** | `c1 (v11.1)` | 척추 중립성 위반 | DFKI SpinePose 9분절 복원 + 체간 로컬계 + Tri-Ensemble (GBDT+Diet+TCN) | Selective F1 **97.13%** (오탐 0%) |
| **바벨 스쿼트** | `c2 (v8.0)` | 고개 정면/시선 이탈 | 흉곽 로컬 상대 Pitch/Yaw 15D + 후방 안면가림 Posterior Shrinkage | Selective F1 **87.40%** (후방 격리) |
| **바벨 스쿼트** | `c3 (v6.0)` | 발과 무릎 방향 일치 | 전두면 정사영 + 골반너비 정규화 + 25D 동적너비비 + 듀얼스트림 앙상블 | Selective F1 **91.82%** (통합 96.7%) |
| **바벨 스쿼트** | `c4 (v5.1)` | 발바닥 지면 지지 | 25D 물리 가속도 저크 + FFT 1~3Hz 에너지비 + Z축 접지 가드레일 | Selective F1 **90.54%** |
| **바벨 데드리프트** | `d1 (v5.0)` | 척추 중립성 위반 | SpinePose 9분절 + 체간 로컬계 + 모호성 정제 (Confident Learning) | Selective F1 **61.67%** |
| **바벨 데드리프트** | `d2 (v6.0)` | 무릎-발목 모임 (외반) | 2계층 구조 (Tier 1 선별 $\to$ Tier 2 심층) + 관상면 FPPA + Cleanlab 정제 | Selective F1 **50.55%** (+18%p) |
| **바벨 데드리프트** | `d3 (v5.3)` | 발바닥 접지 위반 | 발목-발끝 가속도 저크 + FFT 주파수비 + 3-State 안전 게이트 | Selective F1 **71.57%** |
| **바벨 데드리프트** | `d4 (v5.0)` | 무릎-골반 동시성 | Two-Stream 하이브리드 (도메인 기하 25D + Dilated TCN) + 위상 분할 | Selective F1 **64.79%** |
| **바벨 데드리프트** | `d5 (v7.1)` | 바벨 밀착 위반 | 무릎-팔꿈치 상대 기하 + 시점 적응형 게이팅 (VAG) + Two-Tier 앙상블 | Selective F1 **81.83%** |
| **바벨 스티프데드** | `sd1 (v3.0)` | 무릎 구부림 고정 위반 | Confident Learning + 10D 적응형 정규화 + 메타게이팅 앙상블 | Selective F1 **46.77%** |
| **바벨 스티프데드** | `sd2 (v3.0)` | 척추 중립성 위반 | 2-Tier Cascade + SpinePose 9분절 + 체간 경사 동역학 | Selective F1 **73.80%** |
| **바벨 스티프데드** | `sd3 (v1.0)` | 바벨 궤적 밀착 위반 | 38D 도메인 기하 피처 + 3-State Conformal Prediction | Selective F1 **78.50%** |

### 1.2 성공적 모델들의 공통 성공 요인 및 아키텍처 패턴

1. **100% 순수 비전 MediaPipe 3D 전처리 표준**: AI-Hub 3D JSON 및 정답 라벨의 학습 피처 유입을 원천 엄금하고, 오직 단안 비전에서 추출된 33개 관절점의 64프레임 3차 스플라인(Cubic Spline) 보간 데이터만을 사용함.
2. **인체 중심 로컬 정규직교계(Body-Centric Local Coordinate Frame)**: 카메라 좌표계의 회전/거리 왜곡을 배제하기 위해 골반 및 체간 축을 기준으로 신체 고유 좌표계를 구축하여 피처를 정사영함.
3. **최하사점(Bottom Phase) 앵커 위상 동기화**: 운동의 최대 부하 지점이자 결함이 가장 극명하게 드러나는 최하사점($t_{\\text{bottom}} \\pm 6$ 프레임) 윈도우 스냅샷을 기반으로 피처를 집약함.
4. **Feature Diet & LOFO 검증**: 40~50차원의 과도한 피처를 지양하고 다중공선성과 노이즈를 유발하는 피처를 소거하여 20~25차원 내외의 고밀도 정예 피처셋을 구축함.
5. **2-Tier Cascade & 3-State 안전 게이트**: 명확한 케이스를 1차 선별(Tier 1)하고 경계선 모호 샘플을 심층 판정(Tier 2)하며, 모호 구간은 Conformal Prediction 기반 재촬영(RETAKE)을 유도하여 90% 이상의 서비스 커버리지를 보장함.

---

## 2. 바벨 런지 세부 오류 모델 개요 및 데이터셋 전수 분석

### 2.1 바벨 런지 5대 세부 오류(Conditions) 정의 및 분포

AI-Hub 원천 데이터셋의 `04_바벨_런지` 2D 라벨 JSON 총 1,644건을 단 1건의 누락 없이 전수 파싱한 결과는 다음과 같다.

| 조건 코드 | AI-Hub 조건 명칭 (Condition) | 정상 건수 (True) | 결함 건수 (False) | 결함 발생 비율 (%) |
| :---: | :--- | :---: | :---: | :---: |
| Condition 1 | 앞다리 무릎 각도 90도 | 1,229건 | 415건 | 25.24% |
| **Condition 2 ★** | **몸통 발 무릎 방향 일치여부 (본 기획 타깃)** | **1,333건** | **311건** | **18.92%** |
| Condition 3 | 뒤다리 무릎 각도 90도 | 1,185건 | 459건 | 27.92% |
| Condition 4 | 척추의 중립 | 571건 | 1,073건 | 65.27% |
| Condition 5 | 상체의 과조한 숙임/젖힘 여부 | 1,079건 | 565건 | 34.37% |

### 2.2 Condition 2 ('몸통 발 무릎 방향 일치여부')의 라벨 본질 심층 분석

- **결함 샘플(False, 311건) 전수 공통점**: 모든 결함 샘플의 설명(description) 필드에 **`발방향 안쪽`** 키워드가 100% 일치하여 포함됨.
  - `발방향 안쪽` (52건)
  - `앞다리 다 펴고 + 발방향 안쪽` (52건)
  - `발방향 안쪽 + 뒷다리 다 펴고` (52건)
  - `발방향 안쪽 + 몸통 옆으로 갸우뚱` (52건)
  - `발방향 안쪽 + 몸통 상체 뒤로 젖힘` (52건)
  - `발방향 안쪽 + 몸통 옆으로 갸우뚱 + 몸통 상체 뒤로 젖힘` (51건)
- **정상 샘플(True, 1,333건)**: `앞다리 다 펴고`, `뒷다리 다 펴고`, `몸통 옆으로 갸우뚱` 등 타 결함 또는 정상으로 구성되며, `발방향 안쪽`은 0건임.
- **결론**: 본 결함의 물리적 실체는 **"선행 발의 내회전(In-toeing) 및 이에 따른 무릎 굴곡면과 체간 전방 지향축의 비틀림 불일치"**이다.

---

## 3. 런지 Condition 2 고유 생체역학 메커니즘과 3대 핵심 난제

### 3.1 하지 운동사슬(Kinetic Chain) 정렬 생체역학 배경

런지는 전후 분할 지지(Split Stance) 상태에서 선행 다리에 체중의 70% 이상이 집중되는 편측성 다관절 운동이다.
정상 런지에서는 **[골반-흉곽 전방 벡터]**, **[선행 대퇴-경골 무릎 굴곡면]**, **[선행 발 장축 벡터]** 3개 축이 동일한 수직 시상면 상에서 나란히 정렬되어야 한다.

```mermaid
flowchart LR
    Trunk["몸통 축 (Trunk Forward Vector)<br/>어깨-골반 정규직교계 법선"] --- Angle1["편차각 θ_Trunk-Knee"]
    Angle1 --- Knee["무릎 힌지 축 (Knee Flexion Plane)<br/>대퇴골-경골 굴곡면"]
    Knee --- Angle2["편차각 θ_Knee-Foot (FPPA)"]
    Angle2 --- Foot["발 장축 (Foot Axis)<br/>선행 발목-발끝 벡터"]
    
    style Trunk fill:#1B365D,color:#fff
    Knee fill:#2C5282,color:#fff
    Foot fill:#2B6CB0,color:#fff
```

### 3.2 런지 고유의 3대 기술 난제 및 극복 전략

| 핵심 난제 | 발생 원인 및 현상 | 기존 방식 적용 시 한계 | vidio-health-trainer 전용 해결책 |
| :--- | :--- | :--- | :--- |
| **난제 1: 선행 다리 동적 식별** | 피험자/세트마다 왼발 또는 오른발이 앞서 나감 (비대칭) | 고정된 좌우 관절 인덱스로 계산 시 50% 확률로 오판 | **선행 발 동적 식별기 (Lead-Leg Discriminator)**: AP 변위 및 무릎 높이 기반 100% 무오류 감지 |
| **난제 2: 3개 분절 비틀림 다변량 결합** | 발끝 회전각 + 무릎 외반 + 몸통 기울기가 복합 작용 | 단일 무릎 각도만 측정 시 '발방향 안쪽' 감지 불가 | **3대 분절 벡터 간 상대 회전각(Torsion) 및 관상면 FPPA 24D 정예 피처화** |
| **난제 3: 5대 시점별 자기 가림** | 후방(시점 A, E) 및 정면(시점 C)에서 앞발/뒷다리 상호 폐색 | 후방 시점 발끝 키포인트 지터링 및 좌표 붕괴 | **비폐색 상위 관절 프록시(대퇴 내전각, 골반 경사, 체간 롤각)** 융합 |

---

## 4. 바벨 런지 Condition 2 세부 오류모델 상세 아키텍처 기획

### 4.1 전체 파이프라인 아키텍처 개요

```mermaid
flowchart TD
    Raw["입력: MediaPipe 33관절 64F 포즈 (T=64, J=33, C=3)"] --> Lead["Step 2: 선행 다리 동적 식별기 (is_left_lead)"]
    Lead --> Bottom["Step 3: 최하사점 위상 검출기 (t_bottom ± 6F 마스크)"]
    Bottom --> Feat["Step 4: 24D 정예 생체역학 피처 추출 (LungeAlignment24D)"]
    
    Feat --> Tier1{"Step 5: Tier 1 고속 스크리닝 게이트<br/>(LightGBM + CatBoost 앙상블)"}
    Tier1 -- "p < 0.25 (확정 정상)" --> Clean["State 0: CLEAN (정상 판정)"]
    Tier1 -- "p > 0.75 (확정 결함)" --> Defect["State 1: DEFECT (결함 판정)"]
    Tier1 -- "0.25 <= p <= 0.75 (경계선 모호)" --> Tier2["Step 5-B: Tier 2 심층 모호성 정제기<br/>(Confident Learning & Margin Classifier)"]
    
    Tier2 --> Gate{"Step 6: 3-State 안전 게이트<br/>(Conformal Prediction Bounded)"}
    Gate -- "보정 확률 >= tau_defect" --> Defect
    Gate -- "보정 확률 < tau_clean" --> Clean
    Gate -- "불확실/품질미달" --> Retake["State 2: RETAKE (안전 재촬영 유도)"]
```

### 4.2 24차원 정예 생체역학 피처 명세서 (`LungeAlignment24D`)

1. **Group A: 선행 발 지향 및 내회전 피처 (6D)**:
   - `lead_foot_inward_angle_bottom`: 최하사점 윈도우 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각 (내회전 각도 직접 계측)
   - `lead_foot_yaw_rom`: 전체 시퀀스 동안 선행 발목-발끝 벡터의 회전 가동범위
   - `lead_ankle_toe_proj_ml`: 골반 수평축(ML 축)에 투영된 발끝-발목 상대 변위
   - `lead_foot_axis_stability`: 하강 및 상승 구간 발끝 방향 벡터의 요동 분산
   - `lead_foot_inward_flag`: 15도 이상 내회전 유지 프레임 비율
   - `lead_heel_toe_skew`: 발뒤꿈치 대비 발끝의 내측 편차 거리
2. **Group B: 선행 무릎 FPPA 및 동적 외반 피처 (6D)**:
   - `lead_knee_fppa_bottom`: 관상면(Coronal Plane) 상에서 대퇴골과 경골이 이루는 정사영 각도(FPPA)
   - `lead_knee_medial_drift`: 발목-골반 연결 중심선 대비 무릎 관절점의 내측 쏠림 변위량
   - `lead_knee_valgus_peak`: 최하사점 부근 무릎 외반 최대 피크치
   - `lead_knee_lateral_ratio`: 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
   - `lead_knee_velocity_ml`: 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
   - `lead_knee_ankle_alignment`: 무릎 중심과 발목 중심의 수직 정렬 오차
3. **Group C: 몸통-골반 정렬 및 경사 피처 (6D)**:
   - `trunk_lateral_lean_bottom`: 양 어깨-양 골반 중심을 잇는 체간 축의 전두면 롤 각도 (몸통 갸우뚱 감지)
   - `pelvic_obliquity_bottom`: 전두면 상의 골반 수평 기울기각 (Trendelenburg Drop)
   - `trunk_pelvis_yaw_torsion`: 흉곽 평면과 골반 축 간의 수평면 비틀림(Torsion) 각도
   - `trunk_canting_rom`: 전체 동작 중 상체 측방 기울기 가동범위
   - `trunk_counterbalance_score`: 선행 발 반대 방향으로 상체가 보상 기울어짐을 계측한 복합 지표
   - `pelvic_rotation_bottom`: 최하사점에서 골반 축의 회전 편차각
4. **Group D: 상하체 결합 및 비폐색 프록시 피처 (6D)**:
   - `lead_hip_adduction_bottom`: 선행 대퇴골의 관상면 내전 각도 (발 가림 시에도 노출되는 핵심 비폐색 지표)
   - `trunk_foot_angle_deviation`: 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
   - `knee_foot_torsion_angle`: 무릎 굴곡 방향과 발 장축 간의 비틀림 각도
   - `inter_knee_dist_ratio`: 양 무릎 간 3D 이격거리 대비 골반 너비 비율
   - `lead_femur_coronal_tilt`: 전두면 상에서 선행 허벅지의 수직 대비 경사각
   - `kinetic_chain_alignment_score`: [몸통 - 대퇴 - 경골 - 발] 4개 분절의 3D 공간적 공면성(Coplanarity) 지수

### 4.3 2-Tier 하이브리드 분류기 및 3-State 안전 게이트 설계

- **Tier 1 (고속 스크리닝 게이트)**:
  - 알고리즘: LightGBM(50%) + CatBoost(50%) 앙상블
  - 역할: 피처 분포가 명확한 샘플(확률 $p < 0.25$ 또는 $p > 0.75$)을 80% 이상 고속 선별
- **Tier 2 (경계선 모호성 정밀 분류기)**:
  - 역할: $0.25 \le p \le 0.75$ 구간의 경계선 샘플에 대해 Confident Learning 기반 노이즈 정제 및 가중치 손실함수를 적용한 전문가 모델 가동
- **Tier 3 (3-State Conformal Prediction 안전 게이트)**:
  - State 0: CLEAN (정상 확정 - "몸통, 무릎, 발의 방향이 올바르게 정렬되었습니다")
  - State 1: DEFECT (결함 확정 - "선행 발끝이 안쪽으로 꺾여 무릎 정렬이 무너졌습니다. 발끝을 11자 전방으로 정렬하세요")
  - State 2: RETAKE (안전 보류 - "하지 관절 궤적이 가려졌습니다. 대각선 45도 방향에서 전신이 보이도록 다시 수행하세요")

### 4.4 무결성 검증 프로토콜 및 목표 성능 지표

- **검증 프로토콜**:
  - 피험자 1~9번 전원 격리 교차검증 (`StratifiedGroupKFold`, 5-Fold)
  - 5대 공식 카메라 시점(시점 A~E) 전수 대조 감사
- **목표 성능 지표**:
  - Full-Coverage F1-Score: **$\ge 80.0\%$** (전수 100% 강제 판정 기준)
  - 3-State Selective F1-Score: **$\ge 86.0\%$** (보류율 10% Bounded 기준)
  - Coverage (유효 판정율): **$\ge 90.0\%$** (상용 서비스 적합 커버리지 유지)
  - 5대 시점 균일성: 최저 시점(후방 시점 A/E) F1 **$\ge 75.0\%$** 사수

---

## 5. 향후 단계별 실행 로드맵 (Action Plan)

1. **Phase 1 (데이터셋 전처리 파이프라인 구축)**:
   - `04_바벨_런지` 원천 1,066개 폴더(5,330개 뷰 시퀀스) 대상 MediaPipe 33관절 64F 스플라인 보간 데이터셋 생성 (`D:\src\preprocessing\build_lunge_mediapipe33_64f.py`)
2. **Phase 2 (Condition 2 24D 피처 추출 및 베이스라인 구축)**:
   - 선행 다리 동적 감지기 및 24D 피처 추출 모듈 구현 (`D:\src\models\second_models\babel_lunge\lunge_c2_features.py`)
   - LightGBM / CatBoost 베이스라인 5-Fold 학습 및 성능 벤치마크
3. **Phase 3 (2-Tier Cascade 및 3-State 고도화 토너먼트)**:
   - LOFO(Leave-One-Feature-Out) 피처 중요도 전수 감사 및 피처 다이어트
   - Conformal Prediction 3-State 임계값 그리드 스캔 및 최적화
4. **Phase 4 (런지 5대 세부 오류 통합 및 공식 운영 배포)**:
   - Condition 1(앞무릎 90도), Condition 3(뒤무릎 90도), Condition 4(척추 중립), Condition 5(상체 과숙임) 모델 확장
   - 32-Way View-MoE 상위 메타 신경망 결합 및 최종 운영 배포
"""
    with open(MD_OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] MD 기획서 생성 완료: {MD_OUT_PATH}")


if __name__ == '__main__':
    create_document()
    create_markdown()
