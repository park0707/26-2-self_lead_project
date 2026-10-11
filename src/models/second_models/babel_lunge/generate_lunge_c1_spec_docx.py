# -*- coding: utf-8 -*-
"""
generate_lunge_c1_spec_docx.py
- vidio-health-trainer 바벨 런지 c1(앞다리 무릎 각도 90도) 세부 오류 모델 구축 기획서 생성 스크립트
- 출력:
  1. D:\\자주프 문서\\6주차\\바벨_런지_c1_앞다리_무릎각도_세부오류모델_구축기획서.docx
  2. D:\\자주프 문서\\6주차\\바벨_런지_c1_앞다리_무릎각도_세부오류모델_구축기획서.md
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = r"D:\자주프 문서\6주차\바벨_런지_c1_앞다리_무릎각도_세부오류모델_구축기획서.docx"
MD_OUT = r"D:\자주프 문서\6주차\바벨_런지_c1_앞다리_무릎각도_세부오류모델_구축기획서.md"


def set_cell_background(cell, hex_color):
    """표 셀 배경색 지정"""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    """표 셀 여백 설정 (dxa 단위: 1pt = 20dxa)"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout_box(doc, text, title="핵심 요약 (Key Takeaway)", bg_color="F0F4F8", border_color="1F4E79"):
    """세련된 콜아웃 강조 박스 추가"""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=160, bottom=160, left=240, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"■ {title}\n")
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(10.5)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    run_body = p.add_run(text)
    run_body.font.name = "맑은 고딕"
    run_body.font.size = Pt(9.5)
    run_body.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


def format_table(tbl, col_widths, headers, rows_data):
    """표 스타일링 공통 함수"""
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Header Row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=160, right=160)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "맑은 고딕"
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for r_idx, row in enumerate(rows_data):
        row_cells = tbl.add_row().cells
        bg_col = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(9.0)
                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Set Column Widths
    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)


def build_lunge_c1_spec():
    doc = Document()

    # 페이지 여백 설정 (0.85인치)
    for s in doc.sections:
        s.top_margin = Inches(0.85)
        s.bottom_margin = Inches(0.85)
        s.left_margin = Inches(0.85)
        s.right_margin = Inches(0.85)

    # ---------------------------------------------------------
    # 문서 타이틀
    # ---------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("[vidio-health-trainer] 바벨 런지 c1(앞다리 무릎 각도 90도)\n세부 오류 모델 구축 사양 및 공학적 기획서")
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("스쿼트·데드리프트 성공 모델 아키텍처 계승 기반 비대칭 단일관절 결함 판정 엔진 설계")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # 메타데이터 표
    meta_table = doc.add_table(rows=1, cols=4)
    meta_headers = ["프로젝트 공식 명칭", "문서 버전 / 상태", "대상 운동 및 오류", "작성 일자 / 표준"]
    meta_rows = [
        [
            "vidio-health-trainer",
            "v1.0 (정식 기획 확정)",
            "04_바벨_런지\nc1: 앞다리 무릎 각도 90도",
            "2026. 10. 03\nAGENTS.md 표준 준수"
        ]
    ]
    format_table(meta_table, [1.6, 1.5, 1.8, 1.6], meta_headers, meta_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ---------------------------------------------------------
    # 핵심 요약 콜아웃 박스
    # ---------------------------------------------------------
    add_callout_box(
        doc,
        "1. 연구 목표: 바벨 런지의 제1 핵심 결함인 '앞다리 무릎 각도 90도 미준수(Under/Over-flexion)'를 고정밀 판정하는 독립 세부 오류 모델 c1을 설계한다.\n"
        "2. 선행 아키텍처 계승: 스쿼트(c1~c4), 데드리프트(d1~d4), 스티프 데드리프트(sd1~sd3 v4)에서 실증된 '64F 3차 스플라인 표준', '피험자 9명 전수 격리(StratifiedGroupKFold)', '5대 시점 전수 감사', 'Two-Tier 모호성 분할', '3-State UX(Clean/Defect/Retake)' 구조를 100% 계승한다.\n"
        "3. 런지 특화 혁신: 대칭 운동과 다른 '비대칭(Unilateral)' 특성을 극복하기 위해 ① 앞다리 자동 판별 엔진(Lead Leg Auto-Detection), ② 시점별 자가 가림(Self-occlusion) 보정 기하학, ③ 최저 전환점(Inflection Point) 윈도우 한정 생체역학 32D 피처셋을 도입한다.\n"
        "4. 데이터 무결성: AI-Hub 1,644개 시퀀스 전수 전처리 및 순수 MediaPipe 33관절 3D 비전 데이터만 활용하여 데이터 누수(Data Leakage)를 원천 차단한다.",
        title="기획 핵심 요약 (Executive Summary)"
    )

    # ---------------------------------------------------------
    # 제1장. 기획 개요 및 공학적 배경
    # ---------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("1. 기획 개요 및 생체역학적 배경")
    run_h1.font.name = "맑은 고딕"
    run_h1.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "바벨 런지(Barbell Lunge)는 양발이 동시에 지면을 지지하며 수직으로 하강하는 스쿼트나 데드리프트와 달리, "
        "한 발이 전방으로 뻗어 나가 체중의 대부분(약 70~80%)을 지탱하고 반대 발이 후방에서 보조하는 대표적인 일측성(Unilateral) 복합 하지 운동이다. "
        "따라서 인공지능 기반 자세 교정 파이프라인에서 가장 먼저 해결해야 할 최우선 과제는 전방 지지 다리의 생체역학적 안정성을 좌우하는 '앞다리 무릎 각도 90도' 달성 여부이다."
    )

    doc.add_paragraph(
        "국가대표 및 임상 스포츠 생체역학 연구에 따르면, 런지 하강 최저점(Peak Flexion)에서 앞다리 대퇴골(Femur)과 정강이뼈(Tibia)가 이루는 사잇각이 90도에 미달하거나 과도하게 구부러질 경우 다음과 같은 중대한 문제가 발생한다:"
    )

    p_bio = doc.add_paragraph()
    p_bio.add_run("• 과소 굴곡(Under-flexion / 앞다리 다 펴고 앉지 않음): ").bold = True
    p_bio.add_run("대퇴사두근 및 둔근(Gluteus)에 유효한 기계적 장력이 전달되지 않아 운동 효과가 전무하며, 불완전한 하강으로 인해 요추부 반동(Lumbar bouncing)을 유발함.\n")
    p_bio.add_run("• 과도 굴곡(Over-flexion / 무릎이 발끝을 과도하게 돌파): ").bold = True
    p_bio.add_run("슬개대퇴관절(Patellofemoral Joint)에 가해지는 전단력(Shear force)이 급격히 증가하여 슬개건염(Patellar tendinitis) 및 반월상 연골 손상을 초래함.\n")
    p_bio.add_run("• 정강이 전방 붕괴(Tibia Forward Tilt): ").bold = True
    p_bio.add_run("앞다리 정강이가 지면과 수직을 유지하지 못하고 전방으로 쏠릴 경우 아킬레스건 및 족저근막에 비정상적인 부하가 집중됨.")

    # ---------------------------------------------------------
    # 제2장. AI-Hub 원천 데이터 정밀 감사 및 분석
    # ---------------------------------------------------------
    h2 = doc.add_heading(level=1)
    run_h2 = h2.add_run("2. AI-Hub 원천 데이터 정밀 감사 (Data Audit & Ground Truth Profile)")
    run_h2.font.name = "맑은 고딕"
    run_h2.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "본 기획을 위해 D:\\Data\\01_raw_data\\labels_json\\04_바벨_런지 및 원천 이미지 시퀀스 전수를 정밀 감사하였다. "
        "감사 결과, 총 1,644개의 JSON 라벨 및 이미지 디렉토리가 완벽히 구축되어 있음을 확인하였다."
    )

    # 라벨 통계 표
    lbl_table = doc.add_table(rows=1, cols=4)
    lbl_headers = ["평가 조건 (Conditions)", "정상(True) 시퀀스", "오류(False) 시퀀스", "결함 비율 (%)"]
    lbl_rows = [
        ["c1: 앞다리 무릎 각도 90도 (타깃 결함)", "1,229건", "415건", "25.24% (불균형 ~1:3)"],
        ["c2: 몸통 발 무릎 방향 일치여부", "1,333건", "311건", "18.92%"],
        ["c3: 뒤다리 무릎 각도 90도", "1,185건", "459건", "27.92%"],
        ["c4-A: 척추의 중립 위반", "571건", "1,073건", "65.27%"],
        ["c4-B: 상체의 과도한 숙임/젖힘 여부", "1,079건", "565건", "34.37%"]
    ]
    format_table(lbl_table, [2.5, 1.3, 1.3, 1.4], lbl_headers, lbl_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "특히 타깃 결함인 'c1: 앞다리 무릎 각도 90도'의 415건 오류 라벨에 명시된 원천 세부 기술(Description)을 교차 전수 분석한 결과, "
        "다음과 같이 8가지 복합 오류 양상이 각각 약 52건씩 완벽히 균등하게 분포되어 있음을 밝혀내었다:"
    )

    # 세부 기술 표
    desc_table = doc.add_table(rows=1, cols=4)
    desc_headers = ["순번", "원천 세부 설명 (Description)", "샘플 수", "오류 유형 분석 및 생체역학적 해석"]
    desc_rows = [
        ["1", "앞다리 다 펴고", "52건", "순수 c1 단독 결함 (최저점 도달 실패, 과소 굴곡)"],
        ["2", "앞다리 다 펴고 발방향 안쪽", "52건", "c1(무릎 미달) + c2(발목 내회전/외반) 복합 결함"],
        ["3", "앞다리 다 펴고 뒷다리 다 펴고", "52건", "c1(앞무릎 미달) + c3(뒷무릎 미달) 양측 하지 결함"],
        ["4", "앞다리 다 펴고 몸통 옆으로 갸우뚱", "52건", "c1(앞무릎 미달) + 골반/체간 측방 불균형(Lateral Tilt)"],
        ["5", "앞다리 다 펴고 상체 뒤로 젖힘", "52건", "c1(앞무릎 미달) + 요추 과신전(Lumbar Lordosis)"],
        ["6", "바닥보고 앞다리 다 펴고", "52건", "c1(앞무릎 미달) + 경추 굴곡(시선 바닥 이탈)"],
        ["7", "하늘보고 앞다리 다 펴고", "52건", "c1(앞무릎 미달) + 경추 신전(시선 상방 이탈)"],
        ["8", "왼쪽보고 앞다리 다 펴고", "51건", "c1(앞무릎 미달) + 경추 회전(시선 측면 편향)"]
    ]
    format_table(desc_table, [0.6, 2.2, 0.9, 2.8], desc_headers, desc_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 피험자 및 시점 감사
    doc.add_paragraph(
        "AI-Hub 피험자 9명 전원(피험자1~9)의 데이터 분포는 다음과 같이 완벽히 보존되어 있으며, "
        "AGENTS.md 원칙에 따라 5-Fold StratifiedGroupKFold(피험자 격리)로 검증을 수행한다:\n"
        "• 피험자 분포: 피험자1(222), 피험자2(179), 피험자3(190), 피험자4(191), 피험자5(160), 피험자6(190), 피험자7(192), 피험자8(192), 피험자9(128) = 총 1,644개 시퀀스.\n"
        "• 카메라 5대 시점: 시점 A(view5, 우측 후방 45°), 시점 B(view2, 우측 전방 45°), 시점 C(view3, 정면 0°), 시점 D(view4, 좌측 전방 45°), 시점 E(view1, 좌측 후방 45°) 전수 완비."
    )

    # ---------------------------------------------------------
    # 제3장. 기존 세부 오류 모델 구성안 분석 및 계승 원칙
    # ---------------------------------------------------------
    h3 = doc.add_heading(level=1)
    run_h3 = h3.add_run("3. 기존 세부 오류 모델 구성안 분석 및 핵심 계승 원칙")
    run_h3.font.name = "맑은 고딕"
    run_h3.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "본 프로젝트(vidio-health-trainer)에서 기 구축되어 검증된 바벨 스쿼트(c1~c4), 바벨 데드리프트(d1~d4), "
        "바벨 스티프 데드리프트(sd1~sd3)의 세부 오류 모델 진화 과정에서 확립된 5대 불변 공학 규격(Core Invariants)을 바벨 런지 c1 모델에 전면 계승한다."
    )

    # 계승 표
    inherit_table = doc.add_table(rows=1, cols=3)
    inherit_headers = ["기존 모델 성공 사례 및 핵심 기술", "바벨 런지 c1 적용 및 계승 방식", "기대 효과 및 공학적 보장"]
    inherit_rows = [
        [
            "스쿼트 c1: Active Phase Clamping\n(최저점 전후 윈도우 한정 분석)",
            "런지 전체 64프레임 중 최저점(Inflection Point) 전후 ±5프레임(약 10프레임) Active Bottom Phase를 추출하여 무릎 각도 평가 집중",
            "동작 시작/종료 시의 선 자세(기립) 구간 노이즈를 100% 제거하고 최저 굴곡 결함 검출력 극대화"
        ],
        [
            "스티프 sd1: LOFO 피처 다이어트\n(다중공선성 및 간섭 피처 제거)",
            "대퇴-정강이 관절각과 상호작용 피처 간의 간섭을 Leave-One-Feature-Out(LOFO) 전수 감사로 사전 스크리닝",
            "불필요한 고차원 피처로 인한 과적합 방지 및 단일 피처의 생체역학적 해석력(Explainability) 확보"
        ],
        [
            "스티프 sd2 v3: Two-Tier 계층 평가\n(Level 1~4 모호성 체계 도입)",
            "Level 1(명확군: 90도 vs 다 펴짐)과 Level 2~3(경계선 80~95도 및 복합 결함군)을 분리하여 2단계 계층적 평가",
            "명확 결함 F1 95%+ 보장 및 경계선 구간에서의 무리한 오분류 원천 차단"
        ],
        [
            "스티프 sd3 v4: Multi-View Dynamic Gating\n(5대 시점별 가중치 차등 및 블라인드 극복)",
            "대각선 뷰(시점 B, D)의 높은 가시성을 주축으로 하고, 정면(시점 C) 및 반대편 가림 시점(A, E)의 불확실성을 동적 게이트로 흡수",
            "단일 카메라 환경에서도 시점 편향 없이 전 시점 평균 F1 85%+ 달성"
        ],
        [
            "공통: Selective 3-State UX\n(Clean / Defect / Retake 분기)",
            "신뢰도 마진(|p - 0.5|)이 낮은 불확실 구간(5~10%)에 대해 '판정보류(Retake Prompt)' 및 핀포인트 촬영 각도 가이드 제공",
            "사용자 강제 오진단(Forced Misclassification) 방지 및 전문 트레이너 수준의 신뢰도 획득"
        ]
    ]
    format_table(inherit_table, [2.2, 2.3, 2.0], inherit_headers, inherit_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ---------------------------------------------------------
    # 제4장. 바벨 런지 c1 특화 도메인 피처 엔지니어링 설계
    # ---------------------------------------------------------
    h4 = doc.add_heading(level=1)
    run_h4 = h4.add_run("4. 바벨 런지 c1 특화 도메인 피처 엔지니어링 설계")
    run_h4.font.name = "맑은 고딕"
    run_h4.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "바벨 런지는 스쿼트나 데드리프트와 달리 '비대칭(Unilateral)' 운동이므로, 기존의 단순 좌우 관절 평균이나 대칭적 기하학을 그대로 적용할 경우 "
        "앞다리와 뒷다리가 혼선되어 모델이 붕괴된다. 따라서 본 기획에서는 3대 전용 메커니즘을 최초 설계한다."
    )

    # 전용 메커니즘 설명
    p_mech = doc.add_paragraph()
    p_mech.add_run("■ 메커니즘 1: 앞다리 자율 식별 엔진 (Lead Leg Auto-Detection Engine)\n").bold = True
    p_mech.add_run(
        "런지 수행 중 피험자가 왼발을 앞으로 내밀었는지 오른발을 내밀었는지 시퀀스 초반(하강 개시 시점, 프레임 10~25)에서 자동으로 판별한다.\n"
        "• 판별 수식: $\\Delta Z_{\\text{ankle}} = Z_{\\text{left\\_ankle}} - Z_{\\text{right\\_ankle}}$ 및 $\\Delta X_{\\text{knee\\_disp}}$\n"
        "• 카메라 전방 방향으로 발목이 더 많이 이동하고, 하강 굴곡(Flexion velocity)이 먼저 일어나는 쪽을 '앞다리(Lead Leg)', 반대쪽을 '뒷다리(Rear Leg)'로 동적 라우팅한다.\n"
        "• 이를 통해 모델 내부에서는 피험자의 발 전환(Left/Right switch) 여부와 무관하게 항상 `lead_knee`, `rear_knee` 표준 슬롯으로 정규화된다."
    )

    p_mech2 = doc.add_paragraph()
    p_mech2.add_run("■ 메커니즘 2: 5대 시점별 자가 가림(Self-Occlusion) 극복 기하학\n").bold = True
    p_mech2.add_run(
        "예를 들어 왼다리가 앞다리일 때, 우측 후방(시점 A)이나 우측 전방(시점 B) 카메라에서는 앞다리가 몸통이나 뒷다리에 의해 부분 가림(Partial Occlusion)된다.\n"
        "• 이를 해결하기 위해 MediaPipe의 Visibility/Presence 신뢰도를 가중치로 사용하는 3차원 골반-대퇴-하퇴 회전 불변 로컬 좌표계(Local Anatomical Coordinate System)를 구성한다.\n"
        "• 2D 화면 투영 오차를 제거하기 위해 3D 관절 벡터 사잇각 $\\arccos\\left(\\frac{\\vec{u}\\cdot\\vec{v}}{\\|\\vec{u}\\|\\|\\vec{v}\\|}\\right)$을 1차 기준으로 삼고, 2D 관절각을 보조 시선 피처로 융합한다."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 32차원 피처 명세 표
    doc.add_paragraph("본 기획에서 최종 정의한 바벨 런지 c1 전용 32차원 생체역학 피처셋(32D Feature Set)의 상세 구조는 다음과 같다:")

    feat_table = doc.add_table(rows=1, cols=4)
    feat_headers = ["피처 그룹 (차원)", "대표 피처 변수명", "생체역학적 계산 정의 및 수식", "결함 판정 기여도 및 물리적 의미"]
    feat_rows = [
        [
            "그룹 1:\n앞다리 무릎 각도\n(8차원)",
            "lead_knee_angle_min\nlead_knee_angle_mean\nlead_knee_90_error_abs\nlead_knee_flexion_rom",
            "최저점 구간(Active Phase) 내 3D 대퇴-정강이 벡터 사잇각의 최소치, 평균치, |최소치 - 90°| 편차, 가동범위(ROM)",
            "c1 결함의 핵심 직접 팩터. 90도 도달 여부 및 덜 구부림(앞다리 다 펴고)을 직접 식별"
        ],
        [
            "그룹 2:\n정강이/대퇴 기하\n(8차원)",
            "lead_tibia_vertical_tilt\nlead_femur_horizontal_tilt\nlead_knee_toe_overhang\nlead_shank_ankle_coupling",
            "지면 수직선 대비 정강이 각도, 수평선 대비 대퇴골 각도, 발끝 대비 무릎 전방 돌출 거리, 족관절 배측굴곡비",
            "정강이가 수직을 유지하는지, 무릎이 과도하게 앞으로 밀리거나 덜 내려가 대퇴가 지면과 평행을 못 이루는지 감지"
        ],
        [
            "그룹 3:\n양 하지 비대칭 협응\n(8차원)",
            "lead_rear_knee_ratio\npelvic_drop_angle\npelvic_axial_rotation\ntrunk_forward_lean_lead",
            "앞다리 무릎 각도 대비 뒷다리 무릎 각도 비율, 골반 좌우 기울기(Trendelenburg sign), 골반 회전각, 상체 전경각",
            "단독 c1 결함뿐만 아니라 골반 비틀림, 상체 보상 작용(상체 젖힘/숙임)으로 인한 가짜 90도 보상 패턴 분리"
        ],
        [
            "그룹 4:\n시계열 템포/시점불변\n(8차원)",
            "bottom_phase_dwell_ratio\nflexion_velocity_peak\nlead_knee_dtw_distance\nview_invariant_flexion_3d",
            "최저점 윈도우 체류 시간 비율, 최대 굴곡 각속도, 골드 정상 템플릿과의 DTW 유사도 거리, 시점 불변 투영 스케일러",
            "정상 90도 수행자의 리드미컬한 감속-정체-가속 템포와 결함자의 얕은 반동 튕김(Bounce) 동작을 정밀 감별"
        ]
    ]
    format_table(feat_table, [1.5, 1.8, 1.8, 1.4], feat_headers, feat_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ---------------------------------------------------------
    # 제5장. 모델 아키텍처 및 Two-Tier 3-State 파이프라인 설계
    # ---------------------------------------------------------
    h5 = doc.add_heading(level=1)
    run_h5 = h5.add_run("5. 모델 아키텍처 및 Two-Tier / 3-State 파이프라인 설계")
    run_h5.font.name = "맑은 고딕"
    run_h5.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "바벨 런지 c1 판별 엔진은 스티프 데드리프트 sd2 v3 및 sd3 v4에서 완성된 **계층형 4-단계 추론 파이프라인(Tier 0 ~ Tier 3)**을 기반으로 설계된다."
    )

    # 파이프라인 계층 표
    arch_table = doc.add_table(rows=1, cols=4)
    arch_headers = ["파이프라인 계층", "구성 모듈 및 알고리즘", "동작 메커니즘 및 판정 기준", "출력 상태 (UX Mapping)"]
    arch_rows = [
        [
            "Tier 0:\n입력 센서 게이트",
            "MediaPipe Visibility Check &\nBounding Box Onset Gate",
            "앞다리/뒷다리 무릎 및 발목 키포인트 가시성 평균 < 0.50 이거나 신체가 화면 밖으로 이탈한 경우 차단",
            "State 2: RETAKE\n'신체 하체가 화면에 다 들어오도록 뒤로 물러서세요'"
        ],
        [
            "Tier 1:\n전문가 앙상블",
            "Tri-GBDT (LightGBM + CatBoost + XGBoost)\n+ 1D-CNN 시계열 잔차 융합",
            "앞서 정의된 32D 피처셋을 입력받아 5-Fold OOF 예측 기반 가중 연성 투표(Soft-voting) 수행",
            "연속 결함 확률 $P(\\text{Defect} \\mid X)$\n및 신뢰 마진 $|P - 0.5|$ 산출"
        ],
        [
            "Tier 2:\nTwo-Tier 모호성 완충",
            "Level 1~4 모호성 체계 분류기\n(Ambiguity Partitioning Gate)",
            "• Level 1(명확군): 90도 완벽 정상 vs 다 펴고 결함\n• Level 2(경계선군): 82°~88° 애매한 구간\n• Level 3(복합 결함군): 시선/상체 복합 오류\n• Level 4(가림군): 반대편 시점 블라인드",
            "계층형 이원화 평가 메타러너\n(Tier 1 고확신 판정 vs Tier 2 전체 보정 판정)"
        ],
        [
            "Tier 3:\n3-State 의사결정 UX",
            "Dynamic Thresholding UX\n($\\tau_{\\text{clean}}=0.38, \\tau_{\\text{defect}}=0.52$)",
            "• $P < 0.38$: 정상 확정 (CLEAN)\n• $P \\ge 0.52$: 결함 확정 (DEFECT)\n• $0.38 \\le P < 0.52$: 판정보류 (RETAKE)",
            "State 0 (CLEAN): '완벽한 90도 런지!'\nState 1 (DEFECT): '무릎을 90도까지 더 굽히세요'\nState 2 (RETAKE): '45도 측면에서 재촬영'"
        ]
    ]
    format_table(arch_table, [1.4, 1.8, 1.8, 1.5], arch_headers, arch_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ---------------------------------------------------------
    # 제6장. 단계별 구현 로드맵 및 검증 계획
    # ---------------------------------------------------------
    h6 = doc.add_heading(level=1)
    run_h6 = h6.add_run("6. 단계별 구현 로드맵 및 검증 계획 (Action Plan)")
    run_h6.font.name = "맑은 고딕"
    run_h6.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "본 기획서 승인 즉시 착수할 바벨 런지 c1 모델 구축의 5단계 정밀 실행 로드맵은 다음과 같다:"
    )

    # 로드맵 표
    plan_table = doc.add_table(rows=1, cols=4)
    plan_headers = ["단계 (Phase)", "주요 작업 내용", "생성 소스코드 및 산출물 파일", "검증 기준 (Target KPI)"]
    plan_rows = [
        [
            "Step 1:\n원천 데이터 전처리\n(Full-Coverage)",
            "04_바벨_런지 1,644개 시퀀스 전수에 대해 MediaPipe 33관절 추출 및 64프레임 3차 스플라인 균일 보간 수행",
            "D:\\Data\\전처리_데이터\\MediaPipe_Full33_64F\\04_바벨_런지\nprecompute_lunge_mediapipe_64f.py",
            "1,644개 시퀀스 결측치(NaN/Inf) 0%\n추출 성공률 100%"
        ],
        [
            "Step 2:\n앞다리 자동식별 &\n피처 추출기 구축",
            "앞다리(Lead Leg) 자율 탐지 로직 및 32차원 생체역학 도메인 피처 추출 모듈 구현",
            "D:\\src\\features\\lunge_c1_features.py\ntest_lunge_c1_features.py",
            "앞다리 좌우 식별 정확도 99%+\n32D 피처 연산 시간 < 15ms"
        ],
        [
            "Step 3:\nGBDT 토너먼트 &\n베이스라인 수립",
            "피험자 9명 StratifiedGroupKFold (5-Fold) 기반 LightGBM, CatBoost, XGBoost 단독 및 앙상블 학습",
            "D:\\src\\models\\second_models\\babel_lunge\\train_and_eval_lunge_c1.py",
            "결함 F1-Score ≥ 85.0%\nROC-AUC ≥ 88.0%"
        ],
        [
            "Step 4:\nTwo-Tier 모호성 고도화\n& LOFO 피처 감사",
            "Level 1~4 모호성 분할 캐시 구축, LOFO 기반 피처 다이어트, 3-State Dynamic Gating 적용",
            "D:\\src\\models\\second_models\\babel_lunge\\train_and_eval_lunge_c1_twotier.py",
            "Tier 1 결함 F1 ≥ 95.0%\nTier 2 결함 F1 ≥ 90.0%\n보류율 ≤ 10.0%"
        ],
        [
            "Step 5:\n5대 시점 전수 감사 &\n공식 배포 번들 패키징",
            "공식 5대 시점(A~E) 세부 성능 대조 감사, 단일 추론 인터페이스 클래스 구축, 공식 보고서 배포",
            "lunge_c1_classifier.py\ntest_lunge_c1_classifier.py\n바벨_런지_c1_성능성과보고서.docx",
            "5대 시점 전수 F1 ≥ 75.0% (시점편향 차단)\n단위 테스트 100% 통과"
        ]
    ]
    format_table(plan_table, [1.4, 2.0, 1.8, 1.3], plan_headers, plan_rows)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ---------------------------------------------------------
    # 제7장. 기대 효과 및 결론
    # ---------------------------------------------------------
    h7 = doc.add_heading(level=1)
    run_h7 = h7.add_run("7. 기대 효과 및 향후 확장 계획")
    run_h7.font.name = "맑은 고딕"
    run_h7.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph(
        "1. **비대칭 운동 AI 진단의 새로운 표준 수립**: "
        "기존 양측 대칭 운동(스쿼트, 데드리프트)에 머물렀던 AI 진단 엔진을 비대칭 단일관절 운동인 바벨 런지로 성공적으로 확장하며, "
        "앞다리 자율 탐지 엔진을 통해 사용자 경험을 극대화함.\n"
        "2. **임상 수준의 고신뢰도 진단**: "
        "Two-Tier 모호성 완충 체계를 통해 명확한 결함에 대해서는 95% 이상의 압도적 F1을 제공하고, 경계선 구간에서는 무리한 오진단을 방지하여 "
        "부상 방지(Patellofemoral Joint 보호)라는 헬스케어 본연의 가치 실현.\n"
        "3. **후속 오류 모델(c2, c3, c4)의 탄탄한 초석**: "
        "본 c1 모델에서 구축된 '앞다리/뒷다리 분절 슬롯'과 '32D 기본 피처 파이프라인'은 곧바로 c2(몸통-발-무릎 방향 일치), c3(뒤다리 무릎 각도 90도), c4(척추 중립) 모델의 "
        "공통 척추-하지 백본으로 재사용되어 향후 개발 속도를 3배 이상 단속시킬 것임."
    )

    # 저장 실행
    os.makedirs(os.path.dirname(DOCX_OUT), exist_ok=True)
    doc.save(DOCX_OUT)
    print(f"[SUCCESS] Word Document saved to: {DOCX_OUT}")


def build_markdown_report():
    md_content = """# [vidio-health-trainer] 바벨 런지 c1(앞다리 무릎 각도 90도) 세부 오류 모델 구축 사양 및 공학적 기획서

> **프로젝트 공식 명칭**: `vidio-health-trainer`  
> **문서 버전 / 상태**: v1.0 (정식 기획 확정)  
> **대상 운동 및 오류**: `04_바벨_런지` / `c1`: 앞다리 무릎 각도 90도 (Lead Knee 90° Flexion Defect)  
> **핵심 패러다임**: 앞다리 자율 식별 엔진 + 32D 생체역학 도메인 피처 + Two-Tier 3-State 아키텍처  
> **작성 일자**: 2026년 10월 3일  
> **규격 준수**: AGENTS.md 표준 (64F 3차 스플라인, 피험자 9명 전수, 5대 시점 A~E 감사, 누수 원천 차단)  

---

## 1. 기획 핵심 요약 (Executive Summary)

1. **연구 목표**: 바벨 런지의 제1 핵심 결함인 '앞다리 무릎 각도 90도 미준수(Under/Over-flexion)'를 고정밀 판정하는 독립 세부 오류 모델 c1을 설계한다.
2. **선행 아키텍처 계승**: 스쿼트(c1~c4), 데드리프트(d1~d4), 스티프 데드리프트(sd1~sd3 v4)에서 실증된 '64F 3차 스플라인 표준', '피험자 9명 전수 격리(StratifiedGroupKFold)', '5대 시점 전수 감사', 'Two-Tier 모호성 분할', '3-State UX(Clean/Defect/Retake)' 구조를 100% 계승한다.
3. **런지 특화 혁신**: 대칭 운동과 다른 '비대칭(Unilateral)' 특성을 극복하기 위해 ① 앞다리 자동 판별 엔진(Lead Leg Auto-Detection), ② 시점별 자가 가림(Self-occlusion) 보정 기하학, ③ 최저 전환점(Inflection Point) 윈도우 한정 생체역학 32D 피처셋을 도입한다.
4. **데이터 무결성**: AI-Hub 1,644개 시퀀스 전수 전처리 및 순수 MediaPipe 33관절 3D 비전 데이터만 활용하여 데이터 누수(Data Leakage)를 원천 차단한다.

---

## 2. 기획 개요 및 생체역학적 배경

바벨 런지(Barbell Lunge)는 양발이 동시에 지면을 지지하며 수직으로 하강하는 스쿼트나 데드리프트와 달리, 한 발이 전방으로 뻗어 나가 체중의 대부분(약 70~80%)을 지탱하고 반대 발이 후방에서 보조하는 대표적인 일측성(Unilateral) 복합 하지 운동이다. 따라서 인공지능 기반 자세 교정 파이프라인에서 가장 먼저 해결해야 할 최우선 과제는 전방 지지 다리의 생체역학적 안정성을 좌우하는 '앞다리 무릎 각도 90도' 달성 여부이다.

국가대표 및 임상 스포츠 생체역학 연구에 따르면, 런지 하강 최저점(Peak Flexion)에서 앞다리 대퇴골(Femur)과 정강이뼈(Tibia)가 이루는 사잇각이 90도에 미달하거나 과도하게 구부러질 경우 다음과 같은 중대한 문제가 발생한다:
* **과소 굴곡(Under-flexion / 앞다리 다 펴고 앉지 않음)**: 대퇴사두근 및 둔근(Gluteus)에 유효한 기계적 장력이 전달되지 않아 운동 효과가 전무하며, 불완전한 하강으로 인해 요추부 반동(Lumbar bouncing)을 유발함.
* **과도 굴곡(Over-flexion / 무릎이 발끝을 과도하게 돌파)**: 슬개대퇴관절(Patellofemoral Joint)에 가해지는 전단력(Shear force)이 급격히 증가하여 슬개건염(Patellar tendinitis) 및 반월상 연골 손상을 초래함.
* **정강이 전방 붕괴(Tibia Forward Tilt)**: 앞다리 정강이가 지면과 수직을 유지하지 못하고 전방으로 쏠릴 경우 아킬레스건 및 족저근막에 비정상적인 부하가 집중됨.

---

## 3. AI-Hub 원천 데이터 정밀 감사 (Data Audit & Ground Truth Profile)

총 1,644개의 JSON 라벨 및 이미지 디렉토리를 전수 감사한 결과는 다음과 같다:

### 3.1 조건별 라벨 분포
| 평가 조건 (Conditions) | 정상(True) 시퀀스 | 오류(False) 시퀀스 | 결함 비율 (%) |
| :--- | :---: | :---: | :---: |
| **c1: 앞다리 무릎 각도 90도 (타깃 결함)** | **1,229건** | **415건** | **25.24% (불균형 ~1:3)** |
| c2: 몸통 발 무릎 방향 일치여부 | 1,333건 | 311건 | 18.92% |
| c3: 뒤다리 무릎 각도 90도 | 1,185건 | 459건 | 27.92% |
| c4-A: 척추의 중립 위반 | 571건 | 1,073건 | 65.27% |
| c4-B: 상체의 과도한 숙임/젖힘 여부 | 1,079건 | 565건 | 34.37% |

### 3.2 c1 결함 415건의 원천 세부 기술(Description) 전수 분석
| 순번 | 원천 세부 설명 (Description) | 샘플 수 | 오류 유형 분석 및 생체역학적 해석 |
| :---: | :--- | :---: | :--- |
| 1 | 앞다리 다 펴고 | 52건 | 순수 c1 단독 결함 (최저점 도달 실패, 과소 굴곡) |
| 2 | 앞다리 다 펴고 발방향 안쪽 | 52건 | c1(무릎 미달) + c2(발목 내회전/외반) 복합 결함 |
| 3 | 앞다리 다 펴고 뒷다리 다 펴고 | 52건 | c1(앞무릎 미달) + c3(뒷무릎 미달) 양측 하지 결함 |
| 4 | 앞다리 다 펴고 몸통 옆으로 갸우뚱 | 52건 | c1(앞무릎 미달) + 골반/체간 측방 불균형(Lateral Tilt) |
| 5 | 앞다리 다 펴고 상체 뒤로 젖힘 | 52건 | c1(앞무릎 미달) + 요추 과신전(Lumbar Lordosis) |
| 6 | 바닥보고 앞다리 다 펴고 | 52건 | c1(앞무릎 미달) + 경추 굴곡(시선 바닥 이탈) |
| 7 | 하늘보고 앞다리 다 펴고 | 52건 | c1(앞무릎 미달) + 경추 신전(시선 상방 이탈) |
| 8 | 왼쪽보고 앞다리 다 펴고 | 51건 | c1(앞무릎 미달) + 경추 회전(시선 측면 편향) |

### 3.3 피험자 및 시점 전수 분포
* **피험자 9명 전원**: 피험자1(222), 피험자2(179), 피험자3(190), 피험자4(191), 피험자5(160), 피험자6(190), 피험자7(192), 피험자8(192), 피험자9(128) = 총 1,644개 시퀀스.
* **공식 5대 시점**: 시점 A(view5, 우측 후방 45°), 시점 B(view2, 우측 전방 45°), 시점 C(view3, 정면 0°), 시점 D(view4, 좌측 전방 45°), 시점 E(view1, 좌측 후방 45°) 전수 완비.

---

## 4. 기존 세부 오류 모델 구성안 분석 및 핵심 계승 원칙

| 기존 모델 성공 사례 및 핵심 기술 | 바벨 런지 c1 적용 및 계승 방식 | 기대 효과 및 공학적 보장 |
| :--- | :--- | :--- |
| **스쿼트 c1: Active Phase Clamping**<br>(최저점 전후 윈도우 한정 분석) | 런지 전체 64프레임 중 최저점(Inflection Point) 전후 ±5프레임(약 10프레임) Active Bottom Phase를 추출하여 무릎 각도 평가 집중 | 동작 시작/종료 시의 선 자세(기립) 구간 노이즈를 100% 제거하고 최저 굴곡 결함 검출력 극대화 |
| **스티프 sd1: LOFO 피처 다이어트**<br>(다중공선성 및 간섭 피처 제거) | 대퇴-정강이 관절각과 상호작용 피처 간의 간섭을 Leave-One-Feature-Out(LOFO) 전수 감사로 사전 스크리닝 | 불필요한 고차원 피처로 인한 과적합 방지 및 단일 피처의 생체역학적 해석력(Explainability) 확보 |
| **스티프 sd2 v3: Two-Tier 계층 평가**<br>(Level 1~4 모호성 체계 도입) | Level 1(명확군: 90도 vs 다 펴짐)과 Level 2~3(경계선 80~95도 및 복합 결함군)을 분리하여 2단계 계층적 평가 | 명확 결함 F1 95%+ 보장 및 경계선 구간에서의 무리한 오분류 원천 차단 |
| **스티프 sd3 v4: Multi-View Dynamic Gating**<br>(5대 시점별 가중치 차등 및 블라인드 극복) | 대각선 뷰(시점 B, D)의 높은 가시성을 주축으로 하고, 정면(시점 C) 및 반대편 가림 시점(A, E)의 불확실성을 동적 게이트로 흡수 | 단일 카메라 환경에서도 시점 편향 없이 전 시점 평균 F1 85%+ 달성 |
| **공통: Selective 3-State UX**<br>(Clean / Defect / Retake 분기) | 신뢰도 마진(|p - 0.5|)이 낮은 불확실 구간(5~10%)에 대해 '판정보류(Retake Prompt)' 및 핀포인트 촬영 각도 가이드 제공 | 사용자 강제 오진단(Forced Misclassification) 방지 및 전문 트레이너 수준의 신뢰도 획득 |

---

## 5. 바벨 런지 c1 특화 도메인 피처 엔지니어링 설계

```mermaid
flowchart LR
    A["입력: 64프레임 런지 시퀀스 (33관절 3D)"] --> B["[메커니즘 1] 앞다리 자율 식별 엔진<br/>(ΔZ 발목 변위 + 굴곡 속도 판별)"]
    B --> C["정규화 슬롯 매핑<br/>(lead_knee, rear_knee)"]
    C --> D["[메커니즘 2] 3D 회전 불변 각도 & 가림 보정"]
    D --> E["[메커니즘 3] Active Bottom Phase Clamping"]
    E --> F["32차원 특화 피처 벡터 산출"]
```

### 5.1 32차원 전용 도메인 피처셋(32D Feature Set) 명세

1. **그룹 1: 앞다리 무릎 각도 피처 (8D)**
   * `lead_knee_angle_min`: 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
   * `lead_knee_angle_mean`: 최저점 윈도우 내 평균 무릎 굴곡 각도 (°)
   * `lead_knee_90_error_abs`: $|\\text{lead\\_knee\\_angle\\_min} - 90^\\circ|$ 절대 오차
   * `lead_knee_flexion_rom`: 선 자세 대비 최저점까지의 무릎 굴곡 가동범위(ROM)
   * `lead_knee_under_90_frames`: 90도 미만(불충분 굴곡) 체류 프레임 수
   * `lead_knee_flex_velocity_max`: 하강 구간 최대 무릎 굴곡 각속도 (deg/s)
   * `lead_knee_accel_inflection`: 최저점 방향 전환 순간의 각가속도
   * `lead_knee_rebound_jitter`: 전환점 전후 미세 흔들림 표준편차

2. **그룹 2: 정강이(Tibia) 및 대퇴(Femur) 기하 피처 (8D)**
   * `lead_tibia_vertical_tilt`: 지면 수직선과 앞다리 정강이 벡터 사잇각 (수직 유지 여부)
   * `lead_femur_horizontal_tilt`: 지면 수평선과 앞다리 대퇴골 벡터 사잇각 (대퇴 평행도)
   * `lead_knee_toe_overhang`: 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (mm 정규화)
   * `lead_shank_ankle_coupling`: 하퇴 분절과 발목 배측굴곡(Dorsiflexion) 각도 커플링 비율
   * `lead_tibia_sagittal_sway`: 하강 중 정강이뼈의 시상면 흔들림 폭
   * `lead_knee_valgus_varus`: 관상면 상 무릎 안쪽/바깥쪽 꺾임 각도
   * `lead_femur_length_ratio`: 3D 포즈 스케일 대비 정규화 대퇴 길이
   * `lead_tibia_length_ratio`: 3D 포즈 스케일 대비 정규화 정강이 길이

3. **그룹 3: 양 하지 비대칭 협응 피처 (8D)**
   * `lead_rear_knee_ratio`: 앞다리 무릎 최소각 대비 뒷다리 무릎 최소각 비율
   * `pelvic_drop_angle`: 관상면 상 골반 좌우 기울기 각도 (골반 처짐)
   * `pelvic_axial_rotation`: 횡단면 상 골반 좌우 회전각
   * `trunk_forward_lean_lead`: 체간 전경각과 앞다리 허벅지 간 협응 사잇각
   * `hip_knee_flexion_coupling`: 고관절 굴곡각 대비 슬관절 굴곡각 비율
   * `stride_length_normalized`: 양 발목 사이 전후 거리 (보폭 정규화)
   * `step_width_normalized`: 양 발목 사이 좌우 너비
   * `center_of_mass_disp_z`: 골반 중심 전후방 이동 궤적 적분치

4. **그룹 4: 시공간 템포 및 시점 불변 피처 (8D)**
   * `bottom_phase_dwell_ratio`: 전체 64프레임 중 최저점 부근 체류 프레임 비율
   * `descent_ascent_tempo_ratio`: 하강 소요 시간 대비 상승 소요 시간 비율
   * `lead_knee_dtw_distance`: 이상적 90도 표준 골드 궤적 대비 DTW 거리
   * `view_invariant_flexion_3d`: 카메라 시점 독립적 로컬 3D 굴곡각
   * `camera_view_azimuth_est`: 카메라 상대 방위각 추정치 (A~E 시점 정규화 보조)
   * `lead_knee_visibility_mean`: 최저점 구간 앞무릎 키포인트 평균 신뢰도
   * `kinetic_energy_inflection`: 최저점 전환 시 관절 운동에너지 극소치
   * `jerk_metric_lead_leg`: 앞다리 무릎 궤적 저크(Jerk, 3차 미분) 평활도

---

## 6. 모델 아키텍처 및 Two-Tier / 3-State 파이프라인 설계

```mermaid
flowchart TD
    A["입력: 64프레임 런지 시퀀스"] --> T0{"Tier 0: 입력 센서 게이트<br/>(MediaPipe Visibility < 0.50 or 프레임 이탈)"}
    
    T0 -- "센서 불량 (가림/이탈)" --> R0["[State 2: RETAKE]<br/>'전신 하체가 화면에 다 들어오도록 뒤로 2걸음 물러서세요'"]
    T0 -- "센서 통과" --> T1["Tier 1: 전문가 앙상블<br/>(LightGBM + CatBoost + XGBoost + TCN)"]
    
    T1 --> T1_Out["연속 결함 확률 P(Defect | X) 산출<br/>신뢰 마진 |P - 0.5| 계산"]
    
    T1_Out --> T2["Tier 2: Two-Tier 모호성 완충 게이트<br/>(Level 1 명확군 vs Level 2~3 모호군 분할)"]
    
    T2 --> T3{"Tier 3: 3-State 의사결정<br/>(tau_clean=0.38, tau_defect=0.52)"}
    
    T3 -- "P(Defect) < 0.38" --> S0["[State 0: CLEAN (정상 확정)]<br/>Clean F1 >= 90% | '완벽한 90도 런지 자세입니다!'"]
    T3 -- "P(Defect) >= 0.52" --> S1["[State 1: DEFECT (결함 확정)]<br/>Defect F1 >= 92% | '앞다리 무릎을 90도까지 더 굽혀주세요'"]
    T3 -- "0.38 <= P(Defect) < 0.52" --> S2["[State 2: RETAKE (판정 보류)]<br/>보류율 <= 10.0% | 대각선 45도 맞춤형 재촬영 유도"]
```

---

## 7. 단계별 구현 로드맵 및 검증 계획 (Action Plan)

| 단계 (Phase) | 주요 작업 내용 | 생성 소스코드 및 산출물 파일 | 검증 기준 (Target KPI) |
| :--- | :--- | :--- | :--- |
| **Step 1: 원천 데이터 전처리** | 04_바벨_런지 1,644개 시퀀스 전수 MediaPipe 33관절 추출 및 64프레임 3차 스플라인 균일 보간 | `D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지`<br>`precompute_lunge_mediapipe_64f.py` | 1,644개 시퀀스 결측치 0%<br>추출 성공률 100% |
| **Step 2: 피처 추출기 구축** | 앞다리 자율 탐지 엔진 및 32차원 생체역학 도메인 피처 추출 모듈 구현 | `D:\src\features\lunge_c1_features.py`<br>`test_lunge_c1_features.py` | 앞다리 식별 정확도 ≥ 99%<br>피처 연산 시간 < 15ms |
| **Step 3: GBDT 토너먼트** | 피험자 9명 StratifiedGroupKFold (5-Fold) 기반 LightGBM/CatBoost/XGBoost 학습 | `D:\src\models\second_models\babel_lunge\train_and_eval_lunge_c1.py` | 결함 F1-Score ≥ 85.0%<br>ROC-AUC ≥ 88.0% |
| **Step 4: Two-Tier 고도화** | Level 1~4 모호성 체계 구축, LOFO 피처 다이어트, 3-State Dynamic Gating 적용 | `D:\src\models\second_models\babel_lunge\train_and_eval_lunge_c1_twotier.py` | Tier 1 결함 F1 ≥ 95.0%<br>Tier 2 결함 F1 ≥ 90.0%<br>보류율 ≤ 10.0% |
| **Step 5: 5대 시점 감사 & 배포** | 공식 5대 시점(A~E) 성능 대조 감사, 단일 추론 인터페이스 클래스 패키징, 공식 보고서 생성 | `lunge_c1_classifier.py`<br>`test_lunge_c1_classifier.py`<br>`바벨_런지_c1_성능성과보고서.docx` | 5대 시점 전수 F1 ≥ 75.0%<br>단위 테스트 100% 통과 |

---

## 8. 기대 효과 및 결론

1. **비대칭 운동 AI 진단의 새로운 표준 수립**: 스쿼트/데드리프트(대칭)를 넘어 런지(비대칭) 영역으로 파이프라인을 성공적으로 확장하며, 앞다리 자율 식별 엔진을 통해 사용자 편의성 극대화.
2. **임상 수준의 고신뢰도 진단**: Two-Tier 모호성 완충 체계를 통해 명확한 결함에 대해서는 95% 이상의 압도적 F1을 제공하고, 경계선 구간에서는 무리한 오진단을 방지하여 슬개대퇴관절(Patellofemoral Joint) 부상 예방이라는 핵심 가치 달성.
3. **후속 오류 모델(c2, c3, c4)의 탄탄한 백본 확보**: 본 c1 모델에서 구축된 '앞다리/뒷다리 분절 슬롯'과 '32D 기본 피처 파이프라인'은 곧바로 c2(발-무릎 방향 일치), c3(뒤다리 무릎 각도 90도), c4(척추 중립) 모델의 공통 백본으로 재사용되어 개발 효율을 극대화함.
"""
    os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[SUCCESS] Markdown Document saved to: {MD_OUT}")


if __name__ == "__main__":
    print("=" * 60)
    print("Building Barbell Lunge c1 Specification Documents...")
    print("=" * 60)
    build_lunge_c1_spec()
    build_markdown_report()
    print("All tasks completed successfully!")
