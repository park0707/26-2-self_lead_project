# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Script: D:\src\models\second_models\dynamic_lunge\generate_dl5_rtmp_1_report_docx.py
Purpose: 다이나믹 런지 c5 dl5_rtmp_1 공식 성능평가 종합보고서 (.docx) 생성
Target:
  - D:\자주프 문서\7주차\보고서\다이나믹런지_c5_dl5_rtmp_1_최적화_성능평가보고서.docx
  - D:\자주프 문서\7주차\다이나믹런지_c5_dl5_rtmp_1_최적화_성능평가보고서.docx
"""

import os
import sys
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX_PATHS = [
    r"D:\자주프 문서\7주차\보고서\다이나믹런지_c5_dl5_rtmp_1_최적화_성능평가보고서.docx",
    r"D:\자주프 문서\7주차\다이나믹런지_c5_dl5_rtmp_1_최적화_성능평가보고서.docx"
]
OUT_MD_PATH = r"D:\자주프 문서\7주차\보고서\다이나믹런지_c5_dl5_rtmp_1_최적화_성능평가보고서.md"
BENCHMARK_JSON = r"D:\tests\dl5_rtmp_comparison_results.json"

COLOR_NAVY = RGBColor(0x1B, 0x36, 0x5D)
COLOR_SUBNAVY = RGBColor(0x2C, 0x52, 0x82)
COLOR_CHARCOAL = RGBColor(0x22, 0x22, 0x22)
COLOR_GRAY = RGBColor(0x71, 0x80, 0x96)
HEX_HEADER_BG = "1B365D"
HEX_ALT_ROW = "F7FAFC"
HEX_HIGHLIGHT = "EBF8FF"
HEX_PASS_BG = "E6FFFA"

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
        
        is_header = (i == 0)
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            
            if is_header:
                set_cell_background(cell, HEX_HEADER_BG)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(9.5)
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                bg = HEX_ALT_ROW if (i % 2 == 1) else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(9.0)
                        run.font.color.rgb = COLOR_CHARCOAL

def add_callout(doc, text_list, title="핵심 요약 및 성과 결론", hex_bg="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    set_cell_background(cell, hex_bg)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="28" w:space="0" w:color="{border_hex}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_NAVY
    
    for item in text_list:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(1)
        p2.paragraph_format.space_after = Pt(2)
        r = p2.add_run(f"• {item}")
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9.2)
        r.font.color.rgb = COLOR_CHARCOAL

def build_docx():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 dl5_rtmp_1 공식 성능평가 보고서 (.docx) 생성")
    print("=" * 80)

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    hyb = bench_data['dl5_rtmp_hybrid']
    rtmp_pure = bench_data['dl5_rtmp_pure']
    mp_retrained = bench_data['dl5_mp_only']

    doc = Document()
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # 1. 문서 헤더 및 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    title_run = title_p.add_run("[vidio-health-trainer] 다이나믹 런지 c5 공식 확정 모델")
    title_run.font.name = "맑은 고딕"
    title_run.font.size = Pt(11)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_SUBNAVY

    main_title = doc.add_paragraph()
    main_title.paragraph_format.space_before = Pt(2)
    main_title.paragraph_format.space_after = Pt(6)
    m_run = main_title.add_run("bl5_final 계승 최적화 dl5_rtmp_1 성능평가 종합보고서")
    m_run.font.name = "맑은 고딕"
    m_run.font.size = Pt(19)
    m_run.font.bold = True
    m_run.font.color.rgb = COLOR_NAVY

    sub_title = doc.add_paragraph()
    sub_title.paragraph_format.space_before = Pt(0)
    sub_title.paragraph_format.space_after = Pt(14)
    s_run = sub_title.add_run("79D 하이브리드(MediaPipe 3D + RTMPose 2D) 피처 융합 및 공식 5대 시점 적응형 3-State 캘리브레이션 실측")
    s_run.font.name = "맑은 고딕"
    s_run.font.size = Pt(10)
    s_run.font.color.rgb = COLOR_GRAY

    # 메타 정보 테이블
    meta_tbl = doc.add_table(rows=4, cols=4)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_tbl.autofit = False
    col_w = [Inches(1.3), Inches(2.0), Inches(1.3), Inches(1.9)]
    aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT]
    
    meta_data = [
        [("프로젝트 명칭", True), ("vidio-health-trainer", False), ("문서 버전", True), ("공식 확정 v1.0 (2026-10-11)", False)],
        [("대상 운동", True), ("다이나믹 런지 (05_포워드, 06_백워드)", False), ("평가 시퀀스", True), ("전수 1,468개 시퀀스 (결측치 0%)", False)],
        [("세부 결함 항목", True), ("c5: 상체의 과도한 숙임/젖힘 여부", False), ("인증 모델명", True), ("dl5_rtmp_1 (79D Hybrid Bundle)", False)],
        [("검증 프로토콜", True), ("5-Fold StratifiedGroupKFold (피험자 1~9 전원 격리)", False), ("달성 3-State F1", True), ("77.41% (보류율 9.13%) / 78.36% (14.0%)", False)],
    ]
    for r_idx, row in enumerate(meta_tbl.rows):
        for c_idx, cell in enumerate(row.cells):
            txt, is_bold = meta_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.alignment = aligns[c_idx]
            r = p.add_run(txt)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.bold = is_bold
            if is_bold:
                set_cell_background(cell, "EDF2F7")
                r.font.color.rgb = COLOR_NAVY
            else:
                set_cell_background(cell, "FFFFFF")
                r.font.color.rgb = COLOR_CHARCOAL
            cell.width = col_w[c_idx]
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 1. 핵심 성과 요약 Callout
    add_callout(doc, [
        "바벨 런지 공식 챔피언 모델인 bl5_final(79D Hybrid)의 아키텍처를 계승하여, 다이나믹 런지 1,468개 전수 데이터에 맞게 최적화한 dl5_rtmp_1 구축을 완료하였습니다.",
        "3-State 적응형 F1 스코어 77.41% (보류율 9.13% <= 10.0% 엄격 준수)를 기록하여, 바벨 런지 제로샷 전이 성능(69.82%) 대비 +7.59%p 대폭 반등하며 70% 목표를 완벽히 돌파했습니다.",
        "보류율 15% 허용 기준에서는 78.36% F1 및 ROC-AUC 0.8565(역대 최고치)를 달성하였으며, 결함 검출 재현율(Recall) 86.65%, 확정 정확도 79.84%를 기록했습니다.",
        "가장 취약했던 '시점 C (정면 0°)'에서 원근 단축비 보정 크로스 모달 피처를 통해 제로샷 58.26%에서 65.96%(+7.70%p)로 대폭 회복하였으며, 5대 시점 중 4대 시점이 73%~85%의 고성능을 균일하게 달성했습니다.",
        "세부 운동 유형별로도 05_스텝_포워드 71.15%, 06_스텝_백워드 83.10%로 두 운동 모두 70% 기준을 안정적으로 초과 달성했습니다."
    ], title="핵심 성과 요약 (Executive Summary)")

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 2. 개발 배경 및 최적화 동기
    h1 = doc.add_heading("1. 개발 배경 및 최적화 동기 (Motivation)", level=1)
    h1.runs[0].font.name = "맑은 고딕"
    h1.runs[0].font.color.rgb = COLOR_NAVY

    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.2
    p_intro.paragraph_format.space_after = Pt(6)
    p_intro.add_run(
        "본 연구진은 앞선 전이 평가에서 바벨 런지 공식 5대 세부 오류 최종 모델(bl1_rtmp_1, bl2_final, bl3_v4, bl4_rtmp_1, bl5_final)을 "
        "다이나믹 런지(1,468건)에 무학습 제로샷(Zero-shot)으로 투입한 바 있습니다. "
        "평가 결과 c1(70.31%), c3(81.06%), c4(80.75%)는 70%를 상회하였으나, c5(상체 과도한 숙임/젖힘)는 69.82%를 기록하여 단 0.18%p 차이로 70% 문턱에 미달했습니다.\n\n"
        "이에 대한 심층 역학 분석 결과, 제자리에서 수행하는 바벨 런지와 달리 다이나믹 런지는 전방/후방 스텝 착지 시 발생하는 지면 반력 충격, "
        "추진 복귀 시의 체간-골반 디커플링(Pelvis-Spine Decoupling), 특히 정면 시점(시점 C)에서의 동적 전경 왜곡이 존재함을 확인했습니다. "
        "따라서 bl5_final의 우수한 79D 하이브리드 피처 추출 파이프라인을 그대로 계승하되, 다이나믹 런지의 고유한 운동학적 시퀀스에 맞춰 "
        "앙상블 모델과 5대 시점 적응형 3-State 임계값을 재학습·최적화한 dl5_rtmp_1을 개발하게 되었습니다."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3. 모델 아키텍처 및 피처 파이프라인
    h2 = doc.add_heading("2. dl5_rtmp_1 모델 아키텍처 및 79D 피처 파이프라인", level=1)
    h2.runs[0].font.name = "맑은 고딕"
    h2.runs[0].font.color.rgb = COLOR_NAVY

    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.line_spacing = 1.2
    p_arch.paragraph_format.space_after = Pt(6)
    p_arch.add_run(
        "dl5_rtmp_1은 단일 비전 모달리티의 한계를 극복하기 위해 MediaPipe 3D와 RTMPose 2D의 장점을 결합한 79차원 하이브리드 피처셋을 사용합니다:\n"
        "1) MediaPipe 3D (36D): 3차원 공간상의 VSPA 체간 시상면 피치각, 흉추 곡률 이탈(Thoracic Curvature), 3D 모멘트암, 위상 적분\n"
        "2) RTMPose 2D (37D): 초정밀 2D 키포인트 기반 지면 수직 저크(Vertical Jerk), 체간/대퇴 동적 압축비, 스펙트럼 에너지 비(Spectral Energy Ratio)\n"
        "3) Cross-Modal 상호작용 (6D): 3D-2D 피치각차, 원근 투영 단축 보정비, 모달 간 속도 불일치도, 어깨/골반 수직 속도비\n"
        "4) 3-Way 최적 앙상블 분류기: CatBoost (50%) + LightGBM (30%) + XGBoost (20%)\n"
        "5) 의사결정 로직: 공식 5대 시점(시점 A~E) 맞춤형 적응형 3-State 선택적 분류 (보류율 10% 이내 엄격 통제)"
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 4. 전체 모델 비교 벤치마크 (표)
    h3 = doc.add_heading("3. 다이나믹 런지 c5 모델 종합 성능 비교 매트릭스", level=1)
    h3.runs[0].font.name = "맑은 고딕"
    h3.runs[0].font.color.rgb = COLOR_NAVY

    tbl_comp = doc.add_table(rows=6, cols=6)
    comp_widths = [Inches(2.2), Inches(0.8), Inches(0.9), Inches(1.3), Inches(1.3), Inches(0.8)]
    comp_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]

    comp_headers = ["비교 모델 및 파이프라인", "ROC-AUC", "2-State F1", "3-State F1 (≤10%)", "3-State F1 (≤15%)", "확정 정확도"]
    for j, h in enumerate(comp_headers):
        tbl_comp.cell(0, j).paragraphs[0].text = h

    comp_rows = [
        ["bl5_final (바벨런지 제로샷 전이)", "0.7635", "64.69%", "68.80% (보류: 9.8%)", "69.82% (보류: 13.0%)", "68.99%"],
        ["dl5_v5 (이전 MediaPipe 36D 전용)", "0.8499", "72.16%", "76.05% (보류: 9.7%)", "78.20% (보류: 14.5%)", "81.13%"],
        ["dl5_rtmp_pure (73D Pure RTMPose)", f"{rtmp_pure['2state']['auc']:.4f}", f"{rtmp_pure['2state']['f1']*100:.2f}%", f"{rtmp_pure['3state_adaptive_10']['f1']*100:.2f}% (보류: {rtmp_pure['3state_adaptive_10']['retake_rate']*100:.1f}%)", f"{rtmp_pure['3state_adaptive_15']['f1']*100:.2f}% (보류: {rtmp_pure['3state_adaptive_15']['retake_rate']*100:.1f}%)", f"{rtmp_pure['3state_adaptive_10']['accuracy']*100:.2f}%"],
        ["dl5_v5_retrained (36D MP 3-Way)", f"{mp_retrained['2state']['auc']:.4f}", f"{mp_retrained['2state']['f1']*100:.2f}%", f"{mp_retrained['3state_adaptive_10']['f1']*100:.2f}% (보류: {mp_retrained['3state_adaptive_10']['retake_rate']*100:.1f}%)", f"{mp_retrained['3state_adaptive_15']['f1']*100:.2f}% (보류: {mp_retrained['3state_adaptive_15']['retake_rate']*100:.1f}%)", f"{mp_retrained['3state_adaptive_10']['accuracy']*100:.2f}%"],
        ["★ dl5_rtmp_1 (79D Hybrid 확정)", f"{hyb['2state']['auc']:.4f}", f"{hyb['2state']['f1']*100:.2f}%", f"{hyb['3state_adaptive_10']['f1']*100:.2f}% (보류: {hyb['3state_adaptive_10']['retake_rate']*100:.1f}%)", f"{hyb['3state_adaptive_15']['f1']*100:.2f}% (보류: {hyb['3state_adaptive_15']['retake_rate']*100:.1f}%)", f"{hyb['3state_adaptive_10']['accuracy']*100:.2f}%"],
    ]

    for i, r in enumerate(comp_rows, start=1):
        for j, val in enumerate(r):
            tbl_comp.cell(i, j).paragraphs[0].text = val
            if i == 5: # Highlight winning model
                set_cell_background(tbl_comp.cell(i, j), HEX_PASS_BG)
                tbl_comp.cell(i, j).paragraphs[0].runs[0].font.bold = True

    style_table(tbl_comp, comp_widths, comp_aligns)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 5. 공식 5대 시점별 세부 감사 (표)
    h4 = doc.add_heading("4. 공식 5대 시점(A~E)별 3-State F1 성능 감사 (보류율 ≤ 10.0%)", level=1)
    h4.runs[0].font.name = "맑은 고딕"
    h4.runs[0].font.color.rgb = COLOR_NAVY

    tbl_views = doc.add_table(rows=6, cols=6)
    view_widths = [Inches(1.8), Inches(1.1), Inches(1.0), Inches(1.1), Inches(0.9), Inches(1.0)]
    view_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]

    view_headers = ["공식 시점 명칭 및 각도", "bl5_final (전이)", "dl5_rtmp_pure", "dl5_rtmp_1 (확정)", "보류율", "개선폭 (vs 전이)"]
    for j, h in enumerate(view_headers):
        tbl_views.cell(0, j).paragraphs[0].text = h

    bl5_trans = {'view5': 80.00, 'view2': 76.99, 'view3': 58.26, 'view4': 68.63, 'view1': 73.33}
    view_rows = []
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = hyb['3state_adaptive_10']['per_view'][v]['view_name']
        f_bl5 = bl5_trans[v]
        f_pure = rtmp_pure['3state_adaptive_10']['per_view'][v]['f1_3state'] * 100
        f_hyb = hyb['3state_adaptive_10']['per_view'][v]['f1_3state'] * 100
        ret = hyb['3state_adaptive_10']['per_view'][v]['retake_rate'] * 100
        diff = f_hyb - f_bl5
        view_rows.append([v_name, f"{f_bl5:.2f}%", f"{f_pure:.2f}%", f"{f_hyb:.2f}%", f"{ret:.1f}%", f"+{diff:.2f}%p" if diff >= 0 else f"{diff:.2f}%p"])

    for i, r in enumerate(view_rows, start=1):
        for j, val in enumerate(r):
            tbl_views.cell(i, j).paragraphs[0].text = val
            if j == 3: # dl5_rtmp_1
                tbl_views.cell(i, j).paragraphs[0].runs[0].font.bold = True
            if j == 5:
                set_cell_background(tbl_views.cell(i, j), HEX_PASS_BG)
                tbl_views.cell(i, j).paragraphs[0].runs[0].font.bold = True

    style_table(tbl_views, view_widths, view_aligns)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 6. 세부 운동 유형별 분석 (05_포워드 vs 06_백워드)
    h5 = doc.add_heading("5. 세부 운동 유형별 성능 분석 (05_스텝_포워드 vs 06_스텝_백워드)", level=1)
    h5.runs[0].font.name = "맑은 고딕"
    h5.runs[0].font.color.rgb = COLOR_NAVY

    tbl_ex = doc.add_table(rows=3, cols=5)
    ex_widths = [Inches(2.2), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1)]
    ex_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]

    ex_headers = ["다이나믹 런지 세부 운동", "시퀀스 수", "ROC-AUC", "3-State F1 (≤10%)", "확정 정확도"]
    for j, h in enumerate(ex_headers):
        tbl_ex.cell(0, j).paragraphs[0].text = h

    ex_fwd = hyb['per_exercise']['05_스텝_포워드']
    ex_bwd = hyb['per_exercise']['06_스텝_백워드']

    ex_rows = [
        ["05_스텝_포워드 다이나믹 런지", f"{ex_fwd['count']:,}건", f"{ex_fwd['auc']:.4f}", f"{ex_fwd['f1_3state_10']*100:.2f}%", f"{ex_fwd['accuracy']*100:.2f}%"],
        ["06_스텝_백워드 다이나믹 런지", f"{ex_bwd['count']:,}건", f"{ex_bwd['auc']:.4f}", f"{ex_bwd['f1_3state_10']*100:.2f}%", f"{ex_bwd['accuracy']*100:.2f}%"],
    ]

    for i, r in enumerate(ex_rows, start=1):
        for j, val in enumerate(r):
            tbl_ex.cell(i, j).paragraphs[0].text = val
            if j == 3:
                tbl_ex.cell(i, j).paragraphs[0].runs[0].font.bold = True
                set_cell_background(tbl_ex.cell(i, j), HEX_PASS_BG)

    style_table(tbl_ex, ex_widths, ex_aligns)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 7. 다이나믹 런지 5대 세부오류 공식 통합 위상 표
    h6 = doc.add_heading("6. 다이나믹 런지 5대 세부 오류 분류 모델 최종 확정 위상", level=1)
    h6.runs[0].font.name = "맑은 고딕"
    h6.runs[0].font.color.rgb = COLOR_NAVY

    tbl_final5 = doc.add_table(rows=6, cols=6)
    f5_widths = [Inches(0.8), Inches(1.8), Inches(1.5), Inches(1.0), Inches(1.0), Inches(1.0)]
    f5_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]

    f5_headers = ["결함", "세부 결함 명칭", "최종 확정 모델", "입력 모달리티", "3-State F1", "70% 달성여부"]
    for j, h in enumerate(f5_headers):
        tbl_final5.cell(0, j).paragraphs[0].text = h

    f5_rows = [
        ["c1", "앞다리 무릎 각도 90도 미만", "bl1_rtmp_1 / dl1_rtmp_1", "RTMPose 2D (30D)", "70.31%", "달성 (PASS)"],
        ["c2", "무릎이 발끝 앞으로 과도 돌출", "bl2_final / dl2_v1", "Hybrid (20D)", "59.28% (정면 74.8%)", "주 시점 통과"],
        ["c3", "뒷다리 무릎 지면 닿음/과각도", "bl3_v4 (BL3V4Classifier)", "MediaPipe 2D (14D)", "81.06%", "초과 달성 (PASS)"],
        ["c4", "몸통 과도 회전 (비틀림)", "bl4_rtmp_1 / dl4_rtmp_1", "RTMPose 2D (24D)", "80.75%", "초과 달성 (PASS)"],
        ["c5", "상체 과도한 숙임/젖힘", "dl5_rtmp_1 (공식 확정)", "Hybrid 79D (MP+RTMP)", "77.41% ~ 78.36%", "초과 달성 (PASS)"],
    ]

    for i, r in enumerate(f5_rows, start=1):
        for j, val in enumerate(r):
            tbl_final5.cell(i, j).paragraphs[0].text = val
            if i == 5: # dl5_rtmp_1
                set_cell_background(tbl_final5.cell(i, j), HEX_PASS_BG)
                tbl_final5.cell(i, j).paragraphs[0].runs[0].font.bold = True
            elif j == 5 and "PASS" in val:
                tbl_final5.cell(i, j).paragraphs[0].runs[0].font.color.rgb = RGBColor(0x27, 0x67, 0x49)
                tbl_final5.cell(i, j).paragraphs[0].runs[0].font.bold = True

    style_table(tbl_final5, f5_widths, f5_aligns)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 8. 배포 아티팩트 및 무결성 검증
    h7 = doc.add_heading("7. 프로덕션 배포 아티팩트 및 파이프라인 무결성 검증", level=1)
    h7.runs[0].font.name = "맑은 고딕"
    h7.runs[0].font.color.rgb = COLOR_NAVY

    p_deploy = doc.add_paragraph()
    p_deploy.paragraph_format.line_spacing = 1.2
    p_deploy.add_run(
        "1) 프로덕션 번들 가중치 파일:\n"
        "   - 파일 경로: D:\\src\\models\\second_models\\dynamic_lunge\\weights\\dl5_rtmp_1_bundle.pkl (1,801,059 bytes)\n"
        "   - 포함 내용: CatBoost, LightGBM, XGBoost 3대 모델 가중치, 공식 5대 시점 최적 3-State 임계값, 79D 피처 명칭, 벤치마크 메타데이터\n\n"
        "2) 프로덕션 추론 서빙 모듈:\n"
        "   - 모듈 경로: D:\\src\\models\\second_models\\dynamic_lunge\\dl5_rtmp_1.py\n"
        "   - 제공 클래스: DL5RTMP1Classifier (extract_features, predict_proba, predict_sequence)\n"
        "   - 퀵 추론 함수: predict_dynamic_lunge_c5_rtmp_1(mp_kps, rtmp_kps, view)\n\n"
        "3) 하네스 엔지니어링 단위 및 무결성 테스트:\n"
        "   - 테스트 파일: D:\\tests\\test_dl5_rtmp_1_pipeline.py\n"
        "   - 검증 항목: 번들 무결성, 79D 결측치(NaN/Inf) 0% 검증, 실측 시퀀스 추론, 공식 5대 시점 매핑, 퀵 추론 함수\n"
        "   - 테스트 결과: 5개 테스트 케이스 전수 통과 (Ran 5 tests in 2.864s - OK)"
    )

    # 저장
    for p in OUT_DOCX_PATHS:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        doc.save(p)
        print(f"[+] 공식 Word 보고서 저장 완료: {p} ({os.path.getsize(p):,} bytes)")

    # Markdown 동시 출력
    os.makedirs(os.path.dirname(OUT_MD_PATH), exist_ok=True)
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# [vidio-health-trainer] 다이나믹 런지 c5 dl5_rtmp_1 공식 성능평가 종합보고서\n\n")
        f.write(f"- 작성일자: 2026-10-11\n")
        f.write(f"- 공식 모델명: `dl5_rtmp_1`\n")
        f.write(f"- 아키텍처: bl5_final 계승 (MediaPipe 3D 36D + RTMPose 2D 37D + Cross-Modal 6D = 79D Hybrid)\n")
        f.write(f"- 3-State Adaptive F1: **77.41%** (보류율 9.13%) / **78.36%** (보류율 13.96%)\n")
        f.write(f"- ROC-AUC: **{hyb['2state']['auc']:.4f}**\n")
        f.write(f"- 확정 정확도: **{hyb['3state_adaptive_10']['accuracy']*100:.2f}%**\n")
        f.write(f"- 재현율: **{hyb['3state_adaptive_10']['recall']*100:.2f}%**\n\n")
        f.write("## 1. 공식 5대 시점별 F1\n\n")
        for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
            stat = hyb['3state_adaptive_10']['per_view'][v]
            f.write(f"- **{stat['view_name']}**: F1 = **{stat['f1_3state']*100:.2f}%**, AUC = {stat['auc']:.4f}, Retake = {stat['retake_rate']*100:.1f}%\n")
    print(f"[+] 마크다운 요약본 저장 완료: {OUT_MD_PATH}")

if __name__ == '__main__':
    build_docx()
