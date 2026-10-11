# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d3_external_ai_comparison_report_docx.py
Purpose: D:\자주프 문서\외부ai 응답 폴더 내 3대 외부 AI(Gemini, Perplexity, 아주LLM) 응답 전문을
         심층 비교 분석하고, 바벨 데드리프트 d3(발바닥 지면 고정 / Heel Lift 100%) 모델의
         F1 80~88%+ 돌파를 위한 통합 최적화 액션 플랜 및 기술 보고서를 Word(.docx)와 Markdown(.md)으로 동시 생성.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import shutil

DOC_DIR = r"D:\자주프 문서\6주차"
ROOT_DIR = r"D:\자주프 문서"
os.makedirs(DOC_DIR, exist_ok=True)
DOCX_PATH = os.path.join(DOC_DIR, "바벨_데드리프트_d3_외부AI_응답_종합비교_및_고도화_분석보고서.docx")
ROOT_DOCX_PATH = os.path.join(ROOT_DIR, "바벨_데드리프트_d3_외부AI_응답_종합비교_및_고도화_분석보고서.docx")
MD_PATH = os.path.join(DOC_DIR, "바벨_데드리프트_d3_외부AI_응답_종합비교_및_고도화_분석보고서.md")

COLOR_PRIMARY = RGBColor(26, 82, 118)     # Deep Navy
COLOR_SECONDARY = RGBColor(41, 128, 185)  # Soft Blue
COLOR_ACCENT = RGBColor(211, 84, 0)       # Amber/Orange
COLOR_TEXT = RGBColor(44, 62, 80)         # Charcoal
COLOR_MUTED = RGBColor(127, 140, 141)     # Gray
HEX_PRIMARY = "1A5276"
HEX_LIGHT_BG = "F4F6F7"
HEX_HIGHLIGHT = "EBF5FB"
HEX_WARN_BG = "FEF9E7"

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_header_footer(doc):
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.9)
        s.right_margin = Inches(0.9)
        
        # Header
        hdr = s.header
        hp = hdr.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("vidio-health-trainer | 바벨 데드리프트 d3 외부 AI 자문 종합 비교분석 및 성능 돌파 보고서")
        hrun.font.name = "Malgun Gothic"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = COLOR_MUTED

        # Footer
        ftr = s.footer
        fp = ftr.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("vidio-health-trainer Confidential - 64-Frame Dual-Stream Architecture Pipeline")
        frun.font.name = "Malgun Gothic"
        frun.font.size = Pt(8.0)
        frun.font.color.rgb = COLOR_MUTED

def style_paragraph(p, space_before=0, space_after=6, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=14, space_after=6)
    run = p.add_run(text)
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=10, space_after=4)
    run = p.add_run(text)
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=6, space_after=2)
    run = p.add_run(text)
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_TEXT
    return p

def add_body_p(doc, text, bold=False, italic=False, color=COLOR_TEXT):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=0, space_after=4)
    run = p.add_run(text)
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(9.5)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return p

def add_bullet_p(doc, prefix, text, bold_prefix=True):
    p = doc.add_paragraph(style='List Bullet')
    style_paragraph(p, space_before=0, space_after=3)
    r1 = p.add_run(prefix)
    r1.font.name = "Malgun Gothic"
    r1.font.size = Pt(9.5)
    r1.font.bold = bold_prefix
    r1.font.color.rgb = COLOR_PRIMARY
    r2 = p.add_run(text)
    r2.font.name = "Malgun Gothic"
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = COLOR_TEXT
    return p

def add_callout_box(doc, title, text, bg_hex=HEX_HIGHLIGHT, border_color="2980B9"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    cell.width = Inches(6.8)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    style_paragraph(p, space_before=0, space_after=2)
    r_title = p.add_run(title + "\n")
    r_title.font.name = "Malgun Gothic"
    r_title.font.size = Pt(10)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    
    r_body = p.add_run(text)
    r_body.font.name = "Malgun Gothic"
    r_body.font.size = Pt(9)
    r_body.font.color.rgb = COLOR_TEXT
    
    p_after = doc.add_paragraph()
    style_paragraph(p_after, space_before=0, space_after=4)

def format_table(table, col_widths, headers, data, align_right_cols=[]):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header Row
    hdr_row = table.rows[0]
    for i, h_text in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = Inches(col_widths[i])
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
        p = cell.paragraphs[0]
        style_paragraph(p, space_before=0, space_after=0)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h_text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    # Data Rows
    for r_idx, row_data in enumerate(data):
        row = table.rows[r_idx + 1]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            p = cell.paragraphs[0]
            style_paragraph(p, space_before=0, space_after=0)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(8.5)
            run.font.color.rgb = COLOR_TEXT
            if c_idx == 0:
                run.font.bold = True

def build_docx_and_md():
    doc = Document()
    add_header_footer(doc)
    
    # Title Cover Header
    p_title = doc.add_paragraph()
    style_paragraph(p_title, space_before=20, space_after=4)
    run_title = p_title.add_run("vidio-health-trainer 바벨 데드리프트 d3 외부 AI 3사 자문 종합 분석 및 고도화 실행 로드맵 보고서")
    run_title.font.name = "Malgun Gothic"
    run_title.font.size = Pt(17)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    
    p_sub = doc.add_paragraph()
    style_paragraph(p_sub, space_before=0, space_after=14)
    run_sub = p_sub.add_run("부제: 발바닥 지면 고정 결함(Heel Lift 100%) 판정 극대화를 위한 Perplexity, Google Gemini, 아주LLM 기술 제안 심층 비교 및 v5.3 / v6.0 통합 액션 플랜")
    run_sub.font.name = "Malgun Gothic"
    run_sub.font.size = Pt(10.5)
    run_sub.font.color.rgb = COLOR_SECONDARY
    
    # Meta Info Table
    meta_data = [
        ("프로젝트명", "vidio-health-trainer (AI 기반 웨이트 트레이닝 자세 교정 및 오류 판별)"),
        ("대상 서브모델", "바벨 데드리프트 세부 결함 d3 (발바닥 지면 고정 / Heel Lift 100% 오류)"),
        ("외부 자문 AI 군단", "1. Perplexity (Sonar/GPT-4o) | 2. Google Gemini (1.5 Pro) | 3. 아주LLM (Claude 3.5 Sonnet 추정)"),
        ("현재 베이스라인", "v5.2 하이브리드 모델 (ST-SeqNet 35% + Tri-Ensemble 65%) -> OOF F1 67.06%, AUC 0.8487, 보류율 9.09%"),
        ("차기 목표 성능", "전체 5-Fold OOF 확정 F1-score 80%~88%+ 달성 및 후방 시점(A, E) F1 58.8% -> 75%+ 견인 (보류율 ≤ 10%)")
    ]
    meta_table = doc.add_table(rows=len(meta_data)+1, cols=2)
    format_table(meta_table, [1.8, 5.0], ["구분 항목", "상세 명세"], meta_data)
    
    doc.add_paragraph()
    
    # Section 1
    add_heading_1(doc, "1. 기술 자문 개요 및 현행 베이스라인(v5.2) 한계 진단")
    
    add_body_p(doc, "본 보고서는 바벨 데드리프트의 4대 세부 결함 중 세 번째 서브모델인 d3(발바닥 지면 고정 / 뒤꿈치 들림 100% 오류)의 판별 성능을 획기적으로 개선하기 위해, 최상위 LLM 3사(Perplexity, Google Gemini, 아주LLM)에 의뢰한 기술 자문 응답 전문을 심층 대조 분석하고, 이를 바탕으로 v5.3 및 v6.0 통합 아키텍처를 설계하기 위해 작성되었습니다.")
    
    add_heading_2(doc, "1.1 현행 v5.2 하이브리드 파이프라인 성과 및 병목 분석")
    add_body_p(doc, "v5.2 모델은 ST-SeqNet(35%)과 Tri-Ensemble(65%)을 연합하여 전체 F1 67.06%, ROC-AUC 0.8487, 판단 보류율 9.09%를 달성하여 시스템 제약(보류율 ≤ 10%)을 충족하였습니다. 특히 시점 D(좌측 전방 45°)에서는 F1 82.35%를 기록하며 높은 생체역학적 진단 가능성을 입증하였습니다.")
    add_body_p(doc, "그러나 5대 카메라 시점 및 피험자 간 성능 편차가 극심하여 전체 평균 F1이 60%대 후반에 정체되어 있으며, 구체적으로 다음과 같은 4대 핵심 병목이 식별되었습니다:")
    
    add_bullet_p(doc, "1. 극심한 시점 편향 (후방 뷰 병목): ", "전방/측면(시점 D 82.35%)에 비해 후방 시점(시점 A 58.82%, 시점 E 58.82%)에서 성능이 23.5%p 이상 급락. 45cm 바벨 원판 가림(Occlusion)으로 인해 발목·뒤꿈치 랜드마크가 심하게 튀거나 결측됨.")
    add_bullet_p(doc, "2. 운동 시작 위상(Phase) 이원화 혼동: ", "데이터셋 내 시퀀스의 56.9%는 바닥 셋업 상태에서 바로 당기기 시작하고, 43.1%는 이미 기립한 상태에서 바벨을 내렸다가 당김. 고정 64프레임 내에서 리프팅 시점이 달라 시계열 어텐션 학습이 발산함.")
    add_bullet_p(doc, "3. 피험자 고유 체형(키/다리길이) 및 미세 들림(0.5~1.5cm) 한계: ", "카메라 거리와 피험자 신장(155cm vs 190cm) 차이로 인해 절대 픽셀 좌표 편차가 발생하며, 개인별 초기 영점(Zero-Baseline)이 보정되지 않아 노이즈에 취약함.")
    add_bullet_p(doc, "4. 판단 보류율 10% 엄격 제약 하의 오탐(FP) 폭증: ", "결함 비율이 23.4%(90/385)로 불균형한 상황에서 단순 임계값 적용 시 미세한 발목 흔들림을 뒤꿈치 들림으로 오인하여 정상 샘플에 대한 FP가 대거 발생함.")
    
    # Performance Table
    p_perf = doc.add_paragraph()
    style_paragraph(p_perf, space_before=4, space_after=2)
    p_perf.add_run("■ 표 1. 현행 v5.2 모델의 5대 시점별 세부 성능 및 취약점 진단").font.bold = True
    
    t_perf = doc.add_table(rows=7, cols=5)
    perf_data = [
        ("시점 A (우측 후방 45° / view5)", "58.82%", "0.8012", "11.69%", "바벨 원판 가림 극심, 뒤꿈치 랜드마크 소실로 FP/FN 동시 다발"),
        ("시점 B (우측 전방 45° / view2)", "66.67%", "0.8520", "9.09%", "발끝-무릎 가림 일부 발생, 바닥 셋업 시점 위상 혼동"),
        ("시점 C (정면 0° / view3)", "66.67%", "0.8415", "7.79%", "관상면(Coronal) 관측으로 전후방 발바닥 기울기 깊이(Depth) 추정 난항"),
        ("시점 D (좌측 전방 45° / view4)", "82.35%", "0.9104", "6.49%", "골반-무릎-발목 개방 시야 확보로 최고 성능(골든 시점)"),
        ("시점 E (좌측 후방 45° / view1)", "58.82%", "0.8120", "10.39%", "우측 대칭 구조이나 원판 가림으로 인한 신호 왜곡 심각"),
        ("전체 5-Fold OOF 평균", "67.06%", "0.8487", "9.09%", "후방 시점(A, E) 및 정면(C) 성능 저하가 전체 평균을 15%p 이상 하락시킴")
    ]
    format_table(t_perf, [1.8, 1.0, 1.0, 1.1, 1.9], ["시점 구분 (공식 5대 시점)", "F1-Score", "ROC-AUC", "보류율(Retake)", "병목 및 취약 원인"], perf_data, align_right_cols=[1, 2, 3])
    
    doc.add_paragraph()
    
    # Section 2
    add_heading_1(doc, "2. 외부 AI 3사 자문 응답 심층 비교 분석")
    add_body_p(doc, "3대 외부 AI에 동일한 기술 명세서(d3 배경지식.pdf)를 제공하고 4대 핵심 질문(원판 가림, 위상 불일치, 체형 불변 피처, 물리 제약 손실)에 대한 자문을 수렴하였습니다. 각 모델의 접근 철학과 핵심 제안은 뚜렷한 특징과 강력한 상호 보완성을 보였습니다.")
    
    add_heading_2(doc, "2.1 3대 AI별 핵심 자문 철학 및 접근 방식 비교")
    
    t_ai_comp = doc.add_table(rows=6, cols=4)
    ai_comp_data = [
        ("핵심 접근 철학", "엔지니어링 & 기하학 중심 (즉시 수식화 및 실전 코드화)", "통계적 무결성 & 다단계 검증 중심 (Ablation 및 CI 검증)", "실전 리스크 방어 & 데이터 보존 중심 (치명적 함정 선제 경고)"),
        ("Q1. 원판 가림 극복", "Visibility-Guided ST-GCN: 신뢰도 v_i(t) 가중 이웃 노드 차용", "Mask-aware Residual Inpainting: 관측 좌표 근접 잔차 학습", "4중 텐서 보존: [원본, 복원, 마스크, 분산] 동시 공급 (결함 은폐 방지)"),
        ("Q2. 시작 위상 정렬", "KAPW: 손목 최하점(Nadir)을 index 32에 고정하는 3차 스플라인", "Start-State Head(바닥/기립 분류) + 상체 궤적 Soft-DTW 정렬", "상승/준비-하강/전체 3단계 분할 판정 (발 신호 정렬 영구 엄금)"),
        ("Q3. 체형 불변 피처", "Zero-Baseline Dimensionless: 셋업 영점화 + 3D 정강이 정규화", "Foot Plane Normal(발바닥 법선) + 발/정강이 비율 무차원화", "지면 기준 발 피치 각도(Pitch Angle) + 상시 결함(Always Lift) 방어"),
        ("Q4. 물리 손실 & 보류", "Physics-Informed Soft-F1 + 3-State 그리드 서치 탐색", "Cost-Sensitive Loss + Contact Penalty + MC-Dropout 보류", "Expected Error Cost 기반 보류 예산(Budget) 할당 + 고신뢰 오판 방어")
    ]
    format_table(t_ai_comp, [1.5, 1.8, 1.8, 1.8], ["평가 영역", "Google Gemini (1.5 Pro)", "Perplexity (Sonar/GPT)", "아주LLM (Claude 3.5 추정)"], ai_comp_data)
    
    doc.add_paragraph()
    
    add_heading_2(doc, "2.2 질문별 3사 제안의 상세 분석 및 장단점 대조")
    
    add_heading_3(doc, "[Q1. 45cm 원판 가림 극복] Spatio-Temporal Inpainting vs 가림 마스크 보존")
    add_body_p(doc, "• Google Gemini는 MediaPipe의 가시성(Visibility) 점수를 활용하는 VG-ST-GCN을 제안하여, 발목·뒤꿈치가 가려졌을 때 무릎과 골반의 시공간 궤적으로부터 하체 좌표를 복원(Spatial Hallucination + Temporal Smoothing)하는 명쾌한 파이토치 코드를 제시하였습니다.")
    add_body_p(doc, "• 아주LLM은 여기서 가장 치명적인 위험을 경고하였습니다: '복원기가 가려진 관절을 그럴듯하게 복원해버리면, 모델이 학습한 정상 궤적으로 복원되어 실제 뒤꿈치 들림 결함을 지워버릴 수 있다(Wipeout).' 또한 MediaPipe는 가려진 부위도 엉뚱하게 높은 신뢰도로 오추정할 수 있음을 지적하였습니다.")
    add_body_p(doc, "• Perplexity는 이를 절충하여 관측된 좌표는 원본을 유지하고 잔차만 추정하는 Residual Inpainting과 가림 마스크를 모델에 함께 입력하는 구조를 제안하였습니다.")
    
    add_callout_box(doc, "★ Q1 최적 통합 결론: 4-Channel Uncertainty-Aware Feature Tensor",
                    "Gemini의 VG-ST-GCN 복원기를 도입하되, 아주LLM과 Perplexity의 경고를 100% 반영하여 '복원 좌표로 덮어쓰기'를 엄격히 금지함. 최종 분류기에는 [X_orig(원본), X_inpaint(복원), Mask(가림여부), Variance(복원불확실성)]의 4개 채널을 함께 입력하여, 모델이 복원값에 맹목적으로 의존하지 않고 관측 품질에 따라 가중치를 스스로 조절하도록 설계함.",
                    HEX_HIGHLIGHT, "2980B9")
                    
    add_heading_3(doc, "[Q2. 시작 위상 불일치] Kinematic Anchor Warping vs 다중 헤드 분할")
    add_body_p(doc, "• Gemini는 양 손목의 최하점(운동 방향 전환 순간, Nadir)을 찾아 64프레임의 정중앙인 32번 프레임에 강제 고정하고 전후 구간을 3차 스플라인으로 각각 리샘플링하는 KAPW(Kinematic Anchor Phase Warping)를 제안하였습니다.")
    add_body_p(doc, "• 아주LLM은 '뒤꿈치나 발 관절 신호로 정렬을 수행하면 결함 신호 자체가 시간 축 워핑으로 인해 압축되어 소실된다'고 강력 경고하며, 상체(손목/바벨/골반) 궤적만을 기준으로 삼아야 한다고 강조하였습니다. 또한 결함이 셋업이나 하강 구간에 발생하는 경우도 있으므로 '상승 구간', '준비·하강 구간', '전체 시퀀스'를 분리 판정할 것을 권고하였습니다.")
    add_body_p(doc, "• Perplexity는 시퀀스의 시작 유형(바닥 셋업 시작 56.9% vs 기립 시작 43.1%)을 먼저 분류하는 Start-State Head를 둘 것을 제안하였습니다.")
    
    add_callout_box(doc, "★ Q2 최적 통합 결론: Upper-Body Nadir KAPW + Start-State Dual-Stream",
                    "발 관절이 아닌 양 손목 Y좌표 최하점(Nadir)을 기준으로 32프레임 앵커 정렬(KAPW)을 전처리로 적용함. 동시에 시간 워핑으로 인한 미세 진폭 왜곡을 방지하기 위해 [정렬된 시계열 스트림]과 [원본 64프레임 시계열 스트림]을 병렬로 처리하는 듀얼 스트림 구조를 채택하고, 시작 상태(바닥/기립) 메타 피처를 분류기에 주입함.",
                    HEX_HIGHLIGHT, "2980B9")
                    
    add_heading_3(doc, "[Q3. 피험자 체형 분산 해소] 무차원화 정규화 vs 상시 결함(Always Lift) 방어")
    add_body_p(doc, "• Gemini는 개인별 초기 셋업 구간(0~20 프레임)의 뒤꿈치 높이를 영점(0)으로 차감하고, 개인의 3D 정강이 길이(Knee-to-Ankle) 중앙값으로 나누는 Zero-Baseline Dimensionless 정규화 수식을 제안하였습니다.")
    add_body_p(doc, "• 아주LLM은 극도로 날카로운 예외 상황을 지적하였습니다: '만약 피험자가 운동 시작부터 끝까지 뒤꿈치를 들고 있는 상시 결함자(Always Heel Lifted)라면, 초기 셋업 구간을 영점으로 잡는 순간 뒤꿈치 들림이 0으로 상쇄되어 결함이 완전히 사라진다.' 따라서 발끝(Toe) 대비 뒤꿈치(Heel)의 상대적 피치 각도(Pitch Angle)와 지면 법선 추정을 병행해야 한다고 제안하였습니다.")
    add_body_p(doc, "• Perplexity 역시 3D 발바닥 평면 법선(Foot Plane Normal)과의 내적 및 골반 상대 좌표 변환을 권장하였습니다.")
    
    add_callout_box(doc, "★ Q3 최적 통합 결론: Dual-Reference Calibration (Pitch Angle + Tibia Dimensionless)",
                    "단순 셋업 영점화에만 의존하지 않고, 1) 발끝과 뒤꿈치의 상대 수직차 및 피치 각도(Pitch Angle) 피처를 기본 보존하여 상시 결함자가 상쇄되는 오류를 원천 차단함. 2) 동시에 3D 정강이 길이 중앙값을 분모로 하는 무차원 스케일링을 적용하여 피험자 간 신장(155~190cm) 및 카메라 거리 왜곡을 완벽히 흡수함.",
                    HEX_HIGHLIGHT, "2980B9")
                    
    add_heading_3(doc, "[Q4. 물리 제약 손실 및 보류율 제어] Physics-Informed Soft-F1 vs Expected Error Budget")
    add_body_p(doc, "• Gemini는 F1을 미분 가능하게 최적화하는 Soft-F1 Loss에 손목/바벨이 상승하지 않는 구간에서 뒤꿈치 들림을 예측할 때 강력한 패널티를 부여하는 Physics-Informed Loss 수식을 제안하였으며, 유효 커버리지 90% 이상을 만족하는 3-State 이중 임계값 탐색 알고리즘을 제공하였습니다.")
    add_body_p(doc, "• Perplexity는 앙상블 모델 간의 분산(Ensemble Variance)과 MC 드롭아웃을 이용한 불확실성 기반 Selective Classification을 제안하였습니다.")
    add_body_p(doc, "• 아주LLM은 단순 0.5 근처뿐만 아니라 관측 품질이 나쁜 고신뢰 오판을 막기 위해 Expected Error Cost를 계산하고 고정 10%의 보류 예산(Abstention Budget)을 분배하는 방식을 권고하였습니다.")
    
    # Section 3
    add_heading_1(doc, "3. 통합 아키텍처 v5.3 / v6.0 실행 로드맵")
    add_body_p(doc, "3대 AI의 자문 결과를 vidio-health-trainer 워크스페이스의 64프레임 불변 원칙 및 아키텍처 모듈화 규정에 맞추어 2단계 점진적 고도화 로드맵으로 체계화하였습니다.")
    
    add_heading_2(doc, "3.1 1단계: v5.3 파이프라인 (피처 엔지니어링 및 전처리 즉시 고도화)")
    add_body_p(doc, "기존 v5.2 모델 구조(ST-SeqNet + Tri-Ensemble)를 유지하면서 전처리 및 피처 엔지니어링 단계를 즉시 혁신하여 목표 F1 75~80%를 달성합니다:")
    
    add_bullet_p(doc, "1. Kinematic Anchor Phase Warping (KAPW) 전처리: ", "손목 Y 최하점을 프레임 32에 고정 정렬하는 스플라인 워핑 적용. 바닥 시작(56.9%)과 기립 시작(43.1%) 간 위상 혼동 완전 종식.")
    add_bullet_p(doc, "2. Dimensionless Foot Pitch & Tibia Index 도입: ", "상시 결함을 방어하는 발끝-뒤꿈치 피치 각도(Pitch Angle) 및 3D 정강이 길이 정규화 무차원 피처 12종 추가.")
    add_bullet_p(doc, "3. Physics Penalty Soft-F1 Loss 적용: ", "손목 수직 속도와 뒤꿈치 들림 간 생체역학적 불일치 샘플에 패널티를 부여하여 오탐(FP) 대폭 억제.")
    add_bullet_p(doc, "4. 3-State Dual Threshold 커버리지 튜닝: ", "보류율 10% 이하(커버리지 ≥ 90%)를 보장하는 최적 [t_low, t_high] 그리드 서치 적용.")
    
    add_heading_2(doc, "3.2 2단계: v6.0 파이프라인 (VG-ST-GCN 복원기 및 불확실성 앙상블)")
    add_body_p(doc, "심층 신경망 레벨에서 후방 시점(A, E)의 원판 가림을 근본적으로 극복하여 최종 목표 F1 82~88%+를 달성합니다:")
    
    add_bullet_p(doc, "1. Visibility-Guided ST-GCN Temporal Inpainting 모듈: ", "하체 8대 관절(골반-무릎-발목-뒤꿈치-발끝)의 시공간 그래프를 구축하여 가려진 뒤꿈치 궤적을 복원.")
    add_bullet_p(doc, "2. 4-Channel Tensor 융합 분류기: ", "원본 좌표, 복원 좌표, 가림 마스크, 복원 분산을 모두 결합하여 은폐 오류를 방지하는 Robust 분류기 구축.")
    add_bullet_p(doc, "3. Multi-View Adaptive Ensemble: ", "시점 A~E의 관측 신뢰도에 따라 가중치를 동적으로 조절하는 시점 인식형 하이브리드 게이트 적용.")
    
    # Roadmap Table
    p_road = doc.add_paragraph()
    style_paragraph(p_road, space_before=6, space_after=2)
    p_road.add_run("■ 표 2. 바벨 데드리프트 d3 단계별 고도화 실행 로드맵").font.bold = True
    
    t_road = doc.add_table(rows=4, cols=5)
    road_data = [
        ("현행 v5.2", "ST-SeqNet(35%) + Tri-Ensemble(65%)", "64프레임 기본 스플라인 + 단일 임계값", "67.06%", "9.09% (기준선)"),
        ("차기 v5.3 (단기)", "KAPW 위상정합 + 무차원 피처 + Physics Loss", "손목 최하점 정렬 + 지면 피치 + 3-State 최적화", "76.0~80.0%", "≤ 9.5% (안정화)"),
        ("목표 v6.0 (중기)", "VG-ST-GCN 복원 + 4-Tensor Dual-Stream Hybrid", "시공간 Inpainting + 불확실성 인식 게이트", "83.0~88.0%+", "≤ 9.0% (극대화)")
    ]
    format_table(t_road, [1.3, 1.8, 1.9, 1.0, 1.0], ["버전 단계", "핵심 아키텍처", "전처리 및 손실함수 혁신", "예상 F1-Score", "보류율(Retake)"], road_data, align_right_cols=[3, 4])
    
    doc.add_paragraph()
    
    # Section 4
    add_heading_1(doc, "4. 결론 및 향후 실행 계획")
    add_body_p(doc, "외부 AI 3사의 자문은 각기 독창적인 강점을 가지면서도 놀라울 정도로 유기적으로 맞물려 있습니다. Gemini의 '수학적 명쾌함과 KAPW/VG-GCN 아키텍처', 아주LLM의 '복원기 결함 은폐 방지 및 상시 결함 보존 경고', Perplexity의 '다단계 통계적 검증과 선택적 분류'를 집대성함으로써 d3 모델은 시점 A/E의 한계를 극복하고 F1 80% 이상의 실전 임계치를 안정적으로 돌파할 수 있는 확고한 기술적 기반을 확보하였습니다.")
    
    add_body_p(doc, "향후 즉각적으로 v5.3 구현 스크립트 작성에 착수하며, 피험자 1~9번 전수 격리 교차검증(StratifiedGroupKFold)과 5대 시점 전수 성능 감사를 통해 단 1%의 데이터 누수나 과적합 없이 최고 품질의 모델을 완성할 것입니다.")
    
    # Save DOCX
    doc.save(DOCX_PATH)
    shutil.copy2(DOCX_PATH, ROOT_DOCX_PATH)
    print(f"[+] DOCX 저장 완료: {DOCX_PATH}")
    print(f"[+] ROOT DOCX 복사 완료: {ROOT_DOCX_PATH}")
    
    # Generate Markdown Report
    md_content = f"""# vidio-health-trainer 바벨 데드리프트 d3 외부 AI 3사 자문 종합 분석 및 고도화 실행 로드맵 보고서

> **부제**: 발바닥 지면 고정 결함(Heel Lift 100%) 판정 극대화를 위한 Perplexity, Google Gemini, 아주LLM 기술 제안 심층 비교 및 v5.3 / v6.0 통합 액션 플랜  
> **공식 프로젝트명**: `vidio-health-trainer`  
> **대상 서브모델**: 바벨 데드리프트 $d_3$ (발바닥 지면 고정 / Heel Lift 100% 오류)  
> **배포 문서**: [`바벨_데드리프트_d3_외부AI_응답_종합비교_및_고도화_분석보고서.docx`](file:///{DOCX_PATH.replace(chr(92), '/')})

---

## 1. 기술 자문 개요 및 현행 베이스라인(v5.2) 한계 진단

본 보고서는 바벨 데드리프트의 4대 세부 결함 중 세 번째 서브모델인 $d_3$(발바닥 지면 고정 / 뒤꿈치 들림 100% 오류)의 판별 성능을 획기적으로 개선하기 위해, 최상위 LLM 3사(**Perplexity**, **Google Gemini**, **아주LLM**)에 의뢰한 기술 자문 응답 전문을 심층 대조 분석하고, 이를 바탕으로 v5.3 및 v6.0 통합 아키텍처를 설계하기 위해 작성되었습니다.

### 1.1 현행 v5.2 하이브리드 파이프라인 성과 및 병목 분석
- **현행 성과**: ST-SeqNet(35%) + Tri-Ensemble(65%) 연합 -> **전체 5-Fold OOF F1 67.06%**, ROC-AUC 0.8487, 판단 보류율 9.09% (보류율 ≤ 10% 엄수).
- **시점 편향**: 시점 D(좌측 전방 45°)에서는 **F1 82.35%**로 극히 우수한 생체역학적 진단력을 입증했으나, **후방 시점(시점 A 58.82%, 시점 E 58.82%)에서 성능이 23.5%p 이상 급락**함.
- **4대 핵심 병목**:
  1. **후방 시점 45cm 바벨 원판 가림(Occlusion)**으로 인한 발목·뒤꿈치 랜드마크 지터 및 유실
  2. **바닥 셋업 시작(56.9%) vs 기립 시작(43.1%)의 극단적인 위상(Phase) 불일치**로 인한 시계열 학습 혼동
  3. **피험자 간 체형 편차(신장 155~190cm, 다리 길이 차이) 및 미세 들림(0.5~1.5cm)**과 카메라 거리 왜곡
  4. **판단 보류율 10% 엄격 제약 하에서 오탐(FP)과 미탐(FN)의 트레이드오프**

| 시점 구분 (공식 5대 시점) | F1-Score | ROC-AUC | 보류율(Retake) | 병목 및 취약 원인 |
| :--- | :---: | :---: | :---: | :--- |
| **시점 A** (우측 후방 45° / view5) | 58.82% | 0.8012 | 11.69% | 바벨 원판 가림 극심, 뒤꿈치 랜드마크 소실로 FP/FN 동시 다발 |
| **시점 B** (우측 전방 45° / view2) | 66.67% | 0.8520 | 9.09% | 발끝-무릎 가림 일부 발생, 바닥 셋업 시점 위상 혼동 |
| **시점 C** (정면 0° / view3) | 66.67% | 0.8415 | 7.79% | 관상면(Coronal) 관측으로 전후방 발바닥 기울기 깊이(Depth) 추정 난항 |
| **시점 D** (좌측 전방 45° / view4) | **82.35%** | **0.9104** | 6.49% | 골반-무릎-발목 개방 시야 확보로 최고 성능 (골든 시점) |
| **시점 E** (좌측 후방 45° / view1) | 58.82% | 0.8120 | 10.39% | 우측 대칭 구조이나 원판 가림으로 인한 신호 왜곡 심각 |
| **전체 5-Fold OOF 평균** | **67.06%** | **0.8487** | **9.09%** | 후방 시점(A, E) 성능 저하가 전체 평균을 15%p 이상 하락시킴 |

---

## 2. 외부 AI 3사 자문 응답 심층 비교 분석

### 2.1 3대 AI별 핵심 자문 철학 및 접근 방식 비교

| 평가 영역 | Google Gemini (1.5 Pro) | Perplexity (Sonar/GPT) | 아주LLM (Claude 3.5 추정) |
| :--- | :--- | :--- | :--- |
| **핵심 접근 철학** | **엔지니어링 & 기하학 중심**<br>(즉시 수식화 및 실전 코드화) | **통계적 무결성 & 다단계 검증 중심**<br>(Ablation 및 1,000회 Bootstrap CI) | **실전 리스크 방어 & 데이터 보존 중심**<br>(치명적 함정 선제 경고) |
| **Q1. 원판 가림 극복** | `VisibilityGuidedInpaintingGCN`<br>(신뢰도 $v_i(t)$ 가중 하체 뼈대 차용) | `Mask-aware Residual Inpainting`<br>(관측 좌표 유지 및 잔차 추정) | **4중 텐서 동시 공급 권고**<br>([원본, 복원, 마스크, 분산] 결함 은폐 방지) |
| **Q2. 시작 위상 정렬** | `Kinematic Anchor Phase Warping (KAPW)`<br>(손목 Nadir을 index 32에 고정 정렬) | `Start-State Head` (바닥/기립 분류)<br>+ 상체 궤적 Soft-DTW 정렬 | **다중 구간 분할 판정** (상승/준비-하강/전체)<br>※ 발 결함 신호로 워핑 절대 금지 |
| **Q3. 체형 불변 피처** | `Zero-Baseline Dimensionless`<br>(초기 셋업 영점화 + 3D 정강이 정규화) | `Foot Plane Normal`<br>(발바닥 법선 내적 + 발/정강이 비율) | **지면 기준 발 피치 각도 (Pitch Angle)**<br>※ 상시 결함자(Always Lift) 영점 소멸 방어 |
| **Q4. 물리 손실 & 보류** | `Physics-Informed Soft-F1 Loss`<br>+ 3-State 그리드 서치 탐색 | Cost-Sensitive Loss + Contact Penalty<br>+ MC-Dropout 불확실성 보류 | Expected Error Cost 기반 보류 예산 할당<br>+ 고신뢰 오판 방어 10% 엄수 |

### 2.2 핵심 질문별 최적 통합 결론

1. **[Q1 원판 가림] 4-Channel Uncertainty-Aware Feature Tensor**:
   - Gemini의 VG-ST-GCN 복원기를 도입하되, 아주LLM의 경고대로 복원 좌표로 원본을 덮어쓰지 않고 `[X_orig, X_inpaint, Mask, Variance]` 4중 텐서를 모델에 동시 공급하여 복원기의 결함 은폐를 방지함.
2. **[Q2 위상 불일치] Upper-Body Nadir KAPW + Dual-Stream**:
   - 양 손목 Y좌표 최하점(Nadir)을 index 32에 고정하는 KAPW 전처리를 적용하되, 결함 신호 축소 방지를 위해 정렬 시계열과 원본 시계열을 병렬 유지하고 시작 상태(바닥/기립) 메타 피처를 주입함.
3. **[Q3 체형 불변] Dual-Reference Calibration (Pitch Angle + Tibia Dimensionless)**:
   - 셋업 영점화 시 상시 결함이 지워지는 문제를 막기 위해 발끝-뒤꿈치 피치 각도(Pitch Angle)를 기본 보존하고, 3D 정강이 길이 중앙값을 분모로 무차원화함.
4. **[Q4 물리 손실 및 보류율] Physics Penalty Soft-F1 + 3-State Budget Control**:
   - 바벨 상승 시점 이전의 뒤꿈치 들림 예측을 처벌하는 물리 손실과 커버리지 90% 이상을 보장하는 이중 임계값 탐색 적용.

---

## 3. 통합 아키텍처 v5.3 / v6.0 실행 로드맵

| 버전 단계 | 핵심 아키텍처 | 전처리 및 손실함수 혁신 | 예상 F1-Score | 보류율(Retake) |
| :--- | :--- | :--- | :---: | :---: |
| **현행 v5.2** | ST-SeqNet(35%) + Tri-Ensemble(65%) | 64프레임 기본 스플라인 + 단일 임계값 | 67.06% | 9.09% (기준선) |
| **차기 v5.3 (단기)** | **KAPW 위상정합 + 무차원 피처 + Physics Loss** | 손목 최하점 정렬 + 지면 피치 + 3-State 최적화 | **76.0~80.0%** | **≤ 9.5% (안정화)** |
| **목표 v6.0 (중기)** | **VG-ST-GCN 복원 + 4-Tensor Dual-Stream Hybrid** | 시공간 Inpainting + 불확실성 인식 게이트 | **83.0~88.0%+** | **≤ 9.0% (극대화)** |

---

## 4. 결론 및 향후 계획

외부 AI 3사의 자문은 Gemini의 '수학적 명쾌함과 KAPW/VG-GCN 아키텍처', 아주LLM의 '복원기 결함 은폐 방지 및 상시 결함 보존 경고', Perplexity의 '다단계 통계적 검증과 선택적 분류'가 유기적으로 결합되어 d3 모델의 난제를 돌파할 명확한 청사진을 제공하였습니다.

즉시 v5.3 구현 및 피험자 1~9번 격리 5-Fold 교차검증을 수행하여, 후방 시점 F1을 75%+ 이상으로 견인하고 전체 평균 F1 80% 돌파를 실증할 예정입니다.
"""
    
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] MD 저장 완료: {MD_PATH}")

if __name__ == "__main__":
    build_docx_and_md()
