# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d4_external_ai_comparison_report_docx.py
Purpose: D:\\자주프 문서\\외부ai 응답 폴더 내 3대 외부 AI(Gemini, Perplexity, 아주LLM) 응답 전문을
         심층 비교 분석하고, 바벨 데드리프트 d4(무릎-골반 동시 펴짐 위반, 힙 슈트업) 모델의
         F1 65~72%+ 돌파를 위한 통합 최적화 액션 플랜 및 기술 보고서를 Word(.docx)와 Markdown(.md)으로 동시 생성.
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
DOCX_PATH = os.path.join(DOC_DIR, "바벨_데드리프트_d4_외부AI_응답_종합비교_및_고도화_분석보고서.docx")
ROOT_DOCX_PATH = os.path.join(ROOT_DIR, "바벨_데드리프트_d4_외부AI_응답_종합비교_및_고도화_분석보고서.docx")
MD_PATH = os.path.join(DOC_DIR, "바벨_데드리프트_d4_외부AI_응답_종합비교_및_고도화_분석보고서.md")

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
        hrun = hp.add_run("vidio-health-trainer | 바벨 데드리프트 d4 외부 AI 자문 종합 비교분석 및 성능 돌파 보고서")
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
    run.font.size = Pt(10.0)
    run.font.bold = True
    run.font.color.rgb = COLOR_ACCENT
    return p

def add_body_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=0, space_after=4)
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = "Malgun Gothic"
        r_b.font.size = Pt(9.5)
        r_b.font.bold = True
        r_b.font.color.rgb = COLOR_TEXT
    r = p.add_run(text)
    r.font.name = "Malgun Gothic"
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_TEXT
    return p

def add_callout(doc, text, title="핵심 권고 요약", hex_bg=HEX_HIGHLIGHT, hex_border=HEX_PRIMARY):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    c.width = Inches(6.5)
    set_cell_background(c, hex_bg)
    set_cell_margins(c, top=100, bottom=100, left=150, right=150)
    
    tcPr = c._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{hex_border}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = c.paragraphs[0]
    style_paragraph(p, space_before=2, space_after=4)
    r_t = p.add_run(f"★ [{title}]\n")
    r_t.font.name = "Malgun Gothic"
    r_t.font.size = Pt(10)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_PRIMARY
    
    r_b = p.add_run(text)
    r_b.font.name = "Malgun Gothic"
    r_b.font.size = Pt(9.0)
    r_b.font.color.rgb = COLOR_TEXT
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def generate_report():
    doc = Document()
    add_header_footer(doc)

    # Title
    tp = doc.add_paragraph()
    style_paragraph(tp, space_before=12, space_after=4)
    tr = tp.add_run("vidio-health-trainer | 인공지능 연구 자문 종합 분석 보고서")
    tr.font.name = "Malgun Gothic"
    tr.font.size = Pt(11)
    tr.font.color.rgb = COLOR_MUTED
    tr.font.bold = True

    tp2 = doc.add_paragraph()
    style_paragraph(tp2, space_before=2, space_after=8)
    tr2 = tp2.add_run("바벨 데드리프트 d4(무릎-골반 동시 펴짐 위반, 힙 슈트업)\n3대 외부 AI 자문 종합 비교분석 및 v4.0 성능 돌파 보고서")
    tr2.font.name = "Malgun Gothic"
    tr2.font.size = Pt(18)
    tr2.font.bold = True
    tr2.font.color.rgb = COLOR_PRIMARY

    # Meta Table
    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("공식 프로젝트 명칭", "vidio-health-trainer (바벨 데드리프트 하위 결함 d4 전문 진단 엔진)"),
        ("분석 대상 외부 AI 자문", "1) Perplexity Deep Research  2) Google Gemini 1.5 Pro  3) 아주LLM (Vision & Dynamics Expert)"),
        ("데이터셋 및 검증 기준", "770개 시퀀스 (정상 635개 82.5%, 결함 135개 17.5%), 9명 피험자 격리 StratifiedGroupKFold & LOSO"),
        ("핵심 연구 목표", "v3.0(F1 33.87%, 3-State 40.14%) 한계 돌파 → 2-State F1 50~58%+, 3-State F1 65~72%+ 달성")
    ]
    for i, (k, v) in enumerate(meta_data):
        c0 = meta_tbl.cell(i, 0)
        c1 = meta_tbl.cell(i, 1)
        c0.width = Inches(1.8)
        c1.width = Inches(4.7)
        set_cell_background(c0, HEX_LIGHT_BG)
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
        
        p0 = c0.paragraphs[0]
        style_paragraph(p0, 0, 0)
        r0 = p0.add_run(k)
        r0.font.name = "Malgun Gothic"
        r0.font.size = Pt(8.5)
        r0.font.bold = True
        r0.font.color.rgb = COLOR_TEXT
        
        p1 = c1.paragraphs[0]
        style_paragraph(p1, 0, 0)
        r1 = p1.add_run(v)
        r1.font.name = "Malgun Gothic"
        r1.font.size = Pt(8.5)
        r1.font.color.rgb = COLOR_TEXT
        
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 1. Executive Summary
    add_heading_1(doc, "1. 핵심 분석 요약 (Executive Summary)")
    add_body_p(doc, 
        "D:\\자주프 문서\\외부ai 응답 폴더에 수합된 3대 최상위 AI(Google Gemini, Perplexity Deep Research, 아주LLM)의 자문 응답 전문을 "
        "정밀 교차 대조한 결과, 세 모델 모두 d4(무릎-골반 동시 펴짐 위반 / 힙 슈트업, Stripper Deadlift) 결함 진단의 병목 원인과 돌파 해법에 대해 "
        "놀라울 정도로 일치된 공통 진단과 강력한 상호보완적 솔루션을 제시하였습니다.")
    
    add_callout(doc, 
        "1. d4의 본질적 기전 규명: '값(Magnitude)'이 아니라 '시간 선후관계(Temporal Order / 1~3프레임 위상차)'의 결함\n"
        "   - 힙 슈트업은 First Pull(바닥~무릎) 초기 무릎 신전이 고관절 신전보다 약 0.05~0.1초 먼저 시작되는 미세 위상차(Phase Lag)가 핵심입니다.\n"
        "   - 기존 37D 피처셋은 전 구간의 요약 통계량(평균/최대/표준편차) 위주로 구성되어 이 미세한 시간차 정보가 소실되었습니다.\n\n"
        "2. 시계열 딥러닝(1D-TCN) + 생체역학 GBDT Two-Stream 융합 합의\n"
        "   - 세 AI 모두 64프레임 시퀀스를 직접 수용하는 경량 1D-TCN을 보조 스트림으로 도입하고, GBDT OOF 확률과 Late Fusion할 것을 강력 권고하였습니다.\n\n"
        "3. 정면 뷰(시점 C / view3) Z축 깊이 결손 완전 극복: HS-VVR (골반-어깨 수직 속도비)\n"
        "   - MediaPipe의 불완전한 단안 Z축 좌표 대신, 2D 투영 평면에서 불변인 골반-어깨 수직 속도비(HS-VVR)를 도입하여 정면 결함 미탐을 원천 차단합니다.\n\n"
        "4. 클래스 불균형(17.5%) 극복: 비대칭 초점 손실함수(Asymmetric Focal Loss) 및 Hard Negative Mining\n"
        "   - 무작위 오버샘플링(SMOTE)을 배제하고 결함(소수 클래스)의 Recall을 보존하되 정상의 오탐(FP)을 억제하는 손실함수 설계를 적용합니다.",
        title="3대 외부 AI 공통 진단 및 핵심 합의 결론"
    )

    # 2. Comparison Matrix
    add_heading_1(doc, "2. 3대 외부 AI 응답 비교 매트릭스 (Comparison Matrix)")
    add_body_p(doc, "본 프로젝트의 5대 핵심 기술 질의(Q1~Q5)에 대한 세 AI 모델의 구체적인 답변 및 방법론을 비교한 매트릭스입니다:")

    tbl = doc.add_table(rows=6, cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["질의 영역", "Google Gemini 1.5 Pro", "Perplexity Deep Research", "아주LLM (Vision Expert)"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        style_paragraph(p, 0, 0)
        r = p.add_run(h)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    matrix_rows = [
        ("Q1. 클래스 불균형\n(17.5% 결함)",
         "• Asymmetric Focal Loss (ASL)\n• γ+=1.0, γ-=3.0, λ-=0.5\n• 결함 Recall 보존 및 정상 오탐(FP) 강력 억제",
         "• Group-stratified Cross-Entropy + Asymmetric Focal Loss\n• SMOTE 배제, scale_pos_weight + OOF threshold 튜닝\n• PR-AUC/MCC 감사",
         "• Class-balanced Asymmetric Focal Loss (CBAFL)\n• Hard-Negative Mining (FP 억제 패널티)\n• 보조 손실(Supervised Contrastive) 제안"),
        ("Q2. 64프레임\n시계열 융합",
         "• 경량 1D-TCN (Dilation 1, 2, 4)\n• 4대 관절 12D 시계열 수용\n• GBDT OOF + TCN OOF 메타 융합 (Logistic Regression)",
         "• Receptive Field 64 전수 커버 경량 TCN (양방향/Full rep)\n• ST-GCN 15~17개 관절 축소\n• OOF 누수 방지 엄수",
         "• 정규화된 64프레임 각도/속도 TCN\n• GBDT(37D+신규) + TCN Two-Stream\n• View/Quality-aware Gating Network 메타 융합"),
        ("Q3. 정면(시점 C)\nZ축 결손 극복",
         "• MediaPipe Z축 완전 배제 권고\n• ★ HS-VVR (골반-어깨 수직 속도비)\n• 2D Y축(수직) 미분 비율로 힙 슈트업 완벽 포착",
         "• 단안 깊이 맹신 경계\n• 정규화된 Foreshortening Proxy (상완/전완 투영비)\n• 관측가능성 게이트 분리",
         "• 시점 불변 상대 기하학 5종:\n 1) Pelvis-Shoulder Axial Rise Ratio\n 2) Trunk-Femur Projected Coupling\n 3) Scale-free Progress Mismatch"),
        ("Q4. d4 타깃\n생체역학 피처",
         "• 운동 사슬 위상 지연시간 (ΔT)\n  (무릎 최고각속도 시점 - 고관절 최고각속도 시점)\n• 수평 모멘트암 대리 지표 (MA_proxy)",
         "• Event-based Phase Lag (신전 개시 시점 차)\n• Hip-rise vs Bar-rise Decoupling\n• Early Velocity Coupling Ratio",
         "• Onset Latency (신전 개시 시간차)\n• Early Impulse Ratio (초기 30% 충격량 비)\n• Knee-Hip Acceleration Lead\n• Quasi-static Trunk Moment"),
        ("Q5. 보류율 10% 제약\n3-State 전략",
         "• Constrained Dual-Threshold Grid Search\n• Decline Rate ≤ 10% 엄수 조건 하 F1 최대화\n• OOF 확률 최적화",
         "• 전체 보류율뿐 아니라 ★ 각 시점별 보류율 ≤ 10% 제약 준수\n• Temperature Scaling 캘리브레이션 필수\n• Brier Score 검증",
         "• 보류율 치팅(보류 남발로 F1 상승) 원천 금지\n• Risk-based Abstention (Binary Entropy + Disagreement)\n• LOSO 9-Fold 교차검증 병행 의무화")
    ]

    for row_idx, r_data in enumerate(matrix_rows):
        for col_idx, text in enumerate(r_data):
            cell = tbl.cell(row_idx + 1, col_idx)
            if col_idx == 0:
                set_cell_background(cell, HEX_LIGHT_BG)
                cell.width = Inches(1.4)
            else:
                cell.width = Inches(1.7)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            style_paragraph(p, 0, 0)
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r.font.color.rgb = COLOR_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. Deep Dive
    add_heading_1(doc, "3. 모델별 핵심 인사이트 심층 분석")
    
    # 3.1 Gemini
    add_heading_2(doc, "3.1. Google Gemini: 발상의 전환과 2D 불변 메트릭 (HS-VVR)")
    add_body_p(doc, 
        "Google Gemini는 d4의 운동학적 본질을 '운동 사슬 붕괴(Kinematic Chain Breakdown)'로 정의하고, "
        "정면 카메라(시점 C / view3)에서 발생하는 Z축 결손 문제를 발상의 전환으로 해결하는 킬러 피처를 제안했습니다:")
    add_body_p(doc,
        "리프터가 정상적인 데드리프트를 수행할 때는 바닥에서 무릎까지 바벨을 들어올리는 동안(First Pull) 상체 각도가 일정하게 유지되며 "
        "골반과 어깨가 동일한 수직 속도로 동시에 상승합니다. 그러나 힙 슈트업(d4 결함)이 발생하면 무릎이 조기에 펴지면서 골반만 치솟고 어깨는 거의 정체됩니다. "
        "따라서 불완전한 단안 3D Z축(깊이)을 전혀 쓰지 않고, 순수 2D Y축(수직 높이) 미분만으로 '골반-어깨 수직 속도비'를 계산하면 정면 뷰에서도 완벽하게 결함을 포착할 수 있습니다.",
        bold_prefix="• HS-VVR (Hip-Shoulder Vertical Velocity Ratio)의 원리: "
    )
    add_body_p(doc,
        "HS-VVR(t) = |V_y^hip(t)| / (|V_y^shoulder(t)| + ε)  [First Pull 구간 최대값 추출]\n"
        "- 정상 동작: 골반과 어깨가 함께 상승하므로 비율이 1.0 안팎 유지\n"
        "- d4 결함: 어깨가 정체된 상태에서 골반만 급상승하므로 비율이 2.5~5.0 이상으로 폭증",
        bold_prefix="• 수학 공식 및 판정 기준: "
    )
    add_body_p(doc,
        "L = - (1/N) Σ [ y_i (1 - p_i)^γ+ log(p_i) + (1 - y_i) λ- (p_i)^γ- log(1 - p_i) ]\n"
        "(권장 파라미터: γ+ = 1.0, γ- = 3.0, λ- = 0.5)\n"
        "이 수식은 결함 데이터(y=1)에 대해서는 높은 집중도를 유지하여 Recall을 방어하고, 정상 데이터(y=0)가 조금이라도 맞았을 때 손실 기여도를 급격히 0으로 만들어 오탐(FP)을 원천 차단합니다.",
        bold_prefix="• 비대칭 초점 손실함수 (Asymmetric Focal Loss) 수식: "
    )

    # 3.2 Perplexity
    add_heading_2(doc, "3.2. Perplexity Deep Research: 64프레임 전수 수용 TCN 및 엄격한 누수 차단")
    add_body_p(doc, 
        "Perplexity는 최신 생체역학 문헌(Ippersiel et al., 2024; Escamilla et al., 2002)을 기반으로, "
        "현재 37D 통계 요약 피처가 힙 슈트업의 핵심인 '1~3프레임(약 0.05~0.1초)의 시간적 선후관계'를 보존하지 못한다는 점을 수학적으로 지적했습니다:")
    add_body_p(doc,
        "일반적인 실시간 스트리밍에서는 미래 프레임을 볼 수 없으므로 인과적 패딩(Causal Padding)이 필요하지만, "
        "본 시스템은 이미 1회 반복이 엄밀히 분할되어 64프레임으로 정규화된 시퀀스를 입력받으므로 양방향(Non-causal) 컨볼루션을 사용하여 "
        "리셉티브 필드(Receptive Field)가 64프레임 전체를 빈틈없이 커버하도록 설계해야 합니다 (Dilation 1, 2, 4, 8 조합).",
        bold_prefix="• 양방향 Receptive Field 전수 수용: "
    )
    add_body_p(doc,
        "딥러닝 모델의 출력 임베딩을 추출하여 GBDT에 투입할 때, 동일한 훈련 데이터셋으로 학습된 딥러닝 모델의 예측값을 그대로 넣으면 심각한 정보 누수가 발생합니다. "
        "반드시 5-Fold 피험자 격리 루프 내부에서 Train 폴드로만 TCN을 학습하고, 오직 검증 폴드에 대한 OOF(Out-of-Fold) 예측 확률/임베딩만 생성하여 GBDT 메타 모델에 전달해야 합니다.",
        bold_prefix="• 엄격한 OOF Stacking 누수 차단 원칙: "
    )
    add_body_p(doc,
        "기존 v3.0 명세서에서 전체 보류율은 9.7%였으나 view3(정면)의 보류율은 12.3%로 10% 제약을 초과하였습니다. "
        "이는 특정 어려운 시점(view3)을 시스템이 대거 보류(기권) 처리하여 겉보기 F1을 올린 착시일 수 있으므로, "
        "향후 3-State 최적화 시 '전체 보류율 ≤ 10%'뿐 아니라 '5대 시점 각각의 보류율 ≤ 10%'를 제약 조건으로 명시해야 합니다.",
        bold_prefix="• 시점별 보류율 왜곡 방지 규정: "
    )

    # 3.3 아주LLM
    add_heading_2(doc, "3.3. 아주LLM: 미세 위상차 5대 동역학 피처 & Risk-based Abstention")
    add_body_p(doc, 
        "아주LLM은 컴퓨터 비전 및 생체역학 동역학 관점에서, d4 결함을 직접 겨냥하는 5대 신규 피처와 "
        "단순 확률 구간을 넘어선 '불확실도-불일치도 기반 위험도(Risk-based) 3-State 정책'을 제안했습니다:")
    add_body_p(doc,
        "1) Onset Latency (무릎-고관절 신전 개시 시간차): 무릎과 고관절 신전 각속도가 개별 ROM 기준 10%를 넘는 최초 시점의 차이 (d4 결함 시 음수 폭증)\n"
        "2) Early Impulse Ratio (초기 30% 무릎 충격량 비): First Pull 초기 30% 구간에서 무릎 신전 각속도의 적분값 비율\n"
        "3) Knee-Hip Acceleration Lead (가속도 선행 피크): 무릎 각가속도 최대 피크 시점과 고관절 각가속도 피크 시점 간의 프레임 간격\n"
        "4) Hip-rise / Bar-rise Decoupling: 바벨 상승 대비 골반 상승의 수직 궤적 분리도\n"
        "5) Quasi-static Trunk Moment Proxy: 미드풋 대리점(발목)과 고관절 간의 수평 모멘트암 시계열 변화율",
        bold_prefix="• d4 직접 겨냥 5대 생체역학 동역학 피처: "
    )
    add_body_p(doc,
        "단순히 예측 확률 P가 0.4~0.6 사이에 있다고 보류하는 방식은 확률 보정(Calibration)이 조금만 틀어져도 심각하게 왜곡됩니다. "
        "대신 ① 모델 예측의 정보 엔트로피(Binary Entropy)와 ② GBDT 스트림과 TCN 스트림 간의 예측 불일치도(|P_gbdt - P_tcn|), "
        "③ 관절 가시성/품질(Pose Quality)을 결합한 통합 위험도(Risk Score)를 산출하고, 전체 상위 10% 위험 샘플만 정밀하게 보류하는 방식을 권고하였습니다.",
        bold_prefix="• 불확실도-불일치도 결합 위험도 기반 보류 (Risk-based Abstention): "
    )

    # 4. Golden Action Plan
    add_heading_1(doc, "4. d4 v4.0 통합 돌파를 위한 '골든 액션 플랜' 4대 핵심 과제")
    add_body_p(doc, "세 AI의 공통 합의 및 최우선 제안을 집약하여 즉시 실행 가능한 4단계 고도화 로드맵을 확정하였습니다:")

    action_tbl = doc.add_table(rows=5, cols=5)
    action_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    act_headers = ["과제 순위", "핵심 액션 아이템", "세부 기술 명세 및 수학 수식", "해결 대상 병목", "예상 기대 효과"]
    for j, h in enumerate(act_headers):
        cell = action_tbl.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 80, 80, 80, 80)
        p = cell.paragraphs[0]
        style_paragraph(p, 0, 0)
        r = p.add_run(h)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    act_rows = [
        ("1순위\n(즉시)",
         "6대 미세 위상차\n동역학 피처 추출\n(37D → 43D)",
         "• HS-VVR (골반-어깨 수직 속도비, 2D Y축)\n• Onset Latency (무릎-골반 신전 개시 시차)\n• Early Impulse Ratio (초기 30% 충격량 비)\n• Knee-Hip Acceleration Lead (가속도 선행)\n• Trunk-Femur Coupling & Decoupling",
         "정면(시점 C) F1 31.3% 추락 및\n요약 통계량의 시간 순서\n정보 손실 원천 해소",
         "정면 F1 31.3% → 55%+ 급상승,\n기본 GBDT F1 33.87% → 45~50%+ 견인"),
        ("2순위\n(손실함수)",
         "Class-Balanced\nAsymmetric Focal\n손실함수 도입",
         "• Asymmetric Focal Loss (γ+=1.0, γ-=3.0)\n• 결함 Recall 보존 및 정상 오탐(FP) 억제\n• Hard Negative Mining (애매한 정상 패널티)",
         "17.5% 극심한 클래스 불균형에\n의한 FP 폭증 및 Recall\n동반 추락 딜레마 극복",
         "정상에 대한 오탐(FP) 40% 이상 감축,\nPrecision 37.17% → 50%+ 회복"),
        ("3순위\n(아키텍처)",
         "GBDT + 1D-TCN\nTwo-Stream\nOOF Late Fusion",
         "• Stream A: 43D 확장 생체역학 GBDT\n• Stream B: 64프레임 관절각/속도 경량 TCN\n• 누수 없는 피험자 격리 OOF 확률 생성\n• Meta-Learner: Calibrated Logistic Regression",
         "단일 트리 모델의 시계열\n패턴 수용 한계 극복 및\n앙상블 시너지 극대화",
         "2-State 기본 F1 50~58%+ 달성,\nPR-AUC 0.3541 → 0.60+ 돌파"),
        ("4순위\n(결정정책)",
         "Risk-based\nCalibrated 3-State\n제약 최적화",
         "• Temperature Scaling 기반 확률 캘리브레이션\n• Binary Entropy + Stream Disagreement 위험도\n• 전체 보류율 ≤ 10% 및 시점별 보류율 ≤ 10% 동시 충족 Dual Threshold 탐색",
         "시점 C 보류율 12.3% 편향 왜곡\n방지 및 임계값 튜닝 신뢰도\n무결성 확보",
         "Coverage ≥ 90% 엄수 하에\n최종 3-State F1 65~72%+ 달성")
    ]

    for row_idx, r_data in enumerate(act_rows):
        for col_idx, text in enumerate(r_data):
            cell = action_tbl.cell(row_idx + 1, col_idx)
            if col_idx == 0:
                set_cell_background(cell, HEX_HIGHLIGHT)
                cell.width = Inches(1.0)
            elif col_idx == 1:
                cell.width = Inches(1.5)
            elif col_idx == 2:
                cell.width = Inches(1.8)
            elif col_idx == 3:
                cell.width = Inches(1.3)
            else:
                cell.width = Inches(1.4)
            set_cell_margins(cell, 60, 60, 60, 60)
            p = cell.paragraphs[0]
            style_paragraph(p, 0, 0)
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r.font.color.rgb = COLOR_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5. Pipeline Architecture Diagram & Targets
    add_heading_1(doc, "5. 차세대 d4 v4.0 아키텍처 명세 및 단계별 성능 목표")
    add_body_p(doc, 
        "바벨 데드리프트 d4 모델의 고도화 파이프라인은 100% MediaPipe 비전 키포인트만을 활용하며(AI-Hub JSON 영구 격리 준수), "
        "64프레임 정규화 시퀀스를 기반으로 Two-Stream 하이브리드 구조로 전면 진화합니다:")

    add_callout(doc,
        "[입력 64프레임 키포인트 시퀀스 (770개 전수, 9명 피험자, 5대 전 시점)]\n"
        "      │\n"
        "      ├─► [Stream A: 43D 생체역학 피처 추출기]\n"
        "      │     • 기존 37D (CRP, MVC, 체형정규화 등)\n"
        "      │     • 신규 6D: HS-VVR(2D 수직속도비), Onset Latency, Early Impulse, Accel Lead 등\n"
        "      │     └─► [Tri-Ensemble GBDT (CatBoost, LightGBM, XGBoost)] ──► P_gbdt (OOF)\n"
        "      │\n"
        "      ├─► [Stream B: 64프레임 시계열 딥러닝 인코더]\n"
        "      │     • 4대 관절 12D 시계열 + Dilated Conv (Dilation 1, 2, 4) 경량 1D-TCN\n"
        "      │     • Class-Balanced Asymmetric Focal Loss (γ+=1.0, γ-=3.0)\n"
        "      │     └─► [Temporal Max Pooling + Linear] ──► P_tcn (OOF)\n"
        "      │\n"
        "      └─► [Meta-Fusion & Calibrated Risk Decision]\n"
        "            • Temperature Scaling Calibrated Stacking (Logistic Regression)\n"
        "            • Risk Score = Entropy(P) + |P_gbdt - P_tcn| + (1 - Quality)\n"
        "            • Coverage ≥ 90% (Decline ≤ 10%) 제약 만족 Dual Threshold 탐색\n"
        "            └─► [최종 3-State 판정: 정상 / 보류 / 힙 슈트업 결함]",
        title="d4 v4.0 Two-Stream 메타 융합 아키텍처 다이어그램"
    )

    add_heading_2(doc, "5.1. 버전별 성능 발전 단계 및 최종 목표치")
    
    perf_tbl = doc.add_table(rows=5, cols=6)
    perf_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    p_headers = ["파이프라인 버전", "핵심 기술 구성", "Recall", "Precision", "2-State F1", "3-State F1 (보류 ≤10%)"]
    for j, h in enumerate(p_headers):
        cell = perf_tbl.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 80, 80, 80, 80)
        p = cell.paragraphs[0]
        style_paragraph(p, 0, 0)
        r = p.add_run(h)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    perf_rows = [
        ("v1.0 (초기 기준선)", "24D 원시 각도 피처 + Random Forest", "22.2%", "28.5%", "25.0%", "28.3%"),
        ("v2.0 (개선 모델)", "31D 확장 피처 + LightGBM 단일 모델", "27.4%", "33.1%", "29.9%", "34.5%"),
        ("v3.0 (현재 베이스라인)", "37D 생체역학 피처 (CRP/MVC) + Tri-GBDT", "31.11%", "37.17%", "33.87%", "40.14% (보류 9.7%)"),
        ("★ v4.0 (통합 목표)", "43D 동역학 피처 + 1D-TCN Two-Stream OOF 융합", "55.0%+", "62.0%+", "55~58%+", "65.0 ~ 72.0%+")
    ]

    for row_idx, r_data in enumerate(perf_rows):
        for col_idx, text in enumerate(r_data):
            cell = perf_tbl.cell(row_idx + 1, col_idx)
            if row_idx == 3: # Target
                set_cell_background(cell, HEX_HIGHLIGHT)
            elif col_idx == 0:
                set_cell_background(cell, HEX_LIGHT_BG)
            set_cell_margins(cell, 60, 60, 60, 60)
            p = cell.paragraphs[0]
            style_paragraph(p, 0, 0)
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(8.5)
            if row_idx == 3:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r.font.color.rgb = COLOR_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Save files
    doc.save(DOCX_PATH)
    shutil.copyfile(DOCX_PATH, ROOT_DOCX_PATH)
    print(f"Saved docx to: {DOCX_PATH}")
    print(f"Copied docx to root: {ROOT_DOCX_PATH}")

    # Generate Markdown
    md_content = f"""# vidio-health-trainer | 인공지능 연구 자문 종합 분석 보고서
# 바벨 데드리프트 d4(무릎-골반 동시 펴짐 위반, 힙 슈트업) 3대 외부 AI 자문 종합 비교분석 및 v4.0 성능 돌파 보고서

- **분석 대상:** `D:\\자주프 문서\\외부ai 응답` 폴더 내 3대 최상위 AI 모델 응답 전문
  1. `deadlift-d4-f1-maximization.pplx.md.pdf` (Perplexity Deep Research)
  2. `gemini.pdf` (Google Gemini 1.5 Pro)
  3. `새 채팅 _ 아주LLM.pdf` (Vision & Dynamics Expert)
- **공식 프로젝트명:** `vidio-health-trainer`
- **검증 환경:** 770개 시퀀스 (정상 635개 82.5%, 결함 135개 17.5%), 9명 피험자 격리 StratifiedGroupKFold & LOSO, 5대 전 시점(시점 A~E / view1~5) 전수 반영
- **핵심 목표:** v3.0(F1 33.87%, 3-State 40.14%) 한계 돌파 → 2-State F1 50~58%+, 3-State F1 65~72%+ 달성

---

## 1. 핵심 분석 요약 (Executive Summary)

세 외부 AI의 독립적인 답변을 교차 대조한 결과, d4 결함 진단의 한계를 깨뜨릴 결정적인 **4대 공통 해법과 획기적인 신규 피처**가 도출되었습니다:

1. **d4의 본질적 기전 규명: '크기(Magnitude)'가 아닌 '시간 선후관계(Temporal Order / 1~3프레임 위상차)'의 결함**
   - 힙 슈트업(Stripper Deadlift)은 First Pull(바닥~무릎) 초기 무릎 신전이 고관절 신전보다 약 0.05~0.1초 먼저 시작되는 미세 위상차(Phase Lag)가 핵심입니다.
   - 기존 37D 피처셋은 전 구간의 요약 통계량(평균/최대/표준편차) 위주로 구성되어 이 미세한 시간차 정보가 소실되었습니다.
2. **시계열 딥러닝(1D-TCN) + 생체역학 GBDT Two-Stream 융합 합의**
   - 세 AI 모두 64프레임 시퀀스를 직접 수용하는 경량 1D-TCN을 보조 스트림으로 도입하고, GBDT OOF 확률과 Late Fusion할 것을 강력 권고하였습니다.
3. **정면 뷰(시점 C / view3) Z축 깊이 결손 완전 극복: HS-VVR (골반-어깨 수직 속도비)**
   - MediaPipe의 불완전한 단안 Z축 좌표 대신, 2D 투영 평면에서 보존되는 골반-어깨 수직 속도비(HS-VVR)를 도입하여 정면 결함 미탐을 원천 차단합니다.
4. **클래스 불균형(17.5%) 극복: 비대칭 초점 손실함수(Asymmetric Focal Loss) 및 Hard Negative Mining**
   - 무작위 오버샘플링(SMOTE)을 배제하고 결함(소수 클래스)의 Recall을 보존하되 정상의 오탐(FP)을 억제하는 손실함수 설계를 적용합니다.

---

## 2. 3대 외부 AI 응답 비교 매트릭스 (Comparison Matrix)

| 질의 영역 | Google Gemini 1.5 Pro | Perplexity Deep Research | 아주LLM (Vision Expert) |
| :--- | :--- | :--- | :--- |
| **Q1. 클래스 불균형 (17.5% 결함)** | • Asymmetric Focal Loss (ASL)<br>• $\\gamma_+=1.0, \\gamma_-=3.0, \\lambda_-=0.5$<br>• 결함 Recall 보존 및 정상 오탐(FP) 강력 억제 | • Group-stratified Cross-Entropy + Asymmetric Focal Loss<br>• SMOTE 배제, scale_pos_weight + OOF threshold 튜닝<br>• PR-AUC/MCC 감사 | • Class-balanced Asymmetric Focal Loss (CBAFL)<br>• Hard-Negative Mining (FP 억제 패널티)<br>• 보조 손실(Supervised Contrastive) 제안 |
| **Q2. 64프레임 시계열 융합** | • 경량 1D-TCN (Dilation 1, 2, 4)<br>• 4대 관절 12D 시계열 수용<br>• GBDT OOF + TCN OOF 메타 융합 (Logistic Regression) | • Receptive Field 64 전수 커버 경량 TCN (양방향/Full rep)<br>• ST-GCN 15~17개 관절 축소<br>• OOF 누수 방지 엄수 | • 정규화된 64프레임 각도/속도 TCN<br>• GBDT(37D+신규) + TCN Two-Stream<br>• View/Quality-aware Gating Network 메타 융합 |
| **Q3. 정면(시점 C) Z축 결손 극복** | • MediaPipe Z축 완전 배제 권고<br>• **★ HS-VVR (골반-어깨 수직 속도비)**<br>• 2D Y축(수직) 미분 비율로 힙 슈트업 완벽 포착 | • 단안 깊이 맹신 경계<br>• 정규화된 Foreshortening Proxy (상완/전완 투영비)<br>• 관측가능성 게이트 분리 | • 시점 불변 상대 기하학 5종:<br> 1) Pelvis-Shoulder Axial Rise Ratio<br> 2) Trunk-Femur Projected Coupling<br> 3) Scale-free Progress Mismatch |
| **Q4. d4 타깃 생체역학 피처** | • 운동 사슬 위상 지연시간 ($\\Delta T$)<br>  (무릎 최고각속도 시점 - 고관절 최고각속도 시점)<br>• 수평 모멘트암 대리 지표 ($MA_{{proxy}}$) | • Event-based Phase Lag (신전 개시 시점 차)<br>• Hip-rise vs Bar-rise Decoupling<br>• Early Velocity Coupling Ratio | • Onset Latency (신전 개시 시간차)<br>• Early Impulse Ratio (초기 30% 충격량 비)<br>• Knee-Hip Acceleration Lead<br>• Quasi-static Trunk Moment |
| **Q5. 보류율 10% 제약 3-State 전략** | • Constrained Dual-Threshold Grid Search<br>• Decline Rate $\\le 10\\%$ 엄수 조건 하 F1 최대화<br>• OOF 확률 최적화 | • 전체 보류율뿐 아니라 **★ 각 시점별 보류율 $\\le 10\\%$ 제약 준수**<br>• Temperature Scaling 캘리브레이션 필수<br>• Brier Score 검증 | • 보류율 치팅(보류 남발로 F1 상승) 원천 금지<br>• Risk-based Abstention (Binary Entropy + Disagreement)<br>• LOSO 9-Fold 교차검증 병행 의무화 |

---

## 3. 모델별 핵심 인사이트 심층 분석

### 3.1. Google Gemini: 발상의 전환과 2D 불변 메트릭 (HS-VVR)
- **HS-VVR (Hip-Shoulder Vertical Velocity Ratio)의 원리**:
  리프터가 정상적인 데드리프트를 수행할 때는 바닥에서 무릎까지 바벨을 들어올리는 동안(First Pull) 상체 각도가 일정하게 유지되며 골반과 어깨가 동일한 수직 속도로 동시에 상승합니다. 그러나 힙 슈트업(d4 결함)이 발생하면 무릎이 조기에 펴지면서 골반만 치솟고 어깨는 거의 정체됩니다.
  따라서 불완전한 단안 3D Z축(깊이)을 전혀 쓰지 않고, **순수 2D Y축(수직 높이) 미분만으로 '골반-어깨 수직 속도비'를 계산하면 정면 뷰(시점 C)에서도 완벽하게 결함을 포착**할 수 있습니다.
  $$\\text{{HS-VVR}}(t) = \\frac{{|V_y^{{\\text{{hip}}}}(t)|}}{{|V_y^{{\\text{{shoulder}}}}(t)| + \\epsilon}}$$
- **비대칭 초점 손실함수 (Asymmetric Focal Loss)**:
  $$\\mathcal{{L}} = - \\frac{{1}}{{N}} \\sum_{{i=1}}^N \\left[ y_i (1 - p_i)^{{\\gamma_+}} \\log(p_i) + (1 - y_i) \\lambda_- (p_i)^{{\\gamma_-}} \\log(1 - p_i) \\right]$$
  ($\\gamma_+ = 1.0, \\gamma_- = 3.0, \\lambda_- = 0.5$)  
  결함(소수 클래스)은 강하게 학습하여 Recall을 유지하고, 정상(다수 클래스)의 쉬운 예측 손실 기여도를 억제하여 오탐(FP)을 원천 차단합니다.

### 3.2. Perplexity Deep Research: 64프레임 전수 수용 TCN 및 엄격한 누수 차단
- **양방향 Receptive Field 전수 수용**: 1회 완료 시퀀스이므로 인과적 패딩(Causal Padding) 없이 양방향 컨볼루션을 적용하여 64프레임 전체 시계열 패턴을 완벽히 포착.
- **엄격한 OOF Stacking 누수 차단 원칙**: 피험자 격리 루프 내부에서 Train 폴드로만 TCN을 학습하고, 오직 검증 폴드의 OOF 예측 확률만 GBDT 메타 모델에 전달.
- **시점별 보류율 왜곡 방지 규정**: 특정 어려운 시점(view3)을 대거 기권 처리하여 F1을 왜곡하는 현상을 방지하기 위해 전체 및 5대 시점 각각의 보류율 $\\le 10\\%$ 준수.

### 3.3. 아주LLM: 미세 위상차 5대 동역학 피처 & Risk-based Abstention
- **d4 직접 겨냥 5대 생체역학 동역학 피처**:
  1. Onset Latency: 무릎과 고관절 신전 각속도가 10%를 넘는 최초 시점의 차이
  2. Early Impulse Ratio: First Pull 초기 30% 구간에서 무릎 신전 각속도의 적분값 비율
  3. Knee-Hip Acceleration Lead: 무릎 각가속도 최대 피크와 고관절 각가속도 피크 간 프레임 간격
  4. Hip-rise / Bar-rise Decoupling: 바벨 상승 대비 골반 상승의 수직 궤적 분리도
  5. Quasi-static Trunk Moment Proxy: 미드풋(발목)과 고관절 간의 수평 모멘트암 시계열 변화율
- **위험도 기반 3-State 보류 (Risk-based Abstention)**:
  모델 정보 엔트로피 + GBDT-TCN 스트림 불일치도 + 관절 품질(Visibility)을 결합한 통합 위험도(Risk Score) 상위 10% 샘플만 선별 보류.

---

## 4. d4 v4.0 통합 돌파를 위한 '골든 액션 플랜' 4대 핵심 과제

| 과제 순위 | 핵심 액션 아이템 | 세부 기술 명세 및 수학 수식 | 해결 대상 병목 | 예상 기대 효과 |
| :---: | :--- | :--- | :---: | :---: |
| **1순위 (즉시)** | **6대 미세 위상차 동역학 피처 추출 (37D → 43D)** | • HS-VVR (골반-어깨 수직 속도비, 2D Y축)<br>• Onset Latency (무릎-골반 신전 개시 시차)<br>• Early Impulse Ratio (초기 30% 충격량 비)<br>• Knee-Hip Acceleration Lead (가속도 선행) | 정면(시점 C) F1 31.3% 추락 및 요약 통계량의 시간 순서 정보 손실 원천 해소 | 정면 F1 31.3% → 55%+ 급상승, 기본 GBDT F1 33.87% → 45~50%+ 견인 |
| **2순위 (손실함수)** | **Class-Balanced Asymmetric Focal Loss 도입** | • Asymmetric Focal Loss (γ+=1.0, γ-=3.0)<br>• 결함 Recall 보존 및 정상 오탐(FP) 억제<br>• Hard Negative Mining (애매한 정상 패널티) | 17.5% 극심한 클래스 불균형에 의한 FP 폭증 및 Recall 동반 추락 딜레마 극복 | 정상에 대한 오탐(FP) 40% 이상 감축, Precision 37.17% → 50%+ 회복 |
| **3순위 (아키텍처)** | **GBDT + 1D-TCN Two-Stream OOF Late Fusion** | • Stream A: 43D 확장 생체역학 GBDT<br>• Stream B: 64프레임 관절각/속도 경량 TCN<br>• 누수 없는 피험자 격리 OOF 확률 생성<br>• Meta-Learner: Calibrated Logistic Regression | 단일 트리 모델의 시계열 패턴 수용 한계 극복 및 앙상블 시너지 극대화 | 2-State 기본 F1 50~58%+ 달성, PR-AUC 0.3541 → 0.60+ 돌파 |
| **4순위 (결정정책)** | **Risk-based Calibrated 3-State 제약 최적화** | • Temperature Scaling 기반 확률 캘리브레이션<br>• Binary Entropy + Stream Disagreement 위험도<br>• 전체 및 시점별 보류율 ≤ 10% 동시 충족 Dual Threshold | 시점 C 보류율 12.3% 편향 왜곡 방지 및 임계값 튜닝 신뢰도 무결성 확보 | Coverage ≥ 90% 엄수 하에 최종 3-State F1 65~72%+ 달성 |

---

## 5. 차세대 d4 v4.0 아키텍처 명세 및 단계별 성능 목표

```mermaid
flowchart TD
    A["입력 64프레임 키포인트 시퀀스 (770개 전수, 9명 피험자, 5대 전 시점)"] --> B["Stream A: 43D 생체역학 피처 추출기<br/>• 기존 37D (CRP, MVC, 체형정규화)<br/>• 신규 6D (HS-VVR, Onset Latency, Early Impulse 등)"]
    A --> C["Stream B: 64프레임 시계열 딥러닝 인코더<br/>• 4대 관절 12D 시계열 + Dilated Conv 경량 1D-TCN<br/>• Class-Balanced Asymmetric Focal Loss"]
    
    B --> D["Tri-Ensemble GBDT (CatBoost, LightGBM, XGBoost)"]
    C --> E["Temporal Max Pooling + Linear"]
    
    D --> F["P_gbdt (피험자 격리 OOF)"]
    E --> G["P_tcn (피험자 격리 OOF)"]
    
    F & G --> H["Meta-Fusion & Calibrated Risk Decision<br/>• Temperature Scaling Calibrated Stacking<br/>• Risk Score = Entropy + Disagreement + Quality"]
    
    H --> I["최종 3-State 판정<br/>(Coverage ≥ 90% 엄수, F1 65~72%+ 달성)"]
```

### 5.1. 버전별 성능 발전 단계 및 최종 목표치

| 파이프라인 버전 | 핵심 기술 구성 | 결함 Recall | 결함 Precision | 2-State F1 | 3-State F1 (보류 ≤10%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v1.0 (초기 기준선)** | 24D 원시 각도 피처 + Random Forest | 22.2% | 28.5% | 25.0% | 28.3% |
| **v2.0 (개선 모델)** | 31D 확장 피처 + LightGBM 단일 모델 | 27.4% | 33.1% | 29.9% | 34.5% |
| **v3.0 (현재 베이스라인)** | 37D 생체역학 피처 (CRP/MVC) + Tri-GBDT | 31.11% | 37.17% | 33.87% | 40.14% (보류 9.7%) |
| **★ v4.0 (통합 목표)** | 43D 동역학 피처 + 1D-TCN Two-Stream OOF 융합 | **55.0%+** | **62.0%+** | **55~58%+** | **65.0 ~ 72.0%+** |

"""
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved md to: {MD_PATH}")

if __name__ == "__main__":
    generate_report()
