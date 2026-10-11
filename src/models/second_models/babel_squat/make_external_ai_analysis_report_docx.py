# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_external_ai_analysis_report_docx.py
Purpose: 외부 AI 6대 기술 자문 심층 비교 분석 및 v10.0 통합 실행 로드맵 DOCX 보고서 생성
Targets:
  1) D:\자주프 문서\5주차\vidio-health-trainer_외부AI_6대기술자문_심층비교분석_및_v10_통합실행로드맵.docx
  2) D:\자주프 문서\외부ai 응답\vidio-health-trainer_외부AI_6대기술자문_심층비교분석_및_v10_통합실행로드맵.docx
"""

import os
import shutil
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, hex_color):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_callout(doc, title, text, border_color="1A365D", bg_color="F7FAFC"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="none"/>
        <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
    </w:tcBorders>
    '''
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.bold = True
    run_t.font.name = '맑은 고딕'
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor.from_string(border_color)
    
    run_b = p.add_run(text)
    run_b.font.name = '맑은 고딕'
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def style_heading(p, text, level=1):
    p.paragraph_format.space_before = Pt(14 if level==1 else (10 if level==2 else 6))
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = '맑은 고딕'
    if level == 1:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    elif level == 2:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    else:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def format_table(table, col_widths, headers, data, header_bg="1A365D", alt_bg="F7FAFC"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_row = table.rows[0]
    for idx, heading in enumerate(headers):
        cell = hdr_row.cells[idx]
        cell.width = col_widths[idx]
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(heading)
        run.bold = True
        run.font.name = '맑은 고딕'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    # Data rows
    for r_idx, row_data in enumerate(data):
        row = table.rows[r_idx + 1]
        bg_col = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(str(val))
            run.font.name = '맑은 고딕'
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
            
            if c_idx in [0, 2, 3] and len(str(val)) < 25:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="8" w:space="0" w:color="CBD5E0"/>
        <w:left w:val="none"/>
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1A365D"/>
        <w:right w:val="none"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

def create_report(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = Document()

    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    style_normal = doc.styles['Normal']
    style_normal.font.name = '맑은 고딕'
    style_normal.font.size = Pt(9.5)
    style_normal.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    # Title block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("vidio-health-trainer\n외부 AI 6대 기술 자문 심층 비교 분석 및 v10.0 통합 실행 로드맵")
    r_title.bold = True
    r_title.font.name = '맑은 고딕'
    r_title.font.size = Pt(17)
    r_title.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Comparative Synthesis of External AI Technical Consultations: Pathways to Clean F1 75%+ & 4-Defect EMR 80%+")
    r_sub.font.name = '맑은 고딕'
    r_sub.font.size = Pt(9.5)
    r_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_meta = p_meta.add_run("작성일자: 2026. 09. 27 | 시스템 버전: v10.0 Master Blueprint | 프로젝트 공식 명칭: vidio-health-trainer")
    r_meta.font.name = '맑은 고딕'
    r_meta.font.size = Pt(8.5)
    r_meta.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

    doc.add_paragraph()

    # 1. 개요 및 분석 대상 문서 명세
    style_heading(doc.add_paragraph(), "1. 분석 개요 및 외부 AI 자문 문서 명세", level=1)
    doc.add_paragraph(
        "본 보고서는 vidio-health-trainer 바벨 스쿼트 AI 통합 진단 엔진 v9.0의 베이스라인 성능(4-Defect EMR 71.24%, Clean F1 67.63%, Clean Precision 60.27%, RETAKE 3.19%)을 차세대 v10.0 목표(Clean Precision 70%+, Clean F1 75%+, 4-Defect EMR 80%+, RETAKE 4.0% 이하)로 도약시키기 위해 수행된 2건의 최신 외부 AI 심층 기술 자문 결과를 전수 대조·분석하고 구체적인 실행 계획을 확정하기 위해 작성되었습니다."
    )
    
    t_docs = doc.add_table(rows=3, cols=4)
    headers_docs = ["구분", "파일명 (외부 AI 응답)", "분량 / 형태", "핵심 자문 관점 및 특징"]
    widths_docs = [Inches(1.0), Inches(2.4), Inches(1.1), Inches(2.4)]
    data_docs = [
        ["자문 문서 1", "vidio-health-trainer v10.0 Clean F1 75%·4-Defect EMR 80% 달성을 위한 6대 기술 자문.pdf", "14 페이지 (학술/수식)", "학술 논문 기반 수학적 정식화, Reverse-ASL, Exact 16-State CRF/Energy Head, Constrained IK, Cross-View Teacher-Student 증류"],
        ["자문 문서 2", "새 채팅 _ 아주LLM.pdf", "25 페이지 (아키텍처/코드)", "시스템 구현 관점, False-Clean Focal Loss, GNN Factor Graph, 14D 관절 채널 ST-GCN+Transformer 하이브리드, OOF Precision 제약 임계값 탐색"]
    ]
    format_table(t_docs, widths_docs, headers_docs, data_docs, header_bg="1A365D")
    doc.add_paragraph()

    add_callout(
        doc,
        "양대 자문 문서의 공통 결론 및 제1원칙 (Universal Core Consensus)",
        "1. False Clean 억제가 최우선: Clean Precision(60.27%)의 병목은 실제 결함 샘플이 미세 결함 상태(Near-Clean)로 Clean에 흡수되는 현상이므로, 결함 샘플에 대한 False Clean 오판에 비대칭적 강력 페널티를 부과해야 한다.\n"
        "2. 9명 피험자(2,605 시퀀스) 데이터 환경 존중: 대형 End-to-End 신경망을 성급히 단독 구축하면 피험자 격리 검증에서 치명적 과적합이 발생한다. 반드시 기존 v9.0 MoE GBDT 챔피언 모델을 베이스로 보존하고, 신규 딥러닝 모듈은 Late Fusion 잔차(Residual) 보완 전문가로 결합해야 한다.\n"
        "3. 원천 좌표 보존(Dual-Stream): 뼈 길이 불변 정규화 시 발목 접지 좌표까지 단순 Forward Kinematics로 누적 이동시키면 c4(발 지면고정) 절대 변위 피처가 파괴된다. 반드시 Retargeted Stream과 Raw Normalized Stream을 이원화해야 한다.",
        border_color="2B6CB0", bg_color="EBF8FF"
    )

    # 2. 6대 핵심 기술 자문 전수 비교 및 심층 분석
    style_heading(doc.add_paragraph(), "2. 6대 핵심 기술 자문 전수 비교 및 심층 분석", level=1)

    # Q1
    style_heading(doc.add_paragraph(), "2.1 [질문 1] Clean Precision 70%+ 달성을 위한 손실 함수 및 분류 전략", level=2)
    p_q1 = doc.add_paragraph(
        "■ 현상 진단: 현재 Clean Recall은 77.05%에 달하지만 Precision은 60.27%에 머무릅니다. 즉, '결함이 하나라도 존재하는 샘플(c1~c4)'을 정상(0,0,0,0)으로 통과시키는 'False Clean'이 전체 정상 판정의 39.73%를 차지하고 있습니다. 이는 단일 결함(Near-Clean) 샘플의 손실 기여도가 대다수 결함 샘플에 묻히기 때문입니다.\n"
        "■ 자문 문서 1 (Reverse-ASL): 일반 Asymmetric Loss(ASL)는 음성을 억제하지만, Clean 판정 입장에서는 결함(음성)을 정상(양성)으로 오판하는 False Clean이 최대 손실입니다. 따라서 Clean 이진 헤드에 결함 샘플이 높은 Clean 확률을 가질 때 지수적 패널티를 주는 Reverse-ASL을 제안합니다.\n"
        "■ 자문 문서 2 (False-Clean Asymmetric Focal Loss & OOF Thresholding): 비대칭 Focal Loss와 함께, Clean 판정 임계값을 손실 함수에만 의존하지 않고 5-Fold OOF 결합 분포에서 Precision 70% 이상을 만족하는 조건부 최대 Recall 임계값(tau_clean)을 1회 탐색하여 고정하는 제약 최적화 전략을 권고합니다."
    )
    p_q1.paragraph_format.line_spacing = 1.15

    t_q1 = doc.add_table(rows=3, cols=3)
    headers_q1 = ["기법 명칭", "수학적 정식화 및 구현 수식", "장점 및 실전 적용 가이드"]
    widths_q1 = [Inches(1.8), Inches(3.0), Inches(2.1)]
    data_q1 = [
        ["Reverse-ASL\n(False-Clean Asymmetric)", "L_clean = -α*y*(1-p)^γ+ * log(p) - λ_FC*(1-y)*p^γ- * log(1-p)\n(단, y=1[c1=c2=c3=c4=0], λ_FC=3.0, γ-=2.0, γ+=1.0)", "결함 샘플의 높은 Clean 확률 p에 대해 p^2 및 λ_FC=3.0의 초강력 패널티 부과. False Clean 직접 차단."],
        ["OOF Precision-Constrained\nThresholding", "max_τ Recall_clean(τ)\ns.t. Precision_clean(τ) ≥ 0.70\n(5-Fold OOF 통합 확률 분포에서 전역 1회 고정)", "Fold별 threshold 재탐색 누출(Leakage)을 원천 차단하며, Recall 손실을 최소화하면서 정확히 목표 Precision 70% 달성."]
    ]
    format_table(t_q1, widths_q1, headers_q1, data_q1, header_bg="2B6CB0")
    doc.add_paragraph()

    # Q2
    style_heading(doc.add_paragraph(), "2.2 [질문 2] c2 후방 시점(시점 A/E) 경추-흉추-견갑대 운동학 연동 및 부분관측 극복", level=2)
    p_q2 = doc.add_paragraph(
        "■ 물리적 한계 경고 (양측 일치): 후방 시점에서 얼굴 랜드마크(코, 눈, 귀)가 보이지 않을 때 흉추/어깨 각도만으로 시선 각도를 100% 결정론적으로 재구성할 수 있다는 가정은 생체역학적으로 성립하지 않습니다. 머리는 체간에 대해 독립적인 회전 자유도를 가지므로, 무리한 3D 회전 추정은 오히려 치명적인 과신(Overconfidence)을 유발합니다.\n"
        "■ 해결책 1 (불확실성 인식 c2 사후확률): 모델이 c2 결함 확률 logit(μ)과 함께 추정 불확실성 분산(log σ^2)을 동시 출력하게 하고, 불확실성이 크면 로짓을 자동으로 0.5 근처로 수축시킵니다 (p_c2 = σ(μ / sqrt(1 + πσ^2/8))).\n"
        "■ 해결책 2 (생체역학 흉곽 로컬 좌표계 48D): 좌우 어깨선과 상부 흉추-골반 벡터로 흉곽 로컬 직교 좌표계 R_thorax(t)를 구축하고, 어깨선 roll, 견갑 비대칭, 흉추-경추 위상차(Phase lag), 상호상관(Cross-correlation) 피처를 추출합니다.\n"
        "■ 해결책 3 (Cross-View Teacher-Student 증류): 전방 시점(시점 C/B/D)에서 얼굴이 온전히 보이는 교사 모델의 c2 로짓을, 후방 시점 학생 모델(얼굴 마스킹 훈련)이 증류(KD)받도록 학습합니다."
    )
    p_q2.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()

    # Q3
    style_heading(doc.add_paragraph(), "2.3 [질문 3] 16-상태 결함 의존성 구조화 모델링 (Factor Graph / Energy Head)", level=2)
    p_q3 = doc.add_paragraph(
        "■ 고정 분류기 체인(Classifier Chain)의 결함: c4 → c3 → c1 순서의 고정 순차 체인은 앞단 분류기의 오류가 뒷단으로 눈덩이처럼 증폭되는 오차 전파(Cascading Error)에 취약합니다.\n"
        "■ 혁신적 설계 - Exact 16-State CRF / Energy-Based Head:\n"
        "스쿼트의 결함 변수는 c1, c2, c3, c4 단 4개의 이진 변수이므로 전체 상태 공간은 2^4 = 16개에 불과합니다. 따라서 복잡한 MCMC 표본추출이나 근사 추론 없이 16개 상태 전수를 매 프레임 정확히 열거(Exact Enumeration)할 수 있습니다.\n"
        "  - Unary Potential (4D): 각 결함 단독 발생 경향성 (u_k(x))\n"
        "  - Pairwise Potential (6D): 결함 쌍 간 상호작용 (v_kl(x), 예: c4 발 불안정과 c3 무릎 모임의 공존 보상)\n"
        "  - Triple Chain Potential (1D): 생체역학 운동사슬(c4 발들림 × c3 무릎모임 × c1 척추말림) 트리플 상호작용 (r(x))\n"
        "이 상호작용 계수들은 고정 상수가 아니라 39D 메타 피처 및 ST-GCN 임베딩으로부터 동적으로 생성되어 시점과 운동 위상에 맞게 변화합니다. 최종 16개 상태 확률은 Softmax(Energy)로 계산되며, 4개 Marginal BCE와 16-State Weighted NLL을 동시에 최적화합니다."
    )
    p_q3.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()

    # Q4
    style_heading(doc.add_paragraph(), "2.4 [질문 4] Skeletal Length Normalization (뼈 길이 불변 정규화) 및 Dual-Stream", level=2)
    p_q4 = doc.add_paragraph(
        "■ 왜 필요한가: MediaPipe 3D 관절 좌표는 깊이(z) 방향 예측의 고유한 불안정성으로 인해 동일 피험자의 대퇴골, 정강이뼈 길이가 프레임마다 심하게 늘어나거나 줄어듭니다(Wobble 노이즈). 이는 각도 및 속도 계산의 왜곡을 초래합니다.\n"
        "■ 기준 길이(L_ref) 산출: 스쿼트 직립 안정 구간(Stand Phase)에서 관절 간 유클리드 거리를 측정하고, 10%~90% 분위수 내 중앙값(Median)을 취해 이상치에 견고한 피험자 고유 뼈 길이를 산출합니다.\n"
        "■ Forward Kinematics (FK) 계층적 투영: Root(골반)에서 시작하여 각 관절의 방향 벡터 v_norm에 L_ref를 곱해 좌표를 재배치합니다.\n"
        "■ 절대 불변 원칙 (Dual-Stream 유지): 골반부터 발끝까지 단순 FK로 누적 투영하면 발의 절대 위치가 밀려나 c4(발바닥 지면고정)의 지면 접지 변위 피처가 파괴됩니다. 따라서 반드시 (1) Retargeted Stream(관절각, ST-GCN 입력)과 (2) Raw Normalized Stream(발 접지 절대변위 피처)을 병렬 유지하고, 두 스트림의 잔차 크기(||x_raw - x_retarget||)를 관측 신뢰도 피처로 추가합니다."
    )
    p_q4.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()

    # Q5
    style_heading(doc.add_paragraph(), "2.5 [질문 5] ST-GCN + Temporal Transformer 하이브리드 결합 방식", level=2)
    p_q5 = doc.add_paragraph(
        "■ 입력 채널 표준화 (14D per Joint): 64프레임 × 33관절 시퀀스에 단순 3D 좌표만 주지 않고, (1) Root 상대 3D, (2) 속도 Δx 3D, (3) 가속도 Δ^2x 3D, (4) 뼈 방향 벡터 3D, (5) 신뢰도 1D, (6) Raw-Retarget 잔차 1D의 총 14D 채널을 입력합니다.\n"
        "■ 2-Layer ST-GCN: 해부학적 인접행렬(A_anat), 좌우 대칭 인접행렬(A_sym), 생체역학 기능적 인접행렬(A_func: 발목↔무릎↔골반↔흉추)을 결합한 공간-시간 그래프 컨볼루션을 수행합니다.\n"
        "■ 4-Layer Temporal Transformer: d_model=128, heads=4, dropout=0.25의 초경량 구조로 하강기-최하사점-상승기의 장거리 운동 의존성을 학습합니다.\n"
        "■ Late Fusion Gating 원칙 (피험자 과적합 차단): 2,605개 시퀀스/9명 피험자 데이터셋에서 End-to-End 대형 모델은 그룹 과적합으로 붕괴합니다. 따라서 검증된 v9.0 MoE(GBDT 챔피언)를 메인 줄기로 보존하고, STGT 임베딩은 게이팅 가중치 g를 통해 Late Fusion 잔차 형태로 결합합니다."
    )
    p_q5.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()

    # Q6
    style_heading(doc.add_paragraph(), "2.6 [질문 6] 신뢰도 보정(Calibration) 및 선택적 판정(Selective Prediction) 고도화", level=2)
    p_q6 = doc.add_paragraph(
        "■ Scalar Temperature Scaling의 한계: 전체 시점과 16개 클래스에 단일 스칼라 T를 적용하면, 후방 시점의 c2 과신을 낮추려다 전방 시점의 높은 정확도까지 뭉개지는 부작용이 발생합니다.\n"
        "■ View-Conditioned Vector Scaling: 시점별·클래스별 scale 벡터 a_v와 bias b_v를 학습(z_cal = a_v ⊙ z + b_v)하여, 특히 Clean 상태 로짓과 후방 c2 로짓을 독립적으로 정밀 보정합니다.\n"
        "■ Quality-Aware Bounded Temperature: 고비용 GP 대신 전체 품질 지수, 가림 여부, c2 불확실성으로 구성된 저차원 품질 벡터로부터 T(r) ∈ [0.6, 2.5]를 산출하여 저품질/후방 시퀀스의 과신을 물리적으로 제한합니다.\n"
        "■ Risk-Aware Selective Prediction (고정 데드밴드 대체): 기존의 [0.32, 0.38] 고정 데드밴드를 폐기하고, 불확실성 엔트로피 H(P), 품질 점수, c2 분산으로 정의된 위험도 함수 r(x)를 기준으로 RETAKE율 ≤ 4.0% 제약 하에서 재촬영을 선별합니다."
    )
    p_q6.paragraph_format.line_spacing = 1.15
    doc.add_paragraph()

    # 3. 양대 자문안 비교 및 시너지 총괄표
    style_heading(doc.add_paragraph(), "3. 양대 자문안 세부 기법 비교 및 상호 시너지 분석", level=1)
    
    t_comp = doc.add_table(rows=7, cols=4)
    headers_comp = ["기술 영역", "자문 문서 1 (이론/정식화 중심)", "자문 문서 2 (구현/아키텍처 중심)", "v10.0 통합 채택 결론"]
    widths_comp = [Inches(1.2), Inches(2.0), Inches(2.0), Inches(1.7)]
    data_comp = [
        ["손실 함수\n(Q1)", "Reverse-ASL\nClean 마진 손실", "False-Clean Focal BCE\nNear-Clean Supervised Contrastive", "Reverse-ASL + OOF 제약 임계값 탐색 채택 (SupCon은 P1 이후 점진 도입)"],
        ["c2 후방 복원\n(Q2)", "Face 마스킹 교사-학생 증류\n48D 협응 동역학 피처", "흉곽 로컬 좌표계 48D\n불확실성 분산 추정 및 로짓 축소", "흉곽 로컬 좌표계 48D + 불확실성 분산 출력 + 교사-학생 증류 하이브리드"],
        ["16상태 모델\n(Q3)", "Exact 16-State CRF\nDynamic Pair/Triple Energy", "4-Defect Factor Graph GNN\n상태 전수 열거 추론", "Exact 16-State Energy Head 채택 (Unary 4 + Pair 6 + Triple 1)"],
        ["뼈 길이 정규화\n(Q4)", "Stand 구간 Median 추정\nConstrained IK 최적화", "계층적 FK 순회 투영\nDual-Stream 원천 좌표 보존", "계층적 FK + Dual-Stream 보존 (Raw Normalized는 c4 지면고정에 필수 보존)"],
        ["ST-GCN 결합\n(Q5)", "FiLM 중간 주입\nResidual Late Fusion", "14D 채널 구성\n2-STGCN + 4-Transformer Late Fusion", "14D 입력 채널 + 2-STGCN/2-Transformer + Gated Late Fusion"],
        ["신뢰도 보정\n(Q6)", "Diagonal/Low-Rank Scaling\nDirichlet Calibration", "View-Conditioned Vector Scaling\nBounded Temperature Network", "View-Conditioned Vector Scaling + Bounded Temperature (RETAKE ≤ 4% 제약)"]
    ]
    format_table(t_comp, widths_comp, headers_comp, data_comp, header_bg="1A365D")
    doc.add_paragraph()

    # 4. 단계별 실행 로드맵 (P0 -> P1 -> P2)
    style_heading(doc.add_paragraph(), "4. 단계별 통합 실행 로드맵 및 검증 프로토콜", level=1)
    doc.add_paragraph(
        "외부 AI의 강력한 경고대로 9명 피험자 환경에서 모든 신기술을 일거에 도입하면 과적합 원인 규명이 불가능해집니다. 따라서 위험도가 낮고 효과가 즉각적인 기법부터 점진적으로 계승(Copy-and-Refine)하는 3단계 로드맵을 수립합니다."
    )

    t_p = doc.add_table(rows=4, cols=4)
    headers_p = ["우선순위", "적용 기술 및 모듈", "목표 지표 및 기대 효과", "작업 영역 및 기존 코드 영향"]
    widths_p = [Inches(1.0), Inches(2.2), Inches(2.0), Inches(1.7)]
    data_p = [
        ["Phase 0\n(즉시 실행\nDay 1~2)", "1) False-Clean Asymmetric Loss\n2) 5-Fold OOF Precision 70% 제약 임계값 탐색\n3) Skeletal Length Normalization (Dual-Stream)", "Clean Precision ≥ 68.0%\nClean F1 ≥ 72.0%\n좌표계 흔들림 제거", "기존 c1~c4 GBDT 및 MoE 구조 100% 보존. 전처리 및 손실/후처리만 수정."],
        ["Phase 1\n(구조 혁신\nDay 3~4)", "1) Exact 16-State CRF/Energy Head (Unary+Pair+Triple)\n2) c2 불확실성 출력 및 흉곽 로컬 48D\n3) View-Conditioned Vector Scaling & Risk RETAKE", "4-Defect EMR ≥ 76.0%\nClean Precision ≥ 70.0%\n후방 시점 c2 안정화", "MoE 상단 헤드를 Exact 16-State Energy Head로 교체. c2 모델에 분산 헤드 추가."],
        ["Phase 2\n(딥러닝 확장\nDay 5~6)", "1) 14D 입력 2-STGCN + 2-Transformer\n2) MoE 챔피언과의 Gated Late Fusion\n3) Cross-View Teacher-Student 증류", "4-Defect EMR ≥ 80.0%\nClean F1 ≥ 75.0%\nRETAKE ≤ 3.5%", "신규 딥러닝 보완 서브모듈 구축 및 Late Fusion 연결. 파라미터 1M 이하 엄격 제한."]
    ]
    format_table(t_p, widths_p, headers_p, data_p, header_bg="2B6CB0")
    doc.add_paragraph()

    # 5. 다차원 감사 및 엄격한 검증 체크리스트
    style_heading(doc.add_paragraph(), "5. 다차원 성능 감사 및 검증 불변식 (Quality Invariants)", level=1)
    doc.add_paragraph(
        "vidio-health-trainer 작업 표준 규정에 따라, v10.0 배포 전 다음의 6대 감사 기준을 전수 통과해야 합니다.\n"
        "1. 피험자 9명 전원 격리 검증: StratifiedGroupKFold(subject_id) 5-Fold OOF 성능으로만 평가하며, validation fold 내 임계값 재탐색 누출을 영구 금지합니다.\n"
        "2. 5대 시점(A~E) 전수 감사: 시점 C(정면) 성능만 보지 않고, 취약 시점인 시점 A(우측후방), 시점 E(좌측후방)의 c2 F1 및 결함별 오분류율을 전수 대조합니다.\n"
        "3. Near-Clean 오분류율 감사: 실제 1개 결함 샘플이 Clean으로 잘못 분류되는 비율을 독립 측정하여 이전 버전 대비 50% 이상 감소했는지 확인합니다.\n"
        "4. c4 지면 접지 변위 무결성 검증: Skeletal Normalization 후에도 c4의 발들림/발회전 신호가 유실되지 않았는지 c4 단독 F1을 비교 감사합니다.\n"
        "5. 파라미터 경량화 및 추론 속도: STGT 모듈 추가 후에도 단일 시퀀스(64프레임) 추론 지연시간이 30ms 이내를 유지하는지 프로파일링합니다."
    )

    doc.save(output_path)
    print(f"Report successfully saved: {output_path}")

if __name__ == '__main__':
    targets = [
        r"D:\자주프 문서\5주차\vidio-health-trainer_외부AI_6대기술자문_심층비교분석_및_v10_통합실행로드맵.docx",
        r"D:\자주프 문서\외부ai 응답\vidio-health-trainer_외부AI_6대기술자문_심층비교분석_및_v10_통합실행로드맵.docx"
    ]
    for target in targets:
        create_report(target)
