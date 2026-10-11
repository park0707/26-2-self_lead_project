# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_week5_standard_weekly_report.py
Purpose: D:\\자주프 문서\\기타\\보고서 양식.md 표준 규격을 100% 준수한 5주차 주차 연구 보고서 생성
Target: D:\\자주프 문서\\5주차\\보고서\\26-2학기_자기주도프로젝트_5주차_보고서_박현준.docx
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

def set_cell_background(cell, hex_color):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def style_heading(p, text, level=1):
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.font.name = '맑은 고딕'
        run.font.size = Pt(14)
        run.bold = True
        run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.name = '맑은 고딕'
        run.font.size = Pt(11.5)
        run.bold = True
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    elif level == 3:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.font.name = '맑은 고딕'
        run.font.size = Pt(10)
        run.bold = True
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def add_table_title(doc, number, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run_num = p.add_run(f"표 {number}. ")
    run_num.bold = True
    run_num.font.name = '맑은 고딕'
    run_num.font.size = Pt(9.5)
    run_num.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    run_title = p.add_run(title)
    run_title.font.name = '맑은 고딕'
    run_title.font.size = Pt(9.5)
    run_title.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def add_figure_caption(doc, number, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_num = p.add_run(f"그림 {number}. ")
    run_num.bold = True
    run_num.font.name = '맑은 고딕'
    run_num.font.size = Pt(9)
    run_num.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    run_title = p.add_run(title)
    run_title.font.name = '맑은 고딕'
    run_title.font.size = Pt(9)
    run_title.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

def format_standard_table(table, col_widths, headers, data, header_bg="1A365D", note=None, doc=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_background(hdr_cells[i], header_bg)
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = '맑은 고딕'
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.font.size = Pt(9)
            
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=75, bottom=75, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 and len(str(val)) < 25 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = '맑은 고딕'
                r.font.size = Pt(8.5)
                if "▲" in str(val) or "돌파" in str(val) or "최고" in str(val) or "완결" in str(val) or "확정" in str(val):
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x1B, 0x4D, 0x3E)
                elif "▼" in str(val):
                    r.font.color.rgb = RGBColor(0x9B, 0x2C, 0x2C)
                else:
                    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    for row in table.rows:
        for c_idx, w in enumerate(col_widths):
            row.cells[c_idx].width = w
            
    if note and doc:
        p_note = doc.add_paragraph()
        p_note.paragraph_format.space_before = Pt(2)
        p_note.paragraph_format.space_after = Pt(8)
        r_note = p_note.add_run(f"※ {note}")
        r_note.font.name = '맑은 고딕'
        r_note.font.size = Pt(8)
        r_note.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

def add_code_or_box(doc, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, "F7FAFC")
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="2B6CB0"/>
        <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>
    </w:tcBorders>
    '''
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.20
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def main():
    doc = Document()
    
    # 여백: A4 용지 상하좌우 20mm (약 56.7pt = 0.787인치)
    for s in doc.sections:
        s.top_margin = Inches(0.79)
        s.bottom_margin = Inches(0.79)
        s.left_margin = Inches(0.79)
        s.right_margin = Inches(0.79)

    # ========================================================
    # 2. 표지 및 기본 정보 (양식 규칙 2)
    # ========================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(8)
    title_p.paragraph_format.space_after = Pt(6)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t_run = title_p.add_run("26-2학기 자기주도프로젝트 5주차 보고서")
    t_run.font.name = '맑은 고딕'
    t_run.font.size = Pt(20)
    t_run.bold = True
    t_run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    author_p = doc.add_paragraph()
    author_p.paragraph_format.space_before = Pt(0)
    author_p.paragraph_format.space_after = Pt(4)
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a_run = author_p.add_run("202220722 소프트웨어학과 박현준")
    a_run.font.name = '맑은 고딕'
    a_run.font.size = Pt(11)
    a_run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(16)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_run = sub_p.add_run("과제명: vidio-health-trainer 바벨 스쿼트 세부 오류 모델 고도화 및 v11.0 통합 진단 AI 엔진 구축")
    s_run.font.name = '맑은 고딕'
    s_run.font.size = Pt(10)
    s_run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    # ========================================================
    # 3. 목차 규칙 (양식 규칙 3: 3개 열 표)
    # ========================================================
    p_toc_title = doc.add_paragraph()
    p_toc_title.paragraph_format.space_before = Pt(6)
    p_toc_title.paragraph_format.space_after = Pt(4)
    r_toc = p_toc_title.add_run("목차")
    r_toc.font.name = '맑은 고딕'
    r_toc.font.size = Pt(12)
    r_toc.bold = True
    r_toc.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    toc_headers = ["장", "대주제", "세부 목차"]
    toc_widths = [Inches(0.8), Inches(2.7), Inches(3.4)]
    toc_data = [
        ["01", "1. 이번 주 연구 목표 및 범위", "1.1 주차 목표\n1.2 연구 범위 및 주요 과제"],
        ["02", "2. 수행 내용", "2.1 조사·분석·설계 내용\n2.2 구현·전처리·실험 내용"],
        ["03", "3. 결과 및 검증", "3.1 정량 결과\n3.2 정성 결과 및 해석\n3.3 표·그림·예시 자료"],
        ["04", "4. 이슈 및 한계점", "4.1 확인된 문제\n4.2 원인 분석 및 대응 방안"]
    ]
    t_toc = doc.add_table(rows=len(toc_data) + 1, cols=len(toc_headers))
    format_standard_table(t_toc, toc_widths, toc_headers, toc_data, header_bg="1A365D")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ========================================================
    # 본문 1. 이번 주 연구 목표 및 범위 (양식 표준 구조)
    # ========================================================
    p_c1 = doc.add_paragraph()
    style_heading(p_c1, "1. 이번 주 연구 목표 및 범위", level=1)

    p_s11 = doc.add_paragraph()
    style_heading(p_s11, "1.1 주차 목표", level=2)

    p_txt11 = doc.add_paragraph()
    p_txt11.paragraph_format.line_spacing = 1.25
    p_txt11.paragraph_format.space_after = Pt(4)
    r = p_txt11.add_run(
        "본 연구의 공식 명칭은 vidio-health-trainer이며, 5주차 연구 목표는 바벨 스쿼트(Barbell Squat) 운동의 AI 기반 다중 결함 및 정상/오류 통합 진단 엔진을 완성하는 것으로 설정하였다. "
        "초기 통합 모델(v6.0)은 정상 판정 시 부위 간 모순 충돌(False Clean 오진)과 후방 카메라 시점에서의 얼굴 미관측 오진, 카메라 깊이 왜곡에 따른 무릎 오진 등의 한계로 인해 완전 일치율(EMR)이 69.40%에 머물렀다. "
        "따라서 이번 주차에는 외부 AI 자문단이 제시한 6대 핵심 기술 로드맵을 체계적으로 검증 및 탑재하여, 4대 세부 결함 모델(c1~c4)을 고도화하고 최종 마스터 프로덕션 챔피언인 v11.0 통합 진단 엔진을 구축함으로써 4-Defect EMR 75% 이상, Clean F1 75% 이상의 상용 배포 수준 무결성을 확보하는 것을 주차 목표로 확정하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    p_s12 = doc.add_paragraph()
    style_heading(p_s12, "1.2 연구 범위 및 주요 과제", level=2)

    p_txt12 = doc.add_paragraph()
    p_txt12.paragraph_format.line_spacing = 1.25
    p_txt12.paragraph_format.space_after = Pt(6)
    r = p_txt12.add_run(
        "이번 주차에 설정된 구체적인 연구 범위 및 중점 수행 과제는 다음과 같다.\n"
        "• 과제 1 (4대 세부 오류 모델 고도화): c1(척추 중립성), c2(시선 및 고개 정면), c3(무릎 외반), c4(발바닥 지면 고정) 각각의 신체 도메인 특화 피처와 앙상블 아키텍처를 구축하고 표준 3-State 인터페이스를 정립한다.\n"
        "• 과제 2 (메타 융합 아키텍처 구축): Level 1 계층적 로짓 온도 보정(ECE 1.62%)과 39D View-Conditioned Mixture-of-Experts(MoE) 메타 신경망, Reverse-ASL 손실 함수를 개발하여 부위 간 상호작용을 통합한다.\n"
        "• 과제 3 (시공간 신경망 잔차 결합): MediaPipe 33관절 기반 14D 관절 채널 시공간 텐서를 구축하고, 7.5만 파라미터 경량 ST-GCN + Temporal Transformer 잔차 헤드를 Late Fusion 방식으로 결합한다.\n"
        "• 과제 4 (시점별 벡터 스케일링 및 위험도 게이트): 5대 카메라 시점(시점 A~E)별 기준 로짓 편향을 교정하는 16D 벡터 스케일러와 엔트로피 마진 동적 위험도 게이트를 구축한다.\n"
        "• 과제 5 (전수 실측 검증 및 마스터 동결): AI-Hub 2,605건 전수 데이터셋(9인 전 피험자, 5대 전 시점)에 대해 5-Fold StratifiedGroupKFold 교차검증을 수행하고 최종 마스터 v11.0으로 영구 확정한다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    # ========================================================
    # 본문 2. 수행 내용 (양식 표준 구조)
    # ========================================================
    p_c2 = doc.add_paragraph()
    style_heading(p_c2, "2. 수행 내용", level=1)

    p_s21 = doc.add_paragraph()
    style_heading(p_s21, "2.1 조사·분석·설계 내용", level=2)

    p_txt21 = doc.add_paragraph()
    p_txt21.paragraph_format.line_spacing = 1.25
    p_txt21.paragraph_format.space_after = Pt(6)
    r = p_txt21.add_run(
        "1) AI-Hub 바벨 스쿼트 데이터셋 분석\n"
        "AI-Hub 운동 영상 데이터셋 중 바벨 스쿼트 2,605건을 전수 분석하였다. 피험자 9명(피험자 1~9번)과 5대 카메라 시점(시점 A: 우측 후방 45°, 시점 B: 우측 전방 45°, 시점 C: 정면 0°, 시점 D: 좌측 전방 45°, 시점 E: 좌측 후방 45°)의 구성을 파악하였다. "
        "특히 후방 시점(시점 A, E)에서는 신체 구조상 안면 랜드마크가 완전히 가려져 c2(시선) 모델의 오진율이 급증하는 시점 편향(View Bias) 문제를 식별하였다.\n\n"
        "2) 1단계 운동 분류 모델 과적합 원인 분석 및 대조 설계\n"
        "과거 1단계 운동 분류 모델이 AI-Hub 스튜디오의 특정 배경, 카메라 높이, 피험자 신장 등 원시 좌표(Raw Coordinates)를 딥러닝이 직접 암기하여 외부 일반화 성능이 저하되었던 문제를 분석하였다. "
        "이를 원천 차단하기 위해 2단계 진단 엔진에서는 원시 좌표 학습을 완전히 배제하고, 신체 크기나 카메라 거리에 무관한 '체간 로컬 좌표계', '하지 뼈 길이 정규화', '무차원 생체역학 비율'만을 입력 피처로 채택하도록 수학적 불변성(Invariance) 설계를 완료하였다.\n\n"
        "3) 외부 AI 자문 6대 기술 로드맵 설계\n"
        "• 해결책 1 (Reverse-ASL): 결함 샘플을 정상으로 오판하는 False Clean에 비대칭 강력 가중치(lambda_fc=3.0)를 부여하는 비대칭 손실 함수를 설계하였다.\n"
        "• 해결책 2 (c2 v8 Posterior Shrinkage): 흉곽 로컬 축을 생성하여 가슴 기준 머리 각도를 추출하고, 후방 시점에서 안면 미관측 시 불확실성 분산(sigma^2=0.85)을 통해 사후 확률을 0.5로 자동 수축시키는 메커니즘을 설계하였다.\n"
        "• 해결책 3 (Exact 16-State Structured Energy Head): 4대 결함의 16개 조합 상태를 마르코프 랜덤 필드(MRF) 에너지 포텐셜로 모델링하여 생체역학적 상호 모순을 수학적으로 억제하도록 설계하였다.\n"
        "• 해결책 4 (Skeletal Length Normalization): 대퇴골과 경골 길이를 1.0 표준으로 재조정하는 Stream A(정규화)와 Stream B(원천)의 Dual-Stream 잔차 앙상블(3:7)을 설계하였다.\n"
        "• 해결책 5 (14D ST-GCN + Transformer): 33개 관절의 공간 그래프 합성곱과 64프레임 시간 트랜스포머 인코더를 결합하되, 파라미터를 7.5만 개로 극소화하고 잔차 가중치를 8%(g=0.08)로 묶는 Bounded Late Fusion을 설계하였다.\n"
        "• 해결책 6 (View-Conditioned Vector Scaling & Dynamic Risk Gate): 5대 시점별 전용 스케일/바이어스 보정 및 엔트로피 위험도 R(x) <= 0.45 기반 선택적 거부권을 설계하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    p_s22 = doc.add_paragraph()
    style_heading(p_s22, "2.2 구현·전처리·실험 내용", level=2)

    p_txt22 = doc.add_paragraph()
    p_txt22.paragraph_format.line_spacing = 1.25
    p_txt22.paragraph_format.space_after = Pt(6)
    r = p_txt22.add_run(
        "1) 데이터 전처리 및 14D 관절 채널 텐서 구축\n"
        "프로젝트 표준 규정(Rule 2)에 따라 AI-Hub 2,605건 원천 이미지 시퀀스로부터 MediaPipe Full Pose 33개 관절을 추출하고, 64프레임 3차 스플라인(Cubic Spline) 보간 데이터셋을 구축하였다. "
        "이를 기반으로 골반 중심 상대 좌표(3D), 관절 속도(3D), 관절 가속도(3D), 뼈 단위 방향 벡터(3D), 뼈 길이(1D), 속력 스칼라(1D)로 구성된 14D 시공간 텐서 (2605, 14, 64, 33)를 고속 추출하여 squat_14d_spatiotemporal_tensor_2605.npz로 캐싱하였다.\n\n"
        "2) 4대 세부 오류 모델 구현\n"
        "• c1 모델 (c1_model_v11_1.py): SpinePose 척추 9분절 회귀 모델과 3-Way Tri-Ensemble(GBDT 10% + Diet-GBDT 40% + TCN 50%)을 결합하고, 정상 확신 구간(p<=0.35)에서 TCN 분산을 면제하는 Gating Waiver를 구현하였다.\n"
        "• c2 모델 (c2_model_v8.py): 흉곽 로컬 좌표계 19D 피처 추출기 및 후방 시점 안면 비가시성 감지 시 불확실성 사후 수축 알고리즘을 구현하였다.\n"
        "• c3 모델 (c3_model_v6.py): 관상면 정사영 투영 22D 피처와 하지 뼈 정규화 Stream A(30%) + Stream B(70%) 잔차 결합기를 구현하였다.\n"
        "• c4 모델 (c4_model_v5_1.py): 족부 6개 랜드마크 접지 변위 추적 및 켑스트럼 FFT 스펙트럼 저크(Jerk), 미세 떨림(Wobble) 25D 결합 모델을 구현하였다.\n\n"
        "3) 경량 시공간 신경망 구현 및 피험자 격리 5-Fold OOF 학습\n"
        "stgcn_transformer_net.py에 2-Layer SGCN(14->32->64) 및 2-Layer Temporal Transformer Encoder를 총 75,765개 파라미터로 구현하였다. "
        "9인 피험자를 엄격히 분리하는 StratifiedGroupKFold(n_splits=5) 교차검증을 수행하여 검증 폴드 피험자에 대한 OOF 잔차 로짓(oof_delta_z16)을 산출하고, 최적 전방 잔차 게이트 가중치(g_front=0.08)를 그리드 탐색으로 도출하였다.\n\n"
        "4) 통합 진단 엔진 v11.0 구현\n"
        "babel_squat_integrated_v11.py에 5계층 파이프라인(Layer 1: 서브모델 -> Layer 2: ECE 온도 보정 및 MoE -> Layer 3: ST-GCN 잔차 융합 -> Layer 4: 시점별 벡터 스케일링 -> Layer 5: 품질 게이트 및 동적 위험도 선별)을 구축하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    # ========================================================
    # 본문 3. 결과 및 검증 (양식 표준 구조)
    # ========================================================
    p_c3 = doc.add_paragraph()
    style_heading(p_c3, "3. 결과 및 검증", level=1)

    p_s31 = doc.add_paragraph()
    style_heading(p_s31, "3.1 정량 결과", level=2)

    p_txt31 = doc.add_paragraph()
    p_txt31.paragraph_format.line_spacing = 1.25
    p_txt31.paragraph_format.space_after = Pt(4)
    r = p_txt31.add_run(
        "AI-Hub 2,605건 전수 데이터셋에 대해 5주차 연구 기간 동안 진화한 버전별 핵심 정량 성능과 최종 확정된 v11.0의 세부 지표를 산출하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    # 표 1. 주간 버전별 성능 진화 추이
    add_table_title(doc, 1, "바벨 스쿼트 통합 진단 엔진 주간 버전별 성능 진화 추이 (AI-Hub 2,605건)")
    t1_headers = ["버전 명칭", "핵심 적용 기술", "Clean F1", "Precision", "Recall", "4-EMR", "Coverage"]
    t1_widths = [Inches(1.0), Inches(2.3), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9)]
    t1_data = [
        ["v6.0", "초기 계층적 융합 베이스라인", "67.17%", "67.67%", "66.67%", "69.40%", "87.50%"],
        ["v10.0", "해결책 1: Reverse-ASL 손실", "71.60%", "75.41%", "68.15%", "71.60%", "90.06%"],
        ["v10.1", "해결책 3: 구조화 에너지 헤드", "75.28%", "72.03%", "78.85%", "73.34%", "91.30%"],
        ["v10.2", "해결책 2: c2 v8 불확실성 사후 수축", "75.46%", "72.03%", "79.23%", "73.69%", "91.30%"],
        ["v10.3", "해결책 4: 하지 뼈 정규화 Dual-Stream", "77.26%", "73.29%", "81.68%", "73.77%", "91.20%"],
        ["v10.4", "해결책 6: 시점별 벡터 스케일링", "76.92%", "71.92%", "82.68%", "76.40%", "89.94%"],
        ["v11.0 (최종)", "해결책 5: ST-GCN 잔차 융합 확정", "77.21% ▲", "72.41% ▲", "82.68%", "76.42% ▲", "90.02% ▲"]
    ]
    t1 = doc.add_table(rows=len(t1_data) + 1, cols=len(t1_headers))
    format_standard_table(t1, t1_widths, t1_headers, t1_data, header_bg="1A365D",
                          note="AI-Hub 바벨 스쿼트 2,605건 전수 표본 대상, 9인 전 피험자 및 5대 전 시점 격리 검증 기준", doc=doc)

    # 표 2. 4대 세부 결함 모델 실측 성능
    add_table_title(doc, 2, "v11.0 탑재 4대 세부 결함(c1~c4) 모델 실측 성능 감사 지표")
    t2_headers = ["결함 코드", "진단 부위 및 명칭", "탑재 세부 모델", "F1-Score", "Precision", "Recall", "무결성 상태"]
    t2_widths = [Inches(0.8), Inches(1.8), Inches(2.2), Inches(0.9), Inches(0.9), Inches(0.9), Inches(1.0)]
    t2_data = [
        ["c1", "척추의 중립성", "BabelSquatC1ModelV11_1", "97.13% ▲", "95.88%", "98.41%", "프로덕션 최상위"],
        ["c2", "고개 정면 및 시선", "BabelSquatC2ModelV8", "87.40% ▲", "85.20%", "89.72%", "역대 최고 경신"],
        ["c3", "무릎 외반 궤적", "BabelSquatC3ModelV6", "96.77%", "96.12%", "97.43%", "Dual-Stream 안정"],
        ["c4", "발바닥 지면 고정", "BabelSquatC4ModelV5_1", "92.20%", "90.85%", "93.59%", "92% 돌파 유지"]
    ]
    t2 = doc.add_table(rows=len(t2_data) + 1, cols=len(t2_headers))
    format_standard_table(t2, t2_widths, t2_headers, t2_data, header_bg="2B6CB0",
                          note="v11.0 선택적 판정(Coverage 90.02%) 기준 산출", doc=doc)

    # 표 3. 5대 카메라 시점별 성능 감사 지표
    add_table_title(doc, 3, "v11.0 공식 5대 카메라 시점(시점 A~E) 실측 감사 지표")
    t3_headers = ["공식 시점 명칭", "시점 코드", "Clean F1", "Clean Precision", "Clean Recall", "4-EMR", "시점 특화 전략"]
    t3_widths = [Inches(1.8), Inches(0.8), Inches(0.9), Inches(1.0), Inches(0.9), Inches(0.9), Inches(1.3)]
    t3_data = [
        ["시점 A: 우측 후방 (45도)", "view5", "71.43%", "64.52%", "80.00%", "69.63%", "보수적 잔차 게이트 (g=0)"],
        ["시점 B: 우측 전방 (45도)", "view2", "80.70%", "74.19%", "88.46%", "76.72% ▲", "전방 잔차 게이트 (g=0.08)"],
        ["시점 C: 정면 (0도)", "view3", "85.71% ▲", "77.42%", "96.00%", "81.29% ▲", "좌우 대칭성 보정 (g=0.08)"],
        ["시점 D: 좌측 전방 (45도)", "view4", "82.14%", "79.31%", "85.19%", "79.96%", "전방 잔차 게이트 (g=0.08)"],
        ["시점 E: 좌측 후방 (45도)", "view1", "63.83%", "65.22%", "62.50%", "73.30%", "후방 불확실성 수축 적용"]
    ]
    t3 = doc.add_table(rows=len(t3_data) + 1, cols=len(t3_headers))
    format_standard_table(t3, t3_widths, t3_headers, t3_data, header_bg="2D3748",
                          note="공식 5대 카메라 시점 명명 규격(Rule 4) 전수 준수", doc=doc)

    p_s32 = doc.add_paragraph()
    style_heading(p_s32, "3.2 정성 결과 및 해석", level=2)

    p_txt32 = doc.add_paragraph()
    p_txt32.paragraph_format.line_spacing = 1.25
    p_txt32.paragraph_format.space_after = Pt(6)
    r = p_txt32.add_run(
        "1) 정상 판정 시 부위 간 모순 충돌 해결 (Clean F1 67.2% -> 77.2%)\n"
        "초기 모델은 4대 결함 모델이 독립적으로 정상이라 판단하더라도, 상호 연관성이 없는 상태에서 결합되어 정상 재현율이 66.7%로 낮았다. "
        "v11.0은 Exact 16-State 구조화 에너지 헤드를 통해 부위 간 생체역학적 결합 에너지(Pairwise Compatibility)를 사전에 정의함으로써 비정상적인 조합을 억제하고 완전 정상(State 0)의 판별력을 10%p 이상 끌어올렸다.\n\n"
        "2) 후방 카메라 시점의 안면 미관측 딜레마 극복 (시점 A, E 무결성)\n"
        "후방에서 얼굴이 가려질 때 기존 모델은 억지로 시선 오류를 추론하여 오진을 양산하였다. "
        "c2 v8의 Posterior Shrinkage와 v11.0의 OBSERVABLE 등급화는 얼굴이 안 보일 때 불확실성 분산을 부여하여 사후 확률을 0.5로 수축시키고, c2 평가를 'NOT_EVALUATED'로 안전 격리하여 후방 시점의 Clean F1을 71.4%로 방어하였다.\n\n"
        "3) 카메라 원근 왜곡 및 무릎 z축 요동 해소 (c3 96.8% 달성)\n"
        "피험자의 다리 길이 차이나 카메라 거리에 따라 무릎 간격이 왜곡되는 문제를 하지 뼈 길이 정규화 Dual-Stream(7:3)으로 해결하였다. "
        "대퇴골과 경골의 물리적 길이가 1.0 표준으로 고정됨에 따라 순수한 외반 각도만을 측정할 수 있게 되었다.\n\n"
        "4) 정면 시점의 동적 타이밍 제압 (시점 C Clean F1 85.7% 급등)\n"
        "정면 시점(시점 C)에서는 ST-GCN + Transformer 잔차 결합을 통해 좌우 무릎과 발목의 미세한 동적 대칭성을 포착함으로써 Clean F1이 84.21%에서 85.71%로 +1.50%p 급등하였고, 4-EMR 또한 81.29%를 달성하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    p_s33 = doc.add_paragraph()
    style_heading(p_s33, "3.3 표·그림·예시 자료", level=2)

    # 그림 1 도식
    add_code_or_box(doc,
"""[입력 64프레임 33관절 3D 시퀀스 (64, 33, 3) + 카메라 시점]
       │
       ├───> [Layer 1: 4대 독립 서브모델 진단]
       │     • c1 (척추): SpinePose 9분절 3-Way Tri-Ensemble (F1 97.1%)
       │     • c2 (시선): 흉곽 로컬 좌표계 + 후방 불확실성 사후 수축 (F1 87.4%)
       │     • c3 (무릎): 하지 뼈 정규화 Dual-Stream 7:3 앙상블 (F1 96.8%)
       │     • c4 (발바닥): 접지선 랜드마크 + 스펙트럼 저크 융합 (F1 92.2%)
       │
       ├───> [Layer 2: Level 1 온도 보정 & 39D View-MoE 메타 신경망]
       │     • 3대 서브전문가 + Exact 16-State 구조화 에너지 헤드 (모순 억제)
       │
       ├───> [Layer 3: 14D 관절 채널 경량 시공간 신경망 (ST-GCN + Transformer)]
       │     • 7.5만 파라미터, 시점별 차등 잔차 게이팅 (전방 g=0.08 / 후방 g=0.00)
       │
       ├───> [Layer 4: 5대 시점별 16D 벡터 스케일링] ──> 시점별 로짓 편향 보정
       │
       └───> [Layer 5: 품질 게이트(q_min=0.05) & 동적 위험도(R<=0.45) 선별]
                   │
                   ▼
       [최종 출력: 종합 상태(CLEAN/DEFECT/RETAKE) + 4대 부위별 상세 상태 및 맞춤 코칭]"""
    )
    add_figure_caption(doc, 1, "vidio-health-trainer v11.0 바벨스쿼트 통합 진단 엔진 5계층 아키텍처 다이어그램")

    # 코드 예시 1
    add_code_or_box(doc,
"""// v11.0 통합 진단 엔진 출력 JSON 표준 규격 예시
{
  "session_summary": {
    "status": "CLEAN",                 // 종합 판정: CLEAN / DEFECT / RETAKE
    "clean_tier": "VERIFIED",          // 정상 신뢰 등급: VERIFIED / OBSERVABLE / NONE
    "is_clean": true,
    "is_retake": false,
    "soft_risk_score": 0.082,          // 0.0 ~ 1.0 연속형 위험도 점수
    "summary_text": "바벨 스쿼트 4대 주요 부위의 정렬 및 동적 안정성이 완벽하게 유지되었습니다."
  },
  "component_reports": {
    "c1": { "status": "NORMAL", "probability": 0.041, "reason_code": "CLEAR_DECISION", "feedback": "척추의 중립 정렬이 바르게 유지되었습니다." },
    "c2": { "status": "NORMAL", "probability": 0.092, "reason_code": "CLEAR_DECISION", "feedback": "고개와 경추가 정면을 향해 안정적으로 유지되었습니다." },
    "c3": { "status": "NORMAL", "probability": 0.078, "reason_code": "CLEAR_DECISION", "feedback": "무릎이 발끝 방향과 완벽히 일치하여 외반 붕괴가 없습니다." },
    "c4": { "status": "NORMAL", "probability": 0.035, "reason_code": "CLEAR_DECISION", "feedback": "발바닥 전체가 지면에 안정적으로 밀착되었습니다." }
  }
}"""
    )
    add_figure_caption(doc, 2, "v11.0 통합 진단 엔진 출력 데이터 표준 JSON 규격 예시")

    # ========================================================
    # 본문 4. 이슈 및 한계점 (양식 표준 구조)
    # ========================================================
    p_c4 = doc.add_paragraph()
    style_heading(p_c4, "4. 이슈 및 한계점", level=1)

    p_s41 = doc.add_paragraph()
    style_heading(p_s41, "4.1 확인된 문제", level=2)

    p_txt41 = doc.add_paragraph()
    p_txt41.paragraph_format.line_spacing = 1.25
    p_txt41.paragraph_format.space_after = Pt(6)
    r = p_txt41.add_run(
        "• 문제 1 (상체 뼈 정규화 시 척추 모델 붕괴): 뼈 길이 정규화(v10.3) 기획 초기, 상하체 전신에 뼈 정규화를 강제 적용했을 때 c1(척추) 모델의 F1이 96.3%에서 94.1%로 -2.2%p 급락하는 성능 퇴행이 발생하였다.\n"
        "• 문제 2 (후방 시점 시공간 신경망 깊이 왜곡): 14D ST-GCN + Transformer 잔차 결합(v10.5) 초기 실험 시, 후방 시점(시점 A, E)에 잔차를 전방과 동일하게 가산했을 때 시점 A Clean F1이 71.4%에서 69.1%로 하락하는 현상이 관찰되었다.\n"
        "• 문제 3 (고정 데드밴드의 시점별 편향): 기존 고정 데드밴드([0.30, 0.36]) 필터링이 정면과 후방 시점 간의 기본 로짓 차이를 반영하지 못하여 특정 시점에서 불필요한 재촬영 권고가 발생하는 한계가 확인되었다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    p_s42 = doc.add_paragraph()
    style_heading(p_s42, "4.2 원인 분석 및 대응 방안", level=2)

    p_txt42 = doc.add_paragraph()
    p_txt42.paragraph_format.line_spacing = 1.25
    p_txt42.paragraph_format.space_after = Pt(6)
    r = p_txt42.add_run(
        "• 원인 1 분석 및 조치: SpinePose 척추 9분절 모델은 사전 학습 당시 원천 흉곽과 어깨-골반 실제 비례를 기반으로 척추 곡률을 추정하므로, 상체 뼈 길이를 인위적으로 리스케일링하면 척추 기하학적 정렬이 파괴됨을 규명하였다. "
        "대응 방안으로 상체 정규화를 즉시 취소하고, 하체(대퇴-경골)만 정규화하는 Dual-Stream 앙상블 구조를 채택하여 c1은 원천 Stream B(100%)로 보존하고 c3만 정규화 Stream A(30%)를 융합하여 문제를 완벽히 해결하였다.\n\n"
        "• 원인 2 분석 및 조치: 후방 카메라 시점에서는 신체 전면 랜드마크(안면, 무릎 전면)의 z축 깊이 추정 오차가 전방 대비 컸으며, 이에 따라 3차원 그래프 전파 시 노이즈가 유입됨을 확인하였다. "
        "대응 방안으로 시점별 차등 잔차 게이팅(View-Conditioned Gating)을 도입하여, 전방 3대 시점(B, C, D)에는 g_front=0.08을 적용하여 타이밍 결함을 정밀 보정하고, 후방 2대 시점(A, E)에는 g_rear=0.00을 적용하여 베이스라인의 안정성을 100% 보존하였다.\n\n"
        "• 원인 3 분석 및 조치: 5대 시점별로 정상 시점 기준 로짓이 상이함(-1.16 ~ -1.90)에 기인함을 규명하였다. "
        "대응 방안으로 5대 시점 전용 16D 벡터 스케일러(a_v, b_v)를 탑재하고, 고정 구간 대신 16상태 엔트로피와 상위 2개 상태 간 마진 갭을 결합한 동적 위험도 지표 R(x) <= 0.45를 구축하여 불필요한 재촬영을 10.06%에서 9.98%로 감소시키고 Coverage 90.02%를 달성하였다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)

    # 최종 문서 저장
    out_docx_path = r"D:\자주프 문서\5주차\보고서\26-2학기_자기주도프로젝트_5주차_보고서_박현준.docx"
    os.makedirs(os.path.dirname(out_docx_path), exist_ok=True)
    doc.save(out_docx_path)
    print(f"[SUCCESS] Saved Standard Weekly Report to: {out_docx_path}")

if __name__ == '__main__':
    main()
