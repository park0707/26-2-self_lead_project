"""
외부 AI 자문용 기술 배경지식 명세서 생성 스크립트
파일명: 외부 ai 용 배경지식 문서.docx -> 외부 ai 용 배경지식 문서.pdf
프로젝트: vidio-health-trainer
"""

import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_callout(doc, title, text, border_color="1A365D", bg_color="F7FAFC"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
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
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}")
    run_t.font.name = '맑은 고딕'
    run_t.font.size = Pt(10.5)
    run_t.bold = True
    run_t.font.color.rgb = RGBColor(int(border_color[:2], 16), int(border_color[2:4], 16), int(border_color[4:], 16))
    
    p2 = cell.add_paragraph()
    p2.paragraph_format.line_spacing = 1.30
    p2.paragraph_format.space_after = Pt(2)
    run_b = p2.add_run(text)
    run_b.font.name = '맑은 고딕'
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def format_table(table, col_widths, headers, data, header_bg="1A365D"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    hdr_row = table.rows[0]
    for idx, heading in enumerate(headers):
        cell = hdr_row.cells[idx]
        cell.text = heading
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = '맑은 고딕'
            run.font.size = Pt(9)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    for row_idx, row_data in enumerate(data):
        row = table.rows[row_idx + 1]
        bg = "FFFFFF" if row_idx % 2 == 0 else "F8FAFC"
        for col_idx, cell_value in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = str(cell_value)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            p = cell.paragraphs[0]
            if col_idx == 0 or len(cell_value) > 20:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = '맑은 고딕'
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
                
    for row in table.rows:
        for c_idx, w in enumerate(col_widths):
            row.cells[c_idx].width = w

def style_heading(p, text, level=1):
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = '맑은 고딕'
    run.bold = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    elif level == 2:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    elif level == 3:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def add_body_paragraph(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(6)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = '맑은 고딕'
        rb.font.size = Pt(9.5)
        rb.bold = True
        rb.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    r = p.add_run(text)
    r.font.name = '맑은 고딕'
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    return p

def build_document():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # --------------------------------------------------------
    # 헤더 및 타이틀
    # --------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_after = Pt(2)
    t_run = title_p.add_run("vidio-health-trainer AI 엔진 기술 배경지식 명세서")
    t_run.font.name = '맑은 고딕'
    t_run.font.size = Pt(18)
    t_run.bold = True
    t_run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run(
        "바벨 스쿼트 4대 세부 오류 모델 및 통합 진단 AI 엔진(v11.0) 아키텍처 상세, 입력/처리 파이프라인, 현행 성능 벤치마크 및 성능 극대화(Clean F1 & EMR 80%+) 자문 의뢰서"
    )
    sub_run.font.name = '맑은 고딕'
    sub_run.font.size = Pt(10.5)
    sub_run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    # Callout: 외부 AI를 위한 문서 전달 목적
    add_callout(
        doc,
        title="외부 AI 전문가 자문 의뢰 개요 (Consultation Objective)",
        text="본 문서는 외부 최상위 AI 모델(Claude 3.7 Sonnet, GPT-4.5/o1/o3 등)에게 본 프로젝트(vidio-health-trainer)의 운동 자세 진단 파이프라인 전체를 완벽히 이해시키고, "
             "현재 달성된 [Clean F1 77.21%] 및 [4-Defect EMR 76.42%]를 차세대 아키텍처 개선을 통해 [80.0% ~ 85.0% 이상]으로 극대화하기 위한 "
             "구체적이고 실현 가능한 아키텍처 혁신안, 손실 함수 설계, 멀티모달 융합 방안을 도출받기 위해 작성된 고밀도 기술 명세서입니다.\n\n"
             "본 문서에 명시된 데이터셋 스펙, 피처 엔지니어링 인터페이스, 신경망 레이어 구성, 3-State 선택적 분류 규칙, 평가 지표 정의를 기반으로 "
             "이론적 직관과 실제 코드 레벨의 정밀한 모델링 솔루션을 제안해 주십시오.",
        border_color="1B4D3E",
        bg_color="F0FDF4"
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 1. 프로젝트 및 당면 핵심 목표
    # ========================================================
    p1 = doc.add_paragraph()
    style_heading(p1, "1. 프로젝트 개요 및 핵심 고도화 목표", level=1)

    add_body_paragraph(
        doc,
        "vidio-health-trainer 프로젝트는 단일 스마트폰/웹캠 환경에서 딥러닝 기반 포즈 추정(MediaPipe Pose)을 활용하여 "
        "사용자의 3대 웨이트 트레이닝(바벨 스쿼트, 데드리프트, 벤치프레스) 자세를 실시간 분석하고, 인체 관절별 결함을 정밀 코칭하는 지능형 피트니스 AI 시스템입니다. "
        "본 문서의 타깃 종목은 가장 높은 관절 자유도와 시점 왜곡을 지닌 '바벨 스쿼트(Barbell Squat)'입니다."
    )

    headers_goal = ["핵심 지표 (Key Metric)", "현행 공식 마스터 (v11.0)", "차세대 목표 (Target)", "핵심 도전 과제 및 의미"]
    widths_goal = [Inches(1.8), Inches(1.3), Inches(1.3), Inches(2.5)]
    data_goal = [
        ["4-Defect EMR (완전 일치율)", "76.42%", "80.0% ~ 82.0% ▲", "4개 부위 동시 100% 무결점 판정 비율 (+4~6%p 증대)"],
        ["Clean F1-Score (정상 F1)", "77.21%", "80.0% ~ 85.0% ▲", "무결점 정상 스쿼트의 정밀도·재현율 조화평균 극대화"],
        ["Clean Precision (정상 정밀도)", "72.41%", "75.0% 이상 유지", "거짓 정상(False Clean) 방지 하드 제약(70%+) 지속 준수"],
        ["Coverage (판정 커버리지)", "90.02%", "90.0% 이상 유지", "상용화 최소 기준선인 90% 이상 유지 (무리한 보류 방지)"],
        ["RETAKE (안전 보류율)", "9.98%", "10.0% 이하 유지", "모호 구간 안전 보류율을 10% 미만으로 엄격히 통제"]
    ]
    t_goal = doc.add_table(rows=len(data_goal) + 1, cols=len(headers_goal))
    format_table(t_goal, widths_goal, headers_goal, data_goal, header_bg="1A365D")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 2. 데이터셋 규격 및 전처리 파이프라인
    # ========================================================
    p2 = doc.add_paragraph()
    style_heading(p2, "2. 데이터셋 규격 및 전처리 파이프라인 (Data & Preprocessing)", level=1)

    add_body_paragraph(
        doc,
        "■ [AI-Hub 원천 데이터 규격 및 전수 활용 원칙]:\n"
        "AI-Hub 피트니스 자세 교정 데이터셋에서 바벨 스쿼트 원천 시퀀스 총 2,605건을 사용합니다. "
        "일부 연구에서 시도하던 임의 데이터 삭제나 샘플링 편향을 원천 차단하기 위해, "
        "1번부터 9번까지 9인 전 피험자와 5대 전 카메라 시점을 단 1건의 누락 없이 100% 전수 학습 및 검증에 투입합니다."
    )

    headers_data = ["구분", "스펙 및 정의", "공학적 설계 및 처리 이유"]
    widths_data = [Inches(1.5), Inches(2.2), Inches(3.2)]
    data_data = [
        ["총 데이터 볼륨", "2,605개 모션 시퀀스", "AI-Hub 바벨 스쿼트 전수 데이터 (무삭제 전수 벤치마크)"],
        ["피험자 구성", "피험자 1번 ~ 9번 (전원 9명)", "체형, 신장, 근력 편차 반영. StratifiedGroupKFold로 분할"],
        ["교차 검증 전략", "5-Fold GroupKFold (피험자 격리)", "동일 피험자가 Train/Validation에 섞이는 Data Leakage 원천 차단"],
        ["시점 A (view5)", "우측 후방 (Right Back 45°)", "등 근육, 척추 곡률 관측 우수. 안면 가림 발생"],
        ["시점 B (view2)", "우측 전방 (Right Front 45°)", "무릎 가동범위 및 상체 기울기 동시 관측 최적 뷰"],
        ["시점 C (view3)", "정면 (Front View 0°)", "무릎 외반/내반(좌우 대칭) 관측 우수. Z축 깊이 왜곡 극심"],
        ["시점 D (view4)", "좌측 전방 (Left Front 45°)", "시점 B와 대칭적 관측 특성 보유"],
        ["시점 E (view1)", "좌측 후방 (Left Back 45°)", "후방 척추 관측 우수. 얼굴 랜드마크 완전 결손"],
        ["프레임 전처리 표준", "64프레임 3차 스플라인 보간", "원천 가변 프레임(30~120F)을 64프레임 균일 텐서로 정규화"],
        ["실서비스 입력 전처리", "단일 횟수 분할 후 64F 스플라인", "연속 영상에서 1회 반복을 엄밀 분할 후 64F 3차 스플라인 변환"]
    ]
    t_data = doc.add_table(rows=len(data_data) + 1, cols=len(headers_data))
    format_table(t_data, widths_data, headers_data, data_data, header_bg="2B6CB0")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 3. 4대 세부 오류 모델 (c1 ~ c4) 상세 아키텍처 및 입력/처리
    # ========================================================
    p3 = doc.add_paragraph()
    style_heading(p3, "3. 4대 세부 오류 모델 (c1 ~ c4) 상세 아키텍처 및 입출력 명세", level=1)

    add_body_paragraph(
        doc,
        "통합 엔진(Layer 1)의 입력으로 투입되는 4대 세부 결함 모델은 인체 해부학적 특성에 맞추어 고도로 독립화된 "
        "전용 피처 추출기와 이종(Heterogeneous) 머신러닝/딥러닝 앙상블로 구성되어 있습니다. 각 모델의 세부 사항은 다음과 같습니다."
    )

    # c1 상세
    style_heading(doc.add_paragraph(), "3.1 c1 척추 중립성 위반 모델 (BabelSquatC1ModelV11_1)", level=2)
    add_body_paragraph(
        doc,
        "1) 진단 대상 및 난제: 요추 굴곡(버트윙크, Butt Wink), 굿모닝 스쿼트(상체 과숙임), 요추 과신전(허리 꺾임). MediaPipe는 어깨와 골반 랜드마크만 제공하므로 척추 뼈마디가 부재하며, 정면/후방에서 Z축 압축 왜곡이 극심함.\n"
        "2) 입력 데이터: 64프레임 x 33개 3D 관절 좌표 (T=64, J=33, C=3).\n"
        "3) 핵심 기술 및 피처 엔지니어링:\n"
        "   - DFKI SpinePose 9분절 3D 가상 복원: 어깨-골반 사이 S1(천골), L5/L3/L1(요추), T8/T3(흉추), C7/C4/C1(경추)의 가상 관절점을 기하학적으로 복원하여 요추 2차 곡률(poly_a) 및 분절별 Cobb 각도 산출.\n"
        "   - 체간 로컬 정규 직교 기저계: 좌우 어깨와 골반 벡터로 인체 몸통 자체의 로컬 축(x, y, z)을 정의하여 카메라 회전 오차 소거.\n"
        "   - Tri-Ensemble 하이브리드: GBDT 38D(10%) + Diet 27D(40%) + Dilated TCN 딥러닝(50%) 결합.\n"
        "   - 선택적 거부 및 정상 분산 면제: 가동범위 미달 시 RETAKE, 경계 구간([0.40, 0.60])은 UNDETERMINED. 단 p <= 0.35인 고신뢰 정상 영역은 Fold 분산 면제.\n"
        "4) 현행 성능: Selective F1 97.13%, 판정 커버리지 90.2%, 보류율 9.8%, 정상군 허위 오탐 0.00% (완전 제로화)."
    )

    # c2 상세
    style_heading(doc.add_paragraph(), "3.2 c2 고개 정면 및 시선 이탈 모델 (BabelSquatC2ModelV8)", level=2)
    add_body_paragraph(
        doc,
        "1) 진단 대상 및 난제: 바닥 응시, 목 과굴곡/과신전, 시선 분산. 후방 시점(시점 A, E)에서 얼굴 전체가 뒤통수에 가려지는 물리적 관측 한계.\n"
        "2) 입력 데이터: 64프레임 x 33개 3D 관절 좌표 중 코, 귀, 눈, 어깨, 골반 좌표.\n"
        "3) 핵심 기술 및 피처 엔지니어링:\n"
        "   - 흉곽 로컬 정규 직교계: 어깨-골반 기준 흉곽 평면을 세우고, 가슴 기준 머리의 상대적 피치/요 각도(15D) 추출.\n"
        "   - 골반-머리 운동 협응 지연: 스쿼트 하강 시 골반 이동과 머리 각도 변화 간의 상호상관(Cross-correlation) 계측.\n"
        "   - 불확실성 사후 수축(Posterior Shrinkage): 후방 시점에서 안면 가림 시 분산 σ²=0.85를 부여하여 예측 확률을 0.5(모호)로 강제 수축.\n"
        "   - OBSERVABLE_CLEAN 후방 격리: 후방 시점에서는 시선 판정을 'NOT_EVALUATED'로 안전 격리하여 오진 원천 차단.\n"
        "4) 현행 성능: Selective F1 87.40%, 판정 커버리지 85.0%, 보류율 15.0%, 후방 시점 오탐 0건."
    )

    # c3 상세
    style_heading(doc.add_paragraph(), "3.3 c3 무릎 정렬(외반/내반) 모델 (BabelSquatC3ModelV9)", level=2)
    add_body_paragraph(
        doc,
        "1) 진단 대상 및 난제: 무릎이 안으로 모이는 외반(Knee Valgus) 및 바깥으로 벌어지는 내반(Varus). 측면/후방에서 관절 겹침(Occlusion) 및 원근감 왜곡 발생.\n"
        "2) 입력 데이터: 64프레임 골반, 무릎, 발목, 발가락 3D 관절 좌표.\n"
        "3) 핵심 기술 및 피처 엔지니어링:\n"
        "   - 전두면(Coronal Plane) 정사영 투영 및 신체 비례(ASIS 너비, 대퇴-경골 길이) 정규화.\n"
        "   - 25차원 무릎-발목 너비 동적 비율(knee_ankle_ratio) 시계열 통계량.\n"
        "   - 듀얼 스트림 앙상블: 25D LightGBM(70%) + 1D Conv-Transformer(30%) 결합.\n"
        "4) 현행 성능: Selective F1 91.82%, 판정 커버리지 89.4%, 보류율 10.6%, 허위 오탐 0건."
    )

    # c4 상세
    style_heading(doc.add_paragraph(), "3.4 c4 발바닥 지면 지지 고정성 모델 (BabelSquatC4ModelV8)", level=2)
    add_body_paragraph(
        doc,
        "1) 진단 대상 및 난제: 뒤꿈치 들림, 앞꿈치 들림, 좌우 롤링. 2D 카메라 해상도 상 2~3cm 미만의 미세 지면 분리를 포착하기 어려움.\n"
        "2) 입력 데이터: 64프레임 발목, 뒤꿈치, 발가락 좌표.\n"
        "3) 핵심 기술 및 피처 엔지니어링:\n"
        "   - 25D 피처셋: 가속도 저크(Jerk) 피크, FFT 1~3Hz 주파수 에너지 비, Z축 접지 특이점 가드레일.\n"
        "   - 3-State 안전 게이트: 미세 요동 구간 보류 처리.\n"
        "4) 현행 성능: Selective F1 90.54%, 판정 커버리지 87.9%, 보류율 12.1%, 허위 오탐 0건."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 4. 바벨 스쿼트 정상·오류 통합 진단 AI 엔진 (v11.0) 5계층 아키텍처
    # ========================================================
    p4 = doc.add_paragraph()
    style_heading(p4, "4. 바벨 스쿼트 정상·오류 통합 진단 AI 엔진 (v11.0) 5계층 아키텍처", level=1)

    add_body_paragraph(
        doc,
        "통합 엔진 v11.0은 4대 세부 모델의 예측 결과를 단순 결합하는 데 그치지 않고, "
        "입력 텐서부터 최종 판정까지 총 5개 계층(Layer 1 ~ Layer 5)으로 연결된 계층적 메타 신경망 아키텍처를 취하고 있습니다."
    )

    headers_layer = ["계층 (Layer)", "계층 명칭", "입력 텐서 규격", "핵심 처리 및 적용 기술", "출력 텐서 규격"]
    widths_layer = [Inches(1.0), Inches(1.4), Inches(1.3), Inches(2.2), Inches(1.0)]
    data_layer = [
        ["Layer 1", "L1 Pre-Ensemble", "원천 64F 포즈 (64, 33, 3)", "c1~c4 세부 모델별 피처 추출 및 개별 모델 추론", "[p1, p2, p3, p4] (4D)"],
        ["Layer 1-T", "Temperature Scaling", "[p1, p2, p3, p4]", "시점별 신뢰도 캘리브레이션 (온도 스케일링 T)", "보정 확률 (4D)"],
        ["Layer 2", "39D View-MoE Meta", "39D 메타 피처 벡터", "시점 라우팅 게이트 + 전문가 MLP (Reverse-ASL λ=3.0)", "16-Way 확률 (16D)"],
        ["Layer 3", "ST-GCN 잔차 융합", "64F 그래프 텐서", "경량 시공간 그래프 합성곱(ST-GCN) 특징 추출", "잔차 벡터 (16D)"],
        ["Layer 4", "Vector Scaling", "Late Fusion 확률", "시점 A~E별 다변량 벡터 스케일링 보정", "최종 다중확률"],
        ["Layer 5", "3-State 안전 게이트", "보정 확률 & ROM", "가동범위 컷오프 + 부상 리스크 가중 3-State 판정", "NORMAL/DEFECT/RETAKE"]
    ]
    t_layer = doc.add_table(rows=len(data_layer) + 1, cols=len(headers_layer))
    format_table(t_layer, widths_layer, headers_layer, data_layer, header_bg="1A365D")

    add_body_paragraph(
        doc,
        "■ [39차원 통합 피처 벡터 구성 상세]:\n"
        "1) 4대 서브모델 캘리브레이션 예측 확률 (4D): [p_c1, p_c2, p_c3, p_c4]\n"
        "2) c1 척추 핵심 생체역학 피처 (10D): 요추 2차 곡률(poly_a), 천골-흉추 Cobb 각도, 체간 전경각, 최저점 버트윙크 지수 등\n"
        "3) c2 시선 핵심 피처 (5D): 흉곽 기준 상대 머리 피치각, 골반-머리 협응 시차 상관계수, 후방 가림 플래그 등\n"
        "4) c3 무릎 핵심 피처 (10D): 전두면 무릎-발목 거리 비율(최솟값, 분산, 변곡점), 정강이 좌우 기울기 등\n"
        "5) c4 발바닥 핵심 피처 (5D): 발목 수직 변위 저크 피크, 1~3Hz FFT 파워 스펙트럼 에너지 비 등\n"
        "6) 카메라 시점 원핫 인코딩 (5D): [view1, view2, view3, view4, view5]"
    )

    add_body_paragraph(
        doc,
        "■ [16-Way 다중 클래스 매핑과 3-State 판정 로직]:\n"
        "4개 결함의 조합은 2⁴ = 16개의 상태(Class 0: [0,0,0,0] 무결점 정상 ~ Class 15: [1,1,1,1] 전 부위 결함)로 표현됩니다. "
        "Layer 2는 이 16개 클래스에 대한 소프트맥스 확률 분포를 출력합니다.\n"
        "- 무결점 정상(CLEAN) 판정 조건: Class 0 확률 P(Clean) >= θ_clean(시점별 0.45~0.55) AND 개별 부위 결함 확률 p_i <= θ_defect_max(0.35).\n"
        "- 재촬영(RETAKE) 유도 조건: 가동범위(골반 하강 깊이) 미달 OR 최대 클래스 확률이 0.35 미만으로 극단적 모호 구간일 때.\n"
        "- 결함(DEFECT) 판정 조건: 정상 및 RETAKE 조건을 제외한 확정 케이스에서 16개 클래스 아르그맥스(Argmax)를 통해 4자리 이진 라벨 확정."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 5. 현행 벤치마크 및 지표 정의
    # ========================================================
    p5 = doc.add_paragraph()
    style_heading(p5, "5. 현행 통합 엔진 성능 진화 및 평가지표 벤치마크", level=1)

    headers_hist = ["버전", "핵심 파이프라인 기술", "4-Defect EMR", "Clean F1", "Clean Prec.", "Clean Rec.", "Coverage", "RETAKE"]
    widths_hist = [Inches(0.9), Inches(2.2), Inches(0.8), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.6)]
    data_hist = [
        ["v1.0", "단순 Hard OR 결합 (임계값 0.5)", "54.62%", "52.80%", "51.20%", "54.50%", "100.0%", "0.0%"],
        ["v2.0", "c1 척추 29D 연속확률 소프트 게이팅", "58.39%", "57.60%", "56.40%", "58.90%", "100.0%", "0.0%"],
        ["v3.0", "2-Track 캐스케이드 정상 선별 스크리닝", "59.92%", "68.00%", "64.20%", "72.30%", "100.0%", "0.0%"],
        ["v6.0", "2,605건 무삭제 64F 스플라인 베이스라인", "69.40%", "67.17%", "67.67%", "66.67%", "87.50%", "12.50%"],
        ["v10.0", "39D View-MoE + Reverse-ASL (λ=3.0)", "71.60%", "71.60%", "75.41%", "68.15%", "90.06%", "9.94%"],
        ["v10.4", "시점별 벡터 스케일링 & 위험도 게이트", "76.40%", "76.92%", "71.92%", "82.68%", "89.94%", "10.06%"],
        ["v11.0", "5계층 ST-GCN 잔차 Late Fusion 챔피언", "76.42% ★", "77.21% ★", "72.41% ★", "82.68% ★", "90.02% ★", "9.98% ▼"]
    ]
    t_hist = doc.add_table(rows=len(data_hist) + 1, cols=len(headers_hist))
    format_table(t_hist, widths_hist, headers_hist, data_hist, header_bg="1A365D")

    add_body_paragraph(
        doc,
        "■ [핵심 평가지표 수학적 정의]:\n"
        "1) 4-Defect EMR (Exact Match Ratio): 1/N Σ I(y_pred == y_true). 4개 부위 [c1, c2, c3, c4] 라벨이 단 1개의 오차도 없이 일치해야 정답 인정 (올-오어-나씽 지표, 현행 76.42%).\n"
        "2) Clean F1-Score: 2 * (Clean Prec * Clean Rec) / (Clean Prec + Clean Rec). 4개 결함이 모두 0인 완전 정상 스쿼트의 종합 식별력 (현행 77.21%).\n"
        "3) Clean Precision: TP_clean / (TP_clean + FP_clean). AI가 '정상'이라 판정한 것 중 실제 정상 비율 (현행 72.41%, 70%+ 하드 제약 준수).\n"
        "4) Coverage: N_determined / N_total * 100%. 보류(RETAKE) 없이 확정 판정을 내린 비율 (현행 90.02%)."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ========================================================
    # 6. 외부 AI를 위한 핵심 자문 의뢰 질문 (5대 집중 과제)
    # ========================================================
    p6 = doc.add_paragraph()
    style_heading(p6, "6. 외부 AI 전문가를 위한 핵심 자문 의뢰 질문 (5대 집중 혁신 과제)", level=1)

    add_callout(
        doc,
        title="외부 AI 전문가 연구 과제: Clean F1 80%+ 및 EMR 80%+ 돌파를 위한 설계 의뢰",
        text="현재 v11.0은 76.42% EMR과 77.21% Clean F1으로 프로젝트 역대 최고치를 달성하였으나, "
             "상용 헬스케어 AI의 완벽성을 위해 두 지표 모두 '마의 80% 벽'을 돌파해야 합니다. "
             "다음 5대 기술 질문에 대해 수식, 신경망 아키텍처 다이어그램, 파이썬/PyTorch 코드 구조를 포함하여 구체적 해법을 제시해주십시오.",
        border_color="C53030",
        bg_color="FFF5F5"
    )

    questions = [
        (
            "Q1. [다중 라벨 간 생체역학적 조건부 의존성(Label Dependency) 모델링 혁신]",
            "현재 Layer 2는 16개 클래스를 독립적인 1-of-16 소프트맥스로 취급하여, 클래스 간 위계(예: 버트윙크 c1이 발생하면 무릎 외반 c3가 연쇄적으로 일어나는 생체역학적 결합성)를 충분히 반영하지 못합니다.\n"
            "→ 16-Way 독립 분류 대신, Classifier Chains, Directed Graph Neural Network, 또는 Transformer Decoder 기반의 Auto-regressive Multi-Label 의존성 모델링을 어떻게 구성해야 EMR을 극대화할 수 있습니까?"
        ),
        (
            "Q2. [EMR 및 Clean F1 직접 최적화를 위한 손실 함수(Loss Function) 혁신]",
            "현재 손실 함수는 거짓 정상을 패널티화하는 Reverse-ASL(Asymmetric Loss, λ_fc=3.0)을 사용하고 있습니다. 그러나 이는 여전히 교차 엔트로피 대리 손실에 불과하여 EMR(All-or-Nothing 일치율)과 Clean F1을 직접 최적화하지 못합니다.\n"
            "→ 미분 불가능한 EMR 및 F1 지표를 직접 공략할 수 있는 부드러운 대리 손실(Soft-EMR Loss / Lovász Extension / Margin-calibrated Asymmetric Loss)의 수식 및 PyTorch 구현 방안은 무엇입니까?"
        ),
        (
            "Q3. [스켈레톤 ST-GCN과 39D 도메인 피처 간의 Cross-Attention 융합 기법]",
            "현재 Layer 3의 ST-GCN은 64프레임 포즈 그래프에서 시공간 특징을 추출한 뒤, 최종단에서 Layer 2의 16차원 확률 벡터와 단순 덧셈/가중합(Late Fusion)으로 결합됩니다.\n"
            "→ 단순 Late Fusion의 정보 병목을 해소하고, 39차원 도메인 피처와 ST-GCN의 관절 노드 특징 간에 Cross-Attention 또는 Gated Bilinear Fusion을 적용하여 시너지를 극대화하는 아키텍처는 어떻게 구현합니까?"
        ),
        (
            "Q4. [5대 카메라 시점 불변성 및 시점 적응형 대조 학습(Contrastive Learning)]",
            "동일한 스쿼트 동작도 시점 A/E(후방 대각)와 시점 C(정면), 시점 B/D(전방 대각)에서 관측되는 2D/3D 좌표의 왜곡 양상이 매우 큽니다. 현재는 5차원 시점 원핫 벡터로 MoE를 라우팅하고 있습니다.\n"
            "→ 시점 라우팅을 넘어, 동일 피험자의 동시 촬영 다시점 데이터를 활용하여 시점 불변 관절 표현(View-invariant Representation)을 학습시키는 다시점 대조 학습(Multi-View Contrastive Loss)이나 도메인 적응 기법 도입 방안은 무엇입니까?"
        ),
        (
            "Q5. [Conformal Prediction 기반 3-State 동적 신뢰도 경계 최적화]",
            "현재 Layer 5는 고정된 확률 임계값([0.40, 0.60] 및 시점별 벡터 스케일링)을 통해 90.02% 커버리지와 9.98% RETAKE를 달성했습니다.\n"
            "→ 10% 미만의 재촬영률(RETAKE)을 보장하면서도 확정 진단군에서의 EMR을 80% 이상으로 끌어올릴 수 있는 Conformal Prediction 기반 유효성 보장(Coverage Guarantee) 알고리즘이나 Pareto-Optimal 다목적 경계 탐색 기법은 어떻게 구축합니까?"
        )
    ]

    for q_title, q_desc in questions:
        pq = doc.add_paragraph()
        style_heading(pq, q_title, level=2)
        add_body_paragraph(doc, q_desc)

    # 저장 경로
    out_docx_path = r"D:\자주프 문서\외부 ai 용 배경지식 문서.docx"
    os.makedirs(os.path.dirname(out_docx_path), exist_ok=True)
    doc.save(out_docx_path)
    print(f"[SUCCESS] Word document saved to: {out_docx_path}")

    # PDF 변환 실행
    out_pdf_path = r"D:\자주프 문서\외부 ai 용 배경지식 문서.pdf"
    convert_docx_to_pdf(out_docx_path, out_pdf_path)

    # Markdown 동반 파일 생성 (외부 AI 프롬프트 복사/붙여넣기 및 파일 첨부용)
    out_md_path = r"D:\자주프 문서\외부 ai 용 배경지식 문서.md"
    generate_markdown_version(out_md_path)
    print(f"[SUCCESS] Markdown companion saved to: {out_md_path}")

def generate_markdown_version(md_path):
    content = r"""# [vidio-health-trainer] AI 엔진 기술 배경지식 명세서 및 고도화 자문 의뢰서
## 바벨 스쿼트 4대 세부 오류 모델 및 통합 진단 AI 엔진(v11.0) 아키텍처 상세, 입출력 파이프라인, 현행 성능 및 고도화(Clean F1 & EMR 80%+) 자문

- **공식 프로젝트명**: `vidio-health-trainer`
- **배포 일시**: 2026-09-28
- **문서 규격**: PDF(`외부 ai 용 배경지식 문서.pdf`), Word(`외부 ai 용 배경지식 문서.docx`), Markdown 동시 배포

---

### ■ 외부 AI 전문가 자문 의뢰 개요 (Consultation Objective)
본 문서는 외부 최상위 AI 모델(Claude 3.7 Sonnet, GPT-4.5/o1/o3 등)에게 본 프로젝트(`vidio-health-trainer`)의 운동 자세 진단 파이프라인 전체를 완벽히 이해시키고, 현재 달성된 **[Clean F1 77.21%]** 및 **[4-Defect EMR 76.42%]**를 차세대 아키텍처 개선을 통해 **[80.0% ~ 85.0% 이상]**으로 극대화하기 위한 구체적이고 실현 가능한 아키텍처 혁신안, 손실 함수 설계, 멀티모달 융합 방안을 도출받기 위해 작성된 고밀도 기술 명세서입니다.

본 문서에 명시된 데이터셋 스펙, 피처 엔지니어링 인터페이스, 신경망 레이어 구성, 3-State 선택적 분류 규칙, 평가 지표 정의를 기반으로 이론적 직관과 실제 코드 레벨의 정밀한 모델링 솔루션을 제안해 주십시오.

---

## 1. 프로젝트 개요 및 핵심 고도화 목표

`vidio-health-trainer` 프로젝트는 단일 스마트폰/웹캠 환경에서 딥러닝 기반 포즈 추정(MediaPipe Pose)을 활용하여 사용자의 3대 웨이트 트레이닝(바벨 스쿼트, 데드리프트, 벤치프레스) 자세를 실시간 분석하고, 인체 관절별 결함을 정밀 코칭하는 지능형 피트니스 AI 시스템입니다. 본 문서의 타깃 종목은 가장 높은 관절 자유도와 시점 왜곡을 지닌 **바벨 스쿼트(Barbell Squat)**입니다.

| 핵심 지표 (Key Metric) | 현행 공식 마스터 (v11.0) | 차세대 목표 (Target) | 핵심 도전 과제 및 의미 |
| :--- | :---: | :---: | :--- |
| **4-Defect EMR (완전 일치율)** | **76.42%** | **80.0% ~ 82.0% ▲** | 4개 부위 동시 100% 무결점 판정 비율 (+4~6%p 증대) |
| **Clean F1-Score (정상 F1)** | **77.21%** | **80.0% ~ 85.0% ▲** | 무결점 정상 스쿼트의 정밀도·재현율 조화평균 극대화 |
| **Clean Precision (정상 정밀도)** | **72.41%** | **75.0% 이상 유지** | 거짓 정상(False Clean) 방지 하드 제약(70%+) 지속 준수 |
| **Coverage (판정 커버리지)** | **90.02%** | **90.0% 이상 유지** | 상용화 최소 기준선인 90% 이상 유지 (무리한 보류 방지) |
| **RETAKE (안전 보류율)** | **9.98%** | **10.0% 이하 유지** | 모호 구간 안전 보류율을 10% 미만으로 엄격히 통제 |

---

## 2. 데이터셋 규격 및 전처리 파이프라인 (Data & Preprocessing)

### 2.1 AI-Hub 원천 데이터 규격 및 전수 활용 원칙
AI-Hub 피트니스 자세 교정 데이터셋에서 바벨 스쿼트 원천 시퀀스 총 2,605건을 사용합니다. 일부 연구에서 시도하던 임의 데이터 삭제나 샘플링 편향을 원천 차단하기 위해, 1번부터 9번까지 9인 전 피험자와 5대 전 카메라 시점을 단 1건의 누락 없이 100% 전수 학습 및 검증에 투입합니다.

- **총 데이터 볼륨**: 2,605개 모션 시퀀스 (무삭제 전수 벤치마크)
- **피험자 구성**: 피험자 1번 ~ 9번 (전원 9명, 체형/신장/근력 편차 반영)
- **교차 검증 전략**: 5-Fold GroupKFold (피험자 격리 교차 검증으로 Data Leakage 원천 차단)
- **카메라 5대 시점 체계**:
  - **시점 A (view5)**: 우측 후방 45° (등 근육, 척추 곡률 관측 우수 / 안면 가림 발생)
  - **시점 B (view2)**: 우측 전방 45° (무릎 가동범위 및 상체 기울기 동시 관측 최적 뷰)
  - **시점 C (view3)**: 정면 0° (무릎 외반/내반 좌우 대칭 관측 우수 / Z축 깊이 왜곡 극심)
  - **시점 D (view4)**: 좌측 전방 45° (시점 B와 대칭적 관측 특성)
  - **시점 E (view1)**: 좌측 후방 45° (후방 척추 관측 우수 / 얼굴 랜드마크 완전 결손)
- **프레임 전처리 표준**: 64프레임 3차 스플라인 보간(Cubic Spline Interpolation) 단일 표준화. 원천 가변 프레임(30~120F)을 64프레임 균일 텐서로 정규화.
- **실서비스 입력 전처리**: 연속 영상에서 1회 반복(단일 횟수)을 엄밀 분할한 후 64프레임 3차 스플라인 변환.

---

## 3. 4대 세부 오류 모델 (c1 ~ c4) 상세 아키텍처 및 입출력 명세

통합 엔진(Layer 1)의 입력으로 투입되는 4대 세부 결함 모델은 인체 해부학적 특성에 맞추어 고도로 독립화된 전용 피처 추출기와 이종(Heterogeneous) 머신러닝/딥러닝 앙상블로 구성되어 있습니다.

### 3.1 c1 척추 중립성 위반 모델 (BabelSquatC1ModelV11_1)
- **진단 대상 및 난제**: 요추 굴곡(버트윙크, Butt Wink), 굿모닝 스쿼트(상체 과숙임), 요추 과신전(허리 꺾임). MediaPipe는 어깨와 골반점만 줄 뿐 척추 뼈마디가 없으며, 정면/후방에서 Z축 압축 왜곡 극심.
- **입력 데이터**: 64프레임 x 33개 3D 관절 좌표 $(T=64, J=33, C=3)$.
- **핵심 기술 및 피처 엔지니어링**:
  1. **DFKI SpinePose 9분절 3D 가상 복원**: 어깨-골반 사이 S1(천골), L5/L3/L1(요추), T8/T3(흉추), C7/C4/C1(경추)의 가상 3D 좌표를 기하학적으로 복원하여 요추 2차 곡률(`poly_a`) 및 분절별 Cobb 각도 산출.
  2. **체간 로컬 정규 직교 기저계**: 좌우 어깨와 골반 벡터로 인체 몸통 자체의 로컬 직교계(x, y, z)를 정의하여 외부 카메라 각도 편향 소거.
  3. **Tri-Ensemble 하이브리드**: GBDT 38D(10%) + Diet 27D(40%) + Dilated TCN 딥러닝(50%) 결합.
  4. **선택적 거부 및 정상 분산 면제**: 가동범위 미달 시 RETAKE, 경계 구간([0.40, 0.60])은 UNDETERMINED. 단 $p \le 0.35$인 고신뢰 정상 영역은 Fold 분산 면제 승인.
- **현행 성능**: Selective F1 97.13%, 판정 커버리지 90.2%, 보류율 9.8%, 정상군 오탐률 0.00% (완전 제로화).

### 3.2 c2 고개 정면 및 시선 이탈 모델 (BabelSquatC2ModelV8)
- **진단 대상 및 난제**: 바닥 응시, 목 과굴곡/과신전, 시선 분산. 후방 시점(시점 A, E)에서 얼굴 전체가 뒤통수에 가려지는 물리적 관측 한계.
- **입력 데이터**: 64프레임 x 33개 관절 중 코, 귀, 눈, 어깨, 골반 좌표.
- **핵심 기술 및 피처 엔지니어링**:
  1. **흉곽 로컬 정규 직교계**: 어깨-골반 기준 흉곽 평면을 세우고 가슴 기준 머리의 상대적 피치/요 각도(15D) 추출.
  2. **골반-머리 운동 협응 지연**: 스쿼트 하강 시 골반 이동과 머리 각도 변화 간의 상호상관(Cross-correlation) 계측.
  3. **불확실성 사후 수축(Posterior Shrinkage)**: 후방 시점에서 안면 가림 시 분산 $\sigma^2=0.85$를 부여하여 예측 확률을 0.5(모호)로 강제 수축.
  4. **OBSERVABLE_CLEAN 후방 격리**: 후방 시점에서는 시선 판정을 'NOT_EVALUATED'로 안전 격리하여 오진 원천 차단.
- **현행 성능**: Selective F1 87.40%, 판정 커버리지 85.0%, 보류율 15.0%, 후방 시점 오탐 0건.

### 3.3 c3 무릎 정렬(외반/내반) 모델 (BabelSquatC3ModelV9)
- **진단 대상 및 난제**: 무릎이 안으로 모이는 외반(Knee Valgus) 및 바깥으로 벌어지는 내반(Varus). 측면/후방에서 관절 겹침(Occlusion) 및 원근감 왜곡 발생.
- **입력 데이터**: 64프레임 골반, 무릎, 발목, 발가락 3D 관절 좌표.
- **핵심 기술 및 피처 엔지니어링**:
  1. **전두면(Coronal Plane) 정사영 투영 및 신체 비례 정규화**: ASIS 너비 및 대퇴-경골 길이를 기준으로 스케일 정규화.
  2. **25차원 무릎-발목 너비 동적 비율(`knee_ankle_ratio`)**: 64프레임 동적 궤적 통계량.
  3. **듀얼 스트림 앙상블**: 25D LightGBM(70%) + 1D Conv-Transformer(30%) 결합.
- **현행 성능**: Selective F1 91.82%, 판정 커버리지 89.4%, 보류율 10.6%, 허위 오탐 0건.

### 3.4 c4 발바닥 지면 지지 고정성 모델 (BabelSquatC4ModelV8)
- **진단 대상 및 난제**: 뒤꿈치 들림, 앞꿈치 들림, 좌우 롤링. 2D 카메라 상 2~3cm 미만의 미세 지면 분리를 포착하기 어려움.
- **입력 데이터**: 64프레임 발목, 뒤꿈치, 발가락 좌표.
- **핵심 기술 및 피처 엔지니어링**:
  1. **25D 피처셋**: 가속도 저크(Jerk) 피크, FFT 1~3Hz 주파수 에너지 비, Z축 접지 특이점 가드레일.
  2. **3-State 안전 게이트**: 미세 요동 구간 보류 처리.
- **현행 성능**: Selective F1 90.54%, 판정 커버리지 87.9%, 보류율 12.1%, 허위 오탐 0건.

---

## 4. 바벨 스쿼트 정상·오류 통합 진단 AI 엔진 (v11.0) 5계층 아키텍처

통합 엔진 v11.0은 4대 세부 모델의 예측 결과를 단순 결합하는 데 그치지 않고, 입력 텐서부터 최종 판정까지 총 5개 계층(Layer 1 ~ Layer 5)으로 연결된 계층적 메타 신경망 아키텍처를 취하고 있습니다.

| 계층 (Layer) | 계층 명칭 | 입력 텐서 규격 | 핵심 처리 및 적용 기술 | 출력 텐서 규격 |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | L1 Pre-Ensemble | 원천 64F 포즈 (64, 33, 3) | c1~c4 세부 모델별 피처 추출 및 개별 모델 추론 | `[p1, p2, p3, p4]` (4D) |
| **Layer 1-T** | Temperature Scaling | `[p1, p2, p3, p4]` | 시점별 신뢰도 캘리브레이션 (온도 스케일링 T) | 보정 확률 (4D) |
| **Layer 2** | 39D View-MoE Meta | 39D 메타 피처 벡터 | 시점 라우팅 게이트 + 전문가 MLP (Reverse-ASL $\lambda=3.0$) | 16-Way 확률 (16D) |
| **Layer 3** | ST-GCN 잔차 융합 | 64F 그래프 텐서 | 경량 시공간 그래프 합성곱(ST-GCN) 특징 추출 | 잔차 벡터 (16D) |
| **Layer 4** | Vector Scaling | Late Fusion 확률 | 시점 A~E별 다변량 벡터 스케일링 보정 | 최종 다중확률 (16D) |
| **Layer 5** | 3-State 안전 게이트 | 보정 확률 & ROM | 가동범위 컷오프 + 부상 리스크 가중 3-State 판정 | NORMAL / DEFECT / RETAKE |

### 4.1 39차원 통합 피처 벡터 구성 상세
1. **4대 서브모델 캘리브레이션 예측 확률 (4D)**: `[p_c1, p_c2, p_c3, p_c4]`
2. **c1 척추 핵심 생체역학 피처 (10D)**: 요추 2차 곡률(`poly_a`), 천골-흉추 Cobb 각도, 체간 전경각, 최저점 버트윙크 지수 등
3. **c2 시선 핵심 피처 (5D)**: 흉곽 기준 상대 머리 피치각, 골반-머리 협응 시차 상관계수, 후방 가림 플래그 등
4. **c3 무릎 핵심 피처 (10D)**: 전두면 무릎-발목 거리 비율(최솟값, 분산, 변곡점), 정강이 좌우 기울기 등
5. **c4 발바닥 핵심 피처 (5D)**: 발목 수직 변위 저크 피크, 1~3Hz FFT 파워 스펙트럼 에너지 비 등
6. **카메라 시점 원핫 인코딩 (5D)**: `[view1, view2, view3, view4, view5]`

### 4.2 16-Way 다중 클래스 매핑과 3-State 판정 로직
4개 결함의 조합은 $2^4 = 16$개의 상태(Class 0: `[0,0,0,0]` 무결점 정상 ~ Class 15: `[1,1,1,1]` 전 부위 결함)로 표현됩니다. Layer 2는 이 16개 클래스에 대한 소프트맥스 확률 분포를 출력합니다.
- **무결점 정상(CLEAN) 판정 조건**: Class 0 확률 $P(\text{Clean}) \ge \theta_{clean}$(시점별 0.45~0.55) AND 개별 부위 결함 확률 $p_i \le \theta_{defect\_max}(0.35)$.
- **재촬영(RETAKE) 유도 조건**: 가동범위(골반 하강 깊이) 미달 OR 최대 클래스 확률이 0.35 미만으로 극단적 모호 구간일 때.
- **결함(DEFECT) 판정 조건**: 정상 및 RETAKE 조건을 제외한 확정 케이스에서 16개 클래스 아르그맥스(Argmax)를 통해 4자리 이진 라벨 확정.

---

## 5. 현행 통합 엔진 성능 진화 및 평가지표 벤치마크

| 버전 | 핵심 파이프라인 기술 | 4-Defect EMR | Clean F1 | Clean Prec. | Clean Rec. | Coverage | RETAKE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1.0** | 단순 Hard OR 결합 (임계값 0.5) | 54.62% | 52.80% | 51.20% | 54.50% | 100.0% | 0.0% |
| **v2.0** | c1 척추 29D 연속확률 소프트 게이팅 | 58.39% | 57.60% | 56.40% | 58.90% | 100.0% | 0.0% |
| **v3.0** | 2-Track 캐스케이드 정상 선별 스크리닝 | 59.92% | 68.00% | 64.20% | 72.30% | 100.0% | 0.0% |
| **v6.0** | 2,605건 무삭제 64F 스플라인 베이스라인 | 69.40% | 67.17% | 67.67% | 66.67% | 87.50% | 12.50% |
| **v10.0** | 39D View-MoE + Reverse-ASL ($\lambda=3.0$) | 71.60% | 71.60% | 75.41% | 68.15% | 90.06% | 9.94% |
| **v10.4** | 시점별 벡터 스케일링 & 위험도 게이트 | 76.40% | 76.92% | 71.92% | 82.68% | 89.94% | 10.06% |
| **v11.0** | 5계층 ST-GCN 잔차 Late Fusion 챔피언 | **76.42% ★** | **77.21% ★** | **72.41% ★** | **82.68% ★** | **90.02% ★** | **9.98% ▼** |

### 5.1 핵심 평가지표 수학적 정의
1. **4-Defect EMR (Exact Match Ratio)**: $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{\mathbf{y}}_i == \mathbf{y}_i)$. 4개 부위 `[c1, c2, c3, c4]` 라벨이 단 1개의 오차도 없이 일치해야 정답 인정 (올-오어-나씽 지표, 현행 76.42%).
2. **Clean F1-Score**: $2 \times \frac{\text{Clean Prec} \times \text{Clean Rec}}{\text{Clean Prec} + \text{Clean Rec}}$. 4개 결함이 모두 0인 완전 정상 스쿼트의 종합 식별력 (현행 77.21%).
3. **Clean Precision**: $\frac{TP_{clean}}{TP_{clean} + FP_{clean}}$. AI가 '정상'이라 판정한 것 중 실제 정상 비율 (현행 72.41%, 70%+ 하드 제약 준수).
4. **Coverage**: $\frac{N_{determined}}{N_{total}} \times 100\%$. 보류(RETAKE) 없이 확정 판정을 내린 비율 (현행 90.02%).

---

## 6. 외부 AI 전문가를 위한 핵심 자문 의뢰 질문 (5대 집중 혁신 과제)

> **🎯 자문 목표**: 현재 v11.0의 76.42% EMR과 77.21% Clean F1을 모두 **[80.0% ~ 85.0% 이상]**으로 극대화하면서, Coverage 90% 이상과 False Clean 제약(70%+ Precision)을 동시에 달성하는 최적의 아키텍처 및 손실 함수 설계.

### Q1. [다중 라벨 간 생체역학적 조건부 의존성(Label Dependency) 모델링 혁신]
현재 Layer 2는 16개 클래스를 독립적인 1-of-16 소프트맥스로 취급하여, 클래스 간 위계(예: 버트윙크 c1이 발생하면 무릎 외반 c3가 연쇄적으로 일어나는 생체역학적 결합성)를 충분히 반영하지 못합니다.
- **질문**: 16-Way 독립 분류 대신, Classifier Chains, Directed Graph Neural Network, 또는 Transformer Decoder 기반의 Auto-regressive Multi-Label 의존성 모델링을 어떻게 구성해야 EMR을 극대화할 수 있습니까?

### Q2. [EMR 및 Clean F1 직접 최적화를 위한 손실 함수(Loss Function) 혁신]
현재 손실 함수는 거짓 정상을 패널티화하는 Reverse-ASL(Asymmetric Loss, $\lambda_{fc}=3.0$)을 사용하고 있습니다. 그러나 이는 여전히 교차 엔트로피 대리 손실에 불과하여 EMR(All-or-Nothing 일치율)과 Clean F1을 직접 최적화하지 못합니다.
- **질문**: 미분 불가능한 EMR 및 F1 지표를 직접 공략할 수 있는 부드러운 대리 손실(Soft-EMR Loss / Lovász Extension / Margin-calibrated Asymmetric Loss)의 수식 및 PyTorch 구현 방안은 무엇입니까?

### Q3. [스켈레톤 ST-GCN과 39D 도메인 피처 간의 Cross-Attention 융합 기법]
현재 Layer 3의 ST-GCN은 64프레임 포즈 그래프에서 시공간 특징을 추출한 뒤, 최종단에서 Layer 2의 16차원 확률 벡터와 단순 덧셈/가중합(Late Fusion)으로 결합됩니다.
- **질문**: 단순 Late Fusion의 정보 병목을 해소하고, 39차원 도메인 피처와 ST-GCN의 관절 노드 특징 간에 Cross-Attention 또는 Gated Bilinear Fusion을 적용하여 시너지를 극대화하는 아키텍처는 어떻게 구현합니까?

### Q4. [5대 카메라 시점 불변성 및 시점 적응형 대조 학습(Contrastive Learning)]
동일한 스쿼트 동작도 시점 A/E(후방 대각)와 시점 C(정면), 시점 B/D(전방 대각)에서 관측되는 2D/3D 좌표의 왜곡 양상이 매우 큽니다. 현재는 5차원 시점 원핫 벡터로 MoE를 라우팅하고 있습니다.
- **질문**: 시점 라우팅을 넘어, 동일 피험자의 동시 촬영 다시점 데이터를 활용하여 시점 불변 관절 표현(View-invariant Representation)을 학습시키는 다시점 대조 학습(Multi-View Contrastive Loss)이나 도메인 적응 기법 도입 방안은 무엇입니까?

### Q5. [Conformal Prediction 기반 3-State 동적 신뢰도 경계 최적화]
현재 Layer 5는 고정된 확률 임계값([0.40, 0.60] 및 시점별 벡터 스케일링)을 통해 90.02% 커버리지와 9.98% RETAKE를 달성했습니다.
- **질문**: 10% 미만의 재촬영률(RETAKE)을 보장하면서도 확정 진단군에서의 EMR을 80% 이상으로 끌어올릴 수 있는 Conformal Prediction 기반 유효성 보장(Coverage Guarantee) 알고리즘이나 Pareto-Optimal 다목적 경계 탐색 기법은 어떻게 구축합니까?
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.strip())


def convert_docx_to_pdf(docx_path, pdf_path):
    print(f"[CONVERT] Converting {docx_path} -> {pdf_path} using Word COM...")
    cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{docx_path}')
        $doc.SaveAs([ref]'{pdf_path}', [ref]17)
        $doc.Close()
        Write-Output "[COM SUCCESS] PDF conversion complete."
    }} catch {{
        Write-Error $_
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_path):
        print(f"[SUCCESS] PDF generated successfully: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")
    else:
        print(f"[ERROR] Failed to convert PDF: {res.stderr}")

if __name__ == "__main__":
    build_document()
