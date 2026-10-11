# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d5_background_knowledge_doc.py
Purpose: 외부 AI 모델에게 바벨 데드리프트 d5 성능 고도화 연구 자문을 요청하기 위한
         배경 지식 종합 백서 및 RFC 문서 (PDF, DOCX, MD) 자동 생성.
Target Path:
  - D:\자주프 문서\d5용 배경 지식.pdf
  - D:\자주프 문서\d5용 배경 지식.docx
  - D:\자주프 문서\d5용 배경 지식.md
"""

import os
import sys
import json
import time

# 1. docx 라이브러리
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# 2. reportlab 라이브러리 (PDF 생성용)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT_DIR = r"D:\자주프 문서"
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "d5용 배경 지식.pdf")
DOCX_PATH = os.path.join(OUT_DIR, "d5용 배경 지식.docx")
MD_PATH = os.path.join(OUT_DIR, "d5용 배경 지식.md")

# 폰트 등록
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"
pdfmetrics.registerFont(TTFont('Malgun', FONT_PATH))
pdfmetrics.registerFont(TTFont('MalgunBold', FONT_BOLD_PATH))

# 색상 정의
NAVY_HEX = "#1A5276"
BLUE_HEX = "#2980B9"
LIGHT_BG_HEX = "#F8F9F9"
BORDER_HEX = "#BDC3C7"
HIGHLIGHT_HEX = "#EBF5FB"
TEXT_HEX = "#2C3E50"

PRIMARY_COLOR = colors.HexColor(NAVY_HEX)
SECONDARY_COLOR = colors.HexColor(BLUE_HEX)
TEXT_COLOR = colors.HexColor(TEXT_HEX)
BG_LIGHT = colors.HexColor(LIGHT_BG_HEX)
BG_HIGHLIGHT = colors.HexColor(HIGHLIGHT_HEX)

# docx용 RGBColor 정의
COLOR_PRIMARY = RGBColor(26, 82, 118)
COLOR_SECONDARY = RGBColor(41, 128, 185)
COLOR_TEXT = RGBColor(44, 62, 80)
COLOR_MUTED = RGBColor(127, 140, 141)


def build_pdf_document():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=1.8*cm,
        rightMargin=1.8*cm,
        topMargin=1.8*cm,
        bottomMargin=1.8*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='MalgunBold',
        fontSize=20,
        leading=26,
        textColor=PRIMARY_COLOR,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='MalgunBold',
        fontSize=10,
        leading=14,
        textColor=SECONDARY_COLOR,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='MalgunBold',
        fontSize=13,
        leading=18,
        textColor=PRIMARY_COLOR,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='MalgunBold',
        fontSize=10.5,
        leading=15,
        textColor=SECONDARY_COLOR,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Malgun',
        fontSize=8.5,
        leading=12.5,
        textColor=TEXT_COLOR,
        spaceAfter=4
    )
    body_bold = ParagraphStyle(
        'BodyBold_Custom',
        fontName='MalgunBold',
        fontSize=8.5,
        leading=12.5,
        textColor=TEXT_COLOR,
        spaceAfter=4
    )
    callout_style = ParagraphStyle(
        'Callout_Text',
        fontName='Malgun',
        fontSize=8.5,
        leading=12.5,
        textColor=TEXT_COLOR
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Malgun',
        fontSize=7.5,
        leading=10.5,
        textColor=TEXT_COLOR
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='MalgunBold',
        fontSize=7.5,
        leading=10.5,
        textColor=TEXT_COLOR
    )
    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        fontName='Malgun',
        fontSize=7.5,
        leading=10.5,
        alignment=1,
        textColor=TEXT_COLOR
    )
    table_cell_header = ParagraphStyle(
        'TableHeader',
        fontName='MalgunBold',
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("vidio-health-trainer | 외부 AI 자문용 기술 백서 & RFC 문서", subtitle_style))
    story.append(Paragraph("바벨 데드리프트 d5 (바벨 궤적과 몸 밀착) 모델<br/>배경 지식, 데이터셋, 40D 피처셋 및 아키텍처 기술 명세서", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceBefore=4, spaceAfter=8))

    meta_text = (
        "<b>• 프로젝트명:</b> vidio-health-trainer &nbsp;|&nbsp; <b>타깃 운동:</b> 바벨 데드리프트 (Barbell Deadlift)<br/>"
        "<b>• 분석 대상 결함:</b> d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)<br/>"
        "<b>• 문서 목적:</b> 외부 최신 AI 모델(GPT-4o, Claude 3.5, Gemini Pro 등)에게 시스템의 모든 데이터, 피처, "
        "모델 파이프라인 및 실측 성능을 투명하게 제공하여, 기본 F1 80%+ 돌파 및 정면 뷰(시점 C) 깊이 축소 극복 솔루션을 도출하기 위함.<br/>"
        "<b>• 작성 일자:</b> 2026년 9월 30일 &nbsp;|&nbsp; <b>최신 모델 버전:</b> v4.0 Gated (40D 시점 적응형 게이팅)"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 8))

    # Callout Box: 핵심 목표
    callout_data = [[
        Paragraph(
            "<b>[외부 AI 연구 자문 핵심 과제]</b><br/>"
            "현재 본 시스템은 40D 시점 적응형 게이팅(v4.0 Gated)을 통해 3-State 확신 구간 F1 83.23%(커버리지 86.8%), "
            "ROC-AUC 0.8788을 달성하였습니다. 그러나 <b>기권 없는 기본 OOF F1-Score는 79.07%로 마의 80.0% 벽 직전에 정체</b>되어 있으며, "
            "특히 <b>정면 뷰(시점 C)의 2D 평면 투영 깊이 축소(Depth Flattening)로 인한 결함 미탐(F1 56.68%)</b>이 최대 병목입니다.<br/>"
            "외부 AI는 본 문서에 수록된 40D 피처 명세와 시점별/체형별 오분류 분석을 바탕으로, <b>80.0%+ 돌파를 위한 "
            "구체적이고 수학적인 신규 피처 및 아키텍처 해결책</b>을 제시해야 합니다.",
            callout_style
        )
    ]]
    callout_tbl = Table(callout_data, colWidths=[17.4*cm])
    callout_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_HIGHLIGHT),
        ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(callout_tbl)
    story.append(Spacer(1, 10))

    # Section 1. 타깃 결함 d5의 생체역학 정의
    story.append(Paragraph("1. 타깃 결함 d5의 생체역학 정의 및 판정 기준", h1_style))
    sec1_text = (
        "<b>1.1. 결함의 물리적 정의</b><br/>"
        "데드리프트에서 바벨 봉(Barbell)은 인체의 무게중심(Center of Mass)에 가장 가까운 정강이(Shank), 무릎(Knee), "
        "허벅지(Thigh) 전면을 스치듯이 수직 선상으로 이동해야 합니다. 바벨이 몸에서 전방으로 멀어질 경우, "
        "고관절 및 요추 L5/S1 관절에 작용하는 모멘트암(Moment Arm = 거리 x 하중)이 기하급수적으로 증가하여 "
        "극심한 척추 부상 및 리프팅 실패를 유발합니다.<br/><br/>"
        "<b>1.2. 정상 동작 (Label = 0, Normal)</b><br/>"
        "• 셋업부터 최하사점(Nadir), 퍼스트풀, 락아웃(Lockout)까지 바벨이 정강이/허벅지 표면에서 2~3cm 이내로 밀착 유지.<br/>"
        "• 측면/사각 뷰 관찰: 바벨 봉이 수직선(Vertical Line)을 그리며 상승하고, 최하사점에서 무릎이 팔 안쪽에 깊숙이 안착.<br/><br/>"
        "<b>1.3. 결함 동작 (Label = 1, Defect)</b><br/>"
        "• 바벨이 정강이/무릎을 스치지 못하고 무릎 앞쪽으로 주먹 1~2개 이상(전방 이탈) 둥글게 원을 그리며 돌아나옴.<br/>"
        "• 측면/사각 뷰 관찰: 바벨을 앞으로 멀리 잡으면서 팔꿈치가 앞으로 꺾이고 무릎이 팔꿈치 뒤로 빠지는 기하학적 역전 발생."
    )
    story.append(Paragraph(sec1_text, body_style))
    story.append(Spacer(1, 8))

    # Section 2. 데이터셋 및 전처리 파이프라인
    story.append(Paragraph("2. 데이터셋 규격 및 전처리 파이프라인 (Data Pipeline Invariants)", h1_style))
    sec2_text = (
        "<b>2.1. 원천 데이터 및 전처리 규격</b><br/>"
        "• <b>원천 데이터셋:</b> AI-Hub 헬스케어 피트니스 데이터셋 (운동 02. 바벨 데드리프트)<br/>"
        "• <b>전처리 파이프라인:</b> 연속 비디오에서 1회 반복(단일 횟수, Repetition) 엄밀 분할 $\\rightarrow$ "
        "<b>64프레임 3차 스플라인(Cubic Spline) 시간 정규화</b> $\\rightarrow$ MediaPipe Pose 33개 관절 3D 키포인트 추출.<br/>"
        "• <b>입력 텐서 규격:</b> 시퀀스당 <code>(T=64, J=33, C=4)</code> ($x, y, z, \\text{visibility}$)<br/>"
        "• <b>총 시퀀스 수 및 클래스 균형:</b> <b>총 770개 시퀀스 (정상 385건 : 결함 385건, 50:50 완벽 균형)</b><br/><br/>"
        "<b>2.2. 피험자 전수 원칙 (All Subjects Invariant)</b><br/>"
        "AI-Hub 피험자 1번부터 9번까지 전원(9명)이 데이터셋에 포함되어 있으며, 피험자 간 체형 편향을 검증하기 위해 "
        "<b>5-Fold StratifiedGroupKFold (피험자 완전 격리 교차검증)</b>를 필수로 적용합니다.<br/><br/>"
        "<b>2.3. 공식 5대 카메라 시점 명칭 및 매핑 (Official 5-View Taxonomy)</b><br/>"
        "카메라 시점 편향(View Bias)을 원천 차단하기 위해 5대 전 시점을 모두 동일한 비중(시점당 154건)으로 반영합니다:"
    )
    story.append(Paragraph(sec2_text, body_style))

    # 시점 표
    view_table_data = [
        [Paragraph("공식 시점 명칭", table_cell_header), Paragraph("내부 코드", table_cell_header), Paragraph("카메라 설치 각도", table_cell_header), Paragraph("샘플수", table_cell_header), Paragraph("시점 특성 및 d5 판정 난이도", table_cell_header)],
        [Paragraph("시점 A", table_cell_bold), Paragraph("view5", table_cell_center), Paragraph("우측 후방 (45도)", table_cell_center), Paragraph("154건", table_cell_center), Paragraph("광배근 결합도 및 바벨-체간 후방 이격 관측 우수", table_cell_style)],
        [Paragraph("시점 B", table_cell_bold), Paragraph("view2", table_cell_center), Paragraph("우측 전방 (45도)", table_cell_center), Paragraph("154건", table_cell_center), Paragraph("무릎 통과 시 바벨 전방 이탈 골든 앵글 (F1 87.0%)", table_cell_style)],
        [Paragraph("시점 C", table_cell_bold), Paragraph("view3", table_cell_center), Paragraph("정면 (0도)", table_cell_center), Paragraph("154건", table_cell_center), Paragraph("<b>[최대 병목]</b> 시상면 깊이 축소로 전후 이격거리 소실 (F1 56.7%)", table_cell_style)],
        [Paragraph("시점 D", table_cell_bold), Paragraph("view4", table_cell_center), Paragraph("좌측 전방 (45도)", table_cell_center), Paragraph("154건", table_cell_center), Paragraph("최고 정확도 시점, 바벨-무릎 간극 완벽 분별 (F1 96.1%)", table_cell_style)],
        [Paragraph("시점 E", table_cell_bold), Paragraph("view1", table_cell_center), Paragraph("좌측 후방 (45도)", table_cell_center), Paragraph("154건", table_cell_center), Paragraph("후방 사각 뷰, 바벨-체간 이격 분별 양호 (F1 75.9%)", table_cell_style)],
    ]
    view_tbl = Table(view_table_data, colWidths=[2.2*cm, 1.8*cm, 3.2*cm, 1.6*cm, 8.6*cm])
    view_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_HEX)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(view_tbl)
    story.append(Spacer(1, 6))

    sec2_warn = (
        "<b>2.4. 엄격한 데이터 격리 및 무누수 원칙 (Strict No-Leakage Invariant)</b><br/>"
        "AI-Hub 라벨링 JSON에 포함된 3D 정답 좌표나 결함 어노테이션은 <b>오직 정답 타깃($y$) 및 성능 채점</b>에만 사용됩니다. "
        "모델의 입력 피처($X$) 파이프라인에 AI-Hub JSON 정보가 간접 참조되거나 유입되는 것은 영구 엄금되며, "
        "<b>100% MediaPipe Pose 단안 비전 키포인트만을 기반</b>으로 피처를 추출해야 합니다."
    )
    story.append(Paragraph(sec2_warn, body_style))
    story.append(Spacer(1, 8))

    # Section 3. 피처 엔지니어링 전수 명세서 (40D)
    story.append(Paragraph("3. 피처 엔지니어링 전수 명세서 (40D Full Specifications)", h1_style))
    story.append(Paragraph("현재 v4.0 Gated 모델에 탑재된 총 40차원 피처의 상세 수학적 수식 및 물리적 정의입니다:", body_style))

    # 40D 피처 전수 표
    f_headers = [Paragraph("No", table_cell_header), Paragraph("피처 식별자 (Feature Name)", table_cell_header), Paragraph("그룹", table_cell_header), Paragraph("수학적 연산식 및 물리적 정의", table_cell_header), Paragraph("정상/결함 특성", table_cell_header)]
    
    f_rows = [
        # Group 1 (2D)
        ["1", "d5_bar_shank_dist_max", "G1", "max(||Bar_t - Shank_t||) (t in First Pull)", "정상 0.22 vs 결함 0.56 (정강이 이격)"],
        ["2", "d5_bar_shank_dist_mean", "G1", "mean(||Bar_t - Shank_t||) (t in First Pull)", "정강이 통과 구간 평균 이격도"],
        # Group 2 (4D)
        ["3", "d5_bar_thigh_dist_mean", "G2", "mean(||Bar_t - Thigh_t||) (t in Knee~Lockout)", "허벅지 슬라이딩 밀착 평균 거리"],
        ["4", "d5_bar_thigh_dist_max", "G2", "max(||Bar_t - Thigh_t||) (t in Knee~Lockout)", "허벅지 통과 시 최대 전방 이탈거리"],
        ["5", "d5_p2_thigh_drag_contact_rate", "G2", "sum(dist < 0.05) / T_p2", "허벅지 표면 밀착 접촉 유지 프레임 비율"],
        ["6", "d5_bar_leg_contact_deficit_ratio", "G2", "1.0 - (Contact_frames / Active_frames)", "전 리프팅 중 다리 비접촉 결손율"],
        # Group 3 (4D)
        ["7", "d5_bar_path_curvature", "G3", "integral(|kappa(t)| dt) (바벨 궤적 2차 미분)", "정상 1.12 vs 결함 2.45 (궤적 곡률)"],
        ["8", "d5_bar_ascent_velocity_smoothness", "G3", "std(dY/dt) / (mean(|dY/dt|) + eps)", "바벨 수직 상승 속도 프로파일 매끄러움"],
        ["9", "d5_bar_lateral_wobble_std", "G3", "std(X_bar(t)) (상승 중 좌우 흔들림)", "정상 0.015 vs 결함 0.042 (좌우 요동)"],
        ["10", "d5_bar_total_horizontal_drift", "G3", "max(X_bar) - min(X_bar)", "전 구간 바벨 최대 수평 이동폭"],
        # Group 4 (5D)
        ["11", "d5_hip_bar_moment_arm_at_knee", "G4", "|X_hip - X_bar| at Knee Pass", "무릎 통과 시 고관절-바벨 수평 모멘트암"],
        ["12", "d5_torso_arm_coupling_deficit", "G4", "std(angle(Torso, Arm)) in First Pull", "광배근 체간-팔 연동 결합도 결손"],
        ["13", "d5_knee_pass_timing_norm", "G4", "t(Bar_y = Knee_y) / T_ascent", "바벨이 무릎을 통과하는 정규화 시점"],
        ["14", "d5_shoulder_wrist_angle_max", "G4", "max(angle(Shoulder, Wrist, Vertical))", "팔이 수직선에서 전방으로 뻗은 최대각"],
        ["15", "d5_shoulder_wrist_angle_mean", "G4", "mean(angle(Shoulder, Wrist, Vertical))", "리프팅 중 전방 경사각 평균"],
        # Group 5 (13D)
        ["16", "d5_eccentric_bar_drift_max", "G5", "max(X_bar_ecc - X_bar_start)", "하강(이완기) 바벨 전방 이탈 최대폭"],
        ["17", "d5_eccentric_thigh_dist_mean", "G5", "mean(||Bar_ecc - Thigh_ecc||)", "하강 시 허벅지 밀착 이격 평균"],
        ["18", "d5_eccentric_velocity_std", "G5", "std(dY_ecc / dt)", "하강 시 통제력 상실 및 속도 변동성"],
        ["19", "d5_setup_bar_shin_gap", "G5", "||Bar_0 - Ankle_0|| / Scale", "바닥 셋업 시 바벨-발목 초기 세팅 간격"],
        ["20", "d5_lockout_bar_pelvis_dist", "G5", "||Bar_top - Hip_top|| / Scale", "최상단 완전 신전 시 바벨-골반 밀착도"],
        ["21", "d5_sagittal_proj_drift", "G5", "dot(Bar - Hip, n_sagittal)", "시상면 정사영 전방 드리프트"],
        ["22", "d5_coronal_overlap_deficit", "G5", "1.0 - Overlap_Area(Bar, Legs)", "관상면 바벨-다리 오버랩 결손율"],
        ["23", "d5_bar_reversal_smoothness", "G5", "d2Y/dt2 at Apex (방향 전환 가속도)", "최정점 전환 시 충격도 및 부드러움"],
        ["24", "d5_ascent_descent_path_diff", "G5", "Frechet_Distance(Path_up, Path_down)", "상승 궤적과 하강 궤적 간의 불일치도"],
        ["25", "d5_knee_clearance_margin", "G5", "X_bar - X_knee at Y_bar = Y_knee", "무릎 통과 시 바벨과 무릎의 수평 마진"],
        ["26", "d5_shank_contact_duration", "G5", "t_contact / t_first_pull", "정강이 표면 접촉 유지 시간 비율"],
        ["27", "d5_hip_extension_sync_error", "G5", "|t_bar_top - t_hip_extended|", "바벨 최상단 도달과 고관절 신전 시차"],
        ["28", "d5_total_energy_expenditure", "G5", "integral(F_bar * v_bar dt)", "이상 궤적 대비 불필요한 일률 낭비량"],
        # Group 6 (4D)
        ["29", "d5_arm_foreshortening_min_ratio", "G6", "min(L_2D / L_max) (cos theta)", "2D 평면 투영 시 팔 단축률 최소치"],
        ["30", "d5_virtual_sagittal_angle_max", "G6", "arccos(min_ratio) * 180 / pi", "정면 뷰 팔 단축률 역산 가상 전방경사각"],
        ["31", "d5_virtual_sagittal_angle_mean", "G6", "mean(virtual_angle_seq)", "리프팅 구간 가상 전방 경사각 평균"],
        ["32", "d5_frontal_bar_knee_clearance_y", "G6", "(Y_bar - Y_knee) at peak velocity", "정면 뷰 바벨-무릎 수직 상대변위"],
        # Group 7 (8D - Gated)
        ["33★", "d5_gate_is_oblique", "G7", "1.0 if view in [B,D,A,E] else 0.0", "사각/후방 뷰 활성화 게이트 지시자"],
        ["34★", "d5_gated_ke_angle_sek_deg", "G7", "angle(Shoulder, Elbow, Knee) at Nadir", "정상 117.3° vs 결함 99.8° (p=2.11e-9)"],
        ["35★", "d5_gated_ke_elbow_knee_ant_diff", "G7", "(e_ant - k_ant) (정면 C일 때 0.0 차단)", "시점 D p=2.15e-8, 시점 B p=5.07e-5"],
        ["36★", "d5_gated_ke_width_ratio_nadir", "G7", "W_knee / (W_elbow + eps) at Nadir", "무릎/팔꿈치 관상면 너비비율"],
        ["37★", "d5_gated_ke_dist_nadir", "G7", "||Knee - Elbow||_2 / Scale at Nadir", "최하사점 무릎-팔꿈치 3D 유클리드 거리"],
        ["38★", "d5_gated_ke_dist_pull_mean", "G7", "mean(||K_t - E_t||) in First Pull", "퍼스트풀 12프레임 무릎-팔꿈치 평균거리"],
        ["39★", "d5_gated_ke_z_diff_nadir", "G7", "(Z_knee - Z_elbow) (정면 C일 때 0.0 차단)", "최하사점 무릎-팔꿈치 Z축 상대깊이"],
        ["40★", "d5_gated_ke_signed_x_nadir", "G7", "(X_elbow - X_knee) / Scale at Nadir", "최하사점 무릎-팔꿈치 부호화 수평 오프셋"]
    ]

    f_table_data = [f_headers]
    for r in f_rows:
        f_table_data.append([
            Paragraph(r[0], table_cell_center),
            Paragraph(r[1], table_cell_bold if "★" in r[0] else table_cell_style),
            Paragraph(r[2], table_cell_center),
            Paragraph(r[3], table_cell_style),
            Paragraph(r[4], table_cell_style)
        ])

    feat_tbl = Table(f_table_data, colWidths=[0.8*cm, 4.4*cm, 1.0*cm, 6.2*cm, 5.0*cm])
    feat_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_HEX)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(feat_tbl)
    story.append(Spacer(1, 10))

    # Section 4. 모델 아키텍처 및 3-State 파레토 메커니즘
    story.append(Paragraph("4. 현재 모델 아키텍처 및 3-State 파이프라인 (v4.0 Gated)", h1_style))
    sec4_text = (
        "<b>4.1. 3-Way Tri-Ensemble 하이브리드 분류기</b><br/>"
        "현재 d5 모델은 결정 트리 기반 GBDT와 앙상블 기법을 결합한 3-Way 블렌딩을 사용합니다:<br/>"
        "• <b>Branch 1 (LightGBM 50%):</b> n_estimators=180, max_depth=5, lr=0.04, num_leaves=20, colsample=0.85<br/>"
        "• <b>Branch 2 (CatBoost 30%):</b> iterations=180, depth=5, lr=0.04, colsample=0.85<br/>"
        "• <b>Branch 3 (Random Forest 20%):</b> n_estimators=180, max_depth=6 (과적합 완충재 역할)<br/>"
        "• <b>최종 결함 확률:</b> $P_{\\text{final}} = 0.50 \\times p_{\\text{LGB}} + 0.30 \\times p_{\\text{CAT}} + 0.20 \\times p_{\\text{RF}}$<br/><br/>"
        "<b>4.2. 시점 적응형 피처 게이팅 메커니즘 (스쿼트 c2 모델 계승)</b><br/>"
        "정면 뷰(시점 C)에서는 관상면 시야각 한계로 전후 깊이 왜곡 노이즈가 발생하므로, "
        "정면일 때는 전방 돌출도(`d5_gated_ke_elbow_knee_ant_diff`)와 Z축 차이(`d5_gated_ke_z_diff_nadir`)를 "
        "<b><code>0.0</code>으로 완벽 차단(Zero-Masking)</b>하고, 단축 보정(Fore-shortening) 피처에만 집중하게 유도합니다.<br/><br/>"
        "<b>4.3. 3-State 불확실성 기권(Selective Abstention) 파레토 최적화</b><br/>"
        "실서비스에서 애매한 경계 구간의 오진단을 방지하기 위해 3구간 상태 판정기를 적용합니다:<br/>"
        "• $P_{\\text{final}} \\le 0.405$: <b>정상 (NORMAL) 확정</b><br/>"
        "• $P_{\\text{final}} \\ge 0.595$: <b>결함 (DEFECT) 확정</b><br/>"
        "• $0.405 < P_{\\text{final}} < 0.595$: <b>판정 보류 및 재촬영 안내 (UNDETERMINED / RETAKE)</b> (보류율 13.2%, 커버리지 86.8%)"
    )
    story.append(Paragraph(sec4_text, body_style))
    story.append(Spacer(1, 8))

    # Section 5. 실측 성능 현황 및 시점별 성능 감사
    story.append(Paragraph("5. 실측 성능 현황 및 공식 5대 시점 감사 (Empirical Audit)", h1_style))
    story.append(Paragraph("770개 전수 시퀀스에 대한 5-Fold StratifiedGroupKFold 실측 지표 비교입니다:", body_style))

    # 성능 비교 표
    p_headers = [Paragraph("모델 버전", table_cell_header), Paragraph("피처수", table_cell_header), Paragraph("기본 OOF F1", table_cell_header), Paragraph("ROC-AUC", table_cell_header), Paragraph("3-State F1 (커버리지)", table_cell_header), Paragraph("시점 A", table_cell_header), Paragraph("시점 B", table_cell_header), Paragraph("시점 C(정면)", table_cell_header), Paragraph("시점 D", table_cell_header), Paragraph("시점 E", table_cell_header)]
    p_data_rows = [
        ["v1.0 베이스라인", "28D", "76.15%", "0.8521", "79.19% (90.4%)", "74.1%", "84.2%", "48.1%", "95.8%", "74.3%"],
        ["v2.0 파레토 최적화", "28D", "76.15%", "0.8521", "80.23% (89.7%)", "74.1%", "84.2%", "48.1%", "95.8%", "74.3%"],
        ["v3.0 단축보정 Tri", "32D", "78.06%", "0.8712", "81.49% (90.0%)", "75.2%", "85.9%", "55.9%", "96.0%", "75.5%"],
        ["v4.0 TCN 하이브리드", "32D+TCN", "78.29%", "0.8758", "81.75% (90.3%)", "76.3%", "86.1%", "54.8%", "96.1%", "76.0%"],
        ["★ v4.0 Gated", "40D", "79.07%", "0.8788", "83.23% (86.8%)", "79.2%", "87.0%", "56.7%", "96.1%", "75.9%"]
    ]
    p_table_data = [p_headers]
    for r in p_data_rows:
        p_table_data.append([
            Paragraph(r[0], table_cell_bold if "★" in r[0] else table_cell_style),
            Paragraph(r[1], table_cell_center),
            Paragraph(r[2], table_cell_center),
            Paragraph(r[3], table_cell_center),
            Paragraph(r[4], table_cell_bold if "★" in r[0] else table_cell_style),
            Paragraph(r[5], table_cell_center),
            Paragraph(r[6], table_cell_center),
            Paragraph(r[7], table_cell_center),
            Paragraph(r[8], table_cell_center),
            Paragraph(r[9], table_cell_center)
        ])

    perf_tbl = Table(p_table_data, colWidths=[2.6*cm, 1.1*cm, 1.7*cm, 1.5*cm, 2.7*cm, 1.3*cm, 1.3*cm, 1.6*cm, 1.3*cm, 1.3*cm])
    perf_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_HEX)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(perf_tbl)
    story.append(Spacer(1, 8))

    # Section 6. 현존하는 핵심 병목 및 미해결 과제
    story.append(Paragraph("6. 현존하는 3대 핵심 병목 및 미해결 과제 (Core Bottlenecks)", h1_style))
    sec6_text = (
        "<b>[병목 1] 정면 뷰(시점 C)의 2D 평면 투영 깊이 축소 (Depth Flattening Problem)</b><br/>"
        "• <b>현상:</b> 사각 뷰(시점 B, D)는 F1 87%~96%로 완벽하나, 정면 뷰(시점 C)는 F1 56.68%로 극심한 저하를 보임.<br/>"
        "• <b>원인:</b> 카메라 정면에서는 바벨이 다리를 가로막고, 바벨이 전방으로 15cm 이탈해도 2D 평면상에서는 Y축/X축 변위가 거의 "
        "변하지 않아 전후 거리 정보가 사라짐 (결함 미탐 FN이 39건으로 오분류의 60% 이상 집중).<br/>"
        "• <b>현재 대응:</b> 팔 겉보기 길이의 단축률(Fore-shortening) 역삼각 측량으로 48% $\\rightarrow$ 56.7%까지 끌어올렸으나 여전히 60% 미만임.<br/><br/>"
        "<b>[병목 2] 피험자 골격 체형 편향 (Subject Morphology Domain Shift)</b><br/>"
        "• <b>현상:</b> 팔이 긴 피험자(피험자 3번)는 정상 동작에서도 바벨이 무릎보다 전방에 위치하여 과잉 결함(FP)으로 오진단(FP 80% 집중). "
        "반대로 상체가 긴 피험자(피험자 4번, 9번)는 결함 미탐(FN) 다발.<br/>"
        "• <b>원인:</b> 현재 신체 스케일 정규화가 체간 길이(Torso Length)와 대퇴 길이(Femur Length)의 합에만 의존하여, "
        "상체 대비 팔 길이 비율(Arm-to-Torso Ratio) 및 정강이 대비 대퇴 비율(Femur-to-Tibia Ratio) 편차를 흡수하지 못함.<br/><br/>"
        "<b>[병목 3] 기권 없는 기본 OOF F1 80.0%+ 돌파 과제</b><br/>"
        "• 3-State 확신 구간에서는 F1 83.23%로 80%를 훌륭히 넘겼으나, 기권 없이 전수 판정하는 기본 F1은 현재 79.07%임. "
        "순수 기본 분류 성능 자체를 80.0%+로 끌어올릴 수 있는 획기적인 모델링 기법이 요구됨."
    )
    story.append(Paragraph(sec6_text, body_style))
    story.append(Spacer(1, 8))

    # Section 7. 외부 AI에게 요청하는 5대 핵심 질문 및 답변 양식
    story.append(Paragraph("7. 외부 AI 연구 자문 가이드라인 및 5대 핵심 질문 (Prompt Guide)", h1_style))
    sec7_text = (
        "외부 AI 연구 모델은 본 기술 명세서를 면밀히 검토한 뒤, 다음 5대 핵심 질문에 대해 "
        "<b>구체적인 수학 공식, 생체역학적 근거, 알고리즘 아키텍처 및 Python 구현 코드 스니펫</b>을 포함하여 답변해 주시기 바랍니다:<br/><br/>"
        "<b>■ 질문 1 [정면 깊이 복원]:</b><br/>"
        "정면 2D 단안 영상(시점 C)에서 바벨이 몸에서 전방으로 멀어지는 시상면(Sagittal) 깊이 이탈을 "
        "정밀하게 역산/복원할 수 있는 기하학적 피처나 딥러닝 기법(예: 관절 간 사영 면적, 원근 투영 왜곡비 등)은 무엇인가?<br/><br/>"
        "<b>■ 질문 2 [체형 불변 정규화]:</b><br/>"
        "팔이 비정상적으로 긴 피험자(Ape Index > 1.05)나 상체가 긴 피험자에서도 오분류되지 않는 "
        "사지 비율 불변(Body-Proportion Invariant) 정규화 수식 및 상대 변위 피처 설계안은 무엇인가?<br/><br/>"
        "<b>■ 질문 3 [시계열 및 트랜스포머 융합]:</b><br/>"
        "현재 40D GBDT와 결합할 때 시너지 효과를 낼 수 있는 경량 시계열 신경망(예: Spatial-Temporal Graph ConvNet, "
        "MotionBERT 경량화, 또는 Bi-directional GRU Attention)의 최적 아키텍처 및 입력 텐서 구성안은 무엇인가?<br/><br/>"
        "<b>■ 질문 4 [Mixture-of-Experts 시점 융합]:</b><br/>"
        "현재의 '방식 1(피처 0.0 마스킹)'을 넘어, 정면 전담 서브넷과 사각 전담 서브넷을 동적으로 스위칭하는 "
        "시점 특화 Mixture-of-Experts(MoE) 또는 Dynamic Routing 게이팅 모델 설계안은 무엇인가?<br/><br/>"
        "<b>■ 질문 5 [생체역학 킬러 피처 제안]:</b><br/>"
        "사용자가 발견한 '최하사점 무릎-팔꿈치 상대 기하학'처럼, 데드리프트에서 바벨 밀착 위반을 가장 강력하게 잡아낼 수 있는 "
        "새로운 생체역학적 관절 연동 피처(수학 공식 포함) 3~5개를 추가 제안해 달라."
    )
    story.append(Paragraph(sec7_text, body_style))
    story.append(Spacer(1, 10))

    # 문서 빌드
    doc.build(story)
    print(f"[+] PDF 생성 완료: {PDF_PATH}")

    # DOCX 및 MD 동시 생성
    build_docx_document()
    build_md_document()


def build_docx_document():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    title_p = doc.add_paragraph()
    r_sub = title_p.add_run("vidio-health-trainer | 외부 AI 자문용 기술 백서 & RFC 문서\n")
    r_sub.font.name = 'Malgun Gothic'
    r_sub.font.size = Pt(10)
    r_sub.font.bold = True
    r_sub.font.color.rgb = COLOR_SECONDARY

    r_title = title_p.add_run("바벨 데드리프트 d5 (바벨 궤적과 몸 밀착) 모델\n배경 지식, 데이터셋, 40D 피처셋 및 아키텍처 기술 명세서")
    r_title.font.name = 'Malgun Gothic'
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(10)
    r_meta = meta_p.add_run(
        "• 프로젝트명: vidio-health-trainer | 타깃 운동: 바벨 데드리프트 (Barbell Deadlift)\n"
        "• 분석 대상 결함: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)\n"
        "• 문서 목적: 외부 최신 AI 모델(GPT-4o, Claude 3.5, Gemini Pro 등)에게 시스템의 모든 데이터, 피처, "
        "모델 파이프라인 및 실측 성능을 제공하여 기본 F1 80%+ 돌파 및 정면 뷰(시점 C) 깊이 축소 극복 솔루션을 자문받기 위함.\n"
        "• 작성 일자: 2026년 9월 30일 | 최신 모델 버전: v4.0 Gated (40D 시점 적응형 게이팅)"
    )
    r_meta.font.name = 'Malgun Gothic'
    r_meta.font.size = Pt(9)
    r_meta.font.color.rgb = COLOR_MUTED

    # 본문 섹션들
    sections = [
        ("1. 타깃 결함 d5의 생체역학 정의 및 판정 기준", [
            "■ 1.1. 결함의 물리적 정의\n데드리프트에서 바벨 봉(Barbell)은 인체의 무게중심(Center of Mass)에 가장 가까운 정강이(Shank), 무릎(Knee), 허벅지(Thigh) 전면을 스치듯이 수직 선상으로 이동해야 합니다. 바벨이 몸에서 전방으로 멀어질 경우, 고관절 및 요추 L5/S1 관절에 작용하는 모멘트암(Moment Arm = 거리 x 하중)이 기하급수적으로 증가하여 극심한 척추 부상 및 리프팅 실패를 유발합니다.",
            "■ 1.2. 정상 동작 (Label = 0, Normal)\n• 셋업부터 최하사점(Nadir), 퍼스트풀, 락아웃(Lockout)까지 바벨이 정강이/허벅지 표면에서 2~3cm 이내로 밀착 유지.\n• 측면/사각 뷰 관찰: 바벨 봉이 수직선(Vertical Line)을 그리며 상승하고, 최하사점에서 무릎이 팔 안쪽에 깊숙이 안착.",
            "■ 1.3. 결함 동작 (Label = 1, Defect)\n• 바벨이 정강이/무릎을 스치지 못하고 무릎 앞쪽으로 주먹 1~2개 이상(전방 이탈) 둥글게 원을 그리며 돌아나옴.\n• 측면/사각 뷰 관찰: 바벨을 앞으로 멀리 잡으면서 팔꿈치가 앞으로 꺾이고 무릎이 팔꿈치 뒤로 빠지는 기하학적 역전 발생."
        ]),
        ("2. 데이터셋 규격 및 전처리 파이프라인 (Data Pipeline Invariants)", [
            "■ 2.1. 원천 데이터 및 전처리 규격\n• 원천 데이터셋: AI-Hub 헬스케어 피트니스 데이터셋 (운동 02. 바벨 데드리프트)\n• 전처리 파이프라인: 단일 횟수 분할 -> 64프레임 3차 스플라인(Cubic Spline) 시간 정규화 -> MediaPipe Pose 33개 관절 3D 키포인트 추출.\n• 입력 텐서 규격: 시퀀스당 (T=64, J=33, C=4) (x, y, z, visibility)\n• 총 시퀀스 수 및 클래스 균형: 총 770개 시퀀스 (정상 385건 : 결함 385건, 50:50 완벽 균형)",
            "■ 2.2. 피험자 전수 원칙 (All Subjects Invariant)\nAI-Hub 피험자 1번부터 9번까지 전원(9명)이 데이터셋에 포함되어 있으며, 피험자 간 체형 편향을 검증하기 위해 5-Fold StratifiedGroupKFold (피험자 완전 격리 교차검증)를 필수로 적용합니다.",
            "■ 2.3. 공식 5대 카메라 시점 명칭 및 매핑 (Official 5-View Taxonomy)\n• 시점 A: 우측 후방 45도 (view5, 154건) - 광배근 결합도 및 바벨-체간 후방 이격 관측\n• 시점 B: 우측 전방 45도 (view2, 154건) - 무릎 통과 시 바벨 전방 이탈 골든 앵글 (F1 87.0%)\n• 시점 C: 정면 0도 (view3, 154건) - [최대 병목] 시상면 깊이 축소로 전후 이격거리 소실 (F1 56.7%)\n• 시점 D: 좌측 전방 45도 (view4, 154건) - 최고 정확도 시점, 바벨-무릎 간극 완벽 분별 (F1 96.1%)\n• 시점 E: 좌측 후방 45도 (view1, 154건) - 후방 사각 뷰, 바벨-체간 이격 분별 양호 (F1 75.9%)",
            "■ 2.4. 엄격한 데이터 격리 및 무누수 원칙 (Strict No-Leakage Invariant)\nAI-Hub 라벨링 JSON에 포함된 3D 정답 좌표나 결함 어노테이션은 오직 정답 타깃(y) 및 성능 채점에만 사용됩니다. 모델의 입력 피처(X) 파이프라인에 AI-Hub JSON 정보가 간접 참조되거나 유입되는 것은 영구 엄금되며, 100% MediaPipe Pose 단안 비전 키포인트만을 기반으로 피처를 추출해야 합니다."
        ]),
        ("3. 피처 엔지니어링 전수 명세서 (40D Full Specifications)", [
            "현재 v4.0 Gated 모델에 탑재된 총 40차원 피처의 상세 내역입니다:\n"
            "• Group 1 (2D): 정강이-바벨 최대/평균 이격거리 (d5_bar_shank_dist_max/mean)\n"
            "• Group 2 (4D): 허벅지-바벨 평균/최대 이격거리, 허벅지 밀착 슬라이딩율, 다리 접촉 결손율\n"
            "• Group 3 (4D): 바벨 궤적 곡률, 상승 속도 매끄러움, 좌우 흔들림 표준편차, 최대 수평 이동폭\n"
            "• Group 4 (5D): 고관절 모멘트암, 광배근 체간-팔 연동 결손도, 무릎 통과 시점, 팔 전방 경사각 max/mean\n"
            "• Group 5 (13D): 이완기(하강) 바벨 이탈 최대폭, 하강 허벅지 이격, 하강 속도 변동성, 셋업 초기 간격, 락아웃 밀착도, 시상면 드리프트, 관상면 오버랩 결손율, 정점 가속도, 승강 궤적 불일치도(Frechet Distance), 무릎 통과 마진, 접촉 시간비, 고관절 신전 시차, 일률 낭비량\n"
            "• Group 6 (4D): 팔 단축비(Fore-shortening), 가상 시상면 각도 max/mean, 정면 바벨-무릎 수직 상대변위\n"
            "• Group 7 (8D, Gated): 사각 뷰 게이트 지시자(is_oblique), 어깨-팔꿈치-무릎 사잇각(SEK deg, p=2.11e-9), 체간 전방 돌출도 차이(정면 0.0 차단, p=2.15e-8), 무릎/팔꿈치 너비비, 최하사점 3D 거리, 퍼스트풀 평균거리, Z축 상대깊이(정면 0.0 차단), X축 상대 오프셋"
        ]),
        ("4. 현재 모델 아키텍처 및 3-State 파이프라인 (v4.0 Gated)", [
            "• 3-Way Tri-Ensemble: LightGBM(50%) + CatBoost(30%) + Random Forest(20%)\n"
            "• 시점 적응형 피처 게이팅: 정면 C일 때 전방 돌출도와 Z축 차이를 0.0으로 Zero-Masking 처리하여 깊이 노이즈 원천 차단\n"
            "• 3-State 불확실성 기권: P <= 0.405 (정상 확정), P >= 0.595 (결함 확정), 0.405 < P < 0.595 (판정 보류/재촬영 안내, 보류율 13.2%, 커버리지 86.8%)"
        ]),
        ("5. 실측 성능 현황 및 공식 5대 시점 감사 (Empirical Audit)", [
            "• v1.0 Baseline: F1 76.15% | AUC 0.8521 | 3-State F1 79.19% (Cov 90.4%)\n"
            "• v2.0 Pareto: F1 76.15% | AUC 0.8521 | 3-State F1 80.23% (Cov 89.7%)\n"
            "• v3.0 Dynamic: F1 78.06% | AUC 0.8712 | 3-State F1 81.49% (Cov 90.0%)\n"
            "• v4.0 Hybrid: F1 78.29% | AUC 0.8758 | 3-State F1 81.75% (Cov 90.3%)\n"
            "• ★ v4.0 Gated: 기본 OOF F1 79.07% | ROC-AUC 0.8788 | 3-State F1 83.23% (Cov 86.8%) / 85.33% (Cov 78.8%)\n"
            "• 시점별 F1: 시점 A(79.2%), 시점 B(87.0%), 시점 C(56.7%), 시점 D(96.1%), 시점 E(75.9%) (5대 전 시점 동반 상승 달성)"
        ]),
        ("6. 현존하는 3대 핵심 병목 및 미해결 과제 (Core Bottlenecks)", [
            "1. [병목 1] 정면 뷰(시점 C)의 2D 평면 투영 깊이 축소 (Depth Flattening Problem): 시상면 깊이가 겹쳐 결함 미탐(FN 39건) 집중.\n"
            "2. [병목 2] 피험자 골격 체형 편향 (Subject Morphology Domain Shift): 팔이 긴 피험자(피험자 3)는 정상이어도 과잉 결함(FP 80%), 상체가 긴 피험자(피험자 4, 9)는 결함 미탐(FN) 다발.\n"
            "3. [병목 3] 기권 없는 기본 OOF F1 80.0%+ 돌파: 현재 79.07%에서 기권 없이 80.0%+를 달성하기 위한 차세대 모델링 요구."
        ]),
        ("7. 외부 AI 연구 자문 가이드라인 및 5대 핵심 질문 (Prompt Guide)", [
            "외부 AI 모델은 다음 5대 질문에 대해 수식, 생체역학 근거, Python 구현 코드와 함께 답변해 주시기 바랍니다:\n\n"
            "■ 질문 1 [정면 깊이 복원]: 정면 2D 단안 영상(시점 C)에서 바벨-다리 시상면 깊이 이탈을 정밀하게 역산/복원할 기하학적 피처 또는 딥러닝 기법은?\n"
            "■ 질문 2 [체형 불변 정규화]: 팔 길이 및 상체 비율 편차에 불변(Body-Proportion Invariant)한 정규화 수식 및 상대 변위 피처 설계안은?\n"
            "■ 질문 3 [시계열 및 트랜스포머 융합]: 40D GBDT와 결합할 경량 Spatio-Temporal Graph ConvNet 또는 MotionBERT/GRU Attention 융합 아키텍처는?\n"
            "■ 질문 4 [Mixture-of-Experts 시점 융합]: 정면 전담 서브넷과 사각 전담 서브넷을 동적 스위칭하는 MoE 또는 Dynamic Routing 설계안은?\n"
            "■ 질문 5 [생체역학 킬러 피처 제안]: 바벨 데드리프트 d5 결함을 가장 확실하게 판별할 수 있는 신규 생체역학 피처 3~5개 제안."
        ])
    ]

    for title, paras in sections:
        h = doc.add_heading(title, level=1)
        h.runs[0].font.name = 'Malgun Gothic'
        h.runs[0].font.color.rgb = COLOR_PRIMARY
        for p_txt in paras:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.25
            r = p.add_run(p_txt)
            r.font.name = 'Malgun Gothic'
            r.font.size = Pt(9.5)
            r.font.color.rgb = COLOR_TEXT

    doc.save(DOCX_PATH)
    print(f"[+] DOCX 생성 완료: {DOCX_PATH}")


def build_md_document():
    md_content = """# vidio-health-trainer | 외부 AI 자문용 기술 백서 & RFC 문서
# 바벨 데드리프트 d5 (바벨 궤적과 몸 밀착) 모델 배경 지식, 데이터셋, 40D 피처셋 및 아키텍처 기술 명세서

- **프로젝트명:** `vidio-health-trainer` | **타깃 운동:** 바벨 데드리프트 (Barbell Deadlift)
- **분석 대상 결함:** `d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)`
- **문서 목적:** 외부 최신 AI 모델(GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro 등)에게 시스템의 모든 데이터셋 명세, 40D 피처 수식, 모델 아키텍처, 실측 성능 및 병목 요인을 투명하게 제공하여, **기본 OOF F1 80.0%+ 돌파 및 정면 뷰(시점 C) 깊이 축소 극복 솔루션**을 자문받기 위함.
- **작성 일자:** 2026년 9월 30일 | **최신 모델 버전:** `v4.0 Gated` (40D 시점 적응형 게이팅)

---

## [외부 AI 연구 자문 핵심 과제]
현재 본 시스템은 40D 시점 적응형 게이팅(`v4.0 Gated`)을 통해 **3-State 확신 구간 F1 83.23%(커버리지 86.8%), ROC-AUC 0.8788**을 달성하였습니다.  
그러나 **기권 없는 기본 OOF F1-Score는 79.07%로 80.0% 벽 직전에 정체**되어 있으며, 특히 **정면 뷰(시점 C)의 2D 평면 투영 깊이 축소(Depth Flattening)로 인한 결함 미탐(F1 56.68%)**이 최대 병목입니다.  
외부 AI는 본 문서에 수록된 40D 피처 명세와 시점별/체형별 오분류 분석을 바탕으로, **80.0%+ 돌파를 위한 구체적이고 수학적인 신규 피처 및 아키텍처 해결책**을 제시해야 합니다.

---

## 1. 타깃 결함 d5의 생체역학 정의 및 판정 기준

### 1.1. 결함의 물리적 정의
데드리프트에서 바벨 봉(Barbell)은 인체의 무게중심(Center of Mass)에 가장 가까운 정강이(Shank), 무릎(Knee), 허벅지(Thigh) 전면을 스치듯이 수직 선상으로 이동해야 합니다. 바벨이 몸에서 전방으로 멀어질 경우, 고관절 및 요추 L5/S1 관절에 작용하는 모멘트암(Moment Arm = 거리 x 하중)이 기하급수적으로 증가하여 극심한 척추 부상 및 리프팅 실패를 유발합니다.

### 1.2. 정상 동작 (Label = 0, Normal)
- 셋업부터 최하사점(Nadir), 퍼스트풀, 락아웃(Lockout)까지 바벨이 정강이/허벅지 표면에서 2~3cm 이내로 밀착 유지.
- 측면/사각 뷰 관찰: 바벨 봉이 수직선(Vertical Line)을 그리며 상승하고, 최하사점에서 무릎이 팔 안쪽에 깊숙이 안착.

### 1.3. 결함 동작 (Label = 1, Defect)
- 바벨이 정강이/무릎을 스치지 못하고 무릎 앞쪽으로 주먹 1~2개 이상(전방 이탈) 둥글게 원을 그리며 돌아나옴.
- 측면/사각 뷰 관찰: 바벨을 앞으로 멀리 잡으면서 팔꿈치가 앞으로 꺾이고 무릎이 팔꿈치 뒤로 빠지는 기하학적 역전 발생.

---

## 2. 데이터셋 규격 및 전처리 파이프라인 (Data Pipeline Invariants)

### 2.1. 원천 데이터 및 전처리 규격
- **원천 데이터셋:** AI-Hub 헬스케어 피트니스 데이터셋 (운동 02. 바벨 데드리프트)
- **전처리 파이프라인:** 연속 비디오에서 1회 반복(단일 횟수, Repetition) 엄밀 분할 $\\rightarrow$ **64프레임 3차 스플라인(Cubic Spline) 시간 정규화** $\\rightarrow$ MediaPipe Pose 33개 관절 3D 키포인트 추출.
- **입력 텐서 규격:** 시퀀스당 `(T=64, J=33, C=4)` ($x, y, z, \\text{visibility}$)
- **총 시퀀스 수 및 클래스 균형:** **총 770개 시퀀스 (정상 385건 : 결함 385건, 50:50 완벽 균형)**

### 2.2. 피험자 전수 원칙 (All Subjects Invariant)
AI-Hub 피험자 1번부터 9번까지 전원(9명)이 데이터셋에 포함되어 있으며, 피험자 간 체형 편향을 검증하기 위해 **5-Fold StratifiedGroupKFold (피험자 완전 격리 교차검증)**를 필수로 적용합니다.

### 2.3. 공식 5대 카메라 시점 명칭 및 매핑 (Official 5-View Taxonomy)
| 공식 시점 명칭 | 내부 코드 | 카메라 설치 각도 | 샘플수 | 시점 특성 및 d5 판정 난이도 |
| :--- | :---: | :---: | :---: | :--- |
| **시점 A** | `view5` | 우측 후방 (45도) | 154건 | 광배근 결합도 및 바벨-체간 후방 이격 관측 우수 (F1 79.2%) |
| **시점 B** | `view2` | 우측 전방 (45도) | 154건 | 무릎 통과 시 바벨 전방 이탈 골든 앵글 (F1 87.0%) |
| **시점 C** | `view3` | 정면 (0도) | 154건 | **[최대 병목]** 시상면 깊이 축소로 전후 이격거리 소실 (F1 56.7%) |
| **시점 D** | `view4` | 좌측 전방 (45도) | 154건 | 최고 정확도 시점, 바벨-무릎 간극 완벽 분별 (F1 96.1%) |
| **시점 E** | `view1` | 좌측 후방 (45도) | 154건 | 후방 사각 뷰, 바벨-체간 이격 분별 양호 (F1 75.9%) |

### 2.4. 엄격한 데이터 격리 및 무누수 원칙 (Strict No-Leakage Invariant)
AI-Hub 라벨링 JSON에 포함된 3D 정답 좌표나 결함 어노테이션은 오직 정답 타깃($y$) 및 성능 채점에만 사용됩니다. 모델의 입력 피처($X$) 파이프라인에 AI-Hub JSON 정보가 간접 참조되거나 유입되는 것은 영구 엄금되며, **100% MediaPipe Pose 단안 비전 키포인트만을 기반**으로 피처를 추출해야 합니다.

---

## 3. 피처 엔지니어링 전수 명세서 (40D Full Specifications)

| No | 피처 식별자 (Feature Name) | 그룹 | 수학적 연산식 및 물리적 정의 | 정상 vs 결함 특성 |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `d5_bar_shank_dist_max` | G1 | $\\max(\\|\\text{Bar}_t - \\text{Shank}_t\\|)$ ($t \\in \\text{FirstPull}$) | 정상 0.22 vs 결함 0.56 (정강이 최대 이격) |
| 2 | `d5_bar_shank_dist_mean` | G1 | $\\text{mean}(\\|\\text{Bar}_t - \\text{Shank}_t\\|)$ ($t \\in \\text{FirstPull}$) | 정강이 통과 구간 평균 이격도 |
| 3 | `d5_bar_thigh_dist_mean` | G2 | $\\text{mean}(\\|\\text{Bar}_t - \\text{Thigh}_t\\|)$ ($t \\in \\text{Knee}\\sim\\text{Lockout}$) | 허벅지 슬라이딩 밀착 평균 거리 |
| 4 | `d5_bar_thigh_dist_max` | G2 | $\\max(\\|\\text{Bar}_t - \\text{Thigh}_t\\|)$ ($t \\in \\text{Knee}\\sim\\text{Lockout}$) | 허벅지 통과 시 최대 전방 이탈거리 |
| 5 | `d5_p2_thigh_drag_contact_rate` | G2 | $\\sum(\\text{dist} < 0.05) / T_{p2}$ | 허벅지 표면 밀착 접촉 유지 프레임 비율 |
| 6 | `d5_bar_leg_contact_deficit_ratio`| G2 | $1.0 - (\\text{Contact\\_frames} / \\text{Active\\_frames})$ | 전 리프팅 중 다리 비접촉 결손율 |
| 7 | `d5_bar_path_curvature` | G3 | $\\int |\\kappa(t)| dt$ (바벨 궤적 2차 미분) | 정상 1.12 vs 결함 2.45 (궤적 곡률) |
| 8 | `d5_bar_ascent_velocity_smoothness`| G3 | $\\text{std}(dY/dt) / (\\text{mean}(\\|dY/dt\\|) + \\epsilon)$ | 바벨 수직 상승 속도 매끄러움 |
| 9 | `d5_bar_lateral_wobble_std` | G3 | $\\text{std}(X_{\\text{bar}}(t))$ (상승 중 좌우 흔들림) | 정상 0.015 vs 결함 0.042 (좌우 요동) |
| 10 | `d5_bar_total_horizontal_drift` | G3 | $\\max(X_{\\text{bar}}) - \\min(X_{\\text{bar}})$ | 전 구간 바벨 최대 수평 이동폭 |
| 11 | `d5_hip_bar_moment_arm_at_knee` | G4 | $|X_{\\text{hip}} - X_{\\text{bar}}|$ at Knee Pass | 무릎 통과 시 고관절-바벨 수평 모멘트암 |
| 12 | `d5_torso_arm_coupling_deficit` | G4 | $\\text{std}(\\angle(\\text{Torso}, \\text{Arm}))$ in First Pull | 광배근 체간-팔 연동 결합도 결손 |
| 13 | `d5_knee_pass_timing_norm` | G4 | $t(\\text{Bar}_y = \\text{Knee}_y) / T_{\\text{ascent}}$ | 바벨이 무릎을 통과하는 정규화 시점 |
| 14 | `d5_shoulder_wrist_angle_max` | G4 | $\\max(\\angle(\\text{Shoulder}, \\text{Wrist}, \\text{Vertical}))$ | 팔이 수직선에서 전방으로 뻗은 최대각 |
| 15 | `d5_shoulder_wrist_angle_mean` | G4 | $\\text{mean}(\\angle(\\text{Shoulder}, \\text{Wrist}, \\text{Vertical}))$ | 리프팅 중 전방 경사각 평균 |
| 16 | `d5_eccentric_bar_drift_max` | G5 | $\\max(X_{\\text{bar\\_ecc}} - X_{\\text{bar\\_start}})$ | 하강(이완기) 바벨 전방 이탈 최대폭 |
| 17 | `d5_eccentric_thigh_dist_mean` | G5 | $\\text{mean}(\\|\\text{Bar}_{\\text{ecc}} - \\text{Thigh}_{\\text{ecc}}\\|)$ | 하강 시 허벅지 밀착 이격 평균 |
| 18 | `d5_eccentric_velocity_std` | G5 | $\\text{std}(dY_{\\text{ecc}} / dt)$ | 하강 시 통제력 상실 및 속도 변동성 |
| 19 | `d5_setup_bar_shin_gap` | G5 | $\\|\\text{Bar}_0 - \\text{Ankle}_0\\| / \\text{Scale}$ | 바닥 셋업 시 바벨-발목 초기 세팅 간격 |
| 20 | `d5_lockout_bar_pelvis_dist` | G5 | $\\|\\text{Bar}_{\\text{top}} - \\text{Hip}_{\\text{top}}\\| / \\text{Scale}$ | 최상단 완전 신전 시 바벨-골반 밀착도 |
| 21 | `d5_sagittal_proj_drift` | G5 | $\\text{dot}(\\text{Bar} - \\text{Hip}, \\vec{n}_{\\text{sagittal}})$ | 시상면 정사영 전방 드리프트 |
| 22 | `d5_coronal_overlap_deficit` | G5 | $1.0 - \\text{Overlap\\_Area}(\\text{Bar}, \\text{Legs})$ | 관상면 바벨-다리 오버랩 결손율 |
| 23 | `d5_bar_reversal_smoothness` | G5 | $d^2Y/dt^2$ at Apex (방향 전환 가속도) | 최정점 전환 시 충격도 및 부드러움 |
| 24 | `d5_ascent_descent_path_diff` | G5 | $\\text{Frechet\\_Distance}(\\text{Path}_{\\text{up}}, \\text{Path}_{\\text{down}})$ | 상승 궤적과 하강 궤적 간의 불일치도 |
| 25 | `d5_knee_clearance_margin` | G5 | $X_{\\text{bar}} - X_{\\text{knee}}$ at $Y_{\\text{bar}} = Y_{\\text{knee}}$ | 무릎 통과 시 바벨과 무릎의 수평 마진 |
| 26 | `d5_shank_contact_duration` | G5 | $t_{\\text{contact}} / t_{\\text{first\\_pull}}$ | 정강이 표면 접촉 유지 시간 비율 |
| 27 | `d5_hip_extension_sync_error` | G5 | $|t_{\\text{bar\\_top}} - t_{\\text{hip\\_extended}}|$ | 바벨 최상단 도달과 고관절 신전 시차 |
| 28 | `d5_total_energy_expenditure` | G5 | $\\int F_{\\text{bar}} \\cdot v_{\\text{bar}} dt$ | 이상 궤적 대비 불필요한 일률 낭비량 |
| 29 | `d5_arm_foreshortening_min_ratio`| G6 | $\\min(L_{2D} / L_{\\max})$ ($\\cos\\theta$) | 2D 평면 투영 시 팔 단축률 최소치 |
| 30 | `d5_virtual_sagittal_angle_max` | G6 | $\\arccos(\\text{min\\_ratio}) \\times 180 / \\pi$ | 정면 뷰 팔 단축률 역산 가상 전방경사각 |
| 31 | `d5_virtual_sagittal_angle_mean`| G6 | $\\text{mean}(\\text{virtual\\_angle\\_seq})$ | 리프팅 구간 가상 전방 경사각 평균 |
| 32 | `d5_frontal_bar_knee_clearance_y`| G6 | $(Y_{\\text{bar}} - Y_{\\text{knee}})$ at peak velocity | 정면 뷰 바벨-무릎 수직 상대변위 |
| **33★** | `d5_gate_is_oblique` | G7 | $1.0$ if view in [B,D,A,E] else $0.0$ | 사각/후방 뷰 활성화 게이트 지시자 |
| **34★** | `d5_gated_ke_angle_sek_deg` | G7 | $\\angle(\\text{Shoulder}, \\text{Elbow}, \\text{Knee})$ at Nadir | 정상 117.3° vs 결함 99.8° ($p=2.11 \\times 10^{-9}$) |
| **35★** | `d5_gated_ke_elbow_knee_ant_diff`| G7 | $(e_{\\text{ant}} - k_{\\text{ant}})$ (정면 C일 때 $0.0$ 차단) | 시점 D $p=2.15 \\times 10^{-8}$, 시점 B $p=5.07 \\times 10^{-5}$ |
| **36★** | `d5_gated_ke_width_ratio_nadir` | G7 | $W_{\\text{knee}} / (W_{\\text{elbow}} + \\epsilon)$ at Nadir | 무릎/팔꿈치 관상면 너비비율 |
| **37★** | `d5_gated_ke_dist_nadir` | G7 | $\\|\\text{Knee} - \\text{Elbow}\\|_2 / \\text{Scale}$ at Nadir | 최하사점 무릎-팔꿈치 3D 유클리드 거리 |
| **38★** | `d5_gated_ke_dist_pull_mean` | G7 | $\\text{mean}(\\|K_t - E_t\\|)$ in First Pull | 퍼스트풀 12프레임 무릎-팔꿈치 평균거리 |
| **39★** | `d5_gated_ke_z_diff_nadir` | G7 | $(Z_{\\text{knee}} - Z_{\\text{elbow}})$ (정면 C일 때 $0.0$ 차단) | 최하사점 무릎-팔꿈치 Z축 상대깊이 |
| **40★** | `d5_gated_ke_signed_x_nadir` | G7 | $(X_{\\text{elbow}} - X_{\\text{knee}}) / \\text{Scale}$ at Nadir | 최하사점 무릎-팔꿈치 부호화 수평 오프셋 |

---

## 4. 현재 모델 아키텍처 및 3-State 파이프라인 (v4.0 Gated)

### 4.1. 3-Way Tri-Ensemble 하이브리드 분류기
- **Branch 1 (LightGBM 50%):** `n_estimators=180, max_depth=5, lr=0.04, num_leaves=20, colsample=0.85`
- **Branch 2 (CatBoost 30%):** `iterations=180, depth=5, lr=0.04, colsample=0.85`
- **Branch 3 (Random Forest 20%):** `n_estimators=180, max_depth=6`
- **최종 확률 블렌딩:** $P_{\\text{final}} = 0.50 \\times p_{\\text{LGB}} + 0.30 \\times p_{\\text{CAT}} + 0.20 \\times p_{\\text{RF}}$

### 4.2. 시점 적응형 피처 게이팅 메커니즘 (스쿼트 c2 모델 계승)
정면 뷰(시점 C)에서는 관상면 시야각 한계로 전후 깊이 왜곡 노이즈가 발생하므로, 정면일 때는 전방 돌출도(`d5_gated_ke_elbow_knee_ant_diff`)와 Z축 차이(`d5_gated_ke_z_diff_nadir`)를 **`0.0`으로 완벽 차단(Zero-Masking)**하고, 단축 보정(Fore-shortening) 피처에만 집중하게 유도합니다.

### 4.3. 3-State 불확실성 기권(Selective Abstention) 파레토 최적화
- $P_{\\text{final}} \\le 0.405$: **정상 (NORMAL) 확정**
- $P_{\\text{final}} \\ge 0.595$: **결함 (DEFECT) 확정**
- $0.405 < P_{\\text{final}} < 0.595$: **판정 보류 및 재촬영 안내 (RETAKE)** (보류율 13.2%, 커버리지 86.8%)

---

## 5. 실측 성능 현황 및 공식 5대 시점 감사 (Empirical Audit)

| 모델 버전 | 피처수 | 기본 OOF F1 | ROC-AUC | 3-State F1 (커버리지) | 시점 A | 시점 B | 시점 C(정면) | 시점 D | 시점 E |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1.0 베이스라인** | 28D | 76.15% | 0.8521 | 79.19% (90.4%) | 74.1% | 84.2% | 48.1% | 95.8% | 74.3% |
| **v2.0 파레토 최적화** | 28D | 76.15% | 0.8521 | 80.23% (89.7%) | 74.1% | 84.2% | 48.1% | 95.8% | 74.3% |
| **v3.0 단축보정 Tri** | 32D | 78.06% | 0.8712 | 81.49% (90.0%) | 75.2% | 85.9% | 55.9% | 96.0% | 75.5% |
| **v4.0 TCN 하이브리드** | 32D+TCN | 78.29% | 0.8758 | 81.75% (90.3%) | 76.3% | 86.1% | 54.8% | 96.1% | 76.0% |
| **★ v4.0 Gated** | **40D** | **79.07%** | **0.8788** | **83.23% (86.8%)** | **79.2%** | **87.0%** | **56.7%** | **96.1%** | **75.9%** |

---

## 6. 현존하는 3대 핵심 병목 및 미해결 과제 (Core Bottlenecks)

### [병목 1] 정면 뷰(시점 C)의 2D 평면 투영 깊이 축소 (Depth Flattening Problem)
- **현상:** 사각 뷰(시점 B, D)는 F1 87%~96%로 완벽하나, 정면 뷰(시점 C)는 F1 56.68%로 극심한 저하를 보임.
- **원인:** 카메라 정면에서는 바벨이 다리를 가로막고, 바벨이 전방으로 15cm 이탈해도 2D 평면상에서는 Y축/X축 변위가 거의 변하지 않아 전후 거리 정보가 사라짐 (결함 미탐 FN이 39건으로 오분류의 60% 이상 집중).
- **현재 대응:** 팔 겉보기 길이의 단축률(Fore-shortening) 역삼각 측량으로 48% $\\rightarrow$ 56.7%까지 끌어올렸으나 여전히 60% 미만임.

### [병목 2] 피험자 골격 체형 편향 (Subject Morphology Domain Shift)
- **현상:** 팔이 긴 피험자(피험자 3번)는 정상 동작에서도 바벨이 무릎보다 전방에 위치하여 과잉 결함(FP)으로 오진단(FP 80% 집중). 반대로 상체가 긴 피험자(피험자 4번, 9번)는 결함 미탐(FN) 다발.
- **원인:** 현재 신체 스케일 정규화가 체간 길이(Torso Length)와 대퇴 길이(Femur Length)의 합에만 의존하여, 상체 대비 팔 길이 비율(Arm-to-Torso Ratio) 및 정강이 대비 대퇴 비율(Femur-to-Tibia Ratio) 편차를 흡수하지 못함.

### [병목 3] 기권 없는 기본 OOF F1 80.0%+ 돌파 과제
- 3-State 확신 구간에서는 F1 83.23%로 80%를 훌륭히 넘겼으나, 기권 없이 전수 판정하는 기본 F1은 현재 79.07%임. 순수 기본 분류 성능 자체를 80.0%+로 끌어올릴 수 있는 획기적인 모델링 기법이 요구됨.

---

## 7. 외부 AI 연구 자문 가이드라인 및 5대 핵심 질문 (Prompt Guide)

외부 AI 연구 모델은 본 기술 명세서를 면밀히 검토한 뒤, 다음 5대 핵심 질문에 대해 **구체적인 수학 공식, 생체역학적 근거, 알고리즘 아키텍처 및 Python 구현 코드 스니펫**을 포함하여 답변해 주시기 바랍니다:

1. **질문 1 [정면 깊이 복원]:**  
   정면 2D 단안 영상(시점 C)에서 바벨이 몸에서 전방으로 멀어지는 시상면(Sagittal) 깊이 이탈을 정밀하게 역산/복원할 수 있는 기하학적 피처나 딥러닝 기법(예: 관절 간 사영 면적, 원근 투영 왜곡비 등)은 무엇인가?
2. **질문 2 [체형 불변 정규화]:**  
   팔이 비정상적으로 긴 피험자(Ape Index > 1.05)나 상체가 긴 피험자에서도 오분류되지 않는 사지 비율 불변(Body-Proportion Invariant) 정규화 수식 및 상대 변위 피처 설계안은 무엇인가?
3. **질문 3 [시계열 및 트랜스포머 융합]:**  
   현재 40D GBDT와 결합할 때 시너지 효과를 낼 수 있는 경량 시계열 신경망(예: Spatial-Temporal Graph ConvNet, MotionBERT 경량화, 또는 Bi-directional GRU Attention)의 최적 아키텍처 및 입력 텐서 구성안은 무엇인가?
4. **질문 4 [Mixture-of-Experts 시점 융합]:**  
   현재의 '방식 1(피처 0.0 마스킹)'을 넘어, 정면 전담 서브넷과 사각 전담 서브넷을 동적으로 스위칭하는 시점 특화 Mixture-of-Experts(MoE) 또는 Dynamic Routing 게이팅 모델 설계안은 무엇인가?
5. **질문 5 [생체역학 킬러 피처 제안]:**  
   사용자가 발견한 '최하사점 무릎-팔꿈치 상대 기하학'처럼, 데드리프트에서 바벨 밀착 위반을 가장 확실하게 잡아낼 수 있는 새로운 생체역학적 관절 연동 피처(수학 공식 포함) 3~5개를 추가 제안해 달라.
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] MD 생성 완료: {MD_PATH}")


if __name__ == "__main__":
    build_pdf_document()
