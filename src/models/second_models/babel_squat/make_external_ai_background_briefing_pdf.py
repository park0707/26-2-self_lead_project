# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_external_ai_background_briefing_pdf.py
Purpose: 외부 AI 기술 자문용 바벨 스쿼트 통합 진단 엔진 v9.0 상세 아키텍처 및 배경지식 명세서 PDF/DOCX 생성
Targets:
  1) D:\자주프 문서\vidio-health-trainer_외부AI_자문용_바벨스쿼트_통합진단_배경지식_및_아키텍처_명세서.pdf
  2) D:\자주프 문서\5주차\vidio-health-trainer_외부AI_자문용_바벨스쿼트_통합진단_배경지식_및_아키텍처_명세서.pdf
  3) D:\자주프 문서\5주차\vidio-health-trainer_외부AI_자문용_바벨스쿼트_통합진단_배경지식_및_아키텍처_명세서.docx
"""

import os
import sys
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 1. 폰트 등록
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"
pdfmetrics.registerFont(TTFont("Malgun", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunBold", FONT_BOLD_PATH))

# 2. 색상 정의
COLOR_PRIMARY = colors.HexColor("#1A365D")    # 딥 네이비
COLOR_SECONDARY = colors.HexColor("#2B6CB0")  # 블루
COLOR_ACCENT = colors.HexColor("#319795")     # 틸
COLOR_DARK = colors.HexColor("#2D3748")       # 본문 텍스트
COLOR_LIGHT_BG = colors.HexColor("#F7FAFC")   # 테이블 배경
COLOR_HIGHLIGHT = colors.HexColor("#EBF8FF")  # 강조 배경
COLOR_BORDER = colors.HexColor("#CBD5E0")     # 테두리 회색

def build_pdf(target_pdf_path):
    os.makedirs(os.path.dirname(target_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        target_pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # 커스텀 스타일
    style_title = ParagraphStyle(
        'DocTitle',
        fontName='MalgunBold',
        fontSize=20,
        leading=26,
        textColor=COLOR_PRIMARY,
        alignment=1, # Center
        spaceAfter=6
    )

    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        fontName='Malgun',
        fontSize=11,
        leading=16,
        textColor=COLOR_SECONDARY,
        alignment=1,
        spaceAfter=15
    )

    style_meta = ParagraphStyle(
        'DocMeta',
        fontName='Malgun',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#718096"),
        alignment=2, # Right
        spaceAfter=15
    )

    style_h1 = ParagraphStyle(
        'SectionH1',
        fontName='MalgunBold',
        fontSize=13,
        leading=18,
        textColor=COLOR_PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'SectionH2',
        fontName='MalgunBold',
        fontSize=10.5,
        leading=15,
        textColor=COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'DocBody',
        fontName='Malgun',
        fontSize=8.5,
        leading=13,
        textColor=COLOR_DARK,
        spaceAfter=5
    )

    style_body_bold = ParagraphStyle(
        'DocBodyBold',
        fontName='MalgunBold',
        fontSize=8.5,
        leading=13,
        textColor=COLOR_DARK,
        spaceAfter=5
    )

    style_code = ParagraphStyle(
        'DocCode',
        fontName='Malgun',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName='MalgunBold',
        fontSize=8,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName='Malgun',
        fontSize=7.5,
        leading=10.5,
        textColor=COLOR_DARK
    )

    style_table_cell_center = ParagraphStyle(
        'TableCellCenter',
        fontName='Malgun',
        fontSize=7.5,
        leading=10.5,
        textColor=COLOR_DARK,
        alignment=1
    )

    style_table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='MalgunBold',
        fontSize=7.5,
        leading=10.5,
        textColor=COLOR_PRIMARY,
        alignment=1
    )

    story = []

    # =========================================================================
    # 표지 및 문서 메타데이터
    # =========================================================================
    story.append(Paragraph("vidio-health-trainer", style_subtitle))
    story.append(Paragraph("바벨 스쿼트 AI 통합 진단 엔진 v9.0 상세 아키텍처 및 외부 AI 기술 자문 배경지식 명세서", style_title))
    story.append(Paragraph("Technical Specification & Baseline Briefing for External AI Advisors: Enhancing Clean F1 & 4-Defect EMR", style_subtitle))
    story.append(Paragraph("작성일자: 2026. 09. 27 | 시스템 버전: v9.0 Master Production | 프로젝트 공식 명칭: vidio-health-trainer", style_meta))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=12))

    # =========================================================================
    # 1. 문서 목적 및 자문 핵심 목표 (Executive Mission)
    # =========================================================================
    story.append(Paragraph("1. 문서의 목적 및 외부 AI 자문 핵심 목표 (Executive Mission)", style_h1))
    story.append(Paragraph(
        "본 명세서는 <b>vidio-health-trainer</b> 프로젝트의 바벨 스쿼트 AI 진단 시스템에 대해 외부 최신 LLM 및 멀티모달 AI 연구원(Advisor)에게 현재 시스템의 <b>전처리 파이프라인, 4대 서브모델(Layer 1), 다층 앙상블 및 동적 라우팅(Layer 2~4), 데이터 구조</b>를 완전히 공개하고, <b>[정상 운동자 분류 F1-Score (Clean F1)] 및 [4대 결함 동시 적중률 (4-Defect EMR)]</b>을 추가로 비약시키기 위한 최신 AI 기술 솔루션을 도출하기 위해 작성되었습니다.",
        style_body
    ))

    # 핵심 목표 박스 테이블
    goal_data = [
        [
            Paragraph("<b>핵심 최우선 과제</b>", style_table_cell_bold),
            Paragraph("<b>현재 v9.0 달성 수준 (AI-Hub 골든셋 N=2,605)</b>", style_table_cell_bold),
            Paragraph("<b>차세대 v10.0 목표치</b>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>4-Defect EMR (완전 일치율)</b><br/>4개 결함 여부를 동시 100% 적중", style_table_cell),
            Paragraph("<b>71.24%</b> (선택적 판정 적용 시)<br/>(100% 전수 커버리지 기준: 68.98% / 1,797건 적중)", style_table_cell_center),
            Paragraph("<b>80.00% 이상</b> 돌파<br/>(전수 커버리지 75% 이상)", style_table_cell_center)
        ],
        [
            Paragraph("<b>Clean F1-Score (정상 판정)</b><br/>무결점 정상 운동자 종합 적중 F1", style_table_cell),
            Paragraph("<b>67.63%</b><br/>(Precision: 60.27%, Recall: <b>77.05%</b>)", style_table_cell_center),
            Paragraph("<b>75.00% 이상</b><br/>(Precision 70% 이상)", style_table_cell_center)
        ],
        [
            Paragraph("<b>RETAKE (재촬영 권고율)</b><br/>판정 불확실/노이즈 시 안전 보류", style_table_cell),
            Paragraph("<b>3.19%</b> (83건 / 2,605건)<br/>(데드밴드 [0.32, 0.38], 마진 Δ < 0.05)", style_table_cell_center),
            Paragraph("<b>4.00% 이하</b> 유지<br/>(사용자 경험 저해 최소화)", style_table_cell_center)
        ]
    ]
    t_goal = Table(goal_data, colWidths=[160, 200, 160])
    t_goal.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_HIGHLIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_goal)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 2. 데이터셋 규격 및 전처리 표준 파이프라인
    # =========================================================================
    story.append(Paragraph("2. 데이터셋 규격 및 전처리 표준 파이프라인 (Data Pipeline Invariants)", style_h1))
    story.append(Paragraph(
        "본 프로젝트는 인공지능 학습 및 평가 시 데이터 누수와 시점 편향(View Bias)을 원천 차단하기 위해 엄격한 불변 원칙을 준수합니다.",
        style_body
    ))

    data_pipeline_text = """
    <b>1) 데이터셋 및 피험자 격리 원칙 (Subject Isolation)</b>:<br/>
    • <b>AI-Hub 헬스 피트니스 바벨 스쿼트 골든셋 전수 2,605건</b> 사용.<br/>
    • 피험자 1번부터 9번까지 9명 전수를 학습 및 검증에 반영하며, 평가 시 <b>StratifiedGroupKFold (n_splits=5, groups=subject_id)</b>를 적용하여 학습 세트와 검증 세트 간 피험자 데이터가 절대 섞이지 않도록 완전 격리.<br/><br/>
    <b>2) 공식 5대 카메라 시점 체계 (Official 5-View Taxonomy)</b>:<br/>
    • <b>시점 A (우측 후방 45도 / 내부 코드 view5)</b>: 521건 (고개 및 안면 랜드마크 비가시 영역 발생)<br/>
    • <b>시점 B (우측 전방 45도 / 내부 코드 view2)</b>: 521건 (전신 3차원 깊이 정보 양호)<br/>
    • <b>시점 C (정면 0도 / 내부 코드 view3)</b>: 521건 (좌우 대칭성 및 관상면 무릎 외반 최적 관측)<br/>
    • <b>시점 D (좌측 전방 45도 / 내부 코드 view4)</b>: 521건 (전신 3차원 깊이 정보 양호)<br/>
    • <b>시점 E (좌측 후방 45도 / 내부 코드 view1)</b>: 521건 (고개 및 안면 랜드마크 비가시 영역 발생)<br/>
    ※ 5대 시점 전수 균등 포함 (각 20% 점유)하여 특정 앵글에 대한 과적합을 방지함.<br/><br/>
    <b>3) 64프레임 3차 스플라인 표준 전처리 (64-Frame Cubic Spline Normalization)</b>:<br/>
    • <b>입력 영상 파이프라인</b>: 원본 연속 영상 -> 단일 횟수(1 Repetition) 시작-최하사점-종료 구간 엄밀 분할 -> <b>64프레임으로 3차 스플라인(Cubic Spline) 보간</b>.<br/>
    • <b>기본 입력 텐서 규격</b>: <b>shape: (64, 33, 3)</b> (MediaPipe Pose 33개 랜드마크의 3D 공간 좌표 X, Y, Z).<br/>
    • <b>보조 텐서 규격 (c1 척추 모델 전용)</b>: <b>shape: (64, 9, 3)</b> (SpinePose 9대 분절: 경추 3점, 흉추 3점, 요추 3점).
    """
    story.append(Paragraph(data_pipeline_text, style_body))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. 4대 독립 서브모델 최신 버전 명세 (Layer 1 Submodels)
    # =========================================================================
    story.append(Paragraph("3. 4대 최신 챔피언 세부 결함 모델 세부 명세 (Layer 1)", style_h1))
    story.append(Paragraph(
        "하위 레이어(Layer 1)는 바벨 스쿼트 4대 필수 체크포인트를 전담하는 독립 모델로 구성되어 있으며, 각각 최고 성능 버전으로 고도화되어 있습니다.",
        style_body
    ))

    # 4대 모델 테이블
    sub_table_data = [
        [
            Paragraph("<b>결함 코드</b>", style_table_header),
            Paragraph("<b>공식 명칭</b>", style_table_header),
            Paragraph("<b>최신 버전</b>", style_table_header),
            Paragraph("<b>피처 및 모델 아키텍처</b>", style_table_header),
            Paragraph("<b>현재 F1 / Acc / AUC</b>", style_table_header)
        ],
        [
            Paragraph("<b>c1</b>", style_table_cell_center),
            Paragraph("<b>척추 중립 위반</b><br/>(요추 과전만/후만)", style_table_cell),
            Paragraph("v11.1 Tri-Ensemble", style_table_cell_center),
            Paragraph("• 38D LightGBM (체간 기저계 투영)<br/>• 27D DietGBM (경량 도메인 피처)<br/>• 24D Dilated TCN (시계열 텐서)<br/>※ 가중치: 0.10:0.40:0.50 결합", style_table_cell),
            Paragraph("<b>F1: 95.76%</b><br/>Acc: 95.89%<br/>AUC: 0.9882", style_table_cell_center)
        ],
        [
            Paragraph("<b>c2</b>", style_table_cell_center),
            Paragraph("<b>시선 및 고개 정면</b><br/>(경추 과신전/굴곡)", style_table_cell),
            Paragraph("v7.1 시점적응형", style_table_cell_center),
            Paragraph("• 19D CatBoost + LightGBM<br/>• 코-귀 관통 벡터, 경추 각도<br/>• <b>후방(시점 A, E) 시선 사각지대 판정 보류/면제 규칙 내장</b>", style_table_cell),
            Paragraph("<b>F1: 83.43%</b><br/>Acc: 82.76%<br/>AUC: 0.9124", style_table_cell_center)
        ],
        [
            Paragraph("<b>c3</b>", style_table_cell_center),
            Paragraph("<b>무릎-발 방향 일치</b><br/>(무릎 외반/내측붕괴)", style_table_cell),
            Paragraph("v6.0 3D 관상면", style_table_cell_center),
            Paragraph("• 22D CatBoost<br/>• 최하사점 절대 무릎/발목 비율(KA Ratio)<br/>• 3D 대퇴-경골 관상각, 비대칭 외반 피크", style_table_cell),
            Paragraph("<b>F1: 94.04%</b><br/>Acc: 94.17%<br/>AUC: 0.9781", style_table_cell_center)
        ],
        [
            Paragraph("<b>c4</b>", style_table_cell_center),
            Paragraph("<b>발바닥 완전 접지</b><br/>(뒤꿈치/앞꿈치 들림)", style_table_cell),
            Paragraph("v5.1 동역학 위상", style_table_cell_center),
            Paragraph("• 25D CatBoost + Dilated TCN<br/>• 앞꿈치/뒤꿈치 Y변위, 착지 면적<br/>• 떨림(Wobble), 저크(Jerk), FFT 0차(F0 Rate)", style_table_cell),
            Paragraph("<b>F1: 87.66%</b><br/>Acc: 87.72%<br/>AUC: 0.9415", style_table_cell_center)
        ]
    ]
    t_sub = Table(sub_table_data, colWidths=[40, 95, 80, 205, 100])
    t_sub.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_sub)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 4. 상위 통합 진단 마스터 엔진 아키텍처 (Layer 1.5 ~ Layer 4)
    # =========================================================================
    story.append(Paragraph("4. 상위 통합 진단 마스터 엔진 v9.0 아키텍처 (Layer 1.5 ~ Layer 4)", style_h1))
    story.append(Paragraph(
        "v9.0 마스터 엔진은 단순 Argmax 결합의 한계를 극복하고 4대 결함 간 상호작용 및 시점별 왜곡을 해결하기 위해 구축된 4계층 파이프라인입니다.",
        style_body
    ))

    arch_layers_text = """
    <b>■ Layer 1.5: 5대 시점 적응형 온도 스케일링 (Temperature Scaling)</b><br/>
    • 목적: 트리 기반 모델 및 신경망의 출력 확률 과확신(Over-confidence) 및 불확실성 왜곡 교정.<br/>
    • 수식: p_cal = 1 / (1 + exp(-logit / T))<br/>
    • 적용 파라미터: T_c1 = 0.456, T_c3 = 1.112, T_c4 = 1.001<br/>
    • c2 모델 시점별 적응형 온도: 시점 A(0.893), 시점 B(0.898), 시점 C(0.963), 시점 D(0.890), 시점 E(0.903)<br/><br/>

    <b>■ Layer 3: 도메인 보정 피처 추출기 (해결책 4 & 해결책 5)</b><br/>
    • <b>[해결책 4] 12D 관측성 품질 게이트 (Quality Gate)</b>:<br/>
      - 체간 길이 상대표준편차(std/mean), 대퇴골 상대표준편차, 경골 상대표준편차 (신체 분절 뼈대 길이 안정성)<br/>
      - 관절 중심 떨림(Wobble), 급격한 가속도 변화(Jerk), 3D Z축 깊이 분산, 관절 가려짐(Occlusion) 비율 등 12개 지표.<br/>
      - 종합 품질 점수 Q_overall 산출 및 이상 샘플 필터링.<br/>
    • <b>[해결책 5] 9D 3단계 위상 운동사슬 피처 (Kinetic Chain Interaction)</b>:<br/>
      - 스쿼트 전 구간을 하강기(Eccentric), 최하사점(Nadir), 상승기(Concentric)의 3위상으로 분할.<br/>
      - 위상별 [발바닥-무릎], [발바닥-체간], [무릎-체간] 분절 간 각속도 상관계수 및 위상차 9차원 벡터.<br/>
      - 발목의 흔들림이 무릎 외반과 요추 굴곡으로 이어지는 보상 작용(Compensatory movement) 포착.<br/><br/>

    <b>■ Layer 2: 39D View-Conditioned Mixture-of-Experts (MoE) 동적 라우팅 (해결책 6)</b><br/>
    • <b>입력 피처 39D 결합 벡터</b>:<br/>
      - [0~4] 5대 시점 One-Hot 인코딩 (5D)<br/>
      - [5~8] Layer 1.5 교정 결함 확률 [p_c1, p_c2, p_c3, p_c4] (4D)<br/>
      - [9~12] 모델별 확신도 마진 및 엔트로피 (4D)<br/>
      - [13~17] c4 발동역학 핵심 피처 (Wobble, Y-Delta, Toe-Lift, Jerk, F0-Rate) (5D)<br/>
      - [18~29] Layer 3 12D 관측성 품질 게이트 피처 (12D)<br/>
      - [30~38] Layer 3 9D 3단계 위상 운동사슬 피처 (9D)<br/>
    • <b>3대 전문가(Sub-Experts) 구조</b>:<br/>
      - <b>Expert 1 (Front/Oblique Specialist, Linear(39->48->16))</b>: 정면 및 45도 전방 시점 판정 특화<br/>
      - <b>Expert 2 (Rear Specialist, Linear(39->48->16))</b>: 후방 45도 시점 및 시선 사각지대 판정 특화<br/>
      - <b>Expert 3 (Kinetic Chain Specialist, Linear(39->48->16))</b>: 발-무릎-체간 다관절 위상 협응 특화<br/>
    • <b>Gating Network</b>: 39D -> 48 -> 3 (Softmax 출력 g1, g2, g3)<br/>
    • <b>출력</b>: 16-상태 결합 확률 분포 P_16 = g1*E1(x) + g2*E2(x) + g3*E3(x) (16-Way: 2^4 결함 조합)<br/>
    • <b>로드 밸런싱 손실 (Load-Balancing Loss)</b>: L_balance = 3 * sum(P_i * f_i) 적용으로 게이트 붕괴 원천 차단 (실제 부하 분산: 30.5% : 34.0% : 35.5% 달성)<br/><br/>

    <b>■ Layer 4: MoE 16-상태 결합 비대칭 선택적 판정 (Quality-Gated Selective Prediction)</b><br/>
    • 정상 확률 정의: P(Clean) = P(State 0) = P((0, 0, 0, 0))<br/>
    • 데드밴드 판정: P(Clean) in [0.32, 0.38] 또는 Top-1과 Top-2 결합 상태 간 마진 ΔP < 0.05인 경우, 억지 추론을 배제하고 안전하게 재촬영 권고(RETAKE)로 분류.<br/>
    • 결과: EMR 71.24% 달성 및 Clean Recall 77.05% 확보, RETAKE율 3.19%로 극소화.
    """
    story.append(Paragraph(arch_layers_text, style_body))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 5. 진화 궤적 및 5대 시점별 세부 감사
    # =========================================================================
    story.append(Paragraph("5. 7단계 마일스톤 진화 궤적 및 5대 시점별 감사 결과", style_h1))

    # 마일스톤 테이블
    m_data = [
        [
            Paragraph("<b>버전</b>", style_table_header),
            Paragraph("<b>핵심 도입 기술</b>", style_table_header),
            Paragraph("<b>4-Defect EMR</b>", style_table_header),
            Paragraph("<b>Clean F1</b>", style_table_header),
            Paragraph("<b>Clean Recall</b>", style_table_header),
            Paragraph("<b>RETAKE율</b>", style_table_header)
        ],
        [
            Paragraph("v6.0", style_table_cell_center),
            Paragraph("단순 독립 4모델 Argmax 결합", style_table_cell),
            Paragraph("15.90%", style_table_cell_center),
            Paragraph("34.43%", style_table_cell_center),
            Paragraph("45.00%", style_table_cell_center),
            Paragraph("0.00%", style_table_cell_center)
        ],
        [
            Paragraph("v8.0", style_table_cell_center),
            Paragraph("온도 보정 + 후방 적응형 휴리스틱", style_table_cell),
            Paragraph("48.02%", style_table_cell_center),
            Paragraph("55.40%", style_table_cell_center),
            Paragraph("58.12%", style_table_cell_center),
            Paragraph("0.00%", style_table_cell_center)
        ],
        [
            Paragraph("v8.2", style_table_cell_center),
            Paragraph("16-상태 결합 신경망 (18D Joint Meta)", style_table_cell),
            Paragraph("52.17%", style_table_cell_center),
            Paragraph("58.73%", style_table_cell_center),
            Paragraph("62.45%", style_table_cell_center),
            Paragraph("0.00%", style_table_cell_center)
        ],
        [
            Paragraph("v8.3", style_table_cell_center),
            Paragraph("비대칭 비용 데드밴드 선택적 판정", style_table_cell),
            Paragraph("56.40%", style_table_cell_center),
            Paragraph("62.45%", style_table_cell_center),
            Paragraph("68.20%", style_table_cell_center),
            Paragraph("4.85%", style_table_cell_center)
        ],
        [
            Paragraph("v8.4", style_table_cell_center),
            Paragraph("12D 관측성 품질 게이트 (해결책 4)", style_table_cell),
            Paragraph("59.94%", style_table_cell_center),
            Paragraph("65.13%", style_table_cell_center),
            Paragraph("72.04%", style_table_cell_center),
            Paragraph("3.80%", style_table_cell_center)
        ],
        [
            Paragraph("v8.5", style_table_cell_center),
            Paragraph("3단계 위상 운동사슬 39D (해결책 5)", style_table_cell),
            Paragraph("66.08%", style_table_cell_center),
            Paragraph("66.43%", style_table_cell_center),
            Paragraph("74.80%", style_table_cell_center),
            Paragraph("3.55%", style_table_cell_center)
        ],
        [
            Paragraph("<b>v9.0 Master</b>", style_table_cell_bold),
            Paragraph("<b>39D View-Conditioned MoE 동적 라우팅</b>", style_table_cell),
            Paragraph("<b>71.24%</b><br/>(전수 68.98%)", style_table_cell_bold),
            Paragraph("<b>67.63%</b>", style_table_cell_bold),
            Paragraph("<b>77.05%</b>", style_table_cell_bold),
            Paragraph("<b>3.19%</b>", style_table_cell_bold)
        ]
    ]
    t_m = Table(m_data, colWidths=[65, 195, 75, 60, 65, 60])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('BACKGROUND', (0,7), (-1,7), COLOR_HIGHLIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_m)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 6. 현재 남아있는 기술적 병목 및 문제점 분석
    # =========================================================================
    story.append(Paragraph("6. 현재 남아있는 기술적 병목 및 문제점 분석 (Current Bottlenecks)", style_h1))
    bottlenecks_text = """
    <b>1) 정상 판정의 정밀도(Precision 60.27%) 대비 재현율(Recall 77.05%) 불균형</b>:<br/>
    • 재현율은 77%에 도달했으나, 1개 이상의 미세 결함이 있는 샘플을 '정상'으로 오분류하는 위양성(False Positive)이 여전히 존재하여 Precision이 60%에 머무름.<br/>
    • 원인: 데이터셋 내 '전체 무결점 정상' 샘플의 희소성(약 10~15%) 및 4개 부위 중 1개 부위만 미세하게 결함인 경계 샘플의 모호성.<br/><br/>

    <b>2) c2 (시선 및 고개) 모델의 성능 한계 (F1 83.43%)</b>:<br/>
    • c1(95.76%), c3(94.04%), c4(87.66%)에 비해 c2가 가장 낮은 F1(83.43%)을 기록하며 전체 4-Defect EMR의 주요 발목을 잡고 있음.<br/>
    • 후방 45도(시점 A, E)에서 얼굴 전면이 가려지므로 척추-경추 각도만으로 추정해야 하는 본질적 가시성 한계 존재.<br/><br/>

    <b>3) 16-상태 결합 공간에서의 희귀 복합 결합(Rare Multi-Defect) 불균형</b>:<br/>
    • 16개 결합 상태 중 단일 결함이나 2개 결함은 샘플이 충분하나, 3개 결함 동시 발생((1,1,1,0) 등)이나 4개 결함 전수 발생((1,1,1,1)) 등 희귀 상태는 학습 표본이 극소수임.<br/><br/>

    <b>4) 뼈대 길이 고정(Skeletal Length Invariance) 전처리의 부재</b>:<br/>
    • MediaPipe 3D 좌표는 카메라 원근감과 피험자 체형에 따라 프레임마다 뼈대 길이(대퇴골, 경골, 체간)가 미세하게 변동함.<br/>
    • 현재는 이를 Layer 3에서 12D 상대표준편차로 검출만 하고 있을 뿐, 좌표 자체를 고정된 스켈레톤 길이로 리타겟팅(Normalization)하지는 않고 있음.
    """
    story.append(Paragraph(bottlenecks_text, style_body))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 7. 외부 AI 자문 요청 질문서 (Specific Technical Inquiries)
    # =========================================================================
    story.append(Paragraph("7. 외부 AI 자문 요청 질문서 (Specific Inquiries for External AI)", style_h1))
    story.append(Paragraph(
        "외부 AI 연구원께서는 위의 배경과 기술 명세를 바탕으로, 다음 6대 질문에 대해 <b>[구체적인 수학적 원리, 네트워크 아키텍처, 손실 함수 수식, 파이썬 구현 가이드]</b>를 포함하여 실질적인 해결책을 제안해 주십시오.",
        style_body_bold
    ))

    inquiries_text = """
    <b>[질문 1: Clean Precision 70%+ 돌파를 위한 손실 함수 및 분류 전략]</b><br/>
    • 정상(Clean) 클래스의 정밀도(Precision)를 현재 60.27%에서 70% 이상으로 끌어올리고 FP를 억제하기 위한 최신 비용 민감 학습(Cost-Sensitive / Asymmetric Focal Loss) 또는 정상 상태 중심 Contrastive Learning(대조 학습) 설계 방안은 무엇인가?<br/><br/>

    <b>[질문 2: c2 시선/고개 모델의 후방 뷰 극복을 위한 경추-흉추 운동학 연동]</b><br/>
    • 후방 45도 시점에서 얼굴 특징점 결측을 보상하기 위해, 흉추(Thoracic Spine)와 견갑대(Scapula)의 3D 회전 및 각속도로부터 경추(Cervical) 각도를 간접 추정하는 생체역학적 커플링(Biomechanical Coupling) 모델링 기법은 무엇인가?<br/><br/>

    <b>[질문 3: 4-Defect EMR 80%+ 달성을 위한 16-상태 결합 모델 고도화]</b><br/>
    • 현재의 MoE(3개 전문가) 구조를 넘어, 4대 결함 간의 조건부 의존성(Conditional Dependency: 예, c4 발목 불안정이 c3 무릎 외반을 유발하고 c1 요추 보상으로 이어지는 인과 그래프)을 직접 모델링하는 <b>Graph Neural Network (GNN) 또는 Directed Acyclic Graph (DAG) 베이지안 신경망</b> 적용 방안은?<br/><br/>

    <b>[질문 4: 뼈대 길이 고정(Skeletal Length Normalization) 알고리즘]</b><br/>
    • 64프레임 (64, 33, 3) 3D 좌표에 대해, 피험자 기립 시의 안정된 세그먼트 길이를 기준으로 프레임별 관절 방향 벡터는 유지하면서 뼈 길이를 완벽히 일정하게 고정하는 스켈레톤 정규화(Kinematic Retargeting) 수식 및 파이프라인 삽입 위치(서브모델 입력 전 vs 통합 피처 추출 전)는?<br/><br/>

    <b>[질문 5: 시공간 통합 신경망 (Spatio-Temporal Transformer / ST-GCN)]</b><br/>
    • 현재 수작업 피처(Hand-crafted 39D) + MoE 방식과 원천 시퀀스 (64, 33, 3)를 직접 입력받는 Spatial-Temporal Graph Convolutional Network (ST-GCN) 또는 Pose-Transformer의 하이브리드 결합 가능성 및 최적 융합(Late Fusion vs Intermediate Feature Fusion) 전략은?<br/><br/>

    <b>[질문 6: 4대 세부 모델의 신뢰도 보정(Calibration) 고도화]</b><br/>
    • 현재 스칼라 온도 스케일링(Temperature Scaling)을 넘어, 피처 벡터(39D)의 공간적 위치에 따라 온도가 국소적으로 변하는 가우시안 프로세스 온도 스케일링(GP-Temperature) 또는 매트릭스 스케일링(Matrix Scaling) 도입이 판정 불확실성 감소에 기여할 수 있는가?
    """
    story.append(Paragraph(inquiries_text, style_body))

    # 문서 빌드
    doc.build(story)
    print(f"PDF build successful: {target_pdf_path}")

if __name__ == "__main__":
    OUT_PDF_1 = r"D:\자주프 문서\vidio-health-trainer_외부AI_자문용_바벨스쿼트_통합진단_배경지식_및_아키텍처_명세서.pdf"
    OUT_PDF_2 = r"D:\자주프 문서\5주차\vidio-health-trainer_외부AI_자문용_바벨스쿼트_통합진단_배경지식_및_아키텍처_명세서.pdf"

    build_pdf(OUT_PDF_1)
    os.makedirs(os.path.dirname(OUT_PDF_2), exist_ok=True)
    shutil.copyfile(OUT_PDF_1, OUT_PDF_2)
    print(f"Copied to: {OUT_PDF_2}")
