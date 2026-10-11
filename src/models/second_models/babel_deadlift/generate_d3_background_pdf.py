# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d3_background_pdf.py
Purpose: 외부 AI 자문용 'd3 배경지식.pdf' 및 'd3 배경지식.docx' 생성 스크립트
         - 외부 AI가 우리 프로젝트를 전혀 모른다는 전제 하에 작성된 완벽한 기술 명세서
         - 데이터셋 구조, 64프레임 텐서 규격, 26D 피처셋 수식, ST-SeqNet 딥러닝 구조,
           3-State 판정 로직, 실측 성능 및 핵심 병목 분석, 외부 AI 핵심 자문 4대 질문 수록
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
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 등록
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"
pdfmetrics.registerFont(TTFont("MalgunGothic", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", FONT_BOLD_PATH))

# 출력 파일 경로 (사용자 명시 경로)
TARGET_DIR_ROOT = r"D:\자주프 문서"
TARGET_DIR_WEEK6 = r"D:\자주프 문서\6주차"
os.makedirs(TARGET_DIR_ROOT, exist_ok=True)
os.makedirs(TARGET_DIR_WEEK6, exist_ok=True)

PDF_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "d3 배경지식.pdf")
DOCX_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "d3 배경지식.docx")
MD_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "d3 배경지식.md")


class NumberedCanvas(canvas.Canvas):
    """2페이지 이상 문서에서 하단에 '페이지 X / Y' 및 헤더 출력"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("MalgunGothic", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # 헤더 (2페이지부터)
        if self._pageNumber > 1:
            self.drawString(36, A4[1] - 30, "vidio-health-trainer | 바벨 데드리프트 d3 모델 고도화 기술 명세서 (외부 AI 자문용)")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(36, A4[1] - 34, A4[0] - 36, A4[1] - 34)
            
        # 푸터
        footer_text = f"페이지 {self._pageNumber} / {page_count}"
        self.drawRightString(A4[0] - 36, 25, footer_text)
        self.drawString(36, 25, "Confidential - For AI Consulting & Architecture Optimization")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 35, A4[0] - 36, 35)
        self.restoreState()


def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    NAVY = colors.HexColor("#1A365D")
    BLUE = colors.HexColor("#2B6CB0")
    DARK = colors.HexColor("#2D3748")
    GRAY = colors.HexColor("#718096")
    BG_LIGHT = colors.HexColor("#F7FAFC")
    BORDER = colors.HexColor("#E2E8F0")
    ALERT_BG = colors.HexColor("#FFF5F5")
    ALERT_BORDER = colors.HexColor("#FEB2B2")
    ACCENT_BG = colors.HexColor("#EBF8FF")

    title_style = ParagraphStyle("DocTitle", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=20, leading=26, textColor=NAVY, spaceAfter=4)
    subtitle_style = ParagraphStyle("DocSubTitle", parent=styles['Normal'], fontName="MalgunGothic", fontSize=10, leading=14, textColor=GRAY, spaceAfter=14)
    h1_style = ParagraphStyle("H1", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=13, leading=17, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=True)
    h2_style = ParagraphStyle("H2", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=10.5, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=4, keepWithNext=True)
    body_style = ParagraphStyle("Body", parent=styles['Normal'], fontName="MalgunGothic", fontSize=8.5, leading=12, textColor=DARK, spaceAfter=4)
    body_bold = ParagraphStyle("BodyBold", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=8.5, leading=12, textColor=DARK)
    bullet_style = ParagraphStyle("Bullet", parent=styles['Normal'], fontName="MalgunGothic", fontSize=8.5, leading=12, textColor=DARK, leftIndent=12, firstLineIndent=-8, spaceAfter=2)
    code_style = ParagraphStyle("Code", parent=styles['Normal'], fontName="Courier", fontSize=7.5, leading=10, textColor=colors.HexColor("#2C5282"))
    th_style = ParagraphStyle("TH", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=8, leading=10, textColor=colors.white, alignment=1)
    td_style = ParagraphStyle("TD", parent=styles['Normal'], fontName="MalgunGothic", fontSize=7.5, leading=9.5, textColor=DARK)
    td_bold = ParagraphStyle("TDBold", parent=styles['Normal'], fontName="MalgunGothic-Bold", fontSize=7.5, leading=9.5, textColor=DARK)
    td_center = ParagraphStyle("TDCenter", parent=styles['Normal'], fontName="MalgunGothic", fontSize=7.5, leading=9.5, textColor=DARK, alignment=1)

    story = []

    # 1. 헤더 타이틀 블록
    story.append(Paragraph("vidio-health-trainer | Technical RFC Document", subtitle_style))
    story.append(Paragraph("바벨 데드리프트 d3 (발바닥 지면 고정) 모델 고도화<br/>외부 AI 자문용 기술 배경지식 명세서", title_style))
    story.append(Paragraph("AI 기반 스마트 피트니스 자세 분석 플랫폼 | 외부 최상위 AI(GPT-4o, Claude 3.5 Sonnet, Gemini Pro) 자문 요청용", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=10))

    # 요약 안내 박스
    summary_html = (
        "<b>■ 본 문서의 목적 (Executive Briefing)</b><br/>"
        "본 문서는 피트니스 모션 분석 프로젝트 <b>`vidio-health-trainer`</b>에 대한 사전 지식이 없는 외부 AI 모델에게 "
        "<b>바벨 데드리프트 세 번째 세부 결함(d3: 발바닥 지면 고정 / Foot Ground Fixation)</b>의 데이터셋 특성, "
        "현재 파이프라인(v5.2), 실측 벤치마크 성능 및 핵심 기술적 병목을 완전하게 공개하여, "
        "<b>'F1-Score를 80%~90%+ 수준으로 도약시키기 위한 최적의 모델 구조, 피처링 및 알고리즘 자문'</b>을 얻기 위해 작성되었습니다."
    )
    story.append(Table([[Paragraph(summary_html, body_style)]], colWidths=[523], style=[
        ('BACKGROUND', (0,0), (-1,-1), ACCENT_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#90CDF4")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 10))

    # 1. 외부 AI 핵심 자문 질문 4가지
    story.append(Paragraph("1. 외부 AI 핵심 자문 요청 항목 (Core RFC Questions)", h1_style))
    story.append(Paragraph("현재 5-Fold 피험자 완전 격리 환경에서 v5.2 모델은 2-State AUC 0.8487, 확정 판정 정확도 84.00%, Recall 79.17%, 확정 F1 67.06%를 기록 중입니다. 아래 4대 핵심 질문에 대한 구체적 해결 방안을 요청합니다:", body_style))
    
    rfc_q = [
        "<b>[자문 1. 45cm 바벨 원판 가림(Occlusion) 극복]</b>: 데드리프트 시 바벨 원판과 봉이 발목/뒤꿈치 1~2cm 앞을 지나며 발생하는 랜드마크 지터와 결측치를 완벽히 복원하거나 노이즈에 불변인 딥러닝/신호처리 구조는 무엇인가?",
        "<b>[자문 2. 시작 위상 이원화(바닥 셋업 vs 기립 락아웃) 정합]</b>: 전체 385건 중 56.9%는 바닥 셋업에서 시작하고, 43.1%는 서서 시작합니다. 고정 시간 슬라이싱이 불가능한 환경에서 두 그룹의 시간 위상을 단일 기준으로 완벽 정합하는 기법은?",
        "<b>[자문 3. 피험자 간 해부학적 체형 분산(Domain Shift) 해소]</b>: 피험자 1번의 정상 뒤꿈치 고도가 피험자 5번의 결함 뒤꿈치 고도보다 4배 높게 계측되는 신체 치수 분산을 완벽히 제거할 정규화 수학 공식 또는 도메인 적응(Domain Adaptation) 기법은?",
        "<b>[자문 4. 판단 보류율 10% 이하 제약 하에서 F1 80%~90%+ 달성]</b>: 판단 보류율을 최대 10% 이하(커버리지 90% 이상)로 엄격히 통제하면서 전체 F1-Score를 80% 이상으로 도약시킬 수 있는 손실 함수(Loss), 앙상블 블렌딩, 또는 그래프 신경망(GNN) 혁신안은?"
    ]
    for q in rfc_q:
        story.append(Paragraph(f"• {q}", bullet_style))
    story.append(Spacer(1, 10))

    # 2. 데이터셋 규격 및 전처리 파이프라인
    story.append(Paragraph("2. 데이터셋 규격 및 전처리 파이프라인 (Dataset Invariants)", h1_style))
    
    ds_data = [
        [Paragraph("항목", th_style), Paragraph("상세 기술 규격", th_style), Paragraph("공학적 설명 및 제약사항", th_style)],
        [Paragraph("원천 데이터", td_bold), Paragraph("AI-Hub 피트니스 자세 교정 3D 모션 데이터셋", td_style), Paragraph("정상/결함 레이블이 포함된 멀티뷰 3D 비디오", td_style)],
        [Paragraph("대상 운동", td_bold), Paragraph("02_바벨_데드리프트 (컨벤셔널 데드리프트)", td_style), Paragraph("바벨을 바닥 또는 랙에서 들어 올리는 복합 관절 운동", td_style)],
        [Paragraph("대상 결함 (d3)", td_bold), Paragraph("발바닥 지면 고정 (Foot Ground Fixation)", td_style), Paragraph("<b>100% 뒤꿈치 들림(Heel Lift) 결함</b> (앞꿈치 들림 부재)", td_style)],
        [Paragraph("피험자 구성", td_bold), Paragraph("총 9명 (피험자 1번 ~ 피험자 9번 전원 전수)", td_style), Paragraph("StratifiedGroupKFold 5-Fold 피험자 완전 격리 검증", td_style)],
        [Paragraph("멀티뷰 시점", td_bold), Paragraph("5대 전 시점 (A, B, C, D, E) 전수 학습 및 평가", td_style), Paragraph("시점 A(우측후방), B(우측전방), C(정면), D(좌측전방), E(좌측후방)", td_style)],
        [Paragraph("총 데이터 수", td_bold), Paragraph("385개 클립 (시점당 77개, 피험자당 약 40~50개)", td_style), Paragraph("정상 295건 (76.6%) vs 결함 90건 (23.4%) - 클래스 불균형", td_style)],
        [Paragraph("입력 텐서 규격", td_bold), Paragraph("<b>(64, 33, 3)</b> [Frames, Keypoints, Coordinates]", td_style), Paragraph("MediaPipe BlazePose 33개 3D 랜드마크, 64F 3차 스플라인 보간", td_style)],
        [Paragraph("데이터 누수 금지", td_bold), Paragraph("AI-Hub 3D JSON 좌표는 평가용 Ground Truth로만 사용", td_style), Paragraph("모델 입력 피처는 100% 비전(MediaPipe) 추출 데이터만 투입", td_style)]
    ]
    t_ds = Table(ds_data, colWidths=[90, 210, 223], style=[
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ])
    story.append(t_ds)
    story.append(Spacer(1, 10))

    # 3. 현재 모델 아키텍처 (v5.2)
    story.append(Paragraph("3. 현재 모델 아키텍처 및 파이프라인 (d3 v5.2)", h1_style))
    story.append(Paragraph(
        "d3 v5.2는 <b>[64프레임 51D 시공간 신경망(ST-SeqNet)]</b>과 <b>[26D 정예 생체역학 Tri-Ensemble]</b>을 융합한 하이브리드 아키텍처입니다:",
        body_style
    ))
    
    arch_html = (
        "<b>1. 시계열 딥러닝 스트림 (ST-SeqNet, 가중치 35%):</b><br/>"
        "   - 입력: 15개 핵심 관절(45D) + 바벨-발목 상대변위(3D) + 뒤꿈치-발가락 전방벡터(3D) = <b>(Batch, 64, 51)</b><br/>"
        "   - 구조: Multi-Scale 1D Conv (k=3, k=5) -> BatchNorm -> ReLU -> 2-Layer BiGRU (hidden 64) -> Temporal Attention Pooling -> Linear<br/>"
        "<b>2. 생체역학 트리 앙상블 스트림 (Tri-Ensemble, 가중치 65%):</b><br/>"
        "   - 입력: 피처 전수 감사를 통해 중복과 노이즈를 걷어낸 <b>26D 정예 생체역학 피처셋</b><br/>"
        "   - 모델: CatBoost (35%) + LightGBM (20%) + Random Forest (10%) 가중 결합 (Class Weight Balanced)<br/>"
        "<b>3. 시점 적응형 3-State 안전 게이트:</b><br/>"
        "   - 사후 결합 확률 $p = 0.35 p_{\\text{DL}} + 0.65 p_{\\text{Tree}}$ 에 대해 시점별 임계치 $[\\tau_{\\text{norm}}, \\tau_{\\text{def}}]$ 적용<br/>"
        "   - $p \\le \\tau_{\\text{norm}}$: NORMAL(0) | $p \\ge \\tau_{\\text{def}}$: DEFECT(1) | $\\tau_{\\text{norm}} < p < \\tau_{\\text{def}}$: RETAKE(2, 보류율 9.09%)"
    )
    story.append(Table([[Paragraph(arch_html, body_style)]], colWidths=[523], style=[
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 10))

    # 4. 26D 정예 피처셋 명세표
    story.append(Paragraph("4. 26D 정예 생체역학 피처 엔지니어링 명세표", h1_style))
    story.append(Paragraph("피처 전수 감사(Audit)를 통해 $r=1.0$ 완전 중복 4쌍과 발날 롤(Roll) 등 16개 노이즈 피처를 소거하고 남긴 핵심 정예 피처셋입니다:", body_style))

    feat_data = [
        [Paragraph("번호", th_style), Paragraph("피처 명칭", th_style), Paragraph("생체역학 계산 수식 / 정의", th_style), Paragraph("도입 목적 및 물리적 의미", th_style)],
        [Paragraph("01", td_center), Paragraph("d3_pitch_max_active_deg", td_bold), Paragraph("$\\max_t (\\theta_{\\text{pitch}}(t)) - \\text{base}$", td_style), Paragraph("리프팅 구간 최대 발 피치각 (트리 중요도 1위 9.42%)", td_style)],
        [Paragraph("02", td_center), Paragraph("d3_foot_pitch_nadir_deg", td_bold), Paragraph("최저 기립 슬라이스 내 평균 피치각", td_style), Paragraph("정적 셋업/기립 기준선 피치각", td_style)],
        [Paragraph("03", td_center), Paragraph("d3_foot_pitch_drop_deg", td_bold), Paragraph("$\\max(\\theta_{\\text{pitch}}) - \\min(\\theta_{\\text{pitch}})$", td_style), Paragraph("전체 구간 피치각 낙폭 및 가동 범위", td_style)],
        [Paragraph("04", td_center), Paragraph("d3_delta_wobble_active", td_bold), Paragraph("$\\text{std}(y_{\\text{toe}} - y_{\\text{heel}})$", td_style), Paragraph("발바닥 전후방 고도차 미세 흔들림 표준편차", td_style)],
        [Paragraph("05", td_center), Paragraph("d3_delta_max_active", td_bold), Paragraph("$\\max_t (y_{\\text{toe}}(t) - y_{\\text{heel}}(t))$", td_style), Paragraph("뒤꿈치가 발가락보다 위로 올라간 최대 변위", td_style)],
        [Paragraph("06", td_center), Paragraph("d3_heel_toe_ydelta_nadir_cm", td_bold), Paragraph("최저 기립 시 뒤꿈치-발가락 수직 변위", td_style), Paragraph("개인별 발바닥 신체 기준 높이", td_style)],
        [Paragraph("07", td_center), Paragraph("d3_delta_jerk_active", td_bold), Paragraph("$\\text{RMS}(\\frac{d^3 y_{\\text{heel}}}{dt^3})$", td_style), Paragraph("뒤꿈치 수직 가속도 저크(급격한 들림 충격도)", td_style)],
        [Paragraph("08", td_center), Paragraph("d3_toe_lift_max_cm", td_bold), Paragraph("$\\max_t (y_{\\text{toe}}(t) - \\text{base}_{\\text{toe}})$", td_style), Paragraph("앞꿈치 들림 억제 대조군 피처", td_style)],
        [Paragraph("09", td_center), Paragraph("d3_coupling_ratio_shank_pitch", td_bold), Paragraph("$\\text{corr}(\\theta_{\\text{shank}}, \\theta_{\\text{foot}})$", td_style), Paragraph("정강이 전경각과 발 피치각 간의 동적 결합비", td_style)],
        [Paragraph("10", td_center), Paragraph("d3_shank_foot_angle_drop_deg", td_bold), Paragraph("정강이-발 3D 로컬 상대 각도 낙폭", td_style), Paragraph("발목 관절의 순수 족배굴곡/저측굴곡 가동폭", td_style)],
        [Paragraph("11", td_center), Paragraph("d3_bar_ankle_min_dist_norm", td_bold), Paragraph("$\\min_t \\|\\mathbf{p}_{\\text{wrist}} - \\mathbf{p}_{\\text{ankle}}\\| / L_{\\text{shank}}$", td_style), Paragraph("바벨이 발목에 가장 근접하는 최소 거리비", td_style)],
        [Paragraph("12", td_center), Paragraph("d3_bar_moment_arm_max", td_bold), Paragraph("$\\max_t (\\mathbf{p}_{\\text{bar}} - \\mathbf{p}_{\\text{ank}}) \\cdot \\mathbf{u}_{\\text{foot}} / L_{\\text{shank}}$", td_style), Paragraph("<b>[신규]</b> 바벨의 발목 전방 최대 모멘트 암 거리", td_style)],
        [Paragraph("13", td_center), Paragraph("d3_phase_start_wrist_y", td_bold), Paragraph("$\\text{clip}((y_{\\text{wrist,start}} - 0.45)/0.35, 0, 1)$", td_style), Paragraph("<b>[신규]</b> 시작 위상 (바닥 셋업 1.0 vs 기립 0.0)", td_style)],
        [Paragraph("14~16", td_center), Paragraph("d3_f0_heel_view_visibility 등 3종", td_bold), Paragraph("시점별 랜드마크 가시성 및 기립 전이율", td_style), Paragraph("카메라 각도별 관절 신뢰도 보정", td_style)],
        [Paragraph("17~21", td_center), Paragraph("d3_view_A ~ d3_view_E", td_bold), Paragraph("공식 5대 시점 원핫 벡터 (5D)", td_style), Paragraph("시점 조건부(View Conditioning) MoE 게이팅", td_style)],
        [Paragraph("22~23", td_center), Paragraph("d3_shank_norm_heel_lift (max/mean)", td_bold), Paragraph("$\\Delta y_{\\text{heel}} / L_{\\text{shank}}$ (Nature Comm OpenCap)", td_style), Paragraph("<b>정강이 길이 무차원화 정규화</b> 동적 뒤꿈치 들림비", td_style)],
        [Paragraph("24~25", td_center), Paragraph("d3_wham_velocity (rms/asym)", td_bold), Paragraph("$\\|v_{\\text{heel}}\\|$ 및 $\\|v_{\\text{heel}} - v_{\\text{toe}}\\|$ (CVPR WHAM)", td_style), Paragraph("물리 영점 수직 속도(Zero Velocity) 접촉 위반도", td_style)],
        [Paragraph("26", td_center), Paragraph("d3_asym_heel_lift_shank_ratio", td_bold), Paragraph("$\\|\\Delta y_{\\text{heel,L}} - \\Delta y_{\\text{heel,R}}\\| / L_{\\text{shank}}$", td_style), Paragraph("좌우 뒤꿈치 들림 비대칭 비율", td_style)]
    ]
    t_feat = Table(feat_data, colWidths=[35, 145, 175, 168], style=[
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ])
    story.append(t_feat)
    story.append(Spacer(1, 10))

    # 5. 현재 실측 벤치마크 성능
    story.append(Paragraph("5. 현재 실측 벤치마크 성능 및 시점별 세부 지표 (v5.2)", h1_style))
    story.append(Paragraph("AI-Hub 피험자 1~9번 전원 전수, 공식 5대 시점 전수 5-Fold StratifiedGroupKFold 피험자 완전 격리 실측 결과입니다:", body_style))

    perf_data = [
        [Paragraph("공식 카메라 시점", th_style), Paragraph("시스템 코드", th_style), Paragraph("2-State AUC", th_style), Paragraph("판단 보류율", th_style), Paragraph("확정 F1-Score", th_style), Paragraph("확정 정확도", th_style), Paragraph("확정 재현율", th_style)],
        [Paragraph("시점 A (우측 후방 45°)", td_bold), Paragraph("view5", td_center), Paragraph("0.7787", td_center), Paragraph("9.09%", td_center), Paragraph("58.06%", td_center), Paragraph("81.43%", td_center), Paragraph("64.29%", td_center)],
        [Paragraph("시점 B (우측 전방 45°)", td_bold), Paragraph("view2", td_center), Paragraph("0.8456", td_center), Paragraph("9.09%", td_center), Paragraph("64.86%", td_center), Paragraph("81.43%", td_center), Paragraph("80.00%", td_center)],
        [Paragraph("시점 C (정면 0°)", td_bold), Paragraph("view3", td_center), Paragraph("0.8719", td_center), Paragraph("9.09%", td_center), Paragraph("73.33%", td_center), Paragraph("88.57%", td_center), Paragraph("78.57%", td_center)],
        [Paragraph("시점 D (좌측 전방 45°)", td_bold), Paragraph("view4", td_center), Paragraph("<b>0.9379</b>", td_center), Paragraph("9.09%", td_center), Paragraph("<b>82.35%</b>", td_center), Paragraph("<b>91.43%</b>", td_center), Paragraph("<b>93.33%</b>", td_center)],
        [Paragraph("시점 E (좌측 후방 45°)", td_bold), Paragraph("view1", td_center), Paragraph("0.8220", td_center), Paragraph("9.09%", td_center), Paragraph("57.89%", td_center), Paragraph("77.14%", td_center), Paragraph("78.57%", td_center)],
        [Paragraph("전체 시점 통합", th_style), Paragraph("Overall", th_style), Paragraph("<b>0.8487</b>", th_style), Paragraph("<b>9.09%</b>", th_style), Paragraph("<b>67.06%</b>", th_style), Paragraph("<b>84.00%</b>", th_style), Paragraph("<b>79.17%</b>", th_style)]
    ]
    t_perf = Table(perf_data, colWidths=[110, 55, 65, 60, 75, 75, 83], style=[
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#2C5282")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, BG_LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ])
    story.append(t_perf)
    story.append(Spacer(1, 10))

    # 6. 직면한 3대 병목 및 외부 AI에게 구하는 상세 자문
    story.append(Paragraph("6. 직면한 3대 기술 병목 및 외부 AI 심층 자문 요청", h1_style))
    
    bottleneck_html = (
        "<b>■ 현재 성능이 전체 F1 67.1%에서 80% 이상으로 도약하지 못하는 3대 원인:</b><br/>"
        "1. <b>원판 가림에 의한 후방 시점(시점 A, E)의 F1 저하 (57~58%)</b>: 전방 시점(시점 D F1 82.4%, AUC 0.938)에 비해 후방 시점은 바벨 원판이 발을 가려 지터가 심함.<br/>"
        "2. <b>시작 위상 불일치 (Phase Asynchrony)</b>: 바닥 셋업 시작(56.9%)과 기립 시작(43.1%)으로 인해 시간 축 합성 시 미세 위상 왜곡 잔존.<br/>"
        "3. <b>클래스 불균형 (정상 76.6% vs 결함 23.4%)</b>: 뒤꿈치 들림 변위가 0.5~1.5cm로 작아 Precision 확보가 까다로움.<br/><br/>"
        "<b>■ 외부 AI 자문 요청 가이드라인:</b><br/>"
        "위 병목을 해결하기 위해 <b>'Spatio-Temporal Graph CNN(ST-GCN/APSA)', 'PoseConv3D 3D 히트맵', '물리 제약 보조 손실(Physics-Informed Loss)', 'FastDTW 위상 정렬'</b> 등 구체적인 수식, 아키텍처 다이어그램 및 파이토치(PyTorch) 구현 코드 아이디어를 제안해 주십시오."
    )
    story.append(Table([[Paragraph(bottleneck_html, body_style)]], colWidths=[523], style=[
        ('BACKGROUND', (0,0), (-1,-1), ALERT_BG),
        ('BOX', (0,0), (-1,-1), 1, ALERT_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] PDF 생성 완료: {filename}")


def build_docx(filename):
    """Word docx 생성"""
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)

    p_sub = doc.add_paragraph()
    r = p_sub.add_run("vidio-health-trainer | Technical RFC Document\n")
    r.font.name = "맑은 고딕"; r.font.size = Pt(10.5); r.font.color.rgb = RGBColor(113, 128, 150); r.bold = True
    
    p_title = doc.add_paragraph()
    r2 = p_title.add_run("바벨 데드리프트 d3 (발바닥 지면 고정) 모델 고도화\n외부 AI 자문용 기술 배경지식 명세서")
    r2.font.name = "맑은 고딕"; r2.font.size = Pt(18); r2.font.color.rgb = RGBColor(26, 54, 93); r2.bold = True

    p_meta = doc.add_paragraph()
    p_meta.add_run("AI 기반 스마트 피트니스 자세 분석 플랫폼 | 외부 최상위 AI(GPT-4o, Claude 3.5, Gemini Pro) 자문 요청용")

    # 본문 및 섹션 추가
    doc.add_heading("1. 외부 AI 핵심 자문 요청 항목 (Core RFC Questions)", level=1)
    doc.add_paragraph("1. 45cm 바벨 원판 가림(Occlusion) 시 발목/뒤꿈치 랜드마크 복원 및 지터 극복 알고리즘\n"
                      "2. 시작 위상(바닥 셋업 56.9% vs 기립 43.1%)의 시간 불일치를 완벽히 정합하는 딥러닝/신호처리 기법\n"
                      "3. 피험자 간 해부학적 체형 분산(Domain Shift)을 원천 차단하는 정규화 수학 공식\n"
                      "4. 판단 보류율 10% 이하 제약 하에서 F1 80%~90%+ 달성을 위한 아키텍처 및 물리 제약 손실 함수")

    doc.add_heading("2. 데이터셋 규격 및 전처리 파이프라인 (Dataset Invariants)", level=1)
    doc.add_paragraph("- 64프레임 3차 스플라인 정규화, MediaPipe BlazePose 33개 3D 랜드마크\n"
                      "- 총 385개 클립 (피험자 1~9번 전원 전수, 공식 5대 시점 A~E 전원 전수)\n"
                      "- 100% 뒤꿈치 들림(Heel Lift) 결함 편향 (정상 295건 vs 결함 90건)")

    doc.add_heading("3. 현재 v5.2 아키텍처 및 26D 정예 피처 명세", level=1)
    doc.add_paragraph("- 64F 51D 시공간 신경망(ST-SeqNet) + 26D 정예 생체역학 Tri-Ensemble 하이브리드 결합\n"
                      "- 완전 중복 4쌍 및 롤(Roll) 등 16개 노이즈 피처 소거 완료\n"
                      "- 바벨-발목 모멘트 암 및 시작 위상 피처 결합")

    doc.add_heading("4. 현재 실측 벤치마크 성능 (5-Fold 피험자 완전 격리)", level=1)
    doc.add_paragraph("- 전체 2-State AUC: 0.8487 | 정확도: 80.52% | F1: 60.32% | Recall: 63.33%\n"
                      "- 전체 3-State 확정 판정 (보류율 9.09%): 정확도 84.00% | Recall 79.17% | F1 67.06%\n"
                      "- 시점 D (좌측 전방 45°): AUC 0.9379 (최고), 확정 F1 82.35%, 정확도 91.43%, Recall 93.33%\n"
                      "- 시점 C (정면 0°): AUC 0.8719, 확정 F1 73.33%, 정확도 88.57%\n"
                      "- 시점 A (우측 후방 45°): AUC 0.7787, 확정 F1 58.06%, 정확도 81.43%")

    doc.save(filename)
    print(f"[+] DOCX 생성 완료: {filename}")


def main():
    # 1. 루트 경로 생성
    build_pdf(PDF_OUTPUT_PATH)
    build_docx(DOCX_OUTPUT_PATH)
    
    # 2. 6주차 폴더로 복사 배포
    pdf_week6 = os.path.join(TARGET_DIR_WEEK6, "d3 배경지식.pdf")
    docx_week6 = os.path.join(TARGET_DIR_WEEK6, "d3 배경지식.docx")
    shutil.copy2(PDF_OUTPUT_PATH, pdf_week6)
    shutil.copy2(DOCX_OUTPUT_PATH, docx_week6)
    print(f"[+] 6주차 폴더 배포 완료: {pdf_week6}")

    # 3. 마크다운 동기화
    with open(MD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# vidio-health-trainer | 바벨 데드리프트 d3 기술 배경지식 명세서 (외부 AI 자문용)\n\n")
        f.write(f"> **공식 PDF 위치**: `{PDF_OUTPUT_PATH}` 및 `{pdf_week6}`  \n")
        f.write(f"> **프로젝트**: `vidio-health-trainer` | **타깃 결함**: d3 발바닥 지면 고정\n\n---\n\n")
        f.write("## ■ 외부 AI 자문 요청 핵심 4대 질문\n\n")
        f.write("1. 45cm 바벨 원판 가림(Occlusion) 시 발목/뒤꿈치 랜드마크 복원 및 지터 극복 알고리즘\n")
        f.write("2. 시작 위상(바닥 셋업 56.9% vs 기립 43.1%) 시간 불일치 완벽 정합 기법\n")
        f.write("3. 피험자 간 해부학적 체형 분산(Domain Shift) 원천 차단 정규화 수학 공식\n")
        f.write("4. 판단 보류율 10% 이하 제약 하에서 F1 80%~90%+ 달성을 위한 GNN 및 물리 제약 손실 함수\n\n")
        f.write("## ■ 현재 v5.2 실측 벤치마크 성능 요약\n\n")
        f.write("- 전체 2-State AUC: 0.8487, 정확도: 80.52%, F1: 60.32%, Recall: 63.33%\n")
        f.write("- 전체 3-State 확정 판정 (보류율 9.09%): 정확도 84.00%, Recall 79.17%, F1 67.06%\n")
        f.write("- 시점 D (좌측 전방 45°): AUC 0.9379 (역대 최고), 확정 F1 82.35%, 정확도 91.43%, Recall 93.33%\n")
    shutil.copy2(MD_OUTPUT_PATH, os.path.join(TARGET_DIR_WEEK6, "d3 배경지식.md"))
    print(f"[+] 마크다운 동기화 완료: {MD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
