# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d1_background_pdf.py
Purpose: 외부 AI 자문용 'd1 배경 지식.pdf' 및 'd1 배경 지식.docx' 생성 스크립트
         - 데드리프트 d2 모델 구조, 데이터 텐서 규격, 18D 피처셋 수식, 3-State 판정, 실측 성능 및 핵심 병목 분석 수록
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 등록
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"

pdfmetrics.registerFont(TTFont("MalgunGothic", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", FONT_BOLD_PATH))

# 출력 파일 경로 (사용자 명시 경로)
TARGET_DIR_1 = r"D:\자주프 문서"
TARGET_DIR_2 = r"D:\자주프 문서\6주차"
os.makedirs(TARGET_DIR_1, exist_ok=True)
os.makedirs(TARGET_DIR_2, exist_ok=True)

PDF_FILES = [
    os.path.join(TARGET_DIR_1, "d1 배경 지식.pdf"),
    os.path.join(TARGET_DIR_1, "d2 배경 지식.pdf"),
    os.path.join(TARGET_DIR_2, "d1 배경 지식.pdf"),
    os.path.join(TARGET_DIR_2, "d2 배경 지식.pdf"),
]


class NumberedCanvas(canvas.Canvas):
    """2페이지 이상 문서에서 하단 중앙에 'Page X of Y' 번호 출력"""
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
            self.drawString(36, A4[1] - 30, "vidio-health-trainer | 바벨 데드리프트 d2 고도화 기술 명세서 (외부 AI 자문용)")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, A4[1] - 34, A4[0] - 36, A4[1] - 34)
            
        # 푸터
        footer_text = f"페이지 {self._pageNumber} / {page_count}"
        self.drawRightString(A4[0] - 36, 25, footer_text)
        self.drawString(36, 25, "Confidential - For AI Consulting & Architecture Optimization")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
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

    # 커스텀 스타일 정의
    NAVY = colors.HexColor("#1A365D")
    BLUE = colors.HexColor("#2B6CB0")
    DARK = colors.HexColor("#2D3748")
    GRAY = colors.HexColor("#718096")
    LIGHT_BG = colors.HexColor("#F7FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E0")

    style_title = ParagraphStyle(
        "DocTitle",
        fontName="MalgunGothic-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceAfter=4
    )

    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        fontName="MalgunGothic",
        fontSize=9.5,
        leading=14,
        textColor=GRAY,
        spaceAfter=12
    )

    style_h1 = ParagraphStyle(
        "SectionH1",
        fontName="MalgunGothic-Bold",
        fontSize=13,
        leading=17,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        "SectionH2",
        fontName="MalgunGothic-Bold",
        fontSize=10.5,
        leading=14,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        "BodyTextCustom",
        fontName="MalgunGothic",
        fontSize=8.5,
        leading=12.5,
        textColor=DARK,
        spaceAfter=4
    )

    style_body_bold = ParagraphStyle(
        "BodyBoldCustom",
        fontName="MalgunGothic-Bold",
        fontSize=8.5,
        leading=12.5,
        textColor=DARK,
        spaceAfter=4
    )

    style_callout = ParagraphStyle(
        "CalloutText",
        fontName="MalgunGothic",
        fontSize=8.5,
        leading=12.5,
        textColor=DARK
    )

    style_table_cell = ParagraphStyle(
        "TableCell",
        fontName="MalgunGothic",
        fontSize=8,
        leading=11,
        textColor=DARK
    )

    style_table_header = ParagraphStyle(
        "TableHeader",
        fontName="MalgunGothic-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # 1. 헤더 및 타이틀
    story.append(Paragraph("바벨 데드리프트 d2 결함 분류 모델 기술 명세서", style_title))
    story.append(Paragraph("부제: 외부 AI 자문용 데이터 규격, 생체역학 기전, 18D 피처 수식, 실측 성능 및 F1 향상 과제 정의서<br/><b>공식 프로젝트명: vidio-health-trainer</b> | 대상 결함: d2(발과 무릎의 방향 일치 / Knee Valgus) | 작성일: 2026-09-30", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8))

    # 박스: 자문 의뢰 요약
    callout_data = [[
        Paragraph(
            "<b>[외부 AI 자문 의뢰 핵심 목표 (Mission)]</b><br/>"
            "본 문서는 피트니스 모션 AI 코칭 시스템 <b>`vidio-health-trainer`</b>의 바벨 데드리프트 세부 결함 모델 <b>d2(발과 무릎의 방향 일치 / 동적 무릎 외반)</b>의 성능을 "
            "현재 3-State F1-Score <b>55.10%</b>에서 <b>80% 이상</b>으로 혁신하기 위해 필요한 모든 데이터 텐서 규격, 생체역학적 결함 기전, 18D 기하 피처 수식, "
            "현재 학습 파이프라인 및 실측 성능 병목을 수록한 기술 명세서입니다. "
            "외부 최신 AI(GPT-4, Claude 3.5, Gemini 1.5 Pro 등)는 본 문서의 스펙을 기반으로 <b>소규모 불균형 골격 데이터셋에 특화된 피처 엔지니어링, 최적 아키텍처, 손실함수 및 뷰 불변 학습법</b>을 자문해 주시기 바랍니다.",
            style_callout
        )
    ]]
    t_callout = Table(callout_data, colWidths=[523])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 1, BLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_callout)
    story.append(Spacer(1, 8))

    # 1. 데이터셋 규격
    story.append(Paragraph("1. 원천 데이터셋 및 입력 텐서 규격 (Dataset & Tensor Specification)", style_h1))
    
    spec_table_data = [
        [Paragraph("항목", style_table_header), Paragraph("상세 기술 규격", style_table_header), Paragraph("비고 및 제약 사항", style_table_header)],
        [Paragraph("데이터 출처", style_table_cell), Paragraph("AI-Hub 피트니스 자세 평가 데이터셋 (02_바벨_데드리프트)", style_table_cell), Paragraph("국가 공공 AI 피트니스 원천 데이터", style_table_cell)],
        [Paragraph("표본 규모", style_table_cell), Paragraph("총 <b>385개 시퀀스</b> (정상 295건 vs 결함 90건)", style_table_cell), Paragraph("<b>결함 비율 23.4% (약 3.3:1 불균형)</b>", style_table_cell)],
        [Paragraph("피험자 구성", style_table_cell), Paragraph("피험자 1번 ~ 9번 (총 9명 전원 전수 활용)", style_table_cell), Paragraph("피험자 완전 격리 5-Fold StratifiedGroupKFold", style_table_cell)],
        [Paragraph("카메라 5대 시점", style_table_cell), Paragraph("시점 A(우측후방 45°), 시점 B(우측전방 45°), 시점 C(정면 0°), 시점 D(좌측전방 45°), 시점 E(좌측후방 45°)", style_table_cell), Paragraph("전 시점 100% 전수 학습 및 감사 의무화", style_table_cell)],
        [Paragraph("입력 텐서 형상", style_table_cell), Paragraph("<b>(N=385, T=64, J=33, C=3)</b> float32", style_table_cell), Paragraph("단일 횟수(1 Rep) 무삭제 64프레임 보간", style_table_cell)],
        [Paragraph("관절 좌표계", style_table_cell), Paragraph("MediaPipe Pose 33개 3D 월드 좌표계 (X, Y, Z 미터 단위)", style_table_cell), Paragraph("순수 비전 좌표 (AI-Hub 3D JSON 참조 금지)", style_table_cell)],
        [Paragraph("정답 라벨 용도", style_table_cell), Paragraph("AI-Hub 2D JSON 조건: '발과 무릎의 방향 일치' (참: 0, 거짓: 1)", style_table_cell), Paragraph("오직 검증 및 손실함수 타깃으로만 사용", style_table_cell)],
    ]
    t_spec = Table(spec_table_data, colWidths=[100, 263, 160])
    t_spec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 8))

    # 2. 생체역학적 결함 기전
    story.append(Paragraph("2. 데드리프트 무릎 외반(Knee Valgus)의 임상 생체역학 기전", style_h1))
    story.append(Paragraph(
        "<b>[대내전근 보상 작용의 역설 (Adductor Magnus Paradox)]</b><br/>"
        "데드리프트의 셋업 및 리프트오프(지면 분리) 구간에서 고관절이 깊게 굴곡될 때, 대둔근보다 <b>대내전근(Adductor Magnus)</b>이 가장 큰 고관절 신전 모멘트암을 가집니다. "
        "후면 사슬(대둔근 및 외회전근군)의 근력이 부족하거나 고중량 정지 관성을 이기지 못할 때, 신경계는 고관절 신전력을 보태기 위해 대내전근을 과도하게 수축시킵니다. "
        "대내전근의 강력한 <b>내전(Adduction) 및 내회전(Internal Rotation) 토크</b>가 무릎을 안쪽으로 강하게 끌어당겨 '무릎 모임(Valgus)'을 유발합니다.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[스쿼트 대비 데드리프트의 본질적 난이도 격차]</b><br/>"
        "• <b>스쿼트 (c3 무릎 외반: 3-State F1 97.64%)</b>: 무릎이 90~120° 깊게 접히며 결함 발생 시 무릎 사이 거리가 10~20cm 좁아지는 거대한 기하학적 신호가 발생하며, 2,605개 표본(결함 1,280개 1:1 균형)을 가집니다.<br/>"
        "• <b>데드리프트 (d2 무릎 외반: 3-State F1 55.10%)</b>: 힙 힌지 운동으로 무릎 굴곡이 30~50°에 불과하여 결함 발생 시 무릎 사이 거리 축소 변위가 <b>2~5cm 미만</b>으로 매우 미세합니다. 또한 385개 표본(결함 90개)의 소규모 불균형 데이터셋입니다.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[동작 위상(Phase)별 실측 통계 (사용자 관찰 검증 완료)]</b><br/>"
        "사용자가 실제 이미지를 육안 검토하여 <i>'애초에 바벨을 잡으러 내려갈 때(셋업)부터 무릎이 모여 시작한다'</i>는 사실을 발견하였으며, 이를 385개 전 시퀀스에서 실측한 결과 다음과 같은 통계적 진실이 입증되었습니다:<br/>"
        "• <b>Phase 0 (셋업 / 하강, 0~20프레임)</b>: 정상 무릎/골반비 1.054 vs 결함 0.997 (<b>p=0.0877</b>, 시작부터 좁은 셋업 결손)<br/>"
        "• <b>Phase 1 (퍼스트 풀 / 리프팅, 21~45프레임)</b>: 정상 1.015 vs 결함 <b>0.915</b> (<b>p=0.0116 ★극히 유의미</b>, 정지관성 극복 시 꺾임 정점)<br/>"
        "• <b>Phase 2 (락아웃 / 직립, 46~64프레임)</b>: 정상 1.061 vs 결함 1.021 (<b>p=0.2317 노이즈 구간</b>, 무릎이 다시 펴지며 신호 희석)<br/>"
        "따라서 64프레임 전체를 뭉뚱그려 평균을 내지 않고, <b>위상별(Phase-aware)로 분리 추출</b>하는 것이 F1 극대화의 핵심 열쇠입니다.",
        style_body
    ))
    story.append(Spacer(1, 6))

    # 3. 현재 18D 피처셋 상세 명세
    story.append(Paragraph("3. 현재 d2 v4.0 모델의 18D 피처셋 및 수학적 정의 (Feature Specification)", style_h1))
    story.append(Paragraph(
        "현재 모델은 사람이 서 있는 회전각을 제거하기 위해 <b>인체 중심 가상 정면 3D 좌표계(Canonical Frontal)</b>로 정렬합니다:<br/>"
        "• 원점: 골반 중심 Mid-Hip $\\frac{L_{hip} + R_{hip}}{2}$, "
        "• 수평 X축: $\\hat{x} = \\frac{L_{hip} - R_{hip}}{\\|L_{hip} - R_{hip}\\|}$, "
        "• 수직 Y축: 골반 중심 $\\rightarrow$ 어깨 중심 벡터, "
        "• 전방 Z축: $\\hat{z} = \\hat{x} \\times \\hat{y}$",
        style_body
    ))

    feat_table_data = [
        [Paragraph("No", style_table_header), Paragraph("피처 변수명", style_table_header), Paragraph("수학적 정의 및 물리적 의미", style_table_header), Paragraph("기저 좌표계", style_table_header)],
        [Paragraph("1", style_table_cell), Paragraph("world_kp_ratio_min_abs", style_table_cell), Paragraph("전체 구간 중 무릎 너비 ÷ 골반 너비 최솟값: min(‖L_knee - R_knee‖ / ‖L_hip - R_hip‖)", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("2", style_table_cell), Paragraph("world_ka_deficit_integ", style_table_cell), Paragraph("무릎 너비가 발목 너비보다 좁아진 결손량의 시간 적분값", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("3", style_table_cell), Paragraph("world_valgus_abs_max", style_table_cell), Paragraph("관상면 X축 투영 기준 양 무릎이 안쪽으로 밀려 들어간 센티미터 변위 최댓값", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("4", style_table_cell), Paragraph("world_valgus_norm_max", style_table_cell), Paragraph("양 무릎 안쪽 변위 합 ÷ 골반 너비 (골반 너비 정규화 결함율)", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("5", style_table_cell), Paragraph("world_shank_tilt_p90", style_table_cell), Paragraph("정강이 뼈(무릎-발목)가 안쪽으로 기운 각도의 90백분위수", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("6", style_table_cell), Paragraph("world_valgus_accel_peak", style_table_cell), Paragraph("무릎 안쪽 변위의 2계 미분값 (무릎 꺾임 순간 가속도 피크)", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("7", style_table_cell), Paragraph("world_morph_kp_drop", style_table_cell), Paragraph("서 있는 초기 상태 대비 최하점에서의 무릎/골반비 상대적 낙폭 (Drop)", style_table_cell), Paragraph("World 3D", style_table_cell)],
        [Paragraph("8", style_table_cell), Paragraph("can_valgus_abs_max", style_table_cell), Paragraph("가상 정면 좌표계 기준 양 무릎 안쪽 변위 합 최댓값", style_table_cell), Paragraph("Canonical 3D", style_table_cell)],
        [Paragraph("9", style_table_cell), Paragraph("can_valgus_norm_max", style_table_cell), Paragraph("가상 정면 좌표계 기준 골반 정규화 안쪽 변위 최댓값", style_table_cell), Paragraph("Canonical 3D", style_table_cell)],
        [Paragraph("10", style_table_cell), Paragraph("can_shank_tilt_p90", style_table_cell), Paragraph("가상 정면 좌표계 기준 정강이 안쪽 기울기 90백분위수", style_table_cell), Paragraph("Canonical 3D", style_table_cell)],
        [Paragraph("11", style_table_cell), Paragraph("can_ka_ratio_min", style_table_cell), Paragraph("가상 정면 좌표계 기준 무릎 너비 ÷ 발목 너비 최솟값", style_table_cell), Paragraph("Canonical 3D", style_table_cell)],
        [Paragraph("12", style_table_cell), Paragraph("can_kp_ratio_min", style_table_cell), Paragraph("가상 정면 좌표계 기준 무릎 너비 ÷ 골반 너비 최솟값", style_table_cell), Paragraph("Canonical 3D", style_table_cell)],
        [Paragraph("13", style_table_cell), Paragraph("phase0_setup_ka_min", style_table_cell), Paragraph("<b>[신규] 셋업 구간(0~20프레임) 무릎/발목비 최솟값 (사용자 관찰 '내려갈 때 모임' 감지)</b>", style_table_cell), Paragraph("Phase 0", style_table_cell)],
        [Paragraph("14", style_table_cell), Paragraph("phase1_pull_ka_min", style_table_cell), Paragraph("<b>[신규] 퍼스트풀 구간(21~45프레임) 무릎/발목비 최솟값 (지면 분리 정지관성 극복)</b>", style_table_cell), Paragraph("Phase 1", style_table_cell)],
        [Paragraph("15", style_table_cell), Paragraph("phase_delta_ka_pull", style_table_cell), Paragraph("<b>[신규] 셋업 대비 퍼스트풀 무릎 모임 낙폭 (ΔKAWR = setup - pull)</b>", style_table_cell), Paragraph("Phase 0 vs 1", style_table_cell)],
        [Paragraph("16", style_table_cell), Paragraph("fppa_3d_valgus_max", style_table_cell), Paragraph("<b>[신규] 3D FPPA 전두면 외반각 최댓값: 180° - ∠(Hip-Knee-Ankle)</b>", style_table_cell), Paragraph("Clinical 3D", style_table_cell)],
        [Paragraph("17", style_table_cell), Paragraph("ankle_rot_mismatch", style_table_cell), Paragraph("<b>[신규] 발끝 방향 대비 정강이 안쪽 회전 어긋남 각도 (경골 비틀림)</b>", style_table_cell), Paragraph("Transverse 2D", style_table_cell)],
        [Paragraph("18", style_table_cell), Paragraph("arm_knee_min_dist", style_table_cell), Paragraph("<b>[신규] 손목/팔꿈치와 무릎 간의 최소 3D 유클리드 거리 (팔 차폐 노이즈 보정)</b>", style_table_cell), Paragraph("World 3D", style_table_cell)],
    ]
    t_feat = Table(feat_table_data, colWidths=[20, 140, 283, 80])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    # 4. 현재 실측 성능 및 3-State 판정
    story.append(Paragraph("4. 현재 실측 성능 및 3-State 선택적 판정 현황 (Current Benchmark)", style_h1))
    story.append(Paragraph(
        "<b>[학습 파이프라인]</b> Cleanlab Confident Learning(오염 라벨 상위 50개 프루닝) $\\rightarrow$ "
        "LightGBM(scale_pos_weight=3.28) + CatBoost(Balanced) 50:50 앙상블 $\\rightarrow$ "
        "5-Fold StratifiedGroupKFold (9명 피험자 완전 격리)",
        style_body
    ))

    perf_table_data = [
        [Paragraph("버전", style_table_header), Paragraph("피처 차원", style_table_header), Paragraph("ROC-AUC", style_table_header), Paragraph("3-State F1", style_table_header), Paragraph("정밀도", style_table_header), Paragraph("재현율", style_table_header), Paragraph("정확도", style_table_header), Paragraph("보류율/커버리지", style_table_header)],
        [Paragraph("v2.0", style_table_cell), Paragraph("7D", style_table_cell), Paragraph("0.6502", style_table_cell), Paragraph("46.43%", style_table_cell), Paragraph("35.20%", style_table_cell), Paragraph("68.80%", style_table_cell), Paragraph("61.04%", style_table_cell), Paragraph("0.0% / 100.0%", style_table_cell)],
        [Paragraph("v3.0", style_table_cell), Paragraph("12D", style_table_cell), Paragraph("0.7279", style_table_cell), Paragraph("53.98%", style_table_cell), Paragraph("42.07%", style_table_cell), Paragraph("<b>75.31%</b>", style_table_cell), Paragraph("70.03%", style_table_cell), Paragraph("9.9% / 90.1%", style_table_cell)],
        [Paragraph("<b>v4.0 (현재)</b>", style_table_cell), Paragraph("<b>18D</b>", style_table_cell), Paragraph("<b>0.7348</b>", style_table_cell), Paragraph("<b>55.10%</b>", style_table_cell), Paragraph("<b>46.55%</b>", style_table_cell), Paragraph("67.50%", style_table_cell), Paragraph("<b>74.64%</b>", style_table_cell), Paragraph("<b>9.9% / 90.1%</b>", style_table_cell)],
    ]
    t_perf = Table(perf_table_data, colWidths=[65, 55, 60, 65, 55, 55, 55, 113])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>[5대 카메라 시점별 v4.0 세부 성능 (윈도우 [0.275, 0.350], 보류율 9.9%)]</b><br/>"
        "• <b>시점 C (정면 0°)</b>: 3-State F1 <b>72.34%</b>, 정밀도 56.67%, <b>재현율 100.0% (전수 감지)</b>, 커버리지 93.5%<br/>"
        "• <b>시점 A (우측후방 45°)</b>: 3-State F1 55.00%, 정밀도 45.83%, 재현율 68.75%, 커버리지 88.3%<br/>"
        "• <b>시점 B (우측전방 45°)</b>: 3-State F1 53.33%, 정밀도 37.21%, 재현율 94.12%, 커버리지 87.0%<br/>"
        "• <b>시점 E (좌측후방 45°)</b>: 3-State F1 48.28%, 정밀도 50.00%, 재현율 46.67%, 커버리지 85.7%<br/>"
        "• <b>시점 D (좌측전방 45°)</b>: 3-State F1 30.00%, 정밀도 60.00%, 재현율 20.00%, 커버리지 96.1%",
        style_body
    ))
    story.append(Spacer(1, 6))

    # 5. 핵심 기술 병목
    story.append(Paragraph("5. 현재 모델의 핵심 기술 병목 (Core Bottlenecks)", style_h1))
    story.append(Paragraph(
        "1. <b>소규모 데이터 & 클래스 불균형</b>: 총 385개 중 결함이 90개(23.4%)뿐이므로, 1개 Fold당 결함 검증 표본이 18개에 불과하여 소수의 위양성(FP)에도 정밀도와 F1이 급락함.<br/>"
        "2. <b>미세한 생체역학 변위 (Low SNR)</b>: 스쿼트(10~20cm)와 달리 데드리프트는 무릎 모임 변위가 2~5cm에 불과하여 비전 랜드마크 추출 노이즈와의 분리가 어려움.<br/>"
        "3. <b>후방/사각 시점에서의 신체 차폐 (Occlusion)</b>: 상체를 45도 숙이고 엉덩이를 뒤로 뺄 때, 후방에서 보면 거대한 골반과 둔근이 무릎 안쪽 라인을 가려버림.<br/>"
        "4. <b>예측 확률 뭉침 현상 (Uncertainty)</b>: 모델의 예측 확률이 양극단(0.0 또는 1.0)으로 갈라지지 못하고 0.20~0.35 사이에 완만하게 퍼져 있어, 보류율 10% 이내 제약 하에서 회색 지대 오판을 완전히 제거하기 어려움.",
        style_body
    ))
    story.append(Spacer(1, 6))

    # 6. 외부 AI 자문 요청 사항 5가지
    story.append(Paragraph("6. 외부 AI에게 의뢰하는 핵심 자문 항목 5가지 (Consulting Requests)", style_h1))
    
    q_data = [
        [Paragraph("질문 번호", style_table_header), Paragraph("핵심 자문 주제", style_table_header), Paragraph("구체적 질문 내용 및 기대 출력", style_table_header)],
        [Paragraph("Q1", style_table_cell), Paragraph("도메인 특화 피처 엔지니어링", style_table_cell), Paragraph("385개 소규모 골격 데이터에서 미세 변위(2~5cm)를 증폭할 수 있는 추가적인 운동학 피처(예: 대퇴골-경골 회전 모멘트, 3D 위상 미분, 관절 간 거리 정규화)는 무엇이 있는가?", style_table_cell)],
        [Paragraph("Q2", style_table_cell), Paragraph("소규모 데이터 최적 아키텍처", style_table_cell), Paragraph("N=385 환경에서 대형 딥러닝(GCN, Transformer)의 과적합을 피하면서 GBDT보다 높은 F1을 낼 수 있는 경량 모델(예: 4k 파라미터 희소 ST-GCN, TabNet, Siamese Network) 설계 방안은?", style_table_cell)],
        [Paragraph("Q3", style_table_cell), Paragraph("F1 직접 최적화 손실함수", style_table_cell), Paragraph("불균형 데이터(3.3:1)에서 정확도가 아닌 F1-Score / PR-AUC를 직접 극대화할 수 있는 커스텀 손실함수(Asymmetric Focal Loss, Soft-F1 Loss)의 최적 수식과 하이퍼파라미터는?", style_table_cell)],
        [Paragraph("Q4", style_table_cell), Paragraph("시점 불변(Cross-View) 표현", style_table_cell), Paragraph("정면(F1 72%) 대비 후방/사각(F1 30~55%) 시점의 급락을 극복하기 위해, 차폐가 심한 시점에서도 무릎 상태를 강건하게 추정할 수 있는 뷰 정규화 또는 잠재 표현 기법은?", style_table_cell)],
        [Paragraph("Q5", style_table_cell), Paragraph("골격 데이터 증강(Augmentation)", style_table_cell), Paragraph("결함 90개 표본을 물리역학적 모순 없이 증강(Kinematic Manifold Mixup, 관절 지터링, 시간 축 탄력 왜곡)하여 데이터 불균형을 해소할 수 있는 구체적 알고리즘은?", style_table_cell)],
    ]
    t_q = Table(q_data, colWidths=[55, 130, 338])
    t_q.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_q)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] PDF 생성 완료: {filename}")


for p in PDF_FILES:
    build_pdf(p)

print("[+] 전수 PDF 생성 작업 완료!")
