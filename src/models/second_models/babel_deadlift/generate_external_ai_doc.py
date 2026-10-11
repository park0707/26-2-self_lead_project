# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_external_ai_doc.py
Purpose: 외부 AI 자문용 기술 명세서 및 가이드라인 생성
         - PDF 출력: D:\자주프 문서\외부ai용 문서.pdf
         - DOCX 출력: D:\자주프 문서\외부ai용 문서.docx
         - MD 출력: D:\자주프 문서\외부ai용 문서.md
"""

import os
import sys
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

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"
pdfmetrics.registerFont(TTFont("MalgunGothic", FONT_PATH))
pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", FONT_BOLD_PATH))

TARGET_DIR_ROOT = r"D:\자주프 문서"
os.makedirs(TARGET_DIR_ROOT, exist_ok=True)

PDF_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "외부ai용 문서.pdf")
DOCX_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "외부ai용 문서.docx")
MD_OUTPUT_PATH = os.path.join(TARGET_DIR_ROOT, "외부ai용 문서.md")


class NumberedCanvas(canvas.Canvas):
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
        if self._pageNumber > 1:
            self.setFont("MalgunGothic", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawString(40, 815, "vidio-health-trainer | 외부 AI 자문용 기술 사양서: 다중 라벨 결함 진단 최적화 가이드")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(40, 808, 555, 808)

        self.setFont("MalgunGothic", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 45, 555, 45)
        self.drawString(40, 32, "CONFIDENTIAL - vidio-health-trainer Deep Learning Core Team")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 32, page_str)
        self.restoreState()


def create_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=48,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#1A365D")
    c_secondary = colors.HexColor("#2B6CB0")
    c_dark = colors.HexColor("#2D3748")
    c_bg_box = colors.HexColor("#EDF2F7")
    c_callout = colors.HexColor("#EBF8FF")
    c_callout_border = colors.HexColor("#3182CE")

    style_title = ParagraphStyle(
        'DocTitle',
        fontName='MalgunGothic-Bold',
        fontSize=20,
        leading=26,
        textColor=c_primary,
        spaceAfter=6,
        alignment=0
    )
    style_subtitle = ParagraphStyle(
        'DocSubTitle',
        fontName='MalgunGothic',
        fontSize=11,
        leading=16,
        textColor=c_secondary,
        spaceAfter=14,
        alignment=0
    )
    style_h1 = ParagraphStyle(
        'H1',
        fontName='MalgunGothic-Bold',
        fontSize=13,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'H2',
        fontName='MalgunGothic-Bold',
        fontSize=10.5,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body',
        fontName='MalgunGothic',
        fontSize=8.8,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=6
    )
    style_bullet = ParagraphStyle(
        'Bullet',
        fontName='MalgunGothic',
        fontSize=8.5,
        leading=13,
        textColor=c_dark,
        leftIndent=12,
        spaceAfter=3
    )
    style_callout = ParagraphStyle(
        'CalloutText',
        fontName='MalgunGothic',
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#2C5282")
    )
    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName='MalgunGothic',
        fontSize=8.0,
        leading=11.5,
        textColor=c_dark
    )
    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName='MalgunGothic-Bold',
        fontSize=8.0,
        leading=11.5,
        textColor=colors.white
    )

    story = []

    # 표지 헤더
    story.append(Paragraph("vidio-health-trainer | AI 연구자문 기술 사양서", style_subtitle))
    story.append(Paragraph("바벨 데드리프트 다중 라벨 결함 진단 엔진<br/>EMR & Clean F1-Score 70% 돌파를 위한 핵심 지식 사양서", style_title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=4, spaceAfter=10))

    meta_text = (
        "<b>문서 버전:</b> v4.2-Official &nbsp;|&nbsp; <b>타깃 프로젝트:</b> vidio-health-trainer &nbsp;|&nbsp; "
        "<b>작성 목적:</b> 외부 AI(GPT-4/Claude 등)의 최신 논문 및 아키텍처 자문 의뢰용<br/>"
        "<b>핵심 과제:</b> 5대 결함 다중 라벨 분류 완전 일치율(EMR) 및 정상 분류 F1-Score의 이론적 한계 돌파"
    )
    callout_data = [[Paragraph(meta_text, style_callout)]]
    t_meta = Table(callout_data, colWidths=[515])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_callout),
        ('BOX', (0,0), (-1,-1), 1, c_callout_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # 섹션 1: 프로젝트 개요 및 문제 정의
    story.append(Paragraph("1. 프로젝트 개요 및 문제 정의", style_h1))
    story.append(Paragraph(
        "본 프로젝트(<b>vidio-health-trainer</b>)는 단일 스마트폰 카메라(2D/3D 비전)로 촬영된 운동 영상을 "
        "MediaPipe Pose 33개 3D 랜드마크(64프레임 정규화)로 추출한 뒤, 웨이트 트레이닝 동작 중 발생하는 "
        "<b>5가지 핵심 자세 결함(Defects)의 공존 상태를 동시 진단</b>하는 엣지/클라우드 AI 시스템입니다.",
        style_body
    ))
    story.append(Paragraph(
        "<b>진단 대상 5대 결함 라벨 ($d_1 \sim d_5$):</b>", style_h2
    ))
    story.append(Paragraph("• <b>$d_1$ (척추 중립 실패)</b>: 요추 굴곡/후만 발생. 요추 추간판(디스크) 파열을 유발하는 최고 위험 부상 결함 (라벨 1=결함)", style_bullet))
    story.append(Paragraph("• <b>$d_2$ (발-무릎 방향 불일치)</b>: 무릎이 안쪽으로 모이는 외반(Valgus). 전방십자인대 및 반월상연골 손상 결함", style_bullet))
    story.append(Paragraph("• <b>$d_3$ (발바닥 지면 고정 실패)</b>: 뒤꿈치 또는 앞꿈치가 바닥에서 들리는 지면 반발력 기저면 불안정 결함", style_bullet))
    story.append(Paragraph("• <b>$d_4$ (무릎-골반 동시 펴짐 실패)</b>: 무릎이 먼저 펴지며 상체가 앞으로 숙여지는 '스트리퍼 풀 / 굿모닝' 역학 결함", style_bullet))
    story.append(Paragraph("• <b>$d_5$ (바벨 궤적 몸 밀착 실패)</b>: 바벨이 정강이/허벅지에서 전방으로 이탈하여 척추 모멘트암이 급증하는 역학 결함", style_bullet))
    story.append(Paragraph("• <b>Clean State ($y_{\\text{clean}}$)</b>: 5개 결함이 모두 0인 완전 정상 시퀀스 ($[0, 0, 0, 0, 0]$)", style_bullet))

    story.append(Spacer(1, 10))

    # 섹션 2: 데이터셋 구조 및 생체역학적 제약 조건
    story.append(Paragraph("2. 데이터셋 구조 및 생체역학적 실측 특성", style_h1))
    story.append(Paragraph(
        "외부 AI 연구자는 모델링 기법 제안 시 아래 <b>AI-Hub 원천 데이터셋의 실측 통계와 불변 제약조건</b>을 반드시 숙지해야 합니다.",
        style_body
    ))

    t_data_stat = [
        [Paragraph("항목", style_table_header), Paragraph("실측 수치 및 명세", style_table_header), Paragraph("엔지니어링 제약 및 의미", style_table_header)],
        [Paragraph("총 샘플 수 ($N$)", style_table_cell), Paragraph("<b>770개 시퀀스</b> (64 프레임 정규화)", style_table_cell), Paragraph("소표본 제약 (N<1000), 오버피팅 주의", style_table_cell)],
        [Paragraph("피험자 구성", style_table_cell), Paragraph("<b>9명 (피험자 1~9 전수)</b>", style_table_cell), Paragraph("StratifiedGroupKFold 완전 격리 교차검증 필수", style_table_cell)],
        [Paragraph("카메라 시점", style_table_cell), Paragraph("<b>5대 전 시점</b> (각 154건 균등 배분)<br/>시점 A(우측후방), B(우측전방), C(정면), D(좌측전방), E(좌측후방)", style_table_cell), Paragraph("시점별 신체 부위 가려짐(Occlusion) 극심<br/>단일 시점별 최적화 또는 시점 조건부 모델 필수", style_table_cell)],
        [Paragraph("정상 시퀀스 ($y_{\\text{clean}}$)", style_table_cell), Paragraph("<b>160건 (20.8%)</b>", style_table_cell), Paragraph("정상:결함 = 1:4의 불균형 데이터셋", style_table_cell)],
        [Paragraph("결함별 양성 비율", style_table_cell), Paragraph("$d_1$=27.3%, $d_2$=28.6%, $d_3$=32.5%<br/>$d_4$=17.5% (최소), $d_5$=50.0% (최대)", style_table_cell), Paragraph("$d_4$는 135건에 불과하여 클래스 붕괴 위험 높음", style_table_cell)],
        [Paragraph("32개 상태 분포", style_table_cell), Paragraph("<b>16개 상태만 존재 (50% 결손)</b><br/>0개 결함: 160건 (20.8%)<br/>1개 결함: 180건, 2개 결함: 270건, 3개 결함: 160건<br/><b>4개/5개 결함: 0건 (0.0%)</b>", style_table_cell), Paragraph("자연 상태에서는 4개 이상 동시 결함이 발생하지 않음.<br/>$d_4$ 발생 시 $d_5$ 100% 동시 발생 (Lift=2.00)<br/>$d_1$과 $d_4$는 동시 발생 거의 없음 (단 5건)", style_table_cell)],
    ]
    t_stat_flow = Table(t_data_stat, colWidths=[95, 175, 245])
    t_stat_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_box]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_stat_flow)
    story.append(Spacer(1, 10))

    # 섹션 3: 카메라 시점별 물리 센싱 특이성 (View Bottleneck)
    story.append(Paragraph("3. 카메라 시점별 물리 센싱 신뢰도 편차 (Core View Bottleneck)", style_h1))
    story.append(Paragraph(
        "단일 RGB 비전 특성상 카메라 앵글에 따라 관측 가능한 물리적 분절이 완전히 다릅니다. "
        "이를 고려하지 않은 단일 전역 분류기는 성능이 심각하게 제한됩니다.",
        style_body
    ))

    t_view_data = [
        [Paragraph("카메라 시점 명칭", style_table_header), Paragraph("주요 관측 유리 부위", style_table_header), Paragraph("센싱 한계 및 가려짐", style_table_header), Paragraph("실측 최적 성능", style_table_header)],
        [Paragraph("<b>시점 D (좌측 전방 45°)</b>", style_table_cell), Paragraph("바벨 수직 궤적, 척추 굴곡, 골반-무릎 타이밍", style_table_cell), Paragraph("반대편 다리 부분 가려짐", style_table_cell), Paragraph("<b>$d_5$ AUC 0.993, F1 97.3%<br/>EMR 40.26% (최고)</b>", style_table_cell)],
        [Paragraph("<b>시점 B (우측 전방 45°)</b>", style_table_cell), Paragraph("바벨 수직 궤적, 우측 무릎 및 척추", style_table_cell), Paragraph("좌측 무릎 가려짐", style_table_cell), Paragraph("$d_5$ AUC 0.917, EMR 31.17%", style_table_cell)],
        [Paragraph("<b>시점 C (정면 0°)</b>", style_table_cell), Paragraph("<b>양쪽 무릎-발 정렬 ($d_2$), 발바닥 ($d_3$)</b>", style_table_cell), Paragraph("깊이(Z축) 블라인드 -> <b>바벨 밀착 측정 불가</b>", style_table_cell), Paragraph("<b>$d_2$ AUC 0.936, F1 78.9%</b><br/>($d_5$ AUC는 0.586으로 무력화)", style_table_cell)],
        [Paragraph("<b>시점 E (좌측 후방 45°)</b>", style_table_cell), Paragraph("요추 및 둔근 신전 타이밍", style_table_cell), Paragraph("바벨 플레이트 및 손목 가려짐", style_table_cell), Paragraph("EMR 24.03%", style_table_cell)],
        [Paragraph("<b>시점 A (우측 후방 45°)</b>", style_table_cell), Paragraph("후방 흉추-경추 각도", style_table_cell), Paragraph("팔과 몸통에 전신 전방 관절 완전 가려짐", style_table_cell), Paragraph("기존 EMR 14.3% -> v4 23.4%", style_table_cell)],
    ]
    t_view_table = Table(t_view_data, colWidths=[110, 140, 150, 115])
    t_view_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_box]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_view_table)
    story.append(Spacer(1, 10))

    # 섹션 4: 기존 아키텍처 진화 및 실험 부검 결과 (v1.0 ~ v4.0)
    story.append(Paragraph("4. 기존 아키텍처 진화 및 실험 부검 결과 (v1.0 ~ v4.0)", style_h1))
    story.append(Paragraph(
        "우리 연구팀이 770건 5-Fold StratifiedGroupKFold 교차검증을 통해 구축하고 실험한 4세대에 걸친 아키텍처 변천사입니다.",
        style_body
    ))

    t_bench = [
        [Paragraph("버전 / 아키텍처 명", style_table_header), Paragraph("핵심 설계 구조", style_table_header), Paragraph("EMR (완전일치율)", style_table_header), Paragraph("Clean Spec (정상특이도)", style_table_header), Paragraph("주요 실패 / 성공 원인", style_table_header)],
        [Paragraph("<b>Baseline</b><br/>단순 병렬 독립 모델", style_table_cell), Paragraph("5개 개별 모델 독립 임계치 추론 ($P \\ge 0.50$)", style_table_cell), Paragraph("24.03% (185/770)", style_table_cell), Paragraph("50.00% (오탐 80건)", style_table_cell), Paragraph("라벨 간 상관관계 무시. 정상 시퀀스 2개 중 1개를 오탐(FP 80건)", style_table_cell)],
        [Paragraph("<b>v1.0</b><br/>32-Way Joint Meta MoE", style_table_cell), Paragraph("16/32개 상태 직접 소프트맥스 분류 신경망<br/>(스쿼트 성공 아키텍처 계승)", style_table_cell), Paragraph("23.64% (182/770)<br/>(-0.39%p)", style_table_cell), Paragraph("66.88% (오탐 53건)", style_table_cell), Paragraph("<b>[치명적 실패]</b> 표본 극소화(클래스당 5~45건)로 소수 결함 $d_4$ F1이 20.8%로 반토막 붕괴 (Collapse to Clean)", style_table_cell)],
        [Paragraph("<b>v2.0</b><br/>생체역학 투-트랙 캐스케이드", style_table_cell), Paragraph("Track A (체간 $d_1,d_5$) vs Track B (하지 $d_2,d_3,d_4$)<br/>+ $d_4 \\rightarrow d_5$ 결합 규칙", style_table_cell), Paragraph("26.49% (204/770)<br/>(+2.46%p)", style_table_cell), Paragraph("66.25% (오탐 54건)", style_table_cell), Paragraph("<b>[성공]</b> 신체 분절별 분해로 $d_4$ F1 48.1% 복구, 정상 오탐 26건 감축. 하지만 시점 편향 존재", style_table_cell)],
        [Paragraph("<b>v3.0</b><br/>시점적응형 투-트랙", style_table_cell), Paragraph("v2.0 + 시점별 AUC 기반 수축(Shrinkage)", style_table_cell), Paragraph("27.01% (207/770)<br/>(+0.52%p)", style_table_cell), Paragraph("66.88% (오탐 53건)", style_table_cell), Paragraph("시점 정보를 소프트 보정하였으나 시점-피처 간 명시적 상호작용 부족", style_table_cell)],
        [Paragraph("<b>v4.0 (현재 SOTA)</b><br/>시점조건부 고밀도 엔진", style_table_cell), Paragraph("<b>100D View-Conditioned Dense Tensor</b><br/>(View x Prob 25D + View x Margin 25D)<br/>+ 시점별 동적 결정 임계치 테이블", style_table_cell), Paragraph("<b>29.22% (225/770)</b><br/>(Baseline 대비 +5.19%p)", style_table_cell), Paragraph("<b>66.88% (오탐 53건)</b><br/>Clean F1: 57.99%", style_table_cell), Paragraph("<b>[최고 기록]</b> 5대 결함 F1 전원 동반 상승 ($d_2, d_4$ 50% 최초 돌파). <b>시점 D 단독 EMR 40.26% 달성</b>", style_table_cell)],
    ]
    t_bench_flow = Table(t_bench, colWidths=[90, 130, 95, 90, 110])
    t_bench_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_box]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench_flow)
    story.append(Spacer(1, 10))

    # 섹션 5: 핵심 이론적 병목 (왜 EMR 70%가 어려운가?)
    story.append(Paragraph("5. 핵심 이론적 병목: EMR과 Clean F1의 수학적 곱셈 법칙", style_h1))
    story.append(Paragraph(
        "다중 라벨 분류에서 완전 일치율(EMR)은 5개 결함이 <b>모두 완벽히 일치해야만 1점으로 인정</b>됩니다. "
        "현재 1단계 세부 모델들의 평균 정확도($p_i$)가 약 75% 수준일 때, 독립 곱셈 법칙에 의해 "
        "$\\text{EMR} = 0.75^5 \\approx \\mathbf{23.7\\%}$이며, 상관관계를 아무리 학습해도 전수 EMR은 29.2%에 묶입니다.",
        style_body
    ))
    story.append(Paragraph(
        "<b>전수 100% 커버리지에서 EMR 70%가 되기 위한 절대 조건:</b><br/>"
        "$$X^5 = 0.70 \\implies X = (0.70)^{1/5} \\approx \\mathbf{93.1\\%}$$<br/>"
        "즉, 5개 개별 모델 각각의 정확도가 93.1% 이상(F1 기준 90% 이상)이어야만 전체 EMR 70%가 수학적으로 성립합니다.",
        style_body
    ))
    story.append(Paragraph(
        "<b>3-State 선택적 판정(Selective Classification / Reject Option) 실측 결과:</b><br/>"
        "• <b>Clean F1 70%</b>: Coverage 40% (보류율 60%) 구간에서 <b>72.07%로 달성 가능</b><br/>"
        "• <b>EMR 70%</b>: 상위 10% 미만의 극단적 고신뢰 샘플만 남기고 <b>90%를 보류(Abstain) 처리해야 도달 가능</b> (실서비스 적용 곤란)",
        style_body
    ))
    story.append(Spacer(1, 10))

    # 섹션 6: 외부 AI 자문 요청 핵심 4대 과제
    story.append(Paragraph("6. 외부 AI 연구자 자문 의뢰 4대 핵심 과제 (Key RFP Questions)", style_h1))
    story.append(Paragraph(
        "아래 4가지 핵심 질문에 대해 구체적인 수학적 수식, 아키텍처 다이어그램, 최신 논문 인용(2022~2025 SOTA) 및 파이토치/사이킷런 구현 아이디어를 제시해 주십시오.",
        style_body
    ))

    t_rfp = [
        [Paragraph("과제 번호 및 핵심 주제", style_table_header), Paragraph("구체적 질문 및 자문 요청 내용", style_table_header)],
        [
            Paragraph("<b>과제 1<br/>[아키텍처/손실함수]<br/>EMR 곱셈 장벽 돌파</b>", style_table_cell),
            Paragraph(
                "• $N=770$, 5개 결함, 16개 유효 상태 환경에서 독립 모델 곱($0.75^5$)의 한계를 깨고 EMR을 35%~50% 이상으로 끌어올릴 수 있는 구조는 무엇인가?<br/>"
                "• Label Powerset의 과적합/소수 클래스 붕괴를 막는 <b>Asymmetric Focal Loss</b> 또는 <b>Distribution-Balanced Loss</b> 적용 방안<br/>"
                "• <b>Probabilistic Classifier Chains (PCC)</b> 또는 <b>GNN 기반 Label Co-occurrence Graph</b> 모델의 소표본 적용 가능성",
                style_table_cell
            )
        ],
        [
            Paragraph("<b>과제 2<br/>[정상 식별 메커니즘]<br/>Clean F1 70% 달성</b>", style_table_cell),
            Paragraph(
                "• 정상 시퀀스(20.8%)를 선별하면서도, 허리 디스크($d_1$)나 굿모닝($d_4$) 같은 치명적 결함을 정상으로 오판(False Clean)하지 않는 <b>계층적 안전 게이트키퍼(Hierarchical Safe Gatekeeper)</b> 설계법<br/>"
                "• 단일 Cutoff 임계치 대신 <b>데드밴드(Dead-band) 및 양방향 신뢰 마진(Bi-directional Confidence Margin)</b> 수학적 최적화",
                style_table_cell
            )
        ],
        [
            Paragraph("<b>과제 3<br/>[시점 특화 퓨전]<br/>View-Conditioned SOTA</b>", style_table_cell),
            Paragraph(
                "• 사용자가 '시점 D(전방 45도)' 등으로 촬영 시점을 지정했을 때, 시점별 물리 센싱 장단점(정면=무릎 우세, 측면=바벨 우세)을 가장 극대화할 수 있는 <b>Cross-Attention 또는 FiLM (Feature-wise Linear Modulation) 기반 시점 조건부 레이어</b> 구성법<br/>"
                "• 단일 시점 모델 5개 앙상블 vs 통합 시점 조건부 모델 간 최적의 트레이드오프",
                style_table_cell
            )
        ],
        [
            Paragraph("<b>과제 4<br/>[3-State 선택적 분류]<br/>실서비스 신뢰도 극대화</b>", style_table_cell),
            Paragraph(
                "• 피트니스 AI 코칭에서 허용 가능한 현실적 보류율(10~15%, Coverage $\\ge 85\\%$) 내에서 EMR과 Clean F1을 극대화하는 <b>다중 라벨 신뢰도 지표(Joint Confidence Metric)</b> 산출 공식<br/>"
                "• SelectiveNet, Deep Gamblers, Chow's Rule을 다중 라벨 상호의존성 환경에 확장 적용하는 최신 기법",
                style_table_cell
            )
        ],
    ]
    t_rfp_flow = Table(t_rfp, colWidths=[130, 385])
    t_rfp_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_box]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_rfp_flow)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] PDF 생성 완료: {output_path}")


def create_docx(output_path):
    doc = docx.Document()

    # 페이지 여백 설정 (A4 기본 1인치)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # 스타일 헬퍼
    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = "맑은 고딕"
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(26, 54, 93)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = "맑은 고딕"
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(43, 108, 176)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_body(text, bold_prefix=""):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.bold = True
            r_b.font.name = "맑은 고딕"
            r_b.font.size = Pt(9.5)
            r_b.font.color.rgb = RGBColor(45, 55, 72)
        r = p.add_run(text)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(45, 55, 72)
        return p

    # 헤더
    p_top = doc.add_paragraph()
    r_top = p_top.add_run("vidio-health-trainer | AI 연구자문 기술 사양서")
    r_top.font.name = "맑은 고딕"
    r_top.font.size = Pt(10)
    r_top.font.color.rgb = RGBColor(43, 108, 176)

    p_title = doc.add_paragraph()
    r_title = p_title.add_run("바벨 데드리프트 다중 라벨 결함 진단 엔진\nEMR & Clean F1-Score 70% 돌파를 위한 핵심 지식 사양서")
    r_title.bold = True
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = RGBColor(26, 54, 93)

    # 콜아웃 박스
    tbl_callout = doc.add_table(rows=1, cols=1)
    tbl_callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_cell = tbl_callout.cell(0, 0)
    c_cell.width = Inches(6.8)
    tcPr = c_cell._tc.get_or_add_tcPr()
    shd = parse_xml(r'<w:shd {} w:fill="EBF8FF"/>'.format(nsdecls('w')))
    tcPr.append(shd)
    borders = parse_xml(r'<w:tcBorders {}><w:top w:val="single" w:sz="6" w:space="0" w:color="3182CE"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="3182CE"/><w:left w:val="single" w:sz="12" w:space="0" w:color="3182CE"/><w:right w:val="single" w:sz="6" w:space="0" w:color="3182CE"/></w:tcBorders>'.format(nsdecls('w')))
    tcPr.append(borders)
    p_meta = c_cell.paragraphs[0]
    r_meta = p_meta.add_run(
        "문서 버전: v4.2-Official  |  타깃 프로젝트: vidio-health-trainer  |  작성 목적: 외부 AI(GPT-4/Claude 등)의 최신 논문 및 아키텍처 자문 의뢰용\n"
        "핵심 과제: 5대 결함 다중 라벨 분류 완전 일치율(EMR) 및 정상 분류 F1-Score의 이론적 한계 돌파"
    )
    r_meta.font.name = "맑은 고딕"
    r_meta.font.size = Pt(8.5)
    r_meta.font.color.rgb = RGBColor(44, 82, 130)

    doc.add_paragraph().paragraph_format.space_before = Pt(8)

    # 1. 개요
    add_h1("1. 프로젝트 개요 및 문제 정의")
    add_body("본 프로젝트(vidio-health-trainer)는 단일 스마트폰 카메라(2D/3D 비전)로 촬영된 운동 영상을 MediaPipe Pose 33개 3D 랜드마크(64프레임 정규화)로 추출한 뒤, 웨이트 트레이닝 동작 중 발생하는 5가지 핵심 자세 결함(Defects)의 공존 상태를 동시 진단하는 엣지/클라우드 AI 시스템입니다.")
    add_h2("진단 대상 5대 결함 라벨 (d1 ~ d5):")
    add_body(" 요추 굴곡/후만 발생. 요추 추간판(디스크) 파열을 유발하는 최고 위험 부상 결함 (라벨 1=결함)", "• d1 (척추 중립 실패):")
    add_body(" 무릎이 안쪽으로 모이는 외반(Valgus). 전방십자인대 및 반월상연골 손상 결함", "• d2 (발-무릎 방향 불일치):")
    add_body(" 뒤꿈치 또는 앞꿈치가 바닥에서 들리는 지면 반발력 기저면 불안정 결함", "• d3 (발바닥 지면 고정 실패):")
    add_body(" 무릎이 먼저 펴지며 상체가 앞으로 숙여지는 '스트리퍼 풀 / 굿모닝' 역학 결함", "• d4 (무릎-골반 동시 펴짐 실패):")
    add_body(" 바벨이 정강이/허벅지에서 전방으로 이탈하여 척추 모멘트암이 급증하는 역학 결함", "• d5 (바벨 궤적 몸 밀착 실패):")
    add_body(" 5개 결함이 모두 0인 완전 정상 시퀀스 ([0, 0, 0, 0, 0])", "• Clean State (y_clean):")

    # 2. 데이터셋 구조
    add_h1("2. 데이터셋 구조 및 생체역학적 실측 특성")
    add_body("외부 AI 연구자는 모델링 기법 제안 시 아래 AI-Hub 원천 데이터셋의 실측 통계와 불변 제약조건을 반드시 숙지해야 합니다.")

    stat_data = [
        ["항목", "실측 수치 및 명세", "엔지니어링 제약 및 의미"],
        ["총 샘플 수 (N)", "770개 시퀀스 (64 프레임)", "소표본 제약 (N<1000), 오버피팅 엄격 통제 필요"],
        ["피험자 구성", "9명 (피험자 1~9 전수)", "StratifiedGroupKFold 완전 격리 교차검증 필수"],
        ["카메라 시점", "5대 전 시점 (각 154건 균등)\n시점 A(우측후방), B(우측전방), C(정면), D(좌측전방), E(좌측후방)", "시점별 신체 부위 가려짐(Occlusion) 극심\n시점 조건부 모델 필수"],
        ["정상 시퀀스", "160건 (20.8%)", "정상:결함 = 1:4 불균형 데이터셋"],
        ["결함별 양성 비율", "d1=27.3%, d2=28.6%, d3=32.5%\nd4=17.5% (최소), d5=50.0% (최대)", "d4는 135건에 불과하여 클래스 붕괴 위험"],
        ["32개 상태 분포", "16개 상태만 존재 (50% 결손)\n0개 결함: 160건, 1개: 180건, 2개: 270건, 3개: 160건\n4개/5개 결함: 0건 (0.0%)", "자연 상태에서 4개 이상 동시 결함 없음\nd4 발생 시 d5 100% 동시 발생 (Lift=2.00)\nd1과 d4는 동시 발생 거의 없음 (단 5건)"],
    ]
    t1 = doc.add_table(rows=len(stat_data), cols=3)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(stat_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.name = "맑은 고딕"
            p.runs[0].font.size = Pt(8.5)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(parse_xml(r'<w:shd {} w:fill="1A365D"/>'.format(nsdecls('w'))))
            else:
                if r_idx % 2 == 1:
                    tcPr = cell._tc.get_or_add_tcPr()
                    tcPr.append(parse_xml(r'<w:shd {} w:fill="F7FAFC"/>'.format(nsdecls('w'))))

    # 3. 시점별 물리 편차
    add_h1("3. 카메라 시점별 물리 센싱 신뢰도 편차 (Core View Bottleneck)")
    add_body("단일 RGB 비전 특성상 카메라 앵글에 따라 관측 가능한 물리적 분절이 완전히 다릅니다. 이를 고려하지 않은 단일 전역 분류기는 성능이 심각하게 제한됩니다.")
    add_body("• 시점 D (좌측 전방 45°): d5(바벨) AUC 0.993 (F1 97.3% 완벽), 단독 EMR 40.26% 최고 성능")
    add_body("• 시점 C (정면 0°): d2(무릎 모임) AUC 0.936 (F1 78.9% 완벽), 그러나 깊이 Z축 상실로 d5(바벨) AUC는 0.586으로 무력화")
    add_body("• 시점 A (우측 후방 45°): 팔과 몸통에 전방 관절이 완전히 가려져 기존 EMR이 14.3%에 불과했음")

    # 4. 아키텍처 진화
    add_h1("4. 기존 아키텍처 진화 및 실험 부검 결과 (v1.0 ~ v4.0)")
    bench_data = [
        ["버전", "핵심 설계 구조", "EMR", "Clean Spec", "결과 요약 및 부검"],
        ["Baseline", "5개 독립 모델 병렬 추론", "24.03%", "50.00%", "정상 2개 중 1개 오탐 (FP 80건)"],
        ["v1.0", "32-Way Joint Meta MoE", "23.64%", "66.88%", "[실패] 표본 극소화로 d4 F1 20.8% 반토막 붕괴"],
        ["v2.0", "생체역학 투-트랙 캐스케이드", "26.49%", "66.25%", "[성공] 체간/하지 분해로 d4 F1 48.1% 복구"],
        ["v3.0", "시점적응형 투-트랙", "27.01%", "66.88%", "시점별 신뢰도 수축 적용으로 207건 일치"],
        ["v4.0 (SOTA)", "100D 시점조건부 고밀도 엔진", "29.22%", "66.88%", "[최고기록] 5대 결함 F1 전원 동반 상승, 시점 D EMR 40.26%"],
    ]
    t2 = doc.add_table(rows=len(bench_data), cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(bench_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.name = "맑은 고딕"
            p.runs[0].font.size = Pt(8.5)
            if r_idx == 0:
                p.runs[0].bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(parse_xml(r'<w:shd {} w:fill="1A365D"/>'.format(nsdecls('w'))))

    # 5. 이론적 병목
    add_h1("5. 핵심 이론적 병목: EMR과 Clean F1의 수학적 곱셈 법칙")
    add_body("다중 라벨 분류에서 EMR은 5개 결함이 모두 완벽히 일치해야만 1점으로 인정됩니다. 개별 모델의 평균 정확도가 약 75% 수준일 때 독립 곱셈 법칙에 의해 0.75^5 = 23.7%이며, 상관관계를 아무리 학습해도 현재 29.22%입니다.")
    add_body("전수 100% 커버리지에서 EMR 70%가 되려면 개별 모델 5개 각각의 정확도가 (0.70)^(1/5) = 93.1% 이상(F1 90% 이상)이어야 합니다.")
    add_body("3-State 선택적 판정에서 Clean F1은 Coverage 40%에서 72.07%로 도달 가능하지만, EMR 70%는 90% 이상의 샘플을 보류(Abstain)해야 하므로 실서비스에 바로 쓰기 어렵습니다.")

    # 6. RFP 4대 과제
    add_h1("6. 외부 AI 연구자 자문 의뢰 4대 핵심 과제 (Key RFP Questions)")
    rfp_items = [
        ("과제 1 [아키텍처/손실함수 - EMR 장벽 돌파]", "N=770 소표본에서 독립 모델 곱의 한계를 깨고 EMR을 35%~50% 이상으로 끌어올릴 수 있는 최신 멀티라벨 아키텍처는 무엇인가? Asymmetric Focal Loss, Distribution-Balanced Loss, Probabilistic Classifier Chains(PCC), 또는 GNN 기반 Label Co-occurrence Graph 모델의 소표본 적용 방안을 제시해 주십시오."),
        ("과제 2 [정상 식별 메커니즘 - Clean F1 70% 달성]", "정상 시퀀스(20.8%)를 선별하면서도 허리 디스크(d1)나 굿모닝(d4) 같은 치명적 결함을 정상으로 놓치지 않는(False Clean 방지) 계층적 안전 게이트키퍼(Hierarchical Safe Gatekeeper) 설계법과 데드밴드(Dead-band) 신뢰 마진 제어 알고리즘을 제안해 주십시오."),
        ("과제 3 [시점 특화 퓨전 - View-Conditioned SOTA]", "사용자가 촬영 시점을 지정했을 때 시점별 물리 센싱 장단점(정면=무릎 우세, 측면=바벨 우세)을 극대화할 수 있는 Cross-Attention 또는 FiLM 기반 시점 조건부 신경망 레이어 구성법을 제안해 주십시오."),
        ("과제 4 [3-State 선택적 분류 - 실서비스 신뢰도 극대화]", "피트니스 AI 서비스의 현실적 허용 보류율(10~15%, Coverage 85% 이상) 내에서 EMR과 Clean F1을 극대화할 수 있는 다중 라벨 결합 신뢰도 지표(Joint Confidence Metric) 산출 공식을 제시해 주십시오."),
    ]
    for title, desc in rfp_items:
        add_h2(title)
        add_body(desc)

    doc.save(output_path)
    print(f"[+] DOCX 생성 완료: {output_path}")


def create_markdown(output_path):
    md_content = """# vidio-health-trainer | AI 연구자문 기술 사양서
## 바벨 데드리프트 다중 라벨 결함 진단 엔진: EMR & Clean F1-Score 70% 돌파를 위한 핵심 지식 사양서

> **문서 버전:** v4.2-Official  
> **타깃 프로젝트:** vidio-health-trainer  
> **작성 목적:** 외부 AI(GPT-4, Claude 3.5, Gemini Pro 등)의 최신 논문 및 아키텍처 자문 의뢰용  
> **핵심 과제:** 5대 결함 다중 라벨 분류 완전 일치율(EMR) 및 정상 분류 F1-Score의 이론적 한계 돌파  

---

## 1. 프로젝트 개요 및 문제 정의
본 프로젝트(**vidio-health-trainer**)는 단일 스마트폰 카메라(2D/3D 비전)로 촬영된 운동 영상을 MediaPipe Pose 33개 3D 랜드마크(64프레임 정규화)로 추출한 뒤, 웨이트 트레이닝 동작 중 발생하는 **5가지 핵심 자세 결함(Defects)의 공존 상태를 동시 진단**하는 엣지/클라우드 AI 시스템입니다.

### 진단 대상 5대 결함 라벨 ($d_1 \sim d_5$):
* **$d_1$ (척추 중립 실패)**: 요추 굴곡/후만 발생. 요추 추간판(디스크) 파열을 유발하는 최고 위험 부상 결함 (라벨 1=결함)
* **$d_2$ (발-무릎 방향 불일치)**: 무릎이 안쪽으로 모이는 외반(Valgus). 전방십자인대 및 반월상연골 손상 결함
* **$d_3$ (발바닥 지면 고정 실패)**: 뒤꿈치 또는 앞꿈치가 바닥에서 들리는 지면 반발력 기저면 불안정 결함
* **$d_4$ (무릎-골반 동시 펴짐 실패)**: 무릎이 먼저 펴지며 상체가 앞으로 숙여지는 '스트리퍼 풀 / 굿모닝' 역학 결함
* **$d_5$ (바벨 궤적 몸 밀착 실패)**: 바벨이 정강이/허벅지에서 전방으로 이탈하여 척추 모멘트암이 급증하는 역학 결함
* **Clean State ($y_{\\text{clean}}$)**: 5개 결함이 모두 0인 완전 정상 시퀀스 ($[0, 0, 0, 0, 0]$)

---

## 2. 데이터셋 구조 및 생체역학적 실측 특성
외부 AI 연구자는 모델링 기법 제안 시 아래 **AI-Hub 원천 데이터셋의 실측 통계와 불변 제약조건**을 반드시 숙지해야 합니다.

| 항목 | 실측 수치 및 명세 | 엔지니어링 제약 및 의미 |
| :--- | :--- | :--- |
| **총 샘플 수 ($N$)** | **770개 시퀀스** (64 프레임 정규화) | 소표본 제약 ($N < 1000$), 오버피팅 엄격 통제 필요 |
| **피험자 구성** | **9명 (피험자 1~9 전수)** | StratifiedGroupKFold 완전 격리 교차검증 필수 |
| **카메라 시점** | **5대 전 시점** (각 154건 균등 배분)<br/>시점 A(우측후방), B(우측전방), C(정면), D(좌측전방), E(좌측후방) | 시점별 신체 부위 가려짐(Occlusion) 극심<br/>시점 조건부 모델 필수 |
| **정상 시퀀스 ($y_{\\text{clean}}$)** | **160건 (20.8%)** | 정상:결함 = 1:4의 불균형 데이터셋 |
| **결함별 양성 비율** | $d_1$=27.3%, $d_2$=28.6%, $d_3$=32.5%<br/>$d_4$=17.5% (최소), $d_5$=50.0% (최대) | $d_4$는 135건에 불과하여 클래스 붕괴 위험 |
| **32개 상태 분포** | **16개 상태만 존재 (50% 결손)**<br/>0개 결함: 160건 (20.8%)<br/>1개: 180건, 2개: 270건, 3개: 160건<br/>**4개/5개 결함: 0건 (0.0%)** | 자연 상태에서 4개 이상 동시 결함 없음<br/>$d_4$ 발생 시 $d_5$ 100% 동시 발생 (Lift=2.00)<br/>$d_1$과 $d_4$는 동시 발생 거의 없음 (단 5건) |

---

## 3. 카메라 시점별 물리 센싱 신뢰도 편차 (Core View Bottleneck)
단일 RGB 비전 특성상 카메라 앵글에 따라 관측 가능한 물리적 분절이 완전히 다릅니다:
* **시점 D (좌측 전방 45°)**: 바벨 수직 궤적과 척추 굴곡이 가장 선명하게 관측됨 $\rightarrow$ **$d_5$ AUC 0.993 (F1 97.3% 완벽), 단독 EMR 40.26% 최고 성능**
* **시점 C (정면 0°)**: 관상면 무릎 모임($d_2$)과 발바닥($d_3$)이 가장 완벽하게 잡힘 $\rightarrow$ **$d_2$ AUC 0.936 (F1 78.9% 완벽)**, 그러나 깊이(Z축) 정보 상실로 바벨 밀착($d_5$) AUC는 0.586으로 무력화
* **시점 A (우측 후방 45°)**: 팔과 몸통에 전방 관절이 완전히 가려져 기존 단독 EMR이 14.3%에 불과했음 (v4에서 23.4%로 구제)

---

## 4. 기존 아키텍처 진화 및 실험 부검 결과 (v1.0 ~ v4.0)

| 버전 / 아키텍처 명 | 핵심 설계 구조 | EMR (완전일치율) | Clean Spec (정상특이도) | 주요 실패 / 성공 원인 |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline**<br/>단순 병렬 독립 모델 | 5개 개별 모델 독립 임계치 추론 ($P \ge 0.50$) | 24.03% (185/770) | 50.00% (오탐 80건) | 라벨 간 상관관계 무시. 정상 시퀀스 2개 중 1개를 오탐 |
| **v1.0**<br/>32-Way Joint Meta MoE | 16/32개 상태 직접 소프트맥스 분류 신경망 | 23.64% (182/770) | 66.88% (오탐 53건) | **[실패]** 표본 극소화로 소수 결함 $d_4$ F1이 20.8%로 반토막 붕괴 |
| **v2.0**<br/>생체역학 투-트랙 캐스케이드 | Track A (체간) vs Track B (하지)<br/>+ $d_4 \rightarrow d_5$ 결합 규칙 | 26.49% (204/770) | 66.25% (오탐 54건) | **[성공]** 신체 분절별 분해로 $d_4$ F1 48.1% 복구 |
| **v3.0**<br/>시점적응형 투-트랙 | v2.0 + 시점별 AUC 기반 수축(Shrinkage) | 27.01% (207/770) | 66.88% (오탐 53건) | 시점 신뢰도 수축 적용으로 207건 일치 |
| **v4.0 (현재 SOTA)**<br/>시점조건부 고밀도 엔진 | **100D View-Conditioned Dense Tensor**<br/>(View x Prob 25D + View x Margin 25D)<br/>+ 시점별 동적 결정 임계치 테이블 | **29.22% (225/770)** | **66.88% (오탐 53건)**<br/>Clean F1: 57.99% | **[최고기록]** 5대 결함 F1 전원 동반 상승 ($d_2, d_4$ 50% 최초 돌파). **시점 D 단독 EMR 40.26% 달성** |

---

## 5. 핵심 이론적 병목: EMR과 Clean F1의 수학적 곱셈 법칙
다중 라벨 분류에서 완전 일치율(EMR)은 5개 결함이 모두 완벽히 일치해야만 1점으로 인정됩니다:
$$\\text{EMR} = P(\\text{correct}_{d_1} \\cap \\text{correct}_{d_2} \\cap \\text{correct}_{d_3} \\cap \\text{correct}_{d_4} \\cap \\text{correct}_{d_5})$$
현재 개별 모델들의 평균 정확도가 약 75% 수준일 때, 독립 곱셈 법칙에 의해 $0.75^5 \\approx \\mathbf{23.7\\%}$이며, 상관관계를 학습한 현재 최고 EMR은 29.22%입니다.

전수 100% 커버리지에서 EMR 70%가 되기 위해서는:
$$X^5 = 0.70 \\implies X = (0.70)^{1/5} \\approx \\mathbf{93.1\\%}$$
즉, 5개 개별 모델 각각의 정확도가 93.1% 이상(F1 기준 90% 이상)이어야만 전체 EMR 70%가 수학적으로 성립합니다.

3-State 선택적 판정(Selective Classification) 실측 결과:
* **Clean F1 70%**: Coverage 40% (보류율 60%) 구간에서 **72.07%로 달성 가능**
* **EMR 70%**: 상위 10% 미만의 극단적 고신뢰 샘플만 남기고 **90%를 보류(Abstain) 처리해야 도달 가능**

---

## 6. 외부 AI 연구자 자문 의뢰 4대 핵심 과제 (Key RFP Questions)

### 과제 1 [아키텍처/손실함수 - EMR 장벽 돌파]
* $N=770$, 5개 결함, 16개 유효 상태 환경에서 독립 모델 곱($0.75^5$)의 한계를 깨고 EMR을 35%~50% 이상으로 끌어올릴 수 있는 최신 멀티라벨 아키텍처는 무엇인가?
* Label Powerset의 과적합/소수 클래스 붕괴를 막는 **Asymmetric Focal Loss** 또는 **Distribution-Balanced Loss** 적용 방안을 제시해 주십시오.
* **Probabilistic Classifier Chains (PCC)** 또는 **GNN 기반 Label Co-occurrence Graph** 모델의 소표본 적용 가능성을 검토해 주십시오.

### 과제 2 [정상 식별 메커니즘 - Clean F1 70% 달성]
* 정상 시퀀스(20.8%)를 선별하면서도, 허리 디스크($d_1$)나 굿모닝($d_4$) 같은 치명적 결함을 정상으로 놓치지 않는(False Clean 방지) **계층적 안전 게이트키퍼(Hierarchical Safe Gatekeeper)** 설계법을 제안해 주십시오.
* 단일 Cutoff 임계치 대신 **데드밴드(Dead-band) 및 양방향 신뢰 마진(Bi-directional Confidence Margin)** 수학적 최적화 공식을 제시해 주십시오.

### 과제 3 [시점 특화 퓨전 - View-Conditioned SOTA]
* 사용자가 '시점 D(전방 45도)' 등으로 촬영 시점을 지정했을 때, 시점별 물리 센싱 장단점(정면=무릎 우세, 측면=바벨 우세)을 극대화할 수 있는 **Cross-Attention 또는 FiLM (Feature-wise Linear Modulation) 기반 시점 조건부 신경망 레이어** 구성법을 제안해 주십시오.

### 과제 4 [3-State 선택적 분류 - 실서비스 신뢰도 극대화]
* 피트니스 AI 코칭에서 허용 가능한 현실적 보류율(10~15%, Coverage $\ge 85\%$) 내에서 EMR과 Clean F1을 극대화할 수 있는 **다중 라벨 결합 신뢰도 지표(Joint Confidence Metric)** 산출 공식을 제시해 주십시오.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Markdown 생성 완료: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("  [vidio-health-trainer] 외부 AI 자문용 기술 사양서 및 가이드라인 일괄 생성")
    print("=" * 80)
    create_pdf(PDF_OUTPUT_PATH)
    create_docx(DOCX_OUTPUT_PATH)
    create_markdown(MD_OUTPUT_PATH)
    print("\n[+] 모든 산출물 생성 완료:")
    print(f"  - PDF : {PDF_OUTPUT_PATH}")
    print(f"  - DOCX: {DOCX_OUTPUT_PATH}")
    print(f"  - MD  : {MD_OUTPUT_PATH}")
