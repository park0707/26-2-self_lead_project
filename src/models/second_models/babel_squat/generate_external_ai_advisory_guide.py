# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_external_ai_advisory_guide.py
Purpose: 바벨 스쿼트 v6.0 통합 엔진 아키텍처 및 성능 저하 원인 정밀 분석 설명서 생성 (PDF + Word 동시 생성)
Target PDF: D:\자주프 문서\5주차\외부_AI_자문용_설명서_바벨스쿼트_v6_통합엔진_구조_및_성능저하분석.pdf
Target DOCX: D:\자주프 문서\5주차\외부_AI_자문용_설명서_바벨스쿼트_v6_통합엔진_구조_및_성능저하분석.docx
"""

import os
import sys
import shutil
import numpy as np

# ReportLab for PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Python-docx for DOCX
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOC_DIR_5 = r"D:\자주프 문서\5주차"
DOC_DIR_ROOT = r"D:\자주프 문서"

PDF_NAME = "외부_AI_자문용_설명서_바벨스쿼트_v6_통합엔진_구조_및_성능저하분석.pdf"
DOCX_NAME = "외부_AI_자문용_설명서_바벨스쿼트_v6_통합엔진_구조_및_성능저하분석.docx"

PDF_PATH_5 = os.path.join(DOC_DIR_5, PDF_NAME)
PDF_PATH_ROOT = os.path.join(DOC_DIR_ROOT, PDF_NAME)
DOCX_PATH_5 = os.path.join(DOC_DIR_5, DOCX_NAME)
DOCX_PATH_ROOT = os.path.join(DOC_DIR_ROOT, DOCX_NAME)


def register_fonts():
    font_regular = r"C:\Windows\Fonts\malgun.ttf"
    font_bold = r"C:\Windows\Fonts\malgunbd.ttf"
    if os.path.exists(font_regular):
        pdfmetrics.registerFont(TTFont("Malgun", font_regular))
    if os.path.exists(font_bold):
        pdfmetrics.registerFont(TTFont("MalgunBold", font_bold))


def build_pdf():
    register_fonts()
    doc = SimpleDocTemplate(
        PDF_PATH_5,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='MalgunBold',
        fontSize=18,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Malgun',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='MalgunBold',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='MalgunBold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Malgun',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )
    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        fontName='MalgunBold',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='Malgun',
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=12,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        'Callout_Text',
        fontName='Malgun',
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor('#1A202C')
    )
    th_style = ParagraphStyle(
        'TH_Style',
        fontName='MalgunBold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD_Style',
        fontName='Malgun',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748'),
        alignment=1
    )
    td_bold = ParagraphStyle(
        'TD_Bold',
        fontName='MalgunBold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1A365D'),
        alignment=1
    )

    story = []

    # 1. 헤더 & 타이틀
    story.append(Paragraph("vidio-health-trainer 시스템 기술 자문 설명서", title_style))
    story.append(Paragraph("바벨 스쿼트 통합 진단 엔진 v6.0 아키텍처 명세 및 전수 일치율(66%)·정상 검출률(23.6%) 저하 원인 분석 보고서 (외부 AI 검토용)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A365D'), spaceAfter=10))

    # 메타 테이블
    meta_data = [
        [
            Paragraph("<b>프로젝트명</b>", td_style),
            Paragraph("vidio-health-trainer", td_bold),
            Paragraph("<b>대상 과제</b>", td_style),
            Paragraph("바벨 스쿼트 정상/오류 통합 엔진 v6.0", td_bold)
        ],
        [
            Paragraph("<b>평가 표본수</b>", td_style),
            Paragraph("2,605건 전수 (AI-Hub 마스터 데이터셋)", td_bold),
            Paragraph("<b>핵심 자문 이슈</b>", td_style),
            Paragraph("유효 EMR(66%) 및 Strict Clean F1(23.6%)", td_bold)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[80, 175, 80, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 섹션 1
    story.append(Paragraph("1. 연구 배경 및 외부 AI 자문 요청 목적", h1_style))
    story.append(Paragraph(
        "본 프로젝트(<b>vidio-health-trainer</b>)는 단일 2D 모바일/웹캠 카메라로 촬영된 사용자의 바벨 스쿼트 운동 영상을 입력받아, "
        "<b>MediaPipe Pose 33개 관절의 64프레임 시계열 3D 좌표(3차 스플라인 보간)</b>만을 활용하여 "
        "스쿼트의 '정상 여부' 및 4대 세부 결함(척추 중립, 고개 정면, 무릎 정렬, 발바닥 접지)을 고정밀 진단하는 온디바이스/클라우드 AI 엔진을 개발하고 있습니다.",
        body_style
    ))
    story.append(Paragraph(
        "최근 4대 세부 결함 모델 각각의 최신 공식 버전을 구축하고, 게이트키퍼 병목을 제거한 <b>부위별 독립 3-State E2E 통합 엔진 v6.0</b>을 완성하였습니다. "
        "그러나 2,605건 전수(5-Fold 피험자 완전 격리) 실측 평가 결과 다음의 <b>2대 심각한 성능 정체 현상</b>이 확인되어 외부 AI 전문가의 기술 자문을 구하고자 합니다:",
        body_style
    ))
    story.append(Paragraph("• <b>문제 1 (유효 라벨 전수 일치율 저조):</b> 4개 부위 중 판단 불가 부위를 제외한 유효 판정들에 대해 모든 라벨을 100% 맞춘 완전 일치율(Exact Match Ratio)이 <b>66.35% (1,725건 / 2,600건)</b>에 머물고 있습니다.", bullet_style))
    story.append(Paragraph("• <b>문제 2 (엄격 정상 인식률 폭락):</b> 4대 결함 부위가 모두 결함이 없는 순수 무결점 정상 스쿼트(총 135건)에 대해, 시스템이 4개 부위 모두 정상으로 맞춘 엄격 정상 F1-Score가 <b>23.60% (Recall 14.07%, 19건 적중)</b>로 극도로 저조합니다. 이전 프로토타입 서브셋 평가에서는 60%대를 기록했으나 전수 기준에서 폭락한 원인 규명이 필요합니다.", bullet_style))

    # 섹션 2
    story.append(Paragraph("2. 데이터셋 규격 및 클래스 분포 특성", h1_style))
    story.append(Paragraph(
        "• <b>총 표본수:</b> 2,605건 (AI-Hub 피트니스 운동 자세 평가 마스터 데이터셋)<br/>"
        "• <b>피험자:</b> 1번~9번 전원 (9명) 완전 격리 (StratifiedGroupKFold 교차검증)<br/>"
        "• <b>5대 전 시점 균등 분할:</b> view1(완전 후방 180°, 521건), view2(우측 45°, 521건), view3(완전 정면 0°, 521건), view4(좌측 45°, 521건), view5(우측 후방 45°, 521건)<br/>"
        "• <b>극단적 클래스 불균형:</b><br/>"
        "  - 4개 부위 모두 무결점 정상인 표본(Strict Clean): <b>135건 (5.18%)</b><br/>"
        "  - 1개 이상 결함을 포함하는 표본(Defect): <b>2,470건 (94.82%)</b> (약 19:1의 극심한 결함 편향)",
        body_style
    ))

    # 섹션 3
    story.append(Paragraph("3. 현재 v6.0 통합 엔진 아키텍처 및 4대 서브모델 스펙", h1_style))
    story.append(Paragraph(
        "통합 엔진 v6.0은 연산 절감을 위한 거시 게이트키퍼(상·하체 조기 차단)를 전면 제거하고, 4대 서브모델이 MediaPipe 64프레임 시퀀스를 병렬로 입력받아 "
        "독립적으로 <b>3-State [0: 정상(NORMAL), 1: 결함(DEFECT), -1: 판단 불가(UNDETERMINED)]</b>를 반환하도록 설계되었습니다.",
        body_style
    ))

    sub_spec_data = [
        [
            Paragraph("결함 코드 및 항목", th_style),
            Paragraph("모델 버전 및 아키텍처", th_style),
            Paragraph("입력 피처 규격", th_style),
            Paragraph("3-State 판정 임계치", th_style),
            Paragraph("단독 실측 F1 / 정확도", th_style)
        ],
        [
            Paragraph("<b>c1</b>. 척추의 중립 위반<br/>(Spine Neutrality)", td_style),
            Paragraph("v11.0 3-Way Tri-Ensemble<br/>(GBDT 10% + Diet 40% + TCN 50%)", td_style),
            Paragraph("38D + 27D + 24D 시계열<br/>(Nadir + 생체역학 + TCN)", td_style),
            Paragraph("tau_norm: 0.40<br/>tau_defect: 0.60<br/>(Fold 분산/불일치 게이트)", td_style),
            Paragraph("F1: <b>97.09%</b><br/>Acc: <b>96.93%</b><br/>Cov: 75.1%", td_style)
        ],
        [
            Paragraph("<b>c2</b>. 고개 정면 미유지<br/>(Head & Gaze)", td_style),
            Paragraph("v7.0 생체역학 앙상블<br/>(CatBoost + LightGBM)", td_style),
            Paragraph("19D 하이브리드 피처<br/>(경추 협응 + 가시귀 피치)", td_style),
            Paragraph("전면(v2,3,4): 0.40 / 0.60<br/>후방(v1,5): 0.30 / 0.70<br/>(코 미관측 시 경추 전환)", td_style),
            Paragraph("F1: <b>83.86%</b><br/>Acc: <b>83.00%</b><br/>Cov: 63.9%", td_style)
        ],
        [
            Paragraph("<b>c3</b>. 무릎-발 방향 일치<br/>(Knee Valgus)", td_style),
            Paragraph("v6.0 순수 비전 앙상블<br/>(CatBoost + LightGBM)", td_style),
            Paragraph("20D 관상면 비율 피처<br/>(골반 관상면 투영 KaRatio)", td_style),
            Paragraph("tau_norm: 0.40<br/>tau_defect: 0.60<br/>(모델 불일치 게이트)", td_style),
            Paragraph("F1: <b>87.42%</b><br/>Acc: <b>87.59%</b><br/>Cov: 92.2%", td_style)
        ],
        [
            Paragraph("<b>c4</b>. 발바닥 지면 고정<br/>(Foot Ground Fixation)", td_style),
            Paragraph("v5.0 하이브리드 앙상블<br/>(CatBoost 60% + LGB 30% + RF 10%)", td_style),
            Paragraph("25D 접촉 상태 머신<br/>(F0 가시성 + 정강이 상관계수)", td_style),
            Paragraph("tau_norm: 0.40<br/>tau_defect: 0.60<br/>(모델 불일치 게이트)", td_style),
            Paragraph("F1: <b>82.81%</b><br/>Acc: <b>82.78%</b><br/>Cov: 85.6%", td_style)
        ]
    ]
    t_sub = Table(sub_spec_data, colWidths=[95, 125, 115, 115, 65])
    t_sub.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_sub)
    story.append(Spacer(1, 8))

    # 섹션 4: 실측 매트릭스
    story.append(Paragraph("4. 2,605건 전수 실측 2D 교차 검증 매트릭스 (k vs m)", h1_style))
    story.append(Paragraph(
        "• <b>열 (k):</b> 4개 부위 중 '판단 불가(-1)'가 발생한 개수 (0~4)<br/>"
        "• <b>행 (m):</b> 유효하게 판정된 (4-k)개 라벨 중 실제 골든 정답과 완벽히 일치한 개수 (0~4-k)",
        body_style
    ))

    matrix_data = [
        [
            Paragraph("맞춘 개수 (m) \\ 판단불가 (k)", th_style),
            Paragraph("k=0 (4개판정)", th_style),
            Paragraph("k=1 (3개판정)", th_style),
            Paragraph("k=2 (2개판정)", th_style),
            Paragraph("k=3 (1개판정)", th_style),
            Paragraph("k=4 (전부불가)", th_style),
            Paragraph("행 합계", th_style)
        ],
        [
            Paragraph("<b>m=4</b> (4개 전수적중)", td_style),
            Paragraph("<b>638건 (62.9%)</b>", td_bold),
            Paragraph("-", td_style), Paragraph("-", td_style), Paragraph("-", td_style), Paragraph("-", td_style),
            Paragraph("638건", td_style)
        ],
        [
            Paragraph("<b>m=3</b> (3개 적중)", td_style),
            Paragraph("296건 (29.2%)", td_style),
            Paragraph("<b>722건 (65.6%)</b>", td_bold),
            Paragraph("-", td_style), Paragraph("-", td_style), Paragraph("-", td_style),
            Paragraph("1,018건", td_style)
        ],
        [
            Paragraph("<b>m=2</b> (2개 적중)", td_style),
            Paragraph("72건 (7.1%)", td_style),
            Paragraph("330건 (30.0%)", td_style),
            Paragraph("<b>298건 (73.0%)</b>", td_bold),
            Paragraph("-", td_style), Paragraph("-", td_style),
            Paragraph("700건", td_style)
        ],
        [
            Paragraph("<b>m=1</b> (1개 적중)", td_style),
            Paragraph("8건 (0.8%)", td_style),
            Paragraph("48건 (4.4%)", td_style),
            Paragraph("98건 (24.0%)", td_style),
            Paragraph("<b>67건 (87.0%)</b>", td_bold),
            Paragraph("-", td_style),
            Paragraph("221건", td_style)
        ],
        [
            Paragraph("<b>m=0</b> (0개 적중)", td_style),
            Paragraph("0건 (0.0%)", td_style),
            Paragraph("1건 (0.1%)", td_style),
            Paragraph("12건 (2.9%)", td_style),
            Paragraph("10건 (13.0%)", td_style),
            Paragraph("5건", td_style),
            Paragraph("28건", td_style)
        ],
        [
            Paragraph("<b>열 합계 (Coverage)</b>", td_bold),
            Paragraph("1,014건 (38.9%)", td_bold),
            Paragraph("1,101건 (42.3%)", td_bold),
            Paragraph("408건 (15.7%)", td_bold),
            Paragraph("77건 (3.0%)", td_bold),
            Paragraph("5건 (0.2%)", td_bold),
            Paragraph("<b>2,605건 (100%)</b>", td_bold)
        ]
    ]
    t_mat = Table(matrix_data, colWidths=[125, 65, 65, 65, 65, 65, 65])
    t_mat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (2, 2), (2, 2), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (3, 3), (3, 3), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (4, 4), (4, 4), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#EDF2F7')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 8))

    # 섹션 5: 심층 원인 분석
    story.append(Paragraph("5. 핵심 성능 정체 원인 심층 분석", h1_style))
    
    story.append(Paragraph("<b>[원인 A] 유효 라벨 전수 일치율(66.35%) 정체의 수학적 원인</b>", h2_style))
    story.append(Paragraph(
        "1. <b>다중 라벨 결합 확률 곱의 수학적 한계:</b> 4개 모델이 완전히 독립적으로 동작할 때 4개가 동시에 모두 맞을 확률은 각 모델 정확도의 곱입니다.<br/>"
        "   - 이론적 결합 확률: P(c1) × P(c2) × P(c3) × P(c4) = 0.9693 × 0.8300 × 0.8759 × 0.8278 = <b>58.33%</b><br/>"
        "   - 실제 실측치: <b>66.35%</b> (1,725건 / 2,600건)로, 신체 관절 간의 생체역학적 상관관계 덕분에 독립 곱보다 8.0%p 높게 나오지만, "
        "<b>c2(83.0%)와 c4(82.8%)라는 80%대 초반 서브모델의 병목</b>이 전체 전수 일치율을 60%대로 구속하고 있습니다.<br/>"
        "2. <b>오답 1개 편중 특성:</b> 2,605건 중 오답이 1개 이하인 고신뢰도 비율은 <b>94.19% (2,449건)</b>에 달합니다. "
        "즉, 34%의 미스매치 중 대다수는 4개 부위 중 3개는 완벽히 맞추고 1개 부위에서 경계값 오판이 난 케이스입니다.",
        body_style
    ))

    story.append(Paragraph("<b>[원인 B] 엄격 정상 F1-Score(23.60%) 폭락의 원인: 3-State 판단 불가의 연쇄 누적</b>", h2_style))
    story.append(Paragraph(
        "과거 서브셋 평가에서 정상 F1이 60.32%가 나왔던 것은 '판단 보류(RETAKE)로 빠진 샘플을 모수에서 제외한 유효 부분집합(N=2,374)' 기준이었으나, "
        "전수 기준 엄격 정상 평가는 <b>135건 전체 골든 정상을 분모</b>로 둡니다. 이때 다음과 같은 치명적 연쇄 누적이 발생합니다:<br/>"
        "• <b>135건의 순수 정상 표본 중 서브모델별 판정 결과:</b><br/>"
        "  - c1 v11: 정상 84건, 결함 1건, <b>판단 불가 50건 (37.0% 누락)</b> (TCN Fold 분산 및 브랜치 불일치 게이트 때문)<br/>"
        "  - c2 v7: 정상 63건, 결함 15건, <b>판단 불가 57건 (42.2% 누락)</b> (후방 시점 얼굴 미관측 때문)<br/>"
        "  - c3 v6: 정상 122건, 결함 7건, 판단 불가 6건<br/>"
        "  - c4 v5: 정상 103건, 결함 18건, 판단 불가 14건<br/>"
        "• <b>결과:</b> 135건의 정상 표본 중 <b>무려 99건(73.3%)에서 최소 1개 이상의 모델이 '판단 불가'</b>를 선언하여 '엄격 정상' 후보에서 탈락했습니다. "
        "여기에 38건의 거짓 결함(False Alarm)이 겹치면서, <b>최종적으로 4개 모두 정상으로 통과된 표본은 단 19건(Recall 14.07%)</b>에 불과하게 되었고, "
        "이로 인해 정상 판정 F1-Score가 <b>23.60%로 폭락</b>한 것입니다.",
        body_style
    ))

    # 섹션 6: 외부 AI 자문 요청 사항
    story.append(Paragraph("6. 외부 AI 전문가 핵심 자문 요청 과제 (5대 질문)", h1_style))
    
    q_data = [
        [
            Paragraph("<b>자문 과제 1: 다중 라벨 의사결정 융합 전략 (Multi-Label Decision Fusion)</b><br/>"
                      "현재는 4개 모델이 모두 NORMAL이어야만 세션 정상을 인정하는 '엄격한 논리곱(Strict AND)' 방식을 취하고 있어, "
                      "각 모델의 판단 불가(73%)와 사소한 거짓 결함(28%)으로 인해 정상 재현율이 14%로 무너집니다. "
                      "각 모델의 독립성을 해치지 않으면서 전체 세션 정상을 합리적으로 판정할 수 있는 <b>소프트 스코어 가중합, 베이지안 신뢰도 융합, "
                      "또는 신뢰도 조건부 정상 인정 규칙</b>을 어떻게 설계해야 할까요?", callout_style)
        ],
        [
            Paragraph("<b>자문 과제 2: 판단 불가(UNDETERMINED) 상태의 안전한 정상 복구 (Abstention Recovery)</b><br/>"
                      "정상인 동작임에도 c2(후방 시점 얼굴 미관측)와 c1(TCN 분산)에서 판단 불가가 대량 발생합니다. "
                      "후방 시점처럼 물리적으로 얼굴이 안 보이는 환경에서 시선(c2)이 판단 불가일 때, 나머지 척추/무릎/발이 완벽한 정상이면 "
                      "이를 안전하게 정상으로 승인해 주는 <b>시점 인지형 조건부 마스킹 규칙(View-Adaptive Masked Clean Rule)</b>이 타당한지, "
                      "그 수학적 정당성과 위험성 통제 방안은 무엇일까요?", callout_style)
        ],
        [
            Paragraph("<b>자문 과제 3: 82~83% 서브모델(c2, c4)의 병목 돌파 및 전수 일치율 향상 방안</b><br/>"
                      "현재 c1(F1 97%)과 c3(F1 87%)에 비해 c2(83.8%)와 c4(82.8%)가 병목이 되어 전수 일치율(66%)을 끌어내리고 있습니다. "
                      "단안 2D 키포인트 노이즈(발목 4~8px 떨림, 후방 경추 가림) 환경에서 이 두 모델의 정확도를 88~90% 이상으로 끌어올리기 위한 "
                      "<b>추천 피처 엔지니어링, 손실 함수 가중치, 또는 모델 간 상호 생체역학 커플링(Coupling) 기법</b>은 무엇이 있을까요?", callout_style)
        ],
        [
            Paragraph("<b>자문 과제 4: 95:5 극단적 클래스 불균형 하에서의 정상/결함 최적화</b><br/>"
                      "정상 데이터가 5.2%(135건)에 불과한 상황에서, 결함 오진을 방지하면서도 정상을 효과적으로 살리기 위한 "
                      "<b>비대칭 손실 함수(Focal Loss, Asymmetric Loss), 비용 민감 학습(Cost-sensitive), 또는 임계치 튜닝(Threshold Calibration) 전략</b>은?", callout_style)
        ],
        [
            Paragraph("<b>자문 과제 5: 사용자 피드백 UX 관점에서의 평가 지표 재정의 제안</b><br/>"
                      "현재 사용자는 부위별로 '척추: 정상, 시선: 결함, 무릎: 정상, 발: 판단불가'라는 멀티카드 피드백을 원하고 있습니다. "
                      "이러한 서비스 환경에서 '전체 동작이 완전 정상인가'라는 획일적 이진 F1-Score 대신, "
                      "<b>서비스 체감 품질과 직결되는 학술적/산업적 추천 다중 라벨 평가 지표 프레임워크</b>가 있다면 제안을 부탁드립니다.", callout_style)
        ]
    ]
    t_q = Table(q_data, colWidths=[515])
    t_q.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor('#2B6CB0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_q)

    doc.build(story)
    print(f"[+] PDF 생성 완료: {PDF_PATH_5} ({os.path.getsize(PDF_PATH_5):,} Bytes)")


def build_docx():
    os.makedirs(DOC_DIR_5, exist_ok=True)
    doc = Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    def set_cell_bg(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_mar(cell, top=100, bottom=100, left=140, right=140):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def set_tbl_borders(table, color="D3D3D3"):
        tblPr = table._tbl.tblPr
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>
                <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1A365D"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
                <w:insideV w:val="none"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr.append(borders)

    # 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    tr = title_p.add_run("vidio-health-trainer 시스템 기술 자문 설명서")
    tr.font.name = "맑은 고딕"
    tr.font.size = Pt(17)
    tr.font.bold = True
    tr.font.color.rgb = RGBColor(26, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sr = sub_p.add_run("바벨 스쿼트 통합 진단 엔진 v6.0 아키텍처 명세 및 전수 일치율(66%)·정상 검출률(23.6%) 저하 원인 분석 보고서 (외부 AI 검토용)")
    sr.font.name = "맑은 고딕"
    sr.font.size = Pt(10)
    sr.font.color.rgb = RGBColor(113, 128, 150)

    # 메타 박스
    meta_tbl = doc.add_table(rows=2, cols=4)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_tbl_borders(meta_tbl)
    w_m = [Inches(1.6), Inches(1.8), Inches(1.6), Inches(1.8)]
    m_h = ["프로젝트명", "vidio-health-trainer", "대상 과제", "바벨 스쿼트 통합 엔진 v6.0"]
    m_v = ["평가 표본수", "2,605건 (AI-Hub 마스터 전수)", "핵심 자문 이슈", "유효 EMR(66%) 및 Strict Clean F1(23.6%)"]
    for i in range(4):
        c0 = meta_tbl.rows[0].cells[i]
        c0.width = w_m[i]
        set_cell_bg(c0, "1A365D" if i % 2 == 0 else "F7FAFC")
        set_cell_mar(c0)
        p = c0.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(m_h[i])
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(26, 54, 93)

        c1 = meta_tbl.rows[1].cells[i]
        c1.width = w_m[i]
        set_cell_bg(c1, "1A365D" if i % 2 == 0 else "F7FAFC")
        set_cell_mar(c1)
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(m_v[i])
        r1.font.name = "맑은 고딕"
        r1.font.size = Pt(9)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 섹션 1
    h1 = doc.add_heading(level=1)
    hr1 = h1.add_run("1. 연구 배경 및 외부 AI 자문 요청 목적")
    hr1.font.name = "맑은 고딕"
    hr1.font.color.rgb = RGBColor(26, 54, 93)

    p1 = doc.add_paragraph()
    p1.paragraph_format.line_spacing = 1.25
    p1.add_run(
        "본 프로젝트(vidio-health-trainer)는 단일 2D 모바일/웹캠 카메라로 촬영된 사용자의 바벨 스쿼트 운동 영상을 입력받아, "
        "MediaPipe Pose 33개 관절의 64프레임 시계열 3D 좌표(3차 스플라인 보간)만을 활용하여 "
        "스쿼트의 '정상 여부' 및 4대 세부 결함(척추 중립, 고개 정면, 무릎 정렬, 발바닥 접지)을 고정밀 진단하는 온디바이스/클라우드 AI 엔진을 개발하고 있습니다.\n\n"
        "최근 4대 세부 결함 모델 각각의 최신 공식 버전을 구축하고, 게이트키퍼 병목을 제거한 부위별 독립 3-State E2E 통합 엔진 v6.0을 완성하였습니다. "
        "그러나 2,605건 전수(5-Fold 피험자 완전 격리) 실측 평가 결과 다음의 2대 심각한 성능 정체 현상이 확인되어 외부 AI 전문가의 기술 자문을 구하고자 합니다:\n"
        "1. 문제 1 (유효 라벨 전수 일치율 저조): 4개 부위 중 판단 불가 부위를 제외한 유효 판정들에 대해 모든 라벨을 100% 맞춘 완전 일치율(Exact Match Ratio)이 66.35% (1,725건 / 2,600건)에 머물고 있습니다.\n"
        "2. 문제 2 (엄격 정상 인식률 폭락): 4대 결함 부위가 모두 결함이 없는 순수 무결점 정상 스쿼트(총 135건)에 대해, 시스템이 4개 부위 모두 정상으로 맞춘 엄격 정상 F1-Score가 23.60% (Recall 14.07%, 19건 적중)로 극도로 저조합니다. 이전 프로토타입 서브셋 평가에서는 60%대를 기록했으나 전수 기준에서 폭락한 원인 규명이 필요합니다."
    )

    # 섹션 2
    h2 = doc.add_heading(level=1)
    hr2 = h2.add_run("2. 데이터셋 규격 및 클래스 분포 특성")
    hr2.font.name = "맑은 고딕"
    hr2.font.color.rgb = RGBColor(26, 54, 93)

    p2 = doc.add_paragraph()
    p2.paragraph_format.line_spacing = 1.25
    p2.add_run(
        "• 총 표본수: 2,605건 (AI-Hub 피트니스 운동 자세 평가 마스터 데이터셋 전수)\n"
        "• 피험자: 1번~9번 전원 (9명) 완전 격리 (StratifiedGroupKFold 교차검증)\n"
        "• 5대 전 시점 균등 분할: view1(완전 후방 180°, 521건), view2(우측 45°, 521건), view3(완전 정면 0°, 521건), view4(좌측 45°, 521건), view5(우측 후방 45°, 521건)\n"
        "• 극단적 클래스 불균형:\n"
        "  - 4개 부위 모두 무결점 정상인 표본(Strict Clean): 135건 (5.18%)\n"
        "  - 1개 이상 결함을 포함하는 표본(Defect): 2,470건 (94.82%) (약 19:1의 극심한 결함 편향)"
    )

    # 섹션 3
    h3 = doc.add_heading(level=1)
    hr3 = h3.add_run("3. 현재 v6.0 통합 엔진 아키텍처 및 4대 서브모델 스펙")
    hr3.font.name = "맑은 고딕"
    hr3.font.color.rgb = RGBColor(26, 54, 93)

    p3 = doc.add_paragraph()
    p3.paragraph_format.line_spacing = 1.25
    p3.add_run(
        "통합 엔진 v6.0은 연산 절감을 위한 거시 게이트키퍼(상·하체 조기 차단)를 전면 제거하고, 4대 서브모델이 MediaPipe 64프레임 시퀀스를 병렬로 입력받아 "
        "독립적으로 3-State [0: 정상(NORMAL), 1: 결함(DEFECT), -1: 판단 불가(UNDETERMINED)]를 반환하도록 설계되었습니다."
    )

    tbl_s = doc.add_table(rows=5, cols=5)
    tbl_s.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_tbl_borders(tbl_s)
    w_sub = [Inches(1.5), Inches(1.8), Inches(1.4), Inches(1.3), Inches(0.8)]

    sub_rows = [
        ["결함 항목", "모델 아키텍처", "피처 규격", "3-State 임계치", "F1 / Acc"],
        ["c1. 척추 중립 위반", "v11.0 3-Way Tri-Ensemble\n(GBDT 10% + Diet 40% + TCN 50%)", "38D+27D+24D 시계열\n(Nadir+생체역학+TCN)", "tau: [0.40, 0.60]\n(Fold분산/불일치 게이트)", "97.1% / 96.9%\n(Cov 75.1%)"],
        ["c2. 고개 정면 미유지", "v7.0 생체역학 앙상블\n(CatBoost + LightGBM)", "19D 하이브리드 피처\n(경추협응+가시귀피치)", "전면: [0.40, 0.60]\n후방: [0.30, 0.70]", "83.9% / 83.0%\n(Cov 63.9%)"],
        ["c3. 무릎-발 방향 일치", "v6.0 순수 비전 앙상블\n(CatBoost + LightGBM)", "20D 관상면 비율 피처\n(골반투영 KaRatio)", "tau: [0.40, 0.60]\n(불일치 게이트)", "87.4% / 87.6%\n(Cov 92.2%)"],
        ["c4. 발바닥 지면 고정", "v5.0 접촉머신 앙상블\n(CB 60% + LGB 30% + RF 10%)", "25D 접촉 상태 머신\n(F0 가시성+정강이상관)", "tau: [0.40, 0.60]\n(불일치 게이트)", "82.8% / 82.8%\n(Cov 85.6%)"]
    ]
    for r_i, r_data in enumerate(sub_rows):
        for c_i, val in enumerate(r_data):
            cell = tbl_s.rows[r_i].cells[c_i]
            cell.width = w_sub[c_i]
            set_cell_bg(cell, "1A365D" if r_i == 0 else "FFFFFF")
            set_cell_mar(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.bold = (r_i == 0)
            r.font.color.rgb = RGBColor(255, 255, 255) if r_i == 0 else RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 섹션 4: 실측 2D 매트릭스
    h4 = doc.add_heading(level=1)
    hr4 = h4.add_run("4. 2,605건 전수 실측 2D 교차 검증 매트릭스 (k vs m)")
    hr4.font.name = "맑은 고딕"
    hr4.font.color.rgb = RGBColor(26, 54, 93)

    p4 = doc.add_paragraph()
    p4.paragraph_format.line_spacing = 1.25
    p4.add_run(
        "• 열 (k): 4개 부위 중 '판단 불가(-1)'가 발생한 개수 (0~4)\n"
        "• 행 (m): 유효하게 판정된 (4-k)개 라벨 중 실제 골든 정답과 완벽히 일치한 개수 (0~4-k)"
    )

    tbl_m = doc.add_table(rows=7, cols=7)
    tbl_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_tbl_borders(tbl_m)
    w_mat = [Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(1.0)]

    mat_rows = [
        ["맞춘 개수 (m) \\ 판단불가 (k)", "k=0 (4개판정)", "k=1 (3개판정)", "k=2 (2개판정)", "k=3 (1개판정)", "k=4 (전부불가)", "행 합계"],
        ["m=4 (4개 전수적중)", "638건 (62.9%)", "-", "-", "-", "-", "638건"],
        ["m=3 (3개 적중)", "296건 (29.2%)", "722건 (65.6%)", "-", "-", "-", "1,018건"],
        ["m=2 (2개 적중)", "72건 (7.1%)", "330건 (30.0%)", "298건 (73.0%)", "-", "-", "700건"],
        ["m=1 (1개 적중)", "8건 (0.8%)", "48건 (4.4%)", "98건 (24.0%)", "67건 (87.0%)", "-", "221건"],
        ["m=0 (0개 적중)", "0건 (0.0%)", "1건 (0.1%)", "12건 (2.9%)", "10건 (13.0%)", "5건", "28건"],
        ["열 합계 (Coverage)", "1,014건 (38.9%)", "1,101건 (42.3%)", "408건 (15.7%)", "77건 (3.0%)", "5건 (0.2%)", "2,605건 (100%)"]
    ]
    for r_i, r_data in enumerate(mat_rows):
        for c_i, val in enumerate(r_data):
            cell = tbl_m.rows[r_i].cells[c_i]
            cell.width = w_mat[c_i]
            is_diag = (r_i == 1 and c_i == 1) or (r_i == 2 and c_i == 2) or (r_i == 3 and c_i == 3) or (r_i == 4 and c_i == 4)
            bg = "1A365D" if r_i == 0 else ("E2E8F0" if is_diag else ("EDF2F7" if r_i == 6 else "FFFFFF"))
            set_cell_bg(cell, bg)
            set_cell_mar(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.bold = (r_i == 0 or r_i == 6 or is_diag)
            r.font.color.rgb = RGBColor(255, 255, 255) if r_i == 0 else RGBColor(26, 54, 93) if is_diag else RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 섹션 5: 심층 원인 분석
    h5 = doc.add_heading(level=1)
    hr5 = h5.add_run("5. 핵심 성능 정체 원인 심층 분석")
    hr5.font.name = "맑은 고딕"
    hr5.font.color.rgb = RGBColor(26, 54, 93)

    p5 = doc.add_paragraph()
    p5.paragraph_format.line_spacing = 1.25
    p5.add_run(
        "■ [원인 A] 유효 라벨 전수 일치율(66.35%) 정체의 수학적 원인\n"
        "1. 다중 라벨 결합 확률 곱의 수학적 한계: 4개 모델이 완전히 독립적으로 동작할 때 4개가 동시에 모두 맞을 확률은 각 모델 정확도의 곱입니다.\n"
        "   - 이론적 결합 확률: P(c1) × P(c2) × P(c3) × P(c4) = 0.9693 × 0.8300 × 0.8759 × 0.8278 = 58.33%\n"
        "   - 실제 실측치: 66.35% (1,725건 / 2,600건)로, 신체 관절 간의 생체역학적 상관관계 덕분에 독립 곱보다 8.0%p 높게 나오지만, c2(83.0%)와 c4(82.8%)라는 80%대 초반 서브모델의 병목이 전체 전수 일치율을 60%대로 구속하고 있습니다.\n"
        "2. 오답 1개 편중 특성: 2,605건 중 오답이 1개 이하인 고신뢰도 비율은 94.19% (2,449건)에 달합니다. 즉, 34%의 미스매치 중 대다수는 4개 부위 중 3개는 완벽히 맞추고 1개 부위에서 경계값 오판이 난 케이스입니다.\n\n"
        "■ [원인 B] 엄격 정상 F1-Score(23.60%) 폭락의 원인: 3-State 판단 불가의 연쇄 누적\n"
        "과거 서브셋 평가에서 정상 F1이 60.32%가 나왔던 것은 '판단 보류(RETAKE)로 빠진 샘플을 모수에서 제외한 유효 부분집합(N=2,374)' 기준이었으나, 전수 기준 엄격 정상 평가는 135건 전체 골든 정상을 분모로 둡니다. 이때 다음과 같은 치명적 연쇄 누적이 발생합니다:\n"
        "• 135건의 순수 정상 표본 중 서브모델별 판정 결과:\n"
        "  - c1 v11: 정상 84건, 결함 1건, 판단 불가 50건 (37.0% 누락) (TCN Fold 분산 및 브랜치 불일치 게이트 때문)\n"
        "  - c2 v7: 정상 63건, 결함 15건, 판단 불가 57건 (42.2% 누락) (후방 시점 얼굴 미관측 때문)\n"
        "  - c3 v6: 정상 122건, 결함 7건, 판단 불가 6건\n"
        "  - c4 v5: 정상 103건, 결함 18건, 판단 불가 14건\n"
        "• 결과: 135건의 정상 표본 중 무려 99건(73.3%)에서 최소 1개 이상의 모델이 '판단 불가'를 선언하여 '엄격 정상' 후보에서 탈락했습니다. 여기에 38건의 거짓 결함(False Alarm)이 겹치면서, 최종적으로 4개 모두 정상으로 통과된 표본은 단 19건(Recall 14.07%)에 불과하게 되었고, 이로 인해 정상 판정 F1-Score가 23.60%로 폭락한 것입니다."
    )

    # 섹션 6: 외부 AI 자문 요청
    h6 = doc.add_heading(level=1)
    hr6 = h6.add_run("6. 외부 AI 전문가 핵심 자문 요청 과제 (5대 질문)")
    hr6.font.name = "맑은 고딕"
    hr6.font.color.rgb = RGBColor(26, 54, 93)

    questions = [
        ("자문 과제 1: 다중 라벨 의사결정 융합 전략 (Multi-Label Decision Fusion)",
         "현재는 4개 모델이 모두 NORMAL이어야만 세션 정상을 인정하는 '엄격한 논리곱(Strict AND)' 방식을 취하고 있어, 각 모델의 판단 불가(73%)와 사소한 거짓 결함(28%)으로 인해 정상 재현율이 14%로 무너집니다. 각 모델의 독립성을 해치지 않으면서 전체 세션 정상을 합리적으로 판정할 수 있는 소프트 스코어 가중합, 베이지안 신뢰도 융합, 또는 신뢰도 조건부 정상 인정 규칙을 어떻게 설계해야 할까요?"),
        ("자문 과제 2: 판단 불가(UNDETERMINED) 상태의 안전한 정상 복구 (Abstention Recovery)",
         "정상인 동작임에도 c2(후방 시점 얼굴 미관측)와 c1(TCN 분산)에서 판단 불가가 대량 발생합니다. 후방 시점처럼 물리적으로 얼굴이 안 보이는 환경에서 시선(c2)이 판단 불가일 때, 나머지 척추/무릎/발이 완벽한 정상이면 이를 안전하게 정상으로 승인해 주는 시점 인지형 조건부 마스킹 규칙(View-Adaptive Masked Clean Rule)이 타당한지, 그 수학적 정당성과 위험성 통제 방안은 무엇일까요?"),
        ("자문 과제 3: 82~83% 서브모델(c2, c4)의 병목 돌파 및 전수 일치율 향상 방안",
         "현재 c1(F1 97%)과 c3(F1 87%)에 비해 c2(83.8%)와 c4(82.8%)가 병목이 되어 전수 일치율(66%)을 끌어내리고 있습니다. 단안 2D 키포인트 노이즈(발목 4~8px 떨림, 후방 경추 가림) 환경에서 이 두 모델의 정확도를 88~90% 이상으로 끌어올리기 위한 추천 피처 엔지니어링, 손실 함수 가중치, 또는 모델 간 상호 생체역학 커플링(Coupling) 기법은 무엇이 있을까요?"),
        ("자문 과제 4: 95:5 극단적 클래스 불균형 하에서의 정상/결함 최적화",
         "정상 데이터가 5.2%(135건)에 불과한 상황에서, 결함 오진을 방지하면서도 정상을 효과적으로 살리기 위한 비대칭 손실 함수(Focal Loss, Asymmetric Loss), 비용 민감 학습(Cost-sensitive), 또는 임계치 튜닝(Threshold Calibration) 전략은?"),
        ("자문 과제 5: 사용자 피드백 UX 관점에서의 평가 지표 재정의 제안",
         "현재 사용자는 부위별로 '척추: 정상, 시선: 결함, 무릎: 정상, 발: 판단불가'라는 멀티카드 피드백을 원하고 있습니다. 이러한 서비스 환경에서 '전체 동작이 완전 정상인가'라는 획일적 이진 F1-Score 대신, 서비스 체감 품질과 직결되는 학술적/산업적 추천 다중 라벨 평가 지표 프레임워크가 있다면 제안을 부탁드립니다.")
    ]

    for q_title, q_desc in questions:
        box_t = doc.add_table(rows=1, cols=1)
        box_t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_tbl_borders(box_t, color="2B6CB0")
        b_cell = box_t.rows[0].cells[0]
        b_cell.width = Inches(6.8)
        set_cell_bg(b_cell, "F7FAFC")
        set_cell_mar(b_cell, top=100, bottom=100, left=140, right=140)
        p = b_cell.paragraphs[0]
        p.paragraph_format.line_spacing = 1.2
        r_t = p.add_run(f"■ {q_title}\n")
        r_t.font.name = "맑은 고딕"
        r_t.font.size = Pt(9.5)
        r_t.font.bold = True
        r_t.font.color.rgb = RGBColor(26, 54, 93)

        r_d = p.add_run(q_desc)
        r_d.font.name = "맑은 고딕"
        r_d.font.size = Pt(8.5)
        r_d.font.color.rgb = RGBColor(45, 55, 72)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    doc.save(DOCX_PATH_5)
    print(f"[+] DOCX 생성 완료: {DOCX_PATH_5} ({os.path.getsize(DOCX_PATH_5):,} Bytes)")


def copy_to_root():
    os.makedirs(DOC_DIR_ROOT, exist_ok=True)
    if os.path.exists(PDF_PATH_5):
        shutil.copy2(PDF_PATH_5, PDF_PATH_ROOT)
        print(f"[+] PDF 복사 완료 -> {PDF_PATH_ROOT}")
    if os.path.exists(DOCX_PATH_5):
        shutil.copy2(DOCX_PATH_5, DOCX_PATH_ROOT)
        print(f"[+] DOCX 복사 완료 -> {DOCX_PATH_ROOT}")


if __name__ == '__main__':
    print("=" * 105)
    print("  [vidio-health-trainer] 외부 AI 자문용 기술 설명서 (PDF + Word) 동시 생성")
    print("=" * 105)
    build_pdf()
    build_docx()
    copy_to_root()
    print("=" * 105)
