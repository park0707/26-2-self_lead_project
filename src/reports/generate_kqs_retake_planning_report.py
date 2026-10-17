# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/reports/generate_kqs_retake_planning_report.py
Purpose: 8주차 멀티비전 키포인트 품질 평가(KQS) 및 재촬영/보류 게이팅 상세 기획 보고서 자동 생성기
         - 출력 Word 문서: D:\자주프 문서\8주차\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.docx
         - 출력 참조 MD:  D:\자주프 문서\8주차\md\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.md
"""

import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = r"D:\자주프 문서\8주차\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.docx"
MD_OUT = r"D:\자주프 문서\8주차\md\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="2B579A"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_callout_box(doc, text_list, bg_hex="F0F4F8", border_color="2B579A"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    c.width = Inches(6.7)
    set_cell_background(c, bg_hex)
    
    tcPr = c._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = c.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    for i, t in enumerate(text_list):
        if i > 0:
            p = c.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
        run = p.add_run(t)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def generate_reports():
    os.makedirs(os.path.dirname(DOCX_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)

    doc = Document()

    # 페이지 여백 설정 (0.85인치)
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # 1. 문서 헤더 / 제목
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("[vidio-health-trainer] 8주차 공식 기술 기획 보고서")
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(21)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(43, 87, 154)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(12)
    run_sub = sub_p.add_run("멀티비전(MediaPipe + RTMPose) 키포인트 품질 지수(KQS) 산출 및 재촬영/보류 게이팅 상세 기획서")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(13)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(100, 116, 139)

    # 메타데이터 표
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("공식 프로젝트명", "vidio-health-trainer (AI 기반 웨이트 트레이닝 자세 진단 및 코칭 시스템)"),
        ("문서 분류 / 작성 주차", "8주차 공식 기술 기획서 (아키텍처 확장 및 데이터 파이프라인 무결성 강화)"),
        ("작성 기준 일시", "2026년 10월 17일 (토요일) - 8주차 세션 산출물"),
        ("배포 파일 경로", "D:\\자주프 문서\\8주차\\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.docx"),
        ("참조 마크다운 경로", "D:\\자주프 문서\\8주차\\md\\8주차_키포인트품질(KQS)_및_재촬영게이팅_상세기획보고서.md")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_k = row.cells[0]
        cell_v = row.cells[1]
        cell_k.width = Inches(1.8)
        cell_v.width = Inches(4.9)
        set_cell_background(cell_k, "F8FAFC")
        pk = cell_k.paragraphs[0]
        pk.paragraph_format.space_before = Pt(2)
        pk.paragraph_format.space_after = Pt(2)
        rk = pk.add_run(k)
        rk.font.name = "맑은 고딕"
        rk.font.size = Pt(9.5)
        rk.font.bold = True
        rk.font.color.rgb = RGBColor(71, 85, 105)

        pv = cell_v.paragraphs[0]
        pv.paragraph_format.space_before = Pt(2)
        pv.paragraph_format.space_after = Pt(2)
        rv = pv.add_run(v)
        rv.font.name = "맑은 고딕"
        rv.font.size = Pt(9.5)
        rv.font.color.rgb = RGBColor(15, 23, 42)
    set_table_borders(meta_table)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Executive Summary 박스
    add_callout_box(doc, [
        "■ [Executive Summary] 기획안 핵심 요약",
        "1. 기획 배경: 사용자가 직접 촬영한 일반화 영상(스마트폰 1인칭/3인칭, 헬스장 환경) 투입 시, 신체 가림(바벨 원판, 조명 음영)이나 화각 이탈로 인한 2단계 결함 분류기 오진단(False Positive)을 원천 차단하기 위해 키포인트 품질 평가 및 재촬영 게이팅을 설계함.",
        "2. 멀티비전 관절 품질 지수(KQS): MediaPipe의 가시성(visibility)과 RTMPose의 히트맵 사후 신뢰도(confidence)를 융합하고, 운동 종목별 핵심 관절 가중치 및 연속 결손 페널티(Consecutive Occlusion Penalty)를 적용하여 정밀 점수화함.",
        "3. 2-Tier 다단계 게이팅: [Tier 1 전역 비디오 게이팅]으로 식별 불가 영상을 입구 차단하고, [Tier 2 횟수 단위 게이팅]으로 가림이 발생한 특정 횟수만 선별적으로 결함 추론을 바이패스(Bypass)하여 즉시 맞춤형 재촬영 안내를 제공함.",
        "4. 타깃형 재촬영 딕셔너리: 단순 '재촬영' 통보를 넘어, 가림 부위(발목/무릎/골반/블러)에 따라 사용자가 즉각 수정 가능한 3대 물리적 조치(거리 확보, 원판 각도 조절, 조명 개선)를 명시적으로 안내함."
    ], bg_hex="F0F7FF", border_color="2B579A")

    def add_section_header(title, level=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(title)
        run.font.name = "맑은 고딕"
        run.font.bold = True
        if level == 1:
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(43, 87, 154)
        elif level == 2:
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(30, 41, 59)
        else:
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(71, 85, 105)

    def add_body_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(51, 65, 85)
        return p

    # -------------------------------------------------------------------------
    # 1. 기획 배경 및 문제 정의
    # -------------------------------------------------------------------------
    add_section_header("1. 기획 배경 및 문제 정의 (Background & Problem Definition)")
    add_body_p(
        "사용자가 직접 촬영하는 일반화 영상(스마트폰 촬영)은 AI-Hub 스튜디오 정제 데이터와 달리 "
        "다양한 외부 변수(바벨 원판에 의한 무릎 가림, 카메라 화각 이탈로 인한 발목 짤림, 체육관 조명에 따른 역광, 동적 모션 블러)를 필연적으로 포함합니다."
    )
    add_body_p(
        "만약 비전 추출기(MediaPipe / RTMPose)가 관절 위치를 잘못 추정하거나 소실된 상태에서 "
        "이를 그대로 2단계 머신러닝 결함 진단 모델(XGBoost, Random Forest)에 입력하면, "
        "정상적인 자세로 운동을 수행했음에도 왜곡된 관절 각도로 인해 '무릎 외반 98%', '척추 굴곡 결함' 등 치명적인 오진단(False Positive)을 내리게 됩니다."
    )
    add_body_p(
        "따라서 연구 계획서 §3.2 및 §3.3에 명시된 '모델 신뢰도 및 MediaPipe visibility 기준 미달 시 자세 교정 대신 재촬영 안내를 출력한다'는 원칙을 "
        "단순한 예외 처리가 아닌, 지능형 품질 평가(KQS)와 다단계 게이팅(Gating) 시스템으로 고도화하여 파이프라인의 무결성을 확보하고자 합니다."
    )

    # -------------------------------------------------------------------------
    # 2. 멀티비전 관절 품질 지수 (KQS) 수학적 모델링
    # -------------------------------------------------------------------------
    add_section_header("2. 멀티비전 관절 품질 지수 (Keypoint Quality Score, KQS) 수학적 모델링")
    add_body_p(
        "단순히 전체 관절의 산술 평균을 계산하면 안면 관절(눈, 코, 귀)이 선명하다는 이유로 무릎이나 발목이 가려진 영상을 통과시키는 심각한 오류가 발생합니다. "
        "이를 방지하기 위해 (1) 듀얼 비전 융합, (2) 종목별 핵심 관절 가중치, (3) 시간축 연속 결손 페널티의 3단계 수식을 정립합니다."
    )

    add_section_header("2.1 단일 관절 프레임 신뢰도 융합 수식", level=2)
    add_body_p(
        "MediaPipe의 2D/3D 공간 가시성(visibility)과 RTMPose의 히트맵 사후 신뢰도(confidence)를 융합하여 프레임 t에서 관절 j의 통합 품질 q_t(j)를 정의합니다:"
    )
    add_callout_box(doc, [
        "q_t(j) = α · vis_MP(t, j) + (1 - α) · conf_RTM(t, j)",
        "- vis_MP(t, j) ∈ [0.0, 1.0]: MediaPipe 관측 가시성 (화면 이탈 및 차폐 민감)",
        "- conf_RTM(t, j) ∈ [0.0, 1.0]: RTMPose Halpe26 히트맵 신뢰도 (미세 관절 위치 모호성 민감)",
        "- α: 융합 비율 가중치 (기본값 α = 0.50, RTMPose 비활성화 시 α = 1.00 자동 폴백)"
    ], bg_hex="F8FAFC", border_color="64748B")

    add_section_header("2.2 운동 종목별 핵심 관절 가중치 매트릭스 (W_exercise)", level=2)
    add_body_p(
        "각 운동 종목의 생체역학적 핵심 관측 관절에 가중치를 집중 부여하고, 비관련 관절(안면 등)은 배제합니다:"
    )

    kqs_table = doc.add_table(rows=4, cols=4)
    kqs_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    kqs_headers = ["운동 기능 그룹", "해당 운동 종목", "핵심 평가 관절 (Critical Joints)", "가중치 배분 (W_joint)"]
    for c_idx, h in enumerate(kqs_headers):
        cell = kqs_table.rows[0].cells[c_idx]
        set_cell_background(cell, "2B579A")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(h)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    kqs_rows = [
        ("스쿼트 / 데드리프트 계열", "바벨 스쿼트, 데드리프트, 스티프데드, 굿모닝", "슬관절(좌우 무릎), 족관절/발끝, 요추/골반, 견관절", "무릎 0.35 / 발목 0.30 / 요추 0.25 / 어깨 0.10"),
        ("런지 계열 (단방향/교차)", "바벨 런지, 다이나믹 런지, 사이드 런지, 크로스 런지", "앞무릎, 뒷무릎, 전족부/발목, 골반 수평, 체간 척추", "앞무릎 0.35 / 뒷무릎 0.25 / 발목 0.20 / 골반 0.20"),
        ("공통 비핵심 관절", "전 종목 공통", "안면(코, 눈, 귀), 손목, 전완", "가중치 0.00 (품질 점수 산정에서 완전 제외)")
    ]
    for r_idx, row_data in enumerate(kqs_rows, start=1):
        row = kqs_table.rows[r_idx]
        bg = "FFFFFF" if r_idx % 2 != 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(51, 65, 85)
    set_table_borders(kqs_table)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_section_header("2.3 시간축 연속 결손 페널티 (Consecutive Occlusion Penalty)", level=2)
    add_body_p(
        "동작 수행 중 1~2프레임의 순간적 떨림은 3차 스플라인으로 복원할 수 있으나, "
        "운동의 최하단(Bottom) 또는 전환 지점에서 연속으로 8프레임(약 0.25초 이상) 관절이 가려지면 치명적인 정보 손실이 발생합니다. 이를 감점하는 공식을 적용합니다:"
    )
    add_callout_box(doc, [
        "KQS_rep = ( 1/64 · ∑_{t=1}^{64} Q_t ) × [ 1.0 - StreakPenalty ]",
        "- Q_t = ∑_{j ∈ Critical} w_j · q_t(j): 프레임 t의 핵심 관절 가중 품질",
        "- StreakPenalty = min(0.50, (max(L_drop) / 64) × 2.0)",
        "- L_drop: Q_t < 0.40 상태로 지속된 최대 연속 프레임 길이"
    ], bg_hex="F8FAFC", border_color="64748B")

    # -------------------------------------------------------------------------
    # 3. 2-Tier 다단계 게이팅 파이프라인 아키텍처
    # -------------------------------------------------------------------------
    add_section_header("3. 2-Tier 다단계 게이팅 파이프라인 아키텍처 (Gating Architecture)")
    add_body_p(
        "사용자의 컴퓨팅 자원 낭비를 줄이고 사용자 경험(UX)을 극대화하기 위해, 게이팅을 '비디오 전역 게이팅'과 '단일 횟수 게이팅'의 2단계로 분리 운영합니다."
    )

    tier_table = doc.add_table(rows=3, cols=4)
    tier_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tier_headers = ["게이팅 계층", "진단 시점", "트리거 조건", "수행 조치 및 반환 결과"]
    for c_idx, h in enumerate(tier_headers):
        cell = tier_table.rows[0].cells[c_idx]
        set_cell_background(cell, "2B579A")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(h)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    tier_rows = [
        ("Tier 1: 비디오 전역 게이팅 (Video Global Gate)", "포즈 추출 직후 (횟수 분할 전)", "비디오 전체 평균 KQS < 0.40 또는 사람 미검출 프레임 비율 > 30%", "횟수 분할 및 2단계 모델 추론 전면 중단. 즉시 '전신 화각 이탈 전역 재촬영' 에러 반환 (불필요한 GPU/CPU 연산 원천 차단)"),
        ("Tier 2: 단일 횟수 게이팅 (Repetition Local Gate)", "개별 횟수(64F) 슬라이싱 직후", "해당 횟수 KQS_rep < 0.60 또는 특정 핵심 관절 평균 신뢰도 < 0.35", "해당 회차만 2단계 결함 추론 바이패스(Bypass). '판정 보류' 상태 지정 및 가림 부위 맞춤형 재촬영 피드백 큐를 최우선 노출 (정상 회차는 분석 보존)")
    ]
    for r_idx, row_data in enumerate(tier_rows, start=1):
        row = tier_table.rows[r_idx]
        bg = "FFFFFF" if r_idx % 2 != 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(51, 65, 85)
    set_table_borders(tier_table)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # 4. 결함 부위 타깃형 재촬영 코칭 딕셔너리
    # -------------------------------------------------------------------------
    add_section_header("4. 결함 부위 타깃형 재촬영 코칭 딕셔너리 (RETAKE_PROMPT_CATALOG)")
    add_body_p(
        "사용자에게 '인식 실패로 재촬영하세요'라는 무의미한 에러 메시지를 제공하지 않고, "
        "품질 저하가 발생한 특정 신체 분절을 파악하여 즉시 촬영 환경을 교정할 수 있는 1문장 물리적 행동 지침을 제공합니다."
    )

    prompt_table = doc.add_table(rows=5, cols=4)
    prompt_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    prompt_headers = ["결손 분류 코드", "취약 관절 부위", "주요 발생 원인", "타깃형 재촬영 행동 지침 큐 (Action Cue)"]
    for c_idx, h in enumerate(prompt_headers):
        cell = prompt_table.rows[0].cells[c_idx]
        set_cell_background(cell, "2B579A")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(h)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    prompt_rows = [
        ("OCC_FEET", "하지 족관절 / 발끝", "스마트폰 각도가 높아 발목이 화면 하단 밖으로 벗어남", "발목과 발바닥이 화면 하단 밖으로 벗어났습니다. 카메라 각도를 살짝 아래로 내리거나 스마트폰을 1m 뒤로 물려 발끝 지면 접지면이 모두 보이도록 재촬영해 주세요."),
        ("OCC_KNEE", "슬관절 (좌/우 무릎)", "화면 각도 또는 장애물에 의해 무릎 관절이 가려짐", "무릎 관절이 화면 또는 물체에 가려져 정밀 분석이 어렵습니다. 무릎이 온전히 보이도록 카메라 각도와 위치를 조절하여 재촬영해 주세요."),
        ("OCC_SPINE", "요추 / 골반 / 체간", "헐렁한 상의 착용, 체육관 역광, 또는 바벨에 체간이 가려짐", "골반 및 척추선 인식이 불완전합니다. 상체 윤곽이 드러나는 복장을 착용하거나 조명이 피사체를 정면으로 비추는 환경에서 재촬영해 주세요."),
        ("OCC_BLUR", "전신 동적 블러 / 흔들림", "스마트폰 손떨림 또는 급격한 움직임으로 인한 셔터스피드 부족", "카메라 흔들림이나 모션 블러로 관절 신뢰도가 낮습니다. 스마트폰을 삼각대나 거치대에 고정하고 템포를 통제하며 재촬영해 주세요.")
    ]
    for r_idx, row_data in enumerate(prompt_rows, start=1):
        row = prompt_table.rows[r_idx]
        bg = "FFFFFF" if r_idx % 2 != 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(51, 65, 85)
    set_table_borders(prompt_table)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # 5. 데이터 스키마 및 파이프라인 연동 설계
    # -------------------------------------------------------------------------
    add_section_header("5. 데이터 스키마 및 파이프라인 연동 설계 (Interface Integration)")
    add_body_p(
        "KQS 지수 및 재촬영 안내를 기존 웹 서비스와 하위 호환성을 유지하며 매끄럽게 연결하기 위해 "
        "schemas.py, feedback_engine.py, video_processor.py의 인터페이스를 다음과 같이 확장 설계합니다."
    )

    add_section_header("5.1 schemas.py 스키마 확장 명세", level=2)
    add_callout_box(doc, [
        "class RepetitionDetail(BaseModel):",
        "    ...",
        "    kqs_score: float = Field(default=1.0, description='키포인트 가중 품질 지수 (0.0~1.0)')",
        "    is_retake_required: bool = Field(default=False, description='신뢰도 미달 재촬영 필요 여부')",
        "    occluded_joints: List[str] = Field(default_factory=list, description='가림/소실된 주요 관절 목록')",
        "    retake_cue: Optional[str] = Field(default=None, description='타깃형 재촬영 행동 지침 큐')"
    ], bg_hex="F8FAFC", border_color="2B579A")

    add_section_header("5.2 video_processor.py 파이프라인 연동 시퀀스", level=2)
    add_body_p(
        "video_processor.py 내부에서 1회 반복 64프레임 생성 직후, KQS를 먼저 평가하여 분기합니다:"
    )
    add_callout_box(doc, [
        "[파이프라인 흐름]",
        "1. 64F 스플라인 관절 시퀀스 (kps_seq) 및 가시성/신뢰도 배열 (vis_seq, conf_seq) 확보",
        "2. kqs_score, occluded_list = evaluate_keypoint_quality(kps_seq, vis_seq, conf_seq, exercise_id)",
        "3. IF kqs_score < 0.60: (품질 미달 발생)",
        "       retake_info = get_retake_feedback(occluded_list, camera_view)",
        "       -> 2단계 결함 분류기 추론 생략 (Bypass), RepetitionDetail에 retake_info 탑재 후 다음 횟수로 이동",
        "   ELSE: (품질 정상)",
        "       -> 정상적으로 2단계 세부 결함 추론(evaluate_single_repetition) 및 Tier A/B/C 코칭 생성 진행"
    ], bg_hex="F0F4F8", border_color="10B981")

    # -------------------------------------------------------------------------
    # 6. 향후 구현 및 검증 로드맵
    # -------------------------------------------------------------------------
    add_section_header("6. 향후 구현 및 검증 로드맵 (Action Items)")
    add_body_p(
        "사용자께서 직접 일반화 영상을 촬영하여 시스템에 투입하기 전, 아래의 순서로 단계적 개발 및 검증을 완료할 예정입니다:"
    )
    add_body_p(
        "1단계: FeedbackEngine 및 VideoAnalysisPipeline 내 KQS 산출 모듈 및 딕셔너리 구현\n"
        "2단계: 가상 가림(Synthetic Occlusion) 노이즈 주입 하네스 단위 테스트(test_kqs_gating.py) 작성 및 100% 바이패스 검증\n"
        "3단계: 사용자 촬영 일반화 영상 투입 시 실시간 재촬영 가이드 동작 및 결함 오탐 방지 실증"
    )

    doc.save(DOCX_OUT)
    print(f"[SUCCESS] Word Report Generated: {DOCX_OUT}")

    # MD 파일 동시 생성
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("# [vidio-health-trainer] 8주차 공식 기술 기획 보고서\n\n")
        f.write("## 멀티비전(MediaPipe + RTMPose) 키포인트 품질 지수(KQS) 산출 및 재촬영/보류 게이팅 상세 기획서\n\n")
        f.write("- **공식 프로젝트명**: `vidio-health-trainer`\n")
        f.write("- **문서 분류**: 8주차 공식 기술 기획서 (아키텍처 확장 및 데이터 파이프라인 무결성 강화)\n")
        f.write("- **작성 일시**: 2026년 10월 17일 (토)\n")
        f.write(f"- **Word 배포 파일**: `{DOCX_OUT}`\n\n")
        f.write("---\n\n")
        f.write("### ■ [Executive Summary] 기획안 핵심 요약\n\n")
        f.write("1. **기획 배경**: 일반화 영상(스마트폰 1인칭/3인칭, 헬스장 환경) 투입 시, 신체 가림(바벨 원판, 조명 음영)이나 화각 이탈로 인한 2단계 결함 분류기 오진단(False Positive)을 원천 차단하기 위해 키포인트 품질 평가 및 재촬영 게이팅을 설계함.\n")
        f.write("2. **멀티비전 관절 품질 지수(KQS)**: MediaPipe의 가시성(visibility)과 RTMPose의 히트맵 사후 신뢰도(confidence)를 융합하고, 운동 종목별 핵심 관절 가중치 및 연속 결손 페널티를 적용하여 정밀 점수화함.\n")
        f.write("3. **2-Tier 다단계 게이팅**: [Tier 1 전역 비디오 게이팅]으로 식별 불가 영상을 입구 차단하고, [Tier 2 횟수 단위 게이팅]으로 가림이 발생한 특정 횟수만 선별적으로 결함 추론을 바이패스(Bypass)하여 즉시 맞춤형 재촬영 안내를 제공함.\n")
        f.write("4. **타깃형 재촬영 딕셔너리**: 단순 '재촬영' 통보를 넘어, 가림 부위(발목/무릎/골반/블러)에 따라 사용자가 즉각 수정 가능한 3대 물리적 조치(거리 확보, 원판 각도 조절, 조명 개선)를 명시적으로 안내함.\n\n")
        f.write("---\n\n")
        f.write("### 1. 기획 배경 및 문제 정의\n\n")
        f.write("사용자가 직접 촬영하는 일반화 영상은 바벨 원판에 의한 무릎 가림, 카메라 화각 이탈로 인한 발목 짤림 등을 필연적으로 포함합니다. "
                "이를 그대로 2단계 머신러닝 모델에 투입하면 정상 자세임에도 '무릎 외반 98%' 등의 오진단을 내리게 됩니다. "
                "따라서 연구 계획서 §3.2 및 §3.3 명세에 따라 품질 미달 시 결함 추론을 차단하고 재촬영으로 직결시키는 시스템이 필수적입니다.\n\n")
        f.write("---\n\n")
        f.write("### 2. 멀티비전 관절 품질 지수 (KQS) 수식\n\n")
        f.write("$$q_t(j) = \\alpha \\cdot vis_{MP}(t, j) + (1 - \\alpha) \\cdot conf_{RTM}(t, j)$$\n\n")
        f.write("$$KQS_{rep} = \\left( \\frac{1}{64} \\sum_{t=1}^{64} \\sum_{j \\in Critical} w_j \\cdot q_t(j) \\right) \\times [ 1.0 - StreakPenalty ]$$\n\n")
        f.write("- **핵심 관절 가중치**: 스쿼트/데드 기준 슬관절 0.35, 족관절 0.30, 요추 0.25, 어깨 0.10 (안면 관절 0.00 제외)\n")
        f.write("- **StreakPenalty**: 최하단 구간 등에서 연속 결손 프레임이 길어질수록 지수적 감점 적용\n\n")
        f.write("---\n\n")
        f.write("### 3. 2-Tier 다단계 게이팅 파이프라인\n\n")
        f.write("| 게이팅 계층 | 진단 시점 | 트리거 조건 | 수행 조치 및 반환 결과 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write("| **Tier 1: 비디오 전역 게이팅** | 포즈 추출 직후 | 전체 평균 KQS < 0.40 또는 사람 미검출 > 30% | 횟수 분할 및 2단계 추론 전면 중단. '전신 화각 이탈 전역 재촬영' 에러 반환 |\n")
        f.write("| **Tier 2: 단일 횟수 게이팅** | 1회 반복 슬라이싱 직후 | KQS_rep < 0.60 또는 핵심 관절 신뢰도 < 0.35 | 해당 회차 2단계 결함 추론 바이패스. '판정 보류' 상태 및 가림 부위 타깃 재촬영 큐 출력 |\n\n")
        f.write("---\n\n")
        f.write("### 4. 타깃형 재촬영 코칭 딕셔너리 (RETAKE_PROMPT_CATALOG)\n\n")
        f.write("1. **발목/발바닥 결손 (`OCC_FEET`)**: *\"발목과 발바닥이 화면 하단 밖으로 벗어났습니다. 카메라 각도를 살짝 아래로 내리거나 스마트폰을 1m 뒤로 물려 발끝 지면 접지면이 모두 보이도록 재촬영해 주세요.\"*\n")
        f.write("2. **슬관절(무릎) 결손 (`OCC_KNEE`)**: *\"무릎 관절이 화면 또는 물체에 가려져 정밀 분석이 어렵습니다. 무릎이 온전히 보이도록 카메라 각도와 위치를 조절하여 재촬영해 주세요.\"*\n")
        f.write("3. **골반/요추 결손 (`OCC_SPINE`)**: *\"골반 및 척추선 인식이 불완전합니다. 상체 윤곽이 드러나는 복장을 착용하거나 조명이 피사체를 정면으로 비추는 환경에서 재촬영해 주세요.\"*\n")
        f.write("4. **전신 동적 블러 (`OCC_BLUR`)**: *\"카메라 흔들림이나 모션 블러로 관절 신뢰도가 낮습니다. 스마트폰을 삼각대나 거치대에 고정하고 템포를 통제하며 재촬영해 주세요.\"*\n\n")
        f.write("---\n\n")
        f.write("### 5. 데이터 스키마 및 파이프라인 연동 설계\n\n")
        f.write("- `RepetitionDetail`에 `kqs_score`, `is_retake_required`, `occluded_joints`, `retake_cue` 필드 확장.\n")
        f.write("- `video_processor.py`에서 2단계 추론 진입 전 KQS < 0.60 판정 시 2단계 추론 생략 및 재촬영 큐 즉시 직결.\n")
    print(f"[SUCCESS] Markdown Report Generated: {MD_OUT}")

if __name__ == '__main__':
    generate_reports()
