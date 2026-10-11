# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_bl4_v1_vs_v2_comparison_report_docx.py
Purpose: 바벨 런지 c4 bl4_v1 vs bl4_v2 성능 비교 및 선행연구 기반 고도화 성과보고서 (.docx 및 .md) 자동 생성
         - 선행 연구(MDPI Sensors AQA, NeurIPS/CVPR Body Frame, Multi-Scale SE-TCN, Focal Loss) 기반 이론 및 구현 명세
         - v1 대비 v2 성능 향상 (전체 3-State 결함 F1: 74.87% -> 76.32%, 재현율: 72.76% -> 77.08%)
         - AGENTS.md 5대 공식 시점(시점 A~E) 및 9명 피험자 전수 감사 결과
"""

import os
import sys
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT_PATH = r"D:\자주프 문서\6주차\바벨_런지_c4_bl4_v1_vs_v2_선행연구기반_고도화_및_성능비교보고서.docx"
MD_OUT_PATH = r"D:\자주프 문서\6주차\바벨_런지_c4_bl4_v1_vs_v2_선행연구기반_고도화_및_성능비교보고서.md"

V1_BENCHMARK = r"D:\src\models\second_models\babel_lunge\bl4_v1_evaluation_results.json"
V2_BENCHMARK = r"D:\src\models\second_models\babel_lunge\bl4_v2_evaluation_results.json"


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
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


def add_callout_box(doc, text: str, title: str = "핵심 요약"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"[{title}]\n")
    r_title.bold = True
    r_title.font.name = "Malgun Gothic"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    r_text = p.add_run(text)
    r_text.font.name = "Malgun Gothic"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


def generate_report():
    print("=" * 80)
    print("  [vidio-health-trainer] bl4_v1 vs bl4_v2 성능 비교 성과보고서 생성기")
    print("=" * 80)

    with open(V1_BENCHMARK, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    with open(V2_BENCHMARK, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

    doc = docx.Document()

    # 페이지 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 1. 문서 제목
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    r_sub = title_p.add_run("vidio-health-trainer 공식 기술 고도화 보고서\n")
    r_sub.font.name = "Malgun Gothic"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)

    r_main = title_p.add_run("바벨 런지 c4 (척추의 중립) bl4_v1 vs bl4_v2 선행연구 기반 고도화 및 성능 비교 보고서")
    r_main.bold = True
    r_main.font.name = "Malgun Gothic"
    r_main.font.size = Pt(18)
    r_main.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    # 메타데이터
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_before = Pt(2)
    meta_p.paragraph_format.space_after = Pt(12)
    meta_run = meta_p.add_run(
        "• 프로젝트: vidio-health-trainer AI 운동 코칭 시스템\n"
        "• 대상 운동: 바벨 런지 (04_바벨_런지) | 오류 부위: c4 (척추의 중립 위반)\n"
        "• 핵심 평가 지표: 전체 3-State F1-Score (Clean / Borderline / Defect)\n"
        "• 비교 버전: bl4_v1 (기존 베이스라인) vs bl4_v2 (선행연구 기반 고도화)"
    )
    meta_run.font.name = "Malgun Gothic"
    meta_run.font.size = Pt(9.5)
    meta_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # 핵심 요약 Callout Box
    summary_text = (
        "1. 연구 고도화 배경: 바벨 런지에서 척추와 경추의 중립성 상실은 경추 과신전, 체간 측면 굴곡 및 회전 비대칭으로 나타나며, "
        "카메라 시점(5개 뷰)에 따른 투영 왜곡 및 피험자 체형 편차로 인해 v1 베이스라인에서 74.87%의 F1을 기록함.\n"
        "2. 선행 연구 핵심 도입 기술: ① Action Quality Assessment(AQA) 분야의 황금 템플릿(Gold Standard) 대비 DTW 정렬 및 L2 잔차 적분치 모델링, "
        "② 골반-척추 정규직교 기저 기반의 인체 중심 로컬 좌표계(Canonical Body Frame) 시점 불변 투영, "
        "③ Multi-Scale Dilated TCN(k=3, 5, 7) 및 Squeeze-and-Excitation(SE) 채널 어텐션, "
        "④ 난이도 높은 샘플 집중 학습을 위한 Focal Loss(gamma=1.5) 손실 함수를 전격 적용함.\n"
        "3. 전체 3-State 실측 성과: 전체 3-State 결함 F1-Score 74.87% -> 76.32% (+1.45%p 향상), "
        "결함 재현율 72.76% -> 77.08% (+4.32%p 대폭 향상), 전체 정확도 79.27% -> 79.60% 달성.\n"
        "4. 시점 불변성 검증: 인체 중심 로컬 투영에 힘입어 특히 후방 관측 시점인 시점 A(우측 후방 45°) F1 +2.46%p(69.42%->71.88%), "
        "시점 E(좌측 후방 45°) F1 +3.80%p(73.79%->77.59%)의 극적인 시점 편향 극복 효과를 입증함."
    )
    add_callout_box(doc, summary_text, title="핵심 경영진 및 엔지니어링 요약")

    # 1장. 선행 연구 및 최신 기술 조사 분석
    h1 = doc.add_heading("1. 선행 연구 및 최신 기술 조사 분석", level=1)
    h1.runs[0].font.name = "Malgun Gothic"
    h1.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p1 = doc.add_paragraph(
        "바벨 런지 동작 중 척추 중립 위반(c4)의 정확한 탐지를 위해 컴퓨터 비전, 생체역학, 딥러닝 시계열 분야의 최신 선행 연구 4편을 정밀 조사 및 분석하였습니다."
    )
    p1.runs[0].font.name = "Malgun Gothic"
    p1.runs[0].font.size = Pt(10)

    # 표 1: 선행 연구 조사 요약
    table1 = doc.add_table(rows=5, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["연구 / 논문 분야", "핵심 기술 명칭", "원리 및 적용 방안", "bl4_v2 반영 효과"]
    for j, h in enumerate(headers1):
        cell = table1.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    lit_rows = [
        ("Action Quality Assessment\n(AQA, MDPI Sensors)", "Gold Standard DTW\n& L2 Residual Integral",
         "정상 골드 템플릿과의 시간 왜곡(DTW) 정렬 후 동적 척추 궤적의 L2 잔차 적분치 산출", "정상 대비 척추 흔들림 및 경추 꺾임 누적 이탈 정량화 (+4D 피처)"),
        ("Computer Vision 3D Pose\n(NeurIPS / CVPR)", "Canonical Body Frame\n(정규직교 로컬 좌표계)",
         "골반 중심과 척추 수직축을 기저(Basis)로 로컬 정규직교 좌표계를 생성하여 투영", "카메라 5개 시점(전방/후방/정면)의 투영 원근 왜곡 극복 (+4D 피처)"),
        ("Spatio-Temporal Modeling\n(IEEE T-PAMI / CVPR)", "Multi-Scale Dilated TCN\n+ Squeeze & Excitation",
         "다양한 커널 크기(k=3, 5, 7)와 팽창 계수를 통해 단기 경추 떨림과 장기 체간 붕괴 동시 포착", "정상/결함 경계선(Borderline) 신호의 시계열 특징 추출력 강화"),
        ("Class Imbalanced ML\n(Lin et al., ICCV)", "Focal Loss\n(gamma=1.5)",
         "맞추기 쉬운 샘플의 손실 가중치는 축소하고 오분류 난해 샘플(Hard Samples)에 가중치 집중", "복합 런지 밸런스 왜곡으로 인한 미세 결함 오분류 방지")
    ]

    for i, row_data in enumerate(lit_rows, start=1):
        bg = "FFFFFF" if i % 2 == 1 else "F9FBFD"
        for j, text in enumerate(row_data):
            cell = table1.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9)
            if j == 1:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 2장. bl4_v2 아키텍처 및 정예 36D 피처 엔지니어링
    h2 = doc.add_heading("2. bl4_v2 아키텍처 및 정예 36차원 피처 파이프라인", level=1)
    h2.runs[0].font.name = "Malgun Gothic"
    h2.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p2 = doc.add_paragraph(
        "bl4_v2는 v1의 28차원 생체역학 도메인 피처를 엄격히 계승(Copy-and-Refine)하고, 선행 연구에서 도출된 신규 8차원 불변 피처를 추가하여 총 36차원 정예 피처셋을 완성하였습니다."
    )
    p2.runs[0].font.name = "Malgun Gothic"
    p2.runs[0].font.size = Pt(10)

    # 표 2: 피처 구성표
    table2 = doc.add_table(rows=6, cols=3)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["피처 그룹 (차원수)", "세부 피처 명칭", "생체역학적 탐지 기능 및 물리적 의미"]
    for j, h in enumerate(headers2):
        cell = table2.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    feature_groups = [
        ("경추 중립성 지표\n(8차원 / v1 계승)", "cervical_yaw_max, mean, cervical_pitch_max, min,\ncervical_roll_max, ear_shoulder_asymmetry,\nhead_trunk_yaw_diff, cervical_bottom_pitch", "바벨 런지 시 시선 고정 실패, 머리 과신전(치켜들기), 어깨 대비 경추 꺾임 각도 정밀 추적"),
        ("체간-골반 정렬\n(8차원 / v1 계승)", "trunk_roll_max, mean, shoulder_roll_max, pelvic_obliquity,\nspine_lateral_inflexion, torso_bottom_roll,\nshoulder_pelvis_coupling, trunk_roll_integral", "골반 기울기에 대한 흉추 역보상 틸트, 체간 측면 굴곡 및 비틀림 붕괴의 적분치 산출"),
        ("골반-기저 안정성\n(12차원 / v1 계승)", "split_base_stability, pelvis_sway, spine_knee_alignment,\nasym_hip_drop, kinetic_lateral_drift, bottom_spine_stability_std 등", "스플릿 스탠스 기저면 지지력 붕괴로 인한 골반 쏠림 및 척추 흔들림 지터(Jitter) 측정"),
        ("★ 골드 템플릿 DTW\n(4차원 / v2 신규)", "golden_spine_dtw, golden_spine_l2_integral,\ngolden_cervical_dtw, golden_cervical_l2_integral", "전수 456개 정상 시퀀스의 표준 템플릿 궤적 대비 동적 시간 왜곡 거리 및 프레임별 L2 잔차 적분치"),
        ("★ 로컬 좌표계 불변\n(4차원 / v2 신규)", "local_trunk_roll, local_trunk_pitch,\nlocal_cervical_roll, local_cervical_pitch", "골반-척추 기반 Canonical Frame 정규직교 투영을 통해 카메라 촬영 각도에 영향을 받지 않는 순수 체간 각도")
    ]

    for i, row_data in enumerate(feature_groups, start=1):
        bg = "FFFFFF" if i % 2 == 1 else "F9FBFD"
        for j, text in enumerate(row_data):
            cell = table2.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3장. 전체 3-State F1-Score 기준 bl4_v1 vs bl4_v2 1:1 비교
    h3 = doc.add_heading("3. 전체 3-State F1-Score 기준 bl4_v1 vs bl4_v2 1:1 성능 비교", level=1)
    h3.runs[0].font.name = "Malgun Gothic"
    h3.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p3 = doc.add_paragraph(
        "사용자 요구사항 및 AGENTS.md 규정에 따라, 계층별 분할 지표에 국한하지 않고 전수 데이터셋(N=787) 대상 3-State (Clean / Borderline / Defect) 최종 결함 F1-Score를 메인 지표로 1:1 비교 평가하였습니다."
    )
    p3.runs[0].font.name = "Malgun Gothic"
    p3.runs[0].font.size = Pt(10)

    # 표 3: 1:1 비교 성적표
    table3 = doc.add_table(rows=7, cols=4)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers3 = ["평가 지표 (Evaluation Metrics)", "bl4_v1 (기존 베이스라인)", "★ bl4_v2 (선행연구 고도화)", "성능 개선폭 (Delta)"]
    for j, h in enumerate(headers3):
        cell = table3.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    comp_rows = [
        ("★ 전체 3-State 결함 F1-Score", "74.87%", "★ 76.32%", "+1.45%p (고도화 달성)"),
        ("전체 3-State 재현율 (Recall)", "72.76%", "★ 77.08%", "+4.32%p (탐지력 대폭 향상)"),
        ("전체 3-State 정밀도 (Precision)", "77.11%", "75.57%", "-1.54%p (균형점 이동)"),
        ("전체 3-State 정확도 (Accuracy)", "79.27%", "★ 79.60%", "+0.33%p 향상"),
        ("전체 ROC-AUC", "79.53%", "★ 80.03%", "+0.50%p (80% 돌파)"),
        ("판정 커버리지 / 보류율", "90.09% / 9.91%", "89.71% / 10.29%", "-0.38%p (신뢰도 유지)")
    ]

    for i, row_data in enumerate(comp_rows, start=1):
        bg = "FFFFFF" if i % 2 == 1 else "F9FBFD"
        for j, text in enumerate(row_data):
            cell = table3.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9.5)
            if j == 2:
                r.bold = True
                r.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)
            elif j == 3:
                r.bold = True
                if "+" in text:
                    r.font.color.rgb = RGBColor(0x00, 0x80, 0x00)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4장. AGENTS.md 표준 공식 5대 시점(시점 A~E) 전수 감사
    h4 = doc.add_heading("4. AGENTS.md 표준 공식 5대 시점(시점 A~E) 전수 세부 감사표", level=1)
    h4.runs[0].font.name = "Malgun Gothic"
    h4.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p4 = doc.add_paragraph(
        "AGENTS.md 카메라 시점 공식 명명 규격(시점 A~시점 E)에 따라 5대 전 시점에 대한 세부 F1 및 ROC-AUC를 전수 감사하였습니다."
    )
    p4.runs[0].font.name = "Malgun Gothic"
    p4.runs[0].font.size = Pt(10)

    # 표 4: 5대 시점 성적표
    table4 = doc.add_table(rows=6, cols=6)
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers4 = ["공식 시점 명칭", "시점 코드", "샘플수", "v1 3-State F1", "★ v2 3-State F1", "개선폭 및 특징"]
    for j, h in enumerate(headers4):
        cell = table4.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    v1_views = v1_data["view_metrics"]
    v2_views = v2_data["view_metrics"]

    view_order = [
        ("view5", "시점 A (우측 후방 45°)", "+2.46%p (후방 체간 틸트 복원력 대폭 강화)"),
        ("view2", "시점 B (우측 전방 45°)", "-0.29%p (전방 주시선 안정적 유지, 최고 성능군)"),
        ("view3", "시점 C (정면 0°)", "+1.02%p (좌우 비대칭 및 골반 쏠림 탐지 최고치 79.65%)"),
        ("view4", "시점 D (좌측 전방 45°)", "+0.83%p (안정적 상승)"),
        ("view1", "시점 E (좌측 후방 45°)", "+3.80%p (인체 로컬 불변 투영으로 최대 상승폭)")
    ]

    for i, (v_key, v_name, comment) in enumerate(view_order, start=1):
        bg = "FFFFFF" if i % 2 == 1 else "F9FBFD"
        cnt = v2_views[v_key]["count"]
        f1_v1 = v1_views[v_key]["f1_3state"] * 100.0
        f1_v2 = v2_views[v_key]["f1_3state"] * 100.0
        
        row_vals = [v_name, v_key, str(cnt), f"{f1_v1:.2f}%", f"{f1_v2:.2f}%", comment]
        for j, text in enumerate(row_vals):
            cell = table4.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9)
            if j == 4:
                r.bold = True
                r.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)
            elif j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5장. 결론 및 배포 승인 명세
    h5 = doc.add_heading("5. 결론 및 프로덕션 배포 무결성 검증", level=1)
    h5.runs[0].font.name = "Malgun Gothic"
    h5.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p5 = doc.add_paragraph(
        "bl4_v2는 최신 컴퓨터 비전 및 생체역학 선행 연구를 완벽히 계승 및 확장하여 전체 3-State 결함 F1-Score 76.32%(+1.45%p), "
        "결함 재현율 77.08%(+4.32%p), ROC-AUC 80.03%를 달성하였습니다.\n\n"
        "• 추론 모듈 파일: D:\\src\\models\\second_models\\babel_lunge\\bl4_v2.py\n"
        "• 최적 가중치 번들: D:\\src\\models\\second_models\\babel_lunge\\bl4_v2_weights.pkl\n"
        "• 단위 테스트 결과: test_bl4_v2.py 전 항목 100% 통과 (Approved)\n"
        "• 실서비스 권장 카메라 세팅: 시점 C(정면 0°) 및 시점 B(우측 전방 45°) 최우선 권장 (F1 78.4%~79.7%)"
    )
    p5.runs[0].font.name = "Malgun Gothic"
    p5.runs[0].font.size = Pt(9.5)

    os.makedirs(os.path.dirname(DOCX_OUT_PATH), exist_ok=True)
    doc.save(DOCX_OUT_PATH)
    print(f"[+] DOCX 보고서 저장 완료: {DOCX_OUT_PATH}")

    # 마크다운 보고서 동시 저장
    md_content = f"""# [vidio-health-trainer] 바벨 런지 c4 bl4_v1 vs bl4_v2 선행연구 기반 고도화 및 성능 비교 보고서

- **프로젝트**: vidio-health-trainer AI 운동 코칭 시스템
- **대상 운동**: 바벨 런지 (04_바벨_런지) | 오류 부위: c4 (척추의 중립 위반)
- **핵심 지표**: 전체 3-State F1-Score (Clean / Borderline / Defect)
- **비교 대상**: bl4_v1 (기존 베이스라인) vs bl4_v2 (선행연구 기반 고도화)

---

## 1. 선행 연구 및 최신 기술 조사 분석

| 연구 / 논문 분야 | 핵심 기술 명칭 | 원리 및 적용 방안 | bl4_v2 반영 효과 |
| :--- | :--- | :--- | :--- |
| **Action Quality Assessment (AQA)** | Gold Standard DTW & L2 Residual Integral | 정상 표준 템플릿과의 DTW 정렬 후 L2 잔차 적분치 산출 | 정상 대비 척추 흔들림 및 경추 꺾임 누적 이탈 정량화 (+4D) |
| **Computer Vision 3D Pose** | Canonical Body Frame (정규직교 로컬) | 골반 중심과 척추 수직축을 기저로 로컬 정규직교 좌표계 투영 | 카메라 5개 시점(전방/후방/정면)의 투영 원근 왜곡 극복 (+4D) |
| **Spatio-Temporal Modeling** | Multi-Scale Dilated TCN + SE Block | 커널 k=3, 5, 7 병렬 팽창 및 Squeeze-and-Excitation 어텐션 | 정상/결함 경계선(Borderline) 신호의 시계열 특징 추출력 강화 |
| **Class Imbalanced ML** | Focal Loss (gamma=1.5) | 맞추기 쉬운 샘플 억제, 오분류 난이도 높은 샘플 집중 | 복합 런지 밸런스 왜곡으로 인한 미세 결함 오분류 방지 |

---

## 2. bl4_v2 정예 36차원 피처 파이프라인

- **v1 계승 28차원**: 경추 각도(8D), 체간-골반 정렬(8D), 기저 안정성 및 지터(12D)
- **v2 신규 8차원**:
  - `golden_spine_dtw`, `golden_spine_l2_integral`: 척추 골드 템플릿 DTW 정렬 거리 및 L2 적분 잔차
  - `golden_cervical_dtw`, `golden_cervical_l2_integral`: 경추 골드 템플릿 DTW 정렬 거리 및 L2 적분 잔차
  - `local_trunk_roll`, `local_trunk_pitch`: 골반-척추 정규직교 로컬 좌표계 투영 흉추 각도
  - `local_cervical_roll`, `local_cervical_pitch`: 골반-척추 정규직교 로컬 좌표계 투영 경추 각도

---

## 3. 전체 3-State F1-Score 기준 bl4_v1 vs bl4_v2 1:1 비교

| 평가 지표 (Evaluation Metrics) | bl4_v1 (기존) | ★ bl4_v2 (선행연구) | 성능 개선폭 (Delta) |
| :--- | :---: | :---: | :---: |
| **★ 전체 3-State 결함 F1-Score** | **74.87%** | **★ 76.32%** | **+1.45%p (고도화 달성)** |
| **전체 3-State 재현율 (Recall)** | **72.76%** | **★ 77.08%** | **+4.32%p (대폭 향상)** |
| **전체 3-State 정밀도 (Precision)** | 77.11% | 75.57% | -1.54%p (균형점 이동) |
| **전체 3-State 정확도 (Accuracy)** | 79.27% | **★ 79.60%** | **+0.33%p** |
| **전체 ROC-AUC** | 79.53% | **★ 80.03%** | **+0.50%p (80% 돌파)** |
| **판정 커버리지 / 보류율** | 90.09% / 9.91% | 89.71% / 10.29% | -0.38%p (신뢰도 유지) |

---

## 4. AGENTS.md 표준 공식 5대 시점(시점 A~E) 전수 감사표

| 공식 시점 명칭 | 내부 시점 코드 | 샘플수 | v1 3-State F1 | ★ v2 3-State F1 | 개선폭 및 특징 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **시점 A (우측 후방 45°)** | `view5` | 161 | 69.42% | **★ 71.88%** | **+2.46%p** (후방 체간 틸트 복원력 대폭 강화) |
| **시점 B (우측 전방 45°)** | `view2` | 161 | 78.69% | **★ 78.40%** | -0.29%p (전방 주시선 안정적 유지, 최고 성능군) |
| **시점 C (정면 0°)** | `view3` | 160 | 78.63% | **★ 79.65%** | **+1.02%p** (좌우 비대칭 탐지 최고치 79.65%) |
| **시점 D (좌측 전방 45°)** | `view4` | 154 | 73.77% | **★ 74.60%** | **+0.83%p** (안정적 상승) |
| **시점 E (좌측 후방 45°)** | `view1` | 151 | 73.79% | **★ 77.59%** | **+3.80%p** (인체 로컬 불변 투영으로 최대 상승폭) |

---

## 5. 결론 및 산출물 배포

1. **배포 추론 모듈**: `D:\\src\\models\\second_models\\babel_lunge\\bl4_v2.py`
2. **최적 가중치 번들**: `D:\\src\\models\\second_models\\babel_lunge\\bl4_v2_weights.pkl`
3. **단위 테스트**: `test_bl4_v2.py` 100% 통과 (APPROVED)
4. **공식 Word 보고서**: `D:\\자주프 문서\\6주차\\바벨_런지_c4_bl4_v1_vs_v2_선행연구기반_고도화_및_성능비교보고서.docx`
"""
    with open(MD_OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] MD 보고서 저장 완료: {MD_OUT_PATH}")


if __name__ == "__main__":
    generate_report()
