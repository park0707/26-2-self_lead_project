# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c1_v6_report_docx.py
Purpose: 바벨 스쿼트 c1 척추의 중립 v6.0 피처 최적화 소거 실험 및 v5.0 실측 비교 보고서 (.docx, .md) 생성
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

DOCX_PATH = "D:/자주프 문서/5주차/바벨스쿼트_c1_v6_피처최적화_소거실험_및_v5_성능비교보고서.docx"
MD_PATH = "D:/자주프 문서/5주차/바벨스쿼트_c1_v6_피처최적화_소거실험_및_v5_성능비교보고서.md"

LOFO_JSON_PATH = "d:/src/models/second_models/babel_squat/c1_v5_lofo_results.json"
TOURNAMENT_JSON_PATH = "d:/src/models/second_models/babel_squat/c1_v6_tournament_results.json"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'<w:{edge} {nsdecls("w")} w:val="{edge_data.get("val","single")}" w:sz="{edge_data.get("sz","4")}" w:space="0" w:color="{edge_data.get("color","auto")}"/>'
            tcBorders.append(parse_xml(tag))
    tcPr.append(tcBorders)

def add_callout(doc, title, text, bg_color="F0F4F8", border_color="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    set_cell_border(cell, left={"sz": 24, "val": "single", "color": border_color},
                          top={"val": "none"}, right={"val": "none"}, bottom={"val": "none"})
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.bold = True
    run_t.font.name = "Malgun Gothic"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    run_b = p.add_run(text)
    run_b.font.name = "Malgun Gothic"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def format_table_header(row, col_widths, titles, bg_color="1B365D"):
    for idx, (cell, w, title) in enumerate(zip(row.cells, col_widths, titles)):
        cell.width = w
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.bold = True
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(9.0)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

def format_table_row(row, col_widths, values, bg_color="FFFFFF", aligns=None, bold_cols=None):
    if aligns is None:
        aligns = [WD_ALIGN_PARAGRAPH.CENTER] * len(values)
    if bold_cols is None:
        bold_cols = [False] * len(values)
    for idx, (cell, w, val, align, is_bold) in enumerate(zip(row.cells, col_widths, values, aligns, bold_cols)):
        cell.width = w
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        set_cell_border(cell,
                        bottom={"sz": 4, "val": "single", "color": "E2E8F0"},
                        top={"sz": 4, "val": "single", "color": "E2E8F0"})
        p = cell.paragraphs[0]
        p.alignment = align
        run = p.add_run(str(val))
        run.bold = is_bold
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

def main():
    print("[*] c1 v6.0 피처 최적화 및 v5 실측 비교 보고서 생성 시작...")

    with open(LOFO_JSON_PATH, 'r', encoding='utf-8') as f:
        lofo_data = json.load(f)

    with open(TOURNAMENT_JSON_PATH, 'r', encoding='utf-8') as f:
        tour_data = json.load(f)

    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # 1. 메타데이터 및 타이틀
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(2)
    run_meta = p_meta.add_run("vidio-health-trainer | AI-Hub 운동 데이터 2단계 세부 결함 진단 연구 리포트")
    run_meta.font.name = "Malgun Gothic"
    run_meta.font.size = Pt(9)
    run_meta.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(8)
    run_title = p_title.add_run("바벨 스쿼트 c1(척추의 중립) v6.0 피처 최적화 소거 실험 및 v5.0 실측 비교 보고서\n: 34D LOFO 소거 분석, 다중공선성(|r|>=0.80) 해소 및 정예 모델(v6-A 29D / v6-B 21D) 벤치마크")
    run_title.font.name = "Malgun Gothic"
    run_title.font.size = Pt(16)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("작성일: 2026년 9월 | 작성 엔진: vidio-health-trainer 생체역학 파이프라인 | 평가 대상: AI-Hub 9명 전수 (1,906건 및 실무군 1,577건)")
    run_sub.font.name = "Malgun Gothic"
    run_sub.font.size = Pt(9.5)
    run_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    # 1. Executive Summary 박스
    add_callout(
        doc,
        "Executive Summary (핵심 연구 성과 및 발견 요약)",
        "1. 성능 정체의 원인 규명 (피처 과포화 및 분기 희석 현상):\n"
        "   - SpinePose 도입으로 6D를 추가하여 34D로 확장했으나, F1-Score가 77.6% 부근에서 정체됨.\n"
        "   - 원인 1: 척추 저크 피처 2종 중복, 경추 피치각과 흉요추 곡률각 간의 극심한 다중공선성(r=0.920) 발생.\n"
        "   - 원인 2: 트리 앙상블(CatBoost/LightGBM)이 유효 분기 후보(Split candidate)를 탐색할 때 유사 피처들에 가중치가 분산되는 '피처 희석(Feature Dilution)' 현상 발생.\n\n"
        "2. 34D 피처 전수 LOFO(Leave-One-Feature-Out) 소거 실험 성과:\n"
        "   - 34개 피처를 각각 1개씩 배제하며 5-Fold 재학습을 수행한 결과, 제거 시 오히려 F1이 상승하는 '성능 저해 피처' 13종을 정밀 특정함.\n"
        "   - 1위 저해 피처: c1_v4_signed_lumbopelvic_flex_delta (#21, 소거 시 F1 +0.80%p 상승!)\n"
        "   - 2위 저해 피처: c1_v4_bottom_curvature_jerk_max (#23, 소거 시 F1 +0.73%p 상승!)\n"
        "   - 3위 저해 피처: cervical_angle_min (#1, 소거 시 F1 +0.65%p 상승!)\n"
        "   - 4위 저해 피처: c1_v5_cervicothoracic_hinge_pitch (#31, r=0.920 다중공선성, 소거 시 F1 +0.60%p 상승!)\n\n"
        "3. 실측 토너먼트 성과 (v5 대비 대약진):\n"
        "   - [v6-A 정예 모델 (29D, 상위 5대 저해 소거)]:\n"
        "     * 원천 1,906건: Opt F1 80.04% 달성 (마의 80% 벽 최초 돌파!), Cascade F1 78.45% (+0.29%p), 클린 정상 FP 37건 -> 34건 감소.\n"
        "     * Level 1~3 실무군 (1,577건): CatBoost F1 84.96% (+0.69%p), Opt F1 85.96%, ROC-AUC 92.88%.\n"
        "   - [v6-B 컴팩트 모델 (21D, 13대 저해/중복 전수 다이어트, 피처 38.2% 절감)]:\n"
        "     * 원천 1,906건: 기본 0.50 기준 Cascade F1 78.73% (+0.57%p, 역대 최고치 경신!), CatBoost F1 78.41% (+0.78%p).\n"
        "     * Level 1~3 실무군 (1,577건): Opt F1 86.36% (+0.47%p), ROC-AUC 93.09% (93% 고지 돌파!)."
    )

    # 1장
    h1 = doc.add_heading("1. c1 모델 성능 정체 원인 분석: 피처 과포화 및 분기 희석", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph()
    p1.paragraph_format.line_spacing = 1.2
    p1.add_run(
        "SpinePose 9대 척추 분절을 도입하여 6D 피처를 추가함으로써 34D 피처셋을 구축하였으나, 5-Fold 실측 결과 F1-Score가 "
        "기존 v4-A(28D) 대비 큰 폭의 상승 없이 77.6% 부근에서 수렴하는 정체 현상이 관찰되었다. "
        "본 연구진은 이에 대해 세 가지 구조적 원인을 도출하였다:\n\n"
        "1) 정보 중복 및 다중공선성(Multicollinearity)에 의한 트리 분기 희석:\n"
        "   - GBDT(Gradient Boosting Decision Tree) 계열 모델은 매 노드 분할 시 최적의 분기 피처를 그리디(Greedy)하게 탐색한다.\n"
        "   - 유사하거나 상호 상관도가 0.80 이상으로 높은 피처들이 다수 존재할 경우, 트리가 특정 핵심 피처에 가중치를 집중하지 못하고 "
        "유사 피처들 사이에서 분기가 흩어지는 '피처 희석(Feature Dilution)' 현상이 발생한다.\n"
        "2) 2차 미분 각가속도 저크(Jerk) 피처의 이중 중복 잡음:\n"
        "   - v4에서 추가된 2차 미분 곡률 저크(bottom_curvature_jerk_max)와 v5 SpinePose 척추 탄젠트 저크(nadir_lumbar_tangent_jerk)가 "
        "동시에 투입되면서 고주파 미세 진동 노이즈가 과적합(Overfitting)을 유발하였다.\n"
        "3) 경추 및 흉요추 사잇각의 극심한 공선성 (|r| = 0.920):\n"
        "   - v4의 signed_thoracolumbar_nadir와 v5의 cervicothoracic_hinge_pitch 간 상관계수가 0.920에 달해 실질적으로 동일한 물리적 거동을 "
        "두 번 측정함으로써 트리의 깊이를 낭비하고 일반화 성능을 저하시켰다."
    )

    # 2장
    h2 = doc.add_heading("2. 34D 피처 전수 LOFO (Leave-One-Feature-Out) 소거 실험", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    p2 = doc.add_paragraph()
    p2.paragraph_format.line_spacing = 1.2
    p2.add_run(
        "어떤 피처가 성능을 향상시키고 어떤 피처가 저해하는지를 엄밀히 증명하기 위해, 34개 피처 전수를 대상으로 "
        "각 피처를 1개씩 배제한 상태에서 5-Fold StratifiedGroupKFold 완전 격리 교차검증을 수행하는 LOFO 소거 실험을 진행하였다.\n"
        "평가 수식: ΔF1 = F1_LOFO(i) - F1_Baseline (여기서 Baseline 34D의 F1@0.50 = 77.63%, Opt F1 = 79.67%, AUC = 88.21%)\n"
        "만약 ΔF1 > 0 이라면 해당 피처는 모델에 투입되었을 때 오히려 정확도를 깎아먹는 '저해 피처(Harmful Feature)'로 판정된다."
    )

    # LOFO 테이블
    p_lt = doc.add_paragraph()
    p_lt.paragraph_format.space_before = Pt(6)
    p_lt.paragraph_format.space_after = Pt(2)
    run_lt = p_lt.add_run("[표 1] c1 v5 34D 피처 LOFO 소거 실측 결과 및 판정 (원천 데이터 1,906건)")
    run_lt.bold = True
    run_lt.font.size = Pt(9.5)
    run_lt.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    tbl_lofo = doc.add_table(rows=1, cols=6)
    tbl_lofo.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_l = [Inches(0.5), Inches(2.3), Inches(0.9), Inches(0.8), Inches(0.8), Inches(1.2)]
    format_table_header(tbl_lofo.rows[0], widths_l, ["No", "피처명 (Feature Name)", "소거 F1", "Δ F1", "소거 AUC", "판정 및 역할"])

    lofo_items = list(lofo_data['lofo'].items())
    # 정렬: delta_f1 내림차순
    lofo_sorted = sorted(lofo_items, key=lambda x: x[1]['delta_f1'], reverse=True)

    for idx, (fname, info) in enumerate(lofo_sorted, start=1):
        df1 = info['delta_f1']
        f1_val = f"{info['f1_50']:.2f}%"
        df1_val = f"{df1:+5.2f}%p"
        auc_val = f"{info['auc']:.2f}%"

        if df1 >= 0.50:
            role = "★ 핵심 저해 (v6-A 소거)"
            bg = "FEE2E2"
        elif df1 > 0.0:
            role = "▲ 경미 저해 (v6-B 소거)"
            bg = "FFF7ED"
        elif df1 <= -0.50:
            role = "■ 절대 필수 (핵심 보존)"
            bg = "DCFCE7"
        else:
            role = "● 정상 보존 피처"
            bg = "FFFFFF" if idx % 2 == 0 else "F8FAFC"

        row = tbl_lofo.add_row()
        format_table_row(
            row, widths_l,
            [f"#{info['index']:02d}", fname, f1_val, df1_val, auc_val, role],
            bg_color=bg,
            aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER],
            bold_cols=[False, False, False, True, False, True]
        )

    # 3장: 다중공선성 및 상관관계 분석
    h3 = doc.add_heading("3. 상관도 분석을 통한 극심한 다중공선성(|r| >= 0.80) 규명", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    p3 = doc.add_paragraph()
    p3.paragraph_format.line_spacing = 1.2
    p3.add_run(
        "34개 피처 간 피어슨 상관계수(|r|) 행렬을 분석한 결과, 상관계수 0.80 이상의 극심한 다중공선성을 갖는 주요 피처 쌍들이 확인되었다:\n\n"
        "1) ear_sh_hip_min <-> neck_waist_dev (r = -0.977):\n"
        "   - 2D 귀-어깨-골반 각도와 목-허리 편차 벡터는 완벽한 역상관을 형성하여 사실상 동일한 기하학적 양을 중복 측정함.\n"
        "2) c1_v4_signed_thoracolumbar_nadir <-> c1_v5_cervicothoracic_hinge_pitch (r = 0.920):\n"
        "   - v4의 흉요추 사잇각과 v5 SpinePose 경추 피치각은 92.0%의 동일한 정보를 공유하여 트리의 최적 분기를 교란함.\n"
        "3) c1_v4_bottom_curvature_jerk_max <-> c1_v5_nadir_lumbar_tangent_jerk (r = 0.884):\n"
        "   - 2차 미분 곡률 저크와 탄젠트 저크가 중복되어 고주파 미세 흔들림에 과도한 패널티를 부여함.\n"
        "4) c1_v4_signed_lumbopelvic_nadir <-> c1_v4_signed_lumbopelvic_flex_delta (r = 0.865):\n"
        "   - 요추-골반 결합 사잇각과 굴곡 변화량의 중복으로 인해 버트 윙크(Butt Wink) 판정 임계값이 왜곡됨."
    )

    # 4장: v6 모델 아키텍처 및 설계
    h4 = doc.add_heading("4. c1 v6.0 최적화 모델 설계: v6-A (29D) 및 v6-B (21D)", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p4 = doc.add_paragraph()
    p4.paragraph_format.line_spacing = 1.2
    p4.add_run(
        "LOFO 소거 실험 결과와 다중공선성 분석을 종합하여, 두 가지 정예 v6.0 최적화 파이프라인을 구축하였다:\n\n"
        "1) c1 v6-A (29D 정예 챔피언 모델 - Opt F1 80.04% 달성):\n"
        "   - 소거 시 F1 상승폭이 가장 컸던 상위 5대 저해 피처(ΔF1 >= +0.55%p)만 핀셋 소거:\n"
        "     * #01 cervical_angle_min (소거 시 +0.65%p)\n"
        "     * #17 lit_spinal_stiffness (소거 시 +0.58%p)\n"
        "     * #21 c1_v4_signed_lumbopelvic_flex_delta (소거 시 +0.80%p)\n"
        "     * #23 c1_v4_bottom_curvature_jerk_max (소거 시 +0.73%p)\n"
        "     * #31 c1_v5_cervicothoracic_hinge_pitch (소거 시 +0.60%p, r=0.920 해소)\n"
        "   - 목적: 기존 생체역학 정보의 손실을 최소화하면서 노이즈만 제거하여 최고 F1 성능 달성.\n\n"
        "2) c1 v6-B (21D 컴팩트 고효율 모델 - Cascade F1 78.73%, AUC 93.09% 달성):\n"
        "   - 성능 저해 피처(ΔF1 > 0) 및 다중공선성 피처 13종을 전수 소거(피처 수 38.2% 절감):\n"
        "     * 소거 인덱스: [1, 17, 18, 20, 21, 22, 23, 24, 25, 29, 30, 31, 33]\n"
        "     * 핵심 잔존 피처: 2차 곡률 포물선 계수(poly_a), 측방 붕괴각, 시선 피치, 목-허리 변위 등 21종 정예 피처.\n"
        "   - 목적: 모바일 및 엣지 디바이스 구동에 최적화된 초경량 구조이면서도 역대 최고 Cascade F1 달성."
    )

    # 5장: 5-Fold 실측 토너먼트 비교 결과
    h5 = doc.add_heading("5. 5-Fold 교차검증 실측 토너먼트 성능 비교", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)

    # 표 2: 원천 전체 1,906건
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(6)
    p_t2.paragraph_format.space_after = Pt(2)
    run_t2 = p_t2.add_run("[표 2] 원천 데이터셋 전체 (1,906건) 5-Fold 실측 비교")
    run_t2.bold = True
    run_t2.font.size = Pt(9.5)
    run_t2.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    tbl_all = doc.add_table(rows=1, cols=8)
    tbl_all.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_t = [Inches(1.5), Inches(0.5), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8)]
    format_table_header(tbl_all.rows[0], widths_t, ["모델 버전", "차원", "CB F1", "CB Opt", "Casc F1", "Casc Opt", "ROC-AUC", "클린 FP"])

    full_dict = tour_data['full1906']
    rows_all_def = [
        ("v4-A Benchmark", "28D", "v4-A_Benchmark_28D"),
        ("v5-A Full", "34D", "v5-A_Full_34D"),
        ("v6-A Pruned (챔피언)", "29D", "v6-A_Pruned_29D"),
        ("v6-B Compact (경량)", "21D", "v6-B_Compact_21D"),
        ("v6-A Cost-Sensitive", "29D", "v6-A_Cost_29D")
    ]

    for name_label, dim_label, key in rows_all_def:
        m = full_dict[key]
        cb_f1 = f"{m['CatBoost']['f1_50']:.2f}%"
        cb_opt = f"{m['CatBoost']['opt_f1']:.2f}%"
        casc_f1 = f"{m['Cascade']['f1_50']:.2f}%"
        casc_opt = f"{m['Cascade']['opt_f1']:.2f}%"
        auc = f"{m['CatBoost']['auc']:.2f}%"
        fp = f"{m['CatBoost']['clean_fps']}건 ({m['CatBoost']['clean_fpr']:.1f}%)"

        bg = "FFFFFF"
        is_champ = False
        if "v6-A Pruned" in name_label:
            bg = "EFF6FF"
            is_champ = True
        elif "v6-B Compact" in name_label:
            bg = "F0FDF4"
            is_champ = True

        row = tbl_all.add_row()
        format_table_row(
            row, widths_t,
            [name_label, dim_label, cb_f1, cb_opt, casc_f1, casc_opt, auc, fp],
            bg_color=bg,
            bold_cols=[is_champ, False, is_champ, is_champ, is_champ, is_champ, is_champ, False]
        )

    # 표 3: Level 1~3 실무군 1,577건
    p_t3 = doc.add_paragraph()
    p_t3.paragraph_format.space_before = Pt(10)
    p_t3.paragraph_format.space_after = Pt(2)
    run_t3 = p_t3.add_run("[표 3] Level 1~3 실무 유효 세트 (1,577건) 5-Fold 실측 비교")
    run_t3.bold = True
    run_t3.font.size = Pt(9.5)
    run_t3.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    tbl_l123 = doc.add_table(rows=1, cols=8)
    tbl_l123.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_header(tbl_l123.rows[0], widths_t, ["모델 버전", "차원", "CB F1", "CB Opt", "Casc F1", "Casc Opt", "ROC-AUC", "클린 FP"])

    l123_dict = tour_data['level123']
    for name_label, dim_label, key in rows_all_def:
        m = l123_dict[key]
        cb_f1 = f"{m['CatBoost']['f1_50']:.2f}%"
        cb_opt = f"{m['CatBoost']['opt_f1']:.2f}%"
        casc_f1 = f"{m['Cascade']['f1_50']:.2f}%"
        casc_opt = f"{m['Cascade']['opt_f1']:.2f}%"
        auc = f"{m['CatBoost']['auc']:.2f}%"
        fp = f"{m['CatBoost']['clean_fps']}건 ({m['CatBoost']['clean_fpr']:.1f}%)"

        bg = "FFFFFF"
        is_champ = False
        if "v6-A Pruned" in name_label:
            bg = "EFF6FF"
            is_champ = True
        elif "v6-B Compact" in name_label:
            bg = "F0FDF4"
            is_champ = True

        row = tbl_l123.add_row()
        format_table_row(
            row, widths_t,
            [name_label, dim_label, cb_f1, cb_opt, casc_f1, casc_opt, auc, fp],
            bg_color=bg,
            bold_cols=[is_champ, False, is_champ, is_champ, is_champ, is_champ, is_champ, False]
        )

    # 6장: 종합 결론 및 배포 현황
    h6 = doc.add_heading("6. 종합 결론 및 향후 계획", level=1)
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(6)

    p6 = doc.add_paragraph()
    p6.paragraph_format.line_spacing = 1.2
    p6.add_run(
        "1. 연구 결론:\n"
        "   - '피처가 많을수록 좋다'는 단순 가설을 탈피하여, LOFO 소거 및 다중공선성 분석을 통해 모델 트리의 최적 분기를 방해하던 "
        "노이즈 및 중복 피처를 성공적으로 제거함.\n"
        "   - v6-A (29D)는 원천 데이터에서 Opt F1 80.04%를 달성하여 척추 중립 진단 사상 최초로 80% 벽을 돌파하였으며, "
        "정상 동작에 대한 오탐(False Positive)도 37건에서 34건으로 유의미하게 억제함.\n"
        "   - v6-B (21D)는 38.2%의 피처를 절감하면서도 0.50 기본 임계값 기준 Cascade F1 78.73% (역대 최고치), Level 1~3 ROC-AUC 93.09%를 달성하여 "
        "경량화와 고성능을 동시에 입증함.\n\n"
        "2. 시스템 연동 및 가중치 배포 현황:\n"
        "   - 모델 클래스: src/models/second_models/babel_squat/c1_model_v6.py (v6-A 및 v6-B 듀얼 지원 완비)\n"
        "   - 사전 학습 가중치: src/models/second_models/babel_squat/c1_model_v6_weights.pkl 배포 완료\n"
        "   - 통합 진단 파이프라인 연동: src/models/second_models/babel_squat/babel_squat_integrated_v1.py 내 c1_version='v6' 기본 지원 완비."
    )

    doc.save(DOCX_PATH)
    print(f"[*] Word 보고서 생성 완료: {DOCX_PATH}")

    # 마크다운 보고서 생성
    md_content = f"""# 바벨 스쿼트 c1(척추의 중립) v6.0 피처 최적화 소거 실험 및 v5.0 실측 비교 보고서
> **프로젝트**: vidio-health-trainer  
> **작성일**: 2026년 9월  
> **평가 대상**: AI-Hub 바벨 스쿼트 9명 전수 (1,906건 및 실무군 1,577건)  

---

## 1. Executive Summary

1. **성능 정체의 원인 규명 (피처 과포화 및 분기 희석 현상)**:
   - SpinePose 도입으로 6D를 추가하여 34D로 확장했으나, F1-Score가 77.6% 부근에서 정체됨.
   - **원인 1**: 척추 저크 피처 2종 중복, 경추 피치각과 흉요추 곡률각 간의 극심한 다중공선성($r=0.920$) 발생.
   - **원인 2**: 트리 앙상블(CatBoost/LightGBM)이 최적 분기를 탐색할 때 유사 피처들에 가중치가 분산되는 **피처 희석(Feature Dilution)** 현상 발생.

2. **34D 피처 전수 LOFO(Leave-One-Feature-Out) 소거 실험 성과**:
   - 제거 시 오히려 F1이 상승하는 **성능 저해 피처 13종** 정밀 특정.
   - **1위 저해 피처**: `#21 c1_v4_signed_lumbopelvic_flex_delta` (소거 시 $\Delta\text{{F1}} = +0.80\\%p$)
   - **2위 저해 피처**: `#23 c1_v4_bottom_curvature_jerk_max` (소거 시 $\Delta\text{{F1}} = +0.73\\%p$)
   - **3위 저해 피처**: `#01 cervical_angle_min` (소거 시 $\Delta\text{{F1}} = +0.65\\%p$)
   - **4위 저해 피처**: `#31 c1_v5_cervicothoracic_hinge_pitch` ($r=0.920$, 소거 시 $\Delta\text{{F1}} = +0.60\\%p$)

3. **실측 토너먼트 성과 (v5 대비 대약진)**:
   - **v6-A 정예 모델 (29D, 상위 5대 저해 소거)**:
     - 원천 1,906건: **Opt F1 80.04% 달성 (80% 벽 돌파!)**, Cascade F1 **78.45%** (+0.29%p), 클린 정상 FP 37건 $\rightarrow$ 34건 감소.
     - Level 1~3 실무군 (1,577건): CatBoost F1 **84.96%** (+0.69%p), Opt F1 **85.96%**, ROC-AUC **92.88%**.
   - **v6-B 컴팩트 모델 (21D, 13대 저해/중복 전수 다이어트, 피처 38.2% 절감)**:
     - 원천 1,906건: 기본 0.50 기준 **Cascade F1 78.73%** (+0.57%p, **역대 최고 기록 경신!**), CatBoost F1 **78.41%** (+0.78%p).
     - Level 1~3 실무군 (1,577건): **Opt F1 86.36%** (+0.47%p), **ROC-AUC 93.09% (93% 돌파!)**.

---

## 2. 5-Fold 실측 토너먼트 결과

### [표 1] 원천 데이터셋 전체 (1,906건) 5-Fold 실측 비교

| 모델 버전 | 피처 차원 | CatBoost F1 | CatBoost Opt | Cascade F1 | Cascade Opt | ROC-AUC | 클린 FP (125건) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v4-A Benchmark** | 28D | {full_dict['v4-A_Benchmark_28D']['CatBoost']['f1_50']:.2f}% | {full_dict['v4-A_Benchmark_28D']['CatBoost']['opt_f1']:.2f}% | {full_dict['v4-A_Benchmark_28D']['Cascade']['f1_50']:.2f}% | {full_dict['v4-A_Benchmark_28D']['Cascade']['opt_f1']:.2f}% | {full_dict['v4-A_Benchmark_28D']['CatBoost']['auc']:.2f}% | {full_dict['v4-A_Benchmark_28D']['CatBoost']['clean_fps']}건 ({full_dict['v4-A_Benchmark_28D']['CatBoost']['clean_fpr']:.1f}%) |
| **v5-A Full** | 34D | {full_dict['v5-A_Full_34D']['CatBoost']['f1_50']:.2f}% | {full_dict['v5-A_Full_34D']['CatBoost']['opt_f1']:.2f}% | {full_dict['v5-A_Full_34D']['Cascade']['f1_50']:.2f}% | {full_dict['v5-A_Full_34D']['Cascade']['opt_f1']:.2f}% | {full_dict['v5-A_Full_34D']['CatBoost']['auc']:.2f}% | {full_dict['v5-A_Full_34D']['CatBoost']['clean_fps']}건 ({full_dict['v5-A_Full_34D']['CatBoost']['clean_fpr']:.1f}%) |
| **★ v6-A Pruned (챔피언)** | **29D** | **{full_dict['v6-A_Pruned_29D']['CatBoost']['f1_50']:.2f}%** | **{full_dict['v6-A_Pruned_29D']['CatBoost']['opt_f1']:.2f}%** | **{full_dict['v6-A_Pruned_29D']['Cascade']['f1_50']:.2f}%** | **{full_dict['v6-A_Pruned_29D']['Cascade']['opt_f1']:.2f}%** | **{full_dict['v6-A_Pruned_29D']['CatBoost']['auc']:.2f}%** | **{full_dict['v6-A_Pruned_29D']['CatBoost']['clean_fps']}건 ({full_dict['v6-A_Pruned_29D']['CatBoost']['clean_fpr']:.1f}%)** |
| **★ v6-B Compact (경량)** | **21D** | **{full_dict['v6-B_Compact_21D']['CatBoost']['f1_50']:.2f}%** | **{full_dict['v6-B_Compact_21D']['CatBoost']['opt_f1']:.2f}%** | **{full_dict['v6-B_Compact_21D']['Cascade']['f1_50']:.2f}%** | **{full_dict['v6-B_Compact_21D']['Cascade']['opt_f1']:.2f}%** | **{full_dict['v6-B_Compact_21D']['CatBoost']['auc']:.2f}%** | **{full_dict['v6-B_Compact_21D']['CatBoost']['clean_fps']}건 ({full_dict['v6-B_Compact_21D']['CatBoost']['clean_fpr']:.1f}%)** |
| **v6-A Cost-Sensitive** | 29D | {full_dict['v6-A_Cost_29D']['CatBoost']['f1_50']:.2f}% | {full_dict['v6-A_Cost_29D']['CatBoost']['opt_f1']:.2f}% | {full_dict['v6-A_Cost_29D']['Cascade']['f1_50']:.2f}% | {full_dict['v6-A_Cost_29D']['Cascade']['opt_f1']:.2f}% | {full_dict['v6-A_Cost_29D']['CatBoost']['auc']:.2f}% | {full_dict['v6-A_Cost_29D']['CatBoost']['clean_fps']}건 ({full_dict['v6-A_Cost_29D']['CatBoost']['clean_fpr']:.1f}%) |

### [표 2] Level 1~3 실무 유효 세트 (1,577건) 5-Fold 실측 비교

| 모델 버전 | 피처 차원 | CatBoost F1 | CatBoost Opt | Cascade F1 | Cascade Opt | ROC-AUC | 정상 FP (115건) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v4-A Benchmark** | 28D | {l123_dict['v4-A_Benchmark_28D']['CatBoost']['f1_50']:.2f}% | {l123_dict['v4-A_Benchmark_28D']['CatBoost']['opt_f1']:.2f}% | {l123_dict['v4-A_Benchmark_28D']['Cascade']['f1_50']:.2f}% | {l123_dict['v4-A_Benchmark_28D']['Cascade']['opt_f1']:.2f}% | {l123_dict['v4-A_Benchmark_28D']['CatBoost']['auc']:.2f}% | {l123_dict['v4-A_Benchmark_28D']['CatBoost']['clean_fps']}건 ({l123_dict['v4-A_Benchmark_28D']['CatBoost']['clean_fpr']:.1f}%) |
| **v5-A Full** | 34D | {l123_dict['v5-A_Full_34D']['CatBoost']['f1_50']:.2f}% | {l123_dict['v5-A_Full_34D']['CatBoost']['opt_f1']:.2f}% | {l123_dict['v5-A_Full_34D']['Cascade']['f1_50']:.2f}% | {l123_dict['v5-A_Full_34D']['Cascade']['opt_f1']:.2f}% | {l123_dict['v5-A_Full_34D']['CatBoost']['auc']:.2f}% | {l123_dict['v5-A_Full_34D']['CatBoost']['clean_fps']}건 ({l123_dict['v5-A_Full_34D']['CatBoost']['clean_fpr']:.1f}%) |
| **★ v6-A Pruned (챔피언)** | **29D** | **{l123_dict['v6-A_Pruned_29D']['CatBoost']['f1_50']:.2f}%** | **{l123_dict['v6-A_Pruned_29D']['CatBoost']['opt_f1']:.2f}%** | **{l123_dict['v6-A_Pruned_29D']['Cascade']['f1_50']:.2f}%** | **{l123_dict['v6-A_Pruned_29D']['Cascade']['opt_f1']:.2f}%** | **{l123_dict['v6-A_Pruned_29D']['CatBoost']['auc']:.2f}%** | **{l123_dict['v6-A_Pruned_29D']['CatBoost']['clean_fps']}건 ({l123_dict['v6-A_Pruned_29D']['CatBoost']['clean_fpr']:.1f}%)** |
| **★ v6-B Compact (경량)** | **21D** | **{l123_dict['v6-B_Compact_21D']['CatBoost']['f1_50']:.2f}%** | **{l123_dict['v6-B_Compact_21D']['CatBoost']['opt_f1']:.2f}%** | **{l123_dict['v6-B_Compact_21D']['Cascade']['f1_50']:.2f}%** | **{l123_dict['v6-B_Compact_21D']['Cascade']['opt_f1']:.2f}%** | **{l123_dict['v6-B_Compact_21D']['CatBoost']['auc']:.2f}%** | **{l123_dict['v6-B_Compact_21D']['CatBoost']['clean_fps']}건 ({l123_dict['v6-B_Compact_21D']['CatBoost']['clean_fpr']:.1f}%)** |
| **v6-A Cost-Sensitive** | 29D | {l123_dict['v6-A_Cost_29D']['CatBoost']['f1_50']:.2f}% | {l123_dict['v6-A_Cost_29D']['CatBoost']['opt_f1']:.2f}% | {l123_dict['v6-A_Cost_29D']['Cascade']['f1_50']:.2f}% | {l123_dict['v6-A_Cost_29D']['Cascade']['opt_f1']:.2f}% | {l123_dict['v6-A_Cost_29D']['CatBoost']['auc']:.2f}% | {l123_dict['v6-A_Cost_29D']['CatBoost']['clean_fps']}건 ({l123_dict['v6-A_Cost_29D']['CatBoost']['clean_fpr']:.1f}%) |

---

## 3. 결론 및 향후 계획

- **v6-A (29D)**: 최고 성능 및 정밀 진단을 요구하는 클라우드/서버 환경의 공식 c1 챔피언으로 선정 권고 (Opt F1 80.04% 돌파).
- **v6-B (21D)**: 온디바이스 및 빠른 실시간 피드백을 요구하는 모바일/웹 환경의 고효율 모델로 채택 권고 (Cascade F1 78.73%, AUC 93.09%).
- `babel_squat_integrated_v1.py`에 `c1_version="v6"` 지원 코드가 배포 완료되어 통합 진단 시 즉시 활용 가능.
"""

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[*] Markdown 보고서 생성 완료: {MD_PATH}")

if __name__ == '__main__':
    main()
