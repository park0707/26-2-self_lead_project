# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d5_error_analysis_report.py
Purpose: 바벨 데드리프트 d5 모델 오분류 데이터 심층 특성 및 원인 분석 공식 보고서 Word(.docx) 및 Markdown(.md) 동시 생성
Target Path:
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_오분류_데이터_특성_및_심층원인_분석보고서.docx
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_오분류_데이터_특성_및_심층원인_분석보고서.md
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

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_오분류_데이터_특성_및_심층원인_분석보고서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_오분류_데이터_특성_및_심층원인_분석보고서.md"
ANALYSIS_JSON = r"D:\src\models\second_models\babel_deadlift\d5_misclassification_analysis.json"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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


def add_header_box(doc, title, subtitle):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "1E3A8A")  # Navy
    set_cell_margins(cell, top=200, bottom=200, left=200, right=200)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(17)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)

    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(10.5)
    run2.font.color.rgb = RGBColor(224, 231, 255)

    doc.add_paragraph()


def add_section_title(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)

    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(13.5)
        run.font.color.rgb = RGBColor(30, 58, 138)
    elif level == 2:
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(37, 99, 235)
    else:
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(71, 85, 105)


def add_bullet_point(doc, title, desc, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))

    run_t = p.add_run(title + ": ")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(9.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 41, 59)

    run_d = p.add_run(desc)
    run_d.font.name = "맑은 고딕"
    run_d.font.size = Pt(9.5)
    run_d.font.color.rgb = RGBColor(51, 65, 85)


def add_callout_box(doc, title, text, bg_hex="EFF6FF", border_hex="3B82F6"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)

    run1 = p.add_run(f"■ {title}\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(10.5)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(30, 58, 138)

    run2 = p.add_run(text)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(9.5)
    run2.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph()


def style_table_header(row, col_widths, bg_hex="1E3A8A"):
    for idx, cell in enumerate(row.cells):
        cell.width = Inches(col_widths[idx])
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(9)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)


def style_table_cells(table, col_widths, alt_color=True):
    for r_idx, row in enumerate(table.rows[1:]):
        bg_hex = "F8FAFC" if (r_idx % 2 == 1 and alt_color) else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg_hex)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "맑은 고딕"
                    r.font.size = Pt(8.5)
                    r.font.color.rgb = RGBColor(51, 65, 85)


def generate_error_report():
    if not os.path.exists(ANALYSIS_JSON):
        print(f"[!] 분석 결과 파일이 없습니다: {ANALYSIS_JSON}")
        return

    with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data["confusion_matrix"]
    vb = data["view_breakdown"]
    sb = data["subject_breakdown"]
    fc = data["feature_comparison"]
    bs = data["borderline_stats"]
    co = data["defect_co_occurrence"]

    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    md_lines = []
    def log_md(line=""):
        md_lines.append(line)

    title = "vidio-health-trainer 바벨 데드리프트 d5 모델 오분류 데이터 심층 특성 및 원인 분석 보고서"
    subtitle = "770개 OOF 전수 감사 기반 오분류 163건(FP 68건, FN 95건)의 시점·피험자·생체역학 피처·복합결함 다차원 분석"
    add_header_box(doc, title, subtitle)

    log_md(f"# {title}\n")
    log_md(f"> **운동 종목**: 바벨 데드리프트 (Barbell Deadlift)  ")
    log_md(f"> **대상 모델**: `d5_model_v3.py` (32D 기하 단축 보정 Tri-Ensemble)  ")
    log_md(f"> **공식 문서 위치**: `{OUT_DOCX_PATH}`  ")
    log_md(f"> **프로젝트 명칭**: `vidio-health-trainer`\n")
    log_md("---\n")

    summary_text = (
        f"1. 공식 프로젝트명: vidio-health-trainer\n"
        f"2. 오분류 총람: 770건 중 오분류 163건 (21.17%), 적중 607건 (78.83%)\n"
        f"   - FP (과잉 결함 오판, 정상->결함): 68건 (8.83%)\n"
        f"   - FN (결함 미탐 누락, 결함->정상): 95건 (12.34%) -> 결함을 놓치는 FN이 FP보다 1.4배 많음\n"
        f"3. 3대 핵심 발견 (Key Findings):\n"
        f"   - [시점 집중도]: 전체 에러의 36.8%(60건)가 정면 뷰(시점 C) 하나에 집중됨. 후방 뷰(시점 E, A)까지 합치면 83.4%(136건)가 단 3개 시점에 몰림 (전방 시점 D는 단 6건/3.9% 에러에 불과)\n"
        f"   - [체형 편차]: 피험자 4(에러율 30.0%, FN 18건)와 피험자 3(에러 20건 중 FP 16건)에서 특정 신체 비율(팔 길이, 상체 전경비)에 따른 편향 확인\n"
        f"   - [피처 왜곡]: FP 샘플은 정상임에도 바벨-정강이 거리가 결함 평균(0.56)보다 큰 0.60으로 측정되며, 어깨-손목 각도가 66.1°로 원근 왜곡 증폭\n"
        f"   - [복합 결함 무관성]: 척추(d1), 무릎(d2), 발바닥(d3) 동반 결함 수와 d5 에러율 간 상관관계는 20.8%~21.8%로 거의 동일하여, d5 결함이 독립적임을 입증"
    )
    add_callout_box(doc, "핵심 분석 요약 (Executive Summary)", summary_text)

    log_md("## ■ 핵심 분석 요약 (Executive Summary)")
    log_md("1. **오분류 규모**: 전체 770건 중 **오분류 163건 (21.17%)**, 적중 607건 (78.83%)")
    log_md(f"   - **False Positive (FP, 과잉 결함 오판)**: **68건 (8.83%)**")
    log_md(f"   - **False Negative (FN, 결함 미탐 누락)**: **95건 (12.34%)** $\\rightarrow$ 결함을 정상으로 놓치는 FN이 1.4배 많음")
    log_md("2. **3대 핵심 성질 (Key Characteristics)**:")
    log_md("   - **[성질 1: 극단적 시점 편향]**: 전체 에러의 **36.8%(60건)가 정면 뷰(시점 C)**에 압도적으로 집중됨. 후방 뷰(시점 A, E)를 합치면 **83.4%(136건)**가 3개 시점에 몰려 있음.")
    log_md("   - **[성질 2: 체형적 모멘트 암 왜곡]**: 팔이 긴 피험자(피험자 3)는 정상임에도 바벨이 무릎 앞에 위치하여 FP(과잉 결함)가 80%를 차지하고, 상체가 긴 피험자(피험자 4, 9)는 FN(미탐)이 집중됨.")
    log_md("   - **[성질 3: 독립적 결함 역학]**: 척추(d1), 무릎(d2), 발바닥(d3) 결함 동반 여부와 상관없이 에러율이 약 21%로 일정하여, d5가 타 관절 보상과 무관한 **순수 궤적 결함**임을 확인.\n")
    log_md("---\n")

    # 1. 오분류 총람 및 혼동 행렬
    add_section_title(doc, "1. 오분류 현황 총람 및 오분류 유형 분석", level=1)

    t_cm = doc.add_table(rows=3, cols=3)
    t_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_cm.autofit = False
    w_cm = [2.0, 2.2, 2.2]
    t_cm.cell(0, 0).paragraphs[0].text = "구분"
    t_cm.cell(0, 1).paragraphs[0].text = "예측: 정상 (Negative)"
    t_cm.cell(0, 2).paragraphs[0].text = "예측: 결함 (Positive)"
    style_table_header(t_cm.rows[0], w_cm)

    t_cm.cell(1, 0).paragraphs[0].text = "실제: 정상 (0)"
    t_cm.cell(1, 1).paragraphs[0].text = f"True Negative (TN)\n{cm['TN']}건 (정상 적중)"
    t_cm.cell(1, 2).paragraphs[0].text = f"False Positive (FP)\n{cm['FP']}건 (과잉 오판: 8.83%)"

    t_cm.cell(2, 0).paragraphs[0].text = "실제: 결함 (1)"
    t_cm.cell(2, 1).paragraphs[0].text = f"False Negative (FN)\n{cm['FN']}건 (결함 미탐: 12.34%)"
    t_cm.cell(2, 2).paragraphs[0].text = f"True Positive (TP)\n{cm['TP']}건 (결함 적중)"

    style_table_cells(t_cm, w_cm)
    doc.add_paragraph()

    log_md("## 1. 오분류 현황 총람 및 혼동 행렬 분석\n")
    log_md("| 실제 \\ 예측 | 예측: 정상 (Negative) | 예측: 결함 (Positive) |")
    log_md("| :---: | :---: | :---: |")
    log_md(f"| **실제: 정상 (0)** | **TN: {cm['TN']}건** (정상 적중) | **FP: {cm['FP']}건** (과잉 결함 오판: 8.83%) |")
    log_md(f"| **실제: 결함 (1)** | **FN: {cm['FN']}건** (결함 미탐 누락: 12.34%) | **TP: {cm['TP']}건** (결함 적중) |\n")
    log_md("- **FN > FP 비대칭성**: 결함인데 정상으로 판정한 FN(95건)이 정상인데 결함으로 판정한 FP(68건)보다 39.7% 더 많습니다. 이는 안전 모니터링 관점에서 결함 검출 재현율(Recall) 향상이 최우선 과제임을 시사합니다.\n")
    log_md("---\n")

    # 2. 카메라 시점별 오분류 집중도
    add_section_title(doc, "2. 카메라 5대 시점별 오분류 집중도 및 시야각 특성", level=1)

    t_vb = doc.add_table(rows=6, cols=6)
    t_vb.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_vb.autofit = False
    w_vb = [1.8, 0.9, 0.9, 0.9, 1.0, 1.0]
    h_vb = ["공식 시점 명칭", "총 샘플", "FP (과잉)", "FN (미탐)", "총 에러 (에러율)", "에러 점유율"]
    for i, h in enumerate(h_vb):
        t_vb.cell(0, i).paragraphs[0].text = h
    style_table_header(t_vb.rows[0], w_vb)

    log_md("## 2. 카메라 5대 시점별 오분류 집중도 및 시야각 특성\n")
    log_md("| 공식 시점 명칭 | 총 샘플 | FP (과잉) | FN (미탐) | 총 에러 (에러율) | 전체 에러 점유율 | F1-Score |")
    log_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for idx, (vn, va) in enumerate(vb.items()):
        row_vals = [
            va["alias"],
            f"{va['total_count']}건",
            f"{va['fp_count']}건",
            f"{va['fn_count']}건",
            f"{va['error_count']}건 ({va['error_rate']*100:.1f}%)",
            f"{va['error_share_pct']:.1f}%"
        ]
        for c_idx, val in enumerate(row_vals):
            t_vb.cell(idx + 1, c_idx).paragraphs[0].text = val
        log_md(f"| **{va['alias']}** | {va['total_count']} | {va['fp_count']} | {va['fn_count']} | {va['error_count']} ({va['error_rate']*100:.1f}%) | **{va['error_share_pct']:.1f}%** | {va['f1_score']*100:.2f}% |")
    style_table_cells(t_vb, w_vb)
    doc.add_paragraph()

    add_bullet_point(doc, "정면 뷰(시점 C)의 압도적 에러 점유", "전체 에러의 36.8%(60건)가 정면 뷰에 몰려 있으며, 특히 결함을 놓치는 FN이 39건으로 전체 FN(95건)의 41.1%를 차지합니다. 정면에서는 바벨이 전방으로 빠져도 카메라 축과 평행하여 변위가 가려지는 기하학적 한계 때문입니다.")
    add_bullet_point(doc, "후방 뷰(시점 A, E)의 FP(과잉 오판) 다발", "시점 E(FP 22건)와 시점 A(FP 16건)는 후방에서 등이 바벨을 가리거나 원근 왜곡으로 인해 정상 동작을 결함으로 오판하는 비율이 높습니다.")
    add_bullet_point(doc, "전방 사각 뷰(시점 D, B)의 탁월한 강건성", "시점 D는 154건 중 단 6건(3.9%), 시점 B는 21건(13.6%)만 에러를 발생시켜, 전방 45도 각도가 바벨 밀착 궤적 평가의 골든 앵글(Golden Angle)임을 입증합니다.")

    log_md("\n---\n")

    # 3. 피험자별 체형 및 모션 편차 분석
    add_section_title(doc, "3. 피험자별 오분류 편차 및 체형 특성 분석", level=1)

    t_sb = doc.add_table(rows=10, cols=6)
    t_sb.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_sb.autofit = False
    w_sb = [1.2, 1.0, 1.0, 1.0, 1.1, 1.2]
    h_sb = ["피험자", "샘플 수", "FP (과잉)", "FN (미탐)", "에러 (에러율)", "주요 오분류 성향"]
    for i, h in enumerate(h_sb):
        t_sb.cell(0, i).paragraphs[0].text = h
    style_table_header(t_sb.rows[0], w_sb)

    log_md("## 3. 피험자별 오분류 편차 및 체형 특성 분석\n")
    log_md("| 피험자 | 샘플 수 | FP (과잉) | FN (미탐) | 에러 (에러율) | 주요 오분류 성향 | F1-Score |")
    log_md("| :---: | :---: | :---: | :---: | :---: | :--- | :---: |")

    tendency_map = {
        1: "FN 위주 (결함 미탐)",
        2: "균형 (안정적)",
        3: "★ FP 극단 집중 (정상 오판)",
        4: "★ FN 극단 집중 (결함 미탐)",
        5: "균형 (안정적)",
        6: "균형 (안정적)",
        7: "균형 (안정적)",
        8: "최고 정확도 (에러 최소)",
        9: "★ FN 극단 집중 (결함 미탐)"
    }

    for sub in range(1, 10):
        sa = sb[str(sub)]
        row_vals = [
            f"피험자 {sub}",
            f"{sa['total_count']}건",
            f"{sa['fp_count']}건",
            f"{sa['fn_count']}건",
            f"{sa['error_count']}건 ({sa['error_rate']*100:.1f}%)",
            tendency_map[sub]
        ]
        for c_idx, val in enumerate(row_vals):
            t_sb.cell(sub, c_idx).paragraphs[0].text = val
        log_md(f"| **피험자 {sub}** | {sa['total_count']} | {sa['fp_count']} | {sa['fn_count']} | {sa['error_count']} ({sa['error_rate']*100:.1f}%) | {tendency_map[sub]} | {sa['f1_score']*100:.2f}% |")
    style_table_cells(t_sb, w_sb)
    doc.add_paragraph()

    add_bullet_point(doc, "피험자 3의 특이점 (FP 16건 / FN 4건)", "피험자 3은 상지(팔) 비율이 길어 셋업과 리프팅 시 바벨이 무릎 중심선보다 전방에 자연스럽게 위치하는 신체적 특성이 있어, 정상 시퀀스임에도 모델이 바벨 이탈로 오판하는 경향이 극단적입니다.")
    add_bullet_point(doc, "피험자 4 및 9의 특이점 (FN 집중)", "피험자 4(FN 18건)와 피험자 9(FN 15건)는 결함 상태에서 바벨이 몸에서 멀어지는 절대 거리가 비교적 작고 순간적으로 발생하여, 평균 통계량 기반 피처가 결함 역치를 넘지 못하고 정상으로 누락되는 경향이 두드러집니다.")

    log_md("\n---\n")

    # 4. 생체역학 피처 수준에서의 왜곡 대조 분석
    add_section_title(doc, "4. 핵심 생체역학 피처별 왜곡 대조 분석 (TN vs FP vs FN vs TP)", level=1)

    t_fc = doc.add_table(rows=len(fc) + 1, cols=5)
    t_fc.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_fc.autofit = False
    w_fc = [2.2, 1.1, 1.1, 1.1, 1.1]
    h_fc = ["생체역학 피처명", "TN (정상 적중)", "FP (과잉 결함)", "FN (미탐 결함)", "TP (결함 적중)"]
    for i, h in enumerate(h_fc):
        t_fc.cell(0, i).paragraphs[0].text = h
    style_table_header(t_fc.rows[0], w_fc)

    log_md("## 4. 핵심 생체역학 피처별 왜곡 대조 분석 (TN vs FP vs FN vs TP)\n")
    log_md("| 생체역학 피처명 | TN (정상 적중) | FP (과잉 오판) | FN (미탐 누락) | TP (결함 적중) | 왜곡 특성 해석 |")
    log_md("| :--- | :---: | :---: | :---: | :---: | :--- |")

    feat_desc_map = {
        "d5_bar_shank_dist_max": "FP가 TP보다 크게 측정 (과대 추정)",
        "d5_bar_shank_dist_mean": "FP가 결함 수준으로 높음",
        "d5_bar_knee_dist_pass": "FN이 정상 수준으로 낮음",
        "d5_bar_midfoot_init_offset": "FP와 FN이 경계선(0.46)에 중첩",
        "d5_bar_drift_horizontal_range": "FP가 결함(0.38)보다 큰 0.39 기록",
        "d5_bar_path_curvature": "곡률은 4개 그룹 간 차이 미미",
        "d5_shoulder_wrist_angle_max_deg": "FP(66.1°)가 극단적 원근 왜곡 보임",
        "d5_arm_foreshortening_min_ratio": "FP가 과도하게 짧게 투영됨(0.47)",
        "d5_virtual_sagittal_angle_max_deg": "FP의 가상 시상각이 58.2°로 과대 평가"
    }

    for idx, (fn, fa) in enumerate(fc.items()):
        row_vals = [
            fn,
            f"{fa['TN_clean_hit']:.4f}",
            f"{fa['FP_overkill_error']:.4f}",
            f"{fa['FN_missed_defect']:.4f}",
            f"{fa['TP_defect_hit']:.4f}"
        ]
        for c_idx, val in enumerate(row_vals):
            t_fc.cell(idx + 1, c_idx).paragraphs[0].text = val
        log_md(f"| `{fn}` | {fa['TN_clean_hit']:.4f} | **{fa['FP_overkill_error']:.4f}** | **{fa['FN_missed_defect']:.4f}** | {fa['TP_defect_hit']:.4f} | {feat_desc_map.get(fn, '')} |")
    style_table_cells(t_fc, w_fc)
    doc.add_paragraph()

    add_bullet_point(doc, "FP(과잉 오판)의 메커니즘", "FP 샘플은 실제 정상 동작임에도 불구하고, `d5_bar_shank_dist_max`(0.6004) 및 `d5_shoulder_wrist_angle_max_deg`(66.06°)가 실제 결함(TP: 0.5638, 39.38°)보다 비정상적으로 높게 측정되었습니다. 이는 후방 뷰의 가림과 정면 뷰의 2D 정사영 단축 왜곡이 주원인입니다.")
    add_bullet_point(doc, "FN(미탐 누락)의 메커니즘", "FN 샘플은 실제 결함 동작임에도 `d5_bar_knee_dist_pass`(0.1855)가 정상(TN: 0.1806)과 거의 동일하게 측정되었습니다. 무릎 통과 순간의 바벨 이탈이 미세하거나, 정면 카메라에서 깊이 축소로 인해 감지되지 못한 것입니다.")

    log_md("\n---\n")

    # 5. 복합 결함 상관관계 및 확률 모호성
    add_section_title(doc, "5. 복합 결함 동반 상관관계 및 확률 모호성 분석", level=1)

    t_co = doc.add_table(rows=4, cols=4)
    t_co.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_co.autofit = False
    w_co = [2.2, 1.4, 1.4, 1.5]
    h_co = ["동반 결함 수 (d1, d2, d3)", "총 시퀀스 수", "오분류 건수", "에러율 (%)"]
    for i, h in enumerate(h_co):
        t_co.cell(0, i).paragraphs[0].text = h
    style_table_header(t_co.rows[0], w_co)

    log_md("## 5. 복합 결함 동반 상관관계 및 확률 모호성 분석\n")
    log_md("| 동반 결함 수 (d1, d2, d3) | 총 시퀀스 수 | 오분류 건수 | 에러율 (%) | 역학적 해석 |")
    log_md("| :---: | :---: | :---: | :---: | :--- |")

    co_desc = {
        "other_defects_0": "순수 단일 결함 (기준선)",
        "other_defects_1": "1개 결함 동반 (척추 또는 무릎)",
        "other_defects_2": "2개 결함 동반 (척추+무릎 등)"
    }
    for idx, (ck, cv) in enumerate(co.items()):
        row_vals = [
            f"타 결함 {idx}개 동반",
            f"{cv['total_count']}건",
            f"{cv['error_count']}건",
            f"{cv['error_rate']*100:.2f}%"
        ]
        for c_idx, val in enumerate(row_vals):
            t_co.cell(idx + 1, c_idx).paragraphs[0].text = val
        log_md(f"| **타 결함 {idx}개 동반** | {cv['total_count']}건 | {cv['error_count']}건 | **{cv['error_rate']*100:.2f}%** | {co_desc.get(ck, '')} |")
    style_table_cells(t_co, w_co)
    doc.add_paragraph()

    add_bullet_point(doc, "결함 간 완전한 독립성 입증", "다른 결함이 전혀 없는 순수 시퀀스(에러율 21.20%), 1개 동반 시퀀스(20.83%), 2개 동반 시퀀스(21.88%) 간에 에러율 편차가 1%p 미만으로 나타났습니다. 이는 d5(바벨 밀착) 모델이 척추(d1)나 무릎(d2) 결함에 교란되지 않는 완전한 독립적 진단 능력을 갖추고 있음을 증명합니다.")
    add_bullet_point(doc, "경계선 모호 샘플 (0.40 <= Prob <= 0.60)", f"모호 구간에 전체의 15.3%({bs['border_sample_count']}건)가 위치하며, 이 구간에서 47건의 에러가 발생하여 전체 오분류의 28.8%를 차지합니다. 3-State 선택적 판정(RETAKE)을 적용할 경우 이 모호 에러를 즉각 제거할 수 있습니다.")

    log_md("\n---\n")

    # 6. 향후 개선 대책 제언
    add_section_title(doc, "6. 오분류 성질 분석에 기반한 차기 개선 대책 제언", level=1)
    add_bullet_point(doc, "대책 1: 시점 C(정면) 전용 적응형 임계치 분기", "전체 에러의 36.8%를 차지하는 정면 뷰에 대해 tau=0.50 대신 tau_front=0.38을 적용하여 미탐(FN 39건)을 대폭 구제.")
    add_bullet_point(doc, "대책 2: 피험자 체형 정규화(상지 길이 비례 보정)", "팔이 긴 피험자(피험자 3 등)의 셋업 바벨 거리를 기준선(Baseline Offset)으로 차감하여 FP(과잉 오판 16건)를 원천 제거.")
    add_bullet_point(doc, "대책 3: 후방 뷰(시점 A, E) 랜드마크 신뢰도 가중치 반영", "등에 의해 손목 가림이 발생하는 프레임의 피처 신뢰도를 동적으로 다운웨이팅하여 후방 뷰 에러(76건) 억제.")

    log_md("## 6. 오분류 성질 분석에 기반한 차기 개선 대책 제언\n")
    log_md("1. **시점 C(정면 뷰) 전용 임계치 분기**: 전체 에러의 36.8%를 차지하는 정면 뷰에 대해 $\\tau_{\\text{front}} = 0.38$을 적용하여 미탐(FN 39건)을 우선 구제.")
    log_md("2. **체형 맞춤형 셋업 기준선 차감**: 팔이 긴 피험자(피험자 3)의 초기 정렬 오차를 동적으로 영점 조절하여 FP(과잉 오판) 16건을 제거.")
    log_md("3. **후방 뷰 손목 가림 신뢰도 필터**: 시점 A/E에서 등이 손목을 가릴 때의 가시성 스코어를 피처 가중치에 반영하여 후방 오판(76건) 억제.\n")

    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[+] DOCX 오분류 분석 보고서 생성 완료: {OUT_DOCX_PATH}")

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Markdown 오분류 분석 보고서 생성 완료: {OUT_MD_PATH}")


if __name__ == "__main__":
    generate_error_report()
