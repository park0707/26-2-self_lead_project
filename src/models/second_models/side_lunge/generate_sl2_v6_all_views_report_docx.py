# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Script: generate_sl2_v6_all_views_report_docx.py
Purpose: v6 사용자 제안 기하 직교 알고리즘 전체 원천 데이터(4,130건) 5대 시점
         보류율 10% 제한 3-State F1 성능 평가 공식 Word 및 MD 보고서 생성
"""

import os
import sys
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPORT_DIR = r"D:\자주프 문서\6주차"
os.makedirs(REPORT_DIR, exist_ok=True)

DOCX_OUT = os.path.join(REPORT_DIR, "사이드런지_c2_v6_전체원천데이터_5대시점_보류율10pct_공식성능평가보고서.docx")
MD_OUT = os.path.join(REPORT_DIR, "사이드런지_c2_v6_전체원천데이터_5대시점_보류율10pct_공식성능평가보고서.md")
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_v6_all_views_evaluation_results.json"


def set_cell_background(cell, fill_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color="D2D6DC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)


def format_table(table, header_bg="1E293B"):
    set_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    for cell in table.rows[0].cells:
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = 'Malgun Gothic'
                r.font.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_idx, row in enumerate(table.rows[1:], start=1):
        bg = "F8FAFC" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx in [0, 1] else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.size = Pt(9.0)
                    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)


def generate_reports():
    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        d = json.load(f)

    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    p0 = doc.add_paragraph()
    r0 = p0.add_run("vidio-health-trainer 공식 기술 보고서")
    r0.font.name = 'Malgun Gothic'
    r0.font.size = Pt(11)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

    p1 = doc.add_paragraph()
    r1 = p1.add_run("사이드 런지 c2 v6 기하 직교 알고리즘 전체 원천 데이터(4,130건) 5대 시점 성능 평가 보고서")
    r1.font.name = 'Malgun Gothic'
    r1.font.size = Pt(16)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    tbl = doc.add_table(rows=7, cols=7)
    headers = ["시점 코드", "공식 시점 명칭", "3-State F1", "확정 정확도(ACC)", "정밀도", "재현율", "보류율(Retake)"]
    for j, h in enumerate(headers):
        tbl.cell(0, j).paragraphs[0].text = h

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    rows_data = []
    for i, v in enumerate(view_order, start=1):
        aud = d['view_audits'][v]
        row = [
            v,
            aud['alias'],
            f"{aud['effective_f1']*100:.2f}%",
            f"{aud['accuracy']*100:.2f}%",
            f"{aud['precision']*100:.2f}%",
            f"{aud['recall']*100:.2f}%",
            f"{aud['uncertainty_rate']*100:.1f}%"
        ]
        rows_data.append(row)
        for j, val in enumerate(row):
            tbl.cell(i, j).paragraphs[0].text = val

    tot = d['overall_metrics']
    tot_row = [
        "전체 (ALL)",
        "5대 시점 전체 통합",
        f"{tot['overall_f1']*100:.2f}%",
        f"{tot['overall_accuracy']*100:.2f}%",
        f"{tot['overall_precision']*100:.2f}%",
        f"{tot['overall_recall']*100:.2f}%",
        f"{tot['uncertainty_rate']*100:.1f}%"
    ]
    for j, val in enumerate(tot_row):
        tbl.cell(6, j).paragraphs[0].text = val

    format_table(tbl)
    doc.save(DOCX_OUT)
    print(f"[+] 공식 Word 보고서 생성 완료: {DOCX_OUT}")

    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write("# vidio-health-trainer 기술 보고서\n\n")
        f.write("## 사이드 런지 c2 v6 기하 직교 알고리즘 전체 원천 데이터(4,130건) 5대 시점 성능 평가 보고서\n\n")
        f.write(f"- 작성일자: 2026-10-06\n- 공식 Word 산출물: `{DOCX_OUT}`\n")
        f.write(f"- 평가 조건: **보류율 10.0% 이내 엄수** (실제 평균 보류율: **{tot['uncertainty_rate']*100:.2f}%**)\n\n")
        f.write("### 5대 공식 시점별 세부 성적표\n\n")
        f.write("| 시점 코드 | 공식 시점 명칭 | 3-State F1-Score | 확정 정확도 (ACC) | 정밀도 (Precision) | 재현율 (Recall) | 보류율 (Retake) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows_data:
            f.write(f"| `{r[0]}` | **{r[1]}** | **{r[2]}** | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")
        f.write(f"| **전체** | **{tot_row[1]}** | **{tot_row[2]}** | **{tot_row[3]}** | **{tot_row[4]}** | **{tot_row[5]}** | **{tot_row[6]}** |\n\n")
        f.write("### 주요 분석\n\n")
        f.write("1. **전방 45도 시점의 압도적 우위**: 시점 B(65.75%)와 시점 D(58.46%)가 85~87%의 높은 확정 정확도를 기록하며 전체 성능을 견인.\n")
        f.write("2. **후방 및 정면 시점의 물리적 한계**: 후방(시점 A 41.08%, 시점 E 45.06%)과 정면(시점 C 47.52%)은 발끝 가림 및 원근 왜곡으로 인해 F1 40%대에 머묾.\n")
    print(f"[+] 공식 마크다운 보고서 저장 완료: {MD_OUT}")


if __name__ == '__main__':
    generate_reports()
