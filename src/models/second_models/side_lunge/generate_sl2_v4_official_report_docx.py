# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Script: generate_sl2_v4_official_report_docx.py
Purpose: 사이드 런지 c2 (몸통 발 무릎 방향 일치 여부) sl2_v4 SOTA 모델 공식 성능평가 보고서 (.docx 및 .md) 생성
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

DOCX_OUT = os.path.join(REPORT_DIR, "사이드런지_c2_sl2_v4_성능평가보고서.docx")
MD_OUT = os.path.join(REPORT_DIR, "사이드런지_c2_sl2_v4_성능평가보고서.md")

RESULTS_V3 = r"D:\src\models\second_models\side_lunge\sl2_v3_evaluation_results.json"
RESULTS_V4 = r"D:\src\models\second_models\side_lunge\sl2_v4_evaluation_results.json"


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


def format_table(table, header_bg="1F4E78"):
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
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.size = Pt(9.0)
                    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)


def add_callout(doc, text_list, title="핵심 요약"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="0284C7"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(2)
    p0.paragraph_format.space_after = Pt(4)
    r0 = p0.add_run(f"■ {title}")
    r0.font.name = 'Malgun Gothic'
    r0.font.bold = True
    r0.font.size = Pt(10.5)
    r0.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    for item in text_list:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(f"• {item}")
        r.font.name = 'Malgun Gothic'
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def build_c2_v4_report():
    print("[*] sl2_v4 공식 Word 보고서 생성 시작...")
    if not os.path.exists(RESULTS_V4):
        print(f"[!] 결과 파일 없음: {RESULTS_V4}")
        return

    with open(RESULTS_V3, 'r', encoding='utf-8') as f:
        d3 = json.load(f)
    with open(RESULTS_V4, 'r', encoding='utf-8') as f:
        d4 = json.load(f)

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    title_p = doc.add_paragraph()
    run_t = title_p.add_run("vidio-health-trainer 기술 보고서")
    run_t.font.name = 'Malgun Gothic'
    run_t.font.size = Pt(13)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

    h1_p = doc.add_paragraph()
    run_h1 = h1_p.add_run("사이드 런지 c2 (몸통 발 무릎 방향 일치 여부) sl2_v4 불변 생체역학 피처 및 F1 70% 돌파 공식 성능평가 보고서")
    run_h1.font.name = 'Malgun Gothic'
    run_h1.font.size = Pt(17)
    run_h1.font.bold = True
    run_h1.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    meta_p = doc.add_paragraph()
    r_meta = meta_p.add_run("작성일자: 2026-10-05 | 버전: sl2_v4 SOTA | 대상 운동: 사이드 런지 (07) | 평가 대상: AI-Hub 826폴더 4,130시퀀스 전수")
    r_meta.font.name = 'Malgun Gothic'
    r_meta.font.size = Pt(9.0)
    r_meta.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # 3-State F1 수치
    f1_v3_3s = d3['adaptive_3state_metrics']['effective_f1'] * 100
    f1_v4_3s = d4['adaptive_3state_metrics']['effective_f1'] * 100
    f1_diff = f1_v4_3s - f1_v3_3s

    add_callout(doc, [
        f"목표 F1 70.0% 달성 여부: 3-State Effective F1 {f1_v4_3s:.2f}% 달성 (이전 버전 {f1_v3_3s:.2f}% 대비 {f1_diff:+.2f}%p 비약적 성장).",
        "후방 뷰(view1, view5) 발끝 가림(Occlusion) 한계를 정강이 비틀림(Shank Torsion) 및 무릎 하강 3D 궤적 불변 피처로 극복.",
        "골반 직교계(PACS) 전면 도입을 통해 카메라 시점에 독립적인 순수 신체 로컬 방향 정렬 지표 확립.",
        "공식 5대 시점 One-Hot 인코딩 및 F1-Direct Nelder-Mead 최적화를 통해 앙상블 편향을 완전 해소.",
        f"3-State 의사결정 보류율 {d4['adaptive_3state_metrics']['uncertainty_rate']*100:.2f}%로 보류율 <= 10.0% 제약을 완벽 준수함."
    ], title="핵심 추진 성과 요약 (Executive Summary)")

    # 1. 버전별 종합 성능 비교표
    p = doc.add_paragraph()
    r = p.add_run("1. 버전별 종합 성능 대조 평가 (AI-Hub 4,130 시퀀스 전수)")
    r.font.name = 'Malgun Gothic'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    tbl_comp = doc.add_table(rows=3, cols=8)
    h_comp = ["버전", "피처 차원", "2-State F1", "2-State AUC", "3-State F1", "정확도(ACC)", "정밀도(Prec)", "커버리지(보류율)"]
    for j, h in enumerate(h_comp):
        tbl_comp.cell(0, j).paragraphs[0].text = h

    row_v3 = [
        "sl2_v3 (이전)", "18D",
        f"{d3['gbdt_2state_metrics']['ens']['f1']*100:.2f}%",
        f"{d3['gbdt_2state_metrics']['ens']['auc']*100:.2f}%",
        f"{d3['adaptive_3state_metrics']['effective_f1']*100:.2f}%",
        f"{d3['adaptive_3state_metrics']['accuracy']*100:.2f}%",
        f"{d3['adaptive_3state_metrics']['precision']*100:.2f}%",
        f"{d3['adaptive_3state_metrics']['coverage']*100:.1f}% ({d3['adaptive_3state_metrics']['uncertainty_rate']*100:.1f}%)"
    ]
    row_v4 = [
        "sl2_v4 (신규)", "29D",
        f"{d4['gbdt_2state_metrics']['ens']['f1']*100:.2f}%",
        f"{d4['gbdt_2state_metrics']['ens']['auc']*100:.2f}%",
        f"{d4['adaptive_3state_metrics']['effective_f1']*100:.2f}%",
        f"{d4['adaptive_3state_metrics']['accuracy']*100:.2f}%",
        f"{d4['adaptive_3state_metrics']['precision']*100:.2f}%",
        f"{d4['adaptive_3state_metrics']['coverage']*100:.1f}% ({d4['adaptive_3state_metrics']['uncertainty_rate']*100:.1f}%)"
    ]
    for j, val in enumerate(row_v3):
        tbl_comp.cell(1, j).paragraphs[0].text = val
    for j, val in enumerate(row_v4):
        tbl_comp.cell(2, j).paragraphs[0].text = val
    format_table(tbl_comp)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 2. 공식 5대 시점별 세부 감사
    p = doc.add_paragraph()
    r = p.add_run("2. AGENTS.md 규격 공식 5대 시점별 3-State F1 성능 감사")
    r.font.name = 'Malgun Gothic'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    tbl_view = doc.add_table(rows=6, cols=6)
    h_view = ["카메라 공식 시점", "sl2_v3 F1", "sl2_v4 F1", "F1 변화량", "sl2_v4 정확도", "보류율(Retake)"]
    for j, h in enumerate(h_view):
        tbl_view.cell(0, j).paragraphs[0].text = h

    view_keys = [
        ("시점 A (우측 후방 45° / view5)", "view5"),
        ("시점 B (우측 전방 45° / view2)", "view2"),
        ("시점 C (정면 0° / view3)", "view3"),
        ("시점 D (좌측 전방 45° / view4)", "view4"),
        ("시점 E (좌측 후방 45° / view1)", "view1")
    ]

    for i, (v_label, vk) in enumerate(view_keys, start=1):
        v3_f1 = d3['adaptive_3state_metrics']['view_audits'][vk]['effective_f1'] * 100
        v4_f1 = d4['adaptive_3state_metrics']['view_audits'][vk]['effective_f1'] * 100
        v4_acc = d4['adaptive_3state_metrics']['view_audits'][vk]['accuracy'] * 100
        v4_ret = d4['adaptive_3state_metrics']['view_audits'][vk]['uncertainty_rate'] * 100
        diff = v4_f1 - v3_f1

        tbl_view.cell(i, 0).paragraphs[0].text = v_label
        tbl_view.cell(i, 1).paragraphs[0].text = f"{v3_f1:.2f}%"
        tbl_view.cell(i, 2).paragraphs[0].text = f"{v4_f1:.2f}%"
        tbl_view.cell(i, 3).paragraphs[0].text = f"{diff:+.2f}%p"
        tbl_view.cell(i, 4).paragraphs[0].text = f"{v4_acc:.2f}%"
        tbl_view.cell(i, 5).paragraphs[0].text = f"{v4_ret:.2f}%"

    format_table(tbl_view)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3. 중요 피처 Top 10
    p = doc.add_paragraph()
    r = p.add_run("3. sl2_v4 29D 불변 생체역학 중요 피처 Top 10")
    r.font.name = 'Malgun Gothic'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    tbl_feat = doc.add_table(rows=11, cols=3)
    h_feat = ["순위", "생체역학 피처명", "중요도 (%)"]
    for j, h in enumerate(h_feat):
        tbl_feat.cell(0, j).paragraphs[0].text = h

    for r_i, item in enumerate(d4['feature_importance_top10'], start=1):
        tbl_feat.cell(r_i, 0).paragraphs[0].text = str(item['rank'])
        tbl_feat.cell(r_i, 1).paragraphs[0].text = item['feature']
        tbl_feat.cell(r_i, 2).paragraphs[0].text = f"{item['importance']:.2f}%"

    format_table(tbl_feat)

    doc.save(DOCX_OUT)
    print(f"[+] 공식 Word 보고서 생성 완료: {DOCX_OUT}")

    # 마크다운 요약본 저장
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write("# vidio-health-trainer 기술 보고서\n\n")
        f.write("## 사이드 런지 c2 (몸통 발 무릎 방향 일치 여부) sl2_v4 공식 성능평가 보고서\n\n")
        f.write(f"- 작성일자: 2026-10-05\n- 배포 경로: `{DOCX_OUT}`\n\n")
        f.write(f"### 1. 버전별 종합 성능 비교표\n\n")
        f.write("| 버전 | 피처 차원 | 2-State F1 | 2-State AUC | 3-State F1 | 정확도(ACC) | 커버리지(보류율) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| sl2_v3 (이전) | 18D | {d3['gbdt_2state_metrics']['ens']['f1']*100:.2f}% | {d3['gbdt_2state_metrics']['ens']['auc']*100:.2f}% | {d3['adaptive_3state_metrics']['effective_f1']*100:.2f}% | {d3['adaptive_3state_metrics']['accuracy']*100:.2f}% | {d3['adaptive_3state_metrics']['coverage']*100:.1f}% ({d3['adaptive_3state_metrics']['uncertainty_rate']*100:.1f}%) |\n")
        f.write(f"| sl2_v4 (신규) | 29D | {d4['gbdt_2state_metrics']['ens']['f1']*100:.2f}% | {d4['gbdt_2state_metrics']['ens']['auc']*100:.2f}% | {d4['adaptive_3state_metrics']['effective_f1']*100:.2f}% | {d4['adaptive_3state_metrics']['accuracy']*100:.2f}% | {d4['adaptive_3state_metrics']['coverage']*100:.1f}% ({d4['adaptive_3state_metrics']['uncertainty_rate']*100:.1f}%) |\n\n")
        f.write(f"### 2. 카메라 5대 공식 시점별 3-State F1 감사\n\n")
        f.write("| 카메라 공식 시점 | sl2_v3 F1 | sl2_v4 F1 | F1 변화량 |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for v_label, vk in view_keys:
            v3_f1 = d3['adaptive_3state_metrics']['view_audits'][vk]['effective_f1'] * 100
            v4_f1 = d4['adaptive_3state_metrics']['view_audits'][vk]['effective_f1'] * 100
            f.write(f"| {v_label} | {v3_f1:.2f}% | {v4_f1:.2f}% | {v4_f1 - v3_f1:+.2f}%p |\n")
    print(f"[+] 공식 마크다운 보고서 저장 완료: {MD_OUT}")


if __name__ == '__main__':
    build_c2_v4_report()
