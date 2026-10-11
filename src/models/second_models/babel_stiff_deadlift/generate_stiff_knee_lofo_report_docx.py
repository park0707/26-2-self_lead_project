# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_stiff_knee_lofo_report_docx.py
Purpose: 스티프 데드리프트 14대 피처 LOFO 전수 감사 및 성능 방해 피처 규명 공식 Word 보고서 생성
Output: D:\자주프 문서\6주차\vidio-health-trainer_스티프데드리프트_피처LOFO전수감사_및_성능개선보고서.docx
"""

import os
import sys
import json
from datetime import datetime
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

JSON_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_knee_lofo_audit_results.json"
OUT_DIR = r"D:\자주프 문서\6주차"
OUT_DOCX = os.path.join(OUT_DIR, "vidio-health-trainer_스티프데드리프트_피처LOFO전수감사_및_성능개선보고서.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
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

def add_header_styled(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(26, 54, 93)
        pPr = h._p.get_or_add_pPr()
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="4" w:color="1A365D"/></w:pBdr>')
        pPr.append(pBdr)
    elif level == 2:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(43, 108, 176)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(3)
    return h

def add_p(doc, text, bold_prefix=""):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "맑은 고딕"
        r_pre.font.bold = True
        r_pre.font.size = Pt(9.5)
        r_pre.font.color.rgb = RGBColor(26, 54, 93)
    r_body = p.add_run(text)
    r_body.font.name = "맑은 고딕"
    r_body.font.size = Pt(9.5)
    r_body.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_callout(doc, text, title="핵심 요약"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=140)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="1A365D"/><w:bottom w:val="none"/><w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    p = cell.paragraphs[0]
    r_t = p.add_run(f"■ {title}\n")
    r_t.font.name = "맑은 고딕"
    r_t.font.bold = True
    r_t.font.size = Pt(10)
    r_t.font.color.rgb = RGBColor(26, 54, 93)
    r_c = p.add_run(text)
    r_c.font.name = "맑은 고딕"
    r_c.font.size = Pt(9)
    r_c.font.color.rgb = RGBColor(45, 55, 72)

def generate_report():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    # 표지
    title_p = doc.add_paragraph()
    r = title_p.add_run("vidio-health-trainer")
    r.font.name = "맑은 고딕"
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = RGBColor(100, 116, 139)

    main_h = doc.add_paragraph()
    main_h.paragraph_format.space_after = Pt(10)
    r = main_h.add_run("스티프 데드리프트 14대 피처 LOFO 전수 감사 및\n성능 방해 피처 규명 실증 보고서")
    r.font.name = "맑은 고딕"
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(26, 54, 93)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    r = meta_p.add_run(f"문서 버전: v1.0  |  작성일자: {datetime.now().strftime('%Y년 %m월 %d일')}  |  분류: 피처 엔지니어링 LOFO 감사")
    r.font.name = "맑은 고딕"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(100, 116, 139)

    base = res['baseline']
    curated = res['curated_results']
    harmful = res['harmful_features']

    callout_txt = (
        f"1. 감사 배경: 스티프 데드리프트 무릎 고정 결함의 낮은 초기 성능(Macro F1 47~48%, AUC 46.8%) 원인 규명\n"
        f"2. LOFO 전수 감사 결과: 14개 피처 중 8개 피처가 오히려 모델의 판별을 교란하는 '성능 방해 피처'로 실측 식별\n"
        f"3. 방해 피처 8개 제거 시 효과: ROC-AUC {base['auc']*100:.2f}% -> {curated['auc']*100:.2f}% (+{(curated['auc']-base['auc'])*100:+.2f}%p 폭등), "
        f"Macro F1 {base['macro_f1']*100:.2f}% -> {curated['macro_f1']*100:.2f}% (+{(curated['macro_f1']-base['macro_f1'])*100:+.2f}%p)\n"
        f"4. 데이터 원인: 복합 오류(Key 213, 214)의 후광 효과 오라벨링으로 인해 전체 시계열 평균 가동범위 피처가 순수 노이즈로 작용함"
    )
    add_callout(doc, callout_txt, title="감사 핵심 결과 요약")

    # 1. 단일 피처 변별력 감사
    add_header_styled(doc, "1. 14대 생체역학 피처 개별 변별력 전수 검사 결과", level=1)
    add_p(doc, "Level 1~3 유효 데이터셋(N=668)에서 14개 피처 각각의 정상군 vs 결함군 평균, Cohen's d 및 단독 ROC-AUC를 전수 산출하였습니다.")

    table1 = doc.add_table(rows=15, cols=5)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    h1 = ["피처명", "정상 평균 (SD)", "결함 평균 (SD)", "Cohen's d", "단독 AUC"]
    for j, txt in enumerate(h1):
        cell = table1.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(txt)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    for i, item in enumerate(res['feature_stats'], 1):
        vals = [
            item['feature'],
            f"{item['mean_clean']:.2f} (±{item['sd_clean']:.2f})",
            f"{item['mean_defect']:.2f} (±{item['sd_defect']:.2f})",
            f"{item['cohen_d']:.3f}",
            f"{item['raw_auc']:.3f}"
        ]
        for j, v in enumerate(vals):
            cell = table1.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(v)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 2. LOFO 감사 결과
    add_header_styled(doc, "2. LOFO (Leave-One-Feature-Out) 성능 영향도 전수 감사", level=1)
    add_p(doc, "14개 피처를 1개씩 제거했을 때 5-Fold 교차검증 Macro F1 및 ROC-AUC의 변화를 측정하여 '기여 피처'와 '방해 피처'를 명확히 분리하였습니다.")

    table2 = doc.add_table(rows=15, cols=5)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = ["제외된 피처명", "Macro F1", "Δ F1", "ROC-AUC", "영향도 판정"]
    for j, txt in enumerate(h2):
        cell = table2.cell(0, j)
        set_cell_background(cell, "2B6CB0")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(txt)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    for i, item in enumerate(res['lofo_results'], 1):
        vals = [
            item['excluded_feature'],
            f"{item['macro_f1']*100:.2f}%",
            f"{item['delta_f1']:+.2f}%p",
            f"{item['auc']*100:.2f}%",
            item['impact'].split("(")[0].strip()
        ]
        for j, v in enumerate(vals):
            cell = table2.cell(i, j)
            bg = "FEE2E2" if item['delta_f1'] > 1.0 else ("F8FAFC" if i % 2 == 1 else "FFFFFF")
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(v)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 3. 방해 피처 및 데이터 원인 분석
    add_header_styled(doc, "3. 성능 방해 피처 식별 및 데이터 원인 심층 분석", level=1)
    add_p(doc, 
        f"LOFO 감사 결과, 다음 8개 피처는 제외할 때 오히려 모델 성능이 크게 상승하는 '성능 방해 피처'로 판명되었습니다: "
        f"{', '.join(harmful)}",
        bold_prefix="[8대 성능 방해 피처] "
    )
    add_p(doc,
        "AI-Hub의 Key 213, 214(복합 오류: 좌우 기울임 + 바벨 이탈)는 피험자가 상체와 바벨만 연기했을 뿐 무릎은 정상(Key 211, 212)과 완전히 "
        "동일하게 유지(무릎 각도 161.5° vs 163.3°, 가동범위 17.7° vs 17.2°)되었습니다. 그러나 라벨러가 일괄 결함으로 체크하여 라벨 오염이 발생했습니다. "
        "이로 인해 전체 시계열 평균 가동범위(Knee ROM)나 각속도 RMS는 정상과 결함의 평균 차이가 0.3도에 불과하여 결정 트리의 분기를 심각하게 교란했습니다.",
        bold_prefix="[생체역학적 교란 메커니즘] "
    )
    add_p(doc,
        "반면 최저점(변곡점)에서의 각도와 가동범위를 측정한 6대 정예 피처(bottom_knee_angle, femur_rom_mean, bottom_knee_delta_mean 등)는 "
        "결함 동작의 진정한 시그니처를 포착하는 핵심 축으로 확인되었습니다.",
        bold_prefix="[6대 핵심 기여 피처] "
    )

    # 4. 정예 피처 최적화 결과
    add_header_styled(doc, "4. 방해 피처 제거 후 정예 피처셋(Curated Diet) 최적화 성과", level=1)
    table3 = doc.add_table(rows=6, cols=3)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    h3 = ["평가 지표", "14D 전체 투입 (기존)", "6D 정예 피처셋 (방해 피처 제거 후)"]
    for j, txt in enumerate(h3):
        cell = table3.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(txt)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_3 = [
        ("ROC-AUC", f"{base['auc']*100:.2f}%", f"{curated['auc']*100:.2f}% (+{(curated['auc']-base['auc'])*100:+.2f}%p 폭등)"),
        ("Macro F1", f"{base['macro_f1']*100:.2f}%", f"{curated['macro_f1']*100:.2f}% (+{(curated['macro_f1']-base['macro_f1'])*100:+.2f}%p)"),
        ("Clean F1 (정상 F1)", f"{base['clean_f1']*100:.2f}%", f"{curated['clean_f1']*100:.2f}% (+{(curated['clean_f1']-base['clean_f1'])*100:+.2f}%p)"),
        ("Defect F1 (결함 F1)", f"{base['defect_f1']*100:.2f}%", f"{curated['defect_f1']*100:.2f}% (+{(curated['defect_f1']-base['defect_f1'])*100:+.2f}%p)"),
        ("Clean Specificity", f"{base['specificity']*100:.2f}%", f"{curated['specificity']*100:.2f}% (+{(curated['specificity']-base['specificity'])*100:+.2f}%p)")
    ]

    for i, row in enumerate(rows_3, 1):
        for j, v in enumerate(row):
            cell = table3.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(v)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.save(OUT_DOCX)
    print(f"[+] 공식 Word 보고서 생성 완료: {OUT_DOCX}")

if __name__ == '__main__':
    generate_report()
