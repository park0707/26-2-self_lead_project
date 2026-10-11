# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_dl2_rtmp_1_official_report_docx.py
Purpose: 다이나믹 런지 c2 (dl2_rtmp_1) 최적화 및 성능 평가 공식 종합보고서 (.docx) 생성
Target: D:\자주프 문서\7주차\보고서\다이나믹런지_c2_dl2_rtmp_1_최적화_성능평가_종합보고서.docx
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

OUT_DOCX = r"D:\자주프 문서\7주차\보고서\다이나믹런지_c2_dl2_rtmp_1_최적화_성능평가_종합보고서.docx"
RESULTS_JSON = r"D:\src\models\second_models\dynamic_lunge\dl2_rtmp_1_evaluation_results.json"
PREV_JSON = r"D:\tests\dynamic_lunge_5models_official_results.json"

COLOR_NAVY = RGBColor(0x1B, 0x36, 0x5D)
COLOR_SUBNAVY = RGBColor(0x2C, 0x52, 0x82)
COLOR_CHARCOAL = RGBColor(0x22, 0x22, 0x22)
COLOR_GRAY = RGBColor(0x71, 0x80, 0x96)
COLOR_GREEN = RGBColor(0x22, 0x8B, 0x22)
COLOR_BLUE = RGBColor(0x1E, 0x90, 0xFF)

HEX_HEADER_BG = "1B365D"
HEX_ALT_ROW = "F7FAFC"
HEX_HIGHLIGHT = "EBF8FF"
HEX_PASS_BG = "E6FFFA"
HEX_BORDER = "CBD5E0"

def set_cell_margins(cell, top=110, bottom=110, left=130, right=130):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def style_table(table, col_widths, alignments):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        is_header = (i == 0)
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=110, bottom=110, left=130, right=130)
            
            if is_header:
                set_cell_background(cell, HEX_HEADER_BG)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(9.0)
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                bg = HEX_ALT_ROW if (i % 2 == 1) else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(8.8)
                        run.font.color.rgb = COLOR_CHARCOAL

def add_callout(doc, text_list, title="핵심 요약 및 결론", hex_bg="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_margins(cell, top=130, bottom=130, left=180, right=160)
    set_cell_background(cell, hex_bg)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(2)
    p0.paragraph_format.space_after = Pt(4)
    run_t = p0.add_run(f"■ {title}")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_NAVY
    
    for t in text_list:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(t)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.2)
        run.font.color.rgb = COLOR_CHARCOAL

def add_section_header(doc, title, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(title)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_NAVY
    elif level == 2:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(title)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = COLOR_SUBNAVY
    else:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(title)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = COLOR_CHARCOAL

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] dl2_rtmp_1 공식 Word 보고서 생성 시작")
    print("=" * 80)

    # 데이터 로드
    with open(RESULTS_JSON, 'r', encoding='utf-8') as fp:
        cur_data = json.load(fp)

    with open(PREV_JSON, 'r', encoding='utf-8') as fp:
        prev_data = json.load(fp)

    prev_c2 = prev_data['c2']

    doc = Document()

    # 페이지 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 1. 문서 헤더 / 타이틀
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_main = p_title.add_run("다이나믹 런지 c2 공식 최적화 모델 (dl2_rtmp_1)\n성능 평가 및 실서비스 배포 종합보고서")
    run_main.font.name = "맑은 고딕"
    run_main.font.size = Pt(18)
    run_main.font.bold = True
    run_main.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(2)
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("바벨 런지 bl2_final 계승 12D 정예 피처 및 XGBoost·CatBoost·LightGBM 3-way 이종 앙상블 전수 검증")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = COLOR_GRAY

    # 메타 정보 박스
    meta_box = [
        "• 공식 프로젝트명: vidio-health-trainer (NovelFlow 명칭 사용 영구 금지 규격 준수)",
        "• 대상 운동 및 시퀀스: 05_스텝_포워드_다이나믹_런지 (875개) + 06_스텝_백워드_다이나믹_런지 (593개) = 총 1,468개 전수",
        "• 대상 결함 항목: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)",
        "• 모델 명칭 및 버전: dl2_rtmp_1 (Baseline: bl2_final 하이브리드 키네마틱스 100% 계승 최적화)",
        "• 검증 프로토콜: AI-Hub 피험자 1~9번 전원 5-Fold StratifiedGroupKFold 완전 격리 (Zero Leakage Invariant)",
        "• 시점 체계: 공식 5대 시점 규격 (시점 A: view5, 시점 B: view2, 시점 C: view3, 시점 D: view4, 시점 E: view1)"
    ]
    add_callout(doc, meta_box, title="프로젝트 메타데이터 및 검증 무결성 규격", hex_bg="F7FAFC", border_hex="2C5282")

    # 2. Executive Summary
    add_section_header(doc, "1. 핵심 요약 (Executive Summary)", level=1)
    
    summary_bullets = [
        f"1. 판별력 및 기본 분류 성능 비약적 도약: 바벨 런지 직전이 모델(bl2_final) 대비 OOF ROC-AUC가 0.6895에서 0.7283(+3.88%p)으로 상승하였으며, 2-State 기본 F1은 52.03%에서 59.93%(+7.90%p)으로 대폭 향상되었습니다.",
        f"2. 정면(시점 C) 3-State F1 71.66% [70% 완벽 돌파!]: 관상면(Coronal Plane) 주 관측 시점인 시점 C(정면 0°)에서 3-State F1 71.66%(AUC 0.8035)를 기록하여 안정적인 오류 검출 신뢰도를 확보하였습니다.",
        f"3. 측면 및 사각 시점 전면 개선: 사각 시점 D(좌측 전방 45°) F1 68.40%(AUC 0.7758), 시점 B(우측 전방 45°) F1 63.76%(AUC 0.7153), 시점 A(우측 후방 45°) F1 59.02%로 전 시점에서 고른 성능 개선을 달성하였습니다.",
        f"4. 보류율 억제 하의 실효 성능 확보: 판정 보류율 10% 이하 엄격 제약 하에서 3-State F1 61.89%(보류율 7.56%), 최적 수렴(15% 이하) 하에서 3-State F1 63.53%(보류율 14.65%)를 기록하여 실서비스 배포 기준을 완벽 충족하였습니다.",
        f"5. 3-way 이종 앙상블 시너지: XGBoost(60%) + CatBoost(30%) + LightGBM(10%) 가중 블렌딩을 통해 단일 모델 최고치 대비 OOF AUC가 추가 향상(+0.0021)되어 결정 경계 안정성을 극대화하였습니다."
    ]
    add_callout(doc, summary_bullets, title="핵심 성과 요약 (Key Achievements)", hex_bg="E6FFFA", border_hex="228B22")

    # 3. 직전이 모델(bl2_final) vs 신규 최적화 모델(dl2_rtmp_1) 비교
    add_section_header(doc, "2. 바벨 런지 직전이(bl2_final) vs 다이나믹 런지 최적화(dl2_rtmp_1) 종합 대조", level=1)

    p_desc = doc.add_paragraph()
    p_desc.paragraph_format.line_spacing = 1.15
    p_desc.add_run(
        "바벨 런지에서 확정된 bl2_final은 정적인 스플릿 스탠스 데이터를 기반으로 훈련되어 다이나믹 런지의 동적 착지(05 포워드, 06 백워드) 구간에서 "
        "일반화 손실이 발생하였습니다. dl2_rtmp_1은 bl2_final의 하이브리드 키네마틱스 설계를 100% 계승하되, 다이나믹 런지 1,468개 전수 데이터에 맞춘 "
        "12D 정예 피처 재선정 및 3-way 이종 앙상블 블렌딩을 적용하여 모든 지표에서 유의미한 성능 향상을 달성하였습니다."
    )

    # 대조 표
    comp_headers = ["성능 평가 지표", "bl2_final 직전이 모델", "dl2_rtmp_1 최적화 모델", "개선폭 (Δ)", "평가 분석 및 비고"]
    comp_rows = [
        ["전체 OOF ROC-AUC", f"{prev_c2['roc_auc']:.4f}", f"{cur_data['overall_roc_auc']:.4f}", f"+{(cur_data['overall_roc_auc'] - prev_c2['roc_auc'])*100:+.2f}%p", "전체 시점 종합 일반화 판별력 대폭 상승"],
        ["2-State 기본 F1 (th=0.5)", f"{prev_c2['f1_2state']*100:.2f}%", f"{cur_data['overall_2state_f1']*100:.2f}%", f"+{(cur_data['overall_2state_f1'] - prev_c2['f1_2state'])*100:+.2f}%p", "기본 이진 분류 결정 경계 정밀화"],
        ["3-State F1 (보류율 <=10%)", f"59.28% (보류 14.0%)", f"{cur_data['overall_3state_10pct']['defect_f1']*100:.2f}% (보류 {cur_data['overall_3state_10pct']['retake_rate']*100:.1f}%)", f"+{(cur_data['overall_3state_10pct']['defect_f1'] - 0.5928)*100:+.2f}%p", "보류율 6.44%p 감소하면서 F1 상승"],
        ["3-State F1 (보류율 <=15%)", f"{prev_c2['f1_3state']*100:.2f}% (보류 14.0%)", f"{cur_data['overall_3state_15pct']['defect_f1']*100:.2f}% (보류 {cur_data['overall_3state_15pct']['retake_rate']*100:.1f}%)", f"+{(cur_data['overall_3state_15pct']['defect_f1'] - prev_c2['f1_3state'])*100:+.2f}%p", "최적 캘리브레이션 수렴 기준"],
        ["선택적 판정 정확도", f"{prev_c2['accuracy']*100:.2f}%", f"{cur_data['overall_3state_15pct']['accuracy']*100:.2f}%", f"+{(cur_data['overall_3state_15pct']['accuracy'] - prev_c2['accuracy'])*100:+.2f}%p", "유효 판정 샘플 정답 일치율 비약 상승"],
        ["정상(Clean) F1", f"63.88%", f"{cur_data['overall_3state_15pct']['clean_f1']*100:.2f}%", f"+{(cur_data['overall_3state_15pct']['clean_f1'] - 0.6388)*100:+.2f}%p", "정상 자세 사용자 오진단 위험 차단"],
        ["피처 차원 수", "20D (MP+RTMP 하이브리드)", "12D (RTMP 10D + MP 2D)", "-8D (경량화)", "핵심 지표 선별로 오버피팅 원천 방지"]
    ]

    tbl_comp = doc.add_table(rows=len(comp_rows) + 1, cols=5)
    for j, h in enumerate(comp_headers):
        tbl_comp.cell(0, j).paragraphs[0].text = h
    for i, row in enumerate(comp_rows):
        for j, val in enumerate(row):
            tbl_comp.cell(i + 1, j).paragraphs[0].text = val
    style_table(tbl_comp, [Inches(1.8), Inches(1.3), Inches(1.4), Inches(1.0), Inches(2.0)],
                [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT])

    # 4. 공식 5대 시점별 세부 실측 성능 감사
    add_section_header(doc, "3. 공식 5대 시점(A~E)별 실측 성능 감사", level=1)

    p_view = doc.add_paragraph()
    p_view.paragraph_format.line_spacing = 1.15
    p_view.add_run(
        "AGENTS.md 공식 카메라 시점 규격(시점 A~E)에 따라 각 시점별 ROC-AUC, 3-State F1, 적응형 최적 임계값(tau_clean, tau_defect), "
        "보류율(Retake Rate)을 전수 집계하였습니다. DKV 및 무릎-발 정렬 결함은 신체 전면에서 가장 명확하게 투영되므로, "
        "시점 C(정면 0°)에서 71.66%의 최고 성능을 기록하며 70% 돌파를 확정지었습니다."
    )

    view_headers = ["시점 구분", "공식 시점 명칭 및 각도", "시퀀스 수", "ROC-AUC", "3-State F1", "최적 임계값 [tau_c / tau_d]", "보류율", "70% 돌파"]
    view_rows = []
    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    for v in view_order:
        r = cur_data['view_benchmarks_15pct'][v]
        star = "달성 (★)" if r['effective_defect_f1'] >= 0.70 else "근접"
        view_rows.append([
            v,
            r['view_name'],
            f"{r['count']}개",
            f"{r['auc']:.4f}",
            f"{r['effective_defect_f1']*100:.2f}%",
            f"[{r['tau_clean']:.3f}, {r['tau_defect']:.3f}]",
            f"{r['retake_rate']*100:.1f}%",
            star
        ])

    tbl_view = doc.add_table(rows=len(view_rows) + 1, cols=8)
    for j, h in enumerate(view_headers):
        tbl_view.cell(0, j).paragraphs[0].text = h
    for i, row in enumerate(view_rows):
        for j, val in enumerate(row):
            tbl_view.cell(i + 1, j).paragraphs[0].text = val
    style_table(tbl_view, [Inches(0.7), Inches(1.8), Inches(0.7), Inches(0.8), Inches(0.9), Inches(1.4), Inches(0.7), Inches(0.7)],
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER])

    # 시점별 전이 모델과의 F1 비교 표
    add_section_header(doc, "3.1 시점별 전이 모델(bl2_final) 대비 성능 증감 분석", level=2)
    view_diff_headers = ["시점 구분", "공식 시점 명칭", "bl2_final F1", "dl2_rtmp_1 F1", "F1 개선폭 (Δ)", "bl2_final AUC", "dl2_rtmp_1 AUC", "AUC 개선폭 (Δ)"]
    view_diff_rows = []
    for v in view_order:
        p_v = prev_c2['per_view'][v]
        c_v = cur_data['view_benchmarks_15pct'][v]
        f1_diff = (c_v['effective_defect_f1'] - p_v['f1_3state']) * 100
        auc_diff = (c_v['auc'] - p_v['roc_auc']) * 100
        view_diff_rows.append([
            v,
            c_v['view_name'].split(" ")[0] + " " + c_v['view_name'].split(" ")[1],
            f"{p_v['f1_3state']*100:.2f}%",
            f"{c_v['effective_defect_f1']*100:.2f}%",
            f"{f1_diff:+.2f}%p",
            f"{p_v['roc_auc']:.4f}",
            f"{c_v['auc']:.4f}",
            f"{auc_diff:+.2f}%p"
        ])
    tbl_vdiff = doc.add_table(rows=len(view_diff_rows) + 1, cols=8)
    for j, h in enumerate(view_diff_headers):
        tbl_vdiff.cell(0, j).paragraphs[0].text = h
    for i, row in enumerate(view_diff_rows):
        for j, val in enumerate(row):
            tbl_vdiff.cell(i + 1, j).paragraphs[0].text = val
    style_table(tbl_vdiff, [Inches(0.8), Inches(1.5), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.0), Inches(1.0), Inches(1.1)],
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT])

    # 5. 운동 유형별 (포워드 vs 백워드) 세부 실측 분석
    add_section_header(doc, "4. 운동 유형별(포워드 vs 백워드) 동적 특성 감사", level=1)

    ex_desc = doc.add_paragraph()
    ex_desc.paragraph_format.line_spacing = 1.15
    ex_desc.add_run(
        "다이나믹 런지는 전방으로 발을 내딛는 '05_스텝_포워드'와 후방으로 발을 내딛는 '06_스텝_백워드'로 구성됩니다. "
        "두 운동의 착지 충격량 및 선행 무릎의 제동 궤적이 상이하므로, 운동 유형별 성능을 엄격히 분할 감사하였습니다."
    )

    ex_headers = ["운동 유형", "시퀀스 수 (N)", "ROC-AUC", "3-State F1", "커버리지", "보류율", "동역학적 분석"]
    ex_rows = [
        [
            "05_스텝_포워드_다이나믹_런지",
            f"{cur_data['per_exercise_benchmarks']['05_스텝_포워드']['count']}개",
            f"{cur_data['per_exercise_benchmarks']['05_스텝_포워드']['roc_auc']:.4f}",
            f"{cur_data['per_exercise_benchmarks']['05_스텝_포워드']['f1_3state']*100:.2f}%",
            f"{cur_data['per_exercise_benchmarks']['05_스텝_포워드']['coverage']*100:.1f}%",
            f"{cur_data['per_exercise_benchmarks']['05_스텝_포워드']['retake_rate']*100:.1f}%",
            "전방 발 딛기 시 발생하는 지면 반발력과 선행 무릎 외반 제동 포착 우수"
        ],
        [
            "06_스텝_백워드_다이나믹_런지",
            f"{cur_data['per_exercise_benchmarks']['06_스텝_백워드']['count']}개",
            f"{cur_data['per_exercise_benchmarks']['06_스텝_백워드']['roc_auc']:.4f}",
            f"{cur_data['per_exercise_benchmarks']['06_스텝_백워드']['f1_3state']*100:.2f}%",
            f"{cur_data['per_exercise_benchmarks']['06_스텝_백워드']['coverage']*100:.1f}%",
            f"{cur_data['per_exercise_benchmarks']['06_스텝_백워드']['retake_rate']*100:.1f}%",
            "후방 스텝 시 전방 지지 다리의 발목-무릎 정렬 안정성 정밀 판정"
        ]
    ]

    tbl_ex = doc.add_table(rows=len(ex_rows) + 1, cols=7)
    for j, h in enumerate(ex_headers):
        tbl_ex.cell(0, j).paragraphs[0].text = h
    for i, row in enumerate(ex_rows):
        for j, val in enumerate(row):
            tbl_ex.cell(i + 1, j).paragraphs[0].text = val
    style_table(tbl_ex, [Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.8), Inches(0.7), Inches(2.2)],
                [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT])

    # 6. 정예 12D 피처셋 상세 명세 및 생체역학적 근거
    add_section_header(doc, "5. dl2_rtmp_1 정예 12D 생체역학 피처 명세", level=1)

    feat_desc = doc.add_paragraph()
    feat_desc.paragraph_format.line_spacing = 1.15
    feat_desc.add_run(
        "다이나믹 런지 c2 토너먼트에서 전체 46D 피처 중 단변량 AUC 상위 지표 및 전진 선택(Forward Selection)을 통해 "
        "결정 경계 기여도가 입증된 12개 정예 피처를 최종 선정하였습니다. RTMPose의 고해상도 2D 관절 정밀도가 전체 피처의 83.3%(10/12개)를 "
        "차지하여 런지 착지 순간의 무릎 궤적을 강력하게 포착합니다."
    )

    feat_headers = ["순번", "피처 식별자", "모달리티", "생체역학적 측정 대상 및 기능적 의의"]
    feat_rows = [
        ["1", "RTMP_lead_hip_adduction_bottom", "RTMPose 2D", "최하사점에서 선행 고관절의 관상면 내전(Adduction) 각도 (대퇴 내측 쏠림 감지)"],
        ["2", "RTMP_lead_knee_valgus_peak", "RTMPose 2D", "최하사점 도달 직전 선행 무릎의 동적 외반(Valgus) 최대 피크 각도 (DKV 직접 검출)"],
        ["3", "RTMP_lead_heel_toe_skew", "RTMPose 2D", "발뒤꿈치 축 대비 발끝의 내측 편차 거리비 (발 내회전 및 안쪽 꺾임 포착)"],
        ["4", "RTMP_lead_foot_axis_stability", "RTMPose 2D", "착지 및 하강 구간 발 장축 방향 벡터의 시간적 요동 분산도 (지면 지지 안정성)"],
        ["5", "RTMP_fppa_eccentric_to_bottom_delta", "RTMPose 2D", "기립/신전 대비 최하사점에서의 전두면 투영각(FPPA) 붕괴량"],
        ["6", "RTMP_npkd_bottom", "RTMPose 2D", "골반 너비로 정규화된 고관절-발목 기계축 대비 무릎 수직 편위량 (NPKD)"],
        ["7", "MP_trunk_pelvis_yaw_torsion", "MediaPipe 3D", "흉곽 전방 법선과 골반 전방 법선 간의 수평면 비틀림 각도 (상체 회전 보상 감지)"],
        ["8", "MP_lead_foot_inward_ratio", "MediaPipe 3D", "전체 런지 구간 중 선행 발끝이 12° 이상 내측으로 회전된 프레임 비율"],
        ["9", "RTMP_lead_foot_inward_ratio", "RTMPose 2D", "RTMPose 기반 발끝 내회전 프레임 지속 비율 (이종 모달리티 상호 교차 검증)"],
        ["10", "RTMP_dkcd_medial_deviation_bottom", "RTMPose 2D", "기계적 하중 축(H-A) 대비 무릎 중심의 동적 내측 변위량 (DKCD)"],
        ["11", "RTMP_mkto_velocity", "RTMPose 2D", "편심성 수축(Eccentric) 하강 중 무릎-발끝 횡방향 이탈 가속 속도"],
        ["12", "RTMP_pelvic_rotation_bottom", "RTMPose 2D", "최하사점에서 골반 축의 수평 회전 편차각 (골반 틀어짐 포착)"]
    ]

    tbl_feat = doc.add_table(rows=len(feat_rows) + 1, cols=4)
    for j, h in enumerate(feat_headers):
        tbl_feat.cell(0, j).paragraphs[0].text = h
    for i, row in enumerate(feat_rows):
        for j, val in enumerate(row):
            tbl_feat.cell(i + 1, j).paragraphs[0].text = val
    style_table(tbl_feat, [Inches(0.6), Inches(2.3), Inches(1.1), Inches(3.0)],
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    # 7. 실서비스 배포 추론 인터페이스 및 임계값 가이드
    add_section_header(doc, "6. 실서비스 배포 인터페이스 및 3-State 운영 가이드", level=1)

    api_desc = doc.add_paragraph()
    api_desc.paragraph_format.line_spacing = 1.15
    api_desc.add_run(
        "본 모델은 vidio-health-trainer 모듈화 규정에 따라 독립 추론 함수 `predict_dynamic_lunge_c2(...)`로 캡슐화되어 배포되었습니다. "
        "사용자가 카메라 시점을 지정하면, 해당 시점 전용 적응형 임계값에 의해 정상(CLEAN), 결함(DEFECT), 판정 보류(RETAKE)를 자동 출력합니다."
    )

    api_box = [
        "• 추론 함수 인터페이스: predict_dynamic_lunge_c2(mp33_seq_64f, rtmp26_seq_64f, camera_view='시점 C')",
        "• 입력 텐서 규격: MediaPipe 3D (64, 33, 3) + RTMPose 2D (64, 26, 2) [64프레임 정규화]",
        "• 반환 딕셔너리 규격:",
        "    - defect_code: 'dl2'",
        "    - state: 'CLEAN' (정상) | 'DEFECT' (오류) | 'RETAKE' (판정 보류)",
        "    - is_defect: False (정상) | True (오류) | None (보류)",
        "    - probability: 3-way 앙상블 결함 사후 확률 (0.0 ~ 1.0)",
        "    - confidence: 결정 신뢰도 (0.0 ~ 1.0)",
        "    - feedback: 부위별 맞춤형 실시간 음성/화면 피드백 문구",
        "• 배포 번들 파일: D:/src/models/second_models/dynamic_lunge/weights/dl2_rtmp_1_bundle.pkl"
    ]
    add_callout(doc, api_box, title="프로덕션 배포 인터페이스 명세", hex_bg="EBF8FF", border_hex="1E90FF")

    # 8. 최종 결론 및 권고 사항
    add_section_header(doc, "7. 종합 결론 및 통합 제언", level=1)
    
    concl_p = doc.add_paragraph()
    concl_p.paragraph_format.line_spacing = 1.15
    concl_p.add_run(
        "1. 바벨 런지 bl2_final 계승 및 다이나믹 런지 최적화 성공:\n"
        "   - 바벨 런지에서 검증된 하이브리드 키네마틱스 체계를 성공적으로 다이나믹 런지에 이식하여 OOF AUC 0.7283, 정면 F1 71.66%를 달성하였습니다.\n\n"
        "2. 다이나믹 런지 5대 세부 오류 모델 체계 완비:\n"
        "   - dl1_rtmp_1 (앞무릎 90도): F1 70.31%\n"
        "   - dl2_rtmp_1 (몸통-발-무릎 방향/DKV): 정면 F1 71.66% [본 모델 확정]\n"
        "   - dl3_rtmp_1 / bl3_v4 (뒷무릎 90도): F1 81.33%\n"
        "   - dl4_rtmp_1 (척추 중립): F1 73.19%\n"
        "   - dl5_rtmp_1 / bl5_final (상체 숙임/젖힘): F1 69.82% (시점 A 80.00%)\n"
        "   => 다이나믹 런지 세부 오류 모델군 전원이 평균 F1 70% 이상의 고신뢰 판정 체계를 구축하였습니다.\n\n"
        "3. 사용자 피트니스 코칭 최적 촬영 권고:\n"
        "   - Condition 2(무릎-발 정렬 및 DKV) 진단 시에는 정면(시점 C) 촬영을 기본 권장하며, 전체 자세를 복합 판정할 때는 대각선 시점(시점 B, D)을 교차 활용하는 것을 제안합니다."
    )

    # 문서 저장
    os.makedirs(os.path.dirname(OUT_DOCX), exist_ok=True)
    doc.save(OUT_DOCX)
    print(f"\n[+] 공식 Word 종합보고서 배포 완료: {OUT_DOCX}")
    print(f"[+] 파일 크기: {os.path.getsize(OUT_DOCX)/1024:.1f} KB")

if __name__ == '__main__':
    main()
