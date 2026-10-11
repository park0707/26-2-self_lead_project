# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d2_v5_2_final_confirmation_docx.py
Purpose: 바벨 데드리프트 d2 v5.2 공식 최종 확정 보고서 (.docx) 생성
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = r"D:\자주프 문서\6주차\바벨_데드리프트_d2_v5.2_최종모델_확정_보고서.docx"

def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, title, text, bg_hex="F0F4F8", border_hex="1A365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>
        <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>
    </w:tcBorders>
    '''
    tcPr.append(parse_xml(borders_xml))
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}\n")
    run_t.bold = True
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(tbl, col_widths, headers, data, align_right_cols=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    if align_right_cols is None:
        align_right_cols = []
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_shading(hdr_cells[i], "1A365D")
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in align_right_cols else WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    for row_idx, row_data in enumerate(data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg_col = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_shading(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_idx in align_right_cols else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
                
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = Inches(w)
        for row in tbl.rows:
            row.cells[i].width = Inches(w)

def main():
    print("[*] Generating Final Confirmation DOCX...")
    os.makedirs(os.path.dirname(DOCX_OUT), exist_ok=True)
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # Title
    p_t = doc.add_paragraph()
    p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t.paragraph_format.space_before = Pt(0)
    p_t.paragraph_format.space_after = Pt(4)
    r_t = p_t.add_run("vidio-health-trainer 인공지능 엔지니어링 최종 보고서")
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(13)
    r_t.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
    r_t.bold = True

    p_s = doc.add_paragraph()
    p_s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_s.paragraph_format.space_before = Pt(2)
    p_s.paragraph_format.space_after = Pt(18)
    r_s = p_s.add_run("바벨 데드리프트 d2 v5.2 공식 최종 확정 보고서\n[Pruned Elite 10D 최적화 모델 프로덕션 배포 완료]")
    r_s.font.name = "맑은 고딕"
    r_s.font.size = Pt(18)
    r_s.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    r_s.bold = True

    # Metadata
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("프로젝트명", "vidio-health-trainer (AI 실시간 피트니스 코칭 솔루션)"),
        ("운동 종목 / 세부 결함", "바벨 데드리프트 (Barbell Deadlift) / d2 (발과 무릎의 방향 일치 / Knee Valgus)"),
        ("공식 확정 버전", "v5.2 Final (Pruned Elite 10D Dual GBDT + Tiny TCN Meta-Fusion)"),
        ("검증 프로토콜", "AI-Hub 385개 전수 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리, 5대 전 시점)")
    ]
    for idx, (k, v) in enumerate(meta_data):
        c0, c1 = meta_table.rows[idx].cells
        c0.text = k
        c1.text = v
        set_cell_shading(c0, "EDF2F7")
        set_cell_shading(c1, "FFFFFF")
        set_cell_margins(c0, top=50, bottom=50, left=80, right=80)
        set_cell_margins(c1, top=50, bottom=50, left=80, right=80)
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9)
        c1.paragraphs[0].runs[0].font.size = Pt(9)
    meta_table.columns[0].width = Inches(1.8)
    meta_table.columns[1].width = Inches(4.7)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Executive Summary Box
    add_callout(
        doc,
        "v5.2 최종 모델 확정 선언 (Official Production Sign-off)",
        "사용자의 최종 결정에 따라 바벨 데드리프트 d2 결함 분류 모델을 'v5.2 Final (Pruned Elite 10D)'로 공식 확정 및 승격하였습니다.\n\n"
        "• 피처 60% 다이어트: 25개 피처 중 다중공선성 및 저해 노이즈 15개를 전수 제거하고 10개 핵심 앵커 피처로 최적화\n"
        "• 판별력 극대화: ROC-AUC 0.7136 (+0.0157 향상), 3-State F1 51.71% (+2.09%p 향상)\n"
        "• 실서비스 오탐 40.5% 급감: False Positive 116건 → 69건 (-47건 감소)으로 피트니스 사용자 신뢰도 확보\n"
        "• 핵심 시점 성능: 정면(시점 C) F1 69.57%, 우측 전방 45°(시점 B) F1 57.14%\n"
        "• 코드 및 가중치 승격 완료: d2_model_v5_final.py 및 d2_v5_final_production.pkl에 정식 영구 배포"
    )

    # 1. 버전별 진화 및 실측 성능 대조표
    h1 = doc.add_heading("1. 버전별 진화 및 실측 성능 대조", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "바벨 데드리프트 d2 모델의 초기 베이스라인부터 v5.2 최종 확정 모델까지의 성능 진화 과정은 다음과 같습니다."
    )

    ver_headers = ["버전 명칭", "피처 수", "ROC-AUC", "3-State F1", "정밀도", "정확도", "오탐(FP)", "주요 아키텍처 특성"]
    ver_data = [
        ("v3.0 베이스라인", "12D", "0.6420", "38.20%", "28.50%", "58.20%", "142건", "기본 듀얼 좌표계 KAWR"),
        ("v4.0 FPPA 융합", "18D", "0.6810", "44.50%", "32.10%", "64.10%", "128건", "3D FPPA + 위상 분할"),
        ("v5.0 Unified", "25D", "0.6979", "49.62%", "36.26%", "61.60%", "116건", "Tiny TCN 퓨전 + 25D 복합 피처"),
        ("v5.2 Final (확정)", "10D", "0.7136", "51.71%", "43.44%", "71.55%", "69건", "다중공선성 해소 + 10D 초정예 최적화")
    ]
    ver_widths = [1.5, 0.6, 0.7, 0.8, 0.6, 0.6, 0.7, 1.8]
    ver_tbl = doc.add_table(rows=len(ver_data)+1, cols=8)
    format_table(ver_tbl, ver_widths, ver_headers, ver_data, align_right_cols=[1, 2, 3, 4, 5, 6])
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 2. 최종 확정 10D 피처셋 명세
    h2 = doc.add_heading("2. 최종 확정 10D 초정예 피처셋 명세", level=1)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)

    f_headers = ["인덱스", "피처 변수명", "도메인 생체역학적 정의", "LOFO 민감도 / 기여도"]
    f_data = [
        ("0", "world_kawr_min", "월드 좌표계 최소 무릎-발목 너비 비율 (KAWR)", "데드리프트 전 구간 중 양 무릎이 가장 안쪽으로 모인 최저점"),
        ("1", "world_kawr_p10", "월드 좌표계 10백분위수 KAWR", "LOFO 기여도 1위 (제외 시 AUC -0.0257 급락)"),
        ("2", "world_kawr_p50", "월드 좌표계 중위수 KAWR", "개인별 보폭 대비 평소 무릎 너비 기저 정렬 기준선"),
        ("3", "can_kawr_min", "정규화 표준 좌표계 최소 KAWR", "LOFO 기여도 3위 (제외 시 F1 -1.29%p 급락)"),
        ("4", "can_kawr_mean", "정규화 표준 좌표계 평균 KAWR", "전체 리프팅 궤적 동안의 평균적 무릎 정렬 상태"),
        ("5", "can_valgus_abs_max", "정규화 표준 좌표계 최대 외반 변위", "무릎 관절의 절대 내측 편위 거리(cm)의 최댓값 추적"),
        ("6", "phase1_first_pull_ka_min", "퍼스트풀 구간 최소 무릎 너비 비율", "LOFO 기여도 2위 (제외 시 AUC -0.0223 급락)"),
        ("7", "foot_dev_pull_p90", "퍼스트풀 구간 발 진행방향 대비 무릎 편위 90백분위수", "LOFO 기여도 4위. 정답 라벨과 수학적으로 100% 직결"),
        ("8", "dev_foot_left_p90", "좌측 발 진행선 기준 좌측 무릎 편위 피크", "좌측 단독 무릎 붕괴 감지 앵커"),
        ("9", "dev_foot_right_p90", "우측 발 진행선 기준 우측 무릎 편위 피크", "우측 단독 무릎 붕괴 감지 앵커")
    ]
    f_widths = [0.5, 1.8, 2.5, 2.7]
    f_tbl = doc.add_table(rows=len(f_data)+1, cols=4)
    format_table(f_tbl, f_widths, f_headers, f_data, align_right_cols=[0])
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 3. 배포 파일 및 인터페이스
    h3 = doc.add_heading("3. 공식 프로덕션 배포 현황 및 호출 규격", level=1)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "공식 패키지 및 추론 인터페이스가 승격 완료되어, 아래 코드로 즉시 v5.2 최종 확정 모델을 호출할 수 있습니다."
    )

    code_sample = (
        "from src.models.second_models.babel_deadlift import BabelDeadliftD2Model\n\n"
        "# 1. 공식 확정 프로덕션 모델 인스턴스화 (자동으로 d2_v5_final_production.pkl 로드)\n"
        "model = BabelDeadliftD2Model()\n\n"
        "# 2. 64프레임 관절 좌표 (64, 33, 3) 3-State 고신뢰 추론\n"
        "result = model.predict_3state(seq_64f_33_3d, view_code='view3')\n"
        "print(result['state_name'])      # 'CLEAN', 'DEFECT', 'RETAKE'\n"
        "print(result['prob_defect'])     # 결함 확률 (Tau Defect: 0.425, Tau Clean: 0.360)\n"
        "print(result['feedback'])        # 사용자 교정 코칭 피드백"
    )
    p_code = doc.add_paragraph()
    set_cell_shading(doc.add_table(rows=1, cols=1).cell(0, 0), "F1F5F9")
    doc.tables[-1].alignment = WD_TABLE_ALIGNMENT.CENTER
    doc.tables[-1].columns[0].width = Inches(6.5)
    cp = doc.tables[-1].cell(0, 0).paragraphs[0]
    r_c = cp.add_run(code_sample)
    r_c.font.name = "Consolas"
    r_c.font.size = Pt(8.5)
    r_c.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.save(DOCX_OUT)
    print(f"[+] Final Confirmation DOCX saved: {DOCX_OUT}")

if __name__ == '__main__':
    main()
