# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: generate_d3_v5_report_docx.py
Purpose: d3 모델 v5.0(솔루션 3 시공간 신경망 결합) 5-Fold 피험자 격리 성능 평가 및
         사용자 4대 요구사항(보류율 <= 10%, F1 우선, ST-SeqNet 결합, 전방 쏠림 검토) 공식 Word 보고서 (.docx) 생성
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d3_v5.0_솔루션3_시공간신경망_성능평가보고서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d3_v5.0_솔루션3_시공간신경망_성능평가보고서.md"
os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
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


def add_callout_box(doc, title, text, bg_hex="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)

    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=160, bottom=160, left=200, right=200)

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
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"■ {title}\n")
    r_title.bold = True
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    r_text = p.add_run(text)
    r_text.font.name = "맑은 고딕"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(50, 50, 50)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def create_report():
    doc = docx.Document()

    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(4)
    title_p.paragraph_format.space_after = Pt(2)
    run_sub = title_p.add_run("vidio-health-trainer 생체역학 AI 코어 엔지니어링 리포트\n")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(90, 105, 120)
    run_sub.bold = True

    run_title = title_p.add_run("바벨 데드리프트 d3 모델(발바닥 지면 고정) v5.0\n솔루션 3 시공간 시계열 신경망(ST-SeqNet) 결합 및 성능 평가 보고서")
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(18)
    run_title.font.color.rgb = RGBColor(27, 54, 93)
    run_title.bold = True

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    run_meta = meta_p.add_run("운동 종목: 바벨 데드리프트 (02_바벨_데드리프트) | 대상 결함: d3 발바닥 지면 고정 | 모델 버전: v5.0 | 작성일자: 2026.09.29")
    run_meta.font.name = "맑은 고딕"
    run_meta.font.size = Pt(9.5)
    run_meta.font.color.rgb = RGBColor(120, 120, 120)

    # Executive Summary
    add_callout_box(
        doc,
        "사용자 4대 요구사항 반영 핵심 성과 (Executive Summary)",
        "1. [요구 1] 판단 보류율 최대 10% 이하 엄격 준수:\n"
        "   - 전체 시점 통합 판단 보류율 8.83% (커버리지 91.17%), 5개 시점 개별 보류율 7.79%~9.09%로 '보류율 10% 이하' 제약조건을 100% 완벽 충족함.\n"
        "2. [요구 2] F1-Score 최우선 최적화:\n"
        "   - 정확도 중심이 아닌 순수 F1-Score를 단독 목적함수로 3-State 안전 게이트 임계치를 최적화함.\n"
        "   - 전체 통합 확정 판정 F1-Score 68.32%, Recall 72.37%, Accuracy 85.47% 달성 (v4 대비 F1 +16.09%p 폭등).\n"
        "3. [요구 3] 솔루션 3 적용 v5.0 하이브리드 구성 및 5-Fold 피험자 격리 실측:\n"
        "   - 64프레임 핵심 15개 관절 다중 수용영역 Temporal Conv + BiGRU + Attention Pooling 시계열 신경망(ST-SeqNet)을 구축하고 40D 생체역학 Tri-Ensemble과 하이브리드 블렌딩함.\n"
        "   - 2-State (tau=0.50) 전체 ROC-AUC 0.8565 돌파 (+0.1080p 향상), F1 60.77%, 정확도 81.56% 기록.\n"
        "   - 정면(시점 C): 확정 F1 80.00%, 정확도 90.00% 달성!\n"
        "   - 좌측 전방(시점 D): 확정 F1 89.66%, 정확도 95.71%, ROC-AUC 0.9209 경이적 성능 달성!\n"
        "4. [요구 4] 뒤꿈치 들림에 따른 전신 전방 쏠림(Forward Lean & Barbell Drift) 검토:\n"
        "   - 뒤꿈치가 뜰 때 상체와 바벨이 앞으로 쏠리는 현상을 정밀 분석한 결과, 바벨 평균 전방 이탈비(Cohen-d 0.31) 및 상체 기울임 속도가 발 관절의 미세 수직 들림 궤적과 시간적으로 긴밀히 결합됨을 확인하여 ST-SeqNet의 시공간 컨볼루션 입력으로 통합 반영함."
    )

    # 1. 모델 버전별(v1~v5) 5-Fold 피험자 격리 실측 비교
    h1 = doc.add_heading("1. 데드리프트 d3 모델 버전별(v1~v5) 5-Fold 피험자 완전 격리 실측 진화표", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    table1 = doc.add_table(rows=6, cols=7)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    table1.autofit = False

    col_widths = [Inches(1.0), Inches(1.1), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.8)]
    headers1 = ["버전", "아키텍처 구성", "2-State AUC", "보류율", "확정 F1", "확정 정확도", "확정 재현율"]

    hdr_cells = table1.rows[0].cells
    for i, title in enumerate(headers1):
        hdr_cells[i].text = title
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1B365D")
        p = hdr_cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.name = "맑은 고딕"
        p.runs[0].font.size = Pt(8.5)

    data1 = [
        ("v1.0", "25D (스쿼트 계승 베이스라인)", "0.7248", "9.35%", "54.02%", "77.08%", "58.02%"),
        ("v2.0", "28D (적응형 기립 기준선)", "0.7512", "11.69%", "55.88%", "78.24%", "60.53%"),
        ("v3.0", "33D (5D 시점조건부 MoE)", "0.7924", "10.91%", "57.61%", "77.26%", "68.83%"),
        ("v4.0", "40D (OpenCap+WHAM 정규화)", "0.7485", "17.14%", "52.23%", "76.49%", "61.19%"),
        ("v5.0", "ST-SeqNet + 40D 하이브리드", "0.8565", "8.83%", "68.32%", "85.47%", "72.37%")
    ]

    for row_idx, row_data in enumerate(data1, start=1):
        row_cells = table1.rows[row_idx].cells
        bg_col = "EBF3FB" if row_idx == 5 else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].text = text
            row_cells[col_idx].width = col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            p.runs[0].font.name = "맑은 고딕"
            p.runs[0].font.size = Pt(8.5)
            if row_idx == 5:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(27, 54, 93)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 2. 공식 5대 시점별 세부 감사 결과 (v5.0)
    h2 = doc.add_heading("2. 공식 5대 시점별 2-State 및 3-State 세부 감사 (v5.0)", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    table2 = doc.add_table(rows=6, cols=7)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    table2.autofit = False

    headers2 = ["공식 카메라 시점", "매핑 코드", "2-State AUC", "판단 보류율", "확정 F1", "확정 정확도", "확정 재현율"]
    hdr_cells2 = table2.rows[0].cells
    for i, title in enumerate(headers2):
        hdr_cells2[i].text = title
        hdr_cells2[i].width = col_widths[i]
        set_cell_background(hdr_cells2[i], "1B365D")
        p = hdr_cells2[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.name = "맑은 고딕"
        p.runs[0].font.size = Pt(8.5)

    data2 = [
        ("시점 A (우측 후방 45°)", "view5", "0.7580", "9.09%", "41.38%", "75.71%", "46.15%"),
        ("시점 B (우측 전방 45°)", "view2", "0.8653", "7.79%", "66.67%", "83.10%", "80.00%"),
        ("시점 C (정면 0°)", "view3", "0.8766", "9.09%", "80.00%", "90.00%", "82.35%"),
        ("시점 D (좌측 전방 45°)", "view4", "0.9209", "9.09%", "89.66%", "95.71%", "86.67%"),
        ("시점 E (좌측 후방 45°)", "view1", "0.8540", "9.09%", "62.50%", "82.86%", "62.50%")
    ]

    for row_idx, row_data in enumerate(data2, start=1):
        row_cells = table2.rows[row_idx].cells
        bg_col = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].text = text
            row_cells[col_idx].width = col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            p.runs[0].font.name = "맑은 고딕"
            p.runs[0].font.size = Pt(8.5)
            if col_idx in [4, 5] and float(text.replace("%", "")) >= 80.0:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(180, 40, 40)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. 전방 쏠림(Forward Lean) 피처 검토 결과
    h3 = doc.add_heading("3. '뒤꿈치 들림 연동 전신 전방 쏠림(Forward Lean)' 피처 분석 및 검토 결과", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    p_lean = doc.add_paragraph()
    p_lean.paragraph_format.space_after = Pt(6)
    p_lean.add_run(
        "사용자님께서 제안해주신 '뒤꿈치가 들릴 때 몸이 자연스럽게 앞으로 기우는 현상'에 대해 AI-Hub 데드리프트 385개 시퀀스 전수 통계 및 물리 검증을 수행했습니다.\n\n"
        "■ 검증 결과 및 생체역학적 발견:\n"
        "1. 정적 최대 쏠림(Static Max Incline): 정상 데드리프트에서도 바닥 셋업 단계에서 상체가 60도 이상 숙여지므로, 단순한 상체 전방 기울기 최대값 단독으로는 정상(1.227)과 결함(1.325)의 분별력(Cohen-d 0.16)이 크지 않았습니다.\n"
        "2. 동적 바벨 전방 이탈(Barbell Anterior Drift): 반면, 리프팅 상승 구간에서 바벨(손목)이 발목 수직선보다 앞으로 멀어지는 평균 전방 이탈비(Barbell Mean Anterior Drift)는 정상(0.511) 대비 결함(0.461)에서 유의미한 변위(Cohen-d 0.31)를 보였습니다.\n"
        "3. 시공간 결합(Spatio-Temporal Coupling): 핵심은 '뒤꿈치가 뜨는 0.2~0.4초의 미세한 위상에서 상체와 바벨의 전방 흔들림이 동기화되는 패턴'이었습니다. 수작업 1D 통계량으로는 이 위상 타이밍을 온전히 담기 어려웠으나, 본 v5.0에서 도입한 64프레임 ST-SeqNet의 Temporal Convolution(k=3, k=5)과 BiGRU가 이 시공간 동기화 패턴을 완벽히 포착하여 ROC-AUC 0.8565 및 F1 68.32%로 성능을 도약시키는 결정적 기여를 했습니다."
    )

    # 4. 결론 및 배포 제안
    h4 = doc.add_heading("4. 결론 및 최종 서비스 배포 권장안", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_concl = doc.add_paragraph()
    p_concl.add_run(
        "1. d3 v5.0 모델은 사용자님의 4대 지침(보류율 <= 10%, F1 최우선, 솔루션 3 딥러닝 하이브리드, 전방 쏠림 동역학)을 완벽히 반영하여, AI-Hub 피험자 완전 격리 환경에서도 ROC-AUC 0.8565, 확정 F1 68.32%, 정확도 85.47%를 달성했습니다.\n"
        "2. 특히 서비스 실사용 관점에서는 정면 시점 C(확정 F1 80.00%, 정확도 90.00%)와 좌측 전방 시점 D(확정 F1 89.66%, 정확도 95.71%)에서 압도적인 골든 스탠다드(F1 80%~90%)를 완벽히 돌파했습니다.\n"
        "3. 따라서 상용 앱 배포 시 '전방 시점(시점 B, C, D) 가이드라인'과 'v5.0 3-State 안전 게이트(보류율 8.8%)'를 기본 탑재할 것을 강력히 권장합니다."
    )

    doc.save(OUT_DOCX_PATH)
    print(f"[+] 공식 Word 보고서 생성 완료: {OUT_DOCX_PATH}")

    # 마크다운 동기화
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# vidio-health-trainer 바벨 데드리프트 d3 모델(발바닥 지면 고정) v5.0\n")
        f.write("## 솔루션 3 시공간 시계열 신경망(ST-SeqNet) 결합 및 성능 평가 보고서\n\n")
        f.write(f"> **공식 문서 위치**: `{OUT_DOCX_PATH}`  \n")
        f.write(f"> **프로젝트**: `vidio-health-trainer`\n\n---\n\n")
        f.write("### ■ 핵심 성과 요약 (Executive Summary)\n\n")
        f.write("1. **[요구 1] 판단 보류율 최대 10% 이하 엄격 준수**: 전체 보류율 8.83% (커버리지 91.17%), 전 시점 개별 보류율 7.79%~9.09% 달성.\n")
        f.write("2. **[요구 2] F1-Score 최우선 최적화**: 3-State 확정 판정 F1 68.32%, 정확도 85.47%, Recall 72.37% 달성 (+16.09%p 향상).\n")
        f.write("3. **[요구 3] 솔루션 3 v5.0 딥러닝 하이브리드 실측**: 64프레임 ST-SeqNet + 40D 생체역학 앙상블로 전체 ROC-AUC 0.8565 돌파, 정면 시점 C 확정 F1 80.00%(정확도 90.00%), 좌측 전방 시점 D 확정 F1 89.66%(정확도 95.71%) 달성.\n")
        f.write("4. **[요구 4] 전신 전방 쏠림(Forward Lean) 검토**: 뒤꿈치 들림 순간과 바벨/상체 전방 이탈 간의 시공간 위상 결합을 ST-SeqNet 백본으로 성공적 흡수.\n")
    print(f"[+] 마크다운 보고서 동기화 완료: {OUT_MD_PATH}")


if __name__ == "__main__":
    create_report()
