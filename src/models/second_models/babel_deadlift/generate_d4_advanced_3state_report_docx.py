# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d4_advanced_3state_report_docx.py
Purpose: 바벨 데드리프트 d4 v5.0 고도화 3-State (F1 60.94% 돌파) 공식 기술 보고서 (.docx) 생성
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d4_v5.0_고도화3State_최적화_보고서.docx"
MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d4_v5.0_고도화3State_최적화_보고서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("바벨 데드리프트 d4 v5.0 고도화 3-State 신뢰도 게이팅 최적화 보고서")
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(24, 43, 73)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("vidio-health-trainer | 기준선 재정렬(tau_defect=0.70)을 통한 F1-Score 60.94% 공식 돌파 및 5대 시점 전수 감사")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(10)
    run_sub.font.color.rgb = RGBColor(100, 110, 120)

    # 1. 2-State vs 기존 3-State vs 고도화 3-State 비교표
    h1 = doc.add_heading("1. 의사결정 전략별 성능 비교 (2-State vs 기존 3-State vs 고도화 3-State)", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    headers = ["평가 지표", "기본 2-State (0.50)", "최적 2-State (0.70)", "기존 3-State (0.50 기반)", "★ 고도화 3-State (0.70 재정렬)", "개선 효과"]
    rows = [
        ["F1-Score", "51.58%", "57.03%", "53.04%", "60.94%", "+9.36%p (60% 벽 최초 돌파!)"],
        ["정밀도 (Precision)", "42.06%", "62.24%", "44.39%", "62.28%", "+20.22%p (오탐 억제 극대화)"],
        ["재현율 (Recall)", "66.67%", "52.59%", "65.87%", "59.66%", "결함 71건 안정적 적발 유지"],
        ["정확도 (Accuracy)", "78.05%", "86.10%", "78.50%", "86.87%", "+8.82%p 안정적 향상"],
        ["Coverage (판정률)", "100.0%", "100.0%", "92.08%", "90.00%", "보류율 10.0% 엄격 통제"],
        ["임계치 구간", "tau=0.50", "tau=0.70", "0.20 ~ 0.62", "0.535 ~ 0.700", "불확실 회색지대 77건 배제"]
    ]

    t1 = doc.add_table(rows=len(rows)+1, cols=len(headers))
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, h in enumerate(headers):
        cell = t1.cell(0, j)
        set_cell_background(cell, "1F497D")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    for i, r in enumerate(rows):
        bg = "F2F5F9" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(r):
            cell = t1.cell(i+1, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in [0, 5] else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.5)
            if j == 4:
                run.font.bold = True
                run.font.color.rgb = RGBColor(180, 40, 40)

    # 2. 공식 5대 시점 전수 감사
    h2 = doc.add_heading("2. 공식 5대 시점 전수 감사 (Official 5-View Audit - 고도화 3-State)", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    v_headers = ["공식 시점 명칭", "내부 코드", "2-State F1", "고도화 3-State F1", "ROC-AUC", "Coverage", "시점별 판정 강건성"]
    v_rows = [
        ["시점 D (좌측 전방 45°)", "view4", "70.59%", "76.60%", "0.9108", "91.6%", "★ F1 76.6%, AUC 0.91 초고정밀 판정"],
        ["시점 B (우측 전방 45°)", "view2", "70.37%", "74.51%", "0.8583", "91.6%", "★ F1 74.5%, AUC 0.86 대각선 전방 완벽 커버"],
        ["시점 E (좌측 후방 45°)", "view1", "61.54%", "65.31%", "0.8186", "90.3%", "후방 뷰에서도 F1 65.3% 고성능 달성"],
        ["시점 C (정면 0°)", "view3", "44.00%", "46.81%", "0.6996", "87.0%", "정면 깊이 왜곡 보류 처리로 46.8% 수복"],
        ["시점 A (우측 후방 45°)", "view5", "33.33%", "35.90%", "0.7224", "89.6%", "우측 후방 시계열 골반 상승 변별"]
    ]

    t2 = doc.add_table(rows=len(v_rows)+1, cols=len(v_headers))
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, h in enumerate(v_headers):
        cell = t2.cell(0, j)
        set_cell_background(cell, "2E5B82")
        set_cell_margins(cell, 120, 120, 140, 140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    for i, r in enumerate(v_rows):
        bg = "F9FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(r):
            cell = t2.cell(i+1, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 100, 100, 120, 120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in [0, 6] else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.5)
            if i in [0, 1] and j == 3:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 120, 50)

    # 3. 고도화 결론
    h3 = doc.add_heading("3. 기술 고도화 결론 및 배포 명세", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(4)

    p_body = doc.add_paragraph()
    p_body.paragraph_format.line_spacing = 1.25
    r_body = p_body.add_run(
        "1. 사용자 제안 기준선 재정렬 완벽 검증:\n"
        "   - 기존 3-State의 기준선(0.50) 대신, 모델의 최적 동작점인 tau_defect=0.70에 맞춰 기준선을 재정렬(tau_normal=0.535, tau_defect=0.700)함으로써 불확실한 77건(10%)을 정확히 보류(RETAKE) 처리했습니다.\n"
        "   - 그 결과, 2-State 최고 성능(57.03%)보다 무려 +3.91%p 추가 상승한 F1-Score 60.94%를 공식 달성했습니다.\n\n"
        "2. 실서비스 적용 매핑:\n"
        "   - p_final <= 0.535: NORMAL (정상 데드리프트, 무릎-골반 동시 신전 준수)\n"
        "   - p_final >= 0.700: DEFECT (결함 감지, d4 무릎-골반 동시 펴짐 위반 / 힙 슈트업)\n"
        "   - 0.535 < p_final < 0.700: RETAKE (불확실 동작, 사용자에게 카메라 각도 조정 후 재촬영 안내)"
    )
    r_body.font.name = "맑은 고딕"
    r_body.font.size = Pt(9.5)

    doc.save(DOCX_PATH)
    print(f"[+] Word 보고서 저장 완료: {DOCX_PATH}")

    # 마크다운 요약본 저장
    md_content = f"""# 바벨 데드리프트 d4 v5.0 고도화 3-State 신뢰도 게이팅 최적화 보고서

> **프로젝트**: vidio-health-trainer | AI-Hub 바벨 데드리프트 결함 d4 모델  
> **보고서 파일**: `{DOCX_PATH}`  
> **작성 일자**: 2026-09-30  
> **최종 성과**: F1-Score **60.94%** 공식 돌파 (Coverage 90.0% 엄수)  

---

## 1. 의사결정 전략별 성능 비교

| 평가 지표 | 기본 2-State (0.50) | 최적 2-State (0.70) | 기존 3-State (0.50 기반) | **★ 고도화 3-State (0.70 재정렬)** | 개선 효과 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **F1-Score** | 51.58% | 57.03% | 53.04% | <span style="color:red">**60.94%**</span> | **+9.36%p (60% 벽 최초 돌파!)** |
| **정밀도 (Precision)**| 42.06% | 62.24% | 44.39% | <span style="color:red">**62.28%**</span> | **+20.22%p (오탐 억제 극대화)** |
| **재현율 (Recall)** | 66.67% | 52.59% | 65.87% | **59.66%** | 결함 71건 안정적 적발 유지 |
| **정확도 (Accuracy)** | 78.05% | 86.10% | 78.50% | **86.87%** | **+8.82%p 안정적 향상** |
| **Coverage (판정률)** | 100.0% | 100.0% | 92.08% | **90.00%** | 보류율 10.0% 엄격 통제 |
| **임계치 구간** | $\tau=0.50$ | $\tau=0.70$ | $0.20 \sim 0.62$ | **$0.535 \sim 0.700$** | 불확실 회색지대 77건 배제 |

---

## 2. 공식 5대 시점 전수 감사 (Official 5-View Audit - 고도화 3-State)

| 공식 시점 명칭 | 내부 코드 | 2-State F1 (0.70) | **고도화 3-State F1** | **ROC-AUC** | Coverage | 시점별 판정 강건성 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **시점 D (좌측 전방 45°)** | view4 | 70.59% | <span style="color:red">**76.60%**</span> | **0.9108** | 91.6% | ★ F1 76.6%, AUC 0.91 초고정밀 판정 |
| **시점 B (우측 전방 45°)** | view2 | 70.37% | <span style="color:red">**74.51%**</span> | **0.8583** | 91.6% | ★ F1 74.5%, AUC 0.86 대각선 전방 완벽 커버 |
| **시점 E (좌측 후방 45°)** | view1 | 61.54% | **65.31%** | **0.8186** | 90.3% | 후방 뷰에서도 F1 65.3% 고성능 달성 |
| **시점 C (정면 0°)** | view3 | 44.00% | **46.81%** | 0.6996 | 87.0% | 정면 깊이 왜곡 보류 처리로 46.8% 수복 |
| **시점 A (우측 후방 45°)** | view5 | 33.33% | **35.90%** | 0.7224 | 89.6% | 우측 후방 시계열 골반 상승 변별 |
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] 마크다운 요약본 저장 완료: {MD_PATH}")

if __name__ == "__main__":
    create_report()
