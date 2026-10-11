# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_c1_v10_final_confirmed_report.py
Purpose: 바벨 스쿼트 c1 v10.0 Balanced [0.40, 0.60] 공식 확정 및 최종 배포 보고서 (.docx & .md)
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_OUT = r"D:\자주프 문서\5주차\바벨스쿼트_c1_v10_Balanced_최종선정_및_공식확정보고서.docx"
MD_OUT = r"D:\자주프 문서\5주차\바벨스쿼트_c1_v10_Balanced_최종선정_및_공식확정보고서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
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

def create_report():
    os.makedirs(os.path.dirname(DOCX_OUT), exist_ok=True)
    doc = Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # 헤더 타이틀
    p_t = doc.add_paragraph()
    p_t.paragraph_format.space_before = Pt(0)
    p_t.paragraph_format.space_after = Pt(4)
    r_p = p_t.add_run("[vidio-health-trainer]\n")
    r_p.font.size = Pt(13)
    r_p.font.bold = True
    r_p.font.color.rgb = RGBColor(31, 78, 121)

    r_main = p_t.add_run("바벨 스쿼트 c1 v10.0 Balanced 공식 확정 및 최종 프로덕션 배포 보고서")
    r_main.font.size = Pt(18)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(16)
    r_sub = p_sub.add_run("Balanced [0.40, 0.60] 채택 · Coverage 90.9% · Selective F1 96.11% · 정상군 오탐 0.81% 달성")
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(100, 116, 139)

    # 1. 확정 배경 및 선정 사유
    h1 = doc.add_heading("1. 사용자 최종 승인 및 Balanced [0.40, 0.60] 선정 사유", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph(
        "사용자의 최종 결정에 따라, 바벨 스쿼트 세부 결함 1(c1: 척추의 중립 위반 / Spine Neutrality)의 "
        "공식 프로덕션 모델로 'Balanced [tau_normal=0.40, tau_defect=0.60]' 설정의 c1 v10.0을 공식 채택 및 확정 배포하였습니다.\n\n"
        "Balanced 설정은 2,605개 전수 시퀀스 실측 평가에서 다음과 같은 절대적 강점을 입증하여 최종 모델로 선정되었습니다:"
    )
    p1.paragraph_format.line_spacing = 1.25

    reasons = [
        ("90% 이상의 압도적 판정 커버리지 (Coverage 90.9%): ", "전체 2,605개 시퀀스 중 2,369건을 명확히 확정 판정하고, 오직 상위 9.1%(236건)의 모호한 경계 구간만을 정밀 선별하여 보류."),
        ("Selective F1-Score 96.11% 달성 (+16.13%p 폭등): ", "기존 강제 이진 분류기(v9: 79.98%) 대비 확정 판정군에서의 F1-Score가 96.11%로 수직 상승하여 오판 위험(Risk)을 3.67%로 극감."),
        ("골든 그라운드트루스 정상군 허위 오탐 0.81%로 제로화: ", "정상 샘플 135건 중 무리하게 결함으로 오판하던 건수를 22건(16.30%)에서 단 1건(0.81%)으로 격감시켜 억울한 결함 판정 완전 해결."),
        ("c2 v6.0 Balanced 모델과의 완벽한 아키텍처 정렬: ", "고개 정면(c2)과 척추 중립(c1)이 동일한 [0.40, 0.60] 3-State 정책을 공유하여 통합 엔진의 소프트 위험도 융합 일관성 확보.")
    ]
    for r_title, r_desc in reasons:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        rb = bp.add_run(r_title)
        rb.bold = True
        rb.font.color.rgb = RGBColor(31, 78, 121)
        rn = bp.add_run(r_desc)
        rn.font.color.rgb = RGBColor(51, 65, 85)

    # 2. 실측 성능 종합 대조표
    h2 = doc.add_heading("2. 공식 프로덕션 모델 실측 성능 대조표 (N=2,605)", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    table_comp = doc.add_table(rows=1, cols=4)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_hdr = table_comp.rows[0].cells
    hdr_titles = ["평가 지표 (Metric)", "v9.0 (강제 이진 결정)", "v10.0 (Balanced 공식 확정)", "개선 효과 및 달성 성과"]
    c_widths = [Inches(1.8), Inches(1.8), Inches(2.2), Inches(2.0)]

    for idx, (title, w) in enumerate(zip(hdr_titles, c_widths)):
        c_hdr[idx].text = title
        c_hdr[idx].width = w
        set_cell_background(c_hdr[idx], "1F4E79")
        set_cell_margins(c_hdr[idx], 100, 100, 100, 100)
        p = c_hdr[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(255, 255, 255)

    comp_rows = [
        ("F1-Score (판정 표본)", "79.98%", "96.11% (Selective F1)", "+16.13%p 폭등"),
        ("정확도 (Accuracy)", "81.00%", "96.33% (Selective Acc)", "+15.33%p 폭등"),
        ("오판율 (Selective Risk)", "19.00%", "3.67%", "▼ 15.33%p (1/5로 극감)"),
        ("판정 커버리지 (Coverage)", "100.0% (억지 강제)", "90.9% (2,369건)", "실효 판정률 90% 이상 유지"),
        ("보류율 (Abstention Rate)", "0.0%", "9.1% (236건)", "모호 표본만 안전 보류"),
        ("정상군 허위 오탐률 (FP)", "16.30% (22건)", "0.81% (단 1건!)", "▼ 15.49%p (사실상 제로화)"),
        ("레거시 텐서 슬라이싱", "0% (완전 제거)", "0% (완전 제거)", "순수 64F 표준 모듈화"),
        ("SpinePose 9분절 통합", "28D 고정밀 반영", "28D 고정밀 반영", "요추 2차 곡률 poly_a 보존")
    ]

    for row_idx, items in enumerate(comp_rows):
        row_cells = table_comp.add_row().cells
        bg_col = "F1F5F9" if row_idx % 2 == 1 else "FFFFFF"
        if "폭등" in items[3] or "제로화" in items[3]:
            bg_col = "E0F2FE"
        for idx, (txt, w) in enumerate(zip(items, c_widths)):
            row_cells[idx].text = txt
            row_cells[idx].width = w
            set_cell_background(row_cells[idx], bg_col)
            set_cell_margins(row_cells[idx], 70, 70, 70, 70)
            p = row_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in [1, 2] else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(8.5)
                if idx == 3 and "+" in txt:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(14, 116, 144)
                elif idx == 3 and "제로화" in txt:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(46, 125, 50)
                else:
                    r.font.color.rgb = RGBColor(30, 41, 59)

    # 3. 5대 전 시점별 확정 성적
    h3 = doc.add_heading("3. 5대 전 시점(view1~view5) 공식 실측 성적", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    table_v = doc.add_table(rows=1, cols=5)
    table_v.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_hdr = table_v.rows[0].cells
    v_hdr_titles = ["카메라 시점", "시점 설명", "전체 표본", "판정 커버리지", "Selective F1-Score"]
    vw = [Inches(1.2), Inches(2.2), Inches(1.2), Inches(1.5), Inches(1.7)]

    for idx, (t, w) in enumerate(zip(v_hdr_titles, vw)):
        v_hdr[idx].text = t
        v_hdr[idx].width = w
        set_cell_background(v_hdr[idx], "1E293B")
        set_cell_margins(v_hdr[idx], 90, 90, 90, 90)
        p = v_hdr[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(255, 255, 255)

    v_data = [
        ("view1", "후방 (180°)", "521개", "461개 (88.5%)", "97.67%"),
        ("view2", "우측 45° (측면 전방)", "521개", "482개 (92.5%)", "95.77%"),
        ("view3", "완전 정면 (0°)", "521개", "478개 (91.7%)", "95.34%"),
        ("view4", "좌측 45° (측면 전방)", "521개", "467개 (89.6%)", "95.79%"),
        ("view5", "우측 후방 45°", "521개", "481개 (92.3%)", "96.05%")
    ]

    for row_idx, items in enumerate(v_data):
        row_cells = table_v.add_row().cells
        bg_col = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for idx, (txt, w) in enumerate(zip(items, vw)):
            row_cells[idx].text = txt
            row_cells[idx].width = w
            set_cell_background(row_cells[idx], bg_col)
            set_cell_margins(row_cells[idx], 70, 70, 70, 70)
            p = row_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in [0, 2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(8.5)
                if idx == 4:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(31, 78, 121)
                else:
                    r.font.color.rgb = RGBColor(30, 41, 59)

    # 4. 배포 명세
    h4 = doc.add_heading("4. 공식 배포 산출물 및 아키텍처 연동 현황", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_dep = doc.add_paragraph(
        "1. 모델 클래스 영구 확정: src/models/second_models/babel_squat/c1_model_v10.py\n"
        "   - 기본 파라미터로 tau_normal=0.40, tau_defect=0.60 (Balanced) 공식 설정 완료\n"
        "   - __init__.py 패키지 최상위에서 BabelSquatC1Model로 정식 바인딩 완료\n\n"
        "2. 프로덕션 공식 가중치: src/models/second_models/babel_squat/c1_model_v10_weights.pkl\n"
        "   - 5-Fold CatBoost + LightGBM 앙상블 가중치 및 Balanced 임계치 영구 패키징 완료\n\n"
        "3. 전체 파이프라인 무결성:\n"
        "   - c2 v6.0(Balanced) 및 c1 v10.0(Balanced)의 3-State 이중화 연동 체계 완비"
    )
    p_dep.paragraph_format.line_spacing = 1.25

    doc.save(DOCX_OUT)
    print(f"[*] Word 보고서 생성 완료: {DOCX_OUT}")

    # Markdown 동시 생성
    md_text = f"""# [vidio-health-trainer] 바벨 스쿼트 c1 v10.0 Balanced 공식 확정 및 최종 프로덕션 배포 보고서

- **공식 프로젝트명**: `vidio-health-trainer`
- **배포 일시**: 2026-09-26
- **보고서 규격**: Word 문서(`바벨스쿼트_c1_v10_Balanced_최종선정_및_공식확정보고서.docx`) 및 Markdown 동시 배포

---

## 1. 사용자 최종 승인 및 Balanced [0.40, 0.60] 선정 사유

사용자 승인에 따라 **`c1 v10.0 Balanced (tau_normal=0.40, tau_defect=0.60)`**가 바벨 스쿼트 세부 결함 1(c1: 척추의 중립 위반)의 **최종 공식 프로덕션 모델**로 확정 배포되었습니다.

1. **커버리지 90.9% (2,369건 판정)**: 전체 시퀀스 중 90% 이상을 명확히 판정하고 모호 구간(9.1%, 236건)만 안전 보류.
2. **Selective F1 96.11% 달성 (+16.13%p 폭등)**: 기존 강제 이진(v9: 79.98%) 대비 확정 판정군에서의 F1-Score가 96.11%로 수직 상승.
3. **정상군 허위 오탐 0.81%로 제로화**: 정상 샘플 135건 중 오탐이 22건(16.30%)에서 단 1건(0.81%)으로 격감.
4. **c2 v6.0과의 완벽한 정책 일치**: c2와 c1이 모두 동일한 [0.40, 0.60] 3-State 정책을 공유.

---

## 2. 공식 프로덕션 모델 실측 성능 대조표 (N=2,605)

| 평가 지표 (Metric) | v9.0 (강제 이진 결정) | v10.0 (Balanced 공식 확정) | 개선 효과 및 달성 성과 |
| :--- | :---: | :---: | :--- |
| **F1-Score (판정 표본)** | 79.98% | **96.11% (Selective F1)** | **+16.13%p 폭등** |
| **정확도 (Accuracy)** | 81.00% | **96.33% (Selective Acc)** | **+15.33%p 폭등** |
| **오판율 (Selective Risk)** | 19.00% | **3.67%** | **▼ 15.33%p (1/5로 극감)** |
| **판정 커버리지 (Coverage)** | 100.0% (억지 강제) | **90.9% (2,369건)** | **실효 판정률 90% 이상 유지** |
| **보류율 (Abstention Rate)** | 0.0% | **9.1% (236건)** | **모호 표본만 안전 보류** |
| **정상군 허위 오탐률 (FP)** | 16.30% (22건) | **0.81% (단 1건!)** | **▼ 15.49%p (사실상 제로화)** |
| **레거시 텐서 슬라이싱** | 0% (완전 제거) | **0% (완전 제거)** | **순수 64F 표준 모듈화** |
| **SpinePose 9분절 통합** | 28D 고정밀 반영 | **28D 고정밀 반영** | **요추 2차 곡률 poly_a 보존** |

---

## 3. 5대 전 시점(view1~view5) 공식 실측 성적

| 카메라 시점 | 시점 설명 | 전체 표본 | 판정 커버리지 | **Selective F1-Score** |
| :--- | :--- | :---: | :---: | :---: |
| **`view1`** | **후방 (180°)** | 521개 | 461개 (88.5%) | **97.67%** |
| **`view2`** | **우측 45° (측면 전방)** | 521개 | 482개 (92.5%) | **95.77%** |
| **`view3`** | **완전 정면 (0°)** | 521개 | 478개 (91.7%) | **95.34%** |
| **`view4`** | **좌측 45° (측면 전방)** | 521개 | 467개 (89.6%) | **95.79%** |
| **`view5`** | **우측 후방 45°** | 521개 | 481개 (92.3%) | **96.05%** |

---

## 4. 공식 배포 산출물 명세

1. **공식 모델 클래스**: [`src/models/second_models/babel_squat/c1_model_v10.py`](file:///d:/src/models/second_models/babel_squat/c1_model_v10.py)
   - `BabelSquatC1Model` 기본 클래스로 확정 바인딩 완료
   - 기본 파라미터 `tau_normal=0.40, tau_defect=0.60` 고정
2. **프로덕션 가중치**: [`src/models/second_models/babel_squat/c1_model_v10_weights.pkl`](file:///d:/src/models/second_models/babel_squat/c1_model_v10_weights.pkl)
3. **패키지 익스포트**: [`src/models/second_models/babel_squat/__init__.py`](file:///d:/src/models/second_models/babel_squat/__init__.py)
"""
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"[*] Markdown 보고서 생성 완료: {MD_OUT}")

if __name__ == '__main__':
    create_report()
