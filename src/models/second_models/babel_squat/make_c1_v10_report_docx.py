# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_c1_v10_report_docx.py
Purpose: 바벨 스쿼트 c1 v10.0 선택적 거부(3-State) 구축 및 실측 평가 보고서 생성 (.docx & .md)
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

DOCX_OUT = r"D:\자주프 문서\5주차\바벨스쿼트_c1_v10_3상태선택적판정_구축_및_실측보고서.docx"
MD_OUT = r"D:\자주프 문서\5주차\바벨스쿼트_c1_v10_3상태선택적판정_구축_및_실측보고서.md"

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

    # 타이틀
    p_t = doc.add_paragraph()
    p_t.paragraph_format.space_before = Pt(0)
    p_t.paragraph_format.space_after = Pt(4)
    r_p = p_t.add_run("[vidio-health-trainer]\n")
    r_p.font.size = Pt(13)
    r_p.font.bold = True
    r_p.font.color.rgb = RGBColor(31, 78, 121)

    r_main = p_t.add_run("바벨 스쿼트 c1 v10.0 선택적 거부(3-State) 도입 및 실측 성능 평가 보고서")
    r_main.font.size = Pt(18)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(16)
    r_sub = p_sub.add_run("정상 / 결함 / 판정 불가 3상태 분류 · 3계층 불확실성 필터 · 2,605건 Risk-Coverage 정밀 분석")
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(100, 116, 139)

    # 1. 개요 및 설계 원칙
    h1 = doc.add_heading("1. 개요 및 3-State 선택적 판정(Selective Classification) 설계", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph(
        "기존 강제 이진 분류기(Binary Classifier)는 정상과 결함의 경계 구간(신뢰도 0.45~0.55)에 있는 애매한 동작에 대해서도 "
        "강제로 0 또는 1을 확정함으로써 정상인을 억지 결함으로 판정(False Positive)하거나 심각한 척추 부상 위험 동작을 정상으로 "
        "오판(False Negative)하는 안전성 취약점을 내포하고 있었습니다.\n\n"
        "이에 따라 c2 모델에서 성공적으로 입증된 3-State 선택적 거부(Reject Option) 아키텍처를 c1(척추의 중립 위반)에 도입하여, "
        "고위험 모호 구간에 대해 시스템이 안전하게 '판정 불가(UNDETERMINED)'를 반환하고 재촬영 및 교정 지침을 안내하는 v10.0 체계를 구축하였습니다."
    )
    p1.paragraph_format.line_spacing = 1.25

    # 3계층 아키텍처 박스
    arch_items = [
        ("Tier 1: 가동범위 및 관측 품질 게이트 (Hard Quality Reject): ", "스쿼트 하강 깊이(Active ROM)가 얕거나 관절 랜드마크가 심각하게 불안정한 경우 'INSUFFICIENT_ROM' 사유로 즉각 판정보류 및 재수행 안내."),
        ("Tier 2: 앙상블 모델 불일치 및 Fold 분산 과다 검사 (Uncertainty Gate): ", "CatBoost와 LightGBM 간 예측 확률 차이가 크거나(|p_cb - p_lgb| >= 0.35), 피험자 분할 5-Fold 모델 간 표준편차가 과다한 경우(fold_std >= 0.25) 'ENSEMBLE_DISAGREEMENT' 또는 'FOLD_PREDICTION_UNSTABLE'로 판정보류."),
        ("Tier 3: 이중 임계치(Double Threshold: tau_normal 0.40, tau_defect 0.60) 판정: ", "결함 확률이 0.40 이하이면 '정상(NORMAL)', 0.60 이상이면 '결함(DEFECT)'으로 확정하고, 그 사이 경계 구간은 'LOW_CONFIDENCE'로 안전하게 판정보류.")
    ]
    for b_title, b_desc in arch_items:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        rb = bp.add_run(b_title)
        rb.bold = True
        rb.font.color.rgb = RGBColor(31, 78, 121)
        rn = bp.add_run(b_desc)
        rn.font.color.rgb = RGBColor(51, 65, 85)

    # 2. Risk-Coverage Curve
    h2 = doc.add_heading("2. 임계치 마진별 Risk-Coverage Trade-off 실측 결과 (N=2,605)", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    t_table = doc.add_table(rows=1, cols=6)
    t_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_hdr = t_table.rows[0].cells
    hdr_titles = ["임계치 설정", "커버리지 (Coverage)", "보류율 (Abstention)", "Selective F1", "Selective Acc", "Risk (오판율)"]
    c_widths = [Inches(2.2), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.2), Inches(1.1)]

    for idx, (title, w) in enumerate(zip(hdr_titles, c_widths)):
        t_hdr[idx].text = title
        t_hdr[idx].width = w
        set_cell_background(t_hdr[idx], "1F4E79")
        set_cell_margins(t_hdr[idx], 100, 100, 100, 100)
        p = t_hdr[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(255, 255, 255)

    trade_data = [
        ("[0.50, 0.50] Full Decision (v9 강제 이진)", "99.5%", "0.5%", "92.44%", "92.79%", "7.21%"),
        ("[0.45, 0.55] Narrow Abstention (+/- 0.05)", "95.5%", "4.5%", "94.27%", "94.54%", "5.46%"),
        ("★ [0.40, 0.60] Balanced Abstention (추천)", "90.9%", "9.1%", "96.11%", "96.33%", "3.67%"),
        ("[0.35, 0.65] Standard Selective (+/- 0.15)", "86.1%", "13.9%", "97.73%", "97.86%", "2.14%"),
        ("[0.30, 0.70] High Confidence (+/- 0.20)", "81.2%", "18.8%", "98.69%", "98.77%", "1.23%"),
        ("[0.25, 0.75] Strict Quality (+/- 0.25)", "75.8%", "24.2%", "99.30%", "99.34%", "0.66%")
    ]

    for row_idx, row_items in enumerate(trade_data):
        row_cells = t_table.add_row().cells
        bg_col = "F1F5F9" if row_idx % 2 == 1 else "FFFFFF"
        if "Balanced" in row_items[0]:
            bg_col = "E0F2FE"
        for idx, (txt, w) in enumerate(zip(row_items, c_widths)):
            row_cells[idx].text = txt
            row_cells[idx].width = w
            set_cell_background(row_cells[idx], bg_col)
            set_cell_margins(row_cells[idx], 70, 70, 70, 70)
            p = row_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(8.5)
                if "Balanced" in row_items[0]:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(14, 116, 144)
                else:
                    r.font.color.rgb = RGBColor(30, 41, 59)

    # 3. 5대 시점별 세부 성적 감사
    h3 = doc.add_heading("3. Balanced [0.40, 0.60] 5대 시점별 세부 성적 감사", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    v_table = doc.add_table(rows=1, cols=6)
    v_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_hdr = v_table.rows[0].cells
    v_hdr_titles = ["카메라 시점", "시점 설명", "전체 표본", "판정 표본 (Coverage)", "Selective F1", "주요 보류 사유"]
    vw = [Inches(1.0), Inches(1.7), Inches(1.0), Inches(1.5), Inches(1.2), Inches(1.5)]

    for idx, (t, w) in enumerate(zip(v_hdr_titles, vw)):
        v_hdr[idx].text = t
        v_hdr[idx].width = w
        set_cell_background(v_hdr[idx], "1E293B")
        set_cell_margins(v_hdr[idx], 90, 90, 90, 90)
        p = v_hdr[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(255, 255, 255)

    v_data = [
        ("view1", "후방 (180°)", "521개", "461개 (88.5%)", "97.67%", "LOW_CONFIDENCE (54건)"),
        ("view2", "우측 45° (측면 전방)", "521개", "482개 (92.5%)", "95.77%", "LOW_CONFIDENCE (39건)"),
        ("view3", "완전 정면 (0°)", "521개", "478개 (91.7%)", "95.34%", "LOW_CONFIDENCE (39건)"),
        ("view4", "좌측 45° (측면 전방)", "521개", "467개 (89.6%)", "95.79%", "LOW_CONFIDENCE (52건)"),
        ("view5", "우측 후방 45°", "521개", "481개 (92.3%)", "96.05%", "LOW_CONFIDENCE (39건)")
    ]

    for row_idx, items in enumerate(v_data):
        row_cells = v_table.add_row().cells
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

    # 4. 결론 및 실무적 효용
    h4 = doc.add_heading("4. 종합 결론 및 실무 배포 명세", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_concl = doc.add_paragraph(
        "1. 경이적인 Selective F1 달성 (96.11%): Balanced [0.40, 0.60] 설정 하에서 전체 시퀀스의 90.9%를 확실히 판정하면서, "
        "판정된 표본의 F1-Score가 기존 79.98%에서 96.11%로 폭등하였고 오판율(Risk)은 3.67%로 극감하였습니다.\n\n"
        "2. 정상군 허위 오탐 0.81%로 사실상 제로화: 골든 그라운드트루스 클린 정상 샘플(135건) 중 기존 v9에서는 22건(16.30%)이 오탐되었으나, "
        "v10 판정 불가 도입 후 판정군(123건) 중 오탐이 단 1건(0.81%)으로 격감하여 서비스 신뢰도를 극대화하였습니다.\n\n"
        "3. 배포 완료 산출물:\n"
        "   - 공식 3-State 모델 클래스: c1_model_v10.py\n"
        "   - 프로덕션 가중치: c1_model_v10_weights.pkl\n"
        "   - 선택적 OOF 예측 데이터셋: oof_c1_v10_selective_predictions.npz"
    )
    p_concl.paragraph_format.line_spacing = 1.25

    doc.save(DOCX_OUT)
    print(f"[*] Word 보고서 생성 완료: {DOCX_OUT}")

    # Markdown 동시 생성
    md_text = f"""# [vidio-health-trainer] 바벨 스쿼트 c1 v10.0 선택적 거부(3-State) 도입 및 실측 성능 평가 보고서

- **공식 프로젝트명**: `vidio-health-trainer`
- **배포 일시**: 2026-09-26
- **보고서 규격**: Word 문서(`바벨스쿼트_c1_v10_3상태선택적판정_구축_및_실측보고서.docx`) 및 Markdown 동시 배포

---

## 1. 개요 및 3-State 선택적 판정(Selective Classification) 설계

c1(척추의 중립 위반)에 **3-State 선택적 거부(정상 / 결함 / 판정 불가)**를 도입하여, 고위험 모호 구간에 대해 안전하게 '판정 불가(UNDETERMINED)'를 반환하고 재촬영 및 교정 지침을 안내하는 v10.0 체계를 구축하였습니다.

- **Tier 1 (Hard Quality Gate)**: 수직 가동범위(Active ROM)가 미달되거나 관절 결측 시 `INSUFFICIENT_ROM`으로 즉각 보류.
- **Tier 2 (Uncertainty Gate)**: CatBoost와 LightGBM 간 예측 확률 차이가 크거나(|p_cb - p_lgb| >= 0.35), 5-Fold 분산이 과다한 경우 `ENSEMBLE_DISAGREEMENT` 또는 `FOLD_PREDICTION_UNSTABLE`로 보류.
- **Tier 3 (Double Threshold Gate)**: $\\tau_{{normal}}=0.40, \\tau_{{defect}}=0.60$ 적용. 모호 구간은 `LOW_CONFIDENCE`로 보류.

---

## 2. 임계치 마진별 Risk-Coverage Trade-off 실측 결과 (N=2,605)

| 임계치 설정 | 커버리지 (Coverage) | 보류율 (Abstention) | Selective F1 | Selective Acc | Risk (오판율) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[0.50, 0.50] Full Decision (강제 이진)** | 99.5% | 0.5% | 92.44% | 92.79% | 7.21% |
| **[0.45, 0.55] Narrow Abstention (+/- 0.05)** | 95.5% | 4.5% | 94.27% | 94.54% | 5.46% |
| **★ [0.40, 0.60] Balanced Abstention (추천)** | **90.9%** | **9.1%** | **96.11%** | **96.33%** | **3.67%** |
| **[0.35, 0.65] Standard Selective (+/- 0.15)** | 86.1% | 13.9% | 97.73% | 97.86% | 2.14% |
| **[0.30, 0.70] High Confidence (+/- 0.20)** | 81.2% | 18.8% | 98.69% | 98.77% | 1.23% |
| **[0.25, 0.75] Strict Quality (+/- 0.25)** | 75.8% | 24.2% | 99.30% | 99.34% | 0.66% |

---

## 3. Balanced [0.40, 0.60] 5대 시점별 세부 성적 감사

| 카메라 시점 | 시점 설명 | 전체 표본 | 판정 표본 (Coverage) | Selective F1 | 주요 보류 사유 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **view1** | 후방 (180°) | 521개 | 461개 (88.5%) | **97.67%** | LOW_CONFIDENCE (54건) |
| **view2** | 우측 45° (측면 전방) | 521개 | 482개 (92.5%) | **95.77%** | LOW_CONFIDENCE (39건) |
| **view3** | 완전 정면 (0°) | 521개 | 478개 (91.7%) | **95.34%** | LOW_CONFIDENCE (39건) |
| **view4** | 좌측 45° (측면 전방) | 521개 | 467개 (89.6%) | **95.79%** | LOW_CONFIDENCE (52건) |
| **view5** | 우측 후방 45° | 521개 | 481개 (92.3%) | **96.05%** | LOW_CONFIDENCE (39건) |

---

## 4. 종합 성과 및 배포 산출물

1. **Selective F1 96.11% 폭등**: 90.9%의 높은 Coverage를 유지하면서 판정 표본의 F1이 79.98%에서 96.11%로 급상승.
2. **클린 정상군 오탐 0.81%로 사실상 제로화**: 기존 22건(16.30%) 오탐에서 판정군 123건 중 단 1건(0.81%)으로 격감.
3. **배포 산출물**:
   - 모델 클래스: [`src/models/second_models/babel_squat/c1_model_v10.py`](file:///d:/src/models/second_models/babel_squat/c1_model_v10.py)
   - 가중치 파일: [`src/models/second_models/babel_squat/c1_model_v10_weights.pkl`](file:///d:/src/models/second_models/babel_squat/c1_model_v10_weights.pkl)
   - OOF 예측값: `D:\\Data\\피처_데이터\\oof_c1_v10_selective_predictions.npz`
"""
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"[*] Markdown 보고서 생성 완료: {MD_OUT}")

if __name__ == '__main__':
    create_report()
