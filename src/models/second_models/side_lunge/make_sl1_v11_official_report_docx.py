# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script to generate official Word (.docx) report for sl1_v11 Option 2
"""

import os
import json
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_report():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    PRIMARY_COLOR = RGBColor(26, 54, 93)     # Deep Navy
    SECONDARY_COLOR = RGBColor(49, 130, 206) # Slate Blue
    TEXT_COLOR = RGBColor(45, 55, 72)        # Dark Charcoal
    MUTED_COLOR = RGBColor(113, 128, 150)    # Gray
    ACCENT_GREEN = RGBColor(40, 167, 69)

    def style_heading(heading, font_size, color, space_before=12, space_after=6):
        heading.paragraph_format.space_before = Pt(space_before)
        heading.paragraph_format.space_after = Pt(space_after)
        heading.paragraph_format.keep_with_next = True
        for run in heading.runs:
            run.font.name = 'Malgun Gothic'
            run.font.size = Pt(font_size)
            run.font.bold = True
            run.font.color.rgb = color

    def add_p(text, bold=False, italic=False, color=TEXT_COLOR, font_size=10, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.2
        p.alignment = align
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        return p

    def set_cell_background(cell, fill_hex):
        shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
        cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def add_callout(text, title="핵심 요약"):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "F7FAFC")
        set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
        
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="3182CE"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
        tcPr.append(borders)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(3)
        r_title = p.add_run(f"📌 {title}\n")
        r_title.font.name = 'Malgun Gothic'
        r_title.font.size = Pt(10)
        r_title.font.bold = True
        r_title.font.color.rgb = SECONDARY_COLOR
        
        r_text = p.add_run(text)
        r_text.font.name = 'Malgun Gothic'
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = TEXT_COLOR

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("사이드 런지 c1 sl1_v11 공식 성능 평가 보고서")
    run_title.font.name = 'Malgun Gothic'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY_COLOR

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("추천 옵션 2 적용: CatBoost 강화 가중치 [0.15, 0.65, 0.20] 및 3-State [0.35, 0.65] 자연적 모호 구간 융합 모델")
    run_sub.font.name = 'Malgun Gothic'
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = MUTED_COLOR

    add_callout(
        "본 보고서는 사이드 런지 c1(앞다리 무릎 각도 90도 미달 결함) 판정 파이프라인에서 추천 옵션 2를 정식 채택하여 개발된 sl1_v11 신규 모델의 검증 결과를 기술합니다.\n"
        "1. CatBoost 가중치 대폭 강화 (LightGBM 0.15, CatBoost 0.65, XGBoost 0.20): 좌표 노이즈 및 가림에 대한 대칭 트리의 최고 강건성 반영\n"
        "2. 3-State [0.35, 0.65] 자연적 모호 구간 결합: 확정 F1-Score 71.93%(0.7193) 달성으로 목표 70% 완벽 돌파, 확정 정밀도 85.96%\n"
        "3. 5대 공식 시점 중 4대 시점(시점 B, C, D, E)에서 72.85% ~ 74.27%의 높은 F1 고른 성능 확보 및 가장 취약한 후방 시점(시점 A)도 65.68%로 방어",
        title="sl1_v11 핵심 성과 요약"
    )

    # 1. 아키텍처 개요
    h1 = doc.add_heading(level=1)
    h1.add_run("1. sl1_v11 아키텍처 및 하이퍼파라미터 개요")
    style_heading(h1, 13, PRIMARY_COLOR)

    add_p("sl1_v11은 이전 버전인 sl1_v10의 패러다임 2, 3, 4 기반 55D 생체역학 불변량 피처셋을 100% 온전히 계승하며, 앙상블 가중치 튜닝 실험에서 가장 높은 성능 향상을 입증한 '추천 옵션 2(CatBoost 편중 앙상블)'를 정식 채택한 신규 버전입니다.")

    table_spec = doc.add_table(rows=6, cols=2)
    table_spec.alignment = WD_TABLE_ALIGNMENT.CENTER
    spec_data = [
        ("항목", "sl1_v11 설정 사양"),
        ("기반 베이스라인", "sl1_v10 (패러다임 2, 3, 4 융합 55D 도메인 피처)"),
        ("데이터셋 규격", "AI-Hub 826개 전체 폴더 (4,130개 시퀀스, 9명 피험자 전수, 5대 시점 전수)"),
        ("앙상블 가중치", "LightGBM: 0.15 | CatBoost: 0.65 | XGBoost: 0.20 (추천 옵션 2)"),
        ("공식 주 정책 (Primary)", "3-State [0.35, 0.65] 자연적 모호 구간 판정보류 정책"),
        ("공식 보조 정책 (Secondary)", "5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0%)"),
    ]
    for r_idx, (col0, col1) in enumerate(spec_data):
        row = table_spec.rows[r_idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.text = col0
        c1.text = col1
        set_cell_margins(c0, top=80, bottom=80, left=120, right=120)
        set_cell_margins(c1, top=80, bottom=80, left=120, right=120)
        if r_idx == 0:
            set_cell_background(c0, "1A365D")
            set_cell_background(c1, "1A365D")
            for c in [c0, c1]:
                p = c.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(255, 255, 255)
        else:
            set_cell_background(c0, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
            set_cell_background(c1, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
            for c in [c0, c1]:
                p = c.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.size = Pt(9.5)

    # 2. 성능 지표 비교
    h2 = doc.add_heading(level=1)
    h2.add_run("2. 모델 버전별 공식 성능 지표 비교 (v10 vs v11)")
    style_heading(h2, 13, PRIMARY_COLOR)

    table_comp = doc.add_table(rows=4, cols=6)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    comp_data = [
        ("평가 정책", "지표", "sl1_v10 (균등 가중치)", "sl1_v11 (옵션 2 적용)", "변화량", "비고"),
        ("3-State [0.35, 0.65]\n(공식 주 정책)", "F1-Score\n정확도\n보류율", "71.39%\n86.19%\n15.67%", "71.93%\n85.96%\n17.89%", "+0.54%p\n-0.23%p\n+2.22%p", "목표 F1 70% 초과 달성\n(최고 안정성)"),
        ("3-State Adaptive\n(보류율 <= 10%)", "F1-Score\n정확도\n보류율", "70.43%\n85.06%\n9.88%", "70.37%\n85.26%\n9.83%", "-0.06%p\n+0.20%p\n-0.05%p", "보류율 10% 이내 통제"),
        ("2-State (Threshold 0.50)\n(폴백 정책)", "F1-Score\n정확도\nROC-AUC", "65.75%\n81.86%\n0.8545", "65.78%\n81.36%\n0.8547", "+0.03%p\n-0.50%p\n+0.0002", "기본 이진 분류선 유지"),
    ]
    for r_idx, row_values in enumerate(comp_data):
        row = table_comp.rows[r_idx]
        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            if r_idx == 0:
                set_cell_background(cell, "2B6CB0")
                p = cell.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
                p = cell.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.size = Pt(9)

    # 3. 5대 공식 시점별 세부 감사
    h3 = doc.add_heading(level=1)
    h3.add_run("3. 5대 공식 시점별 세부 성능 감사 (3-State [0.35, 0.65] 기준)")
    style_heading(h3, 13, PRIMARY_COLOR)

    table_view = doc.add_table(rows=6, cols=6)
    table_view.alignment = WD_TABLE_ALIGNMENT.CENTER
    view_data = [
        ("시점 코드", "공식 시점 명칭", "확정 F1-Score", "확정 정확도", "판정 보류율", "유효 판정 샘플"),
        ("view5", "시점 A (우측 후방 45°)", "65.68%", "82.37%", "20.34%", "658건 / 826건"),
        ("view2", "시점 B (우측 전방 45°)", "73.82%", "87.54%", "19.37%", "666건 / 826건"),
        ("view3", "시점 C (정면 0°)", "72.85%", "87.91%", "17.92%", "678건 / 826건"),
        ("view4", "시점 D (좌측 전방 45°)", "74.27%", "86.20%", "14.89%", "703건 / 826건"),
        ("view1", "시점 E (좌측 후방 45°)", "72.93%", "85.71%", "16.95%", "686건 / 826건"),
    ]
    for r_idx, row_values in enumerate(view_data):
        row = table_view.rows[r_idx]
        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            if r_idx == 0:
                set_cell_background(cell, "1A365D")
                p = cell.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(cell, "F7FAFC" if r_idx % 2 == 1 else "FFFFFF")
                p = cell.paragraphs[0]
                for r in p.runs:
                    r.font.name = 'Malgun Gothic'
                    r.font.size = Pt(9)

    # 4. 결론 및 배포 현황
    h4 = doc.add_heading(level=1)
    h4.add_run("4. 결론 및 프로덕션 배포 현황")
    style_heading(h4, 13, PRIMARY_COLOR)

    add_p("1. sl1_v11 모델은 CatBoost의 높은 강건성을 극대화하여 목표 지표인 F1 70%를 상회하는 71.93%를 공식 달성하였습니다.")
    add_p("2. 공식 프로덕션 번들(sl1_final_bundle.pkl) 및 챔피언 레지스트리(sl1_official_champion_registry.json)가 sl1_v11 기반으로 100% 갱신 완료되었습니다.")
    add_p("3. 단일 시퀀스 추론 및 5-Fold 앙상블 무결성 단위 테스트(test_sl1_final_pipeline.py, test_sl1_v11_pipeline.py)를 전수 통과하여 프로덕션 배포 준비를 완료하였습니다.")

    # Save Word document
    out_docx = r"D:\자주프 문서\7주차\사이드런지_c1_sl1_v11_CatBoost가중치_3State결합_공식성능평가보고서.docx"
    os.makedirs(os.path.dirname(out_docx), exist_ok=True)
    doc.save(out_docx)
    print(f"[+] DOCX 저장 완료: {out_docx}")

    # Save Markdown sibling
    out_md = r"D:\자주프 문서\7주차\사이드런지_c1_sl1_v11_CatBoost가중치_3State결합_공식성능평가보고서.md"
    md_content = f"""# 사이드 런지 c1 sl1_v11 공식 성능 평가 보고서
> **추천 옵션 2 적용: CatBoost 강화 가중치 [0.15, 0.65, 0.20] 및 3-State [0.35, 0.65] 자연적 모호 구간 융합 모델**

## 1. sl1_v11 핵심 성과 요약
- **공식 주 정책 (3-State [0.35, 0.65])**: 확정 F1-Score **71.93% (0.7193)**, 확정 정확도 **85.96%**, 판정 보류율 **17.89% (커버리지 82.11%)**
- **시점별 고른 성능**: 5대 시점 중 4대 시점(시점 B, C, D, E)에서 **72.85% ~ 74.27%** 달성, 취약 후방 시점(시점 A)도 **65.68%**로 견고히 방어
- **가중치 최적화**: LightGBM 0.15 : CatBoost 0.65 : XGBoost 0.20 (노이즈 강건성 극대화)

## 2. 5대 공식 시점별 세부 성능 감사
| 시점 코드 | 공식 시점 명칭 | 확정 F1-Score | 확정 정확도 | 판정 보류율 |
| :---: | :--- | :---: | :---: | :---: |
| `view5` | 시점 A (우측 후방 45°) | **65.68%** | 82.37% | 20.34% |
| `view2` | 시점 B (우측 전방 45°) | **73.82%** | 87.54% | 19.37% |
| `view3` | 시점 C (정면 0°) | **72.85%** | 87.91% | 17.92% |
| `view4` | 시점 D (좌측 전방 45°) | **74.27%** | 86.20% | 14.89% |
| `view1` | 시점 E (좌측 후방 45°) | **72.93%** | 85.71% | 16.95% |

## 3. 프로덕션 산출물 현황
- 학습 및 평가 스크립트: `D:/src/models/second_models/side_lunge/sl1_v11.py`
- 최종 배포 번들: `D:/src/models/second_models/side_lunge/sl1_final_bundle.pkl`
- 단위 테스트: `D:/src/models/second_models/side_lunge/test_sl1_v11_pipeline.py` (100% 통과)
"""
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] MD 저장 완료: {out_md}")

if __name__ == '__main__':
    create_report()
