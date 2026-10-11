# -*- coding: utf-8 -*-
"""
바벨 스쿼트 고정밀 모드 정상 오분류(False Negative) 원인 심층 분석 보고서 (.docx) 생성 스크립트
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX_PATH = r'D:\자주프 문서\4주차\스쿼트_고정밀모드_정상오분류(FN)_원인분석_보고서.docx'
OUT_MD_PATH = r'D:\자주프 문서\4주차\스쿼트_고정밀모드_정상오분류(FN)_원인분석_보고서.md'

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_callout_box(doc, title, text, bg_color='F0F4F8', border_color='1E3A8A'):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.font.name = 'Malgun Gothic'
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    run_b = p.add_run(text)
    run_b.font.name = 'Malgun Gothic'
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def generate_fn_report():
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # 타이틀
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("[오분류 심층 분석 보고서] 바벨 스쿼트 정상의 오류 오판(False Negative) 원인 규명")
    r_title.font.name = 'Malgun Gothic'
    r_title.font.size = Pt(16.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x0F, 0x2A, 0x4A)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("고정밀 모드(정밀도 55.6%)에서 탈락한 정상 74건(FN)의 생체역학·라벨링·피험자 편차 실측 분석")
    r_sub.font.name = 'Malgun Gothic'
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x55, 0x66, 0x77)

    # 메타정보
    meta_table = doc.add_table(rows=2, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_headers = [('작성 일자', '2026-09-19'), ('분석 대상', '고정밀 엄격모드 (th=0.225)'),
                    ('분석 규모', '전체 정상 129건 중 FN 74건'), ('검증 규격', '피험자 격리 5-Fold 전수 OOF')]
    for idx, (h, val) in enumerate(meta_headers):
        r = idx // 2
        c = (idx % 2) * 2
        cell_h = meta_table.cell(r, c)
        cell_v = meta_table.cell(r, c + 1)
        cell_h.width = Inches(1.3)
        cell_v.width = Inches(2.0)
        set_cell_background(cell_h, 'EAEEF3')
        set_cell_background(cell_v, 'F8FAFC')
        
        ph = cell_h.paragraphs[0]
        ph.paragraph_format.space_after = Pt(2)
        rh = ph.add_run(h)
        rh.font.name = 'Malgun Gothic'
        rh.font.size = Pt(9)
        rh.font.bold = True
        
        pv = cell_v.paragraphs[0]
        pv.paragraph_format.space_after = Pt(2)
        rv = pv.add_run(val)
        rv.font.name = 'Malgun Gothic'
        rv.font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 총평 Callout
    exec_summary = (
        "1. [정상인데 왜 오류라고 했는가?]: 실제 정상 129건 중 74건이 오류로 판정된 가장 결정적인 이유는 "
        "'이들이 실제 오류 집단과 생체역학적으로 거의 구별되지 않는 얕은 스쿼트(가동범위 20.2도 vs TP 30.8도)를 수행'했기 때문입니다.\n"
        "2. [인간 라벨러와 AI 기준의 괴리]: AI-Hub 라벨러는 끝까지 깊게 앉지 않아도 4대 조건만 대략 안 틀리면 "
        "'정상'으로 후하게 태깅한 반면, AI 모델은 가동범위와 무릎 간격이 실제 오류 집단(19.5도)과 완벽히 겹치는 것을 감지하여 엄격하게 탈락시켰습니다.\n"
        "3. [피험자 체형 편차]: 특정 피험자(Subject 8: 85.7% FN, Subject 4: 70.6% FN)에게서 오분류가 집중 발생하여, "
        "개인별 고유 골격비 및 스쿼트 깊이 습관이 미학습 일반화(Unseen Subject)에서 기각 요인으로 작용했습니다.\n"
        "4. [결론 및 해결책]: 모델이 멍청해서 틀린 것이 아니라 '오탐(FP) 212건을 박멸하기 위해 문턱을 높인 필연적 결과'이며, "
        "일반 서비스에서는 '균형 모드(th=0.180)' 또는 '고재현 모드(th=0.070, 정상 재현율 72.1%)'를 적용하면 즉시 해결됩니다."
    )
    create_callout_box(doc, "핵심 분석 결론 요약 (Key Findings)", exec_summary, bg_color='EFF6FF', border_color='2563EB')

    # 1장: 현황 진단
    doc.add_heading("1. 현상 진단: TP(정답 정상 55건) vs FN(오분류 정상 74건)", level=1)
    doc.add_paragraph(
        "바벨 스쿼트 고정밀 엄격모드는 오탐(False Positive)을 256건에서 44건으로 82.8% 박멸하고 정밀도를 55.56%까지 끌어올렸으나, "
        "실제 정상 129건 중 55건만 정상으로 인정하고 74건을 오류로 분류(False Negative, 정상 재현율 42.64%)했습니다.\n"
        "이 74건이 왜 오류로 분류되었는지 모델 내부 점수 및 신호를 비교 분석했습니다:"
    )

    # 점수 비교 표
    tbl_scores = doc.add_table(rows=6, cols=4)
    tbl_scores.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_headers = ["내부 판정 신호 (Signals)", "TP (정상 판정 성공 55건)", "FN (정상인데 오류 오판 74건)", "차이 및 분석"]
    s_widths = [Inches(2.2), Inches(1.4), Inches(1.4), Inches(1.5)]
    
    for c_idx, h in enumerate(s_headers):
        cell = tbl_scores.rows[0].cells[c_idx]
        cell.width = s_widths[c_idx]
        set_cell_background(cell, '1E3A8A')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Malgun Gothic'
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    s_rows = [
        ("M0 정상 게이트키퍼 확률", "0.511 (정상 확신)", "0.135 (오류 의심)", "M0 자체가 오류로 강하게 판정"),
        ("3단계 세부 결함 결합 점수", "0.252 (결함 없음)", "0.469 (결함 감지)", "세부 결함 모델들이 결함 의심"),
        ("  - c1. 척추의 중립 결함 확률", "0.132", "0.411 (+0.279p)", "FN에서 상체 숙임/척추 불안정 감지"),
        ("  - c4. 발바닥 지면 고정 결함", "0.217", "0.376 (+0.159p)", "FN에서 미세한 발목 들림 감지"),
        ("최종 소프트 퓨전 점수", "0.383 (통과)", "0.079 (문턱 0.225 미달)", "임계값 0.225에 크게 못 미침")
    ]
    for r_idx, row in enumerate(s_rows):
        cells = tbl_scores.rows[r_idx + 1].cells
        for c_idx in range(4):
            cells[c_idx].width = s_widths[c_idx]
            set_cell_background(cells[c_idx], 'F8FAFC' if r_idx % 2 == 0 else 'FFFFFF')
            set_cell_margins(cells[c_idx], top=60, bottom=60, left=80, right=80)
            p = cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 3 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(row[c_idx])
            r.font.name = 'Malgun Gothic'
            r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 2장: 생체역학적 원인
    doc.add_heading("2. 생체역학적 핵심 차이: 얕은 가동범위(Shallow Squat)와 좁은 무릎", level=1)
    doc.add_paragraph(
        "정상 129건 전체에 대해 169개 관절 기하학 및 키네마틱 피처의 통계적 유의성(Z-score 차이)을 전수 조사한 결과, "
        "정상으로 통과한 TP와 오류로 탈락한 FN 사이에는 명확한 생체역학적 패턴 차이가 존재했습니다:"
    )

    tbl_feat = doc.add_table(rows=7, cols=5)
    tbl_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
    f_headers = ["핵심 생체역학 지표", "TP (정상 통과)", "FN (오류 오판)", "실제 오류 집단 (1,845건)", "기전 분석"]
    f_widths = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.3)]
    for c_idx, h in enumerate(f_headers):
        cell = tbl_feat.rows[0].cells[c_idx]
        cell.width = f_widths[c_idx]
        set_cell_background(cell, '1E3A8A')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Malgun Gothic'
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    f_rows = [
        ("무릎 굴곡 가동범위 (Delta)", "30.83도", "20.25도", "19.52도", "FN이 오류 집단과 가동범위 동일!"),
        ("최하사점 무릎 굴곡각 (Min)", "144.06도", "151.74도", "151.16도", "FN은 덜 굽힌 얕은 스쿼트"),
        ("최하사점 양 무릎 간격", "105.2 cm", "92.6 cm", "83.2 cm", "FN은 무릎을 넓게 벌리지 못함"),
        ("오른쪽 팔꿈치 각도 (Mean)", "38.30도", "50.56도", "58.98도", "FN은 상체/팔꿈치 견착 불안정"),
        ("정강이 수직 각도 (Min)", "5.10도", "7.07도", "8.72도", "FN은 정강이가 과도하게 흔들림"),
        ("무릎 외반각 최대치 (Valgus)", "8.26도", "12.07도", "18.55도", "FN은 무릎 말림 경계선에 위치")
    ]
    for r_idx, row in enumerate(f_rows):
        cells = tbl_feat.rows[r_idx + 1].cells
        for c_idx in range(5):
            cells[c_idx].width = f_widths[c_idx]
            set_cell_background(cells[c_idx], 'F8FAFC' if r_idx % 2 == 0 else 'FFFFFF')
            set_cell_margins(cells[c_idx], top=60, bottom=60, left=80, right=80)
            p = cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 4 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(row[c_idx])
            r.font.name = 'Malgun Gothic'
            r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3장: 왜 이런 문제가 생겼는가? 4대 근본 요인
    doc.add_heading("3. 왜 이런 문제가 생기는가? (4대 근본 요인)", level=1)
    
    p_why = doc.add_paragraph()
    p_why.add_run(
        "① [인간 라벨러의 관용성과 라벨 노이즈 (Annotation Granularity)]\n"
        "   - AI-Hub 원천 데이터 구축 시 라벨러는 '척추 중립', '고개 정면', '무릎 방향', '발바닥 고정'의 4가지만 체크했습니다. "
        "즉, '엉덩이가 충분히 내려갔는가(스쿼트 깊이/가동범위)'는 독립된 결함 조건에 포함되지 않았습니다.\n"
        "   - 따라서 회원이 엉덩이를 20도만 살짝 내리는 쿼터/하프 스쿼트를 하더라도 라벨러는 '정상'으로 태깅했습니다. "
        "그러나 AI 모델은 가동범위 20도짜리 동작을 실제 1,845개의 오류 집단(평균 19.5도)과 동일한 패턴으로 학습하여 엄격하게 기각한 것입니다.\n\n"
        "② [오탐 차단을 위한 임계값 상향의 반작용 (Precision-Recall Trade-off)]\n"
        "   - 기존 단순 게이트키퍼(th=0.15)에서는 이 74건 중 상당수가 정상으로 통과되었습니다(당시 정상 재현율 65.12%). "
        "그러나 그 대가로 1,845개의 오류 중 무려 256개가 정상으로 뚫고 들어와 정밀도가 24.71%로 붕괴되었습니다.\n"
        "   - 256개의 가짜 정상을 44개로 틀어막기 위해 문턱을 0.225로 높이자, 오류 집단과 경계선상에 걸쳐 있던 "
        "'자세가 덜 성숙한 정상(FN 74건)'들이 함께 튕겨져 나간 것입니다.\n\n"
        "③ [피험자 격리(Unseen Subject) 평가의 체형 편차]\n"
        "   - 피험자별 FN 발생률을 분석한 결과, 피험자 1번(35.3%)이나 5번(36.4%)은 정상을 대부분 잘 맞춘 반면, "
        "피험자 8번(85.7% FN)과 피험자 4번(70.6% FN)에게서 오분류가 집중되었습니다.\n"
        "   - 모델을 학습할 때 피험자 8번의 데이터가 완전히 배제된 상태(StratifiedGroupKFold)에서 평가되었기 때문에, "
        "해당 피험자의 유난히 좁은 골반 너비나 얕은 가동범위 습관이 이상치로 분류되었습니다.\n\n"
        "④ [단안 2D-to-3D 정면 시점(View1)의 전후 깊이 모호성]\n"
        "   - 정상 129건은 100% 정면 카메라(View1) 데이터입니다. 정면 시점에서는 전후 방향의 깊이(Z축 변위)가 카메라 축과 평행하여 "
        "고관절의 실제 하강 폭이 시각적으로 압축되어 보입니다. 이로 인해 모델이 하강 가동범위를 실제보다 더 얕게 인식하는 편향이 발생합니다."
    )

    # 4장: 해결 방안
    doc.add_heading("4. 해결 방안 및 현장 적용 가이드", level=1)
    
    sol_text = (
        "1. [운영 모드 스위칭: 현장 목적에 따른 즉시 해결]\n"
        "   - 만약 '정상을 억울하게 틀렸다고 하지 않는 것'이 더 중요한 서비스라면, 모델을 재학습할 필요 없이 "
        "설정값(theta_fused)만 변경하면 즉시 해결됩니다:\n"
        "     * 균형 운영모드 (th=0.180): 정상 재현율 47.29% | 정밀도 45.52% | FP 73건\n"
        "     * 고재현 관용모드 (th=0.070): 정상 재현율 72.09% | 정밀도 29.15% | Balanced Acc 79.92% (최고 균형)\n\n"
        "2. [데이터 엔지니어링: 가동범위(ROM) 기반 라벨 정제]\n"
        "   - 정상 129건 중 무릎 각도 변화량이 25도 미만인 극단적 얕은 스쿼트(Quarter Squat) 30여 건을 "
        "'부분 정상(Partial Normal)' 또는 별도 클래스로 분리 라벨링하여 모델의 경계선 혼란을 방지.\n\n"
        "3. [피험자 정규화(Body Proportion Normalization)]\n"
        "   - 대퇴골 길이 대 정강이 길이 비율(Femur-to-Tibia Ratio)로 가동범위를 정규화하여 피험자 8번과 같은 체형별 개인차 보정."
    )
    create_callout_box(doc, "실무 솔루션 및 제언", sol_text, bg_color='F8FAFC', border_color='0D9488')

    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[+] Word 보고서 생성 완료: {OUT_DOCX_PATH}")

    # 동반 Markdown 생성
    with open(OUT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write("# [오분류 심층 분석 보고서] 바벨 스쿼트 정상의 오류 오판(False Negative) 원인 규명\n\n")
        f.write(f"{exec_summary}\n\n")
        f.write("## 1. TP(정답 정상) vs FN(오분류 정상) 생체역학 지표 실측 비교\n\n")
        f.write("| 핵심 생체역학 지표 | TP (정상 통과 55건) | FN (오류 오판 74건) | 실제 오류 집단 (1,845건) | 기전 분석 |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        for row in f_rows:
            f.write(f"| **{row[0]}** | {row[1]} | **{row[2]}** | {row[3]} | {row[4]} |\n")
        f.write("\n## 2. 왜 이런 문제가 발생하는가? (4대 근본 요인)\n\n")
        f.write("- **① 얕은 스쿼트(Shallow ROM)**: FN 74건은 가동범위가 20.2도로 오류 집단(19.5도)과 동일하여 모델이 기각함.\n")
        f.write("- **② 인간 라벨러의 관용성**: AI-Hub 라벨러는 깊이가 얕아도 4대 결함이 없으면 '정상'으로 후하게 판정함.\n")
        f.write("- **③ 오탐 차단의 필연적 반작용**: 256건의 가짜 정상을 44건으로 차단하는 과정에서 경계선 정상들이 탈락함.\n")
        f.write("- **④ 피험자 격리 체형 편차**: 피험자 8번(85.7% FN), 4번(70.6% FN) 등 미학습 개인 체형 차이가 기각을 유발함.\n")

    print(f"[+] Markdown 보고서 생성 완료: {OUT_MD_PATH}")

if __name__ == '__main__':
    generate_fn_report()
