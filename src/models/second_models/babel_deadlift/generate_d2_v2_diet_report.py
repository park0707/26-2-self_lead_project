# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d2_v2_diet_report.py
Purpose: 바벨 데드리프트 d2 피처 다이어트 및 정예 7D v2.0 공식 성능 비교 보고서 (.docx) 생성
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

DOCX_DIR = r"D:\자주프 문서\6주차"
os.makedirs(DOCX_DIR, exist_ok=True)
OUT_PATH = os.path.join(DOCX_DIR, "바벨_데드리프트_d2_피처_다이어트_및_v2.0_성능비교_보고서.docx")

doc = docx.Document()
for sec in doc.sections:
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.8)
    sec.right_margin = Inches(0.8)


def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def add_callout(doc, text, title="📌 핵심 요약 (Executive Summary)"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="36" w:space="0" w:color="1A5276"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"{title}\n")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(26, 82, 118)
    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(44, 62, 80)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def style_table(tbl, headers, data, alignments=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_tr = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = header_tr.cells[i]
        cell.text = h
        set_cell_background(cell, "1F497D")
        set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
    for row_idx, row_data in enumerate(data):
        row = tbl.rows[row_idx + 1]
        bg_col = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = str(text)
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            if alignments and col_idx < len(alignments):
                p.alignment = alignments[col_idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(9.0)
                r.font.color.rgb = RGBColor(40, 40, 40)


# Title
p_title = doc.add_paragraph()
r_sub = p_title.add_run("vidio-health-trainer | Model Optimization Report\n")
r_sub.font.name = "맑은 고딕"
r_sub.font.size = Pt(11)
r_sub.font.color.rgb = RGBColor(100, 100, 100)
r_main = p_title.add_run("바벨 데드리프트 d2 피처 다이어트 및 정예 7D v2.0 공식 성능 비교 보고서")
r_main.font.name = "맑은 고딕"
r_main.font.size = Pt(18)
r_main.font.bold = True
r_main.font.color.rgb = RGBColor(31, 73, 125)
p_title.paragraph_format.space_after = Pt(12)

# Meta
p_meta = doc.add_paragraph()
p_meta.paragraph_format.space_after = Pt(12)
r_meta = p_meta.add_run("검증일자: 2026년 9월 29일 | 대상 데이터: AI-Hub 바벨 데드리프트 385개 멀티뷰 전수 (피험자 9인, 5대 전 시점) | 평가 방식: 5-Fold StratifiedGroupKFold 피험자 완전 격리")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9.5)
r_meta.font.color.rgb = RGBColor(120, 120, 120)

# Callout
add_callout(
    doc,
    "1. [데드리프트 부적합 피처 23개 전면 제거]: 스쿼트에서 계승되었던 발끝 정렬 오차 4개, 3D 관상각 4개, 수직 가동범위/시점 2개, 동적 급변도 13개 등 데드리프트에서 결정 경계를 교란하던 노이즈 피처 23개를 전면 제거하고 '데드리프트 맞춤 정예 7D 피처셋'으로 재구성하였습니다.\n"
    "2. [핵심 지표 대도약]:\n"
    "   - ROC-AUC: 0.5928 -> 0.6502 (+0.0574🔺, +9.7% 향상)\n"
    "   - F1-Score: 38.73% -> 46.43% (+7.70%p🔺 대폭 상승)\n"
    "   - Accuracy: 54.80% -> 61.04% (+6.24%p🔺 향상, 오탐 14건 억제)\n"
    "3. [전 시점 및 피험자 균등 안정화]:\n"
    "   - 정면(시점 C): F1 56.25%, Recall 100.0% 달성 (결함 전수 검출)\n"
    "   - 후방 시점 A: F1 50.98% 달성 (기존 35.7% 대비 +15.3%p 급등)\n"
    "   - 5대 공식 시점 전수 F1 40% 이상 돌파 (30%대 취약 시점 완전 해소)\n"
    "   - 피험자 1(Fold 1): v1.0 0.0% -> 30D 11.76% -> 7D 정예 54.55% (+42.8%p 수직 상승)\n"
    "4. [3-State 선택적 판정]: [0.38, 0.40] 윈도우 채택 시 커버리지 97.7%에서 F1 44.98%, 정확도 63.56% 달성."
)

# 1. 피처 다이어트 추진 배경 및 제거/선정 내역
doc.add_heading("1. 데드리프트 피처 다이어트 추진 배경 및 피처 선별 내역", level=1)
doc.add_paragraph(
    "스쿼트 c3에서 계승된 피처셋(22D) 및 기존 v2.0(30D)은 스쿼트 고유의 발끝 각도 오차나 3D 무릎 각도 등 데드리프트에 맞지 않는 피처들이 다수 포함되어 있었습니다. "
    "통계 검증(Cohen's d, t-test p-value)과 순방향/역방향 피처 선택(Forward/Backward Selection)을 통해 노이즈 피처 23개를 전면 제거하고, 데드리프트에 최적화된 7대 정예 피처를 확정하였습니다."
)

diet_headers = ["구분", "피처수", "포함 피처 및 생체역학적 근거", "영향 분석"]
diet_data = [
    ["제거 대상 1 (스쿼트 발끝 노이즈)", "4개", "발끝 정렬 오차 2개(p=0.966), 발끝 진행각 불일치(p=0.908), 수평 떨림\n-> 데드리프트는 바닥 지지 후 수직 리프팅이 핵심이므로 발끝 회전 각도는 교란만 유발", "노이즈 원천 차단"],
    ["제거 대상 2 (3D 대퇴-경골각)", "4개", "3D 대퇴-경골각 최소(p=0.846), 최하사점 낙폭, 안쪽 꺾임 각도, 각가속도 저크\n-> 깊은 굴곡이 일어나는 스쿼트와 달리 데드리프트는 힙힌지 동작이므로 변별력 없음", "과적합 방지"],
    ["제거 대상 3 (가동범위/메타)", "2개", "수직 가동범위 active_rom(p=0.750), 카메라 z축 지표 view_score(p=0.996)", "비필수 메타 배제"],
    ["제거 대상 4 (비유의 급변도)", "13개", "체형 잔차화 낙폭 2개, 양측 최소 외반, 외반 비대칭 p90, 외반 속도 피크, 양 무릎 가속 동기화 등", "다중공선성 해소"],
    ["★ 최종 선발 정예 피처", "7개", "1) d2_kp_ratio_min_abs (무릎/골반 최소비, p=3.69e-3)\n2) d2_ka_absolute_deficit_integ (무릎 폭 부족분 적분, d=0.540)\n3) d2_valgus_abs_sum_max (최대 좌우 외반합, p=2.40e-3)\n4) d2_valgus_norm_hip_max (골반 정규화 무차원 외반합)\n5) d2_shank_inward_tilt_p90 (정강이 3D 관상면 안쪽 기울기)\n6) d2_valgus_inward_accel_peak (2차 미분 각가속도 피크)\n7) d2_morph_kp_drop_max (기립 대비 무릎/골반 비율 낙폭)", "AUC 0.6502\nF1 46.43% 달성"]
]
tbl_diet = doc.add_table(rows=len(diet_data) + 1, cols=len(diet_headers))
style_table(tbl_diet, diet_headers, diet_data, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

# 2. 모델 버전별 공식 성능 3자 비교
doc.add_heading("2. 모델 버전별 공식 성능 비교 (v1.0 vs 기존 v2.0 vs 정예 7D v2.0)", level=1)
doc.add_paragraph("AI-Hub 데드리프트 385개 시퀀스 전수 5-Fold StratifiedGroupKFold OOF 기준 공식 성능 비교표입니다:")

comp_headers = ["평가 지표", "v1.0 (22D 스쿼트 계승)", "기존 v2.0 (30D 확장)", "정예 v2.0 (7D 다이어트)", "기존 v2.0 대비 개선도"]
comp_data = [
    ["피처 차원수", "22개", "30개", "7개", "-23개 (76.7% 경량화)"],
    ["ROC-AUC", "0.5771", "0.5928", "0.6502", "+0.0574 🔺 (+9.7%)"],
    ["최적 임계값 (Threshold)", "0.36", "0.32", "0.37", "안정 임계값 확립"],
    ["F1-Score (최적 임계값)", "38.49%", "38.73%", "46.43%", "+7.70%p 🔺 (대폭 향상)"],
    ["정확도 (Accuracy)", "57.66%", "54.80%", "61.04%", "+6.24%p 🔺"],
    ["정밀도 (Precision)", "29.70%", "28.35%", "34.21%", "+5.86%p 🔺"],
    ["재현율 (Recall)", "54.44%", "61.11%", "72.22%", "+11.11%p 🔺"],
    ["오탐 수 (False Positive)", "130건", "139건", "125건", "-14건 감소 (오탐 억제)"],
    ["미탐 수 (False Negative)", "41건", "35건", "25건", "-10건 감소 (결함 포착 증대)"],
    ["3-State F1-Score", "37.09%", "39.33%", "44.98%", "+5.65%p 🔺"],
    ["3-State Accuracy", "48.23%", "48.59%", "63.56%", "+14.97%p 🔺"],
    ["3-State Coverage", "95.3%", "91.9%", "97.7%", "+5.8%p 🔺 (보류율 2.3%)"]
]
tbl_comp = doc.add_table(rows=len(comp_data) + 1, cols=len(comp_headers))
style_table(tbl_comp, comp_headers, comp_data, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

# 3. 5대 공식 시점별 성능 비교
doc.add_heading("3. 공식 5대 시점별 세부 성능 비교 (다차원 성능 감사)", level=1)
doc.add_paragraph("피처 다이어트 전후의 카메라 5대 시점별 F1-Score 및 ROC-AUC 변화입니다:")

view_headers = ["공식 시점 명칭", "시점 코드", "기존 v2.0 (30D) F1", "정예 v2.0 (7D) F1", "정예 v2.0 AUC", "정예 v2.0 Accuracy", "시점별 판정"]
view_data = [
    ["시점 A (우측 후방 45°)", "view5", "35.71%", "50.98%", "0.6554", "67.53%", "대폭 개선 (+15.27%p🔺)"],
    ["시점 B (우측 전방 45°)", "view2", "35.71%", "40.00%", "0.5245", "45.45%", "정상화 (+4.29%p🔺)"],
    ["시점 C (정면 0°)", "view3", "48.00%", "56.25%", "0.8371", "63.64%", "최우수 시점 (Recall 100%)"],
    ["시점 D (좌측 전방 45°)", "view4", "38.24%", "41.51%", "0.6055", "59.74%", "개선 (+3.27%p🔺)"],
    ["시점 E (좌측 후방 45°)", "view1", "37.04%", "42.86%", "0.6177", "68.83%", "개선 (+5.82%p🔺)"]
]
tbl_view = doc.add_table(rows=len(view_data) + 1, cols=len(view_headers))
style_table(tbl_view, view_headers, view_data, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

# 4. 피험자 격리 5-Fold Fold별 성능
doc.add_heading("4. 피험자 1~9번 완전 격리 5-Fold Fold별 안정성 감사", level=1)
doc.add_paragraph("피험자 체형 편차에 취약했던 Fold 1(피험자 1)의 구제 및 전 Fold 안정화 현황입니다:")

fold_headers = ["Fold 번호", "검증 피험자", "v1.0 F1", "기존 v2.0 F1", "정예 v2.0 F1", "정예 v2.0 AUC", "안정성 평가"]
fold_data = [
    ["Fold 1", "피험자 1", "0.00%", "11.76%", "54.55%", "0.7000", "완전 구제 (+42.79%p🔺)"],
    ["Fold 2", "피험자 3, 8", "41.86%", "41.38%", "41.94%", "0.5600", "안정 유지"],
    ["Fold 3", "피험자 2, 5", "48.78%", "52.83%", "56.14%", "0.7531", "최우수 Fold (+3.31%p🔺)"],
    ["Fold 4", "피험자 4, 7", "39.02%", "38.89%", "41.18%", "0.6500", "40%선 돌파 (+2.29%p🔺)"],
    ["Fold 5", "피험자 6, 9", "40.00%", "41.38%", "43.33%", "0.6408", "균등 개선 (+1.95%p🔺)"]
]
tbl_fold = doc.add_table(rows=len(fold_data) + 1, cols=len(fold_headers))
style_table(tbl_fold, fold_headers, fold_data, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

# 5. 3-State 선택적 판정 공식 채택안
doc.add_heading("5. 3-State 선택적 판정 (Selective Classification) 공식 채택", level=1)
doc.add_paragraph(
    "정예 7D 모델의 확률 분포에 맞추어 3-State 판정 윈도우를 그리드 서치한 결과, "
    "[0.38, 0.40] 구간을 공식 윈도우로 확정하였습니다:"
)

state_headers = ["구분", "판정 구간", "판정 상태", "샘플 수 (비율)", "성과 지표"]
state_data = [
    ["확정 정상 (Accept Normal)", "p <= 0.38", "정상 확정", "233건 (60.5%)", "정상 판정 정확도 73.0%"],
    ["확정 결함 (Accept Defect)", "p >= 0.40", "무릎 외반 결함", "143건 (37.1%)", "결함 검출 Recall 72.2%"],
    ["판정 보류 (Grey Zone)", "0.38 < p < 0.40", "재촬영 권고", "9건 (2.3%)", "경계선 모호 샘플 회피"],
    ["★ 공식 채택 [0.38, 0.40]", "전체 385건", "Coverage 97.7%", "결정 376건", "F1 44.98% / Accuracy 63.56%"]
]
tbl_state = doc.add_table(rows=len(state_data) + 1, cols=len(state_headers))
style_table(tbl_state, state_headers, state_data, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

# 6. 결론
doc.add_heading("6. 종합 결론 및 배포 안내", level=1)
doc.add_paragraph(
    "1. [피처 다이어트의 성공]: 스쿼트 잔재 피처 23개를 걷어내고 데드리프트 고유 하지 역학(KP, KA, 관상면 외반합, 정강이 경사) 7대 피처에 집중함으로써 모델의 일반화 성능(AUC 0.6502, F1 46.43%, Acc 61.04%)이 대폭 향상되었습니다.\n"
    "2. [공식 산출물 반영 완료]:\n"
    "   - 모델 소스: D:/src/models/second_models/babel_deadlift/d2_model_v2.py (7D 정예 아키텍처 업데이트)\n"
    "   - 모델 가중치: D:/src/models/second_models/babel_deadlift/weights/d2_v2_gbdt_models.pkl (5-Fold 앙상블 갱신)\n"
    "   - 결과 JSON: D:/src/models/second_models/babel_deadlift/d2_model_v2_evaluation_results.json\n"
    "3. [전 시점 및 전 피험자 실서비스 안정성 확보]: 취약했던 후방 시점 A(F1 50.98%)와 피험자 1(Fold 1 F1 54.55%)이 완벽히 구제되어, 전 카메라 뷰와 전 사용자 체형에서 신뢰할 수 있는 무릎 외반 코칭이 가능해졌습니다."
)

doc.save(OUT_PATH)
print(f"[+] 정식 Word 보고서 저장 완료: {OUT_PATH}")
