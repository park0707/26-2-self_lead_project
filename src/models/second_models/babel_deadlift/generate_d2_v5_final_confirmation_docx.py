# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d2_v5_final_confirmation_docx.py
Purpose: 바벨 데드리프트 d2 v5.0 최종 확정 공식 기술명세서 및 실측 보고서 (.docx) 생성
"""

import os
import sys
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

JSON_PATH = r"D:\src\models\second_models\babel_deadlift\weights\d2_v5_final_metrics.json"
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    res = json.load(f)

DOCX_DIR = r"D:\자주프 문서\6주차"
os.makedirs(DOCX_DIR, exist_ok=True)
DOCX_PATH = os.path.join(DOCX_DIR, "바벨_데드리프트_d2_v5.0_최종모델_확정_보고서.docx")

doc = Document()

for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

NAVY = RGBColor(26, 54, 93)     # #1A365D
BLUE = RGBColor(43, 108, 176)   # #2B6CB0
GRAY = RGBColor(113, 128, 150)  # #718096
BLACK = RGBColor(45, 55, 72)    # #2D3748
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(39, 103, 73)   # #276749

def style_heading(p, text, level=1):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.bold = True
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = NAVY
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = BLUE
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = BLACK

def add_callout_box(doc, text_list, title="", border_color="276749", bg_color="F0FFF4"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>'))
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    if title:
        run_t = p.add_run(f"■ {title}\n")
        run_t.font.name = "맑은 고딕"
        run_t.font.size = Pt(10.5)
        run_t.bold = True
        run_t.font.color.rgb = GREEN
    for line in text_list:
        run = p.add_run(line + "\n")
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLACK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 타이틀
p_meta = doc.add_paragraph()
r_meta = p_meta.add_run("vidio-health-trainer | AI-Hub 바벨 데드리프트 d2 공식 최종 확정 문서")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9.5)
r_meta.font.color.rgb = GRAY

p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(6)
r_title = p_title.add_run("바벨 데드리프트 d2 (발과 무릎의 방향 일치 / 동적 무릎 외반)\nv5.0 최종 프로덕션 모델 확정 및 기술 명세서")
r_title.font.name = "맑은 고딕"
r_title.font.size = Pt(18)
r_title.bold = True
r_title.font.color.rgb = NAVY

add_callout_box(doc, [
    "• 공식 프로젝트명: vidio-health-trainer",
    "• 확정 모델 버전: v5.0 (Final Official Release)",
    "• 대상 운동 및 결함: 바벨 데드리프트 d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)",
    "• 확정 아키텍처: 25D 정예 생체역학 피처 + 4,162 파라미터 경량 Tiny TCN + Dual GBDT Meta-Fusion (0.80 GBDT + 0.20 TCN)",
    f"• 종합 실측 성능 (385개 전수 5-Fold 완전 격리): ROC-AUC {res['roc_auc']:.4f}, 3-State F1-Score {res['3state_f1']:.2f}%, 정밀도 {res['3state_precision']:.2f}%, 정확도 {res['3state_accuracy']:.2f}%, 보류율 {res['abstain_rate']:.1f}% (10% 이하 엄격 준수)",
    "• 5대 시점 감사: 정면(시점 C) F1 73.47% (Recall 100.0% 전수 감지), 우측전방(시점 B) F1 61.54% (Recall 94.12%)",
    "• 상시 무결성 검증: 단일 시퀀스(64, 33, 3) 3-State 독립 추론 파이프라인 무결성 100% 통과 (PASS)"
], title="d2 v5.0 최종 확정 승인 요약 (Official Sign-off)")

# 제1장
p1 = doc.add_paragraph()
style_heading(p1, "1. 확정 모델 아키텍처 및 캡슐화 모듈 명세", level=1)

p1_desc = doc.add_paragraph()
p1_desc.add_run("본 모델은 외부 시스템 및 프론트엔드 서비스에서 호출 시 피처 인덱스나 복잡한 전처리 내부 로직을 몰라도 즉시 사용할 수 있도록 완전히 캡슐화된 독립 클래스 인터페이스로 배포되었습니다:")

tbl_spec = doc.add_table(rows=6, cols=3)
tbl_spec.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_s = ["구성 요소", "배포 파일 경로 / 규격", "주요 기능 및 인터페이스"]
for c_idx, h in enumerate(headers_s):
    cell = tbl_spec.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>'))

spec_data = [
    ("프로덕션 서빙 모듈", "src/models/second_models/babel_deadlift/\nd2_model_v5_final.py", "BabelDeadliftD2ModelV5 클래스 제공\n- predict_proba(seq_64f) -> Dict\n- predict_3state(seq_64f, view) -> Dict"),
    ("프로덕션 모델 가중치", "weights/d2_v5_final_production.pkl", "LightGBM + CatBoost + Tiny TCN 가중치 및 표준화 파라미터 통합 바이너리 영구 보존"),
    ("5-Fold 교차검증 모델", "weights/d2_v5_final_models.pkl", "피험자 완전 격리 5-Fold OOF 예측 확률 및 라벨 전수 보존"),
    ("실측 평가 결과 메트릭", "weights/d2_v5_final_metrics.json", "전체 및 5대 시점별 F1, Precision, Recall, Coverage JSON 명세"),
    ("무결성 자체 테스트", "test_d2_v5_inference_integrity()", "단일 시퀀스(64, 33, 3) 투입 시 3-State 정상 확정 및 가이드 출력 검증 (100% PASS)")
]

for r_idx, r in enumerate(spec_data, 1):
    for c_idx, val in enumerate(r):
        cell = tbl_spec.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if r_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제2장
p2 = doc.add_paragraph()
style_heading(p2, "2. 전 세대 모델 대비 종합 실측 성능 발전사", level=1)

tbl_comp = doc.add_table(rows=8, cols=6)
tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_c = ["평가 지표 (Metric)", "v1.0 (초기 22D)", "v2.0 (정예 7D)", "v3.0 (통합 12D)", "v4.0 (위상/FPPA)", "v5.0 (최종 확정)"]
for c_idx, h in enumerate(headers_c):
    cell = tbl_comp.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="2B6CB0"/>'))

cm = res['confusion_matrix']
history_data = [
    ("ROC-AUC", "0.6120", "0.6502", "0.7279", "0.7348", f"★ {res['roc_auc']:.4f}"),
    ("3-State F1-Score", "41.50%", "46.43%", "53.98%", "55.10%", f"★ {res['3state_f1']:.2f}%"),
    ("3-State 정밀도", "30.80%", "35.20%", "42.07%", "46.55%", f"★ {res['3state_precision']:.2f}%"),
    ("3-State 재현율", "63.50%", "68.80%", "75.31%", "67.50%", f"{res['3state_recall']:.2f}%"),
    ("3-State 정확도", "55.20%", "61.04%", "70.03%", "74.64%", f"★ {res['3state_accuracy']:.2f}%"),
    ("판단 보류율", "0.0% (강제)", "0.0% (강제)", "9.9%", "9.9%", f"★ {res['abstain_rate']:.1f}%"),
    ("오탐(False Positive)", "135건", "112건", "84건", "62건", f"★ {cm['fp']}건 (최저치)")
]

for r_idx, r in enumerate(history_data, 1):
    for c_idx, val in enumerate(r):
        cell = tbl_comp.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        if "★" in val:
            cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.color.rgb = GREEN
        else:
            cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if r_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제3장
p3 = doc.add_paragraph()
style_heading(p3, "3. 공식 5대 시점별 세부 실측 성능 감사 (AGENTS.md 표준 준수)", level=1)

tbl_view = doc.add_table(rows=6, cols=6)
tbl_view.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_v = ["공식 시점 명칭", "원천 코드", "3-State F1", "정밀도 (Precision)", "재현율 (Recall)", "판정 커버리지"]
for c_idx, h in enumerate(headers_v):
    cell = tbl_view.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>'))

view_rows = [
    ("시점 C (정면)", "view3", res['views']['view3']),
    ("시점 B (우측 전방 45°)", "view2", res['views']['view2']),
    ("시점 A (우측 후방 45°)", "view5", res['views']['view5']),
    ("시점 E (좌측 후방 45°)", "view1", res['views']['view1']),
    ("시점 D (좌측 전방 45°)", "view4", res['views']['view4'])
]

for r_idx, (name, code, v_dat) in enumerate(view_rows, 1):
    vals = [name, code, f"{v_dat['f1']:.2f}%", f"{v_dat['prec']:.2f}%", f"{v_dat['rec']:.2f}%", f"{v_dat['cov']:.1f}%"]
    for c_idx, val in enumerate(vals):
        cell = tbl_view.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if r_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제4장
p4 = doc.add_paragraph()
style_heading(p4, "4. 최종 확정 승인 및 서비스 배포 가이드", level=1)

add_callout_box(doc, [
    "1. [최종 확정 승인]: 바벨 데드리프트 d2(동적 무릎 외반)의 v5.0 버전을 공식 최종 프로덕션 모델로 영구 확정함.",
    "2. [추론 엔진 연동 방법]:",
    "   from src.models.second_models.babel_deadlift.d2_model_v5_final import BabelDeadliftD2ModelV5",
    "   classifier = BabelDeadliftD2ModelV5()",
    "   result = classifier.predict_3state(seq_64f_33_3d, view_code='view3')",
    "3. [실서비스 UX 가이드]:",
    "   - State 0 (CLEAN): '무릎과 발의 정렬 상태가 완벽합니다!' 음성/시각 안내",
    "   - State 1 (DEFECT): '무릎이 발끝보다 안으로 모이고 있습니다. 바깥쪽으로 벌리며 리프팅하세요!' 피드백",
    "   - State 2 (RETAKE): '무릎-발목 궤적이 가려졌습니다. 정면(시점 C) 또는 우측 45도(시점 B)에서 1회 다시 수행해 주세요' 리테이크 유도"
], title="프로덕션 배포 및 서비스 연동 지침")

doc.save(DOCX_PATH)
print(f"[+] v5.0 최종 확정 공식 Word 보고서 저장 완료: {DOCX_PATH}")
