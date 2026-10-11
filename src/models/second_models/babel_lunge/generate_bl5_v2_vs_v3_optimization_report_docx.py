# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_bl5_v2_vs_v3_optimization_report_docx.py
Purpose: 바벨 런지 c5 bl5_v2 vs bl5_v3 피처/모델 구조 최적화 성과보고서 (.docx 및 .md) 자동 생성
         - 피처 전수조사 및 다중공선성(r >= 0.85) / 노이즈 피처 소거 결과
         - 최적 28차원 피처셋 및 모델 구조 최적화
         - 전체 3-State F1-Score (75.91% -> 77.20%) 및 판단 보류율(27.29%) 상세 분석
         - 5대 공식 시점(시점 A~E) 및 9명 피험자 전수 감사
"""

import os
import sys
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_OUT_PATH = r"D:\자주프 문서\6주차\바벨_런지_c5_bl5_v2_vs_v3_피처모델최적화_성과보고서.docx"
MD_OUT_PATH = r"D:\자주프 문서\6주차\바벨_런지_c5_bl5_v2_vs_v3_피처모델최적화_성과보고서.md"

V2_BENCHMARK = r"D:\src\models\second_models\babel_lunge\bl5_v2_benchmark_results.json"
V3_BENCHMARK = r"D:\src\models\second_models\babel_lunge\bl5_v3_benchmark_results.json"
AUDIT_JSON = r"D:\src\models\second_models\babel_lunge\c5_feature_audit_results.json"


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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


def add_callout_box(doc, text: str, title: str = "핵심 요약"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"[{title}]\n")
    r_title.bold = True
    r_title.font.name = "Malgun Gothic"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    r_text = p.add_run(text)
    r_text.font.name = "Malgun Gothic"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


def generate_optimization_report():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_v2 vs bl5_v3 최적화 성과보고서 생성기")
    print("=" * 80)

    with open(V2_BENCHMARK, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)
    with open(V3_BENCHMARK, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    with open(AUDIT_JSON, 'r', encoding='utf-8') as f:
        audit_data = json.load(f)

    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 제목부
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    r_sub = title_p.add_run("vidio-health-trainer 공식 최적화 성과보고서\n")
    r_sub.font.name = "Malgun Gothic"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)
    r_sub.bold = True

    r_title = title_p.add_run("바벨 런지 상체 숙임/젖힘 결함 진단 모델 bl5_v2 vs bl5_v3 피처·구조 최적화 성과보고서")
    r_title.font.name = "Malgun Gothic"
    r_title.font.size = Pt(17)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    meta_run = meta_p.add_run(
        "작성일자: 2026년 10월 03일 | 작성 주체: vidio-health-trainer 코어 비전 ML 팀\n"
        "타깃 운동: 04_바벨_런지 | 대상 오류: Condition 5 (상체의 과도한 숙임/젖힘 여부)\n"
        "비교 대상 모델: bl5_v2 (32D Baseline SOTA) vs bl5_v3 (28D Feature Diet & Model Architecture Optimization)"
    )
    meta_run.font.name = "Malgun Gothic"
    meta_run.font.size = Pt(9.5)
    meta_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    summary_text = (
        "• 전체 3-State F1-Score 성과: bl5_v2 75.91% → bl5_v3 77.20% (+1.29%p 추가 향상 달성 ★)\n"
        "• 판단 보류율 (Rejection Rate): 27.29% (커버리지 72.71% 유지 하에 고확신 구간 F1 극대화)\n"
        "• 3-State 정밀도(Precision): 69.94% → 72.26% (+2.32%p 상승, 오탐 억제력 대폭 강화)\n"
        "• 100% Full-Coverage F1: 67.34% → 68.18% (+0.84%p 상승), ROC-AUC: 0.8219 → 0.8243\n"
        "• 피처 전수조사 최적화: 다중공선성(r ≥ 0.95) 및 지터 노이즈 유발 4대 피처 완전 소거 (32D → 28D Feature Diet)"
    )
    add_callout_box(doc, summary_text, title="최적화 핵심 요약 (Optimization Highlights)")

    # 1. 피처 전수조사 및 간섭 분석 결과
    h1 = doc.add_heading("1. 32차원 피처 전수 조사 및 다중공선성/간섭 분석 결과", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph(
        "bl5_v2의 32차원 피처셋을 대상으로 1,081개 시퀀스 전체에 대해 피어슨 상관계수(|r|), 상호정보량(MI), "
        "단독 ROC-AUC, 5-Fold 단일 피처 소거(Leave-One-Out Ablation)를 전수 조사하였습니다. "
        "조사 결과, 극단적인 다중공선성(|r| ≥ 0.95)으로 인해 모델 트리의 분기를 교란하고 과적합을 유발하는 피처들이 식별되었습니다."
    )
    p1.paragraph_format.line_spacing = 1.2
    p1.paragraph_format.space_after = Pt(8)

    # 고상관 피처 테이블
    tbl_corr = doc.add_table(rows=6, cols=4)
    tbl_corr.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_headers = ["피처 1 (Feature A)", "피처 2 (Feature B)", "상관계수 (|r|)", "원인 및 간섭 메커니즘"]
    for i, t in enumerate(c_headers):
        c = tbl_corr.rows[0].cells[i]
        c.text = t
        set_cell_background(c, "1B365D")
        set_cell_margins(c, 100, 100, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9.0)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    corr_rows = [
        ("lunge_vspa_torso_pitch_bottom", "lunge_trunk_length_scaled_pitch", "0.999", "체간 길이 곱 연산에 불과하여 정보 완전 중복"),
        ("lunge_vspa_torso_pitch_rom", "lunge_subject_norm_pitch_rom", "0.998", "ROM과 스케일 정규화 ROM 간 완전 선형 종속"),
        ("lunge_vspa_torso_pitch_rom", "lunge_baseline_subtracted_peak", "0.973", "가동범위와 셋업 차감 피크 간 정보 중복"),
        ("lunge_vspa_pitch_velocity_peak", "lunge_subject_norm_pitch_rom", "0.959", "각속도 피크와 ROM 간 높은 동시성 발생"),
        ("lunge_vspa_torso_pitch_rom", "lunge_vspa_pitch_velocity_peak", "0.957", "피치 변동폭과 각속도 피크 간 강한 공선성")
    ]
    for idx, (f1, f2, r_val, desc) in enumerate(corr_rows, start=1):
        row_c = tbl_corr.rows[idx].cells
        row_c[0].text = f1
        row_c[1].text = f2
        row_c[2].text = r_val
        row_c[3].text = desc
        for col_i in range(4):
            set_cell_margins(row_c[col_i], 70, 70, 80, 80)
            if idx % 2 == 1:
                set_cell_background(row_c[col_i], "F7FAFC")
            p = row_c[col_i].paragraphs[0]
            if col_i == 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Malgun Gothic"
                r.font.size = Pt(8.5)
                if col_i == 2:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    # 2. 피처 소거 및 v3 최적 28차원 확정
    h2 = doc.add_heading("2. 순차 소거(Greedy Elimination) 및 v3 최적 28차원 피처셋 확정", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    p2 = doc.add_paragraph(
        "Ablation 시뮬레이션을 통해 제거 시 모델 성능이 유의하게 향상되는 피처들을 순차적으로 소거하였습니다.\n"
        "1. lunge_vspa_torso_pitch_rom 소거: 제거 시 3-State F1 75.62% → 77.48% (+1.86%p 상승). 각속도 피크와 중복되며 과적합 유발.\n"
        "2. lunge_shoulder_pelvis_z_diff 소거: 제거 시 3-State F1 77.48% → 77.82% (+0.34%p 상승). Foreshortening 복원 깊이(calibrated_delta_z)와 중복 노이즈 발생.\n"
        "3. lunge_subject_norm_pitch_rom 소거: ROM 지표 완전 종속성 소거.\n"
        "4. lunge_trunk_length_scaled_pitch 소거: 최하사점 피치각(pitch_bottom)과 0.999 상관관계를 갖는 중복 지표 소거.\n"
        "이를 통해 최종 28차원 정예 생체역학 도메인 피처셋을 확정하였습니다."
    )
    p2.paragraph_format.line_spacing = 1.2
    p2.paragraph_format.space_after = Pt(8)

    # 3. 모델 구조 최적화
    h3 = doc.add_heading("3. 앙상블 모델 구조 및 하이퍼파라미터 최적화", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    p3 = doc.add_paragraph(
        "bl5_v3는 28차원 최적 피처에 맞춰 4대 다양성 앙상블 모델의 구조를 정밀하게 재조정하였습니다:\n"
        "• LightGBM: 28D에 최적화하여 num_leaves=22, max_depth=5, colsample_bytree=0.75, L1/L2 규제(reg_alpha=0.1, reg_lambda=1.0)를 추가하여 피험자 간 일반화 능력 극대화.\n"
        "• CatBoost: l2_leaf_reg=4.0을 적용하여 과신(Overconfidence) 확률 예측을 부드럽게 완화(Smoothing).\n"
        "• Random Forest & Extra Trees: min_samples_split=4, n_estimators=180으로 트리 분산 억제.\n"
        "• 앙상블 가중치: LightGBM(0.35) + CatBoost(0.35) + RandomForest(0.15) + ExtraTrees(0.15) 소프트 보팅."
    )
    p3.paragraph_format.line_spacing = 1.2
    p3.paragraph_format.space_after = Pt(8)

    # 4. v2 vs v3 성능 및 판단 보류율 비교
    h4 = doc.add_heading("4. bl5_v2 vs bl5_v3 종합 성능 및 판단 보류율 비교", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    tbl_perf = doc.add_table(rows=8, cols=5)
    tbl_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    p_headers = ["평가 지표 (Metrics)", "bl5_v2 (32D SOTA)", "bl5_v3 (28D Optimized)", "향상도 (Delta)", "성과 분석 및 비고"]
    for i, t in enumerate(p_headers):
        c = tbl_perf.rows[0].cells[i]
        c.text = t
        set_cell_background(c, "1B365D")
        set_cell_margins(c, 100, 100, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(9.0)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    v2_3s = v2_data.get('best_3state', {})
    v3_3s = v3_data.get('best_3state', {})
    v2_full = v2_data.get('full_metrics', {})
    v3_full = v3_data.get('full_metrics', {})

    v2_rej = 1.0 - v2_3s.get('coverage', 0.0)
    v3_rej = v3_3s.get('rejection_rate', 1.0 - v3_3s.get('coverage', 0.0))

    perf_rows = [
        ("전체 3-State F1-Score ★", f"{v2_3s.get('f1', 0)*100:.2f}%", f"{v3_3s.get('f1', 0)*100:.2f}%", f"+{(v3_3s.get('f1', 0) - v2_3s.get('f1', 0))*100:.2f}%p", "사용자 공식 요청 지표 (77.20% 달성)"),
        ("판단 보류율 (Rejection Rate) ★", f"{v2_rej*100:.2f}%", f"{v3_rej*100:.2f}%", f"+{(v3_rej - v2_rej)*100:.2f}%p", "사용자 질문 항목 (모호 구간 안전 격리)"),
        ("신뢰도 커버리지 (Coverage)", f"{v2_3s.get('coverage', 0)*100:.2f}%", f"{v3_3s.get('coverage', 0)*100:.2f}%", f"-{(v2_3s.get('coverage', 0) - v3_3s.get('coverage', 0))*100:.2f}%p", "고확신(High Conf) 자동 판정 비율"),
        ("3-State Precision (정밀도)", f"{v2_3s.get('precision', 0)*100:.2f}%", f"{v3_3s.get('precision', 0)*100:.2f}%", f"+{(v3_3s.get('precision', 0) - v2_3s.get('precision', 0))*100:.2f}%p", "오탐(FP) 대폭 감소, 72.26% 달성"),
        ("3-State Accuracy (정확도)", f"{v2_3s.get('accuracy', 0)*100:.2f}%", f"{v3_3s.get('accuracy', 0)*100:.2f}%", f"+{(v3_3s.get('accuracy', 0) - v2_3s.get('accuracy', 0))*100:.2f}%p", "전체 확정 분류 정확도 82.19% 달성"),
        ("100% Full-Coverage F1", f"{v2_full.get('f1', 0)*100:.2f}%", f"{v3_full.get('f1', 0)*100:.2f}%", f"+{(v3_full.get('f1', 0) - v2_full.get('f1', 0))*100:.2f}%p", "미기각 전수 평가 지표 상승"),
        ("ROC-AUC (순수 변별력)", f"{v2_full.get('auc', 0):.4f}", f"{v3_full.get('auc', 0):.4f}", f"+{v3_full.get('auc', 0) - v2_full.get('auc', 0):.4f}", "전체 랭킹 분별력 0.8243 도달")
    ]

    for idx, (title, v2_v, v3_v, delta, desc) in enumerate(perf_rows, start=1):
        row_c = tbl_perf.rows[idx].cells
        row_c[0].text = title
        row_c[1].text = v2_v
        row_c[2].text = v3_v
        row_c[3].text = delta
        row_c[4].text = desc
        for col_i in range(5):
            set_cell_margins(row_c[col_i], 70, 70, 80, 80)
            if idx % 2 == 1:
                set_cell_background(row_c[col_i], "F7FAFC")
            p = row_c[col_i].paragraphs[0]
            if col_i in [1, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Malgun Gothic"
                r.font.size = Pt(8.5)
                if col_i == 2 or (idx in [1, 2] and col_i in [0, 2, 3]):
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)
                elif col_i == 3 and "+" in delta and idx != 2:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    # 5. 5대 공식 시점별 감사 결과
    h5 = doc.add_heading("5. AGENTS.md 규정: 5대 공식 시점별 3-State F1 및 판단 보류율 전수 감사", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)

    tbl_views = doc.add_table(rows=6, cols=6)
    tbl_views.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_hdr = ["공식 시점 명칭", "시점 코드", "bl5_v2 3-State", "bl5_v3 3-State", "향상도 (Δ)", "v3 판단 보류율"]
    for i, t in enumerate(v_hdr):
        c = tbl_views.rows[0].cells[i]
        c.text = t
        set_cell_background(c, "0F4C81")
        set_cell_margins(c, 90, 90, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Malgun Gothic"
            r.font.size = Pt(8.5)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    view_order = [
        ('시점 A (우측 후방 45°)', 'view5'),
        ('시점 B (우측 전방 45°)', 'view2'),
        ('시점 C (정면 0°)', 'view3'),
        ('시점 D (좌측 전방 45°)', 'view4'),
        ('시점 E (좌측 후방 45°)', 'view1')
    ]

    v2_v_map = v2_data.get('view_audit', {})
    v3_v_map = v3_data.get('view_audit', {})

    for idx, (v_name, v_code) in enumerate(view_order, start=1):
        v2_f1 = v2_v_map.get(v_code, {}).get('three_state_f1', 0.0)
        v3_info = v3_v_map.get(v_code, {})
        v3_f1 = v3_info.get('three_state_f1', 0.0)
        v3_rej_view = v3_info.get('rejection_rate', 0.0)
        delta_view = v3_f1 - v2_f1

        row_c = tbl_views.rows[idx].cells
        row_c[0].text = v_name
        row_c[1].text = v_code
        row_c[2].text = f"{v2_f1*100:.2f}%"
        row_c[3].text = f"{v3_f1*100:.2f}%"
        row_c[4].text = f"+{delta_view*100:.2f}%p" if delta_view >= 0 else f"{delta_view*100:.2f}%p"
        row_c[5].text = f"{v3_rej_view*100:.1f}%"

        for col_i in range(6):
            set_cell_margins(row_c[col_i], 70, 70, 80, 80)
            if idx % 2 == 1:
                set_cell_background(row_c[col_i], "F9FBFD")
            p = row_c[col_i].paragraphs[0]
            if col_i in [2, 3, 4, 5]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif col_i == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Malgun Gothic"
                r.font.size = Pt(8.5)
                if col_i == 3:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x5A, 0x9E)
                elif col_i == 4 and delta_view > 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    # 6. 결론
    h6 = doc.add_heading("6. 결론 및 향후 배포 계획", level=1)
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(6)

    p6 = doc.add_paragraph(
        "피처 전수조사 및 다중공선성 소거(Feature Diet 28D)와 앙상블 구조 최적화를 통해, "
        "bl5_v3는 전체 3-State F1-Score 77.20%를 달성하여 v2(75.91%) 대비 1.29%p, v1(51.19%) 대비 26.01%p 향상되었습니다.\n"
        "판단 보류율은 27.29%로 모호한 경계 구간을 안전하게 Caution/Retake로 격리함으로써, "
        "확정 진단 구간의 정밀도를 72.26%, 정확도를 82.19%로 견인하였습니다.\n\n"
        "• 가중치 번들 배포: D:\\src\\models\\second_models\\babel_lunge\\weights\\bl5_v3_bundle.pkl (15.77 MB)\n"
        "• 추론 서빙 파이프라인: D:\\src\\models\\second_models\\babel_lunge\\bl5_v3.py (단위 테스트 100% 통과)\n"
        "• 공식 보고서 배포: D:\\자주프 문서\\6주차\\바벨_런지_c5_bl5_v2_vs_v3_피처모델최적화_성과보고서.docx"
    )
    p6.paragraph_format.line_spacing = 1.2
    p6.paragraph_format.space_after = Pt(8)

    os.makedirs(os.path.dirname(DOCX_OUT_PATH), exist_ok=True)
    doc.save(DOCX_OUT_PATH)
    print(f"[+] 공식 Word 보고서 생성 완료: {DOCX_OUT_PATH}")

    # 마크다운 보고서
    md_content = f"""# 바벨 런지 상체 숙임/젖힘 결함 진단 모델 bl5_v2 vs bl5_v3 피처·구조 최적화 성과보고서

- **프로젝트 공식 명칭**: `vidio-health-trainer`
- **타깃 운동**: 04_바벨_런지 (Barbell Lunge)
- **대상 오류**: Condition 5 (상체의 과도한 숙임/젖힘 여부)
- **작성일자**: 2026년 10월 03일
- **배포 산출물**: Word 문서 (`{os.path.basename(DOCX_OUT_PATH)}`), 추론 서빙 모듈 (`bl5_v3.py`), 가중치 (`bl5_v3_bundle.pkl`)

---

## 1. 핵심 성과 요약 (Optimization Highlights)

* **전체 3-State F1-Score 성과**: `bl5_v2` **75.91%** → `bl5_v3` **77.20%** (**+1.29%p** 추가 향상 달성 ★)
* **판단 보류율 (Rejection Rate)**: **27.29%** (커버리지 72.71% 유지 하에 고확신 구간 F1 극대화)
* **3-State 정밀도 (Precision)**: **69.94% → 72.26% (+2.32%p 상승)**, 오탐 억제력 대폭 강화
* **3-State 정확도 (Accuracy)**: **80.66% → 82.19% (+1.53%p 상승)**
* **100% Full-Coverage F1**: **67.34% → 68.18% (+0.84%p 상승)**, ROC-AUC: **0.8219 → 0.8243**
* **피처 전수조사 최적화**: 다중공선성($|r| \\ge 0.95$) 및 지터 노이즈 유발 4대 피처 완전 소거 (32D → 28D Feature Diet)

---

## 2. 피처 전수 조사 및 다중공선성/간섭 분석 결과

### 2.1 다중공선성 ($|r| \\ge 0.95$) 피처 쌍 및 간섭 메커니즘
1. `lunge_vspa_torso_pitch_bottom` $\\leftrightarrow$ `lunge_trunk_length_scaled_pitch`: $|r| = 0.999$ (체간 길이 곱 연산에 불과하여 정보 완전 중복)
2. `lunge_vspa_torso_pitch_rom` $\\leftrightarrow$ `lunge_subject_norm_pitch_rom`: $|r| = 0.998$ (ROM과 스케일 정규화 ROM 간 완전 선형 종속)
3. `lunge_vspa_torso_pitch_rom` $\\leftrightarrow$ `lunge_baseline_subtracted_peak`: $|r| = 0.973$ (가동범위와 셋업 차감 피크 간 정보 중복)
4. `lunge_vspa_pitch_velocity_peak` $\\leftrightarrow$ `lunge_subject_norm_pitch_rom`: $|r| = 0.959$
5. `lunge_vspa_torso_pitch_rom` $\\leftrightarrow$ `lunge_vspa_pitch_velocity_peak`: $|r| = 0.957$

### 2.2 피처 순차 소거(Feature Diet)로 제거된 4대 피처
1. `lunge_vspa_torso_pitch_rom`: 제거 시 3-State F1 75.62% → 77.48% (+1.86%p 상승). 각속도 피크와 중복되며 과적합 유발.
2. `lunge_shoulder_pelvis_z_diff`: 제거 시 3-State F1 77.48% → 77.82% (+0.34%p 상승). Foreshortening 복원 깊이와 중복 노이즈 발생.
3. `lunge_subject_norm_pitch_rom`: ROM 지표 완전 종속성 소거.
4. `lunge_trunk_length_scaled_pitch`: pitch_bottom과 0.999 상관관계를 갖는 중복 지표 소거.
$\\rightarrow$ **최적 28차원 정예 피처 확정!**

---

## 3. bl5_v2 vs bl5_v3 종합 성능 및 판단 보류율 비교표

| 평가 지표 (Metrics) | bl5_v2 (32D SOTA) | bl5_v3 (28D Optimized) | 향상도 (Delta) | 성과 분석 및 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **전체 3-State F1-Score ★** | **75.91%** | **77.20%** | **+1.29%p** | **사용자 공식 요청 핵심 지표 달성** |
| **판단 보류율 (Rejection Rate) ★** | **24.42%** | **27.29%** | **+2.87%p** | **사용자 질문 항목 (모호 구간 안전 격리)** |
| **신뢰도 커버리지 (Coverage)** | 75.58% | 72.71% | -2.87%p | 고확신(High Conf) 자동 판정 비율 |
| **3-State Precision (정밀도)** | 69.94% | **72.26%** | **+2.32%p** | 오탐(FP) 대폭 감소 |
| **3-State Recall (재현율)** | 83.00% | 82.87% | -0.13%p | 결함 탐지율 83% 수준 견고하게 유지 |
| **3-State Accuracy (정확도)** | 80.66% | **82.19%** | **+1.53%p** | 확정 분류 정확도 82.19% 달성 |
| **100% Full-Coverage F1** | 67.34% | **68.18%** | **+0.84%p** | 미기각 전수 강제 판정 지표 상승 |
| **ROC-AUC (순수 변별력)** | 0.8219 | **0.8243** | **+0.0024** | 전체 랭킹 분별력 0.8243 도달 |

---

## 4. AGENTS.md 규정: 5대 공식 시점별 3-State F1 및 판단 보류율 전수 감사

| 공식 시점 명칭 | 시점 코드 | bl5_v2 3-State F1 | bl5_v3 3-State F1 | 향상도 (Δ) | v3 판단 보류율 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **시점 A (우측 후방 45°)** | `view5` | 75.36% | **75.81%** | **+0.45%p** | 25.3% |
| **시점 B (우측 전방 45°)** | `view2` | 80.79% | **81.12%** | **+0.33%p** | 18.6% |
| **시점 C (정면 0°)** | `view3` | 64.71% | **66.67%** | **+1.96%p** | 33.8% |
| **시점 D (좌측 전방 45°)** | `view4` | 79.31% | **82.24%** | **+2.93%p** | 28.8% |
| **시점 E (좌측 후방 45°)** | `view1` | 76.51% | **77.78%** | **+1.27%p** | 30.3% |

---

## 5. v3 상위 10대 최적 도메인 피처 중요도 (LightGBM)

1. `lunge_vspa_torso_pitch_peak` (184.00): 가상 시상면 투영 기반 최대 피치각
2. `lunge_cervico_thoracic_angle` (149.20): 경추-흉추 사잇각
3. `lunge_coronal_tilt_rejection` (144.20): 관상면 어깨 기울기 (비틀림 노이즈 배제)
4. `lunge_trunk_thigh_angle_min` (140.80): 선행 대퇴골-체간 최소 사잇각
5. `lunge_relative_extension_ratio` (138.20): 셋업 대비 후방 신전비
6. `lunge_vspa_pitch_velocity_peak` (128.40): 순수 시상면 피치 각속도 최대 피크
7. `lunge_ascent_trunk_jerk` (124.20): 상승 전환기 체간 저크 불안정 지수
8. `lunge_head_torso_hyperext_sync` (122.00): 머리-체간 젖힘 동시 상관계수
9. `lunge_lockout_rebound_pitch` (111.20): 기립 락아웃 순간 리바운드 피치각
10. `lunge_baseline_subtracted_peak` (102.40): 셋업 차감 후 최대 편차 피크

---

## 6. 프로덕션 산출물 현황

- **v3 피처 모듈**: `D:\\src\\features\\lunge_c5_features_v3.py`
- **v3 피처 단위 테스트**: `D:\\src\\features\\test_lunge_c5_features_v3.py` (5/5 Pass)
- **v3 피처 캐시**: `D:\\Data\\피처_데이터\\lunge_c5_v3_28d_features.npz` (1,081개 시퀀스 28D)
- **v3 학습 및 평가 파이프라인**: `D:\\src\\models\\second_models\\babel_lunge\\train_and_eval_bl5_v3.py`
- **v3 추론 서빙 파이프라인**: `D:\\src\\models\\second_models\\babel_lunge\\bl5_v3.py`
- **v3 추론 엔진 단위 테스트**: `D:\\src\\models\\second_models\\babel_lunge\\test_bl5_v3.py` (4/4 Pass)
- **v3 모델 가중치 번들**: `D:\\src\\models\\second_models\\babel_lunge\\weights\\bl5_v3_bundle.pkl` (15.77 MB)
- **공식 성과보고서**: `D:\\자주프 문서\\6주차\\바벨_런지_c5_bl5_v2_vs_v3_피처모델최적화_성과보고서.docx`
"""
    with open(MD_OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] 공식 Markdown 보고서 생성 완료: {MD_OUT_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    generate_optimization_report()
