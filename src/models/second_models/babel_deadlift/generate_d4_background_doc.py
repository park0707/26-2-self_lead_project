# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d4_background_doc.py
Purpose: 외부 AI 자문용 기술 백서 및 RFC 배경지식 문서
         - D:\자주프 문서\d4 배경지식.pdf
         - D:\자주프 문서\6주차\d4 배경지식.pdf 및 .docx
         를 자동으로 정밀 생성하고 Word COM을 통해 고품질 PDF로 변환 배포
"""

import os
import sys
import shutil
import subprocess
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT_DOC_DIR = r"D:\자주프 문서"
WEEK6_DOC_DIR = r"D:\자주프 문서\6주차"
os.makedirs(ROOT_DOC_DIR, exist_ok=True)
os.makedirs(WEEK6_DOC_DIR, exist_ok=True)

OUT_DOCX_WEEK6 = os.path.join(WEEK6_DOC_DIR, "d4 배경지식.docx")
OUT_PDF_WEEK6 = os.path.join(WEEK6_DOC_DIR, "d4 배경지식.pdf")
OUT_PDF_ROOT = os.path.join(ROOT_DOC_DIR, "d4 배경지식.pdf")


def set_cell_background(cell, fill_hex):
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


def add_header_box(doc, title, subtitle):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "1E3A8A")  # Deep Navy Blue
    set_cell_margins(cell, top=200, bottom=200, left=200, right=200)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(17)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)

    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(10)
    run2.font.color.rgb = RGBColor(224, 231, 255)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)


def add_callout(doc, text, title="핵심 요약 / KEY TAKEAWAY"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")  # Light Slate Gray
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="1E3A8A"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    r_t = p.add_run(f"📌 {title}\n")
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(10.5)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(30, 58, 138)

    r_b = p.add_run(text)
    r_b.font.name = "맑은 고딕"
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(30, 58, 138)


def add_subheading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(51, 65, 85)


def add_body_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = "맑은 고딕"
        r_b.font.size = Pt(9.5)
        r_b.font.bold = True
        r_b.font.color.rgb = RGBColor(15, 23, 42)
    r_t = p.add_run(text)
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(9.5)
    r_t.font.color.rgb = RGBColor(51, 65, 85)


def style_table(table, col_widths, headers, data, alignments=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].width = Inches(col_widths[i])
        set_cell_background(hdr_cells[i], "1E3A8A")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_col = "FFFFFF" if row_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = Inches(col_widths[c_idx])
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=90, right=90)
            p = row_cells[c_idx].paragraphs[0]
            if alignments and c_idx < len(alignments):
                p.alignment = alignments[c_idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(30, 41, 59)


def convert_docx_to_pdf(docx_path, pdf_path):
    print(f"[*] Word COM을 활용한 PDF 변환 시도: {docx_path} -> {pdf_path}")
    cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{docx_path}')
        $doc.SaveAs([ref]'{pdf_path}', [ref]17)
        $doc.Close()
        Write-Output "[COM SUCCESS] PDF conversion complete."
    }} catch {{
        Write-Error $_
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_path):
        print(f"[+] PDF 변환 성공: {pdf_path} (크기: {os.path.getsize(pdf_path):,} bytes)")
        return True
    else:
        print(f"[!] Word COM 변환 실패, stderr: {res.stderr}")
        return False


def build_document():
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    add_header_box(
        doc,
        "vidio-health-trainer | Technical RFC Document",
        "바벨 데드리프트 d4 (무릎-골반 동시 펴짐 위반) F1-Score 극대화 자문용 기술 배경지식 명세서\n"
        "AI 피트니스 코칭 시스템 | 외부 최상위 AI (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) 자문용"
    )

    add_callout(
        doc,
        "■ 프로젝트명: vidio-health-trainer (피트니스 영상 기반 관절 결함 다중 진단 시스템)\n"
        "■ 자문 핵심 목적: AI-Hub 770개 멀티뷰 3D 시퀀스(불균형 17.5%)를 바탕으로 구축된 바벨 데드리프트 d4 모델의 "
        "F1-Score(현재 33~40%)를 65%~75%+ 이상으로 끌어올리기 위한 생체역학 피처, 모델 아키텍처, 손실함수, 시계열 융합 솔루션 도출.\n"
        "■ 핵심 제약 조건:\n"
        "  1) 3-State 선택적 판정 시 보류율(Decline Rate)은 최대 10% 이하(유효 판정률 Coverage >= 90%)를 엄격히 준수해야 함.\n"
        "  2) AI-Hub 3D JSON 좌표/라벨은 오직 채점용 타깃(y)으로만 사용하며 입력 피처 투입 영구 금지(Zero Data Leakage).\n"
        "  3) 공식 5대 시점(후방, 우측전방, 정면, 좌측전방, 좌측후방) 전수 감사 통과 필수.",
        title="외부 AI 전문가를 위한 핵심 요약 (EXECUTIVE SUMMARY)"
    )

    # 1. 외부 AI 핵심 자문 질의 목록
    add_section_heading(doc, "1. 외부 AI 핵심 자문 질의 (Core RFC Questions)")
    add_body_p(
        doc,
        "외부 AI 연구자분께서는 본 기술 명세서의 데이터 구조, 피처 정의, 현재 실측 벤치마크 및 병목 원인을 검토하신 후, "
        "아래 5대 핵심 질의에 대해 수학적 수식, 구체적인 파이썬 코드, 논문 레퍼런스를 포함하여 답변해 주시기 바랍니다:"
    )

    q_data = [
        ["Q1", "클래스 불균형 (17.5%) 극복 전략", "정상 635건 vs 결함 135건의 극심한 불균형에서 오탐(FP) 폭증 없이 결함 Recall을 60~70% 이상으로 끌어올릴 최적의 손실함수(Focal Loss, SupCon, Class-balanced GBDT 등) 설계 방안은?"],
        ["Q2", "64프레임 시계열 딥러닝 융합 구조", "GBDT(LightGBM/CatBoost)의 요약 통계량 한계를 보완하기 위해, 64프레임 관절 각도/각속도 시계열을 수용하는 1D-TCN / ST-GCN / Transformer 인코더와 GBDT의 최적 Two-Stream 메타 융합 파이프라인은?"],
        ["Q3", "정면 시점(view3) Z축 깊이 결손 극복", "단안 비전(MediaPipe)의 물리적 한계로 정면에서 전후 방향 무릎-골반 상대 변위가 가려지는 문제를 보완할 시점 불변(View-Invariant) 기하학 메트릭은?"],
        ["Q4", "차세대 생체역학 동역학 피처 제안", "연속 상대 위상(CRP), 수정 벡터 코딩(MVC) 외에 무릎 조기 신전(Stripper Deadlift)을 포착할 추가적인 물리역학 지표(모멘트암, 정강이-대퇴 각가속도 연동 등)는?"],
        ["Q5", "보류율 10% 이내 F1 65%+ 달성 로드맵", "보류율 <= 10%(Coverage >= 90%) 제약 하에서 F1-Score 65%~75%+를 달성하기 위한 구체적인 단계별 하이퍼파라미터 및 임계값 튜닝 알고리즘은?"]
    ]
    tbl_q = doc.add_table(rows=1, cols=3)
    style_table(tbl_q, [0.6, 2.3, 3.6], ["번호", "핵심 자문 주제", "세부 질의 내용"], q_data,
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    # 2. 생체역학적 결함 기전 및 운동학적 정의
    add_section_heading(doc, "2. 바벨 데드리프트 d4 결함의 생체역학적 기전")
    add_body_p(
        doc,
        "바벨 데드리프트는 바닥 정지 상태의 고중량 바벨을 기립 상태로 수직 견인하는 전신 복합 운동입니다. "
        "바닥에서 무릎 높이까지 바벨을 뽑아 올리는 First Pull 구간에서는 대퇴사두근에 의한 슬관절(무릎) 신전과 "
        "둔근·햄스트링에 의한 고관절(골반) 신전이 완벽한 조화를 이루며 상체 전경각(Trunk Lean)을 일정하게 유지해야 합니다.",
        bold_prefix="■ 정상 리프트 동역학 (Synchronous Extension): "
    )
    add_body_p(
        doc,
        "피트니스 현장에서 '스트리퍼 데드리프트(Stripper Deadlift)', '엉덩이 솟구침(Butt-shoot)', 또는 '굿모닝 데드리프트화'로 불리는 치명적 결함입니다. "
        "바벨이 바닥에서 떨어지는 순간 대퇴사두근의 지면 반력 전달이 붕괴되어 무릎이 먼저 다 펴지고 엉덩이가 위로 불쑥 치솟습니다. "
        "그 결과 상체가 지면과 수평(90도 전경)으로 꺾이며, 남은 리프트 구간을 오직 허리(기립근)의 힘만으로 들어올리게 되어 요추 전단력과 추간판 탈출 위험이 극대화됩니다.",
        bold_prefix="■ d4 오류 기전 (Early Knee Extension & Hip Shoot Up): "
    )

    # 3. 데이터셋 규격 및 전처리 표준
    add_section_heading(doc, "3. 데이터셋 규격 및 64프레임 전처리 표준 (Invariants)")
    add_body_p(
        doc,
        "1. 원천 데이터: AI-Hub 피트니스 자세 평가 데이터셋 (02_바벨_데드리프트)\n"
        "2. 피험자 전수 원칙: 피험자 1번부터 9번까지 9명 전원 포함 (남녀, 체형, 유연성 편차 포함)\n"
        "3. 카메라 5대 시점 전수 체계 (Official 5-View Taxonomy):\n"
        "   - 시점 A: 우측 후방 45° (view5)\n"
        "   - 시점 B: 우측 전방 45° (view2)\n"
        "   - 시점 C: 정면 0° (view3)\n"
        "   - 시점 D: 좌측 전방 45° (view4)\n"
        "   - 시점 E: 좌측 후방 45° (view1)\n"
        "4. 전처리 표준: Google MediaPipe Pose 모델을 통해 원천 영상에서 33개 관절의 3D World Keypoints를 프레임별 추출한 뒤, "
        "단일 반복 동작(1 Rep) 구간을 엄격히 분할하여 64프레임으로 3차 스플라인(Cubic Spline) 보간 수행 (텐서 규격: 64 x 33 x 3).\n"
        "5. 데이터 분포: 총 770개 멀티뷰 시퀀스 (정상 635건, 82.5% vs 결함 135건, 17.5% - 극심한 클래스 불균형).\n"
        "6. 데이터 누수 차단 원칙: AI-Hub 3D JSON 좌표는 오직 채점용 타깃 라벨(y)로만 사용하며, 모델의 입력 피처는 100% 순수 비전(MediaPipe) 추출 데이터만을 기반으로 함."
    )

    # 4. 현재 모델 아키텍처 및 37D 피처셋 상세 명세
    add_section_heading(doc, "4. 현재 구축된 모델 아키텍처 및 37D 생체역학 피처셋")
    add_body_p(
        doc,
        "현재 vidio-health-trainer에 구축된 d4 v3.0 모델은 LightGBM(50%) + CatBoost(30%) + Random Forest(20%)의 "
        "3-Way Tri-Ensemble 아키텍처를 기반으로 하며, 총 37차원(37D)의 생체역학 피처셋을 사용합니다:"
    )

    feat_spec = [
        ["Group 1\n(7D)", "신전 동시성 & 각속도비", "d4_knee_hip_corr_full, d4_knee_hip_corr_ascent, d4_vel_ratio_first_half,\n"
                                                  "d4_vel_ratio_second_half, d4_vel_ratio_full_ascent, d4_extension_rate_diff_max, d4_angular_accel_mismatch"],
        ["Group 2\n(6D)", "상체 안정도 & 힙 슈트업", "d4_trunk_angle_nadir_deg, d4_trunk_drop_ascent_max_deg, d4_hip_shoot_up_speed_norm,\n"
                                                  "d4_torso_horizontal_ratio, d4_trunk_angle_std_ascent, d4_hip_shoulder_vert_dist_min"],
        ["Group 3\n(5D)", "락아웃 위상차 & ROM", "d4_knee_lock_time_norm, d4_hip_lock_time_norm, d4_lockout_phase_lag_norm,\n"
                                               "d4_knee_rom_deg, d4_hip_rom_deg"],
        ["Group 4\n(6D)", "5대 시점 불변 & 대칭성", "d4_knee_hip_rom_ratio, d4_bilateral_knee_asym_deg, d4_bilateral_hip_asym_deg,\n"
                                                "d4_ascent_duration_ratio, d4_pelvis_y_rise_rate, d4_view_robust_coupling"],
        ["Group 5\n(5D)", "무릎-골반 각도 비율 동역학", "d4_ratio_delta_first_pull, d4_ratio_deriv_max_ascent, d4_ratio_std_ascent,\n"
                                                "d4_ratio_max_ascent, d4_norm_progress_diff_max"],
        ["Group 6\n(3D)", "연속 상대 위상 (CRP)", "d4_crp_mean_first_pull_deg (First Pull 위상차 평균),\n"
                                                "d4_crp_std_ascent_deg (위상 변동성), d4_crp_max_drift_deg (최대 위상 이탈)"],
        ["Group 7\n(3D)", "수정 벡터 코딩 (MVC)", "d4_mvc_knee_dominant_ratio (무릎 우세 0~22.5도 체류비),\n"
                                                "d4_mvc_co_extension_ratio (동시신전 22.5~67.5도 비), d4_mvc_coupling_angle_mean_deg"],
        ["Group 8\n(2D)", "체형 기하학적 정규화", "d4_morph_femur_tibia_ratio (대퇴/하퇴 길이비),\n"
                                                "d4_morph_torso_thigh_ratio (체간/대퇴 길이비)"]
    ]
    tbl_feat = doc.add_table(rows=1, cols=3)
    style_table(tbl_feat, [1.1, 2.0, 3.4], ["그룹 구분", "핵심 생체역학 명칭", "포함 피처 상세 목록"], feat_spec,
                [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    # 5. 실측 벤치마크 및 병목 원인 정밀 진단
    add_section_heading(doc, "5. 실측 벤치마크 결과 및 핵심 병목(Bottleneck) 분석")
    add_subheading(doc, "5.1. 버전별 피험자 격리 5-Fold 교차검증 성능 추이")

    bench_data = [
        ["v1.0 베이스라인 (24D)", "100.0%", "6.67% (9건)", "64.29%", "12.08%", "82.99%", "0.7034", "결함 거의 미검출"],
        ["v2.0 각도비율 추가 (29D)", "100.0%", "25.19% (34건)", "31.48%", "27.98%", "77.27%", "0.6899", "검출률 3.8배 상승"],
        ["★ v3.0 CRP+MVC+체형 (37D)", "100.0%", "31.11% (42건)", "37.17%", "33.87%", "78.70%", "0.6926", "검출률 4.7배 상승"],
        ["★ v3.0 3-State (보류율 9.7%)", "90.26%", "45.90% (56건)", "35.67%", "40.14%", "75.97%", "-", "F1 40% 돌파 달성"]
    ]
    tbl_b = doc.add_table(rows=1, cols=8)
    style_table(tbl_b, [1.8, 0.7, 0.9, 0.7, 0.7, 0.7, 0.6, 1.4],
                ["모델 버전", "커버리지", "결함 Recall", "정밀도", "F1", "정확도", "AUC", "주요 진단"], bench_data,
                [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                 WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_subheading(doc, "5.2. 공식 5대 시점별 세부 감사 성적표 (Official 5-View Audit - v3.0 3-State 기준)")
    v_data = [
        ["시점 E (좌측 후방 45° / view1)", "7.1%", "92.9%", "65.38%", "41.46%", "50.00%", "77.62%", "최고 F1 50% 달성"],
        ["시점 D (좌측 전방 45° / view4)", "10.4%", "89.6%", "54.17%", "41.94%", "47.27%", "78.99%", "안정적 검출"],
        ["시점 B (우측 전방 45° / view2)", "10.4%", "89.6%", "51.85%", "40.00%", "45.16%", "75.36%", "우측 시점 정상화"],
        ["시점 C (정면 0° / view3)", "12.3%", "87.7%", "32.00%", "30.77%", "31.37%", "74.07%", "Z축 깊이 결손 한계"],
        ["시점 A (우측 후방 45° / view5)", "8.4%", "91.6%", "25.00%", "18.52%", "21.28%", "73.76%", "가장 낮은 검출률"]
    ]
    tbl_v = doc.add_table(rows=1, cols=8)
    style_table(tbl_v, [2.0, 0.6, 0.7, 0.8, 0.7, 0.7, 0.7, 1.3],
                ["카메라 시점", "보류율", "커버리지", "Recall", "정밀도", "F1", "정확도", "시점별 진단"], v_data,
                [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                 WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_subheading(doc, "5.3. 현재 모델의 3대 핵심 병목(Bottleneck)")
    add_body_p(
        doc,
        "1. [극심한 클래스 불균형(17.5%)에 의한 음성 예측 편향]:\n"
        "   - 정상 635건 대비 결함이 135건에 불과하여, 모델이 조금만 애매해도 '정상'으로 판정하려는 경향이 강함.\n"
        "   - 비용 민감(Cost-Sensitive) 가중치를 주면 Recall은 올라가나 Precision이 급락하여 F1이 40% 부근에서 정체됨.\n\n"
        "2. [단안 2D 비전의 깊이(Z축) 결손 및 정면 시점 왜곡]:\n"
        "   - 시점 E, D, B 등 측면 각도에서는 무릎-골반 상대 변위가 잘 보여 F1 45~50%에 도달하나,\n"
        "   - 시점 C(완전 정면 0°)에서는 깊이 방향(Z축)으로 일어나는 무릎 전후 이동과 상체 숙임이 심하게 왜곡되어 F1 31%로 저조함.\n\n"
        "3. [GBDT의 시계열 순서성(Temporal Flow) 정보 손실]:\n"
        "   - 현재 피처셋은 평균, 최대치, 표준편차 등 수동 추출 요약 통계량에 의존함.\n"
        "   - 64프레임 동안 일어나는 '무릎 신전 시작 시점과 골반 신전 시작 시점 간의 미세한 위상 차이(1~3프레임)'를 "
        "딥러닝 시계열 인코더(1D-TCN 또는 Transformer)처럼 프레임 단위 연속성으로 포착하지 못함."
    )

    # 6. 외부 AI 연구 자문 가이드라인
    add_section_heading(doc, "6. 외부 AI 연구 자문 가이드라인 (Response Template)")
    add_body_p(
        doc,
        "외부 AI 모델은 상기 배경지식과 병목 분석을 바탕으로 다음 형식에 맞추어 실질적이고 구체적인 해결책을 제시해 주시기 바랍니다:\n\n"
        "1. [진단 평가]: 현재 37D 생체역학 피처셋 및 Tri-Ensemble 구조의 장단점 평가\n"
        "2. [불균형 극복안]: 오탐(FP)을 억제하면서 결함 Recall을 65%+ 이상으로 끌어올릴 손실함수/샘플링 기법\n"
        "3. [시계열 딥러닝 융합 구조]: 64프레임 관절 좌표를 입력받는 경량 1D-TCN / ST-GCN / Transformer와 GBDT의 Two-Stream 메타 융합 파이프라인 (수식 및 파이썬 코드)\n"
        "4. [신규 생체역학 피처]: 정면 시점 Z축 결손을 극복할 수 있는 추가 피처 3~5종 제안\n"
        "5. [예상 성능]: 제안된 솔루션 적용 시 기대 F1-Score 및 검증 전략"
    )

    doc.save(OUT_DOCX_WEEK6)
    print(f"[+] Word 문서 생성 완료: {OUT_DOCX_WEEK6}")

    # Word COM으로 PDF 변환 실행
    success = convert_docx_to_pdf(OUT_DOCX_WEEK6, OUT_PDF_WEEK6)
    if success and os.path.exists(OUT_PDF_WEEK6):
        # D:\자주프 문서\d4 배경지식.pdf 에 복사
        shutil.copy2(OUT_PDF_WEEK6, OUT_PDF_ROOT)
        print(f"[+] 루트 자주프 문서 폴더 배포 완료: {OUT_PDF_ROOT}")
    else:
        print("[!] PDF 변환 실패")


if __name__ == "__main__":
    build_document()
