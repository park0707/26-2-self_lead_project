# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d1_external_ai_doc.py
Purpose: 바벨 데드리프트 d1(척추 중립 위반) 모델의 외부 AI 자문용 기술 백서 및 RFC 문서 (.docx 및 .pdf) 자동 생성
"""

import os
import sys
import subprocess
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_DIR_6 = r"D:\자주프 문서\6주차"
DOCX_DIR_ROOT = r"D:\자주프 문서"
os.makedirs(DOCX_DIR_6, exist_ok=True)

OUT_DOCX_6 = os.path.join(DOCX_DIR_6, "외부ai용 d1.docx")
OUT_PDF_6 = os.path.join(DOCX_DIR_6, "외부ai용 d1.pdf")
OUT_PDF_ROOT = os.path.join(DOCX_DIR_ROOT, "외부ai용 d1.pdf")


def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def add_callout(doc, text, title="핵심 요약 / IMPORTANT"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # 테두리 설정 (좌측 굵은 파란 바)
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="36" w:space="0" w:color="1A5276"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}\n")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(26, 82, 118)

    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(44, 62, 80)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def style_table(tbl, col_widths, headers, data, alignments=None):
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

    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)


def build_document():
    doc = docx.Document()

    # 페이지 마진
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # 1. 문서 헤더 및 메인 타이틀
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_meta = p_meta.add_run("vidio-health-trainer | Technical RFC Document v2.0\n분류 대상: 바벨 데드리프트 d1 (척추 중립 위반)")
    r_meta.font.name = "맑은 고딕"
    r_meta.font.size = Pt(8.5)
    r_meta.font.color.rgb = RGBColor(128, 128, 128)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(4)
    title_p.paragraph_format.space_after = Pt(2)
    title_run = title_p.add_run("바벨 데드리프트 d1(척추 중립 위반) 분류 모델\n아키텍처 명세서 및 고도화 자문 요청서 (RFC)")
    title_run.font.name = "맑은 고딕"
    title_run.font.size = Pt(19)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(26, 82, 118)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_run = sub_p.add_run("AI 기반 멀티뷰 헬스 트레이닝 자세 분석 시스템 | 외부 최상위 AI(GPT-4o, Claude 3.5 Sonnet) 연구 협업용")
    sub_run.font.name = "맑은 고딕"
    sub_run.font.size = Pt(10.5)
    sub_run.font.color.rgb = RGBColor(100, 100, 100)

    # Callout
    add_callout(
        doc,
        "본 문서는 'vidio-health-trainer' 프로젝트의 핵심 모듈인 [바벨 데드리프트 d1: 척추의 중립 위반] 결함 분류 모델의 "
        "구조, 피처 추출 공식, 실측 벤치마크, 한계점을 전수 공개하고, 외부 최신 AI 모델 및 생체역학 연구진에게 "
        "추가적인 성능 향상(F1 71.8% -> 80%+ 달성 및 피험자 일반화)을 위한 구체적 방법론을 자문받기 위해 작성되었습니다.",
        title="문서의 목적 및 협업 요청 개요"
    )

    # 섹션 1. 과제 개요 및 자문 요청 핵심 질문 4선
    h1 = doc.add_heading("1. 과제 개요 및 외부 AI 핵심 자문 항목 (Core RFC Questions)", level=1)
    h1.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    p1 = doc.add_paragraph()
    p1.add_run(
        "바벨 데드리프트 운동 중 가장 부상 위험이 높고 흔하게 발생하는 결함인 'd1: 척추의 중립 위반(Lumbar Flexion / Round Back)'을 "
        "단일 시각 센서(스마트폰/웹캠)와 3D 포즈 추정(MediaPipe) 좌표만으로 실시간 정확히 판정하는 것이 목표입니다. "
        "현재 5-Fold 피험자 완전 격리 교차검증에서 F1 71.77%, 정확도 79.89%(커버리지 90.39%)를 달성했으나, "
        "다음 4가지 핵심 질문에 대해 외부 AI의 심층 분석과 솔루션을 요청합니다:"
    )

    # 질문 표
    q_data = [
        ["Q1. 피험자 일반화\n(Subject Shift)", "특정 피험자(Fold 4: 피험자 5번/8번)에서 F1이 39~42%로 급락하는 도메인 시프트 발생. 피험자별 고유 체형(상체/하체 비율)과 카메라 거리 차이를 완벽히 상쇄할 도메인 불변 정규화 기법은 무엇인가?"],
        ["Q2. 바벨 부재 극복\n생체역학 피처", "현재 2D/3D 바벨 오브젝트 바운딩박스가 없고 인체 33개 관절만 주어짐. 손목-발목 상대 궤적, 힙힌지 신전 동기화 외에 바벨 이륙(Floor Break)과 락아웃을 더 정밀히 포착할 기하학적 공식은?"],
        ["Q3. 시계열 딥러닝\n아키텍처 혁신", "현재 24D Dilated TCN + Attention Pooling(Late Fusion 60%)을 사용 중. ST-GCN(Spatial-Temporal Graph Conv), Mamba(State-Space Model), 혹은 경량 Transformer 중 본 64F 시퀀스에 가장 효과적인 구조는?"],
        ["Q4. 3-State 판정\n신뢰도 캘리브레이션", "현재 최적 임계값 윈도우 [0.28, 0.37]을 휴리스틱 그리드 서치로 도출함. Conformal Prediction이나 몬테카를로 드롭아웃(MC-Dropout)을 적용해 수학적으로 보장된 불확실성 경계를 생성하는 방안은?"]
    ]
    tbl_q = doc.add_table(rows=5, cols=2)
    style_table(tbl_q, [2.0, 5.0], ["자문 영역", "세부 자문 내용 및 해결 희망 과제"], q_data, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 섹션 2. 데이터셋 및 전처리 파이프라인
    h2 = doc.add_heading("2. 데이터셋 규격 및 전처리 표준 파이프라인", level=1)
    h2.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    ds_info = [
        ["데이터셋 출처", "AI-Hub 헬스 피트니스 영상 데이터셋 (02_바벨_데드리프트)"],
        ["피험자 구성", "피험자 1번 ~ 9번 (전수 9명, StratifiedGroupKFold 5-Fold 피험자 완전 격리 평가)"],
        ["시점 (Camera View)", "공식 5대 다각도 시점 전수 반영 (시점 A~E, 각 시점당 77개 시퀀스 균등)\n"
                             "- 시점 A: 우측 후방 45° (view5)\n- 시점 B: 우측 전방 45° (view2)\n- 시점 C: 정면 0° (view3)\n"
                             "- 시점 D: 좌측 전방 45° (view4)\n- 시점 E: 좌측 후방 45° (view1)"],
        ["총 샘플 수", "385개 멀티뷰 시퀀스 (정상: 250건, d1 결함: 135건, 결함 비율 35.1%)"],
        ["입력 텐서 규격", "MediaPipe Pose 33개 관절 (X, Y, Z) x 64 프레임 (3차 스플라인 3D Cubic Spline 보간 완료)"],
        ["골격 확장", "DFKI SpinePose 9분절 가상 척추 모델 (S1, L5, L4, L3, L2, L1, T12, T8, C7) 내부 기하 투영"]
    ]
    tbl_ds = doc.add_table(rows=7, cols=2)
    style_table(tbl_ds, [2.2, 4.8], ["항목", "상세 규격 및 처리 방식"], ds_info, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 섹션 3. 현재 모델 아키텍처 (d1 v2.0)
    h3 = doc.add_heading("3. 현재 d1 v2.0 아키텍처 및 추론 파이프라인", level=1)
    h3.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    p_arch = doc.add_paragraph()
    p_arch.add_run(
        "d1 v2.0은 스쿼트 c1 모델에서 확인된 최하사점(Nadir) 윈도우 노이즈 12개를 영구 소거하고, "
        "데드리프트 고유 생체역학 피처 5개를 신규 주입한 [28D GBDT + 24D Dilated TCN]의 3-Way Tri-Ensemble 하이브리드 구조입니다."
    )

    arch_rows = [
        ["Branch 1: LightGBM", "28D 정제 피처", "n_estimators=160, lr=0.03, depth=4, num_leaves=15, colsample=0.85", "결함 확률 p_lgb 산출"],
        ["Branch 2: CatBoost", "28D 정제 피처", "iterations=160, lr=0.03, depth=4, colsample=0.85", "결함 확률 p_cat 산출"],
        ["Branch 3: Dilated TCN", "24D 시계열 (64, 24)", "In(24)->Proj(48)->ResBlocks(d=1,2,4, k=3)->Multi-Head Attn Pooling", "결함 확률 p_tcn 산출"],
        ["Late Fusion Blending", "앙상블 결합", "P_final = 0.40 * (0.5 * p_lgb + 0.5 * p_cat) + 0.60 * p_tcn", "최적 판정 경계: 0.35"],
        ["3-State Selective Gate", "안전 게이트", "NORMAL: p <= 0.28 | DEFECT: p >= 0.37 | UNDETERMINED: 0.28 < p < 0.37", "보류율 9.61%, 커버리지 90.39%"]
    ]
    tbl_arch = doc.add_table(rows=6, cols=4)
    style_table(tbl_arch, [1.8, 1.3, 2.5, 1.4], ["모듈명", "입력 데이터", "하이퍼파라미터 / 아키텍처", "출력 / 역할"], arch_rows, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 섹션 4. 피처 엔지니어링 전수 명세서
    h4 = doc.add_heading("4. 피처 엔지니어링 전수 명세서 (28D Tabular + 24D Temporal)", level=1)
    h4.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    p_feat = doc.add_paragraph()
    p_feat.add_run("d1 v2.0 모델에서 사용 중인 28차원 정제 타뷸러 피처와 24차원 시계열 텐서의 수학적 정의입니다:")

    doc.add_heading("4.1. 데드리프트 28D 정제 타뷸러 피처셋", level=2)
    feat_rows = [
        ["1", "d1_lumbar_curvature_poly_a", "요추 5개 분절의 체간 시상면(Y-Z) 2차 다항식 곡률 계수 (a). 라운드백 시 양수 급증"],
        ["2", "d1_lumbar_stiffness_sd", "전 64프레임 동안 요추 Cobb 각도의 표준편차 (허리 안정성 및 떨림 감지)"],
        ["3", "d1_spine_lateral_deviation_max", "척추 분절들의 좌우 횡방향(X축) 이탈 최대치 (측만 및 비틀림 감지)"],
        ["4", "d1_trunk_flex_speed_max", "상체 체간 벡터의 각속도 최대 피크 (리프팅 시 허리 급격 굽힘 감지)"],
        ["5", "d1_cervical_min", "경추-상부 흉추 최소 각도 (시선 전방 미유지 및 목 꺾임 감지)"],
        ["6", "d1_signed_thoracolumbar_delta", "흉추와 요추 사이의 부호 있는 상대 각도 최대 변화폭"],
        ["7", "d1_active_rom", "골반 중심의 수직(Y축) 총 이동 변위 (능동 리프팅 높이, 유효 동작 판정)"],
        ["8", "d1_phase_speed_ratio", "상승 리프팅 구간 평균 속도 대비 하강 하강 구간 평균 속도 비"],
        ["9", "d1_hip_to_knee_dist_ratio", "최하단 셋업 시 고관절-무릎 수평 거리 대 대퇴 길이 비 (스쿼트성 주저앉음 감지)"],
        ["10", "d1_cobb_mean_active", "바벨 리프팅 능동 구간(골반 이동 10%~90%) 요추 Cobb 각도 평균"],
        ["11", "d1_cobb_max_flexion", "전 동작 중 요추의 최대 굴곡 각도 (도, deg)"],
        ["12", "d1_cobb_range_active", "능동 리프팅 구간 요추 Cobb 각도 진폭 (Max - Min)"],
        ["13", "d1_shoulder_hip_twist_max", "어깨 축 벡터와 골반 축 벡터 간의 3D 비틀림 각도 최대치"],
        ["14", "d1_lateral_asymmetry_integral", "좌우 척추 비대칭 변위의 전 구간 시간 적분값"],
        ["15", "d1_pelvis_tilt_range", "골반 전방경사(Anterior Tilt) 및 후방경사(Posterior Tilt) 가동범위"],
        ["16", "d1_lumbar_flex_acceleration_peak", "요추 굴곡 각가속도 절대 피크값 (요추 충격 하중 감지)"],
        ["17", "d1_torso_pitch_standing", "최상단 완전 기립(Lockout) 상태에서의 체간 전경각"],
        ["18", "d1_torso_pitch_deepest", "최하단 바벨 셋업 자세에서의 체간 전경각 (힙힌지 깊이)"],
        ["19", "d1_view_score_normal_z", "체간 법선 벡터의 카메라 광축(Z) 투영값 (시점 A~E 인코딩)"],
        ["20", "d1_keypoint_confidence_mean", "척추 및 골반 핵심 랜드마크의 MediaPipe 가시성(Visibility) 평균"],
        ["21", "d1_lumbar_energy_ratio", "요추 분절들의 각운동 에너지 합 대 전신 관절 운동 에너지 총합 비"],
        ["22", "d1_standing_symmetry_score", "피니시 기립 상태에서의 좌우 대칭 점수 (1.0 - 어깨/골반 기울기 오차)"],
        ["23", "d1_lumbar_shear_stress_proxy", "상체 질량 중심(COM)과 요추 L5 간의 수평 모멘트암 x 상체 전경각 기반 전단력 추정치"],
        ["24★", "d1_setup_lumbar_poly_a", "[신규] 바벨 이륙 직전 셋업(Frame 0~8) 구간의 요추 2차 곡률 (시작부터 말린 요추 포착)"],
        ["25★", "d1_floor_break_cobb_delta", "[신규] 바벨이 바닥에서 떨어지는 순간(Floor Break) 요추 Cobb 각도의 급격한 변화량"],
        ["26★", "d1_knee_pass_lumbar_flexion", "[신규] 바벨이 무릎을 통과하는 순간(골반 높이 40~60%)의 요추 굴곡각 유지도"],
        ["27★", "d1_lockout_hyperextension", "[신규] 리프팅 마무리 락아웃 시 상체 뒤로 젖힘(과신전) 각도 오프셋"],
        ["28★", "d1_spine_femur_sync_ratio", "[신규] 상체 신전 속도 대 대퇴 신전 속도 비율 (엉덩이만 먼저 들리는 똥개 데드리프트 감지)"]
    ]
    tbl_f = doc.add_table(rows=29, cols=3)
    style_table(tbl_f, [0.6, 2.5, 3.9], ["No", "피처 명칭", "수학적 정의 및 생체역학적 의도"], feat_rows, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_heading("4.2. TCN 입력용 24D 시계열 텐서 구성 (64 프레임 x 24 채널)", level=2)
    tcn_info = [
        ["1~12 (12D)", "척추 6개 핵심 분절 체간 로컬 좌표", "S1, L5, L3, L1, T8, C7 분절의 체간 로컬 y(상하), z(전후) 좌표"],
        ["13~18 (6D)", "주요 관절 체간 로컬 좌표", "골반 중심, 어깨 중심, 무릎 중심의 체간 로컬 y, z 좌표"],
        ["19~24 (6D)", "6대 생체역학 동적 각도 시계열", "(1) 요추 Cobb 각도, (2) 흉추 후만 각도, (3) 체간 전경각,\n"
                                                      "(4) 고관절 신전각, (5) 무릎 신전각, (6) 골반 경사각"]
    ]
    tbl_tcn = doc.add_table(rows=4, cols=3)
    style_table(tbl_tcn, [1.5, 2.3, 3.2], ["채널 인덱스", "구성 요소", "세부 설명 및 정규화 방식"], tcn_info, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 섹션 5. 실측 벤치마크 및 병목 분석
    h5 = doc.add_heading("5. 실측 벤치마크 결과 및 병목(Bottleneck) 정밀 진단", level=1)
    h5.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    doc.add_heading("5.1. 5-Fold 피험자 격리 성능 비교", level=2)
    perf_rows = [
        ["스쿼트 c1 제로샷 투입", "100.0%", "52.73%", "35.81%", "100.0%", "37.14%", "71.04%", "242건 폭증 (굿모닝 오진)"],
        ["d1 v1.0 (스쿼트 구조 단순 재학습)", "100.0%", "63.94%", "64.18%", "63.70%", "74.81%", "79.42%", "48건 (-80.2% 감소)"],
        ["d1 v2.0 (강제 2진 판정 th=0.35)", "100.0%", "67.41%", "67.41%", "67.41%", "77.14%", "80.91%", "44건 (안정적 베이스라인)"],
        ["★ d1 v2.0 3-State [0.28, 0.37]", "90.39%", "71.77% 🔺", "69.53%", "74.17%", "79.89% 🔺", "82.45%", "39건 (보류 37건, 9.61%)"]
    ]
    tbl_p = doc.add_table(rows=5, cols=8)
    style_table(tbl_p, [1.8, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7, 1.4], ["모델 버전 / 판정 방식", "커버리지", "F1", "Prec", "Rec", "Acc", "AUC", "정상 오진(FP)"], perf_rows, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    doc.add_heading("5.2. 카메라 5대 시점별 세부 감사 (Multi-View Audit)", level=2)
    v_rows = [
        ["시점 A (우측 후방 45° / view5)", "77건", "50 / 27", "65.57%", "58.82%", "74.07%", "76.74%"],
        ["시점 B (우측 전방 45° / view2)", "77건", "50 / 27", "65.52%", "61.29%", "70.37%", "78.96%"],
        ["시점 C (정면 0° / view3)", "77건", "50 / 27", "67.86%", "65.52%", "70.37%", "83.48%"],
        ["시점 D (좌측 전방 45° / view4)", "77건", "50 / 27", "70.83%", "80.95%", "62.96%", "84.37%"],
        ["시점 E (좌측 후방 45° / view1)", "77건", "50 / 27", "68.09%", "80.00%", "59.26%", "84.22%"]
    ]
    tbl_v = doc.add_table(rows=6, cols=7)
    style_table(tbl_v, [2.3, 0.7, 0.9, 0.8, 0.8, 0.8, 0.7], ["카메라 시점 (공식 명칭)", "샘플수", "정상/결함", "F1", "Prec", "Rec", "AUC"], v_rows, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_heading("5.3. 핵심 병목 및 실패 케이스 정밀 분석 (Why F1 is stuck at 71.8%)", level=2)
    p_fail = doc.add_paragraph()
    p_fail.add_run(
        "현재 성능이 71.8%에서 추가 도약하지 못하는 3대 물리적/모델적 병목 원인은 다음과 같습니다:\n\n"
        "1. [피험자 간 신체 골격비 편차 (Subject Domain Shift)]:\n"
        "   - Fold 5(피험자 7, 9번)는 F1 82.2%로 매우 우수하나, Fold 4(피험자 5, 8번)는 F1 39~42%로 급락함.\n"
        "   - 원인: 피험자 5번과 8번은 팔이 길고 대퇴가 짧아 셋업 시 상체 숙임 각도가 다른 피험자보다 15도 이상 얕음. "
        "현재 정규화가 체간 길이(Torso Length)에만 의존하여 사지 비율(Arm-to-Leg Ratio) 차이를 흡수하지 못함.\n\n"
        "2. [바벨 물리 위치의 간접 추정 오차 (Implicit Barbell Tracking Noise)]:\n"
        "   - 손목(Wrist) 키포인트를 바벨 위치로 대리 사용하고 있으나, 데드리프트 중 그립 방식(오버핸드/얼터네이트)과 "
        "팔꿈치 미세 굴곡에 따라 실제 바벨 바닥 이륙(Floor Break) 시점과 손목 가속 시점에 2~4프레임 위상차가 발생함.\n\n"
        "3. [후방 시점(시점 A, E)에서의 척추 전후 깊이(Z축) 노이즈]:\n"
        "   - MediaPipe는 단안 2D RGB에서 3D Z를 예측하므로 후방 45도 뷰에서 옷 주름, 광배근 수축 등에 의해 요추 깊이(Z)가 튀는 현상 존재."
    )

    # 섹션 6. 외부 AI 연구 자문 가이드라인
    h6 = doc.add_heading("6. 외부 AI 연구 자문 가이드라인 (Format for Answers)", level=1)
    h6.runs[0].font.color.rgb = RGBColor(26, 82, 118)

    p_guide = doc.add_paragraph()
    p_guide.add_run(
        "외부 AI 모델은 위 분석 내용을 검토한 뒤, 다음 양식에 맞추어 구체적인 코드, 수학 공식, 논문 기반 해결책을 제시해 주시기 바랍니다:\n\n"
        "■ 답변 권장 양식:\n"
        "1. [진단 평가]: 현재 28D 피처셋 및 TCN 구조의 생체역학적 타당성 및 취약점 평가\n"
        "2. [Q1 솔루션]: 사지 비율(Arm-to-Leg) 불변 체간 정규화 공식 및 코드 (Python)\n"
        "3. [Q2 솔루션]: 바벨 이륙 및 락아웃을 더 정교하게 포착할 신규 피처 3~5개 (수식 포함)\n"
        "4. [Q3 솔루션]: 경량 ST-GCN 또는 Spatial-Temporal Attention Transformer 구현 구조 제안\n"
        "5. [Q4 솔루션]: 3-State 판정을 위한 Conformal Prediction 적용 알고리즘\n"
        "6. [기대 효과]: 제안된 기법 적용 시 예상 F1-Score 향상 폭 및 검증 전략"
    )

    doc.save(OUT_DOCX_6)
    print(f"[+] DOCX 생성 완료: {OUT_DOCX_6}")


if __name__ == "__main__":
    build_document()
