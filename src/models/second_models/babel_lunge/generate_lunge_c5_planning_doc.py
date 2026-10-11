# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_lunge_c5_planning_doc.py
Purpose: 바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부(lunge_c5)' 세부 오류모델 구축 기획서 생성
         - 스쿼트(c1~c4), 데드리프트(d1~d5), 스티프 데드리프트(sd1~sd3 v4) 및 런지 c1, c4의
           모든 선행 아키텍처와 베스트 프랙티스를 100% 계승
         - Word(.docx) 및 Markdown(.md) 동시 생성
Target Paths:
  - D:\자주프 문서\6주차\바벨_런지_c5_상체숙임젖힘_세부오류모델_구축기획서.docx
  - D:\자주프 문서\6주차\바벨_런지_c5_상체숙임젖힘_세부오류모델_구축기획서.md
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_런지_c5_상체숙임젖힘_세부오류모델_구축기획서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_런지_c5_상체숙임젖힘_세부오류모델_구축기획서.md"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
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
    set_cell_background(cell, "1E3A8A")  # Deep Navy
    set_cell_margins(cell, top=220, bottom=220, left=200, right=200)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(16.5)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)

    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(10)
    run2.font.color.rgb = RGBColor(224, 231, 255)

    doc.add_paragraph()


def add_section_title(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)

    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(30, 58, 138)
    elif level == 2:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(37, 99, 235)
    else:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(71, 85, 105)


def add_bullet_point(doc, title, desc, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))

    run_t = p.add_run(title + ": ")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(9.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 41, 59)

    run_d = p.add_run(desc)
    run_d.font.name = "맑은 고딕"
    run_d.font.size = Pt(9.5)
    run_d.font.color.rgb = RGBColor(71, 85, 105)


def add_callout_box(doc, text, title="핵심 안내"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "EFF6FF")  # Blue-50
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)

    r_title = p.add_run(f"[{title}] ")
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(9.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(30, 58, 138)

    r_body = p.add_run(text)
    r_body.font.name = "맑은 고딕"
    r_body.font.size = Pt(9.5)
    r_body.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph()


def add_styled_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1E3A8A")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "맑은 고딕"
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(30, 41, 59)

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()


def build_planning_documents():
    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc = docx.Document()

    # 페이지 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # 0. 헤더
    add_header_box(
        doc,
        "[vidio-health-trainer] 바벨 런지 c5 세부 오류모델 구축 기획서",
        "상체의 과도한 숙임/젖힘 여부(lunge_c5) 고정밀 판별 엔진 공학적 설계 명세서\n"
        "작성일: 2026-10-03 | 대상 종목: 04_바벨_런지 | 대상 오류: 조건 5 (시상면 체간 중립 위반)"
    )

    # 1. 기획 개요 및 생체역학적 배경
    add_section_title(doc, "1. 기획 개요 및 생체역학적 배경")
    p = doc.add_paragraph()
    p.add_run(
        "바벨 런지(Barbell Lunge)는 양발이 동시에 지면을 대칭 지지하는 스쿼트나 데드리프트와 달리, "
        "선행 다리(Lead Leg)가 앞으로 전진 착지하여 체중의 대부분(75~80%)을 감당하고 후행 다리(Trailing Leg)가 보조 균형을 유지하는 "
        "대표적인 편측(Unilateral) 하지 복합 고난도 운동입니다. 이 과정에서 상체(Torso)는 척추 기립근과 코어 근육군의 협응을 통해 "
        "지면 수직선 대비 5°~10° 내외의 미세 전경(Slight Forward Lean)을 유지하며 직립해야 합니다.\n\n"
        "그러나 실제 운동 수행 중 하지 근력 고갈이나 코어 지지력 약화로 인해 다음과 같은 치명적인 생체역학적 보상 결함이 발생합니다:\n"
        "1) 상체 뒤로 과도하게 젖힘 (Torso Hyperextension / Backward Lean): 덤벨/바벨의 무게중심을 뒤로 넘겨 하체 부하를 회피하려는 현상으로, "
        "골반이 전방으로 밀려나며 요추(Lumbar Spine)에 극심한 전만 압박(Lordotic Shear Stress)을 가해 추간판 탈출 및 후관절 증후군을 유발합니다.\n"
        "2) 상체 앞으로 과도하게 고꾸라짐 (Excessive Forward Flexion / Trunk Flexion): 대퇴사두근 및 둔근 근력 부족으로 상체를 20° 이상 앞으로 숙여 "
        "척추 기립근에 무게를 전가하며, 앞무릎에 비정상적인 전단력을 집중시킵니다.\n\n"
        "본 기획서는 바벨 런지의 5번째 핵심 오류인 '상체의 과도한 숙임/젖힘 여부(lunge_c5)'를 완벽히 판별하기 위해, "
        "지금까지 vidio-health-trainer 파이프라인에서 축적된 최고의 베스트 프랙티스들을 집대성한 차세대 전용 오류 판별기를 설계합니다."
    )

    add_callout_box(
        doc,
        "스쿼트 c1~c4, 데드리프트 d1~d5, 스티프 데드리프트 sd1~sd3 v4에서 입증된 "
        "[MediaPipe Full 33관절 64F 무삭제 스플라인] + [노이즈 0% 물리적 골격 증강] + [Two-Tier 계층 구조] + "
        "[3-State 선택적 판정 체계] + [시점 적응형 캘리브레이션]을 100% 계승하여 최초 모델링부터 F1 90% 돌파를 목표로 합니다.",
        title="핵심 기획 철학"
    )

    # 2. AI-Hub 원천 데이터 라벨 전수 분석 및 도메인 분리
    add_section_title(doc, "2. AI-Hub 원천 데이터 라벨 전수 분석 및 도메인 분리")

    headers_lunge_err = ["오류 코드", "공식 오류 항목 명칭", "정상(True) 건수", "오류(False) 건수", "결함 비율", "주요 판별 생체역학 영역"]
    data_lunge_err = [
        ["lunge_c1", "앞다리 무릎 각도 90도", "1,229건", "415건", "25.2%", "시상면 앞무릎 굴곡 각도 (최하사점 90° 미달/초과)"],
        ["lunge_c2", "몸통 발 무릎 방향 일치여부", "1,333건", "311건", "18.9%", "전두면 앞무릎 외반/내반(Valgus/Varus) 및 발목 축 정렬"],
        ["lunge_c3", "뒤다리 무릎 각도 90도", "1,185건", "459건", "27.9%", "시상면 뒷무릎 굴곡 각도 및 지면 여유 간격"],
        ["lunge_c4", "척추의 중립", "571건", "1,073건", "65.3%", "관상면 측굴(몸통 갸우뚱), 수평면 체간 회전, 목 꺾임"],
        ["lunge_c5", "상체의 과도한 숙임/젖힘 여부", "1,079건", "565건", "34.4%", "시상면 체간 피치(Trunk Sagittal Pitch) 전후방 기울기"]
    ]
    add_styled_table(doc, headers_lunge_err, data_lunge_err, [1.1, 1.8, 0.9, 0.9, 0.8, 2.0])

    add_bullet_point(
        doc,
        "원천 데이터 규모 및 피험자 전수성",
        "04_바벨_런지는 총 1,066개 원천 시퀀스 폴더(정상 33개, 오류 1,033개) 및 5개 카메라 시점 환산 시 총 5,330개의 클립으로 구성됩니다. "
        "피험자 1번~9번 전원(9명)이 완비되어 있어 AGENTS.md 규정에 따른 5-Fold StratifiedGroupKFold 피험자 완전 격리 검증을 수행합니다."
    )
    add_bullet_point(
        doc,
        "lunge_c5 결함 라벨(False 565건)의 원천 Key 세부 구성",
        "AI-Hub 라벨링 분석 결과, Key 334(상체 뒤로 젖힘 단독 52건)를 필두로 Key 338(앞다리 펴짐+상체 젖힘), Key 341(발목 안쪽+상체 젖힘), "
        "Key 343(뒷다리 펴짐+상체 젖힘), Key 344(몸통 갸우뚱+상체 젖힘), Key 345~346(3중 복합), Key 347~350(시선 이탈+상체 젖힘) 등 "
        "총 11개 Key 시나리오에 걸쳐 Condition 5 결함이 분포합니다."
    )
    add_bullet_point(
        doc,
        "c4(척추 중립)와 c5(상체 숙임/젖힘)의 엄격한 생체역학적 분리",
        "라벨 분석 결과, Key 334('상체 뒤로 젖힘')는 c4(척추 중립)가 정상(True)이고 c5만 결함(False)으로 처리됩니다. "
        "반면 Key 347~350('하늘/바닥/좌/우 보고 + 상체 젖힘')은 c4와 c5가 동시 결함 처리됩니다. "
        "즉, c4는 관상면 체간 측굴(Spine Roll) 및 경추 3축 비틀림(Cervical Yaw/Pitch)에 집중하고, "
        "c5는 순수한 시상면 체간 전후 경사각(Trunk Pitch)에 100% 집중하여 모델 간 피처 간섭을 원천 차단합니다."
    )

    # 3. 지금까지 진행한 세부 오류 모델의 핵심 구성안 및 베스트 프랙티스 계승
    add_section_title(doc, "3. 선행 세부 오류 모델의 핵심 구성안 및 베스트 프랙티스 계승")

    p = doc.add_paragraph()
    p.add_run(
        "바벨 런지 c5 모델은 vidio-health-trainer 플랫폼이 기개발한 서브 모델들의 핵심 아키텍처 원칙을 그대로 계승하여 "
        "코드의 재사용성과 파이프라인의 무결성을 완벽하게 보장합니다."
    )

    headers_inherit = ["아키텍처 구성 요소", "스쿼트/데드리프트 입증 성과", "바벨 런지 c5 적용 메커니즘"]
    data_inherit = [
        [
            "MediaPipe Full 33관절\n64F 무삭제 스플라인",
            "스쿼트 c1~c4, 데드리프트 d1~d5\n전수 64F 표준화 완료",
            "원천 영상에서 33개 전신 관절(3D 좌표) 추출 후\n64프레임 3차 스플라인 보간으로 시간축 정규화 (JSON 배제)"
        ],
        [
            "노이즈 0% 물리적 골격 증강\n(Physical Augmentation)",
            "스티프 데드리프트 sd2_v3\n결함 F1 91.12% 달성 견인",
            "인위적 난수 가우시안 노이즈 0% 배제, 체형 가변(Bone Scaling)\n및 비선형 시간 축 왜곡(DTW Time Warping)만 적용"
        ],
        [
            "모호성 극복 Two-Tier\n계층형 평가 체계",
            "데드리프트 d4/sd2\nTier 1 F1 97.01%, Tier 2 91.12%",
            "Level 1(명확 단독 결함군)과 Level 2~3(복합/미세 결함군)을\n이원화하여 라벨 노이즈 방어 및 실 서비스 안정성 확보"
        ],
        [
            "최하사점(Bottom Phase)\n위상 인식 윈도우",
            "무빙 런지 서브분류기 및\n스쿼트 c3 발-무릎 일치율",
            "앞무릎 굴곡이 최대가 되는 t_bottom을 기점으로\n±6 프레임 윈도우 스냅샷을 구성하여 착지/기립 노이즈 차단"
        ],
        [
            "시점 적응형 캘리브레이션\n(View-Conditional Calibration)",
            "데드리프트 d5\n정면 뷰 F1 55.8% -> 78.0%+ 돌파",
            "시점 C(정면 0°)의 원근 단축 보정 레이어 및\n시점별 최적화 결정 임계값(tau_view) 동적 분기"
        ],
        [
            "3-State 선택적 분류\n(Selective CRC Option)",
            "전 종목 공통 프로덕션 표준\n확정 판정 F1 93.55% 달성",
            "불확실 확률 구간 [tau - margin, tau + margin] 샘플을\n'보류(Reject/Retake)' 처리하여 무리한 오판 원천 배제"
        ]
    ]
    add_styled_table(doc, headers_inherit, data_inherit, [1.8, 1.8, 2.9])

    # 4. 바벨 런지 c5 특화 생체역학 32D 피처 엔지니어링 명세
    add_section_title(doc, "4. 바벨 런지 c5 특화 생체역학 32차원 피처 엔지니어링 명세")

    p = doc.add_paragraph()
    p.add_run(
        "바벨 런지 c5 모델은 시상면(Sagittal Plane) 체간 기울기뿐만 아니라, 정면 0° 자기 가림(Occlusion) 및 원근 단축(Foreshortening)을 "
        "완벽하게 극복할 수 있도록 엄밀하게 설계된 총 32차원의 다면 생체역학 피처셋을 투입합니다."
    )

    headers_feat = ["피처 패밀리 (차원)", "피처명 (Feature Name)", "생체역학적 정의 및 판정 메커니즘", "목표 해결 결함"]
    data_feat = [
        [
            "1. 시상면 체간 피치각\n(Sagittal Pitch, 8D)",
            "lunge_torso_pitch_bottom_win\nlunge_torso_pitch_peak\nlunge_torso_pitch_min\nlunge_torso_pitch_rom\nlunge_torso_backward_integral\nlunge_torso_forward_integral\nlunge_torso_pitch_velocity_peak\nlunge_torso_bottom_slope",
            "어깨 중점-골반 중점 벡터의 수직축 대비 각도.\n최하사점 윈도우 평균, 전체 시퀀스 최대/최소 피치,\n후방 과신전(-각도) 시간적분 및 전방 과숙임(+각도) 시간적분,\n하강/상승 구간 피치 각속도 피크",
            "상체 뒤로 과도하게 젖힘\n(Key 334 등 11개 Key)\n상체 앞으로 고꾸라짐"
        ],
        [
            "2. 체간-대퇴골 골반 기구학\n(Kinematic Coupling, 6D)",
            "lunge_trunk_thigh_angle_min\nlunge_hip_extension_deficit\nlunge_pelvic_tilt_sagittal\nlunge_trunk_knee_moment_arm\nlunge_trunk_pelvis_sagittal_ratio\nlunge_femur_torso_bottom_angle",
            "체간 축과 선행 다리 대퇴골 사이각, 후행 다리 고관절\n신전 결손량, 시상면 골반 경사각(Pelvic Tilt),\n골반 중심 대비 무릎-상체 모멘트암 비율",
            "하강 시 둔근 약화로 인한\n상체 보상 전방 숙임\n골반 전방 밀림 보상"
        ],
        [
            "3. 정면 원근 수축 극복 3D\n(Frontal Invariance, 6D)",
            "lunge_torso_foreshortening_ratio\nlunge_virtual_sagittal_pitch_3d\nlunge_rib_flare_z_disp\nlunge_shoulder_pelvis_z_diff\nlunge_sagittal_depth_span\nlunge_frontal_compression_integral",
            "정면 카메라(시점 C) 투영 체간 길이 대비 어드레스 길이비\n(L_t / L_0), MediaPipe 3D z축 기반 가상 시상면 투영각,\n후방 젖힘 시 발생하는 흉골/갈비뼈 돌출 변위,\n어깨-골반 깊이 차이 시계열",
            "시점 C(정면 0°)에서의\n원근 소실 및 체간 기울기\n미탐(FN) 원천 극복"
        ],
        [
            "4. 경추-체간 및 락아웃 협응\n(Coordination & Lockout, 6D)",
            "lunge_cervico_thoracic_angle\nlunge_head_torso_hyperext_sync\nlunge_lockout_rebound_pitch\nlunge_ascent_trunk_jerk\nlunge_bottom_pitch_stability_std\nlunge_descent_pitch_smoothness",
            "귀-어깨 라인과 어깨-골반 라인의 경추 꺾임각,\n머리 젖힘과 상체 젖힘의 동시 발생성,\n상승 전환기(Initial Ascent) 및 복귀 시 상체 튕김/저크,\n최하사점 윈도우 내 피치각 표준편차(요동폭)",
            "상체 젖힘 시 머리 젖힘\n동반 패턴 (Key 349 등)\n상승 시 배치기 보상"
        ],
        [
            "5. 어드레스 동적 영점화\n(Zero-Calibration, 6D)",
            "lunge_delta_pitch_from_setup\nlunge_setup_torso_zero_drift\nlunge_subject_norm_pitch_rom\nlunge_trunk_length_scaled_pitch\nlunge_relative_extension_ratio\nlunge_baseline_subtracted_peak",
            "동작 시작 전 정지 상태(Phase 1 어드레스, 0~8F)의\n개인별 고유 상체 기울기를 기준선(Zero)으로 측정 후\n전 구간 상대 변위(Delta) 산출 및 신체 길이 정규화",
            "피험자별 골격/기본자세\n편차로 인한 위양성(FP)/\n위음성(FN) 70% 차단"
        ]
    ]
    add_styled_table(doc, headers_feat, data_feat, [1.5, 1.8, 2.1, 1.1])

    # 5. 차세대 Two-Tier 모델 아키텍처 및 파이프라인 흐름
    add_section_title(doc, "5. 차세대 Two-Tier 모델 아키텍처 및 파이프라인 흐름")

    add_bullet_point(
        doc,
        "1계층 (Tier 1: Level 1 명확 결함군)",
        "Key 334(상체 뒤로 젖힘 단독 결함) 및 완전 정상(Key 329) 중심의 고순도 명확 데이터셋. "
        "순수 상체 젖힘의 물리적 기준선을 학습하여 98% 이상의 초고정밀도(High Precision) 분류 모델을 구축합니다."
    )
    add_bullet_point(
        doc,
        "2계층 (Tier 2: Level 1~3 전체 복합 결함군)",
        "Key 338, 341, 343~350 등 다리 펴짐, 무릎 안쪽 꺾임, 시선 이탈 등이 혼재된 다중 결함 시나리오를 전수 포괄. "
        "타 결함의 간섭 속에서도 상체 기울기 위반을 독립적으로 잡아내는 복합 앙상블을 구성합니다."
    )
    add_bullet_point(
        doc,
        "Tri-Ensemble & 시점 적응형 캘리브레이션 (VCC)",
        "LightGBM(가중치 45%) + CatBoost(가중치 35%) + Random Forest(가중치 20%)의 이종 앙상블로 과적합을 차단하며, "
        "정면 시점(시점 C)과 사선/후방 시점(시점 A, B, D, E)의 확률 보정 레이어(VCC)를 탑재합니다."
    )
    add_bullet_point(
        doc,
        "3-State 선택적 판정 (Selective CRC)",
        "판정 신뢰도가 임계값 부근에 위치하는 경계성 샘플에 대해 '재측정 권고(Abstention)' 옵션을 적용하여, "
        "커버리지 90% 이상 유지 조건에서 서비스 확정 판정 F1 92% 이상을 보장합니다."
    )

    # 6. 정량적 목표 지표 (Target KPI)
    add_section_title(doc, "6. 정량적 목표 지표 (Target KPI Matrix)")

    headers_kpi = ["평가 영역 및 계층", "목표 F1-Score", "목표 정확도 (Accuracy)", "목표 ROC-AUC", "핵심 달성 기준"]
    data_kpi = [
        ["Tier 1 (1계층: Level 1 명확군)", "★ 95.0% ~ 97.0%", "94.0% 이상", "0.9600 이상", "단독 결함에 대한 오분류 제로화"],
        ["Tier 2 (2계층: Level 1~3 전체군)", "★ 88.0% ~ 91.0%", "85.0% 이상", "0.9000 이상", "복합 결함 간섭 극복 및 고검출"],
        ["3-State 선택적 확정 판정", "★ 92.0% ~ 94.0%", "89.0% 이상", "0.9300 이상", "커버리지 88.0%~92.0% 유지 전제"],
        ["시점 C (정면 0°) 단독 성능", "★ 80.0% ~ 84.0%", "78.0% 이상", "0.8600 이상", "정면 원근 단축 극복 (d5 수준 도약)"],
        ["시점 B, D (전방 45°) 성능", "★ 92.0% ~ 95.0%", "90.0% 이상", "0.9400 이상", "시상면 관측 최적 뷰 무결성 확보"],
        ["시점 A, E (후방 45°) 성능", "★ 88.0% ~ 92.0%", "86.0% 이상", "0.9100 이상", "바벨/등 가림 보정 필터 적용"]
    ]
    add_styled_table(doc, headers_kpi, data_kpi, [1.8, 1.2, 1.2, 1.1, 1.2])

    # 7. 단계별 구현 로드맵 및 산출물 배포 계획
    add_section_title(doc, "7. 단계별 구현 로드맵 및 산출물 배포 계획")

    headers_roadmap = ["단계 (Phase)", "핵심 작업 내용", "예상 소요 / 상세 내용", "최종 산출물"]
    data_roadmap = [
        [
            "Phase 1:\n원천 데이터셋 변환",
            "04_바벨_런지 1,066개 폴더(5,330개 클립)로부터\nMediaPipe Full 33관절 추출 및 64F 스플라인 보간",
            "ProcessPoolExecutor 멀티프로세싱 기반 고속 추출\nAI-Hub 3D JSON 영구 배제 준수",
            "D:\\Data\\전처리_데이터\\\nMediaPipe_Full33_64F\\\n04_바벨_런지\\*.npz"
        ],
        [
            "Phase 2:\nc5 특화 32D 피처 및\n모호성 레벨 캐시 구축",
            "lunge_c5_torso_features_32d.py 모듈 개발 및\nLevel 1~4 모호성 분류 메타데이터 생성",
            "t_bottom 위상 탐색 및 32차원 생체역학 텐서 추출\nKey 기반 모호성 체계 확립",
            "lunge_c5_32d_features.npz\nlunge_c5_ambiguity_levels.npz"
        ],
        [
            "Phase 3:\nTwo-Tier Tri-Ensemble\n모델 학습 및 CV 검증",
            "물리적 골격 증강(노이즈 0%) 적용 후\n5-Fold StratifiedGroupKFold 피험자 격리 학습",
            "LightGBM + CatBoost + RF 최적 하이퍼파라미터 튜닝\nTier 1 및 Tier 2 OOF 예측치 산출",
            "train_and_eval_lunge_c5_twotier.py\nlunge_c5_oof_predictions.npz"
        ],
        [
            "Phase 4:\n5대 시점 전수 감사 및\n3-State 최적화",
            "AGENTS.md 공식 5대 시점(시점 A~E)별 F1/AUC 전수 대조\n시점 적응형 임계값 및 거부 마진 확정",
            "정면/후방 뷰 성능 하락 여부 철저 감사\n최적 3-State 결정 경계선 확정",
            "lunge_c5_multiview_audit.json\ntune_lunge_c5_3state.py"
        ],
        [
            "Phase 5:\n표준 추론 엔진 패키징 및\n공식 성과보고서 발간",
            "단일 시퀀스 추론 인터페이스(LungeTorsoClassifier) 배포\n단위 테스트(test_lunge_c5.py) 및 최종 Word 보고서",
            "인터페이스 기반 독립 모듈화 규정 준수\n가중치 번들(.pkl) 패키징",
            "lunge_c5_classifier.py\n바벨_런지_c5_공식성과보고서.docx"
        ]
    ]
    add_styled_table(doc, headers_roadmap, data_roadmap, [1.3, 2.2, 1.8, 1.2])

    # 저장
    doc.save(OUT_DOCX_PATH)
    print(f"[성공] Word 기획서 생성 완료: {OUT_DOCX_PATH}")

    # 마크다운 문서 생성
    generate_markdown_doc()


def generate_markdown_doc():
    md_content = """# [vidio-health-trainer] 바벨 런지 c5(상체의 과도한 숙임/젖힘 여부) 세부 오류모델 구축 기획서

> **문서 버전**: v1.0 Official Architecture Planning  
> **공식 프로젝트명**: `vidio-health-trainer`  
> **대상 운동 종목**: `04_바벨_런지` (AI-Hub 원천 데이터셋 1,066개 폴더 / 5개 시점 5,330개 클립)  
> **대상 세부 오류**: 조건 5. 상체의 과도한 숙임/젖힘 여부 (`lunge_c5`, Torso Sagittal Neutrality Defect)  
> **작성 일자**: 2026년 10월 03일 (6주차)  
> **핵심 아키텍처**: 64프레임 3차 스플라인 + 순수 비전 32D 특화 피처 + Level 1~4 Two-Tier Tri-Ensemble + 3-State 신뢰 보류  
> **준수 규정**: `AGENTS.md` 전수 규정 (카메라 5대 공식 명칭, 피험자 9명 전수, AI-Hub JSON/라벨 엄격 격리)  

---

## 1. 기획 개요 및 생체역학적 배경

바벨 런지(Barbell Lunge)는 양발이 동시에 지면을 대칭 지지하는 스쿼트나 데드리프트와 달리, 선행 다리(Lead Leg)가 앞으로 전진 착지하여 체중의 대부분(75~80%)을 감당하고 후행 다리(Trailing Leg)가 보조 균형을 유지하는 대표적인 편측(Unilateral) 하지 복합 고난도 운동입니다. 이 과정에서 상체(Torso)는 척추 기립근과 코어 근육군의 협응을 통해 지면 수직선 대비 5°~10° 내외의 미세 전경(Slight Forward Lean)을 유지하며 직립해야 합니다.

그러나 실제 운동 수행 중 하지 근력 고갈이나 코어 지지력 약화로 인해 다음과 같은 치명적인 생체역학적 보상 결함이 발생합니다:
1. **상체 뒤로 과도하게 젖힘 (Torso Hyperextension / Backward Lean)**: 바벨의 무게중심을 뒤로 넘겨 하체 부하를 회피하려는 현상으로, 골반이 전방으로 밀려나며 요추(Lumbar Spine)에 극심한 전만 압박(Lordotic Shear Stress)을 가해 추간판 탈출 및 후관절 증후군을 유발합니다.
2. **상체 앞으로 과도하게 고꾸라짐 (Excessive Forward Flexion / Trunk Flexion)**: 대퇴사두근 및 둔근 근력 부족으로 상체를 20° 이상 앞으로 숙여 척추 기립근에 무게를 전가하며, 앞무릎에 비정상적인 전단력을 집중시킵니다.

본 기획서는 바벨 런지의 5번째 핵심 오류인 '상체의 과도한 숙임/젖힘 여부(lunge_c5)'를 완벽히 판별하기 위해, 지금까지 vidio-health-trainer 플랫폼에서 축적된 최고의 베스트 프랙티스들을 집대성한 차세대 전용 오류 판별기를 설계합니다.

> **[핵심 기획 철학]**  
> 스쿼트 c1~c4, 데드리프트 d1~d5, 스티프 데드리프트 sd1~sd3 v4에서 입증된 **[MediaPipe Full 33관절 64F 무삭제 스플라인] + [노이즈 0% 물리적 골격 증강] + [Two-Tier 계층 구조] + [3-State 선택적 판정 체계] + [시점 적응형 캘리브레이션]**을 100% 계승하여 최초 모델링부터 F1 90% 돌파를 목표로 합니다.

---

## 2. AI-Hub 원천 데이터 라벨 전수 분석 및 도메인 분리

| 오류 코드 | 공식 오류 항목 명칭 | 정상(True) 건수 | 오류(False) 건수 | 결함 비율 | 주요 판별 생체역학 영역 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **lunge_c1** | 앞다리 무릎 각도 90도 | 1,229건 | 415건 | 25.2% | 시상면 앞무릎 굴곡 각도 (최하사점 90° 미달/초과) |
| **lunge_c2** | 몸통 발 무릎 방향 일치여부 | 1,333건 | 311건 | 18.9% | 전두면 앞무릎 외반/내반(Valgus/Varus) 및 발목 축 정렬 |
| **lunge_c3** | 뒤다리 무릎 각도 90도 | 1,185건 | 459건 | 27.9% | 시상면 뒷무릎 굴곡 각도 및 지면 여유 간격 |
| **lunge_c4** | 척추의 중립 | 571건 | 1,073건 | 65.3% | 관상면 측굴(몸통 갸우뚱), 수평면 체간 회전, 목 꺾임 |
| **lunge_c5** | **상체의 과도한 숙임/젖힘 여부** | **1,079건** | **565건** | **34.4%** | **시상면 체간 피치(Trunk Sagittal Pitch) 전후방 기울기** |

- **원천 데이터 규모 및 피험자 전수성**: 04_바벨_런지는 총 1,066개 원천 시퀀스 폴더(정상 33개, 오류 1,033개) 및 5개 카메라 시점 환산 시 총 5,330개의 클립으로 구성됩니다. 피험자 1번~9번 전원(9명)이 완비되어 있어 AGENTS.md 규정에 따른 5-Fold StratifiedGroupKFold 피험자 완전 격리 검증을 수행합니다.
- **lunge_c5 결함 라벨(False 565건)의 원천 Key 세부 구성**: AI-Hub 라벨링 분석 결과, Key 334(상체 뒤로 젖힘 단독 52건)를 필두로 Key 338(앞다리 펴짐+상체 젖힘), Key 341(발목 안쪽+상체 젖힘), Key 343(뒷다리 펴짐+상체 젖힘), Key 344(몸통 갸우뚱+상체 젖힘), Key 345~346(3중 복합), Key 347~350(시선 이탈+상체 젖힘) 등 총 11개 Key 시나리오에 걸쳐 Condition 5 결함이 분포합니다.
- **c4(척추 중립)와 c5(상체 숙임/젖힘)의 엄격한 생체역학적 분리**: 라벨 분석 결과, Key 334('상체 뒤로 젖힘')는 c4(척추 중립)가 정상(True)이고 c5만 결함(False)으로 처리됩니다. 반면 Key 347~350('하늘/바닥/좌/우 보고 + 상체 젖힘')은 c4와 c5가 동시 결함 처리됩니다. 즉, c4는 관상면 체간 측굴(Spine Roll) 및 경추 3축 비틀림(Cervical Yaw/Pitch)에 집중하고, c5는 순수한 시상면 체간 전후 경사각(Trunk Pitch)에 100% 집중하여 모델 간 피처 간섭을 원천 차단합니다.

---

## 3. 선행 세부 오류 모델의 핵심 구성안 및 베스트 프랙티스 계승

| 아키텍처 구성 요소 | 스쿼트/데드리프트 입증 성과 | 바벨 런지 c5 적용 메커니즘 |
| :--- | :--- | :--- |
| **MediaPipe Full 33관절<br>64F 무삭제 스플라인** | 스쿼트 c1~c4, 데드리프트 d1~d5<br>전수 64F 표준화 완료 | 원천 영상에서 33개 전신 관절(3D 좌표) 추출 후<br>64프레임 3차 스플라인 보간으로 시간축 정규화 (JSON 배제) |
| **노이즈 0% 물리적 골격 증강<br>(Physical Augmentation)** | 스티프 데드리프트 sd2_v3<br>결함 F1 91.12% 달성 견인 | 인위적 난수 가우시안 노이즈 0% 배제, 체형 가변(Bone Scaling)<br>및 비선형 시간 축 왜곡(DTW Time Warping)만 적용 |
| **모호성 극복 Two-Tier<br>계층형 평가 체계** | 데드리프트 d4/sd2<br>Tier 1 F1 97.01%, Tier 2 91.12% | Level 1(명확 단독 결함군)과 Level 2~3(복합/미세 결함군)을<br>이원화하여 라벨 노이즈 방어 및 실 서비스 안정성 확보 |
| **최하사점(Bottom Phase)<br>위상 인식 윈도우** | 무빙 런지 서브분류기 및<br>스쿼트 c3 발-무릎 일치율 | 앞무릎 굴곡이 최대가 되는 t_bottom을 기점으로<br>±6 프레임 윈도우 스냅샷을 구성하여 착지/기립 노이즈 차단 |
| **시점 적응형 캘리브레이션<br>(View-Conditional Calibration)** | 데드리프트 d5<br>정면 뷰 F1 55.8% -> 78.0%+ 돌파 | 시점 C(정면 0°)의 원근 단축 보정 레이어 및<br>시점별 최적화 결정 임계값(tau_view) 동적 분기 |
| **3-State 선택적 분류<br>(Selective CRC Option)** | 전 종목 공통 프로덕션 표준<br>확정 판정 F1 93.55% 달성 | 불확실 확률 구간 [tau - margin, tau + margin] 샘플을<br>'보류(Reject/Retake)' 처리하여 무리한 오판 원천 배제 |

---

## 4. 바벨 런지 c5 특화 생체역학 32차원 피처 엔지니어링 명세

```
                              [바벨 런지 c5 32차원 특화 피처셋]
                                              │
    ┌─────────────────┬───────────────────────┼───────────────────────┬─────────────────┐
    ▼                 ▼                       ▼                       ▼                 ▼
[1. 시상면 피치 (8D)] [2. 체간-골반 기구학 (6D)] [3. 정면 원근 수축 (6D)] [4. 경추-락아웃 (6D)] [5. 동적 영점화 (6D)]
• 최하사점 윈도우 피치 • 대퇴골-체간 최소각   • 투영 길이 단축비     • 경추-흉추 꺾임각     • 어드레스 영점 차감
• 최대/최소 피치각     • 고관절 신전 결손량   • 3D 가상 시상각       • 머리-상체 젖힘 동기  • 셋업 드리프트
• 후방/전방 시간적분   • 골반 전후 경사각     • 흉골 z축 돌출 변위   • 상승기 체간 저크     • 체형 스케일 정규화
• 각속도 피크          • 모멘트암 비율        • 깊이축 스팬          • 최하사점 안정도      • 상대 신전 비율
```

| 피처 패밀리 (차원) | 피처명 (Feature Name) | 생체역학적 정의 및 판정 메커니즘 | 목표 해결 결함 |
| :--- | :--- | :--- | :--- |
| **1. 시상면 체간 피치각<br>(Sagittal Pitch, 8D)** | `lunge_torso_pitch_bottom_win`<br>`lunge_torso_pitch_peak`<br>`lunge_torso_pitch_min`<br>`lunge_torso_pitch_rom`<br>`lunge_torso_backward_integral`<br>`lunge_torso_forward_integral`<br>`lunge_torso_pitch_velocity_peak`<br>`lunge_torso_bottom_slope` | 어깨 중점-골반 중점 벡터의 수직축 대비 각도.<br>최하사점 윈도우 평균, 전체 시퀀스 최대/최소 피치,<br>후방 과신전(-각도) 시간적분 및 전방 과숙임(+각도) 시간적분,<br>하강/상승 구간 피치 각속도 피크 | 상체 뒤로 과도하게 젖힘<br>(Key 334 등 11개 Key)<br>상체 앞으로 고꾸라짐 |
| **2. 체간-대퇴골 골반 기구학<br>(Kinematic Coupling, 6D)** | `lunge_trunk_thigh_angle_min`<br>`lunge_hip_extension_deficit`<br>`lunge_pelvic_tilt_sagittal`<br>`lunge_trunk_knee_moment_arm`<br>`lunge_trunk_pelvis_sagittal_ratio`<br>`lunge_femur_torso_bottom_angle` | 체간 축과 선행 다리 대퇴골 사이각, 후행 다리 고관절<br>신전 결손량, 시상면 골반 경사각(Pelvic Tilt),<br>골반 중심 대비 무릎-상체 모멘트암 비율 | 하강 시 둔근 약화로 인한<br>상체 보상 전방 숙임<br>골반 전방 밀림 보상 |
| **3. 정면 원근 수축 극복 3D<br>(Frontal Invariance, 6D)** | `lunge_torso_foreshortening_ratio`<br>`lunge_virtual_sagittal_pitch_3d`<br>`lunge_rib_flare_z_disp`<br>`lunge_shoulder_pelvis_z_diff`<br>`lunge_sagittal_depth_span`<br>`lunge_frontal_compression_integral` | 정면 카메라(시점 C) 투영 체간 길이 대비 어드레스 길이비<br>(L_t / L_0), MediaPipe 3D z축 기반 가상 시상면 투영각,<br>후방 젖힘 시 발생하는 흉골/갈비뼈 돌출 변위,<br>어깨-골반 깊이 차이 시계열 | 시점 C(정면 0°)에서의<br>원근 소실 및 체간 기울기<br>미탐(FN) 원천 극복 |
| **4. 경추-체간 및 락아웃 협응<br>(Coordination & Lockout, 6D)** | `lunge_cervico_thoracic_angle`<br>`lunge_head_torso_hyperext_sync`<br>`lunge_lockout_rebound_pitch`<br>`lunge_ascent_trunk_jerk`<br>`lunge_bottom_pitch_stability_std`<br>`lunge_descent_pitch_smoothness` | 귀-어깨 라인과 어깨-골반 라인의 경추 꺾임각,<br>머리 젖힘과 상체 젖힘의 동시 발생성,<br>상승 전환기(Initial Ascent) 및 복귀 시 상체 튕김/저크,<br>최하사점 윈도우 내 피치각 표준편차(요동폭) | 상체 젖힘 시 머리 젖힘<br>동반 패턴 (Key 349 등)<br>상승 시 배치기 보상 |
| **5. 어드레스 동적 영점화<br>(Zero-Calibration, 6D)** | `lunge_delta_pitch_from_setup`<br>`lunge_setup_torso_zero_drift`<br>`lunge_subject_norm_pitch_rom`<br>`lunge_trunk_length_scaled_pitch`<br>`lunge_relative_extension_ratio`<br>`lunge_baseline_subtracted_peak` | 동작 시작 전 정지 상태(Phase 1 어드레스, 0~8F)의<br>개인별 고유 상체 기울기를 기준선(Zero)으로 측정 후<br>전 구간 상대 변위(Delta) 산출 및 신체 길이 정규화 | 피험자별 골격/기본자세<br>편차로 인한 위양성(FP)/<br>위음성(FN) 70% 차단 |

---

## 5. 차세대 Two-Tier 모델 아키텍처 및 파이프라인 흐름

```
[04_바벨_런지 원천 영상 5대 시점 (5,330 Clips)]
                     │
                     ▼
[MediaPipe Full 33관절 3D 추출 & 64F 3차 스플라인 보간] (AI-Hub JSON 100% 배제)
                     │
                     ▼
[최하사점(t_bottom) 탐색 & lunge_c5 생체역학 32D 피처 추출]
                     │
                     ▼
[노이즈 0% 물리적 골격 증강 (Bone Scaling + DTW 시간 왜곡)]
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
[Tier 1: Level 1 명확군]     [Tier 2: Level 1~3 전체 복합군]
 (단독 결함 정밀도 98%+)       (복합 결함 간섭 분별 F1 90%+)
        │                         │
        └────────────┬────────────┘
                     ▼
[Tri-Ensemble: LGBM(45%) + CatBoost(35%) + RF(20%)]
                     │
                     ▼
[시점 적응형 캘리브레이션 (VCC) & 시점 C 원근 보정]
                     │
                     ▼
[3-State 선택적 판정 (정상 / 오류 / 판정보류)]
```

- **Tier 1 (1계층: Level 1 명확 결함군)**: Key 334(상체 뒤로 젖힘 단독 결함) 및 완전 정상(Key 329) 중심의 고순도 데이터셋. 순수 상체 젖힘의 물리적 기준선을 학습하여 98% 이상의 초고정밀도(High Precision) 분류 모델을 구축합니다.
- **Tier 2 (2계층: Level 1~3 전체 복합 결함군)**: Key 338, 341, 343~350 등 다리 펴짐, 무릎 안쪽 꺾임, 시선 이탈 등이 혼재된 다중 결함 시나리오를 전수 포괄. 타 결함의 간섭 속에서도 상체 기울기 위반을 독립적으로 잡아내는 복합 앙상블을 구성합니다.
- **Tri-Ensemble & 시점 적응형 캘리브레이션 (VCC)**: LightGBM(45%) + CatBoost(35%) + Random Forest(20%)의 이종 앙상블로 과적합을 차단하며, 정면 시점(시점 C)과 사선/후방 시점(시점 A, B, D, E)의 확률 보정 레이어(VCC)를 탑재합니다.
- **3-State 선택적 판정 (Selective CRC)**: 판정 신뢰도가 임계값 부근에 위치하는 경계성 샘플에 대해 '재측정 권고(Abstention)' 옵션을 적용하여, 커버리지 90% 이상 유지 조건에서 서비스 확정 판정 F1 92% 이상을 보장합니다.

---

## 6. 정량적 목표 지표 (Target KPI Matrix)

| 평가 영역 및 계층 | 목표 F1-Score | 목표 정확도 (Accuracy) | 목표 ROC-AUC | 핵심 달성 기준 |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1 (1계층: Level 1 명확군)** | **★ 95.0% ~ 97.0%** | **94.0% 이상** | **0.9600 이상** | 단독 결함에 대한 오분류 제로화 |
| **Tier 2 (2계층: Level 1~3 전체군)** | **★ 88.0% ~ 91.0%** | **85.0% 이상** | **0.9000 이상** | 복합 결함 간섭 극복 및 고검출 |
| **3-State 선택적 확정 판정** | **★ 92.0% ~ 94.0%** | **89.0% 이상** | **0.9300 이상** | 커버리지 88.0%~92.0% 유지 전제 |
| **시점 C (정면 0°) 단독 성능** | **★ 80.0% ~ 84.0%** | **78.0% 이상** | **0.8600 이상** | 정면 원근 단축 극복 (d5 수준 도약) |
| **시점 B, D (전방 45°) 성능** | **★ 92.0% ~ 95.0%** | **90.0% 이상** | **0.9400 이상** | 시상면 관측 최적 뷰 무결성 확보 |
| **시점 A, E (후방 45°) 성능** | **★ 88.0% ~ 92.0%** | **86.0% 이상** | **0.9100 이상** | 바벨/등 가림 보정 필터 적용 |

---

## 7. 단계별 구현 로드맵 및 산출물 배포 계획

| 단계 (Phase) | 핵심 작업 내용 | 예상 소요 / 상세 내용 | 최종 산출물 |
| :--- | :--- | :--- | :--- |
| **Phase 1:<br>원천 데이터셋 변환** | 04_바벨_런지 1,066개 폴더(5,330개 클립)로부터<br>MediaPipe Full 33관절 추출 및 64F 스플라인 보간 | ProcessPoolExecutor 멀티프로세싱 기반 고속 추출<br>AI-Hub 3D JSON 영구 배제 준수 | `D:\\Data\\전처리_데이터\\MediaPipe_Full33_64F\\04_바벨_런지\\*.npz` |
| **Phase 2:<br>c5 특화 32D 피처 및<br>모호성 레벨 캐시 구축** | `lunge_c5_torso_features_32d.py` 모듈 개발 및<br>Level 1~4 모호성 분류 메타데이터 생성 | t_bottom 위상 탐색 및 32차원 생체역학 텐서 추출<br>Key 기반 모호성 체계 확립 | `lunge_c5_32d_features.npz`<br>`lunge_c5_ambiguity_levels.npz` |
| **Phase 3:<br>Two-Tier Tri-Ensemble<br>모델 학습 및 CV 검증** | 물리적 골격 증강(노이즈 0%) 적용 후<br>5-Fold StratifiedGroupKFold 피험자 격리 학습 | LightGBM + CatBoost + RF 최적 하이퍼파라미터 튜닝<br>Tier 1 및 Tier 2 OOF 예측치 산출 | `train_and_eval_lunge_c5_twotier.py`<br>`lunge_c5_oof_predictions.npz` |
| **Phase 4:<br>5대 시점 전수 감사 및<br>3-State 최적화** | AGENTS.md 공식 5대 시점(시점 A~E)별 F1/AUC 전수 대조<br>시점 적응형 임계값 및 거부 마진 확정 | 정면/후방 뷰 성능 하락 여부 철저 감사<br>최적 3-State 결정 경계선 확정 | `lunge_c5_multiview_audit.json`<br>`tune_lunge_c5_3state.py` |
| **Phase 5:<br>표준 추론 엔진 패키징 및<br>공식 성과보고서 발간** | 단일 시퀀스 추론 인터페이스(`LungeTorsoClassifier`) 배포<br>단위 테스트(`test_lunge_c5.py`) 및 최종 Word 보고서 | 인터페이스 기반 독립 모듈화 규정 준수<br>가중치 번들(.pkl) 패키징 | `lunge_c5_classifier.py`<br>`바벨_런지_c5_공식성과보고서.docx` |
"""

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[성공] Markdown 기획서 생성 완료: {OUT_MD_PATH}")


if __name__ == "__main__":
    build_planning_documents()
