# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d1_background_docx.py
Purpose: 외부 AI 자문용 'd1 배경 지식.docx' 및 'd2 배경 지식.docx' 생성 스크립트
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

DOCX_FILES = [
    r"D:\자주프 문서\d1 배경 지식.docx",
    r"D:\자주프 문서\d2 배경 지식.docx",
    r"D:\자주프 문서\6주차\d1 배경 지식.docx",
    r"D:\자주프 문서\6주차\d2 배경 지식.docx",
]

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

def style_heading(p, text, level=1):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.bold = True
    if level == 1:
        run.font.size = Pt(15)
        run.font.color.rgb = NAVY
    elif level == 2:
        run.font.size = Pt(12)
        run.font.color.rgb = BLUE
    elif level == 3:
        run.font.size = Pt(10.5)
        run.font.color.rgb = BLACK

def add_callout_box(doc, text_list, title=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F7FAFC"/>'))
    tcPr.append(parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="2B6CB0"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>'))
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    if title:
        r_title = p.add_run(title + "\n")
        r_title.bold = True
        r_title.font.name = "맑은 고딕"
        r_title.font.size = Pt(10.5)
        r_title.font.color.rgb = BLUE
    for line in text_list:
        r = p.add_run(line + "\n")
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9.5)
        r.font.color.rgb = BLACK
    if p.runs:
        p.runs[-1].text = p.runs[-1].text.rstrip("\n")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def style_table_header(row, col_widths, headers):
    for i, h in enumerate(headers):
        cell = row.cells[i]
        cell.width = Inches(col_widths[i])
        set_cell_background(cell, "1A365D")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(h)
        r.bold = True
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9.5)
        r.font.color.rgb = WHITE

# 문서 제목
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after = Pt(4)
r_t = title_p.add_run("바벨 데드리프트 d2 결함 분류 모델 기술 명세서")
r_t.bold = True
r_t.font.name = "맑은 고딕"
r_t.font.size = Pt(18)
r_t.font.color.rgb = NAVY

sub_p = doc.add_paragraph()
sub_p.paragraph_format.space_after = Pt(12)
r_sub = sub_p.add_run("부제: 외부 AI 자문용 데이터 규격, 생체역학 기전, 18D 피처 수식, 실측 성능 및 F1 향상 과제 정의서\n공식 프로젝트명: vidio-health-trainer | 대상 결함: d2(발과 무릎의 방향 일치 / Knee Valgus) | 작성일: 2026-09-30")
r_sub.font.name = "맑은 고딕"
r_sub.font.size = Pt(9.5)
r_sub.font.color.rgb = GRAY

# 자문 목표
add_callout_box(doc, [
    "• 의뢰 목표: 피트니스 모션 AI 코칭 시스템 `vidio-health-trainer`의 바벨 데드리프트 d2(발과 무릎의 방향 일치 / 동적 무릎 외반) 모델의 F1-Score를 현재 55.10%에서 80% 이상으로 극대화하기 위한 외부 AI 전문 기술 자문.",
    "• 핵심 분석 사항: 소규모(385개) 불균형(3.3:1) 골격 데이터에 특화된 피처 엔지니어링, 최적 경량 아키텍처(ST-GCN, TabNet 등), F1 직접 최적화 손실함수, 시점 불변 표현 학습법 자문."
], title="[외부 AI 자문 의뢰 핵심 목표 (Mission)]")

# 1. 원천 데이터 규격
style_heading(doc.add_paragraph(), "1. 원천 데이터셋 및 입력 텐서 규격 (Dataset Specification)", level=1)
tbl_spec = doc.add_table(rows=8, cols=3)
tbl_spec.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_spec.rows[0], [1.5, 3.2, 2.3], ["항목", "상세 기술 규격", "비고 및 제약 사항"])

spec_data = [
    ("데이터 출처", "AI-Hub 피트니스 자세 평가 데이터셋 (02_바벨_데드리프트)", "국가 공공 AI 피트니스 원천 데이터"),
    ("표본 규모", "총 385개 시퀀스 (정상 295건 vs 결함 90건)", "결함 비율 23.4% (약 3.3:1 불균형)"),
    ("피험자 구성", "피험자 1번 ~ 9번 (총 9명 전원 전수 활용)", "피험자 완전 격리 5-Fold StratifiedGroupKFold"),
    ("카메라 5대 시점", "시점 A(우측후방 45°), B(우측전방 45°), C(정면 0°), D(좌측전방 45°), E(좌측후방 45°)", "전 시점 100% 전수 학습 및 감사 의무화"),
    ("입력 텐서 형상", "(N=385, T=64, J=33, C=3) float32", "단일 횟수(1 Rep) 무삭제 64프레임 보간"),
    ("관절 좌표계", "MediaPipe Pose 33개 3D 월드 좌표계 (X, Y, Z 미터 단위)", "순수 비전 좌표 (AI-Hub 3D JSON 참조 금지)"),
    ("정답 라벨 용도", "AI-Hub 2D JSON 조건: '발과 무릎의 방향 일치' (참: 0, 거짓: 1)", "오직 검증 및 지도학습 타깃으로만 사용")
]

for row_idx, data in enumerate(spec_data, 1):
    row = tbl_spec.rows[row_idx]
    bg = "FFFFFF" if row_idx % 2 == 1 else "F7FAFC"
    for col_idx, text in enumerate(data):
        cell = row.cells[col_idx]
        set_cell_background(cell, bg)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(8.5)

# 2. 생체역학 기전
style_heading(doc.add_paragraph(), "2. 데드리프트 무릎 외반(Knee Valgus)의 임상 생체역학 기전", level=1)
p_mech = doc.add_paragraph()
p_mech.paragraph_format.space_after = Pt(6)
p_mech.add_run(
    "1. 대내전근 보상 작용의 역설 (Adductor Magnus Paradox):\n"
    "데드리프트의 셋업 및 리프트오프(지면 분리) 구간에서 고관절이 깊게 굴곡될 때, 대둔근보다 대내전근(Adductor Magnus)이 가장 큰 고관절 신전 모멘트암을 가집니다. "
    "후면 사슬 근력이 부족하거나 고중량 정지 관성을 이기지 못할 때, 신경계는 고관절 신전력을 보태기 위해 대내전근을 과도하게 수축시킵니다. "
    "대내전근의 강력한 내전(Adduction) 및 내회전(Internal Rotation) 토크가 무릎을 안쪽으로 강하게 끌어당겨 '무릎 모임(Valgus)'을 유발합니다.\n\n"
    "2. 스쿼트 대비 데드리프트의 본질적 난이도 격차:\n"
    "• 스쿼트 (c3 무릎 외반: 3-State F1 97.64%): 무릎 굴곡 90~120°로 무릎 모임 시 10~20cm 거대 변위 발생, 2,605개 표본(1:1 균형).\n"
    "• 데드리프트 (d2 무릎 외반: 3-State F1 55.10%): 힙 힌지 운동으로 무릎 굴곡이 30~50°에 불과하여 무릎 모임 변위가 2~5cm 미만으로 극히 미세함. 표본 385개(결함 90개).\n\n"
    "3. 동작 위상(Phase)별 실측 통계 (사용자 관찰 검증 완료):\n"
    "사용자가 실제 이미지를 육안 검토하여 '애초에 바벨을 잡으러 내려갈 때(셋업)부터 무릎이 모여 시작한다'는 사실을 발견하였으며, 실측 결과:\n"
    "• Phase 0 (셋업 / 하강, 0~20프레임): 정상 1.054 vs 결함 0.997 (p=0.0877, 셋업 정렬 결손)\n"
    "• Phase 1 (퍼스트 풀 / 리프팅, 21~45프레임): 정상 1.015 vs 결함 0.915 (p=0.0116 ★핵심 결함)\n"
    "• Phase 2 (락아웃 / 직립, 46~64프레임): 정상 1.061 vs 결함 1.021 (p=0.2317 노이즈 구간)\n"
    "-> 64프레임 전수를 뭉뚱그리지 않고 위상별(Phase-aware)로 분리 추출하는 것이 핵심입니다."
).font.name = "맑은 고딕"

# 3. 18D 피처셋 상세 명세
style_heading(doc.add_paragraph(), "3. 현재 d2 v4.0 모델의 18D 피처셋 및 수학적 정의", level=1)
tbl_feat = doc.add_table(rows=19, cols=4)
tbl_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_feat.rows[0], [0.4, 2.2, 3.4, 1.0], ["No", "피처 변수명", "수학적 정의 및 물리적 의미", "좌표계"])

feat_data = [
    ("1", "world_kp_ratio_min_abs", "전체 구간 중 무릎 너비 ÷ 골반 너비 최솟값: min(‖L_knee - R_knee‖ / ‖L_hip - R_hip‖)", "World 3D"),
    ("2", "world_ka_deficit_integ", "무릎 너비가 발목 너비보다 좁아진 결손량의 시간 적분값", "World 3D"),
    ("3", "world_valgus_abs_max", "관상면 X축 투영 기준 양 무릎이 안쪽으로 밀려 들어간 센티미터 변위 최댓값", "World 3D"),
    ("4", "world_valgus_norm_max", "양 무릎 안쪽 변위 합 ÷ 골반 너비 (골반 너비 정규화 결함율)", "World 3D"),
    ("5", "world_shank_tilt_p90", "정강이 뼈(무릎-발목)가 안쪽으로 기운 각도의 90백분위수", "World 3D"),
    ("6", "world_valgus_accel_peak", "무릎 안쪽 변위의 2계 미분값 (무릎 꺾임 순간 가속도 피크)", "World 3D"),
    ("7", "world_morph_kp_drop", "서 있는 초기 상태 대비 최하점에서의 무릎/골반비 상대적 낙폭 (Drop)", "World 3D"),
    ("8", "can_valgus_abs_max", "가상 정면 좌표계 기준 양 무릎 안쪽 변위 합 최댓값", "Canonical 3D"),
    ("9", "can_valgus_norm_max", "가상 정면 좌표계 기준 골반 정규화 안쪽 변위 최댓값", "Canonical 3D"),
    ("10", "can_shank_tilt_p90", "가상 정면 좌표계 기준 정강이 안쪽 기울기 90백분위수", "Canonical 3D"),
    ("11", "can_ka_ratio_min", "가상 정면 좌표계 기준 무릎 너비 ÷ 발목 너비 최솟값", "Canonical 3D"),
    ("12", "can_kp_ratio_min", "가상 정면 좌표계 기준 무릎 너비 ÷ 골반 너비 최솟값", "Canonical 3D"),
    ("13", "phase0_setup_ka_min", "[신규] 셋업 구간(0~20프레임) 무릎/발목비 최솟값 (사용자 관찰 '내려갈 때 모임' 감지)", "Phase 0"),
    ("14", "phase1_pull_ka_min", "[신규] 퍼스트풀 구간(21~45프레임) 무릎/발목비 최솟값 (지면 분리 정지관성 극복)", "Phase 1"),
    ("15", "phase_delta_ka_pull", "[신규] 셋업 대비 퍼스트풀 무릎 모임 낙폭 (ΔKAWR = setup - pull)", "Phase 0 vs 1"),
    ("16", "fppa_3d_valgus_max", "[신규] 3D FPPA 전두면 외반각 최댓값: 180° - ∠(Hip-Knee-Ankle)", "Clinical 3D"),
    ("17", "ankle_rot_mismatch", "[신규] 발끝 방향 대비 정강이 안쪽 회전 어긋남 각도 (경골 비틀림)", "Transverse 2D"),
    ("18", "arm_knee_min_dist", "[신규] 손목/팔꿈치와 무릎 간의 최소 3D 유클리드 거리 (팔 차폐 노이즈 보정)", "World 3D")
]

for row_idx, data in enumerate(feat_data, 1):
    row = tbl_feat.rows[row_idx]
    bg = "FFFFFF" if row_idx % 2 == 1 else "F7FAFC"
    for col_idx, text in enumerate(data):
        cell = row.cells[col_idx]
        set_cell_background(cell, bg)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2.5)
        p.paragraph_format.space_after = Pt(2.5)
        r = p.add_run(text)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(8.5)

# 4. 실측 성능 및 병목
style_heading(doc.add_paragraph(), "4. 현재 실측 성능 지표 및 핵심 기술 병목", level=1)
tbl_perf = doc.add_table(rows=4, cols=8)
tbl_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_perf.rows[0], [0.8, 0.8, 0.9, 0.9, 0.8, 0.8, 0.8, 1.2], ["버전", "피처수", "ROC-AUC", "3-State F1", "정밀도", "재현율", "정확도", "보류율/커버리지"])

perf_data = [
    ("v2.0", "7D", "0.6502", "46.43%", "35.20%", "68.80%", "61.04%", "0.0% / 100%"),
    ("v3.0", "12D", "0.7279", "53.98%", "42.07%", "75.31%", "70.03%", "9.9% / 90.1%"),
    ("v4.0", "18D", "0.7348", "55.10%", "46.55%", "67.50%", "74.64%", "9.9% / 90.1%")
]

for row_idx, data in enumerate(perf_data, 1):
    row = tbl_perf.rows[row_idx]
    bg = "FFFFFF" if row_idx % 2 == 1 else "F7FAFC"
    for col_idx, text in enumerate(data):
        cell = row.cells[col_idx]
        set_cell_background(cell, bg)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(8.5)

p_bottle = doc.add_paragraph()
p_bottle.paragraph_format.space_before = Pt(6)
p_bottle.paragraph_format.space_after = Pt(6)
p_bottle.add_run(
    "현재 모델의 4대 핵심 기술 병목:\n"
    "1. 소규모 데이터 & 클래스 불균형: N=385 중 결함 90개(23.4%)뿐이므로 소수의 위양성(FP)에도 F1이 급락함.\n"
    "2. 미세한 변위 (Low SNR): 스쿼트 대비 무릎 모임 변위가 2~5cm에 불과하여 비전 랜드마크 지터와 중첩됨.\n"
    "3. 후방/사각 시점에서의 신체 차폐: 엉덩이를 뒤로 뺄 때 골반과 팔에 의해 무릎이 가려져 시점 E(F1 48%), 시점 D(F1 30%)가 급락함.\n"
    "4. 확신도 부족(확률 뭉침): 예측 확률이 0.20~0.35 사이에 몰려 있어 보류율 10% 이내에서 회색 지대를 완전히 분리하지 못함."
).font.name = "맑은 고딕"

# 5. 외부 AI 자문 요청 5가지
style_heading(doc.add_paragraph(), "5. 외부 AI에게 의뢰하는 핵심 자문 항목 5가지 (Consulting Questions)", level=1)
tbl_q = doc.add_table(rows=6, cols=3)
tbl_q.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_q.rows[0], [0.8, 1.8, 4.4], ["번호", "핵심 자문 주제", "구체적 질문 내용 및 기대 출력"])

q_data = [
    ("Q1", "도메인 특화 피처 엔지니어링", "385개 소규모 골격 데이터에서 미세 변위(2~5cm)를 증폭할 수 있는 추가적인 운동학 피처(예: 대퇴골-경골 회전 모멘트, 3D 위상 미분, 관절 간 거리 정규화)는 무엇이 있는가?"),
    ("Q2", "소규모 데이터 최적 아키텍처", "N=385 환경에서 대형 딥러닝(GCN, Transformer)의 과적합을 피하면서 GBDT보다 높은 F1을 낼 수 있는 경량 모델(예: 4k 파라미터 희소 ST-GCN, TabNet, Siamese Network) 설계 방안은?"),
    ("Q3", "F1 직접 최적화 손실함수", "불균형 데이터(3.3:1)에서 정확도가 아닌 F1-Score / PR-AUC를 직접 극대화할 수 있는 커스텀 손실함수(Asymmetric Focal Loss, Soft-F1 Loss)의 최적 수식과 하이퍼파라미터는?"),
    ("Q4", "시점 불변(Cross-View) 표현", "정면(F1 72%) 대비 후방/사각(F1 30~55%) 시점의 급락을 극복하기 위해, 차폐가 심한 시점에서도 무릎 상태를 강건하게 추정할 수 있는 뷰 정규화 또는 잠재 표현 기법은?"),
    ("Q5", "골격 데이터 증강(Augmentation)", "결함 90개 표본을 물리역학적 모순 없이 증강(Kinematic Manifold Mixup, 관절 지터링, 시간 축 탄력 왜곡)하여 데이터 불균형을 해소할 수 있는 구체적 알고리즘은?")
]

for row_idx, data in enumerate(q_data, 1):
    row = tbl_q.rows[row_idx]
    bg = "FFFFFF" if row_idx % 2 == 1 else "F7FAFC"
    for col_idx, text in enumerate(data):
        cell = row.cells[col_idx]
        set_cell_background(cell, bg)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text)
        r.font.name = "맑은 고딕"
        r.font.size = Pt(8.5)

for p in DOCX_FILES:
    doc.save(p)
    print(f"[+] Word 문서 생성 완료: {p}")
