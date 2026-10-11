# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_babel_squat_leakage_audit_report.py
Purpose: 바벨 스쿼트 c1~c4 세부 모델 및 상위 통합 엔진에 대한
         AI-Hub 정답 라벨/3D JSON 좌표 데이터 누수 전수 검사 공식 감사 보고서 (.docx 및 .md) 생성
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUT_DIR = r"D:\자주프 문서\8주차"
DOCX_PATH = os.path.join(OUT_DIR, "바벨스쿼트_c1_to_c4_AIHub라벨및좌표_데이터누수_전수검사_감사보고서.docx")
MD_PATH = os.path.join(OUT_DIR, "바벨스쿼트_c1_to_c4_AIHub라벨및좌표_데이터누수_전수검사_감사보고서.md")

# 색상 팔레트
NAVY = RGBColor(27, 54, 93)       # #1B365D
SLATE = RGBColor(47, 85, 151)     # #2F5597
CHARCOAL = RGBColor(38, 38, 38)   # #262626
RED = RGBColor(192, 0, 0)         # #C00000
GREEN = RGBColor(38, 128, 0)      # #268000
AMBER = RGBColor(180, 100, 0)     # #B46400

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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

def add_header_banner(doc, title, subtitle):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, '1B365D')
    set_cell_margins(cell, top=220, bottom=220, left=240, right=240)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r0 = p.add_run("vidio-health-trainer | 인공지능 모델 파이프라인 무결성 감사 보고서\n")
    r0.font.size = Pt(9.5)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(217, 226, 243)
    r1 = p.add_run(f"{title}\n")
    r1.font.size = Pt(15.0)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(255, 255, 255)
    r2 = p.add_run(subtitle)
    r2.font.size = Pt(9.0)
    r2.font.color.rgb = RGBColor(220, 230, 242)
    doc.add_paragraph()

def add_callout(doc, title, text, bg_hex='F2F4F7', border_hex='1B365D'):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    p = cell.paragraphs[0]
    r_t = p.add_run(f"■ {title}\n")
    r_t.font.bold = True
    r_t.font.size = Pt(10.5)
    r_t.font.color.rgb = NAVY
    r_b = p.add_run(text)
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = CHARCOAL
    doc.add_paragraph()

def add_h1(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    r.font.size = Pt(12.5)
    r.font.bold = True
    r.font.color.rgb = NAVY
    return h

def add_h2(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = SLATE
    return h

def add_p(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    r.font.color.rgb = CHARCOAL
    return p

def format_table(tbl, col_widths, headers, data):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    # 헤더 행
    for col_idx, (h_text, w) in enumerate(zip(headers, col_widths)):
        c = tbl.rows[0].cells[col_idx]
        c.width = Inches(w)
        set_cell_background(c, '1B365D')
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    # 데이터 행
    for row_idx, row in enumerate(data, 1):
        bg = 'F8F9FA' if row_idx % 2 == 1 else 'FFFFFF'
        for col_idx, (val, w) in enumerate(zip(row, col_widths)):
            c = tbl.rows[row_idx].cells[col_idx]
            c.width = Inches(w)
            set_cell_background(c, bg)
            set_cell_margins(c, 70, 70, 90, 90)
            p = c.paragraphs[0]
            if col_idx in [0, 1]:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            r.font.color.rgb = CHARCOAL
            if "FAIL" in str(val) or "치명적" in str(val) or "누수" in str(val):
                r.font.bold = True
                r.font.color.rgb = RED
            elif "PASS" in str(val) or "무결" in str(val) or "0%" in str(val):
                r.font.bold = True
                r.font.color.rgb = GREEN
            elif "주의" in str(val) or "잔재" in str(val):
                r.font.bold = True
                r.font.color.rgb = AMBER

def generate_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # 1. 헤더 배너
    add_header_banner(
        doc,
        title="바벨 스쿼트 c1~c4 모델 AI-Hub 라벨/좌표 데이터 누수 전수 검사 결과 보고서",
        subtitle="공식 프로덕션 파이프라인 전수 소스코드 감사 · 3D JSON 참조 및 정답 라벨 피처화 여부 검증"
    )

    # 2. 핵심 요약
    add_callout(
        doc,
        title="핵심 감사 요약 (Executive Audit Summary)",
        text="1. 감사 대상: 바벨 스쿼트 4대 세부 결함 모델(c1 척추, c2 시선, c3 무릎, c4 발바닥) 및 상위 16-State MoE 통합 진단 엔진 v11.0 전체 소스코드와 학습 피처셋.\n"
             "2. c2, c3, c4 모델 무결성 (PASS): 100% 순수 비전 MediaPipe 33관절 64F 스플라인 보간 좌표만을 입력 피처로 사용하며, AI-Hub 3D JSON 및 정답 라벨을 피처에 유입한 누수가 전무함(0%). 9인 피험자 완전 격리(StratifiedGroupKFold) 준수 확인.\n"
             "3. c1 모델 과거 누수 식별 및 조치 내역 (CRITICAL -> RESOLVED): 과거 Model-A(v9~v11_a)가 AI-Hub 3D JSON(*_3D.json)에서 척추(Neck, Back, Waist) 좌표를 읽어와 38D 피처를 계산하던 치명적 데이터 누수 및 배포 차단 결함이 존재했음. 2026.09.30 원본 영상 SpinePose 순수 비전 모델(Model-B, v11.1_distilled_spinepose)로 전면 교체 완료됨.\n"
             "4. 상위 메타러너 잔재 주의 (WARNING): 통합 엔진 메타러너 학습 데이터셋(meta_fusion_v7_dataset.npz)의 c1 컬럼이 신규 Model-B가 아닌 구 Model-A OOF로 패키징되어 학습된 잔재가 남아 있어, 메타러너 재학습 및 데이터셋 갱신이 필요함."
    )

    # 3. 전수 감사 매트릭스 표
    add_h1(doc, "1. 바벨 스쿼트 모델별 데이터 누수 전수 감사 매트릭스")
    add_p(doc, "프로젝트 코딩 및 아키텍처 무결성 표준(AI-Hub JSON 및 라벨 격리 원칙)에 의거하여, 피처 추출원, AI-Hub JSON 참조, 라벨 피처화, 학습 교차검증 격리를 전수 감사한 종합 결과는 다음과 같습니다.")

    headers_mat = ["모델 / 컴포넌트", "핵심 소스코드", "피처 입력 소스 (X)", "AI-Hub 3D JSON", "정답 라벨 유입", "피험자 분할", "최종 감사 판정"]
    widths_mat = [1.1, 1.2, 1.4, 0.7, 0.7, 0.8, 0.9]
    
    data_mat = [
        ["c1 척추 (구 Model-A)", "c1_model_v11_a.py\ntrain_c1_v11_a.py", "AI-Hub 3D JSON 보간 (Neck, Back, Waist)", "유입 (100%)", "피처 불포함", "9인 완전격리", "FAIL (데이터 누수)"],
        ["c1 척추 (신 Model-B)", "c1_model_v11_1.py\ntrain_c1_distill.py", "RGB -> SpinePose 9분절 + MP33 (순수 비전)", "배제 (0%)", "피처 불포함\n(KD 손실 30%)", "9인 완전격리", "PASS (순수 비전 자립)\n단, 지식증류 잔재"],
        ["c2 시선/고개", "c2_model_v8.py\nc2_model_v7.py", "MediaPipe 33관절 64F (코, 귀, 어깨, 골반)", "배제 (0%)", "배제 (0%)", "9인 완전격리", "PASS (완전 무결)"],
        ["c3 무릎 외반", "c3_model_v6.py\ntrain_c3_v6.py", "MediaPipe 33관절 64F (골반, 무릎, 발목, 발)", "배제 (0%)", "배제 (0%)", "9인 완전격리", "PASS (완전 무결)"],
        ["c4 발바닥 접지", "c4_model_v5_1.py\ntrain_c4_v5.py", "MediaPipe 33관절 64F (골반, 발목, 뒤꿈치, 발끝)", "배제 (0%)", "배제 (0%)", "9인 완전격리", "PASS (완전 무결)"],
        ["보조 피처 (품질/사슬)", "extract_quality.py\nextract_kinetic.py", "MediaPipe 33관절 64F 물리 가속도/동역학", "배제 (0%)", "배제 (0%)", "N/A (비학습)", "PASS (완전 무결)"],
        ["통합 메타러너 (MoE)", "babel_squat_v11.py\ntrain_meta_v10_1.py", "39D 메타 (OOF 확률 + 마진 + 품질 + 사슬)", "배제 (0%)", "배제 (0%)", "9인 완전격리", "WARNING (c1 구형 OOF 잔재)"]
    ]

    tbl_mat = doc.add_table(rows=len(data_mat) + 1, cols=len(headers_mat))
    format_table(tbl_mat, widths_mat, headers_mat, data_mat)
    doc.add_paragraph()

    # 4. 세부 모델별 정밀 감사 보고
    add_h1(doc, "2. 세부 결함 모델별 코드 및 데이터 정밀 검증 결과")

    add_h2(doc, "2.1 c1 (척추의 중립 위반) 모델: AI-Hub 3D JSON 탈피 및 SpinePose 자립화")
    add_p(doc, "• 식별된 과거 누수 (Model-A): train_and_evaluate_c1_v11_a.py의 lines 75~88에서 JSON_DIR(d:/Data/01_raw_data/labels_json/01_바벨_스쿼트/3D/*_3D.json)을 직접 열어 'Neck', 'Back', 'Waist' 좌표를 파싱한 뒤 PCHIP 3차 스플라인으로 9분절을 보간하여 38D 및 27D 피처셋(squat_c1_v11_a_38d_feats.npz)을 구축했습니다. 이는 모바일/웹 카메라 실서비스 환경에서는 존재할 수 없는 정답 좌표를 피처로 활용한 심각한 데이터 누수였습니다.")
    add_p(doc, "• 해결 조치 (Model-B 구축): extract_true_spinepose_64f.py를 신규 개발하여 2,605건 원본 비디오 프레임에서 직접 SpinePose(37 Kps + 3D Lifter)를 구동하여 순수 비전 9대 척추 분절(squat_c1_spinepose_true_vision_64f.npz)을 추출하였습니다. 이 피처로부터 One-Euro 지터 필터 및 뼈 길이 제약을 가한 squat_c1_v11_denoised_spinepose_feats.npz를 생성하여 완전히 자립적인 비전 모델을 완성했습니다.")
    add_p(doc, "• 현재 가중치 상태: c1_model_v11_weights.pkl은 v11.1_distilled_spinepose 버전으로 배포되어 있으며, 실시간 추론 시에는 kps_sequence_3d만을 받아 척추를 추정하므로 AI-Hub JSON 없이 정상 동작합니다.")

    add_h2(doc, "2.2 c2 (시선 및 고개 정면) 모델: 100% 순수 비전 무결성 확인")
    add_p(doc, "• 피처 추출 함수: c2_model_v7.py의 extract_c2_v7_19d_features_from_seq(kps_33_64f)")
    add_p(doc, "• 사용 관절: 코(#0), 좌우 귀(#7, #8), 어깨(#11, #12), 골반(#23, #24) 등 MediaPipe 관절만을 사용하며, 체간 정규직교계(Body-Local Orthonormal Basis) 상에서 경추 굴곡각 및 단일 가시 귀(Visible Ear) 기반 헤드 피치를 산출합니다.")
    add_p(doc, "• 감사 결론: AI-Hub 3D JSON 및 정답 라벨에 대한 참조가 코드 전수에서 완전히 배제되어 있으며(0%), 어떠한 데이터 누수도 존재하지 않는 순수 비전 모델임을 확인하였습니다.")

    add_h2(doc, "2.3 c3 (발과 무릎 외반) 모델: 100% 순수 비전 무결성 확인")
    add_p(doc, "• 피처 추출 함수: c3_model_v6.py의 extract_c3_v6_features_from_seq(kps_33_64f)")
    add_p(doc, "• 사용 관절: 어깨(#11, #12), 골반(#23, #24), 무릎(#25, #26), 발목(#27, #28), 뒤꿈치(#29, #30) 등 MediaPipe 관절 전용.")
    add_p(doc, "• 생체역학 기법: 골반 관상면(Coronal) 로컬 기저계 투영을 통해 절대 무릎-발목 너비 비율(Knee-to-Ankle Ratio) 및 대퇴-경골 3차원 관상각을 측정하며, 하지 뼈 길이 정규화 Dual-Stream(7:3) 역시 비전 스케일링으로만 수행됩니다.")
    add_p(doc, "• 감사 결론: AI-Hub 3D JSON 참조율 0%, 라벨 누수율 0%로 완벽한 무결성을 보유하고 있습니다.")

    add_h2(doc, "2.4 c4 (발바닥 지면 고정) 모델: 100% 순수 비전 무결성 확인")
    add_p(doc, "• 피처 추출 함수: c4_model_v5_1.py의 extract_c4_elite_9d_features 및 25D 지면 접지 피처셋")
    add_p(doc, "• 사용 관절: 골반, 무릎, 발목, 뒤꿈치(#29, #30), 발끝(엄지발가락 #31, #32)의 수직 변위 및 지면 접촉 상태 머신.")
    add_p(doc, "• 감사 결론: 지면 반력 센서나 3D 외부 라벨 없이, 오직 카메라 좌표계의 지면 기준선 중앙값 추정(Median Baseline)과 생체역학 한계 클리핑만으로 동작하여 데이터 누수가 전혀 없습니다.")

    # 5. 상위 메타러너 및 OOF 누수 정밀 진단
    add_h1(doc, "3. 상위 메타러너(MoE) 및 교차검증(OOF) 누수 감사")
    add_p(doc, "1) 교차검증 피험자 격리 무결성 (GroupKFold Integrity):")
    add_p(doc, "   - c1, c2, c3, c4 및 통합 메타러너 전체가 AI-Hub 9인 피험자(Subject 1~9)를 완벽히 격리하는 StratifiedGroupKFold(n_splits=5)를 적용하여, 동일 피험자의 데이터가 훈련셋과 검증셋에 동시 분할되는 환각/과적합 누수를 100% 차단하고 있습니다.")
    add_p(doc, "2) 메타러너 OOF Stacking 무결성 및 c1 잔재 (Critical Finding):")
    add_p(doc, "   - 메타러너 학습 시 하위 서브모델의 예측값으로 훈련셋 확률이 아닌 Out-Of-Fold(OOF) 예측 확률을 사용하여 Stacking 누수 자체는 철저히 방어되었습니다.")
    add_p(doc, "   - 그러나 메타러너 학습 특성 행렬인 meta_fusion_v7_dataset.npz 구성 시, c1 컬럼에 신규 순수 비전 Model-B OOF가 아닌 구형 AI-Hub JSON 기반 Model-A OOF(oof_c1_v11_selective_predictions.npz)가 투입되어 학습되었습니다.")
    add_p(doc, "   - 이로 인해 메타러너는 c1 모델의 F1이 97.13% 수준으로 매우 정밀하다고 가정한 상태에서 라우팅 가중치를 학습했으나, 실제 순수 비전 배포 환경에서는 c1 Model-B(F1 84.02%)가 입력되므로 상위 메타러너의 신뢰도 마진 및 정상 판정에 도메인 불일치(Covariate Shift)가 잔존합니다.")

    # 6. 향후 조치 방안
    add_h1(doc, "4. 무결성 완성 및 프로덕션 정상화 조치 권고사항")
    add_p(doc, "1. [완결 확인] c2, c3, c4 모델은 코드 및 피처 전수 검사 결과 100% 순수 비전 무결성이 입증되었으므로 현행 프로덕션을 유지합니다.")
    add_p(doc, "2. [완결 확인] c1 모델 역시 현행 c1_model_v11_weights.pkl이 SpinePose 순수 비전 가중치(v11.1_distilled_spinepose)로 승격되어 있으므로 추론 시 AI-Hub JSON 의존성이 완전히 제거되었습니다.")
    add_p(doc, "3. [즉시 개선 권고] meta_fusion_v7_dataset.npz의 c1 확률 컬럼을 squat_c1_oof_predictions_a_and_b.npz['oof_p_B'](순수 비전 OOF)로 100% 교체 갱신하고, 상위 MoE 메타러너(joint_meta_16way_v10_1_weights.pth)를 재학습하여 메타 레벨의 잔재 누수 및 공변량 변화를 원천 해소할 것을 권고합니다.")

    doc.save(DOCX_PATH)
    print(f"[OK] Word 감사 보고서 저장 완료: {DOCX_PATH}")

    # 마크다운 보고서 생성
    md_content = f"""# [vidio-health-trainer] 바벨 스쿼트 c1~c4 모델 AI-Hub 라벨/좌표 데이터 누수 전수 검사 결과 보고서

- **문서 버전**: v1.0 (공식 전수 감사본)
- **작성일자**: 2026년 10월 8일
- **프로젝트 명칭**: `vidio-health-trainer`
- **감사 기준**: `Documents/Coding_Rules.md` 및 `d:\\AGENTS.md` (AI-Hub JSON 및 라벨 격리 원칙)

---

## 1. 핵심 감사 요약 (Executive Summary)

1. **감사 대상**: 바벨 스쿼트 4대 세부 결함 모델($c_1$ 척추, $c_2$ 시선, $c_3$ 무릎, $c_4$ 발바닥) 및 상위 16-State MoE 통합 진단 엔진(v11.0) 전체 소스코드와 학습 피처셋.
2. **$c_2, c_3, c_4$ 모델 무결성 (PASS)**: 
   - 100% 순수 비전 MediaPipe 33관절 64F 스플라인 보간 좌표만을 입력 피처로 사용.
   - AI-Hub 3D JSON 및 정답 라벨을 피처에 유입한 누수가 전무함(**0%**).
   - 9인 피험자 완전 격리(`StratifiedGroupKFold`) 준수 확인.
3. **$c_1$ 모델 과거 누수 식별 및 조치 내역 (CRITICAL -> RESOLVED)**:
   - 과거 Model-A(v9~v11_a)가 AI-Hub 3D JSON(`*_3D.json`)에서 척추(`Neck`, `Back`, `Waist`) 좌표를 읽어와 38D 피처를 계산하던 **치명적 데이터 누수 및 배포 차단 결함**이 존재했음.
   - 2026.09.30 원본 영상 SpinePose 순수 비전 모델(Model-B, `v11.1_distilled_spinepose`)로 전면 교체 완료됨.
4. **상위 메타러너 잔재 주의 (WARNING)**:
   - 통합 엔진 메타러너 학습 데이터셋(`meta_fusion_v7_dataset.npz`)의 $c_1$ 컬럼이 신규 Model-B가 아닌 구 Model-A OOF로 패키징되어 학습된 잔재가 남아 있어, 메타러너 재학습 및 데이터셋 갱신이 필요함.

---

## 2. 바벨 스쿼트 모델별 데이터 누수 전수 감사 매트릭스

| 모델 / 컴포넌트 | 핵심 소스코드 | 피처 입력 소스 ($X$) | AI-Hub 3D JSON | 정답 라벨 유입 | 피험자 분할 | 최종 감사 판정 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **$c_1$ 척추 (구 Model-A)** | `c1_model_v11_a.py`<br>`train_c1_v11_a.py` | AI-Hub 3D JSON 보간 (`Neck`, `Back`, `Waist`) | **유입 (100%)** | 피처 불포함 | 9인 완전격리 | **FAIL (데이터 누수)** |
| **$c_1$ 척추 (신 Model-B)** | `c1_model_v11_1.py`<br>`train_c1_distill.py` | RGB -> SpinePose 9분절 + MP33 (순수 비전) | **배제 (0%)** | 피처 불포함<br>*(KD 손실 30%)* | 9인 완전격리 | **PASS (순수 비전 자립)**<br>단, 지식증류 잔재 |
| **$c_2$ 시선/고개** | `c2_model_v8.py`<br>`c2_model_v7.py` | MediaPipe 33관절 64F (코, 귀, 어깨, 골반) | **배제 (0%)** | **배제 (0%)** | 9인 완전격리 | **PASS (완전 무결)** |
| **$c_3$ 무릎 외반** | `c3_model_v6.py`<br>`train_c3_v6.py` | MediaPipe 33관절 64F (골반, 무릎, 발목, 발) | **배제 (0%)** | **배제 (0%)** | 9인 완전격리 | **PASS (완전 무결)** |
| **$c_4$ 발바닥 접지** | `c4_model_v5_1.py`<br>`train_c4_v5.py` | MediaPipe 33관절 64F (골반, 발목, 뒤꿈치, 발끝) | **배제 (0%)** | **배제 (0%)** | 9인 완전격리 | **PASS (완전 무결)** |
| **보조 피처 (품질/사슬)** | `extract_quality.py`<br>`extract_kinetic.py` | MediaPipe 33관절 64F 물리 가속도/동역학 | **배제 (0%)** | **배제 (0%)** | N/A (비학습) | **PASS (완전 무결)** |
| **통합 메타러너 (MoE)** | `babel_squat_v11.py`<br>`train_meta_v10_1.py` | 39D 메타 (OOF 확률 + 마진 + 품질 + 사슬) | **배제 (0%)** | **배제 (0%)** | 9인 완전격리 | **WARNING ($c_1$ 구형 OOF 잔재)** |

---

## 3. 세부 결함 모델별 코드 및 데이터 정밀 검증 결과

### 3.1 $c_1$ (척추의 중립 위반) 모델: AI-Hub 3D JSON 탈피 및 SpinePose 자립화
* **식별된 과거 누수 (Model-A)**:
  * `train_and_evaluate_c1_v11_a.py`의 lines 75~88에서 `JSON_DIR`(`d:/Data/01_raw_data/labels_json/01_바벨_스쿼트/3D/*_3D.json`)을 직접 열어 `Neck`, `Back`, `Waist` 좌표를 파싱한 뒤 PCHIP 3차 스플라인으로 9분절을 보간하여 38D 및 27D 피처셋(`squat_c1_v11_a_38d_feats.npz`)을 구축했습니다.
  * 이는 모바일/웹 카메라 실서비스 환경에서는 존재할 수 없는 정답 좌표를 피처로 활용한 심각한 데이터 누수였습니다.
* **해결 조치 (Model-B 구축)**:
  * `extract_true_spinepose_64f.py`를 신규 개발하여 2,605건 원본 비디오 프레임에서 직접 SpinePose(37 Kps + 3D Lifter)를 구동하여 순수 비전 9대 척추 분절(`squat_c1_spinepose_true_vision_64f.npz`)을 추출하였습니다.
  * 이 피처로부터 One-Euro 지터 필터 및 뼈 길이 제약을 가한 `squat_c1_v11_denoised_spinepose_feats.npz`를 생성하여 완전히 자립적인 비전 모델을 완성했습니다.
* **현재 가중치 상태**:
  * `c1_model_v11_weights.pkl`은 `v11.1_distilled_spinepose` 버전으로 배포되어 있으며, 실시간 추론 시에는 `kps_sequence_3d`만을 받아 척추를 추정하므로 AI-Hub JSON 없이 정상 동작합니다.

### 3.2 $c_2$ (시선 및 고개 정면) 모델: 100% 순수 비전 무결성 확인
* **피처 추출 함수**: `c2_model_v7.py`의 `extract_c2_v7_19d_features_from_seq(kps_33_64f)`
* **사용 관절**: 코(#0), 좌우 귀(#7, #8), 어깨(#11, #12), 골반(#23, #24) 등 MediaPipe 관절만을 사용하며, 체간 정규직교계(Body-Local Orthonormal Basis) 상에서 경추 굴곡각 및 단일 가시 귀(Visible Ear) 기반 헤드 피치를 산출합니다.
* **감사 결론**: AI-Hub 3D JSON 및 정답 라벨에 대한 참조가 코드 전수에서 완전히 배제되어 있으며(**0%**), 어떠한 데이터 누수도 존재하지 않는 순수 비전 모델임을 확인하였습니다.

### 3.3 $c_3$ (발과 무릎 외반) 모델: 100% 순수 비전 무결성 확인
* **피처 추출 함수**: `c3_model_v6.py`의 `extract_c3_v6_features_from_seq(kps_33_64f)`
* **사용 관절**: 어깨(#11, #12), 골반(#23, #24), 무릎(#25, #26), 발목(#27, #28), 뒤꿈치(#29, #30) 등 MediaPipe 관절 전용.
* **생체역학 기법**: 골반 관상면(Coronal) 로컬 기저계 투영을 통해 절대 무릎-발목 너비 비율(Knee-to-Ankle Ratio) 및 대퇴-경골 3차원 관상각을 측정하며, 하지 뼈 길이 정규화 Dual-Stream(7:3) 역시 비전 스케일링으로만 수행됩니다.
* **감사 결론**: AI-Hub 3D JSON 참조율 **0%**, 라벨 누수율 **0%**로 완벽한 무결성을 보유하고 있습니다.

### 3.4 $c_4$ (발바닥 지면 고정) 모델: 100% 순수 비전 무결성 확인
* **피처 추출 함수**: `c4_model_v5_1.py`의 `extract_c4_elite_9d_features` 및 25D 지면 접지 피처셋
* **사용 관절**: 골반, 무릎, 발목, 뒤꿈치(#29, #30), 발끝(엄지발가락 #31, #32)의 수직 변위 및 지면 접촉 상태 머신.
* **감사 결론**: 지면 반력 센서나 3D 외부 라벨 없이, 오직 카메라 좌표계의 지면 기준선 중앙값 추정(Median Baseline)과 생체역학 한계 클리핑만으로 동작하여 데이터 누수가 전혀 없습니다.

---

## 4. 상위 메타러너(MoE) 및 교차검증(OOF) 누수 감사

1. **교차검증 피험자 격리 무결성 (GroupKFold Integrity)**:
   - $c_1, c_2, c_3, c_4$ 및 통합 메타러너 전체가 AI-Hub 9인 피험자(Subject 1~9)를 완벽히 격리하는 `StratifiedGroupKFold(n_splits=5)`를 적용하여, 동일 피험자의 데이터가 훈련셋과 검증셋에 동시 분할되는 환각/과적합 누수를 100% 차단하고 있습니다.
2. **메타러너 OOF Stacking 무결성 및 $c_1$ 잔재 (Critical Finding)**:
   - 메타러너 학습 시 하위 서브모델의 예측값으로 훈련셋 확률이 아닌 Out-Of-Fold(OOF) 예측 확률을 사용하여 Stacking 누수 자체는 철저히 방어되었습니다.
   - **주의 사항**: 메타러너 학습 특성 행렬인 `meta_fusion_v7_dataset.npz` 구성 시, $c_1$ 컬럼에 신규 순수 비전 Model-B OOF가 아닌 구형 AI-Hub JSON 기반 Model-A OOF(`oof_c1_v11_selective_predictions.npz`)가 투입되어 학습되었습니다.
   - 이로 인해 메타러너는 $c_1$ 모델의 F1이 97.13% 수준으로 매우 정밀하다고 가정한 상태에서 라우팅 가중치를 학습했으나, 실제 순수 비전 배포 환경에서는 $c_1$ Model-B(F1 84.02%)가 입력되므로 상위 메타러너의 신뢰도 마진 및 정상 판정에 도메인 불일치(Covariate Shift)가 잔존합니다.

---

## 5. 무결성 완성 및 프로덕션 정상화 조치 권고사항

1. **[완결 확인]** $c_2, c_3, c_4$ 모델은 코드 및 피처 전수 검사 결과 100% 순수 비전 무결성이 입증되었으므로 현행 프로덕션을 유지합니다.
2. **[완결 확인]** $c_1$ 모델 역시 현행 `c1_model_v11_weights.pkl`이 SpinePose 순수 비전 가중치(`v11.1_distilled_spinepose`)로 승격되어 있으므로 추론 시 AI-Hub JSON 의존성이 완전히 제거되었습니다.
3. **[즉시 개선 권고]** `meta_fusion_v7_dataset.npz`의 $c_1$ 확률 컬럼을 `squat_c1_oof_predictions_a_and_b.npz['oof_p_B']`(순수 비전 OOF)로 100% 교체 갱신하고, 상위 MoE 메타러너(`joint_meta_16way_v10_1_weights.pth`)를 재학습하여 메타 레벨의 잔재 누수 및 공변량 변화를 원천 해소할 것을 권고합니다.
"""

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[OK] Markdown 감사 보고서 저장 완료: {MD_PATH}")

if __name__ == '__main__':
    generate_report()
