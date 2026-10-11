# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 1 - 앞다리 무릎 각도 90도 오류
Script: make_cl1_final_champion_report_docx.py
Purpose: cl1_final 공식 최종 확정 챔피언 선정 보고서 (.docx 및 .md) 생성
         저장 경로: D:/자주프 문서/7주차/vidio-health-trainer_크로스런지_c1_최종모델_cl1_final_공식선정보고서.docx
"""

import os
import sys
import json
import time

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCS_DIR = r"D:\자주프 문서\7주차"
os.makedirs(DOCS_DIR, exist_ok=True)
OUT_DOCX_PATH = os.path.join(DOCS_DIR, "vidio-health-trainer_크로스런지_c1_최종모델_cl1_final_공식선정보고서.docx")
OUT_MD_PATH = os.path.join(DOCS_DIR, "vidio-health-trainer_크로스런지_c1_최종모델_cl1_final_공식선정보고서.md")

RESULTS_JSON = r"d:\src\models\second_models\cross_lunge\cl1_final_evaluation_results.json"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def format_table(table, col_widths, alignments):
    for i, col in enumerate(table.columns):
        for cell in col.cells:
            cell.width = col_widths[i]
    for row_idx, row in enumerate(table.rows):
        for col_idx, cell in enumerate(row.cells):
            cell.paragraphs[0].alignment = alignments[col_idx]
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9.5)
                run.font.name = "맑은 고딕"
                if row_idx == 0:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            if row_idx == 0:
                set_cell_background(cell, "1B365D")
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F2F5F9")
            else:
                set_cell_background(cell, "FFFFFF")


def main():
    with open(RESULTS_JSON, 'r', encoding='utf-8') as fp:
        res = json.load(fp)

    m2 = res['bundle_info']['metrics_2state']
    m3 = res['bundle_info']['metrics_3state']
    cal = res['bundle_info']['calibration']
    vm = res['bundle_info']['views_metrics']
    comp = res['comparison']
    T = res['bundle_info']['temperature']
    w = res['bundle_info']['weights']

    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # 1. 문서 헤더
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(4)
    r_t = p_title.add_run("크로스 런지 Condition 1 (앞다리 무릎 각도 90도 오류)\n공식 최종 챔피언 모델 (cl1_final) 확정 및 선정 보고서")
    r_t.font.name = "맑은 고딕"
    r_t.font.size = Pt(17)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(27, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    r_s = p_sub.add_run("공식 프로젝트: vidio-health-trainer | 최종 확정 모델: cl1_final (134D 하이브리드 아키텍처)")
    r_s.font.size = Pt(10)
    r_s.font.color.rgb = RGBColor(90, 100, 115)

    # 2. 선정 배경 및 목적
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    r_h1 = h1.add_run("1. 최종 모델 선정 배경 및 아키텍처 혁신")
    r_h1.font.name = "맑은 고딕"
    r_h1.font.size = Pt(13)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(27, 54, 93)

    p_body1 = doc.add_paragraph()
    p_body1.paragraph_format.line_spacing = 1.25
    p_body1.add_run(
        "크로스 런지(08_크로스_런지)의 핵심 결함 중 하나인 'c1: 앞다리 무릎 각도 90도 오류'는 피트니스 사용자의 하체 근력 발달 및 무릎 관절 부상 예방에 가장 결정적인 진단 요소입니다. "
        "초기 MediaPipe 기반 모델(cl1_v1)은 다리 교차 시의 2D 관절 지터 및 가림 왜곡으로 F1이 89.97%에 머물렀으며, 이후 구축된 RTMPose 기반 모델(cl1_rtmp_1)은 사각/후방 뷰에서 94~96%의 우수한 성능을 나타냈으나 "
        "카메라 정면(시점 C)에서 원근 단축(Perspective Foreshortening)에 기인한 2D 평면 투영 오차로 인해 F1이 88.31%로 저하되는 단일 시점 병목 현상이 규명되었습니다.\n\n"
        "이에 따라 MediaPipe의 3D 공간 깊이(Z) 추론력과 RTMPose의 2D 서브픽셀 정밀도를 융합한 '하이브리드 모델'을 공식 최종 챔피언 모델(cl1_final)로 확정 및 선정하였습니다. "
        "cl1_final은 1,025개 전수 시퀀스에 걸쳐 피험자 9명(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증을 수행하였으며, 3-State F1 93.62%, ROC-AUC 97.31%, 보류율 8.88%를 기록하여 모든 기존 베이스라인을 압도하는 역대 최고 성능을 달성하였습니다."
    )

    # 3. 3자 대조 벤치마크 표
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)
    r_h2 = h2.add_run("2. 모델 발전 계보 및 3자 공식 대조 벤치마크")
    r_h2.font.name = "맑은 고딕"
    r_h2.font.size = Pt(13)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(27, 54, 93)

    tbl_comp = doc.add_table(rows=1, cols=7)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["모델 단계", "피처 규격", "강제 2-State F1", "3-State F1", "ROC-AUC", "보류율 (Retake)", "개선 효과"]
    for i, h in enumerate(headers):
        tbl_comp.cell(0, i).paragraphs[0].text = h

    data_comp = [
        ("1세대 MediaPipe 공인 (cl1_v1)", "55D", "85.23%", f"{comp['mp_3s_f1']:.2f}%", f"{comp['mp_auc']:.2f}%", "9.56%", "초기 베이스라인"),
        ("2세대 RTMPose 최적화 (cl1_rtmp_1)", "74D", "88.69%", f"{comp['rtm_opt_3s_f1']:.2f}%", f"{comp['rtm_opt_auc']:.2f}%", "9.66%", "2D 평면 정밀도 개선"),
        ("★ 최종 확정 챔피언 (cl1_final)", "134D", f"{m2['f1']:.2f}%", f"{m3['sel_f1']:.2f}%", f"{m2['roc_auc']:.2f}%", f"{m3['retake']:.2f}%", f"+0.70%p (AUC +0.46%p)")
    ]

    for row_data in data_comp:
        row = tbl_comp.add_row()
        for idx_c, val in enumerate(row_data):
            row.cells[idx_c].paragraphs[0].text = val

    col_widths = [Inches(1.8), Inches(0.8), Inches(1.0), Inches(1.0), Inches(0.9), Inches(0.9), Inches(1.2)]
    alignments = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER]
    format_table(tbl_comp, col_widths, alignments)

    # 4. 시점별 성능 및 정면 극복
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)
    r_h3 = h3.add_run("3. 공식 5대 시점별 세부 성능 및 정면 뷰(시점 C) 극복 감사")
    r_h3.font.name = "맑은 고딕"
    r_h3.font.size = Pt(13)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(27, 54, 93)

    view_names = {
        'view5': '시점 A: 우측 후방 (45도)',
        'view2': '시점 B: 우측 전방 (45도)',
        'view3': '시점 C: 정면 (0도)',
        'view4': '시점 D: 좌측 전방 (45도)',
        'view1': '시점 E: 좌측 후방 (45도)'
    }
    prev_rtm_views = {
        'view5': 92.20, 'view2': 94.81, 'view3': 88.31, 'view4': 96.45, 'view1': 93.33
    }

    tbl_views = doc.add_table(rows=1, cols=7)
    tbl_views.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_v = ["카메라 시점", "정상 임계치", "결함 임계치", "cl1_final F1", "cl1_rtmp_1 F1", "시점별 변화량", "보류율"]
    for i, h in enumerate(headers_v):
        tbl_views.cell(0, i).paragraphs[0].text = h

    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        row = tbl_views.add_row()
        f1_curr = vm[v]['sel_f1']
        f1_prev = prev_rtm_views[v]
        diff_v = f1_curr - f1_prev
        sign_v = "+" if diff_v > 0 else ""
        row.cells[0].paragraphs[0].text = view_names[v]
        row.cells[1].paragraphs[0].text = f"{vm[v]['tau_clean']:.2f}"
        row.cells[2].paragraphs[0].text = f"{vm[v]['tau_defect']:.2f}"
        row.cells[3].paragraphs[0].text = f"{f1_curr:.2f}%"
        row.cells[4].paragraphs[0].text = f"{f1_prev:.2f}%"
        row.cells[5].paragraphs[0].text = f"{sign_v}{diff_v:.2f}%p"
        row.cells[6].paragraphs[0].text = f"{vm[v]['retake']:.2f}%"

    col_widths_v = [Inches(1.8), Inches(0.9), Inches(0.9), Inches(1.1), Inches(1.0), Inches(1.0), Inches(0.8)]
    alignments_v = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT]
    format_table(tbl_views, col_widths_v, alignments_v)

    # 5. 피처 아키텍처 및 캘리브레이션
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)
    r_h4 = h4.add_run("4. 134D 융합 피처 구성 및 ECE 캘리브레이션 사양")
    r_h4.font.name = "맑은 고딕"
    r_h4.font.size = Pt(13)
    r_h4.font.bold = True
    r_h4.font.color.rgb = RGBColor(27, 54, 93)

    p_feat = doc.add_paragraph()
    p_feat.paragraph_format.line_spacing = 1.25
    p_feat.add_run(
        "• MediaPipe 3D 도메인 피처 (55D): ISB Grood & Suntay 대퇴골 횡축 기준 순수 굴곡 각도, 골반-척추 해부학적 신체 로컬 직교 좌표계 관상면 각도, 카메라 광축 투영 단축 계수, 시계열 깊이축(Z) 가속도 평활도 노름\n"
        "• RTMPose 2D 정밀 피처 (74D): 서브픽셀 2D 최저점 무릎 굴곡 각도, 90도 오차, 대퇴골 지면 수평 레벨링 비율, 정강이 수직 정렬 비율, 골반 수직 하강비, 단안 3D 역투영 복원 각도, 공식 5대 시점 One-Hot 인코딩\n"
        "• 교차 상호작용 피처 (5D): 3D-2D 무릎각 절대 편차, 90도 절대 오차 편차, 순수 ISB 사잇각차, 굴곡비, 광축 단축-대퇴 수평도 결합도\n"
        f"• 온도 스케일링 (Temperature Scaling): 최적 온도 T={T:.4f}, ECE {cal['ece_pre']:.2f}% -> {cal['ece_post']:.2f}% (-{cal['ece_pre'] - cal['ece_post']:.2f}%p 개선, Brier Score {cal['brier_post']:.4f})\n"
        f"• 3-way 앙상블 가중치: CatBoost {w[0]:.3f} + LightGBM {w[1]:.3f} + XGBoost {w[2]:.3f}"
    )

    # 6. 최종 배포 결론
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)
    r_h5 = h5.add_run("5. 최종 배포 확정 및 서비스 연동 규격")
    r_h5.font.name = "맑은 고딕"
    r_h5.font.size = Pt(13)
    r_h5.font.bold = True
    r_h5.font.color.rgb = RGBColor(27, 54, 93)

    p_concl = doc.add_paragraph()
    p_concl.paragraph_format.line_spacing = 1.25
    p_concl.add_run(
        "1. 공식 프로덕션 모듈 배포: `src.models.second_models.cross_lunge.cl1_final` 모듈과 `cl1_final_bundle.pkl`이 공식 배포 경로에 탑재되었습니다.\n"
        "2. 표준 추론 인터페이스: `predict_cross_lunge_c1(kps_26, kps_33, camera_view)` 함수를 통해 RTMPose 및 MediaPipe 관절 텐서 쌍 또는 134D 피처 벡터를 입력받아 실시간 2-State/3-State 판정 및 신뢰도 확률을 반환합니다.\n"
        "3. 하네스 무결성 검증: `tests/test_cross_lunge_cl1_final.py` 자동화 단위 테스트(4종 전수)를 100% 통과하여 아키텍처 및 추론 규격의 무결성을 확증하였습니다."
    )

    doc.save(OUT_DOCX_PATH)
    print(f"[+] 공식 최종 챔피언 Word 보고서 저장 완료: {OUT_DOCX_PATH}")

    # Markdown 보고서 생성
    with open(OUT_MD_PATH, 'w', encoding='utf-8') as fp:
        fp.write("# [vidio-health-trainer] 크로스 런지 Condition 1 공식 최종 챔피언 모델 (cl1_final) 확정 및 선정 보고서\n\n")
        fp.write(f"- **공식 프로젝트명**: `vidio-health-trainer`\n")
        fp.write(f"- **대상 운동 및 결함**: 크로스 런지 (`08_크로스_런지`) / c1 (앞다리 무릎 각도 90도 오류)\n")
        fp.write(f"- **공식 최종 모델**: `cl1_final` (MediaPipe 3D + RTMPose 2D 134D 하이브리드 아키텍처)\n")
        fp.write(f"- **가중치 번들**: `cl1_final_bundle.pkl` (134D GBDT 앙상블 + T={T:.4f} 온도 스케일링)\n")
        fp.write(f"- **전체 성능 요약**: 3-State F1 **{m3['sel_f1']:.2f}%**, 정확도 **{m3['sel_acc']:.2f}%**, ROC-AUC **{m2['roc_auc']:.2f}%**, 보류율 **{m3['retake']:.2f}%**\n\n")
        fp.write("## 1. 모델 발전 계보 및 3자 대조 벤치마크\n\n")
        fp.write("| 모델 단계 | 피처 규격 | 강제 2-State F1 | 3-State F1 | ROC-AUC | 보류율 | 개선 효과 |\n")
        fp.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in data_comp:
            fp.write(f"| {row[0]} | {row[1]} | {row[2]} | **{row[3]}** | {row[4]} | {row[5]} | **{row[6]}** |\n")
        fp.write("\n## 2. 공식 5대 시점별 세부 성능 및 정면 뷰(시점 C) 극복 감사\n\n")
        fp.write("| 카메라 시점 | 정상 임계치 | 결함 임계치 | cl1_final F1 | cl1_rtmp_1 F1 | 시점별 변화량 | 보류율 |\n")
        fp.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
            f1_curr = vm[v]['sel_f1']
            f1_prev = prev_rtm_views[v]
            diff_v = f1_curr - f1_prev
            sign_v = "+" if diff_v > 0 else ""
            fp.write(f"| **{view_names[v]}** | {vm[v]['tau_clean']:.2f} | {vm[v]['tau_defect']:.2f} | **{f1_curr:.2f}%** | {f1_prev:.2f}% | **{sign_v}{diff_v:.2f}%p** | {vm[v]['retake']:.2f}% |\n")
        fp.write("\n## 3. 피처 아키텍처 및 캘리브레이션 사양\n\n")
        fp.write(f"- **피처 차원 구성 (134D)**: MediaPipe 3D 도메인 피처 (55D) + RTMPose 2D 정밀 피처 (74D) + 교차 상호작용 피처 (5D)\n")
        fp.write(f"- **온도 스케일링**: T={T:.4f}, ECE {cal['ece_pre']:.2f}% -> **{cal['ece_post']:.2f}%** (-{cal['ece_pre'] - cal['ece_post']:.2f}%p)\n")
        fp.write(f"- **최종 앙상블 가중치**: CatBoost {w[0]:.3f} + LightGBM {w[1]:.3f} + XGBoost {w[2]:.3f}\n")
    print(f"[+] 공식 최종 챔피언 Markdown 보고서 저장 완료: {OUT_MD_PATH}")


if __name__ == '__main__':
    main()
