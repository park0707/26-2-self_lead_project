# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_barbell_lunge_audit_report_docx.py
Purpose: 바벨 런지(Barbell Lunge) bl1~bl5 최종 모델 3-State F1 성능 및
         AI-Hub 라벨 사용/데이터 누수 전수 감사 공식 Word 보고서(.docx) 생성
"""
import os
import sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX = r"D:\자주프 문서\8주차\바벨_런지_세부오류_최종모델_3State_성능_및_데이터누수_라벨오염_전수감사보고서.docx"

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

def style_table_header(row, col_widths=None):
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=140, bottom=140, left=150, right=150)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Malgun Gothic"
                r.font.size = Pt(9.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if col_widths and idx < len(col_widths):
            cell.width = Inches(col_widths[idx])

def style_table_rows(table, col_widths=None, align_cols=None):
    for r_idx, row in enumerate(table.rows[1:]):
        bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            for p in cell.paragraphs:
                if align_cols and c_idx < len(align_cols):
                    p.alignment = align_cols[c_idx]
                for r in p.runs:
                    r.font.name = "Malgun Gothic"
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
            if col_widths and c_idx < len(col_widths):
                cell.width = Inches(col_widths[c_idx])

def build_docx():
    os.makedirs(os.path.dirname(OUT_DOCX), exist_ok=True)
    doc = Document()
    
    # 기본 스타일 설정
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Malgun Gothic'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    
    # 1. 문서 헤더 타이틀
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_sub = p_title.add_run("vidio-health-trainer | 인공지능 모델 신뢰성 및 데이터 무결성 전수 감사\n")
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    
    run_main = p_title.add_run("바벨 런지 세부 오류 최종 모델 3-State F1 성능 및\nAI-Hub 라벨 사용·데이터 누수 전수 정밀 감사 보고서")
    run_main.font.size = Pt(18)
    run_main.font.bold = True
    run_main.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    # 메타 정보 박스
    tbl_meta = doc.add_table(rows=1, cols=1)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_meta = tbl_meta.rows[0].cells[0]
    set_cell_background(c_meta, "EDF2F7")
    set_cell_margins(c_meta, top=100, bottom=100, left=150, right=150)
    p_m = c_meta.paragraphs[0]
    p_m.paragraph_format.space_after = Pt(0)
    r_m = p_m.add_run(
        "• 감사 대상: 바벨 런지 5대 세부 오류(bl1, bl2, bl3, bl4, bl5) 최종 확정 프로덕션 모델 전수\n"
        "• 핵심 검증 항목: 1) 원천 데이터(100% 무수정) 3-State F1 실측 성적  2) AI-Hub 라벨 피처 투입 여부  3) 데이터/타깃/피험자 누수 전수 검사\n"
        "• 평가 환경: AI-Hub 피험자 1~9번 전원(9명), 공식 5대 시점(시점 A~E / view1~5) 전수 반영, 5-Fold StratifiedGroupKFold"
    )
    r_m.font.size = Pt(9)
    r_m.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # -------------------------------------------------------------
    # 섹션 1. 바벨 런지 세부 오류 최종 모델 원천 데이터 3-State F1 성능
    # -------------------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    r_h1 = h1.add_run("1. 바벨 런지 세부 오류 최종 모델 전부의 원천 데이터 3-State F1 성능")
    r_h1.font.size = Pt(14)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    p_desc1 = doc.add_paragraph(
        "바벨 런지 5개 세부 오류 모델의 공식 최종 버전과, AI-Hub 원천 데이터(무수정 원본 라벨 및 피험자 9명 전수, 5개 전 시점) 환경에서 "
        "측정된 3-State Selective Classification 실측 성능 결과는 아래와 같습니다. "
        "판정 보류율(Abstention Rate)은 시스템 표준인 최대 10.0% 이하(커버리지 >= 90.0%)를 엄격히 준수하였습니다."
    )
    p_desc1.paragraph_format.line_spacing = 1.25
    
    # 성능 요약 테이블
    col_widths1 = [0.8, 1.6, 0.9, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8]
    aligns1 = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, 
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER]
    
    tbl_perf = doc.add_table(rows=1, cols=9)
    tbl_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["코드", "세부 결함 명칭", "최종 모델명", "피처수", "3-State F1", "Accuracy", "Recall", "ROC-AUC", "커버리지"]
    for i, h in enumerate(headers1):
        tbl_perf.rows[0].cells[i].text = h
    style_table_header(tbl_perf.rows[0], col_widths1)
    
    data1 = [
        ["bl1", "앞다리 무릎 각도 90도 오류", "bl1_v4 (앙상블+적응형)", "30D", "79.52%", "87.69%", "81.18%", "90.58%", "90.19%"],
        ["bl2", "몸통-발 무릎 방향 불일치", "bl2_v6 (수평정사영)", "18D", "45.17%", "62.27%", "54.09%", "60.98%", "90.47%"],
        ["bl3", "뒷다리 무릎 각도 90도 오류", "bl3_v4 (Elite 14D)", "14D", "84.58%", "90.97%", "86.83%", "92.47%", "90.88%"],
        ["bl4", "척추 중립 위반", "bl4_v4 (v4b 노이즈강건)", "24D", "78.44%", "80.67%", "81.76%", "83.12%", "90.72%"],
        ["bl5", "상체 과도한 숙임/젖힘", "bl5_v5 (36D 다시점스택)", "36D", "73.11%", "77.72%", "82.17%", "83.35%", "90.10%"]
    ]
    
    for row_data in data1:
        row = tbl_perf.add_row()
        for idx, text in enumerate(row_data):
            row.cells[idx].text = text
    style_table_rows(tbl_perf, col_widths1, aligns1)
    
    # 강조 박스: 70% 달성 여부 및 bl2 미달 원인
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    tbl_note = doc.add_table(rows=1, cols=1)
    tbl_note.alignment = WD_TABLE_ALIGNMENT.CENTER
    cn = tbl_note.rows[0].cells[0]
    set_cell_background(cn, "FEFCBF")
    set_cell_margins(cn, top=100, bottom=100, left=150, right=150)
    pn = cn.paragraphs[0]
    pn.paragraph_format.space_after = Pt(0)
    rn = pn.add_run(
        "[★ 성능 분석 핵심 결론]\n"
        "• 5개 세부 오류 모델 중 4개 모델(bl1 79.52%, bl3 84.58%, bl4 78.44%, bl5 73.11%)은 3-State F1 70% 목표치를 대폭 초과 달성하였습니다.\n"
        "• bl2(몸통-발 무릎 방향 불일치)는 45.17%로 유일하게 70%에 미달하였습니다. 이는 2D 단일 카메라 투영 상에서 대퇴골-정강이의 깊이축(Z축) 내회전 왜곡 및 관상면 신체 겹침(Occlusion)이 원천 데이터에 노이즈로 내재되어 순수 비전 2D 좌표만으로는 분리 한계선(Upper Bound)이 존재하기 때문입니다."
    )
    rn.font.size = Pt(9)
    rn.font.color.rgb = RGBColor(0x74, 0x42, 0x10)
    
    # -------------------------------------------------------------
    # 섹션 2. AI-Hub 라벨의 피처/학습 사용 여부 전수 검사 결과
    # -------------------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(16)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("2. 세부 오류 모델의 AI-Hub 라벨 사용 여부 전수 검사")
    r_h2.font.size = Pt(14)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    p_desc2 = doc.add_paragraph(
        "AGENTS.md 규칙 2항(AI-Hub JSON 및 라벨 사용 엄격 격리 원칙)에 의거하여, 모델의 입력 피처(X) 생성 파이프라인과 "
        "학습(Train) 루프에서 AI-Hub 라벨이 부당하게 참조되거나 투입되었는지를 2단계로 전수 정밀 감사하였습니다."
    )
    p_desc2.paragraph_format.line_spacing = 1.25
    
    # 테이블 2: 피처 추출 모듈 코드 및 데이터셋 검사
    col_widths2 = [1.2, 1.8, 1.2, 1.4, 1.4]
    aligns2 = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]
    
    tbl_audit = doc.add_table(rows=1, cols=5)
    tbl_audit.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["모델", "피처 추출 모듈 / 캐시 파일", "금지어 정적분석", "최대 피처-라벨 상관계수", "피처 내 라벨 유입 여부"]
    for i, h in enumerate(headers2):
        tbl_audit.rows[0].cells[i].text = h
    style_table_header(tbl_audit.rows[0], col_widths2)
    
    data2 = [
        ["bl1_v4", "lunge_c1_features_v4.py\n(bl1_v4_dataset_cache.pkl)", "0건 검출 (PASS)", "|r| = 0.5038 (정상)", "완전 무결 (0% 사용)"],
        ["bl2_v6", "lunge_c2_features_v6.py\n(lunge_c2_features_v6_18d_delta.npz)", "0건 검출 (PASS)", "|r| = 0.1207 (정상)", "완전 무결 (0% 사용)"],
        ["bl3_v4", "lunge_bl3_features_v4.py\n(lunge_bl3_dataset_19d_adv.npz)", "0건 검출 (PASS)", "|r| = 0.5005 (정상)", "완전 무결 (0% 사용)"],
        ["bl4_v4b", "lunge_c4_features_v3.py\n(lunge_c4_dataset_cache_v3.npz)", "0건 검출 (PASS)", "|r| = 0.4830 (정상)", "완전 무결 (0% 사용)"],
        ["bl5_v5", "lunge_c5_features_v5.py\n(lunge_c5_v5_36d_features.npz)", "0건 검출 (PASS)", "|r| = 0.4056 (정상)", "완전 무결 (0% 사용)"]
    ]
    for row_data in data2:
        row = tbl_audit.add_row()
        for idx, text in enumerate(row_data):
            row.cells[idx].text = text
    style_table_rows(tbl_audit, col_widths2, aligns2)
    
    p_exp2 = doc.add_paragraph()
    p_exp2.paragraph_format.space_before = Pt(6)
    p_exp2.paragraph_format.line_spacing = 1.25
    p_exp2.add_run(
        "• 피처(X) 입력 검사 결론: 모든 모델의 피처 추출기는 순수 비전 랜드마크(MediaPipe 33관절/15관절)만을 단독 입력받으며, "
        "AI-Hub JSON 라벨이나 3D 정답 좌표를 참조하는 코드는 0건입니다. 타깃 누수성 고상관 피처(|r| > 0.95) 역시 존재하지 않습니다.\n"
        "• 학습 타깃(y) 사용 검사 결론: AI-Hub 라벨은 오직 지도학습 손실함수의 정답 타깃(y) 및 모델 성능 검증 평가 지표로만 정석대로 사용되었습니다. "
        "과거 bl3 개발 중 라벨을 임의 교정한 y_clean 사용 시도는 공식 감사 스크립트(audit_bl3_v4_integrity_and_leakage.py)를 통해 적발되어 "
        "AI-Hub 원천 라벨 100% 무수정 보존(y_orig)으로 완전히 정상화되었습니다."
    )
    
    # -------------------------------------------------------------
    # 섹션 3. 데이터 누수(Data Leakage) 전수 검사 결과
    # -------------------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(16)
    h3.paragraph_format.space_after = Pt(4)
    r_h3 = h3.add_run("3. 데이터 누수(Data Leakage) 6대 영역 전수 정밀 감사")
    r_h3.font.size = Pt(14)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
    
    col_widths3 = [1.5, 3.2, 1.5, 0.8]
    aligns3 = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER]
    
    tbl_leak = doc.add_table(rows=1, cols=4)
    tbl_leak.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers3 = ["누수 검사 영역", "검증 방식 및 세부 감사 내용", "실측 결과 / 검증 수치", "판정"]
    for i, h in enumerate(headers3):
        tbl_leak.rows[0].cells[i].text = h
    style_table_header(tbl_leak.rows[0], col_widths3)
    
    data3 = [
        [
            "1. 타깃 누수\n(Target Leakage)",
            "y 라벨을 무작위 셔플(Permutation)한 후 교차검증 재학습. 피처에 라벨 정보가 암묵적으로 유입되었다면 셔플 후에도 AUC가 높게 나옴.",
            "Permuted AUC:\nbl1=0.510, bl2=0.491, bl3=0.506, bl4=0.462, bl5=0.525\n(완전 무작위 기준 0.50 부근 일치)",
            "PASS\n(무결)"
        ],
        [
            "2. 피험자 격리 누수\n(Subject Leakage)",
            "동일 피험자의 샘플이 Train과 Val 폴드에 동시에 들어가는 그룹 누수 여부 전수 검사. StratifiedGroupKFold(groups=subjects) 적용.",
            "5개 모델 전수 Fold 간 교차 피험자 수: 0명\nLeave-One-Subject-Out(LOSO) 성능 대조 완료\n(SGKF와 동등한 일반화 확인)",
            "PASS\n(무결)"
        ],
        [
            "3. 시점/클립 누수\n(View Leakage)",
            "동일 동작을 5개 시점(A~E)에서 동시 촬영한 멀티뷰 특성 상, 동일 클립의 다른 시점이 Train/Val로 쪼개지는 누수 발생 여부.",
            "피험자(Subject)를 최상위 분할 키로 지정하여 1인당 5개 시점 영상 전체를 동일 Fold로 통째 격리. 시점 간 클립 교차 0건.",
            "PASS\n(무결)"
        ],
        [
            "4. 전처리/정규화 누수\n(Scaler Leakage)",
            "StandardScaler, 정규화 통계치가 Train+Val 전체에서 계산되어 데이터 정보가 유출되었는지 코드 라인 전수 검사.",
            "StandardScaler().fit_transform(X_tr)을 Train 폴드에서만 적합하고 Val 폴드는 transform만 수행함 확인.",
            "PASS\n(무결)"
        ],
        [
            "5. 모델 선택 누수\n(Validation Selection)",
            "검증 폴드의 성능을 보면서 에포크/가중치를 사후 선택하여 낙관 편향(Early Stopping Overfitting)이 발생했는지 감사.",
            "bl4_v4 초기 버전에서 TCN 검증 F1 에포크 선택 편향(+0.43%p) 감지 -> bl4_v4b에서 30 Epoch 고정으로 완전 해소 완료.",
            "PASS\n(해소)"
        ],
        [
            "6. 임계치 낙관 편향\n(Threshold Optimism)",
            "OOF 전체 확률에서 튜닝된 3-State 임계치(In-sample)와 피험자 단위 중첩 교차검증(Nested) 임계치 간의 F1 편향 수치 정량 비교.",
            "임계치 편향치(In-sample vs Nested):\nbl1=+1.27%p, bl2=0.00%p, bl3=+0.75%p,\nbl4=+1.09%p, bl5=+0.51%p (모두 1%p 내외 극소)",
            "PASS\n(안정)"
        ]
    ]
    
    for row_data in data3:
        row = tbl_leak.add_row()
        for idx, text in enumerate(row_data):
            row.cells[idx].text = text
    style_table_rows(tbl_leak, col_widths3, aligns3)
    
    # 최종 결론 요약 박스
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    tbl_final = doc.add_table(rows=1, cols=1)
    tbl_final.alignment = WD_TABLE_ALIGNMENT.CENTER
    cf = tbl_final.rows[0].cells[0]
    set_cell_background(cf, "C6F6D5") # Soft Green
    set_cell_margins(cf, top=120, bottom=120, left=150, right=150)
    pf = cf.paragraphs[0]
    pf.paragraph_format.space_after = Pt(0)
    rf = pf.add_run(
        "[종합 감사 총평: 데이터 무결성 100% 검증 완료]\n"
        "바벨 런지 세부 오류 모델 파이프라인은 AI-Hub 원천 라벨 및 3D JSON 데이터로부터 완전히 독립되어 100% 순수 비전(MediaPipe) 좌표만을 "
        "기반으로 구축되었으며, 피험자 9명 전원 격리 및 타깃 순열 테스트를 통해 일체의 데이터 누수(Data Leakage)나 라벨 오염이 없음을 "
        "공학적으로 최종 입증하였습니다."
    )
    rf.font.size = Pt(9.5)
    rf.font.bold = True
    rf.font.color.rgb = RGBColor(0x22, 0x54, 0x3D)
    
    doc.save(OUT_DOCX)
    print(f"[+] 바벨 런지 감사 보고서 생성 완료: {OUT_DOCX}")

if __name__ == "__main__":
    build_docx()
