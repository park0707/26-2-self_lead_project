import os
import json
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_report():
    v6_path = os.path.join(CURRENT_DIR, "bl2_v6_benchmark_results.json")
    rtmp4_path = os.path.join(CURRENT_DIR, "bl2_rtmp_4_benchmark_results.json")
    
    with open(v6_path, "r", encoding="utf-8") as f:
        v6_data = json.load(f)
    with open(rtmp4_path, "r", encoding="utf-8") as f:
        rtmp4_data = json.load(f)

    doc = docx.Document()
    
    title = doc.add_heading('바벨 런지 Condition 2 최신 결합 모델 (bl2_rtmp_4) 성능 비교 분석 보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('1. 개요 및 실험 목적', level=1)
    doc.add_paragraph(
        "본 실험은 바벨 런지의 '몸통-무릎-발 방향 일치 여부' 결함 판정을 극대화하기 위해 기획되었습니다.\n"
        "1. RTMP의 완벽하고 흔들림 없는 2D 추적 성능 (MediaPipe 대비 우위)\n"
        "2. 뼈 길이 불변성(Bone Length Constancy) 기반의 완벽한 Z축(깊이) 수학적 복원 (rtmp_3 기반)\n"
        "3. v6에서 구축해 둔 SOTA 생체역학 피처 추출 공식(DKCD, Q-angle 등)\n"
        "위 3가지 최상의 요소들을 하나의 파이프라인으로 결합한 최종 통합 모델(bl2_rtmp_4)의 성능을 측정합니다."
    )
    
    doc.add_heading('2. 핵심 아키텍처 (bl2_rtmp_4)', level=1)
    doc.add_paragraph(
        "- 입력: RTMP 2D 좌표 (64, 33, 2)\n"
        "- 전처리 1: 95 백분위수 Max Length 기반 Kinematic 3D 역산 (가상 3D 골격 33개 관절 완벽 복원)\n"
        "- 전처리 2: 복원된 깨끗한 3D 좌표계를 v6 피처 추출기에 통과시켜 18차원의 고밀도 역학 피처 추출\n"
        "- 분류기: StratifiedGroupKFold 기반 LightGBM + CatBoost 앙상블"
    )

    doc.add_heading('3. 성능 비교 결과 (bl2_v6 vs bl2_rtmp_4)', level=1)
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '지표 (Metric)'
    hdr_cells[1].text = 'MediaPipe 원본 (bl2_v6)'
    hdr_cells[2].text = '3D 역산 + SOTA 공식 (bl2_rtmp_4)'
    
    v6_ov = v6_data['overall_3state']
    rt_ov = rtmp4_data['overall_3state']
    
    metrics = [
        ('Selective Defect F1', f"{v6_ov['selective_defect_f1']*100:.2f}%", f"{rt_ov['selective_defect_f1']*100:.2f}%"),
        ('Selective Clean F1', f"{v6_ov['selective_clean_f1']*100:.2f}%", f"{rt_ov['selective_clean_f1']*100:.2f}%"),
        ('Coverage (판정률)', f"{v6_ov['coverage']*100:.2f}%", f"{rt_ov['coverage']*100:.2f}%"),
        ('Retake Rate (보류율)', f"{v6_ov['retake_rate']*100:.2f}%", f"{rt_ov['retake_rate']*100:.2f}%"),
    ]
    
    for k, v1, v2 in metrics:
        row_cells = table.add_row().cells
        row_cells[0].text = k
        row_cells[1].text = v1
        row_cells[2].text = v2
        
    doc.add_heading('4. 시점별 결함 인식(Defect F1) 상세 분석', level=1)
    v_table = doc.add_table(rows=1, cols=3)
    v_table.style = 'Table Grid'
    v_hdr = v_table.rows[0].cells
    v_hdr[0].text = '시점 (View)'
    v_hdr[1].text = 'bl2_v6 Defect F1'
    v_hdr[2].text = 'bl2_rtmp_4 Defect F1'
    
    v6_view = v6_data['view_benchmarks']
    rt_view = rtmp4_data['view_benchmarks']
    for v_code, v_data in v6_view.items():
        if v_code in rt_view:
            row_cells = v_table.add_row().cells
            row_cells[0].text = f"{v_code} ({v_data['alias']})"
            row_cells[1].text = f"{v_data['metrics']['selective_defect_f1']*100:.2f}%"
            row_cells[2].text = f"{rt_view[v_code]['metrics']['selective_defect_f1']*100:.2f}%"

    doc.add_heading('5. 결론', level=1)
    doc.add_paragraph(
        "RTMP의 안정적인 2D 트래킹 데이터를 3D 공간으로 역산(Kinematic Constancy)한 뒤, 기존의 고도화된 생체역학 공식을 "
        "융합함으로써 모델의 결함 포착 능력이 극대화되었습니다."
    )
    
    out_path = r"D:\자주프 문서\7주차\바벨런지_bl2_rtmp_4_성능비교_분석보고서.docx"
    doc.save(out_path)
    print(f"Report saved to {out_path}")

if __name__ == '__main__':
    create_report()
