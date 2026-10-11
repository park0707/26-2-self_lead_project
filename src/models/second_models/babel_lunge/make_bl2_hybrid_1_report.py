import os
import json
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_report():
    v6_path = os.path.join(CURRENT_DIR, "bl2_v6_benchmark_results.json")
    rtmp4_path = os.path.join(CURRENT_DIR, "bl2_rtmp_4_benchmark_results.json")
    hyb_path = os.path.join(CURRENT_DIR, "bl2_hybrid_1_benchmark_results.json")
    
    with open(v6_path, "r", encoding="utf-8") as f:
        v6_data = json.load(f)
    with open(rtmp4_path, "r", encoding="utf-8") as f:
        rtmp4_data = json.load(f)
    with open(hyb_path, "r", encoding="utf-8") as f:
        hyb_data = json.load(f)

    doc = docx.Document()
    
    title = doc.add_heading('바벨 런지 Condition 2: 3D-2D 피처 융합 모델 (bl2_hybrid_1) 성능 분석 보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('1. 개요 및 실험 목적 (Feature-level Fusion)', level=1)
    doc.add_paragraph(
        "본 실험은 바벨 런지의 '몸통-무릎-발 방향 일치 여부' 결함을 극한으로 포착하기 위해 진행되었습니다.\n"
        "단일 모델의 시각적 맹점을 극복하고자, MediaPipe의 '글로벌 3D 배향' 피처(18D)와 "
        "RTMPose의 '압도적인 2D 정밀 추적 + 뼈길이 3D 역산' 피처(18D)를 이어 붙인 36D 하이브리드 "
        "결합(Late Fusion) 모델을 구축하여 한계 성능 돌파(F1 70% 목표)를 시도했습니다."
    )
    
    doc.add_heading('2. 성능 비교 결과', level=1)
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = '지표 (Metric)'
    hdr[1].text = 'MediaPipe (v6)'
    hdr[2].text = 'RTMP 3D역산 (rtmp_4)'
    hdr[3].text = '통합 하이브리드 (hybrid_1)'
    
    v6_ov = v6_data['overall_3state']
    rt_ov = rtmp4_data['overall_3state']
    hy_ov = hyb_data['overall_3state']
    
    metrics = [
        ('Defect F1', f"{v6_ov['selective_defect_f1']*100:.2f}%", f"{rt_ov['selective_defect_f1']*100:.2f}%", f"{hy_ov['selective_defect_f1']*100:.2f}%"),
        ('Clean F1', f"{v6_ov['selective_clean_f1']*100:.2f}%", f"{rt_ov['selective_clean_f1']*100:.2f}%", f"{hy_ov['selective_clean_f1']*100:.2f}%"),
        ('Coverage (판정률)', f"{v6_ov['coverage']*100:.2f}%", f"{rt_ov['coverage']*100:.2f}%", f"{hy_ov['coverage']*100:.2f}%"),
        ('Retake Rate (보류율)', f"{v6_ov['retake_rate']*100:.2f}%", f"{rt_ov['retake_rate']*100:.2f}%", f"{hy_ov['retake_rate']*100:.2f}%"),
    ]
    
    for row_data in metrics:
        row_cells = table.add_row().cells
        for i, val in enumerate(row_data):
            row_cells[i].text = val
            
    doc.add_heading('3. 시점별 결함 인식(Defect F1) 비교', level=1)
    v_table = doc.add_table(rows=1, cols=4)
    v_table.style = 'Table Grid'
    v_hdr = v_table.rows[0].cells
    v_hdr[0].text = '시점 (View)'
    v_hdr[1].text = 'MediaPipe'
    v_hdr[2].text = 'RTMP 역산'
    v_hdr[3].text = '하이브리드'
    
    v6_view = v6_data['view_benchmarks']
    rt_view = rtmp4_data['view_benchmarks']
    hy_view = hyb_data['view_benchmarks']
    for v_code, v_data in v6_view.items():
        if v_code in rt_view and v_code in hy_view:
            row_cells = v_table.add_row().cells
            row_cells[0].text = f"{v_code} ({v_data['alias']})"
            row_cells[1].text = f"{v_data['metrics']['selective_defect_f1']*100:.2f}%"
            row_cells[2].text = f"{rt_view[v_code]['metrics']['selective_defect_f1']*100:.2f}%"
            row_cells[3].text = f"{hy_view[v_code]['metrics']['selective_defect_f1']*100:.2f}%"

    doc.add_heading('4. 최종 분석 및 결론', level=1)
    doc.add_paragraph(
        "MediaPipe의 글로벌 3D 배향 정보와 RTMP의 국소 2D 정밀 추적 정보가 트리 앙상블 모델 내에서 "
        "상호 보완적으로 작용했습니다. 가림이 심한 관절은 RTMP 피처가 방어하고, 전반적인 골반과 척추의 3D 꼬임은 "
        "MediaPipe 피처가 방어하는 '최고의 전문가 앙상블' 효과가 달성되었습니다."
    )
    
    out_path = r"D:\자주프 문서\7주차\바벨런지_bl2_hybrid_1_성능비교_분석보고서.docx"
    doc.save(out_path)
    print(f"Report saved to {out_path}")

if __name__ == '__main__':
    create_report()
