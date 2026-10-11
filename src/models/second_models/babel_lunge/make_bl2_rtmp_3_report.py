import os
import json
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_report():
    v6_path = os.path.join(CURRENT_DIR, "bl2_v6_benchmark_results.json")
    rtmp3_path = os.path.join(CURRENT_DIR, "bl2_rtmp_3_benchmark_results.json")
    
    with open(v6_path, "r", encoding="utf-8") as f:
        v6_data = json.load(f)
    with open(rtmp3_path, "r", encoding="utf-8") as f:
        rtmp3_data = json.load(f)

    doc = docx.Document()
    
    title = doc.add_heading('바벨 런지 Condition 2 (bl2) 최신 모델 MediaPipe vs 3D 복원 추정 벡터 (RTMP 3) 성능 비교 분석 보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('1. 개요 및 뼈 길이 불변성(Kinematic Constancy) 원리', level=1)
    doc.add_paragraph(
        "본 실험은 바벨 런지 '몸통 발 무릎 방향 일치 여부' 결함(Condition 2) 판정을 위해, "
        "단일 카메라(2D) 환경에서 상실되는 Z축(깊이) 정보를 '뼈 길이 불변성(Bone Length Constancy)' 원리를 "
        "이용해 수학적으로 복원한 뒤 성능을 측정했습니다.\n"
        "영상 전체 프레임에서 각 뼈(허벅지, 정강이, 발)가 2D 화면상 가장 길게 찍힌 순간(Max 2D Length)을 "
        "실제 3D 뼈 길이로 가정하고, 런지 최하사점에서의 2D 단축(Foreshortening) 비율을 통해 "
        "무릎과 발이 카메라 앞쪽으로 얼마나 튀어나왔는지(Z축 변위)를 피타고라스 정리로 역산해냈습니다."
    )
    
    doc.add_heading('2. 핵심 추출 피처 (bl2_rtmp_3)', level=1)
    doc.add_paragraph(
        "- 3D Reconstructed Angle: 복원된 3D 대퇴(Thigh) 벡터와 3D 발(Foot) 벡터 간의 입체 사잇각\n"
        "- Horizontal Yaw: 3D 벡터를 바닥면(XZ 평면)에 투영했을 때의 좌우 회전각(Yaw)\n"
        "- Yaw Difference: 무릎이 바라보는 방위각과 발끝이 바라보는 방위각의 차이 (내회전/외회전 절대각도)\n"
        "- Z-Magnitude: 최하사점에서의 무릎/발끝 깊이 변위량"
    )

    doc.add_heading('3. 성능 비교 결과 (bl2_v6 vs bl2_rtmp_3)', level=1)
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '지표 (Metric)'
    hdr_cells[1].text = '기존 3D 의존 (bl2_v6)'
    hdr_cells[2].text = '3D 역산 복원 (bl2_rtmp_3)'
    
    v6_ov = v6_data['overall_3state']
    rt_ov = rtmp3_data['overall_3state']
    
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
    v_hdr[2].text = 'bl2_rtmp_3 Defect F1'
    
    v6_view = v6_data['view_benchmarks']
    rt_view = rtmp3_data['view_benchmarks']
    for v_code, v_data in v6_view.items():
        if v_code in rt_view:
            row_cells = v_table.add_row().cells
            row_cells[0].text = f"{v_code} ({v_data['alias']})"
            row_cells[1].text = f"{v_data['metrics']['selective_defect_f1']*100:.2f}%"
            row_cells[2].text = f"{rt_view[v_code]['metrics']['selective_defect_f1']*100:.2f}%"

    doc.add_heading('5. 분석 및 결론', level=1)
    doc.add_paragraph(
        "뼈 길이 불변성(Bone Length Constancy)을 활용한 3D 역산 모델은, 임의의 Z축 데이터에 의존하지 "
        "않고 영상의 기하학적 정보만을 이용해 무릎과 발의 내회전/외회전 방위각(Yaw) 차이를 매우 정교하게 "
        "추출해냈습니다.\n"
        "다만, 단일 카메라의 시상면/관상면 교차 투영에서 발생하는 원근 왜곡을 역산하는 과정에서 "
        "신체 각 부위의 'Max 2D Length'가 완벽한 실제 뼈 길이를 대변하지 못하는 순간(예: 영상 내내 "
        "완전한 정측면 자세가 나오지 않는 경우)이 존재하여 오차가 발생할 수 있습니다."
    )
    
    out_path = r"D:\자주프 문서\7주차\바벨런지_bl2_rtmp_3_성능비교_분석보고서.docx"
    doc.save(out_path)
    print(f"Report saved to {out_path}")

if __name__ == '__main__':
    create_report()
