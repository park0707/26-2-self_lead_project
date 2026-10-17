# -*- coding: utf-8 -*-
"""
vidio-health-trainer
일반화 데이터셋 웹 모델 예측 결과 및 사용자 직접 채점표 생성 스크립트
"""

import os
import json
import csv
import shutil

BENCHMARK_JSON = r"D:\data\일반화\generalization_benchmark_results.json"
OUTPUT_DIR = r"D:\data\일반화"
CLIPS_BASE_DIR = r"D:\data\일반화\분할_결과"

# 결함 명칭 및 세부 매핑
DEFECT_META = {
    "바벨 데드": {
        "d1": {"name": "척추의 중립 위반", "eng": "Spine Neutrality", "user_num": "결함 1"},
        "d2": {"name": "무릎-발목 정렬 모임/외반", "eng": "Knee Medial Collapse", "user_num": "결함 2"},
        "d5": {"name": "바벨 궤적과 몸 밀착 위반", "eng": "Barbell Path Proximity", "user_num": "결함 3"},
        "d4": {"name": "무릎과 골반 동시 신전 위반", "eng": "Knee-Hip Desynchronization", "user_num": "결함 4"},
        "d3": {"name": "발바닥 지면 고정 / 힙 슈트업", "eng": "Foot Ground Fixation", "user_num": "결함 5"},
        "order": ["d1", "d2", "d3", "d4", "d5"]
    },
    "스쿼트": {
        "c1": {"name": "척추의 중립 위반", "eng": "Spine Neutrality", "user_num": "결함 1"},
        "c2": {"name": "고개 정면 및 시선 이탈", "eng": "Head & Gaze Orientation", "user_num": "결함 2"},
        "c3": {"name": "무릎-발 방향 일치 위반", "eng": "Knee-Toe Alignment", "user_num": "결함 3"},
        "c4": {"name": "발바닥 지면 고정 위반", "eng": "Foot Grounding", "user_num": "결함 4"},
        "order": ["c1", "c2", "c3", "c4"]
    },
    "스티프 데드": {
        "sd3": {"name": "바벨 궤적과 몸 밀착 위반", "eng": "Barbell Path Proximity", "user_num": "결함 1"},
        "sd1": {"name": "무릎 구부린 채 고정 위반", "eng": "Knee Fixation Violation", "user_num": "결함 2"},
        "sd2": {"name": "척추의 중립 위반", "eng": "Spine Neutrality", "user_num": "결함 3"},
        "order": ["sd1", "sd2", "sd3"]
    }
}

VIEW_FULL_NAME = {
    "시점 A": "시점 A (우측 후방 45° / view5)",
    "시점 B": "시점 B (우측 전방 45° / view2)",
    "시점 C": "시점 C (정면 0° / view3)",
    "시점 D": "시점 D (좌측 전방 45° / view4)",
    "시점 E": "시점 E (좌측 후방 45° / view1)",
}

def get_intent_label(exercise, true_dict):
    true_keys = [k for k, v in true_dict.items() if v.get("true", 0) == 1]
    if not true_keys:
        return "[정상] 완벽한 자세 수행"
    
    meta = DEFECT_META.get(exercise, {})
    labels = []
    for k in true_keys:
        info = meta.get(k, {})
        num = info.get("user_num", k)
        nm = info.get("name", k)
        labels.append(f"[{num}] {nm} ({k})")
    return " & ".join(labels)

def get_model_summary(predictions_raw):
    # state: 0=정상, 1=결함, 2=주의/보류
    defects_detected = []
    warnings_detected = []
    
    for k, v in predictions_raw.items():
        state = v.get("state", 0)
        prob = v.get("prob", 0.0)
        nm = v.get("name", k).split("(")[0].strip()
        if state == 1:
            defects_detected.append(f"[{k}] {nm} ({prob*100:.1f}%, 결함)")
        elif state == 2:
            warnings_detected.append(f"[{k}] {nm} ({prob*100:.1f}%, 주의)")
            
    if defects_detected:
        overall = "결함 감지"
    elif warnings_detected:
        overall = "주의/경고"
    else:
        overall = "정상"
        
    all_notices = defects_detected + warnings_detected
    summary_str = " | ".join(all_notices) if all_notices else "결함 없음 (전 부위 정상)"
    return overall, summary_str

def get_all_probs_str(exercise, predictions_raw):
    order = DEFECT_META.get(exercise, {}).get("order", list(predictions_raw.keys()))
    parts = []
    for k in order:
        v = predictions_raw.get(k, {})
        prob = v.get("prob", 0.0)
        state_label = v.get("state_label", "정상")
        parts.append(f"{k}:{prob*100:.1f}%({state_label})")
    return " | ".join(parts)

def build_sheets():
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    print(f"Loaded {len(items)} items from benchmark JSON.")

    # 1. 통합 채점표 (72개 전체)
    integrated_rows = []
    integrated_headers = [
        "순번",
        "운동종목",
        "카메라시점 (공식표준)",
        "시점코드",
        "클립_파일명",
        "영상_재생링크 (클릭)",
        "영상_상대경로",
        "영상_절대경로",
        "수행의도 (사용자라벨)",
        "모델_최종종합판정",
        "모델_감지결함_요약",
        "전체결함_세부확률_및_상태",
        "세트기준_단순일치여부",
        "[사용자직접채점] 모델판정_타당성 (O/△/X/FN)",
        "[사용자직접채점] 실제발생_결함목록 (영상관찰)",
        "[사용자직접채점] 종합평가_및_특이사항_메모"
    ]

    for x in items:
        idx = x["index"]
        ex = x["exercise_name"]
        view_alias = x["view_alias"]
        view_full = VIEW_FULL_NAME.get(view_alias, view_alias)
        view_code = x["view_code"]
        fn = x["filename"]
        
        # 파일 경로 구성
        rel_clip_path = os.path.join(ex, x["rel_path"].split("\\")[1], fn)
        rel_from_data = os.path.join("분할_결과", rel_clip_path).replace("/", "\\")
        abs_path = os.path.join(CLIPS_BASE_DIR, rel_clip_path).replace("/", "\\")
        
        hyperlink_formula = f'=HYPERLINK("{abs_path}", "영상 재생 ▶")'
        
        intent_label = get_intent_label(ex, x["defects"])
        overall_pred, summary_str = get_model_summary(x["predictions_raw"])
        all_probs_str = get_all_probs_str(ex, x["predictions_raw"])
        
        match_str = "일치(Match)" if x["exact_match"] else "불일치(Mismatch)"
        
        integrated_rows.append([
            idx,
            ex,
            view_full,
            view_code,
            fn,
            hyperlink_formula,
            rel_from_data,
            abs_path,
            intent_label,
            overall_pred,
            summary_str,
            all_probs_str,
            match_str,
            "",  # 사용자 채점 타당성
            "",  # 사용자 실제 발생 결함
            ""   # 메모
        ])

    integrated_path = os.path.join(OUTPUT_DIR, "일반화_모델예측_통합_채점표.csv")
    with open(integrated_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(integrated_headers)
        writer.writerows(integrated_rows)
    print(f"Generated: {integrated_path}")

    # 2. 운동별 상세 채점표 생성
    # 2-1. 바벨 데드 (d1~d5)
    bd_rows = []
    bd_headers = [
        "순번", "카메라시점", "클립_파일명", "영상_재생링크 (클릭)", "영상_상대경로",
        "수행의도_라벨", "모델_최종판정",
        "d1_척추중립_확률", "d1_판정",
        "d2_무릎발목_확률", "d2_판정",
        "d3_발바닥_확률", "d3_판정",
        "d4_동시신전_확률", "d4_판정",
        "d5_바벨밀착_확률", "d5_판정",
        "[사용자확인] d1_실제발생(O/X)",
        "[사용자확인] d2_실제발생(O/X)",
        "[사용자확인] d3_실제발생(O/X)",
        "[사용자확인] d4_실제발생(O/X)",
        "[사용자확인] d5_실제발생(O/X)",
        "[사용자채점] 모델예측_타당성 (O/△/X)",
        "[사용자메모] 상세피드백_및_특이사항"
    ]
    bd_items = [x for x in items if x["exercise_name"] == "바벨 데드"]
    for i, x in enumerate(bd_items, 1):
        fn = x["filename"]
        rel_clip_path = os.path.join("바벨 데드", x["rel_path"].split("\\")[1], fn)
        rel_from_data = os.path.join("분할_결과", rel_clip_path).replace("/", "\\")
        abs_path = os.path.join(CLIPS_BASE_DIR, rel_clip_path).replace("/", "\\")
        link = f'=HYPERLINK("{abs_path}", "영상 재생 ▶")'
        view_full = VIEW_FULL_NAME.get(x["view_alias"], x["view_alias"])
        intent = get_intent_label("바벨 데드", x["defects"])
        overall, _ = get_model_summary(x["predictions_raw"])
        raw = x["predictions_raw"]

        bd_rows.append([
            i, view_full, fn, link, rel_from_data,
            intent, overall,
            f"{raw['d1']['prob']*100:.1f}%", raw['d1']['state_label'],
            f"{raw['d2']['prob']*100:.1f}%", raw['d2']['state_label'],
            f"{raw['d3']['prob']*100:.1f}%", raw['d3']['state_label'],
            f"{raw['d4']['prob']*100:.1f}%", raw['d4']['state_label'],
            f"{raw['d5']['prob']*100:.1f}%", raw['d5']['state_label'],
            "", "", "", "", "", # O/X checks
            "", # 타당성
            ""  # 메모
        ])

    bd_path = os.path.join(OUTPUT_DIR, "일반화_채점표_바벨데드.csv")
    with open(bd_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(bd_headers)
        writer.writerows(bd_rows)
    print(f"Generated: {bd_path}")

    # 2-2. 스쿼트 (c1~c4)
    sq_rows = []
    sq_headers = [
        "순번", "카메라시점", "클립_파일명", "영상_재생링크 (클릭)", "영상_상대경로",
        "수행의도_라벨", "모델_최종판정",
        "c1_척추중립_확률", "c1_판정",
        "c2_고개정면_확률", "c2_판정",
        "c3_무릎발방향_확률", "c3_판정",
        "c4_발바닥_확률", "c4_판정",
        "[사용자확인] c1_실제발생(O/X)",
        "[사용자확인] c2_실제발생(O/X)",
        "[사용자확인] c3_실제발생(O/X)",
        "[사용자확인] c4_실제발생(O/X)",
        "[사용자채점] 모델예측_타당성 (O/△/X)",
        "[사용자메모] 상세피드백_및_특이사항"
    ]
    sq_items = [x for x in items if x["exercise_name"] == "스쿼트"]
    for i, x in enumerate(sq_items, 1):
        fn = x["filename"]
        rel_clip_path = os.path.join("스쿼트", x["rel_path"].split("\\")[1], fn)
        rel_from_data = os.path.join("분할_결과", rel_clip_path).replace("/", "\\")
        abs_path = os.path.join(CLIPS_BASE_DIR, rel_clip_path).replace("/", "\\")
        link = f'=HYPERLINK("{abs_path}", "영상 재생 ▶")'
        view_full = VIEW_FULL_NAME.get(x["view_alias"], x["view_alias"])
        intent = get_intent_label("스쿼트", x["defects"])
        overall, _ = get_model_summary(x["predictions_raw"])
        raw = x["predictions_raw"]

        sq_rows.append([
            i, view_full, fn, link, rel_from_data,
            intent, overall,
            f"{raw['c1']['prob']*100:.1f}%", raw['c1']['state_label'],
            f"{raw['c2']['prob']*100:.1f}%", raw['c2']['state_label'],
            f"{raw['c3']['prob']*100:.1f}%", raw['c3']['state_label'],
            f"{raw['c4']['prob']*100:.1f}%", raw['c4']['state_label'],
            "", "", "", "", # O/X checks
            "", # 타당성
            ""  # 메모
        ])

    sq_path = os.path.join(OUTPUT_DIR, "일반화_채점표_스쿼트.csv")
    with open(sq_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sq_headers)
        writer.writerows(sq_rows)
    print(f"Generated: {sq_path}")

    # 2-3. 스티프 데드 (sd1~sd3)
    sd_rows = []
    sd_headers = [
        "순번", "카메라시점", "클립_파일명", "영상_재생링크 (클릭)", "영상_상대경로",
        "수행의도_라벨", "모델_최종판정",
        "sd1_무릎고정_확률", "sd1_판정",
        "sd2_척추중립_확률", "sd2_판정",
        "sd3_바벨밀착_확률", "sd3_판정",
        "[사용자확인] sd1_실제발생(O/X)",
        "[사용자확인] sd2_실제발생(O/X)",
        "[사용자확인] sd3_실제발생(O/X)",
        "[사용자채점] 모델예측_타당성 (O/△/X)",
        "[사용자메모] 상세피드백_및_특이사항"
    ]
    sd_items = [x for x in items if x["exercise_name"] == "스티프 데드"]
    for i, x in enumerate(sd_items, 1):
        fn = x["filename"]
        rel_clip_path = os.path.join("스티프 데드", x["rel_path"].split("\\")[1], fn)
        rel_from_data = os.path.join("분할_결과", rel_clip_path).replace("/", "\\")
        abs_path = os.path.join(CLIPS_BASE_DIR, rel_clip_path).replace("/", "\\")
        link = f'=HYPERLINK("{abs_path}", "영상 재생 ▶")'
        view_full = VIEW_FULL_NAME.get(x["view_alias"], x["view_alias"])
        intent = get_intent_label("스티프 데드", x["defects"])
        overall, _ = get_model_summary(x["predictions_raw"])
        raw = x["predictions_raw"]

        sd_rows.append([
            i, view_full, fn, link, rel_from_data,
            intent, overall,
            f"{raw['sd1']['prob']*100:.1f}%", raw['sd1']['state_label'],
            f"{raw['sd2']['prob']*100:.1f}%", raw['sd2']['state_label'],
            f"{raw['sd3']['prob']*100:.1f}%", raw['sd3']['state_label'],
            "", "", "", # O/X checks
            "", # 타당성
            ""  # 메모
        ])

    sd_path = os.path.join(OUTPUT_DIR, "일반화_채점표_스티프데드.csv")
    with open(sd_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sd_headers)
        writer.writerows(sd_rows)
    print(f"Generated: {sd_path}")

    # 3. 분할_결과 폴더에도 복사 배치
    for p in [integrated_path, bd_path, sq_path, sd_path]:
        dest = os.path.join(CLIPS_BASE_DIR, os.path.basename(p))
        shutil.copy2(p, dest)
        print(f"Copied to: {dest}")

if __name__ == "__main__":
    build_sheets()
