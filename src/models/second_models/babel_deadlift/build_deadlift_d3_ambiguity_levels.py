# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: build_deadlift_d3_ambiguity_levels.py
Purpose: AI-Hub 385개 전수 시퀀스에 대해 생체역학적 4단계 모호성(Level 1~4) 정밀 분류 및 검증
         - Level 1: 완전 명확군 (Golden Standard / 고순도 정자세 및 명백한 결함)
         - Level 2: 경미 자세 흔들림군 (Mild Variation / 자연스런 미세 떨림)
         - Level 3: 차폐 및 경계선 모호군 (Moderate Ambiguity / 원판 가림 및 Borderline)
         - Level 4: 역전 및 라벨 오염군 (Severe Noise / Physical Inversion) -> 폐기(Pruned)
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.train_and_eval_d3_v6_0 import prepare_v6_0_dataset
from src.models.second_models.babel_deadlift.d3_model_v6_0 import FEATURE_NAMES_36_D3

OUTPUT_NPZ_PATH = os.path.join(CURRENT_DIR, "deadlift_d3_ambiguity_levels.npz")
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "deadlift_d3_ambiguity_summary.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def classify_d3_ambiguity():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d3 (발바닥 지면 고정) 생체역학 4단계 모호성 레벨 체계 산출")
    print("=" * 100)

    X_seq_57, X_feats_36, y, subjects, views, clip_names = prepare_v6_0_dataset()
    N = len(y)

    pitch_nadir = X_feats_36[:, 0]
    wobble = X_feats_36[:, 1]
    ydelta_nadir = X_feats_36[:, 3]
    pitch_max_act = X_feats_36[:, 5]
    pitch_drop = X_feats_36[:, 6]
    toe_lift = X_feats_36[:, 8]
    heel_lift_max = X_feats_36[:, 19]
    vert_vel_rms = X_feats_36[:, 21]
    pitch_angle_max = X_feats_36[:, 26]
    always_lift = X_feats_36[:, 30]

    levels = np.zeros(N, dtype=np.int32)
    reasons = []

    for i in range(N):
        lbl = y[i]
        sub = subjects[i]
        vw = views[i]
        hl = heel_lift_max[i]
        p_act = pitch_max_act[i]
        wb = wobble[i]
        p_max = pitch_angle_max[i]

        # -------------------------------------------------------------
        # 1. Level 4 (라벨 오류 / 물리적 역전 / 노이즈) -> 전면 폐기 대상
        # -------------------------------------------------------------
        # Case 4-A: 후광효과(Halo Effect) 오라벨링 (완벽 접지인데 결함 태깅)
        is_4a = (lbl == 1) and (hl < 0.095 and wb < 0.0135 and p_act < 18.0)
        # Case 4-B: 물리적 역전 오라벨링 (뒤꿈치 붕 뜸/흔들림 폭증 상태에 정상 태깅)
        is_4b = (lbl == 0) and ((hl > 0.45) or (wb > 0.045) or (p_act > 65.0 and hl > 0.35))
        # Case 4-C: 시각적 차폐 왜곡 역전 (원판 간섭 발목 각도 붕괴 상태 정상 태깅)
        is_4c = (lbl == 0) and (p_max > 75.0 and hl > 0.35)

        if is_4a:
            levels[i] = 4
            reasons.append("4-A: 후광효과 오라벨링 (지면 완전고정 상태 결함 태깅)")
        elif is_4b:
            levels[i] = 4
            reasons.append("4-B: 물리적 역전 오라벨링 (뒤꿈치 붕 뜸/흔들림 폭증 상태 정상 태깅)")
        elif is_4c:
            levels[i] = 4
            reasons.append("4-C: 시각적 차폐 왜곡 역전 (원판 간섭 발목 각도 붕괴)")

        # -------------------------------------------------------------
        # 2. Level 3 (시점별 차폐 / 경계선 모호군)
        # -------------------------------------------------------------
        elif vw in ['view1', 'view5'] and (0.17 <= hl <= 0.29 or 24.0 <= p_act <= 38.0):
            levels[i] = 3
            reasons.append("Level 3: 후방 시점 원판 차폐 및 발가락 가림 모호군")
        elif (0.18 <= hl <= 0.26) and (22.0 <= p_act <= 34.0):
            levels[i] = 3
            reasons.append("Level 3: 정상-결함 경계선 모호군 (Borderline Margin)")

        # -------------------------------------------------------------
        # 3. Level 2 (경미 자세 흔들림군)
        # -------------------------------------------------------------
        elif (0.015 <= wb <= 0.024) or (0.13 <= hl < 0.19) or (18.0 <= p_act < 25.0 and lbl == 0):
            levels[i] = 2
            reasons.append("Level 2: 허용 범위 내 경미 발목 흔들림/자연 변동")

        # -------------------------------------------------------------
        # 4. Level 1 (완전 명확군)
        # -------------------------------------------------------------
        else:
            levels[i] = 1
            reasons.append("Level 1: 고순도 기준 정자세 및 명백한 결함 (Golden Standard)")

    reasons = np.array(reasons)
    l4_mask = (levels == 4)

    print("\n[*] 4단계 모호성 레벨 분포 결과 (N=385):")
    level_counts = {}
    for lvl in [1, 2, 3, 4]:
        cnt = int(np.sum(levels == lvl))
        ratio = float(cnt / N * 100.0)
        n_norm = int(np.sum((levels == lvl) & (y == 0)))
        n_def = int(np.sum((levels == lvl) & (y == 1)))
        level_counts[f"level_{lvl}"] = {
            "count": cnt,
            "ratio": ratio,
            "normal_count": n_norm,
            "defect_count": n_def,
            "defect_ratio": float(n_def / max(1, cnt) * 100.0)
        }
        print(f"  - Level {lvl}: {cnt:3d}개 ({ratio:5.2f}%) | 실제정상: {n_norm:3d}개, 실제결함: {n_def:3d}개 (결함율 {n_def/max(1, cnt)*100:4.1f}%)")

    print("\n[*] Level 4 세부 오염 유형 분석:")
    l4_reasons = reasons[l4_mask]
    u_reas, c_reas = np.unique(l4_reasons, return_counts=True)
    l4_details = {}
    for u, c in zip(u_reas, c_reas):
        l4_details[str(u)] = int(c)
        print(f"  - {u}: {c}개")

    print("\n[*] 피험자별 Level 4 분포:")
    subj_l4 = {}
    for s in range(1, 10):
        cnt_s = int(np.sum((subjects == s) & l4_mask))
        tot_s = int(np.sum(subjects == s))
        subj_l4[f"subject_{s}"] = {"l4_count": cnt_s, "total": tot_s, "ratio": float(cnt_s / tot_s * 100.0)}
        print(f"  - 피험자 {s}: {cnt_s:2d} / {tot_s:2d}개 ({cnt_s/tot_s*100:5.1f}%)")

    print("\n[*] 시점별 Level 4 분포:")
    view_l4 = {}
    for vcode in ['view5', 'view2', 'view3', 'view4', 'view1']:
        alias = VIEW_ALIAS_MAP.get(vcode, vcode)
        cnt_v = int(np.sum((views == vcode) & l4_mask))
        tot_v = int(np.sum(views == vcode))
        view_l4[vcode] = {"alias": alias, "l4_count": cnt_v, "total": tot_v, "ratio": float(cnt_v / tot_v * 100.0)}
        print(f"  - {alias}: {cnt_v:2d} / {tot_v:2d}개 ({cnt_v/tot_v*100:5.1f}%)")

    # npz 저장
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        levels=levels,
        reasons=reasons,
        clip_names=np.array(clip_names),
        y=y,
        subjects=subjects,
        views=views
    )
    print(f"\n[+] 4단계 모호성 레벨 NPZ 저장 완료: {OUTPUT_NPZ_PATH}")

    # JSON 저장
    summary_data = {
        "exercise": "02_바벨_데드리프트",
        "defect": "d3_발바닥_지면_고정",
        "total_samples": N,
        "level_distribution": level_counts,
        "level_4_details": l4_details,
        "subject_distribution": subj_l4,
        "view_distribution": view_l4
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 모호성 요약 통계 JSON 저장 완료: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    classify_d3_ambiguity()
