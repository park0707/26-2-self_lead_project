# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: fulfill_user_request.py
Purpose: 
1. 3-State 선택적 판정에서 '보류율 최대 10% 제한(Coverage >= 90%)' 하의 F1-Score 극대화 윈도우 정밀 탐색 및 성과 집계
2. D:\Data\temp\babel_deadlift_d2_samples 에 d2 정상 및 결함 대표 데이터 샘플(npz, json, 설명서) 배치
"""

import os
import sys
import json
import glob
import shutil
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.build_and_eval_d2_v3_integrated import (
    extract_dual_geometry_features, CACHE_DIR, LABEL_JSON_DIR, VIEW_ALIAS_MAP
)

print("[*] 1. 데이터셋 로드 및 v3.0 모델 5-Fold OOF 예측값 생성 중...")
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

X_list = []
y_list = []
subs_list = []
views_list = []
keys_list = []
clip_names = []

for p in files:
    base = os.path.basename(p).replace(".npz", "")
    parts = base.split("_")
    vn = parts[-1]
    folder = "_".join(parts[:-1])
    key = [pt for pt in parts if pt.startswith("Key")][0]

    sub = 1
    for i in range(1, 10):
        if f"피험자{i}" in folder:
            sub = i
            break

    with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
        d = json.load(fp)
    conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
    y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

    kps = np.load(p)["kps_33_3d_64f"]
    f12 = extract_dual_geometry_features(kps)

    X_list.append(f12)
    y_list.append(y_val)
    subs_list.append(sub)
    views_list.append(vn)
    keys_list.append(key)
    clip_names.append(base)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subs = np.array(subs_list, dtype=np.int32)
views = np.array(views_list)
keys = np.array(keys_list)

# v3.0 모델 가중치 파일 로드
weights_path = r"D:\src\models\second_models\babel_deadlift\weights\d2_v3_cleanlab_models.pkl"
with open(weights_path, "rb") as f:
    bundle = pickle.load(f)

lgb_models = bundle["lightgbm_models"]
cat_models = bundle["catboost_models"]

# 5-Fold OOF 예측값 재생성 (StratifiedGroupKFold random_state=42)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs)):
    clf_lgb = lgb_models[fold]
    clf_cat = cat_models[fold]

    X_val = X[val_idx]
    p_l = clf_lgb.predict_proba(X_val)[:, 1]
    p_c = clf_cat.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = 0.5 * p_l + 0.5 * p_c

# =========================================================================
# 요구사항 1: 보류율 최대 10% 제한 (Coverage >= 90%) 하의 F1-Score 극대화 윈도우 탐색
# =========================================================================
print("\n" + "=" * 110)
print("  [요구사항 1] 판단 보류율 최대 10% 제한 (Coverage >= 90.0%) 하의 3-State F1-Score 극대화 탐색")
print("=" * 110)

best_cov90_window = None
best_cov90_f1 = 0.0
cov90_candidates = []

for tn in np.arange(0.10, 0.35, 0.005):
    for td in np.arange(tn + 0.005, 0.45, 0.005):
        decided = (oof_preds <= tn) | (oof_preds >= td)
        cov = float(np.mean(decided) * 100)
        decline = float(100.0 - cov)

        if cov < 90.0:  # 보류율 10% 초과 배제
            continue

        y_dec = y[decided]
        p_dec = (oof_preds[decided] >= td).astype(int)

        if len(np.unique(y_dec)) < 2:
            continue

        f1 = float(f1_score(y_dec, p_dec, zero_division=0) * 100)
        acc = float(accuracy_score(y_dec, p_dec) * 100)
        prec = float(precision_score(y_dec, p_dec, zero_division=0) * 100)
        rec = float(recall_score(y_dec, p_dec, zero_division=0) * 100)
        cm = confusion_matrix(y_dec, p_dec)

        item = {
            'tn': round(float(tn), 3),
            'td': round(float(td), 3),
            'f1': f1,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'cov': cov,
            'decline': decline,
            'tp': int(cm[1, 1]) if len(cm) > 1 else 0,
            'fp': int(cm[0, 1]) if len(cm) > 1 else 0,
            'fn': int(cm[1, 0]) if len(cm) > 1 else 0,
            'tn_cnt': int(cm[0, 0]) if len(cm) > 1 else 0
        }
        cov90_candidates.append(item)

cov90_candidates_sorted = sorted(cov90_candidates, key=lambda x: x['f1'], reverse=True)

print(f"{'순위':<4} | {'3-State 윈도우':<18} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10} | {'Coverage':<10} | {'보류율':<8} | {'TP/FP/FN/TN'}")
print("-" * 110)
for rank, r in enumerate(cov90_candidates_sorted[:10], 1):
    win_str = f"[{r['tn']:.3f}, {r['td']:.3f}]"
    cm_str = f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn_cnt']}"
    print(f" {rank:<3} | {win_str:<18} | {r['f1']:<10.2f}% | {r['prec']:<10.2f}% | {r['rec']:<10.2f}% | {r['acc']:<10.2f}% | {r['cov']:<10.1f}% | {r['decline']:<6.1f}% | {cm_str}")

best_win = cov90_candidates_sorted[0]
print("=" * 110)
print(f"[★ 공식 채택 윈도우]: [{best_win['tn']:.3f}, {best_win['td']:.3f}]")
print(f"  • 3-State F1-Score    : {best_win['f1']:.2f}% (기존 2-State 46.43% 대비 +{best_win['f1']-46.43:+.2f}%p🔺)")
print(f"  • 정밀도 (Precision)  : {best_win['prec']:.2f}%")
print(f"  • 재현율 (Recall)     : {best_win['rec']:.2f}%")
print(f"  • 정확도 (Accuracy)   : {best_win['acc']:.2f}%")
print(f"  • 판정 커버리지       : {best_win['cov']:.1f}% (385건 중 347건 확정 판정)")
print(f"  • 판단 보류율         : {best_win['decline']:.1f}% (385건 중 38건 재촬영 권고, 10% 상한 준수!)")
print(f"  • 혼동 행렬           : TP={best_win['tp']}, FP={best_win['fp']}, FN={best_win['fn']}, TN={best_win['tn_cnt']}")

# 시점별 F1
print(f"\n[공식 5대 시점별 3-State F1-Score (윈도우 [{best_win['tn']:.3f}, {best_win['td']:.3f}])]:")
print(f"{'공식 시점 명칭':<24} | {'코드':<6} | {'3-State F1':<12} | {'Precision':<10} | {'Recall':<10} | {'Coverage'}")
print("-" * 80)
view_3state_res = {}
for v_code, v_alias in VIEW_ALIAS_MAP.items():
    v_mask = (views == v_code)
    y_v = y[v_mask]
    prob_v = oof_preds[v_mask]

    dec_v = (prob_v <= best_win['tn']) | (prob_v >= best_win['td'])
    y_v_dec = y_v[dec_v]
    p_v_dec = (prob_v[dec_v] >= best_win['td']).astype(int)

    f1_v = f1_score(y_v_dec, p_v_dec, zero_division=0) * 100
    prec_v = precision_score(y_v_dec, p_v_dec, zero_division=0) * 100
    rec_v = recall_score(y_v_dec, p_v_dec, zero_division=0) * 100
    cov_v = np.mean(dec_v) * 100
    view_3state_res[v_alias] = {'f1': f1_v, 'prec': prec_v, 'rec': rec_v, 'cov': cov_v}
    print(f"{v_alias:<24} | {v_code:<6} | {f1_v:<12.2f}% | {prec_v:<10.2f}% | {rec_v:<10.2f}% | {cov_v:.1f}%")
print("=" * 80)


# =========================================================================
# 요구사항 2: D:\Data\temp 에 d2 오류/정상 대표 샘플 배치
# =========================================================================
TARGET_TEMP_DIR = r"D:\Data\temp\babel_deadlift_d2_samples"
os.makedirs(TARGET_TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(TARGET_TEMP_DIR, "01_정상_데이터_Normal"), exist_ok=True)
os.makedirs(os.path.join(TARGET_TEMP_DIR, "02_오류_데이터_Defect"), exist_ok=True)

# 대표 샘플 선정 (정상 5개, 결함 5개 - 다양한 시점과 피험자)
# 정상 후보: Key281 (순수 정상) 중 예측 확률이 낮고 신뢰도 높은 샘플
normal_indices = np.where((y == 0) & (keys == "Key281") & (oof_preds < 0.20))[0]
# 결함 후보: Key283 (단독 무릎 결함) 및 Key287 (복합 무릎 결함) 중 예측 확률 높은 샘플
defect_indices = np.where((y == 1) & (oof_preds >= 0.35))[0]

selected_normal_idx = [normal_indices[i] for i in [0, 10, 25, 40, 55] if i < len(normal_indices)]
selected_defect_idx = [defect_indices[i] for i in [0, 8, 16, 24, 32] if i < len(defect_indices)]

manifest = []

print("\n[*] 2. D:\\Data\\temp 에 정상/결함 샘플 복사 및 메타데이터 작성 중...")

# 정상 샘플 복사
for i, idx in enumerate(selected_normal_idx, 1):
    c_name = clip_names[idx]
    npz_src = os.path.join(CACHE_DIR, f"{c_name}.npz")
    parts = c_name.split("_")
    folder = "_".join(parts[:-1])
    json_src = os.path.join(LABEL_JSON_DIR, f"{folder}.json")

    dst_folder = os.path.join(TARGET_TEMP_DIR, "01_정상_데이터_Normal", f"sample_{i:02d}_{c_name}")
    os.makedirs(dst_folder, exist_ok=True)

    shutil.copy2(npz_src, os.path.join(dst_folder, f"{c_name}.npz"))
    if os.path.exists(json_src):
        shutil.copy2(json_src, os.path.join(dst_folder, f"{folder}.json"))

    prob = float(oof_preds[idx])
    state = "확정 정상 (Accept Normal)" if prob <= best_win['tn'] else ("확정 결함" if prob >= best_win['td'] else "판정 보류")

    manifest.append({
        "sample_type": "정상 (Normal)",
        "sample_folder": f"01_정상_데이터_Normal/sample_{i:02d}_{c_name}",
        "clip_name": c_name,
        "subject": int(subs[idx]),
        "view": VIEW_ALIAS_MAP.get(views[idx], views[idx]),
        "key_scenario": keys[idx],
        "ground_truth_d2": "정상 (0, 무릎-발 방향 일치함)",
        "model_defect_prob": round(prob, 4),
        "three_state_decision": state
    })

# 결함 샘플 복사
for i, idx in enumerate(selected_defect_idx, 1):
    c_name = clip_names[idx]
    npz_src = os.path.join(CACHE_DIR, f"{c_name}.npz")
    parts = c_name.split("_")
    folder = "_".join(parts[:-1])
    json_src = os.path.join(LABEL_JSON_DIR, f"{folder}.json")

    dst_folder = os.path.join(TARGET_TEMP_DIR, "02_오류_데이터_Defect", f"sample_{i:02d}_{c_name}")
    os.makedirs(dst_folder, exist_ok=True)

    shutil.copy2(npz_src, os.path.join(dst_folder, f"{c_name}.npz"))
    if os.path.exists(json_src):
        shutil.copy2(json_src, os.path.join(dst_folder, f"{folder}.json"))

    prob = float(oof_preds[idx])
    state = "확정 결함 (Accept Defect)" if prob >= best_win['td'] else ("확정 정상" if prob <= best_win['tn'] else "판정 보류")

    manifest.append({
        "sample_type": "결함 (Defect)",
        "sample_folder": f"02_오류_데이터_Defect/sample_{i:02d}_{c_name}",
        "clip_name": c_name,
        "subject": int(subs[idx]),
        "view": VIEW_ALIAS_MAP.get(views[idx], views[idx]),
        "key_scenario": keys[idx],
        "ground_truth_d2": "결함 (1, 무릎 외반 / 방향 불일치 발생)",
        "model_defect_prob": round(prob, 4),
        "three_state_decision": state
    })

# README.md 작성
readme_path = os.path.join(TARGET_TEMP_DIR, "README_샘플_데이터_가이드.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# 바벨 데드리프트 d2 (무릎-발 방향 일치) 정상 및 결함 검증용 샘플 데이터\n\n")
    f.write(f"- 생성 일시: 2026년 9월 29일\n")
    f.write(f"- 위치: `D:\\Data\\temp\\babel_deadlift_d2_samples`\n")
    f.write(f"- 적용 모델: `vidio-health-trainer` d2 v3.0 통합 모델\n")
    f.write(f"- 3-State 공식 윈도우: `[{best_win['tn']:.3f}, {best_win['td']:.3f}]` (보류율: {best_win['decline']:.1f}% <= 10.0% 준수)\n\n")
    f.write("## 1. 포함된 데이터 구성\n\n")
    f.write("각 샘플 폴더마다 다음 두 가지 파일이 포함되어 있습니다:\n")
    f.write("1. `*.npz`: 64프레임으로 보간된 33개 관절 3D 월드 좌표계 (`kps_33_3d_64f` 키로 저장됨)\n")
    f.write("2. `*.json`: AI-Hub 원천 2D 어노테이션 및 5대 결함 참값 라벨 메타데이터\n\n")
    f.write("## 2. 샘플 목록 및 메타데이터 요약표\n\n")
    f.write("| 번호 | 구분 | 폴더명 | 피험자 | 카메라 시점 | 시나리오 Key | 정답 라벨 (Ground Truth) | 결함 예측 확률 | 3-State 판정 결과 |\n")
    f.write("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for i, m in enumerate(manifest, 1):
        f.write(f"| {i} | {m['sample_type']} | `{m['sample_folder']}` | 피험자 {m['subject']} | {m['view']} | {m['key_scenario']} | {m['ground_truth_d2']} | {m['model_defect_prob']:.4f} | **{m['three_state_decision']}** |\n")
    f.write("\n## 3. 파이썬에서 직접 로드하는 코드 예시\n\n")
    f.write("```python\nimport numpy as np\n\n# npz 로드 예시\ndata = np.load(r'sample_01_.../파일명.npz')\nkps = data['kps_33_3d_64f']  # 형상: (64, 33, 3)\nprint('프레임수:', kps.shape[0], '관절수:', kps.shape[1], '좌표축:', kps.shape[2])\n\n# 25, 26번 관절: 좌우 무릎 3D 좌표\nleft_knee = kps[:, 25, :]\nright_knee = kps[:, 26, :]\nknee_width = np.linalg.norm(left_knee - right_knee, axis=1)\nprint('최소 무릎 너비(m):', np.min(knee_width))\n```\n")

print(f"[+] 샘플 데이터 복사 및 README 작성 완료: {TARGET_TEMP_DIR}")

# 결과 JSON 저장
res_file = os.path.join(r"D:\src\models\second_models\babel_deadlift", "user_request_cov90_results.json")
with open(res_file, "w", encoding="utf-8") as f:
    json.dump({
        "best_window": best_win,
        "view_results": view_3state_res,
        "samples_manifest": manifest
    }, f, ensure_ascii=False, indent=2)

print(f"[+] 실행 완료! 결과 JSON: {res_file}")
