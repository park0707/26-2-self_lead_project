# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/reports/audit_squat_4defects_v2.py
Purpose: 바벨 스쿼트 4대 세부 오류 모델 지표별 합격/불합격 감사 (v2)
  - 합격 기준은 결과 확인 이전에 고정 (아래 CRITERIA). 합산 점수 미사용.
  - (A) 현재 프로덕션 임계치(in-sample, 75클립으로 튜닝됨) 와
    (B) Leave-One-View-Out 임계치(시점 제외 후 튜닝 -> 제외 시점에 적용) 를 모두 보고.
  - 최종 판정은 (A),(B) 모두 만족해야 '통과'. 부트스트랩 95% CI 병기.
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
INPUT_JSON = r"D:\data\일반화\squat_75reps_with_spinepose_benchmark_results.json"
META_PATH = r"D:\src\web\weights\anti_overfitting_master_metadata.json"
OUT_JSON = r"D:\data\일반화\squat_4_defects_audit_v2.json"

# 사전 고정 합격 기준 (모두 '이상')
CRITERIA = {
    "balanced_acc": 70.0,
    "recall": 60.0,
    "specificity": 80.0,
    "precision": 50.0,
    "roc_auc": 70.0,
    "coverage": 80.0,
}
LABEL = {"balanced_acc": "균형 정확도", "recall": "재현율", "specificity": "특이도",
         "precision": "정밀도", "roc_auc": "ROC-AUC", "coverage": "판정 도달률"}
NAMES = {"c1": "상체 과숙임(척추 중립)", "c2": "무릎 모임(고개/시선 계열 c2)", "c3": "무릎-발 방향 불일치", "c4": "발바닥 지면 고정"}
NAMES["c2"] = "고개 정면/시선 위반"
VMAP = {"시점 A": "view5", "시점 B": "view2", "시점 C": "view3", "시점 D": "view4", "시점 E": "view1"}
DEFECTS = ["c1", "c2", "c3", "c4"]

raw = json.load(open(INPUT_JSON, encoding="utf-8"))
meta = json.load(open(META_PATH, encoding="utf-8"))
R = raw["results"]
view = np.array([VMAP[r["view_alias"]] for r in R])
views = sorted(set(view))


def arr(c, k):
    return np.array([float(r["defects"][c][k]) for r in R])


def metrics(y, s, p):
    pb = (s == 1).astype(int)
    tp = int(((y == 1) & (pb == 1)).sum()); tn = int(((y == 0) & (pb == 0)).sum())
    fp = int(((y == 0) & (pb == 1)).sum()); fn = int(((y == 1) & (pb == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "balanced_acc": (rec + spec) / 2 * 100, "recall": rec * 100, "specificity": spec * 100,
            "precision": prec * 100, "f1": f1 * 100,
            "roc_auc": roc_auc_score(y, p) * 100, "coverage": float((s != 2).mean() * 100)}


def prod_states(c):
    p = arr(c, "prob"); by = meta[c]["thresholds"]["by_view"]
    s = np.zeros(len(R), int)
    for i, v in enumerate(view):
        td, tn = by[v]["tau_defect"], by[v]["tau_normal"]
        s[i] = 1 if p[i] >= td else (0 if p[i] <= tn else 2)
    return s


def lovo_states(c):
    y = arr(c, "true").astype(int); p = arr(c, "prob"); s = np.zeros(len(R), int)
    for hv in views:
        tr = view != hv; best = (-1.0, 0.5)
        for t in np.arange(0.2, 0.95, 0.01):
            pb = (p[tr] >= t).astype(int); yt = y[tr]
            rec = ((yt == 1) & (pb == 1)).sum() / max(1, (yt == 1).sum())
            sp = ((yt == 0) & (pb == 0)).sum() / max(1, (yt == 0).sum())
            if (rec + sp) / 2 > best[0]:
                best = ((rec + sp) / 2, t)
        t = best[1]; te = view == hv
        s[te] = np.where(p[te] >= t, 1, np.where(p[te] <= t - 0.06, 0, 2))
    return s


def judge(m):
    chk = {k: {"value": round(m[k], 2), "min": CRITERIA[k], "pass": bool(m[k] >= CRITERIA[k])} for k in CRITERIA}
    n = sum(v["pass"] for v in chk.values())
    verdict = "통과" if n == 6 else ("조건부" if n >= 4 else "불합격")
    return chk, n, verdict


def bootstrap_ci(y, s, p, key, n=2000):
    rng = np.random.default_rng(0); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(set(y[ix])) < 2:
            continue
        vals.append(metrics(y[ix], s[ix], p[ix])[key])
    return [round(float(x), 1) for x in np.percentile(vals, [2.5, 97.5])]


RANK = {"통과": 2, "조건부": 1, "불합격": 0}
out = {"project": "vidio-health-trainer", "exercise": "바벨 스쿼트", "n_clips": len(R),
       "criteria": CRITERIA, "models": {}}
for c in DEFECTS:
    y = arr(c, "true").astype(int); p = arr(c, "prob")
    res = {}
    for tag, s in (("in_sample_production", prod_states(c)), ("leave_one_view_out", lovo_states(c))):
        m = metrics(y, s, p)
        chk, n, verdict = judge(m)
        res[tag] = {"metrics": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()},
                    "checks": chk, "passed": n, "verdict": verdict,
                    "ci95_balanced_acc": bootstrap_ci(y, s, p, "balanced_acc"),
                    "ci95_f1": bootstrap_ci(y, s, p, "f1")}
    final = min((res[t]["verdict"] for t in res), key=lambda v: RANK[v])
    res["final_verdict"] = final
    res["name"] = NAMES[c]
    out["models"][c] = res

json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("합격 기준(사전 고정):", CRITERIA)
for c, r in out["models"].items():
    print(f"\n[{c}] {r['name']}  => 최종: {r['final_verdict']}")
    for tag in ("in_sample_production", "leave_one_view_out"):
        t = r[tag]; m = t["metrics"]
        fails = [LABEL[k] + f"({v['value']}<{v['min']})" for k, v in t["checks"].items() if not v["pass"]]
        print(f"  {tag:22s} {t['verdict']}({t['passed']}/6) TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']} | BalAcc {m['balanced_acc']} CI{t['ci95_balanced_acc']} | 미충족: {', '.join(fails) or '없음'}")
print("\nsaved", OUT_JSON)
