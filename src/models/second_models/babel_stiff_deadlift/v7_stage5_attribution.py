# -*- coding: utf-8 -*-
"""v7 Stage 5: 원시 텐서 통계가 원본 라벨 AUC를 올리는 이유 귀인 분석
 - 신호가 하체(무릎) 채널에서 오는가, 상체/바벨 채널(= 다른 결함의 흔적, 지름길)에서 오는가
 - 풀링 AUC vs 피험자 내 AUC (피험자 간 라벨 비율 차이로 인한 착시 점검)"""
import sys, os, json, re, warnings
import numpy as np
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from v7_ablation_core import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

D = load_dataset(); F = D['feats']; subs = D['subs']; y_o = D['y_orig']; uniq = D['uniq']
RAW = np.load(os.path.join(CURRENT_DIR, 'v7_rawstats_cache.npz'))['raw']       # (N, 490) = [mean,std,min,max,range] x 98ch
V7 = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15, 16, 17]
STATS = ['mean', 'std', 'min', 'max', 'range']

# 채널 이름 (kinematics_3d.py Format B 규격)
EDGES = ['pelvis-torso', 'torso-neck', 'torso-Lsh', 'torso-Rsh', 'Lsh-Lelb', 'Lelb-Lwr', 'Rsh-Relb', 'Relb-Rwr',
         'pelvis-Lhip', 'pelvis-Rhip', 'Lhip-Lknee', 'Lknee-Lank', 'Rhip-Rknee', 'Rknee-Rank']
CH = []
for e in EDGES:
    CH += [f'{e}.dx', f'{e}.dy', f'{e}.dz', f'{e}.len']
CH += ['ang_knee_L', 'ang_knee_R', 'ang_hip_L', 'ang_hip_R', 'ang_ankle_L', 'ang_ankle_R', 'ang_elbow_L', 'ang_elbow_R',
       'torso_pitch', 'torso_roll', 'knee_valgus_L', 'knee_valgus_R', 'sh_hip_twist', 'knee_dist', 'ankle_dist', 'ankle_z_spread',
       'knee_asym', 'hip_asym', 'vel_knee_L', 'vel_knee_R', 'vel_hip_L', 'vel_hip_R', 'vel_ankle_L', 'vel_ankle_R', 'vel_elbow_L', 'vel_elbow_R',
       'neck_spine_align', 'barbell_body_dist', 'knee_hip_synchrony',
       'signed_ankle_z_L', 'signed_ankle_z_R', 'step_dx', 'step_dz', 'shank_vert_L', 'shank_vert_R', 'hip_knee_rom_ratio']
CH += [f'pad{i}' for i in range(98 - len(CH))]
assert len(CH) == 98

LOWER = list(range(32, 56)) + [56, 57, 58, 59, 60, 61] + [66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 84] + list(range(85, 92))
UPPER = list(range(0, 32)) + [62, 63, 64, 65, 68, 80, 81, 82, 83]
KNEE_ONLY = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 74, 75, 78, 79, 89, 90]  # 무릎·정강이 관련 채널만
assert not set(LOWER) & set(UPPER)

def cols_for(chs):
    return [s * 98 + c for s in range(5) for c in chs]

def pooled_and_within(oof, y):
    pooled = roc_auc_score(y, oof) * 100
    ws = [roc_auc_score(y[subs == s], oof[subs == s]) * 100 for s in np.unique(subs) if len(np.unique(y[subs == s])) > 1]
    return pooled, float(np.mean(ws)), len(ws)

def loso(X, mf, seeds):
    oofs = []
    for sd in seeds:
        oof = np.zeros(len(X))
        for s in np.unique(subs):
            va = subs == s
            m = mf(sd); m.fit(X[~va], y_o[~va]); oof[va] = m.predict_proba(X[va])[:, 1]
        oofs.append(oof)
    return np.mean(oofs, axis=0)

LG = lambda sd: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, max_depth=3, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.7, class_weight='balanced', random_state=sd, verbose=-1)
LR = lambda sd: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000))

print("=" * 92 + "\n[S5-1] 피험자별 원본 결함 비율 (피험자 간 라벨 비율 차이 → 풀링 AUC 착시 가능성)\n" + "=" * 92)
for s in np.unique(subs):
    m = subs == s; print(f"  피험자 {int(s)}: N={m.sum():3d} 결함비율={y_o[m].mean()*100:5.1f}%")

print("\n" + "=" * 92 + "\n[S5-2] 채널 그룹별 원본 라벨 예측 (원본라벨 학습, LOSO). 풀링AUC / 피험자내AUC 평균\n" + "=" * 92, flush=True)
sets = [("v7 12D(수작업 무릎 피처)", F[:, V7]),
        ("원시통계 전체 490D", RAW),
        ("하체/무릎 채널만", RAW[:, cols_for(LOWER)]),
        ("무릎·정강이 채널만", RAW[:, cols_for(KNEE_ONLY)]),
        ("상체/바벨 채널만(무릎 정보 없음)", RAW[:, cols_for(UPPER)])]
out = {}
for name, X in sets:
    for mn, mf, seeds in (("LightGBM", LG, (0, 1, 2)), ("LogReg", LR, (0,))):
        oof = loso(X, mf, seeds); p, w, n = pooled_and_within(oof, y_o); ts = three_state(oof, y_o, uniq)
        out[f"{name}|{mn}"] = dict(d=X.shape[1], pooled=p, within=w, f1=ts['f1d'], spec=ts['spec'], prec=ts['prec'])
        print(f"  {name:30s} d={X.shape[1]:3d} {mn:8s} 풀링AUC={p:5.1f} 피험자내AUC={w:5.1f} F1d={ts['f1d']:4.1f} 정밀도={ts['prec']:4.1f} 특이도={ts['spec']:4.1f}", flush=True)

print("\n" + "=" * 92 + "\n[S5-3] 전체 490D LightGBM 피처 기여(gain) 상위 20 (원본라벨 학습, 전체 적합)\n" + "=" * 92)
m = LG(0); m.fit(RAW, y_o); g = m.booster_.feature_importance('gain'); g = g / g.sum() * 100
order = np.argsort(g)[::-1][:20]; grp_gain = {'상체/바벨': 0.0, '하체/무릎': 0.0, '기타': 0.0}
for j in range(RAW.shape[1]):
    c = j % 98; grp_gain['하체/무릎' if c in LOWER else ('상체/바벨' if c in UPPER else '기타')] += g[j]
for j in order:
    c = j % 98; tag = '하체' if c in LOWER else ('상체' if c in UPPER else '기타')
    print(f"  {CH[c]:22s} {STATS[j//98]:5s} gain={g[j]:5.2f}%  [{tag}]")
print("  그룹별 gain 합: " + ", ".join(f"{k}={v:.1f}%" for k, v in grp_gain.items()))
out['gain_by_group'] = grp_gain
json.dump(out, open(os.path.join(CURRENT_DIR, 'v7_stage5_attribution.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print("완료", flush=True)
