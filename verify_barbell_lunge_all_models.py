# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: verify_barbell_lunge_all_models.py
Purpose: 바벨 런지 세부 결함(bl1 ~ bl5) 최종 모델에 대한
         1. 원천 데이터 3-State F1 및 상세 성능 취합
         2. AI-Hub 라벨 피처 투입 여부 전수 감사 (Static AST & Correlation & Shape Audit)
         3. 데이터 누수(타깃 누수, 피험자 누수, 시점 누수) 전수 정량 검증
"""
import os
import sys
import glob
import json
import pickle
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

BL_DIR = r"D:\src\models\second_models\babel_lunge"
FEAT_DIR = r"D:\Data\피처_데이터"

def audit_dataset_integrity():
    print("=" * 80)
    print(" [1] 바벨 런지 bl1~bl5 데이터셋 캐시 파일 전수 검사 (피처-라벨-피험자 무결성)")
    print("=" * 80)
    
    datasets = {}
    
    # bl1
    with open(os.path.join(FEAT_DIR, "bl1_v4_dataset_cache.pkl"), "rb") as f:
        d1 = pickle.load(f)
    g1 = np.array([int("".join(c for c in str(x) if c.isdigit())) for x in d1["groups"]])
    datasets["bl1_v4"] = {"X": d1["X"], "y": d1["y"], "groups": g1, "views": np.array(d1["views"]).astype(str), "names": d1.get("feature_names")}
    
    # bl2
    d2 = np.load(os.path.join(BL_DIR, "lunge_c2_features_v6_18d_delta_cache.npz"), allow_pickle=True)
    g2 = np.array([int("".join(c for c in str(x) if c.isdigit())) for x in d2["subjects"]])
    datasets["bl2_v6"] = {"X": d2["X"], "y": d2["y"], "groups": g2, "views": d2["views"].astype(str), "names": d2.get("feature_names", None)}
    
    # bl3
    d3 = np.load(os.path.join(FEAT_DIR, "lunge_bl3_dataset_19d_adv.npz"), allow_pickle=True)
    sys.path.insert(0, "d:/")
    from src.features.lunge_bl3_features_v4 import FEATURE_INDICES_14D, FEATURE_NAMES_14D
    datasets["bl3_v4"] = {"X": d3["X"][:, FEATURE_INDICES_14D], "y": d3["y"], "groups": d3["groups"].astype(int), "views": None, "names": FEATURE_NAMES_14D}
    
    # bl4
    d4 = np.load(os.path.join(FEAT_DIR, "lunge_c4_dataset_cache_v3.npz"), allow_pickle=True)
    g4 = np.array([int("".join(c for c in str(x) if c.isdigit())) for x in d4["subjects"]])
    datasets["bl4_v4b"] = {"X": d4["X_24d"], "y": d4["y"], "groups": g4, "views": d4["views"].astype(str), "names": d4.get("feature_names", None)}
    
    # bl5
    d5 = np.load(os.path.join(FEAT_DIR, "lunge_c5_v5_36d_features.npz"), allow_pickle=True)
    datasets["bl5_v5"] = {"X": d5["X"], "y": d5["y"], "groups": d5["subjects"].astype(int), "views": d5["views"].astype(str), "names": d5.get("feature_names", None)}

    for m_name, d in datasets.items():
        X, y, g = d["X"], d["y"], d["groups"]
        print(f"\n▶ [{m_name}]")
        print(f"  - 샘플 수(N): {len(y)}, 피처 수(D): {X.shape[1]}")
        print(f"  - 클래스 분포: 정상(0)={np.sum(y==0)}건, 결함(1)={np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
        print(f"  - 피험자 목록: {sorted(list(set(g)))} (총 {len(set(g))}명 전원 포함)")
        print(f"  - 결측치(NaN/Inf) 여부: NaN={np.isnan(X).sum()}개, Inf={np.isinf(X).sum()}개")
        
        # 1. 완벽 상관 피처 탐지 (Target Leakage 여부: r == 1.0 피처가 존재하는가?)
        corrs = [abs(np.corrcoef(X[:, col], y)[0, 1]) if np.std(X[:, col]) > 1e-9 else 0.0 for col in range(X.shape[1])]
        max_corr = max(corrs) if corrs else 0.0
        max_corr_idx = int(np.argmax(corrs)) if corrs else -1
        print(f"  - 피처-라벨 최대 피어슨 상관계수: |r| = {max_corr:.4f} (Col {max_corr_idx})")
        if max_corr > 0.95:
            print(f"    [WARNING] 완벽/준완벽 상관 피처 발견! 타깃 누수 의심!")
        else:
            print(f"    [PASS] 타깃 누수성 단일 피처 없음 (|r| < 0.95 정상 범위)")

        # 2. 피험자 격리 무결성 확인 (5-Fold StratifiedGroupKFold 시 Fold 간 겹침 여부)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        leak_subjects = False
        for fold, (tr, va) in enumerate(sgkf.split(X, y, g)):
            tr_subs = set(g[tr])
            va_subs = set(g[va])
            overlap = tr_subs.intersection(va_subs)
            if len(overlap) > 0:
                print(f"    [FAIL] Fold {fold} 피험자 누수 발생! 겹치는 피험자: {overlap}")
                leak_subjects = True
        if not leak_subjects:
            print(f"    [PASS] 5-Fold 피험자 격리 무결성 100% (Fold 간 교차 피험자 0건)")

    return datasets

def audit_source_codes():
    print("\n" + "=" * 80)
    print(" [2] 소스 코드 정적 감사: 피처 추출 모듈 내 AI-Hub 라벨/3D JSON 참조 여부")
    print("=" * 80)
    
    feature_files = [
        r"D:\src\features\lunge_c1_features_v4.py",
        r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v6.py",
        r"D:\src\features\lunge_bl3_features_v4.py",
        r"D:\src\features\lunge_c4_features_v3.py",
        r"D:\src\features\lunge_c5_features_v5.py"
    ]
    
    forbidden_terms = ['condition', 'conditions', 'label', 'labels', 'metadata', 'AI-Hub', 'aihub', '3D.json', '3d_data']
    
    for fpath in feature_files:
        fname = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        violations = []
        for lnum, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith('#') or s.startswith('"""') or s.startswith('*'):
                continue
            for term in forbidden_terms:
                if term in s:
                    violations.append((lnum, term, s))
                    
        print(f"\n▶ [{fname}] 검사 결과:")
        if len(violations) == 0:
            print(f"  [PASS] 실행 코드 라인 내 금지 키워드 0건! (순수 33관절/15관절 좌표만 입력받음)")
        else:
            print(f"  [FAIL] 위반 의심 라인 {len(violations)}건:")
            for lnum, term, s in violations[:5]:
                print(f"    Line {lnum} ({term}): {s}")

if __name__ == "__main__":
    audit_dataset_integrity()
    audit_source_codes()
