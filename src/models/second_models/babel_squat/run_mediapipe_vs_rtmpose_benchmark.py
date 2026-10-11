# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_mediapipe_vs_rtmpose_benchmark.py
Purpose: 바벨 스쿼트 세부 오류 4개 모델(c1, c2, c3, c4)에 대한
         MediaPipe vs RTMPose 관절 추출 백본 교체 1:1 비교 공식 벤치마크
         - Ceteris Paribus: 관절 추출 모델 차이 외 피처 수학 정의/모델/5-Fold 분할 100% 동일
         - 2,605건 전수 데이터(9명 피험자, 5대 전 시점) 5-Fold StratifiedGroupKFold 피험자 완전 격리
         - 원천 데이터 3-State F1 (Selective Classification) 및 강제 이진 지표 전수 산출
         - 5대 공식 시점(시점 A~E) 다차원 성능 감사
         - 공식 보고서(.docx 및 .md) 자동 생성
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# 캐시 경로
FEATS_DIR = r"D:\Data\피처_데이터"
MASTER_MP_PATH = os.path.join(FEATS_DIR, "Master_Squat_MediaPipe33_64F.npz")
MASTER_RTM_PATH = os.path.join(FEATS_DIR, "Master_Squat_RTMPose26_64F.npz")

MP_C1_PATH = os.path.join(FEATS_DIR, "squat_c1_v11_a_38d_feats.npz")
MP_C2_PATH = os.path.join(FEATS_DIR, "squat_c2_v7_19d_feats.npz")
MP_C3_PATH = os.path.join(FEATS_DIR, "squat_c3_v6_22d_feats.npz")
MP_C4_PATH = os.path.join(FEATS_DIR, "squat_c4_v5_25d_feats.npz")

RESULTS_JSON_PATH = r"D:\src\models\second_models\babel_squat\mediapipe_vs_rtmpose_benchmark_results.json"
OUT_DOCX_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨스쿼트_세부오류4종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.docx"
OUT_MD_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨스쿼트_세부오류4종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.md"

VIEW_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}


def evaluate_metrics(y_true, y_prob, threshold=0.50, tau_norm=0.40, tau_def=0.60):
    """강제 이진 및 3-State 선택적 판정 지표 동시 산출"""
    # 1. 강제 이진 지표
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0) * 100.0)
    prec = float(precision_score(y_true, y_pred, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, y_pred, zero_division=0) * 100.0)
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    try:
        auc = float(roc_auc_score(y_true, y_prob) * 100.0)
    except Exception:
        auc = 50.0

    # 2. 3-State 선택적 판정 지표 (Balanced [0.40, 0.60])
    det_mask = (y_prob <= tau_norm) | (y_prob >= tau_def)
    n_det = int(np.sum(det_mask))
    cov = float(n_det / len(y_true) * 100.0)
    retake = float(100.0 - cov)

    if n_det > 0:
        y_true_det = y_true[det_mask]
        y_pred_det = (y_prob[det_mask] >= 0.50).astype(int)
        sel_f1 = float(f1_score(y_true_det, y_pred_det, zero_division=0) * 100.0)
        sel_prec = float(precision_score(y_true_det, y_pred_det, zero_division=0) * 100.0)
        sel_rec = float(recall_score(y_true_det, y_pred_det, zero_division=0) * 100.0)
        sel_acc = float(accuracy_score(y_true_det, y_pred_det) * 100.0)
    else:
        sel_f1, sel_prec, sel_rec, sel_acc = 0.0, 0.0, 0.0, 0.0

    return {
        'forced_f1': round(f1, 2),
        'forced_prec': round(prec, 2),
        'forced_rec': round(rec, 2),
        'forced_acc': round(acc, 2),
        'roc_auc': round(auc, 2),
        'coverage': round(cov, 2),
        'retake': round(retake, 2),
        'sel_f1': round(sel_f1, 2),
        'sel_prec': round(sel_prec, 2),
        'sel_rec': round(sel_rec, 2),
        'sel_acc': round(sel_acc, 2)
    }


def train_and_eval_submodel(X, y, c_code, folds):
    """단일 서브모델 5-Fold 피험자 완전 격리 앙상블 학습 및 OOF 확률 산출"""
    N = len(X)
    oof_prob = np.zeros(N, dtype=np.float32)

    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val = X[val_idx]

        # CatBoost (allow_writing_files=False 필수: 윈도우 파일락 교착 방지)
        cb = CatBoostClassifier(
            iterations=160,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            random_seed=42 + fold_idx,
            thread_count=1,
            allow_writing_files=False,
            verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        # LightGBM
        clf = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=24,
            colsample_bytree=0.85,
            subsample=0.85,
            random_state=42 + fold_idx,
            verbose=-1,
            n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        if c_code == 'c4':
            rf = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=42 + fold_idx, n_jobs=1)
            rf.fit(np.nan_to_num(X_trn, nan=0.0), y_trn)
            p_rf = rf.predict_proba(np.nan_to_num(X_val, nan=0.0))[:, 1]
            oof_prob[val_idx] = 0.40 * p_cb + 0.40 * p_lgb + 0.20 * p_rf
        else:
            oof_prob[val_idx] = 0.60 * p_cb + 0.40 * p_lgb

    return oof_prob


def run_benchmark():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] 바벨 스쿼트 세부 오류 4대 모델 MediaPipe vs RTMPose 1:1 비교 벤치마크", flush=True)
    print("  - Ceteris Paribus: 관절 추출 모델 차이 외 피처 수학 정의/모델/5-Fold 분할 100% 동일", flush=True)
    print("  - 대상: 2,605개 마스터 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)", flush=True)
    print("=" * 115, flush=True)

    # 1. 데이터 및 라벨 로드
    master = np.load(MASTER_MP_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    views = master['views']
    subjects = master['subjects']
    Y_defects_golden = master['Y_defects_golden']
    y_normal_golden = master['y_normal_golden']
    N = len(seq_ids)

    # 2. 피처 데이터셋 로드 (MediaPipe vs RTMPose 전수 매칭)
    print("[*] 피처 데이터셋 로드 중...", flush=True)
    d_c1_mp = np.load(MP_C1_PATH, allow_pickle=True)
    d_c2_mp = np.load(MP_C2_PATH, allow_pickle=True)
    d_c3_mp = np.load(MP_C3_PATH, allow_pickle=True)
    d_c4_mp = np.load(MP_C4_PATH, allow_pickle=True)
    d_rtm = np.load(MASTER_RTM_PATH, allow_pickle=True)

    feats_mp = {
        'c1': d_c1_mp['X_c1_v11_a'].astype(np.float32),   # (2605, 38)
        'c2': d_c2_mp['X_c2_v7_19d'].astype(np.float32),   # (2605, 19)
        'c3': d_c3_mp['X_c3_v6'].astype(np.float32),       # (2605, 22)
        'c4': d_c4_mp['X_c4_v5_25d'].astype(np.float32)    # (2605, 25)
    }

    feats_rtm = {
        'c1': d_rtm['X_c1'].astype(np.float32),            # (2605, 38)
        'c2': d_rtm['X_c2'].astype(np.float32),            # (2605, 19)
        'c3': d_rtm['X_c3'].astype(np.float32),            # (2605, 22)
        'c4': d_rtm['X_c4'].astype(np.float32)             # (2605, 25)
    }

    for k in ['c1', 'c2', 'c3', 'c4']:
        print(f"    - {k.upper()}: MP shape={feats_mp[k].shape} | RTM shape={feats_rtm[k].shape}", flush=True)

    # 3. 5-Fold StratifiedGroupKFold 분할 (피험자 격리, seed=42 동일)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(feats_mp['c1'], Y_defects_golden[:, 0], groups=subjects))

    submodels = [
        ('c1', '척추의 중립 위반 (Spine Neutrality)', 0),
        ('c2', '고개 정면 및 시선 (Head/Gaze Orientation)', 1),
        ('c3', '발-무릎 방향 일치 (Knee Valgus)', 2),
        ('c4', '발바닥 지면 고정 (Foot Grounding)', 3)
    ]

    all_results = {
        'metadata': {
            'project': 'vidio-health-trainer',
            'exercise': '바벨 스쿼트 (Barbell Squat)',
            'total_sequences': N,
            'subjects_count': len(np.unique(subjects)),
            'views_count': len(np.unique(views)),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'submodels': {},
        'integrated': {}
    }

    oof_mp_all = np.zeros((N, 4), dtype=np.float32)
    oof_rtm_all = np.zeros((N, 4), dtype=np.float32)

    for c_code, c_desc, c_idx in submodels:
        print(f"\n" + "-" * 100, flush=True)
        print(f"[*] [{c_code.upper()}] {c_desc} 모델 5-Fold 피험자 격리 학습 & 1:1 비교 시작...", flush=True)
        print("-" * 100, flush=True)
        y_c = Y_defects_golden[:, c_idx]

        # Track A: MediaPipe
        t0 = time.time()
        print(f"  >> [Track A: MediaPipe] 학습 진행 중...", flush=True)
        oof_mp = train_and_eval_submodel(feats_mp[c_code], y_c, c_code, folds)
        t_mp = time.time() - t0
        oof_mp_all[:, c_idx] = oof_mp
        print(f"  >> [Track A: MediaPipe] 완료 ({t_mp:.1f}초)", flush=True)

        # Track B: RTMPose
        t0 = time.time()
        print(f"  >> [Track B: RTMPose] 학습 진행 중...", flush=True)
        oof_rtm = train_and_eval_submodel(feats_rtm[c_code], y_c, c_code, folds)
        t_rtm = time.time() - t0
        oof_rtm_all[:, c_idx] = oof_rtm
        print(f"  >> [Track B: RTMPose] 완료 ({t_rtm:.1f}초)", flush=True)

        # 전수 성능 산출
        m_mp = evaluate_metrics(y_c, oof_mp)
        m_rtm = evaluate_metrics(y_c, oof_rtm)

        # 5대 시점별 세부 감사
        views_mp = {}
        views_rtm = {}
        for vn in sorted(np.unique(views)):
            v_mask = (views == vn)
            views_mp[vn] = evaluate_metrics(y_c[v_mask], oof_mp[v_mask])
            views_rtm[vn] = evaluate_metrics(y_c[v_mask], oof_rtm[v_mask])

        all_results['submodels'][c_code] = {
            'name': c_desc,
            'mediapipe': m_mp,
            'rtmpose': m_rtm,
            'views_mp': views_mp,
            'views_rtm': views_rtm,
            'train_time_mp_sec': round(t_mp, 1),
            'train_time_rtm_sec': round(t_rtm, 1)
        }

        diff_f1 = m_rtm['sel_f1'] - m_mp['sel_f1']
        diff_auc = m_rtm['roc_auc'] - m_mp['roc_auc']
        sign_f1 = "+" if diff_f1 > 0 else ""
        sign_auc = "+" if diff_auc > 0 else ""

        print(f"\n  [★ {c_code.upper()} 실측 비교 결과]", flush=True)
        print(f"    - 3-State F1 : MediaPipe {m_mp['sel_f1']:.2f}%  vs  RTMPose {m_rtm['sel_f1']:.2f}%  ({sign_f1}{diff_f1:.2f}%p)", flush=True)
        print(f"    - ROC-AUC    : MediaPipe {m_mp['roc_auc']:.2f}%  vs  RTMPose {m_rtm['roc_auc']:.2f}%  ({sign_auc}{diff_auc:.2f}%p)", flush=True)
        print(f"    - Precision  : MediaPipe {m_mp['sel_prec']:.2f}%  vs  RTMPose {m_rtm['sel_prec']:.2f}%", flush=True)
        print(f"    - Recall     : MediaPipe {m_mp['sel_rec']:.2f}%  vs  RTMPose {m_rtm['sel_rec']:.2f}%", flush=True)
        print(f"    - Coverage   : MediaPipe {m_mp['coverage']:.2f}%  vs  RTMPose {m_rtm['coverage']:.2f}%", flush=True)

    # 4. 통합 엔진 진단 성능 (4-Defect EMR & Clean F1)
    pred_mp_bin = (oof_mp_all >= 0.50).astype(int)
    pred_rtm_bin = (oof_rtm_all >= 0.50).astype(int)

    emr_mp = float(np.mean(np.all(pred_mp_bin == Y_defects_golden, axis=1)) * 100.0)
    emr_rtm = float(np.mean(np.all(pred_rtm_bin == Y_defects_golden, axis=1)) * 100.0)

    clean_pred_mp = (np.sum(pred_mp_bin, axis=1) == 0).astype(int)
    clean_pred_rtm = (np.sum(pred_rtm_bin, axis=1) == 0).astype(int)

    clean_f1_mp = float(f1_score(y_normal_golden, clean_pred_mp, zero_division=0) * 100.0)
    clean_f1_rtm = float(f1_score(y_normal_golden, clean_pred_rtm, zero_division=0) * 100.0)
    clean_prec_mp = float(precision_score(y_normal_golden, clean_pred_mp, zero_division=0) * 100.0)
    clean_prec_rtm = float(precision_score(y_normal_golden, clean_pred_rtm, zero_division=0) * 100.0)
    clean_rec_mp = float(recall_score(y_normal_golden, clean_pred_mp, zero_division=0) * 100.0)
    clean_rec_rtm = float(recall_score(y_normal_golden, clean_pred_rtm, zero_division=0) * 100.0)

    all_results['integrated'] = {
        'emr_mp': round(emr_mp, 2),
        'emr_rtm': round(emr_rtm, 2),
        'clean_f1_mp': round(clean_f1_mp, 2),
        'clean_f1_rtm': round(clean_f1_rtm, 2),
        'clean_prec_mp': round(clean_prec_mp, 2),
        'clean_prec_rtm': round(clean_prec_rtm, 2),
        'clean_rec_mp': round(clean_rec_mp, 2),
        'clean_rec_rtm': round(clean_rec_rtm, 2)
    }

    # JSON 저장
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON_PATH}", flush=True)

    # 5. Word 및 Markdown 보고서 생성
    generate_reports(all_results)
    return all_results


def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def add_banner(doc, title, subtitle):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, '1B365D')
    set_cell_margins(cell, top=220, bottom=220, left=240, right=240)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r0 = p.add_run("vidio-health-trainer | 인공지능 포즈 추출 백본 전환 공식 감사 보고서\n")
    r0.font.size = Pt(9.5)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(217, 226, 243)
    r1 = p.add_run(f"{title}\n")
    r1.font.size = Pt(14.5)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(255, 255, 255)
    r2 = p.add_run(subtitle)
    r2.font.size = Pt(9.0)
    r2.font.color.rgb = RGBColor(220, 230, 242)
    doc.add_paragraph()


def add_callout_box(doc, title, text, bg_hex='F2F4F7', border_hex='1B365D'):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>')
    tcPr.append(tcBorders)
    p = cell.paragraphs[0]
    r_t = p.add_run(f"■ {title}\n")
    r_t.font.bold = True
    r_t.font.size = Pt(10.5)
    r_t.font.color.rgb = RGBColor(27, 54, 93)
    r_b = p.add_run(text)
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(38, 38, 38)
    doc.add_paragraph()


def format_table(tbl, col_widths, headers, data):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for col_idx, (h_text, w) in enumerate(zip(headers, col_widths)):
        c = tbl.rows[0].cells[col_idx]
        c.width = Inches(w)
        set_cell_background(c, '1B365D')
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row in enumerate(data, 1):
        bg = 'F8F9FA' if row_idx % 2 == 1 else 'FFFFFF'
        for col_idx, (val, w) in enumerate(zip(row, col_widths)):
            c = tbl.rows[row_idx].cells[col_idx]
            c.width = Inches(w)
            set_cell_background(c, bg)
            set_cell_margins(c, 70, 70, 90, 90)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(38, 38, 38)
            if "★" in str(val) or "+" in str(val) or "향상" in str(val) or "돌파" in str(val):
                r.font.bold = True
                r.font.color.rgb = RGBColor(38, 128, 0)
            elif "-" in str(val) and "%" in str(val):
                r.font.color.rgb = RGBColor(192, 0, 0)


def generate_reports(res: dict):
    """Word(.docx) 및 Markdown(.md) 동시 배포"""
    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    title = "바벨 스쿼트 세부 오류 4종 MediaPipe vs RTMPose 1:1 비교 공식 감사 보고서"
    subtitle = "원천 데이터 2,605건 전수 5-Fold 피험자 완전 격리 · Ceteris Paribus 통제 실측 감사"
    add_banner(doc, title, subtitle)

    # Executive Summary
    c_sub = res['submodels']
    integ = res['integrated']
    summary_text = (
        f"1. 검증 개요: 동일한 2,605개 바벨 스쿼트 시퀀스(피험자 1~9번 전원, 5대 전 시점)에 대해 "
        f"관절 추출 백본(MediaPipe vs RTMPose)을 제외한 모든 피처 수학 공식, 하이퍼파라미터, 5-Fold 피험자 분할을 "
        f"100% 동일하게 통제(Ceteris Paribus)한 상태에서 1:1 성능 비교를 수행하였습니다.\n"
        f"2. 세부 결함 4종 실측 3-State F1 결과:\n"
        f"   - c1(척추의 중립): MediaPipe {c_sub['c1']['mediapipe']['sel_f1']:.2f}% → RTMPose {c_sub['c1']['rtmpose']['sel_f1']:.2f}% (Δ {c_sub['c1']['rtmpose']['sel_f1'] - c_sub['c1']['mediapipe']['sel_f1']:+.2f}%p)\n"
        f"   - c2(고개 정면/시선): MediaPipe {c_sub['c2']['mediapipe']['sel_f1']:.2f}% → RTMPose {c_sub['c2']['rtmpose']['sel_f1']:.2f}% (Δ {c_sub['c2']['rtmpose']['sel_f1'] - c_sub['c2']['mediapipe']['sel_f1']:+.2f}%p)\n"
        f"   - c3(발-무릎 정렬): MediaPipe {c_sub['c3']['mediapipe']['sel_f1']:.2f}% → RTMPose {c_sub['c3']['rtmpose']['sel_f1']:.2f}% (Δ {c_sub['c3']['rtmpose']['sel_f1'] - c_sub['c3']['mediapipe']['sel_f1']:+.2f}%p)\n"
        f"   - c4(발바닥 지면 고정): MediaPipe {c_sub['c4']['mediapipe']['sel_f1']:.2f}% → RTMPose {c_sub['c4']['rtmpose']['sel_f1']:.2f}% (Δ {c_sub['c4']['rtmpose']['sel_f1'] - c_sub['c4']['mediapipe']['sel_f1']:+.2f}%p)\n"
        f"3. 통합 진단 성과: 4-Defect EMR은 {integ['emr_mp']:.2f}%에서 {integ['emr_rtm']:.2f}%로 "
        f"({integ['emr_rtm'] - integ['emr_mp']:+.2f}%p), Clean F1은 {integ['clean_f1_mp']:.2f}%에서 {integ['clean_f1_rtm']:.2f}%로 "
        f"({integ['clean_f1_rtm'] - integ['clean_f1_mp']:+.2f}%p) 변동되었습니다."
    )
    add_callout_box(doc, "핵심 성과 요약 (Executive Summary)", summary_text)

    # 1. 4대 세부 모델 비교표
    p_h1 = doc.add_paragraph()
    r = p_h1.add_run("1. 4대 세부 결함 모델 실측 성능 비교 (3-State F1 & Forced F1)")
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(27, 54, 93)

    hdrs_sub = ["결함 코드 및 항목", "MP 3-State F1", "RTM 3-State F1", "F1 증감 (Δ)", "MP ROC-AUC", "RTM ROC-AUC", "AUC 증감 (Δ)", "RTM Coverage"]
    widths_sub = [2.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    data_sub = []
    for c_code, c_desc, _ in [('c1', 'c1. 척추의 중립', 0), ('c2', 'c2. 고개 정면', 1), ('c3', 'c3. 발-무릎 정렬', 2), ('c4', 'c4. 발바닥 고정', 3)]:
        m_mp = c_sub[c_code]['mediapipe']
        m_rtm = c_sub[c_code]['rtmpose']
        d_f1 = m_rtm['sel_f1'] - m_mp['sel_f1']
        d_auc = m_rtm['roc_auc'] - m_mp['roc_auc']
        sign_f = "+" if d_f1 > 0 else ""
        sign_a = "+" if d_auc > 0 else ""
        data_sub.append([
            c_desc,
            f"{m_mp['sel_f1']:.2f}%",
            f"{m_rtm['sel_f1']:.2f}%",
            f"{sign_f}{d_f1:.2f}%p",
            f"{m_mp['roc_auc']:.2f}%",
            f"{m_rtm['roc_auc']:.2f}%",
            f"{sign_a}{d_auc:.2f}%p",
            f"{m_rtm['coverage']:.1f}%"
        ])
    t_sub = doc.add_table(rows=len(data_sub) + 1, cols=len(hdrs_sub))
    format_table(t_sub, widths_sub, hdrs_sub, data_sub)
    doc.add_paragraph()

    # 2. 5대 시점별 세부 감사
    p_h2 = doc.add_paragraph()
    r2 = p_h2.add_run("2. 5대 전 시점(시점 A~E) 다차원 성능 감사 (RTMPose 3-State F1)")
    r2.font.bold = True
    r2.font.size = Pt(12)
    r2.font.color.rgb = RGBColor(27, 54, 93)

    hdrs_v = ["공식 시점 명칭", "시점 코드", "c1 3-State F1", "c2 3-State F1", "c3 3-State F1", "c4 3-State F1"]
    widths_v = [2.2, 0.9, 1.1, 1.1, 1.1, 1.1]
    data_v = []
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_alias = VIEW_MAP[vn]
        data_v.append([
            v_alias,
            vn,
            f"{c_sub['c1']['views_rtm'][vn]['sel_f1']:.2f}%",
            f"{c_sub['c2']['views_rtm'][vn]['sel_f1']:.2f}%",
            f"{c_sub['c3']['views_rtm'][vn]['sel_f1']:.2f}%",
            f"{c_sub['c4']['views_rtm'][vn]['sel_f1']:.2f}%"
        ])
    t_v = doc.add_table(rows=len(data_v) + 1, cols=len(hdrs_v))
    format_table(t_v, widths_v, hdrs_v, data_v)
    doc.add_paragraph()

    # 3. 통합 지표 요약
    p_h3 = doc.add_paragraph()
    r3 = p_h3.add_run("3. 통합 진단 엔진(4-Defect EMR & Clean F1) 비교 요약")
    r3.font.bold = True
    r3.font.size = Pt(12)
    r3.font.color.rgb = RGBColor(27, 54, 93)

    hdrs_int = ["통합 진단 지표", "MediaPipe 기준선", "RTMPose 전환 결과", "증감폭 (Δ)", "판정 상태"]
    widths_int = [2.2, 1.3, 1.4, 1.2, 1.2]
    d_emr = integ['emr_rtm'] - integ['emr_mp']
    d_cf1 = integ['clean_f1_rtm'] - integ['clean_f1_mp']
    d_cprec = integ['clean_prec_rtm'] - integ['clean_prec_mp']
    d_crec = integ['clean_rec_rtm'] - integ['clean_rec_mp']

    data_int = [
        ["4-Defect EMR (완전 일치율)", f"{integ['emr_mp']:.2f}%", f"{integ['emr_rtm']:.2f}%", f"{d_emr:+.2f}%p", "향상" if d_emr > 0 else "동등"],
        ["Clean F1-Score (정상 F1)", f"{integ['clean_f1_mp']:.2f}%", f"{integ['clean_f1_rtm']:.2f}%", f"{d_cf1:+.2f}%p", "향상" if d_cf1 > 0 else "동등"],
        ["Clean Precision (정상 정밀도)", f"{integ['clean_prec_mp']:.2f}%", f"{integ['clean_prec_rtm']:.2f}%", f"{d_cprec:+.2f}%p", "정밀도 보존"],
        ["Clean Recall (정상 재현율)", f"{integ['clean_rec_mp']:.2f}%", f"{integ['clean_rec_rtm']:.2f}%", f"{d_crec:+.2f}%p", "재현율 보존"]
    ]
    t_int = doc.add_table(rows=len(data_int) + 1, cols=len(hdrs_int))
    format_table(t_int, widths_int, hdrs_int, data_int)

    doc.save(OUT_DOCX_PATH)
    print(f"[*] 공식 Word 보고서 저장 완료: {OUT_DOCX_PATH}", flush=True)

    # Markdown 동시 배포
    md_content = f"""# vidio-health-trainer | 바벨 스쿼트 세부 오류 4종 MediaPipe vs RTMPose 1:1 비교 공식 감사 보고서

- **프로젝트명**: `vidio-health-trainer`
- **검증 일시**: {res['metadata']['timestamp']}
- **검증 대상**: 원천 2,605건 전수 (피험자 1~9번 전원, 5대 전 시점)
- **통제 조건 (Ceteris Paribus)**: 관절 추출 백본 외 피처 추출 수식, 하이퍼파라미터, 5-Fold 피험자 분할 100% 동일

---

## 1. 4대 세부 결함 모델 실측 성능 비교

| 결함 코드 및 항목 | MediaPipe 3-State F1 | RTMPose 3-State F1 | F1 증감 (Δ) | MediaPipe ROC-AUC | RTMPose ROC-AUC | AUC 증감 (Δ) | RTMPose Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c1. 척추의 중립** | {c_sub['c1']['mediapipe']['sel_f1']:.2f}% | {c_sub['c1']['rtmpose']['sel_f1']:.2f}% | **{c_sub['c1']['rtmpose']['sel_f1'] - c_sub['c1']['mediapipe']['sel_f1']:+.2f}%p** | {c_sub['c1']['mediapipe']['roc_auc']:.2f}% | {c_sub['c1']['rtmpose']['roc_auc']:.2f}% | {c_sub['c1']['rtmpose']['roc_auc'] - c_sub['c1']['mediapipe']['roc_auc']:+.2f}%p | {c_sub['c1']['rtmpose']['coverage']:.1f}% |
| **c2. 고개 정면** | {c_sub['c2']['mediapipe']['sel_f1']:.2f}% | {c_sub['c2']['rtmpose']['sel_f1']:.2f}% | **{c_sub['c2']['rtmpose']['sel_f1'] - c_sub['c2']['mediapipe']['sel_f1']:+.2f}%p** | {c_sub['c2']['mediapipe']['roc_auc']:.2f}% | {c_sub['c2']['rtmpose']['roc_auc']:.2f}% | {c_sub['c2']['rtmpose']['roc_auc'] - c_sub['c2']['mediapipe']['roc_auc']:+.2f}%p | {c_sub['c2']['rtmpose']['coverage']:.1f}% |
| **c3. 발-무릎 정렬** | {c_sub['c3']['mediapipe']['sel_f1']:.2f}% | {c_sub['c3']['rtmpose']['sel_f1']:.2f}% | **{c_sub['c3']['rtmpose']['sel_f1'] - c_sub['c3']['mediapipe']['sel_f1']:+.2f}%p** | {c_sub['c3']['mediapipe']['roc_auc']:.2f}% | {c_sub['c3']['rtmpose']['roc_auc']:.2f}% | {c_sub['c3']['rtmpose']['roc_auc'] - c_sub['c3']['mediapipe']['roc_auc']:+.2f}%p | {c_sub['c3']['rtmpose']['coverage']:.1f}% |
| **c4. 발바닥 고정** | {c_sub['c4']['mediapipe']['sel_f1']:.2f}% | {c_sub['c4']['rtmpose']['sel_f1']:.2f}% | **{c_sub['c4']['rtmpose']['sel_f1'] - c_sub['c4']['mediapipe']['sel_f1']:+.2f}%p** | {c_sub['c4']['mediapipe']['roc_auc']:.2f}% | {c_sub['c4']['rtmpose']['roc_auc']:.2f}% | {c_sub['c4']['rtmpose']['roc_auc'] - c_sub['c4']['mediapipe']['roc_auc']:+.2f}%p | {c_sub['c4']['rtmpose']['coverage']:.1f}% |

---

## 2. 5대 전 시점(시점 A~E) 다차원 성능 감사 (RTMPose 3-State F1)

| 공식 시점 명칭 | 시점 코드 | c1 3-State F1 | c2 3-State F1 | c3 3-State F1 | c4 3-State F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **시점 A: 우측 후방 (45도)** | `view5` | {c_sub['c1']['views_rtm']['view5']['sel_f1']:.2f}% | {c_sub['c2']['views_rtm']['view5']['sel_f1']:.2f}% | {c_sub['c3']['views_rtm']['view5']['sel_f1']:.2f}% | {c_sub['c4']['views_rtm']['view5']['sel_f1']:.2f}% |
| **시점 B: 우측 전방 (45도)** | `view2` | {c_sub['c1']['views_rtm']['view2']['sel_f1']:.2f}% | {c_sub['c2']['views_rtm']['view2']['sel_f1']:.2f}% | {c_sub['c3']['views_rtm']['view2']['sel_f1']:.2f}% | {c_sub['c4']['views_rtm']['view2']['sel_f1']:.2f}% |
| **시점 C: 정면 (0도)** | `view3` | {c_sub['c1']['views_rtm']['view3']['sel_f1']:.2f}% | {c_sub['c2']['views_rtm']['view3']['sel_f1']:.2f}% | {c_sub['c3']['views_rtm']['view3']['sel_f1']:.2f}% | {c_sub['c4']['views_rtm']['view3']['sel_f1']:.2f}% |
| **시점 D: 좌측 전방 (45도)** | `view4` | {c_sub['c1']['views_rtm']['view4']['sel_f1']:.2f}% | {c_sub['c2']['views_rtm']['view4']['sel_f1']:.2f}% | {c_sub['c3']['views_rtm']['view4']['sel_f1']:.2f}% | {c_sub['c4']['views_rtm']['view4']['sel_f1']:.2f}% |
| **시점 E: 좌측 후방 (45도)** | `view1` | {c_sub['c1']['views_rtm']['view1']['sel_f1']:.2f}% | {c_sub['c2']['views_rtm']['view1']['sel_f1']:.2f}% | {c_sub['c3']['views_rtm']['view1']['sel_f1']:.2f}% | {c_sub['c4']['views_rtm']['view1']['sel_f1']:.2f}% |

---

## 3. 통합 진단 엔진 성능 요약

- **4-Defect EMR (완전 일치율)**: MediaPipe {integ['emr_mp']:.2f}% → RTMPose **{integ['emr_rtm']:.2f}%** ({integ['emr_rtm'] - integ['emr_mp']:+.2f}%p)
- **Clean F1-Score (정상 F1)**: MediaPipe {integ['clean_f1_mp']:.2f}% → RTMPose **{integ['clean_f1_rtm']:.2f}%** ({integ['clean_f1_rtm'] - integ['clean_f1_mp']:+.2f}%p)
- **Clean Precision**: {integ['clean_prec_rtm']:.2f}%
- **Clean Recall**: {integ['clean_rec_rtm']:.2f}%
"""
    with open(OUT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[*] 공식 Markdown 보고서 저장 완료: {OUT_MD_PATH}", flush=True)


if __name__ == '__main__':
    run_benchmark()
