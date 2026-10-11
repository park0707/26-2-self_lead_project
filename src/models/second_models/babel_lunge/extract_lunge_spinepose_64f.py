# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Module: extract_lunge_spinepose_64f.py
Purpose: 원본 런지 프레임(16F/시점)에서 SpinePose(37Kps + 3D Lifting)를 직접 구동해
         lunge c4 캐시(787 시퀀스)의 9대 척추 분절 (64, 9, 3) 3D 좌표를 추출.
- 100% 순수 비전(이미지) 입력. AI-Hub JSON 좌표/라벨은 입력으로 쓰지 않음 (AGENTS.md 격리 원칙).
- 16개 원본 프레임 추론 후 64프레임 3차 스플라인 보간 (AI-Hub 학습용 데이터 표준).
- 바운딩 박스: 동일 시점 MediaPipe 2D(64f)에서 산출 (RF-DETR 탐지기 생략).
- 시점 매핑(오버레이로 검증): view5=시점A, view2=시점B, view3=시점C, view4=시점D, view1=시점E
  (AGENTS.md 공식 5대 시점 규격과 일치).
- 결과는 전부 메모리에 모아 마지막에 1회 저장 (체크포인트 없음, 실패 시 재실행).
"""

import os
import sys
import time
import numpy as np
import cv2
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]  # S1,L5,L3,L1,T8,T3,C7,C4,C1
VIEW_TO_LETTER = {'view5': 'A', 'view2': 'B', 'view3': 'C', 'view4': 'D', 'view1': 'E'}

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"
CACHE_V3 = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
OUT_PATH = r"D:\Data\피처_데이터\lunge_c4_spinepose_64f.npz"
CKPT_DIR = r"D:\Data\피처_데이터\temp_spinepose_lunge"

_EST = None


def get_estimator():
    global _EST
    if _EST is None:
        from spinepose import SpinePoseEstimator
        _EST = SpinePoseEstimator(mode="small", enable_lifting=True)
    return _EST


def to_64f(arr: np.ndarray) -> np.ndarray:
    T = arr.shape[0]
    if T == 64:
        return arr.astype(np.float32)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(np.linspace(0, 1, T), arr.reshape(T, -1), kind=kind, axis=0, fill_value='extrapolate')
    return f(np.linspace(0, 1, 64)).reshape((64,) + arr.shape[1:]).astype(np.float32)


def process_base(args):
    idx, base, views = args
    ck = os.path.join(CKPT_DIR, f"{idx:04d}.npz")
    if os.path.exists(ck):
        try:
            c = np.load(ck)
            return base, {k: c[k] for k in c.files}, None
        except Exception:
            pass
    folder = os.path.join(RAW_IMG_DIR, base)
    if not os.path.isdir(folder):
        return base, None, f"no folder {folder}"
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png')))
    est = get_estimator()
    res = {}
    for v in views:
        L = VIEW_TO_LETTER[v]
        imgs = [f for f in names if ('시점' + L) in f]
        if len(imgs) < 4:
            continue
        mp_p = os.path.join(MP33_DIR, f"{base}_{v}.npz")
        k2d64 = np.load(mp_p)['kps_33_2d_64f'] if os.path.exists(mp_p) else None
        s3, s2, sc = [], [], []
        for t, fn in enumerate(imgs):
            img = cv2.imdecode(np.fromfile(os.path.join(folder, fn), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if k2d64 is not None:
                k = k2d64[int(round(t * 63 / max(1, len(imgs) - 1)))]
                x0, x1, y0, y1 = k[:, 0].min(), k[:, 0].max(), k[:, 1].min(), k[:, 1].max()
                dx, dy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
                bb = [max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy)]
            else:
                bb = [w * 0.15, h * 0.05, w * 0.85, h * 0.95]
            k2, sco, k3 = est.estimate(img, np.array([bb], np.float32))
            s3.append(k3[0, SPINEPOSE_9_INDICES])
            s2.append(k2[0, SPINEPOSE_9_INDICES])
            sc.append(float(np.mean(sco[0, SPINEPOSE_9_INDICES])))
        res[f"{base}_{v}__s3"] = to_64f(np.array(s3, np.float32))
        res[f"{base}_{v}__s2"] = to_64f(np.array(s2, np.float32))
        res[f"{base}_{v}__sc"] = np.float32(np.mean(sc))
    if res:
        np.savez(ck, **res)
    return base, res, None


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    d = np.load(CACHE_V3, allow_pickle=True)
    clip_ids = [str(c) for c in d['clip_ids']]
    views = [str(v) for v in d['views']]
    groups = {}
    for cid, v in zip(clip_ids, views):
        assert cid.endswith('_' + v), cid
        groups.setdefault(cid[:-(len(v) + 1)], []).append(v)
    tasks = [(i, b, vs) for i, (b, vs) in enumerate(groups.items())]
    print(f"[*] clips={len(clip_ids)}, base folders={len(tasks)}")

    results = {}
    t0 = time.time()
    done = 0
    pending = list(tasks)
    chunk = 12
    attempts = 0
    while pending and attempts < 15:
        batch, rest = pending[:chunk], pending[chunk:]
        ex = ProcessPoolExecutor(max_workers=3)
        futs = {ex.submit(process_base, t): t for t in batch}
        failed = []
        try:
            for fu in as_completed(futs, timeout=900):
                t = futs[fu]
                try:
                    base, res, err = fu.result()
                except Exception as e:  # 워커 크래시 -> 재시도 대상
                    failed.append(t)
                    print(f"[WARN] task {t[0]} 실패: {e}", flush=True)
                    continue
                done += 1
                if err:
                    print(f"[WARN] {base}: {err}")
                elif res:
                    results.update(res)
        except Exception as e:  # 타임아웃(워커 정지) -> 미완료 작업 재시도
            attempts += 1
            failed += [futs[f] for f in futs if not f.done()]
            print(f"[WARN] 청크 타임아웃/오류({e}) -> 재시도 {attempts}/15", flush=True)
        finally:
            for p in list(getattr(ex, '_processes', {}).values()):
                try:
                    p.kill()
                except Exception:
                    pass
            ex.shutdown(wait=False, cancel_futures=True)
        pending = failed + rest
        el = time.time() - t0
        print(f"  {done}/{len(tasks)} | {el/60:.1f}분 경과", flush=True)

    if pending:
        print(f"[ERR] 미처리 폴더 {len(pending)}개")
    N = len(clip_ids)
    S3 = np.zeros((N, 64, 9, 3), np.float32)
    S2 = np.zeros((N, 64, 9, 2), np.float32)
    SC = np.zeros(N, np.float32)
    valid = np.zeros(N, bool)
    for i, cid in enumerate(clip_ids):
        if f"{cid}__s3" in results:
            S3[i], S2[i], SC[i] = results[f"{cid}__s3"], results[f"{cid}__s2"], results[f"{cid}__sc"]
            valid[i] = True
    np.savez_compressed(OUT_PATH, spine3d=S3, spine2d=S2, score=SC, valid=valid,
                        clip_ids=np.array(clip_ids), views=np.array(views))
    print(f"[OK] saved {OUT_PATH} | valid {valid.mean()*100:.1f}% ({valid.sum()}/{N}) | NaN={int(np.isnan(S3).sum())}")


if __name__ == '__main__':
    main()
