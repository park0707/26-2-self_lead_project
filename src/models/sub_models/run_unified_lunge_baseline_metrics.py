# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
try:
    from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_18d_features_v4 as feature_extractor
except ImportError:
    pass

DIRS = {
    '04': r"D:\Data\MediaPipe\전처리\04_바벨_런지",
    '05': r"D:\Data\MediaPipe\전처리\05_스텝_포워드_다이나믹_런지",
    '06': r"D:\Data\MediaPipe\전처리\06_스텝_백워드_다이나믹_런지"
}

def load_data():
    X_list, y_list, group_list, src_list = [], [], [], []
    for src_id, dpath in DIRS.items():
        npz_files = sorted(glob.glob(os.path.join(dpath, "*.npz")))
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                kps = data['keypoints_2d_norm']
                conds = data['conditions']
                try:
                    meta = json.loads(str(data['metadata'].item()))
                except:
                    meta = json.loads(str(data['metadata']))
                
                is_defect = 1 if not conds[2] else 0
                sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
                
                f_vec = feature_extractor(kps)
                X_list.append(f_vec)
                y_list.append(is_defect)
                group_list.append(sub_id)
                src_list.append(src_id)
            except Exception:
                continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(group_list, dtype=np.int32), np.array(src_list)

def print_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # cm[0][0]: True Negative (정상을 정상으로)
    # cm[0][1]: False Positive (정상을 오류로 - 억울한 기각)
    # cm[1][0]: False Negative (오류를 정상으로 - 오진단 통과)
    # cm[1][1]: True Positive (오류를 오류로 - 정확한 지적)
    
    print(f"\n[{name} 모델 결과]")
    print(f" - 정확도(Accuracy): {acc:.4f} | F1 Score: {f1:.4f}")
    print(f" - 정밀도(Precision): {prec:.4f} (에이전트가 '오류'라고 지적했을 때 진짜 오류일 확률)")
    print(f" - 재현율(Recall)   : {rec:.4f} (실제 유저의 '오류' 중 에이전트가 잡아낸 비율)")
    print(f" - 오진 횟수 (총 {len(y_true)}번의 테스트 중):")
    print(f"     * 멀쩡한데 다시 하라고 억울하게 지적함 (False Positive): {cm[0][1]} 번")
    print(f"     * 틀렸는데 잘했다고 통과시켜 줌 (False Negative)    : {cm[1][0]} 번")

def main():
    X, y, groups, sources = load_data()
    
    mask_04 = sources == '04'
    X_04, y_04, groups_04 = X[mask_04], y[mask_04], groups[mask_04]
    
    mask_dyn = (sources == '05') | (sources == '06')
    X_dyn, y_dyn = X[mask_dyn], y[mask_dyn]
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X_04, y_04, groups_04))
    tr_idx, te_idx = splits[0]
    
    X_04_train, y_04_train = X_04[tr_idx], y_04[tr_idx]
    X_04_test, y_04_test = X_04[te_idx], y_04[te_idx]
    
    cb_baseline = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_baseline.fit(X_04_train, y_04_train)
    preds_base = cb_baseline.predict(X_04_test)
    
    X_uni_train = np.vstack([X_04_train, X_dyn])
    y_uni_train = np.hstack([y_04_train, y_dyn])
    
    cb_unified = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_unified.fit(X_uni_train, y_uni_train)
    preds_uni = cb_unified.predict(X_04_test)
    
    print_metrics(y_04_test, preds_base, "A. 기존 방식 (바벨 런지로만 학습)")
    print_metrics(y_04_test, preds_uni, "B. 통합 방식 (다이나믹 런지 추가 학습)")
    
if __name__ == "__main__":
    main()
