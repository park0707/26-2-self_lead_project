"""
stiff_meta_v41.py
v4.1 5-class 스태커의 메타 피처 단일 인터페이스 (학습/추론 공용).
입력: 서브모델 확률(p1, p2, p3), 시점 코드, sd2 30D 생체역학 피처(순수 비전 유래)
출력: (N, 45) 피처 행렬.  외부에서 컬럼 인덱스를 직접 슬라이싱하지 말고 FEATURE_NAMES를 사용할 것.
"""
import numpy as np

COMBOS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1)]  # (sd1, sd2, sd3), 0번=완전 정상
COMBO_ID = {k: i for i, k in enumerate(COMBOS)}
VIEW_IDX = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
N_BIO = 30

FEATURE_NAMES = (
    ['p_sd1', 'p_sd2c', 'p_sd3'] + ['h_sd1', 'h_sd2c', 'h_sd3'] +
    ['cert_sd1', 'cert_sd2c', 'cert_sd3'] + ['p_clean_joint'] +
    [f'is_{v}' for v in VIEW_IDX] + [f'bio_{i:02d}' for i in range(N_BIO)]
)


def _entropy(p):
    e = 1e-7
    return -(p * np.log2(p + e) + (1.0 - p) * np.log2(1.0 - p + e))


def build_meta_features_v41(p1, p2, p3, views, bio30):
    """p2는 생체역학 규칙 보정 전 값이어도 됨(내부에서 max(p2, p1) 적용)."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    p2c = np.maximum(np.asarray(p2, dtype=np.float64).reshape(-1), p1)
    p3 = np.asarray(p3, dtype=np.float64).reshape(-1)
    P = np.column_stack([p1, p2c, p3])
    n = len(p1)
    onehot = np.zeros((n, 5))
    for i, v in enumerate(np.asarray(views).reshape(-1)):
        onehot[i, VIEW_IDX[str(v)]] = 1.0
    bio = np.asarray(bio30, dtype=np.float64).reshape(n, N_BIO)
    X = np.column_stack([P, _entropy(P), np.abs(P - 0.5) * 2.0, (1.0 - P).prod(axis=1), onehot, bio])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X.astype(np.float32)


def decode_class(class_id):
    """클래스 id -> {'sd1':0/1,'sd2':0/1,'sd3':0/1}"""
    a, b, c = COMBOS[int(class_id)]
    return {'sd1': a, 'sd2': b, 'sd3': c}
