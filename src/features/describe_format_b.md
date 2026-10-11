# Format B: 3D 시계열 텐서 피처 (Temporal Kinematics Tensor) 명세서

본 문서는 NovelFlow 파이프라인의 **Format B (64프레임 시공간 시계열 텐서)**의 멀티스트림 구성 체계, 수학적 정의, 차원 규격 및 딥러닝 모델 연계 방식을 상세히 기술합니다.

---

## 📌 1. Format B 개요 및 설계 목적

* **역할**: 운동 1주기(Repetition) 동안 발생하는 인체 관절의 3차원 위치 궤적, 뼈대 벡터, 각도 변화, 각속도 동역학을 **64프레임 균등 시간 보간 시계열 텐서 $(64, D)$**로 변환하여 **시계열 딥러닝 신경망(Bi-GRU, 1D-CNN, TCN, Self-Attention 등)**의 입력으로 사용하는 고차원 피처 텐서.
* **주요 특징**:
  * 동작의 시작(Start) $\rightarrow$ 전환점(Bottom) $\rightarrow$ 복귀(Finish) 전 과정의 **시간적 전후 맥락(Temporal Context)** 완벽 보존.
  * 7개 독립 스트림(뼈대, 각도, 정렬, 비대칭, 각속도, 특화, **취약점 분별**)으로 구조화된 멀티스트림 생체역학 융합 텐서.
  * 채널 드롭아웃(DropKeypoint) 및 시간 왜곡(Time Warping) 증강과 완벽 결합.
  * **[v2.0 업데이트]** 런지 전후 방향성(Signed Ankle Z, Stepping Vector) 및 스티프데드 정강이 수직각(Shank Verticality) 전용 Stream 7($+7\text{D}$) 전격 탑재.

---

## 📐 2. 조건별 Format B 차원 규격

| 데이터 조건 (Condition) | 텐서 크기 (Shape) | 채널 수 ($D$) | 주요 구성 요소 | 생성 모듈 |
| :--- | :---: | :---: | :--- | :--- |
| **Condition 1 (AI Hub 2D)** | `(64, 69)` | **$69\text{D}$** | Bone(42) + Angle(8) + Align(6) + Asym(2) + Vel(8) + Spec(3) | `KinematicsExtractor2D` |
| **Condition 3 (MediaPipe 3D 레거시)** | `(64, 85)` | **$85\text{D}$** | Bone(56) + Angle(8) + Align(8) + Asym(2) + Vel(8) + Spec(3) | `KinematicsExtractor3D` (v1.5) |
| **Condition 3 (MediaPipe 3D 현행 v2.0)** | `(64, 92)` | **$92\text{D}$** | 기본 85D + Stream 7 취약점 분별 7D (`signed_z`, `step_vec`, `shank_angle`) | `KinematicsExtractor3D` (v2.0) |
| **Condition 2 (Fused 2D + 3D)** | `(64, 161)` | **$161\text{D}$** | 2D 텐서 (69D) + 3D 텐서 (92D) 채널 결합 | `FeatureFusionEngine` |

---

## 🔬 3. 3D Format B 현행 v2.0 ($92\text{D}$) 7대 멀티스트림 세부 구조

Format B는 프레임 $t \in [0, 63]$에서 총 7개 스트림으로 구성됩니다.

```
Format B [64, 92]
├── Stream 1: 3D Bone Edges (14개 뼈대 x 4 = 56D) [idx 0..55]
├── Stream 2: 3D Joint Angles (8개 관절 사잇각 = 8D) [idx 56..63]
├── Stream 3: 3D Alignments & Defect (8개 정렬/결함 = 8D) [idx 64..71]
├── Stream 4: 3D Asymmetry (2개 좌우 비대칭률 = 2D) [idx 72..73]
├── Stream 5: 3D Angular Velocity (8개 관절 각속도 = 8D) [idx 74..81]
├── Stream 6: 3D Special Metrics (3개 운동 특화 지표 = 3D) [idx 82..84]
└── Stream 7: [v2.0 신규] Disambiguation (7개 취약점 분별 특화 = 7D) [idx 85..91]
```

### 3.1 Stream 1: 14대 뼈대 분절 벡터 (Bone Edges, $56\text{D}$)
14개 신체 연결 엣지에 대해 $(\Delta X, \Delta Y, \Delta Z, \text{Length}_{3D})$ 4개 채널을 산출 ($14 \times 4 = 56\text{D}$).

| Index 범위 | 뼈대 엣지 명칭 (Skeleton Edge) | 연결 관절 (Src $\rightarrow$ Dst) |
| :--- | :--- | :--- |
| `0 .. 3` | `Pelvis-Torso` | 골반 중심 (0) $\rightarrow$ 체간 중심 (1) |
| `4 .. 7` | `Torso-Neck` | 체간 중심 (1) $\rightarrow$ 목 (14) |
| `8 .. 11` | `Torso-LShoulder` | 체간 중심 (1) $\rightarrow$ 좌측 어깨 (2) |
| `12 .. 15` | `Torso-RShoulder` | 체간 중심 (1) $\rightarrow$ 우측 어깨 (3) |
| `16 .. 19` | `LShoulder-LElbow` | 좌측 어깨 (2) $\rightarrow$ 좌측 팔꿈치 (4) |
| `20 .. 23` | `LElbow-LWrist` | 좌측 팔꿈치 (4) $\rightarrow$ 좌측 손목 (6) |
| `24 .. 27` | `RShoulder-RElbow` | 우측 어깨 (3) $\rightarrow$ 우측 팔꿈치 (5) |
| `28 .. 31` | `RElbow-RWrist` | 우측 팔꿈치 (5) $\rightarrow$ 우측 손목 (7) |
| `32 .. 35` | `Pelvis-LHip` | 골반 중심 (0) $\rightarrow$ 좌측 고관절 (8) |
| `36 .. 39` | `Pelvis-RHip` | 골반 중심 (0) $\rightarrow$ 우측 고관절 (9) |
| `40 .. 43` | `LHip-LKnee` | 좌측 고관절 (8) $\rightarrow$ 좌측 무릎 (10) |
| `44 .. 47` | `LKnee-LAnkle` | 좌측 무릎 (10) $\rightarrow$ 좌측 발목 (12) |
| `48 .. 51` | `RHip-RKnee` | 우측 고관절 (9) $\rightarrow$ 우측 무릎 (11) |
| `52 .. 55` | `RKnee-RAnkle` | 우측 무릎 (11) $\rightarrow$ 우측 발목 (13) |

---

### 3.2 Stream 2 ~ 7 세부 지표 ($36\text{D}$)

| 스트림 | Index | 지표명 (Feature Name) | 물리적 의미 |
| :--- | :---: | :--- | :--- |
| **Stream 2**<br>(Angles 8D) | `56, 57` | `knee_angle_l_3d`, `knee_angle_r_3d` | 좌/우 무릎 3차원 사잇각 |
| | `58, 59` | `hip_angle_l_3d`, `hip_angle_r_3d` | 좌/우 고관절 3차원 사잇각 |
| | `60, 61` | `ankle_angle_l_3d`, `ankle_angle_r_3d` | 좌/우 발목 3차원 사잇각 |
| | `62, 63` | `elbow_angle_l_3d`, `elbow_angle_r_3d` | 좌/우 팔꿈치 3차원 사잇각 |
| **Stream 3**<br>(Align 8D) | `64` | `torso_pitch_3d` | 체간 전방 경사각 (Sagittal) |
| | `65` | `torso_roll_3d` | 체간 좌우 기울기각 (Coronal) |
| | `66, 67` | `knee_valgus_l_3d`, `knee_valgus_r_3d` | 좌/우 무릎 내측 편위 |
| | `68` | `sh_hip_twist_3d` | 어깨-골반 수평 비틀림각 |
| | `69` | `knee_distance_3d` | 양 무릎 간 3D 유클리드 거리 |
| | `70` | `ankle_distance_3d` | 양 발목 간 3D 스탠스 거리 |
| | `71` | `ankle_z_spread_3d` | 양 발목 전후 분리 거리 |
| **Stream 4**<br>(Asym 2D) | `72, 73` | `knee_asymmetry_3d`, `hip_asymmetry_3d` | 무릎 / 고관절 좌우 비대칭률 |
| **Stream 5**<br>(Vel 8D) | `74 .. 81` | 8대 관절 각속도 ($d\theta/dt$) | Stream 2의 1차 시간 미분값 |
| **Stream 6**<br>(Spec 3D) | `82` | `neck_spine_alignment_3d` | 목-체간-골반 직선 정렬 |
| | `83` | `barbell_body_distance_3d` | 손목-무릎 최소 수평 거리 |
| | `84` | `knee_hip_synchrony_3d` | 무릎-고관절 각속도 절대 차이 |
| **Stream 7 [v2.0 신규]**<br>(Disam 7D) | `85, 86` | `signed_ankle_z_rel_l_3d`, `signed_ankle_z_rel_r_3d` | 골반 대비 좌/우 발목 부호화 Z축 변위 (전진: 음수, 후진: 양수) |
| | `87, 88` | `stepping_foot_dx_3d`, `stepping_foot_dz_3d` | 스텝 발 3D 이동 궤적 변위 ($dx, dz$) |
| | `89, 90` | `shank_verticality_angle_l_3d`, `shank_verticality_angle_r_3d` | 좌/우 정강이 지면 수직 각도 ($0^\circ$ 완전 수직) |
| | `91` | `hip_to_knee_rom_ratio_3d` | 고관절 가동범위 대비 무릎 가동범위 비율 ($\Delta \theta_{\text{Hip}} / \Delta \theta_{\text{Knee}}$) |

---

## 🔄 4. 버전 관리 이력 (Changelog)

* **v1.0 (2026-09-02)**: AI Hub 2D 6개 스트림 기반 Format B `(64, 69)` 최초 구축.
* **v1.5 (2026-09-05)**: MediaPipe 3D 6개 스트림 기반 Format B `(64, 85)` 도입.
* **v2.0 (2026-09-09 적용)**: 런지 Z축 부호화 및 정강이 지면 각도/ROM 비율 Stream 7 추가 완료 `(64, 92)`.
