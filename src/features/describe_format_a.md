# Format A: 정형 통계 요약 피처 (Tabular Kinematics Summary) 명세서

본 문서는 NovelFlow 파이프라인의 **Format A (정형 1차원 통계 요약 피처 벡터)**의 구성 체계, 수학적 정의, 차원 규격 및 모델 연계 방식을 상세히 기술합니다.

---

## 📌 1. Format A 개요 및 설계 목적

* **역할**: 시계열 64프레임 전 구간의 인체 골격 운동학 지표들을 통계량(평균, 분산, 최소/최대, 가동범위, 스냅샷)으로 압축하여 **트리 기반 머신러닝 모델(XGBoost, LightGBM, CatBoost, RandomForest 등)**의 초고속 추론(30,000 ~ 78,000 FPS)을 지원하는 1차원 정형 피처 벡터.
* **주요 특징**:
  * $O(1)$ 수준의 초저지연 연산 지원.
  * 회전/신체 크기에 불변(Scale/Rotation Invariant)한 각도 및 비율 기반 피처.
  * 최대 굴곡 시점(Lowest/Bottom Point)의 정밀 스냅샷을 포함하여 동작의 정점(Peak Phase) 형태 보존.
  * **[v2.0 업데이트]** 포워드/백워드 런지 전후 방향성 및 스티프데드 무릎 고정 분별을 위한 생체역학 물리 피처 7종($+40\text{D}$) 전격 반영.

---

## 📐 2. 조건별 Format A 차원 규격

| 데이터 조건 (Condition) | 차원 수 (Dimension) | 주요 구성 요소 | 생성 모듈 |
| :--- | :---: | :--- | :--- |
| **Condition 1 (AI Hub 2D)** | **$97\text{D}$** | 19대 2D 지표 $\times$ 5대 통계량 ($95\text{D}$) + 템포/스냅샷 지표 ($2\text{D}$) | `KinematicsExtractor2D` |
| **Condition 3 (MediaPipe 3D 레거시)** | **$121\text{D}$** | 21대 3D 지표 $\times$ 5대 통계량 ($105\text{D}$) + 최저점 스냅샷 ($16\text{D}$) | `KinematicsExtractor3D` (v1.5) |
| **Condition 3 (MediaPipe 3D 현행 v2.0)** | **$161\text{D}$** | 28대 3D 지표 $\times$ 5대 통계량 ($140\text{D}$) + 최저점 스냅샷 ($21\text{D}$) | `KinematicsExtractor3D` (v2.0) |
| **Condition 2 (Fused 2D + 3D)** | **$258\text{D}$** | 2D 피처 ($97\text{D}$) + 3D 피처 ($161\text{D}$) 직접 결합 | `FeatureFusionEngine` |

---

## 🔬 3. 3D Format A 현행 v2.0 ($161\text{D}$) 세부 피처 구성

### 3.1 28대 3D 생체역학 지표 (FEATURE_NAMES_28_3D)

각 지표마다 **5대 요약 통계량**(`mean`, `std`, `min`, `max`, `delta` [ROM])이 추출되어 $28 \times 5 = 140\text{D}$를 구성합니다.

| 그룹 | 지표명 (Feature Name) | 물리적 의미 및 수식 |
| :--- | :--- | :--- |
| **1. 8대 관절 사잇각**<br>$(8 \times 5 = 40\text{D})$ | `knee_angle_l_3d`, `knee_angle_r_3d` | 좌/우 무릎 굴곡/신전 각도 (Hip - Knee - Ankle) |
| | `hip_angle_l_3d`, `hip_angle_r_3d` | 좌/우 고관절 굴곡/신전 각도 (Torso - Hip - Knee) |
| | `ankle_angle_l_3d`, `ankle_angle_r_3d` | 좌/우 발목 굴곡 각도 (Knee - Ankle - Foot_Ref) |
| | `elbow_angle_l_3d`, `elbow_angle_r_3d` | 좌/우 팔꿈치 사잇각 (Shoulder - Elbow - Wrist) |
| **2. 자세 결함 및 정렬**<br>$(8 \times 5 = 40\text{D})$ | `torso_pitch_3d` | 체간 Sagittal 전방 경사각 ($\arctan2(Z_{\text{spine}}, \|Y_{\text{spine}}\|)$) |
| | `torso_roll_3d` | 체간 Coronal 좌우 기울기 각도 ($\arctan2(X_{\text{spine}}, \|Y_{\text{spine}}\|)$) |
| | `knee_valgus_l_3d`, `knee_valgus_r_3d` | 좌/우 무릎 내측 편위 변위 (Knee Valgus) |
| | `sh_hip_twist_3d` | 어깨선 대비 골반선 3차원 비틀림 회전각 (Twist) |
| | `knee_distance_3d` | 양 무릎 간 3차원 유클리드 거리 ($\|P_{\text{LKnee}} - P_{\text{RKnee}}\|$) |
| | `ankle_distance_3d` | 양 발목 간 3차원 스탠스 거리 ($\|P_{\text{LAnk}} - P_{\text{RAnk}}\|$) |
| | `ankle_z_spread_3d` | 양 발목 간 전후 깊이(Z) 분리 거리 ($\|Z_{\text{LAnk}} - Z_{\text{RAnk}}\|$) |
| **3. 좌우 비대칭률**<br>$(2 \times 5 = 10\text{D})$ | `knee_asymmetry_3d` | 무릎 각도 좌우 비대칭률 ($\|\theta_L - \theta_R\| / \max(\theta_L, \theta_R)$) |
| | `hip_asymmetry_3d` | 고관절 각도 좌우 비대칭률 ($\|\theta_L - \theta_R\| / \max(\theta_L, \theta_R)$) |
| **4. 운동 특화 지표**<br>$(3 \times 5 = 15\text{D})$ | `neck_spine_alignment_3d` | 목-체간-골반 3D 직선성 정렬각 (시선 이탈 검출) |
| | `barbell_body_distance_3d` | 손목-무릎 간 최소 수평 거리 (바벨 이탈 검출) |
| | `knee_hip_synchrony_3d` | 무릎 각속도와 고관절 각속도의 절대 차이 (동조성) |
| **5. [v2.0 신규] 취약점 분별 특화 지표**<br>$(7 \times 5 = 35\text{D})$ | `signed_ankle_z_rel_l_3d`, `signed_ankle_z_rel_r_3d` | 골반 대비 좌/우 발목 부호화 Z축 변위 (전진: 음수, 후진: 양수) |
| | `stepping_foot_dx_3d`, `stepping_foot_dz_3d` | 스텝 발 3차원 이동 궤적 변위 ($dx, dz$) |
| | `shank_verticality_angle_l_3d`, `shank_verticality_angle_r_3d` | 좌/우 정강이 지면 수직 각도 ($0^\circ$ 완전 수직) |
| | `hip_to_knee_rom_ratio_3d` | 고관절 가동범위 대비 무릎 가동범위 비율 ($\Delta \theta_{\text{Hip}} / \Delta \theta_{\text{Knee}}$) |

---

### 3.2 최대 굴곡 시점(Lowest/Bottom Point) 21대 스냅샷 피처 ($21\text{D}$)

무릎 평균 각도($(\theta_{KL} + \theta_{KR})/2$)가 최소가 되는 최하점 프레임($t_{\text{bottom}}$)에서 아래 21개 지표의 정적 순간값을 캡처합니다.

1. `knee_angle_l_3d_bottom`
2. `knee_angle_r_3d_bottom`
3. `hip_angle_l_3d_bottom`
4. `hip_angle_r_3d_bottom`
5. `ankle_angle_l_3d_bottom`
6. `ankle_angle_r_3d_bottom`
7. `elbow_angle_l_3d_bottom`
8. `elbow_angle_r_3d_bottom`
9. `torso_pitch_3d_bottom`
10. `torso_roll_3d_bottom`
11. `knee_valgus_l_3d_bottom`
12. `knee_valgus_r_3d_bottom`
13. `sh_hip_twist_3d_bottom`
14. `knee_distance_3d_bottom`
15. `ankle_z_spread_3d_bottom`
16. `neck_spine_alignment_3d_bottom`
17. `signed_ankle_z_rel_l_3d_bottom`
18. `signed_ankle_z_rel_r_3d_bottom`
19. `shank_verticality_angle_l_3d_bottom`
20. `shank_verticality_angle_r_3d_bottom`
21. `hip_to_knee_rom_ratio_3d_bottom`

**총 차원 수**: $140\text{D} + 21\text{D} = \mathbf{161\text{D}}$

---

## 🔄 4. 버전 관리 이력 (Changelog)

* **v1.0 (2026-09-02)**: AI Hub 2D 19대 지표 기반 Format A ($97\text{D}$) 최초 구축.
* **v1.5 (2026-09-05)**: MediaPipe 3D 21대 지표 기반 Format A ($121\text{D} / 151\text{D}$) 도입.
* **v2.0 (2026-09-09 적용)**: 런지 Z축 부호화 변위 및 데드리프트 정강이 수직각/ROM 비율 7대 신규 물리 피처 반영 완료 ($161\text{D}$).
