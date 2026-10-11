# 이전 스쿼트 파일럿 실험 아카이브 (Legacy Squat Pilot Experiments)

본 디렉토리는 클린 리셋(Clean Reset) 이전에 수행되었던 복잡한 다계층 파이프라인 및 다차원 피처 엔지니어링 실험 스크립트와 결과 캐시들을 안전하게 보관하는 아카이브 폴더입니다.

---

## 1. 보관된 스크립트 목록
* 	est_pilot_squat_sota.py : SOTA 4대 솔루션(최하사점 169D, Veto, 마할라노비스 게이트) 복합 검증기
* 	est_squat_trend_sota.py : 동적 협응(185D) 및 위상 경향성 검증기
* 	sm_1d_features.py : 1D-TSM 시공간 위상차(213D) 피처 추출기
* ine_grain_features.py : AI-Hub 원천 JSON 기반 얼굴/발 세밀 관절 피처 추출기
* orm_coordination.py : 체간-대퇴골 Cyclogram 및 동적 협응 피처 추출기
* calibrated_gating.py : 연속형 소프트 퓨전 및 결함 거부권(Veto) 수식 엔진
* mahalanobis_gate.py : 정상 다양체 마할라노비스 거리 이상 탐지기
* pilot_squat_tournament.py : 4대 세부 결함 6대 알고리즘 토너먼트 선발기
* 	est_pilot_squat.py : 초기 1-Stage 통합 진단 실측 스크립트
* generate_*_report.py : 이전 단계별 Word 보고서 생성 유틸리티

## 2. 보관된 캐시 및 결과 파일 (.npz / .json)
* pilot_squat_oof_cache.npz (169D OOF 확률 캐시)
* pilot_squat_trend_185_cache.npz (185D OOF 확률 캐시)
* pilot_squat_fine_197_cache.npz (197D OOF 확률 캐시)
* pilot_squat_tsm_213_cache.npz (213D OOF 확률 캐시)
* pilot_squat_champions.json (4대 결함 선발 챔피언 메타데이터)
* pilot_squat_*_results.json (이전 실험 결과 요약본)

---
*참고: 현재 공식 운영되는 클린 리셋 단일 모델 스크립트는 상위 디렉토리의 	est_babel_squat.py 입니다.*