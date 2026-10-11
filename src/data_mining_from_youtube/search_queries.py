# -*- coding: utf-8 -*-
"""
9종 운동별 한/영 정밀 검색 쿼리 및 제외어 매트릭스
"""
from typing import Dict, List, Any

GLOBAL_NEGATIVE_KEYWORDS = "-shorts -fail -compilation -meme -podcast -reaction -vlog -tiktok -challenge"

EXERCISE_QUERIES: Dict[str, Dict[str, Any]] = {
    '01_바벨_스쿼트': {
        'idx': 0,
        'name_ko': '바벨 스쿼트',
        'primary_queries': [
            'squat form check side view',
            'barbell back squat form side view',
            '바벨 스쿼트 자세 측면'
        ],
        'disambiguation_filter': '-front -overhead -smith',
        'target_intent': '무릎-골반 동조 굴곡 및 깊은 풀 스쿼트 가동범위 확보'
    },
    '02_바벨_데드리프트': {
        'idx': 1,
        'name_ko': '바벨 데드리프트',
        'primary_queries': [
            '바벨 컨벤셔널 데드리프트 자세',
            'barbell conventional deadlift side view',
            'deadlift form check'
        ],
        'disambiguation_filter': '-sumo -romanian -rdl',
        'target_intent': '바닥 출발 순차적 무릎/고관절 신전 및 수직 바벨 궤적 확보'
    },
    '03_바벨_스티프_데드리프트': {
        'idx': 2,
        'name_ko': '바벨 스티프 데드리프트',
        'primary_queries': [
            '바벨 스티프 레그 데드리프트 자세',
            'barbell stiff leg deadlift SLDL form',
            'stiff legged deadlift side view'
        ],
        'disambiguation_filter': '-romanian -rdl -squat',
        'target_intent': '무릎 관절 고정 상태의 순수 햄스트링 힌지 확보'
    },
    '04_바벨_런지': {
        'idx': 3,
        'name_ko': '바벨 런지 (고정형)',
        'primary_queries': [
            '바벨 런지 정석 자세',
            'barbell static lunge split squat form',
            'stationary barbell lunge'
        ],
        'disambiguation_filter': '-walking -dumbbell',
        'target_intent': '기저면 고정 상태의 전후 다리 비대칭 굴곡 패턴 확보'
    },
    '05_스텝_포워드_다이나믹_런지': {
        'idx': 4,
        'name_ko': '스텝 포워드 다이나믹 런지',
        'primary_queries': [
            '스텝 포워드 런지 자세',
            'barbell step forward lunge tutorial',
            'forward lunge form side view'
        ],
        'disambiguation_filter': '-reverse -backward',
        'target_intent': '전방 발목 스텝 전진 및 착지 후 복귀 3단계 위상 확보'
    },
    '06_스텝_백워드_다이나믹_런지': {
        'idx': 5,
        'name_ko': '스텝 백워드 다이나믹 런지',
        'primary_queries': [
            '스텝 백워드 런지 자세',
            'barbell reverse lunge side view',
            'barbell step backward lunge form'
        ],
        'disambiguation_filter': '-forward',
        'target_intent': '후방 발목 스텝 후퇴 및 기준발 위치 불변 패턴 확보'
    },
    '07_사이드_런지': {
        'idx': 6,
        'name_ko': '사이드 런지',
        'primary_queries': [
            '사이드 런지 자세 정면',
            'barbell lateral lunge form front view',
            'barbell cossack squat side lunge'
        ],
        'disambiguation_filter': '-dumbbell -kettlebell',
        'target_intent': '관상면(X축) 횡방향 확장 및 극단 편측 굴곡 패턴 확보'
    },
    '08_크로스_런지': {
        'idx': 7,
        'name_ko': '크로스 런지 (커시 런지)',
        'primary_queries': [
            '크로스 런지 커시 런지 자세',
            'barbell curtsy lunge cross lunge',
            'curtsy lunge form check'
        ],
        'disambiguation_filter': '-dumbbell',
        'target_intent': '정중선 대각선 교차 궤적 및 골반 회전 비틀림 패턴 확보'
    },
    '09_굿모닝': {
        'idx': 8,
        'name_ko': '굿모닝',
        'primary_queries': [
            '바벨 굿모닝 운동 자세',
            'barbell good morning exercise form side view',
            'good morning workout form check'
        ],
        'disambiguation_filter': '-squat -deadlift',
        'target_intent': '승모근 바벨 견착 유지 및 고관절 힌지 상체 전경 확보'
    }
}

def get_search_query(exercise_key: str, query_index: int = 0) -> str:
    """운동 키 및 쿼리 인덱스를 받아 완전한 yt-dlp 검색 문자열 생성"""
    info = EXERCISE_QUERIES.get(exercise_key)
    if not info:
        # 키 직접 매칭 안될 시 유사 키 탐색
        for k, v in EXERCISE_QUERIES.items():
            if exercise_key in k or exercise_key in v['name_ko']:
                info = v
                break
    if not info:
        raise ValueError(f"Unknown exercise key: {exercise_key}")

    queries = info['primary_queries']
    q = queries[query_index % len(queries)]
    disam = info.get('disambiguation_filter', '')
    return f"{q} {disam} {GLOBAL_NEGATIVE_KEYWORDS}".strip()
