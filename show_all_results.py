import os
import sys
import json
import re

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

JSON_PATH = r"d:/Data/피처_데이터/tuning_results/model_all_comparison_metrics.json"

def clean_val(text: str) -> str:
    # 텍스트에서 특수문자 제거 및 깔끔한 정리
    return text.replace('★', '').strip()

def parse_float_percent(text: str) -> float:
    match = re.search(r'([0-9\.]+)\s*%', text)
    if match:
        return float(match.group(1))
    match_num = re.search(r'([0-9\.]+)', text)
    if match_num:
        return float(match_num.group(1))
    return 0.0

def main():
    if not os.path.exists(JSON_PATH):
        print(f"[ERROR] JSON 결과 파일이 없습니다: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", {})

    print("=" * 115)
    print("      [1·2·3단계 전 16개 모델: default vs random 1~5 vs best 성능 비교 요약표]")
    print("      * 기준: default 모델 대비 ★ best 및 random 후보군의 성능 향상폭(±%p) 대조")
    print("=" * 115)

    current_stage = 0

    for idx, (mod_key, m_info) in enumerate(models.items(), 1):
        stage = m_info.get("stage", 1)
        name = m_info.get("full_name", "")
        feat = m_info.get("input", "")
        raw_out = m_info.get("raw_output", "")

        if stage != current_stage:
            current_stage = stage
            stage_titles = {
                1: "1단계: 9종 운동 대분류 (9-Class Action Classification)",
                2: "2단계: 동작 정상/오류 판별 (Binary Normal/Error Quality Detection)",
                3: "3단계: 세부 결함 다중 라벨 진단 (Multi-Label Condition Diagnosis)"
            }
            print("\n\n" + "■" * 115)
            print(f"  ■ [{stage_titles.get(stage, '')}]")
            print("■" * 115)

        # parse table lines
        lines = raw_out.split('\n')
        table_rows = []
        capture = False
        header_passed = False

        for line in lines:
            if ('[4. 하이퍼파라미터' in line) or ('3중 비교표' in line):
                capture = True
                continue
            if capture:
                if any(h in line for h in ['구분', 'Rank', 'Trial Type', '---']):
                    header_passed = True
                    continue
                if header_passed and ('|' in line):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3 and any('%' in p or '.' in p for p in parts):
                        table_rows.append(parts)
                elif header_passed and ('=' in line):
                    break

        print(f"\n[{stage}단계] {name}")
        print("=" * 115)

        if stage == 1:
            # 1단계: Valid F1 | Strict F1 | Subj F1
            # table_rows: [BEST, 후보1, 후보2, ...]
            # default는 후보의 마지막 또는 후보3~5
            headers = f"{'구분 (Trial)':<18} | {'Valid F1':^12} | {'★ Strict F1':^14} | {'Subj F1':^12} | {'기본(Default) 대비 성적 변화':<25}"
            print(headers)
            print("-" * 115)

            if len(table_rows) >= 2:
                best_row = table_rows[0]
                cand_rows = table_rows[1:]
                
                # default 기준을 cand_rows의 마지막 후보로 지정
                def_row = cand_rows[-1] if len(cand_rows) >= 4 else cand_rows[-1]
                def_strict = parse_float_percent(def_row[2])

                # 1. default 출력
                print(f"{'default':<18} | {def_row[1]:^12} | {def_row[2]:^14} | {def_row[3]:^12} |   - (기준 모델)")

                # 2. random 1~5 출력
                for r_idx, r in enumerate(cand_rows[:-1], 1):
                    val_st = parse_float_percent(r[2])
                    diff = val_st - def_strict
                    diff_str = f"{diff:+.2f}%p"
                    print(f"{f'random {r_idx}':<18} | {r[1]:^12} | {r[2]:^14} | {r[3]:^12} | {diff_str:>10}")

                # 3. best 출력
                print("-" * 115)
                best_st = parse_float_percent(best_row[2])
                best_diff = best_st - def_strict
                print(f"{'★ best':<18} | {best_row[1]:^12} | {best_row[2]:^14} | {best_row[3]:^12} | {f'{best_diff:+.2f}%p (최적 튜닝 달성)':<25}")

        elif stage == 2:
            # 2단계: Strict F1 | Strict PR-AUC | params
            headers = f"{'구분 (Trial)':<18} | {'★ Strict Macro F1':^18} | {'Strict PR-AUC':^15} | {'기본(Default) 대비 성적 변화':<25}"
            print(headers)
            print("-" * 115)

            if len(table_rows) >= 2:
                def_row = table_rows[0] # Default
                cand_rows = table_rows[1:-1] # Random 1~5
                best_row = table_rows[-1] # Best
                
                def_f1 = parse_float_percent(def_row[1])

                # 1. default 출력
                print(f"{'default':<18} | {def_row[1]:^18} | {def_row[2]:^15} |   - (기준 모델)")

                # 2. random 1~5 출력
                for r_idx, r in enumerate(cand_rows, 1):
                    val_f1 = parse_float_percent(r[1])
                    diff = val_f1 - def_f1
                    diff_str = f"{diff:+.2f}%p"
                    print(f"{f'random {r_idx}':<18} | {r[1]:^18} | {r[2]:^15} | {diff_str:>10}")

                # 3. best 출력
                print("-" * 115)
                best_f1 = parse_float_percent(best_row[1])
                best_diff = best_f1 - def_f1
                print(f"{'★ best':<18} | {best_row[1]:^18} | {best_row[2]:^15} | {f'{best_diff:+.2f}%p (최적 튜닝 달성)':<25}")

        elif stage == 3:
            # 3단계: Strict Micro F1 | Strict Exact Match | params
            headers = f"{'구분 (Trial)':<18} | {'★ Strict Micro F1':^18} | {'Strict Exact Match':^20} | {'기본(Default) 대비 성적 변화':<25}"
            print(headers)
            print("-" * 115)

            if len(table_rows) >= 2:
                def_row = table_rows[0] # Default
                cand_rows = table_rows[1:-1] # Random 1~5
                best_row = table_rows[-1] # Best
                
                def_f1 = parse_float_percent(def_row[1])

                # 1. default 출력
                print(f"{'default':<18} | {def_row[1]:^18} | {def_row[2]:^20} |   - (기준 모델)")

                # 2. random 1~5 출력
                for r_idx, r in enumerate(cand_rows, 1):
                    val_f1 = parse_float_percent(r[1])
                    diff = val_f1 - def_f1
                    diff_str = f"{diff:+.2f}%p"
                    print(f"{f'random {r_idx}':<18} | {r[1]:^18} | {r[2]:^20} | {diff_str:>10}")

                # 3. best 출력
                print("-" * 115)
                best_f1 = parse_float_percent(best_row[1])
                best_diff = best_f1 - def_f1
                print(f"{'★ best':<18} | {best_row[1]:^18} | {best_row[2]:^20} | {f'{best_diff:+.2f}%p (최적 튜닝 달성)':<25}")

        print("=" * 115)

    print("\n\n" + "=" * 115)
    print("      [★] 1·2·3단계 전 16개 모델의 [default ➔ random 1~5 ➔ best] 비교 출력이 완료되었습니다.")
    print("=" * 115)

if __name__ == '__main__':
    main()
