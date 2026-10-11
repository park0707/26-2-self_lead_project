import os
import glob
import json

# Description text to expected condition defects
# 5 conditions:
# c1: 앞다리 무릎 각도 90도 (False if '앞다리 조금접')
# c2: 몸통 발 무릎 방향 일치여부 (False if '발방향 바깥쪽')
# c3: 뒤다리 무릎 각도 90도 (False if '안굽히는 다리 구부정')
# c4: 척추의 중립 (False if '몸통 옆으로 갸우뚱' or '보고' in ['왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고'])
# c5: 상체의 과조한 숙임/젖힘 여부 (False if '상체 뒤로 젖힘')

label_dir = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
json_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

def parse_expected_conditions_from_desc(desc: str):
    # Returns expected boolean values for [c1, c2, c3, c4, c5] (True=Normal, False=Defect)
    if "정자세" in desc and "1앞다리 90도" in desc:
        return [True, True, True, True, True]
    
    # Defaults
    c1 = not ("앞다리 조금접" in desc)
    c2 = not ("발방향 바깥쪽" in desc)
    c3 = not ("안굽히는 다리 구부정" in desc)
    
    # c4: 몸통 옆으로 갸우뚱, 또는 왼쪽보고/오른쪽보고/하늘보고/바닥보고
    has_c4_neck = any(k in desc for k in ["왼쪽보고", "오른쪽보고", "하늘보고", "바닥보고"])
    has_c4_body = "몸통 옆으로 갸우뚱" in desc
    c4 = not (has_c4_neck or has_c4_body)
    
    # c5: 상체 뒤로 젖힘
    c5 = not ("상체 뒤로 젖힘" in desc)
    
    return [c1, c2, c3, c4, c5]

def parse_expected_c4_only(desc: str):
    # Only for c4 (척추의 중립)
    has_c4_neck = any(k in desc for k in ["왼쪽보고", "오른쪽보고", "하늘보고", "바닥보고"])
    has_c4_body = "몸통 옆으로 갸우뚱" in desc
    return not (has_c4_neck or has_c4_body)

key_results = []
key_seen = set()

all_match_count = 0
c4_match_count = 0
total_count = len(json_files)

mismatch_examples = []

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        d = json.load(f)
    tinfo = d.get("type_info", {})
    key = tinfo.get("key", "")
    desc = tinfo.get("description", "")
    conds = tinfo.get("conditions", [])
    
    actual_vals = [c.get("value", True) for c in conds]
    # In case there are more/less than 5
    actual_5 = actual_vals[:5] if len(actual_vals) >= 5 else actual_vals + [True]*(5-len(actual_vals))
    
    # expected with neck as c4
    exp_with_neck = parse_expected_conditions_from_desc(desc)
    
    # expected with only body lean as c4 (if neck is ignored)
    exp_no_neck = [
        not ("앞다리 조금접" in desc),
        not ("발방향 바깥쪽" in desc),
        not ("안굽히는 다리 구부정" in desc),
        not ("몸통 옆으로 갸우뚱" in desc),
        not ("상체 뒤로 젖힘" in desc)
    ]
    if "정자세" in desc and "1앞다리 90도" in desc:
        exp_no_neck = [True, True, True, True, True]

    match_all_neck = (actual_5 == exp_with_neck)
    match_all_noneck = (actual_5 == exp_no_neck)
    
    match_c4_neck = (actual_5[3] == exp_with_neck[3])
    match_c4_noneck = (actual_5[3] == exp_no_neck[3])
    
    if key not in key_seen:
        key_seen.add(key)
        key_results.append({
            "key": key,
            "desc": desc,
            "actual_5": actual_5,
            "exp_with_neck": exp_with_neck,
            "exp_no_neck": exp_no_neck,
            "match_all_neck": match_all_neck,
            "match_all_noneck": match_all_noneck,
            "match_c4_neck": match_c4_neck,
            "match_c4_noneck": match_c4_noneck,
        })

out_p = r"D:\comparison_desc_vs_cond.txt"
with open(out_p, "w", encoding="utf-8") as out:
    for kr in sorted(key_results, key=lambda x: int(x["key"]) if x["key"].isdigit() else 999):
        out.write(f"Key {kr['key']}: '{kr['desc']}'\n")
        out.write(f"  Actual in JSON  : {kr['actual_5']} (c4={kr['actual_5'][3]})\n")
        out.write(f"  Exp (with neck) : {kr['exp_with_neck']} (c4={kr['exp_with_neck'][3]}) -> All Match: {kr['match_all_neck']}, c4 Match: {kr['match_c4_neck']}\n")
        out.write(f"  Exp (no neck)   : {kr['exp_no_neck']} (c4={kr['exp_no_neck'][3]}) -> All Match: {kr['match_all_noneck']}, c4 Match: {kr['match_c4_noneck']}\n\n")

print(f"Saved to {out_p}")
