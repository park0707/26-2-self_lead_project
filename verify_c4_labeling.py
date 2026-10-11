import json

# Key descriptions
keys_desc = {
    145: "1앞다리 90도, 2무릎발방향일치, 3한쪽다리바르게 펴기, 4척추 중립, 5상체 중립",
    146: "정자세에서 앞다리 조금접고",
    147: "정자세에서 발방향 바깥쪽",
    148: "정자세에서 안굽히는 다리 구부정",
    149: "정자세에서 몸통 옆으로 갸우뚱",
    150: "정자세에서 상체 뒤로 젖힘",
    151: "정자세에서 앞다리 조금접고 발방향 바깥쪽",
    152: "정자세에서 앞다리 조금접고 안굽히는 다리 구부정",
    153: "정자세에서 앞다리 조금접고 몸통 옆으로 갸우뚱",
    154: "정자세에서 앞다리 조금접고 상체 뒤로 젖힘",
    155: "정자세에서 발방향 바깥쪽 안굽히는 다리 구부정",
    156: "정자세에서 발방향 바깥쪽 몸통 옆으로 갸우뚱",
    157: "정자세에서 발방향 바깥쪽 몸통 상체 뒤로 젖힘",
    158: "정자세에서 안굽히는 다리 구부정 몸통 옆으로 갸우뚱",
    159: "정자세에서 안굽히는 다리 구부정 몸통 상체 뒤로 젖힘",
    160: "정자세에서 몸통 옆으로 갸우뚱 상체 뒤로 젖힘",
    161: "정자세에서 안굽히는 다리 구부정 몸통 옆으로 갸우뚱 몸통 상체 뒤로 젖힘",
    162: "정자세에서 발방향 바깥쪽 몸통 옆으로 갸우뚱 몸통 상체 뒤로 젖힘",
    163: "정자세에서 왼쪽보고 몸통 상체 뒤로 젖힘",
    164: "정자세에서 오른쪽보고 몸통 상체 뒤로 젖힘",
    165: "정자세에서 하늘보고 몸통 상체 뒤로 젖힘",
    166: "정자세에서 바닥보고 몸통 상체 뒤로 젖힘",
    167: "정자세에서 왼쪽보고 몸통 옆으로 갸우뚱",
    168: "정자세에서 오른쪽보고 몸통 옆으로 갸우뚱",
    169: "정자세에서 하늘보고 몸통 옆으로 갸우뚱",
    170: "정자세에서 바닥보고 몸통 옆으로 갸우뚱",
    171: "정자세에서 왼쪽보고 앞다리 조금접힘",
    172: "정자세에서 왼쪽보고 안굽히는 다리 구부정",
    173: "정자세에서 바닥보고 앞다리 조금접힘",
    174: "정자세에서 바닥보고 안굽히는 다리 구부정",
    175: "정자세에서 하늘보고 앞다리 조금접힘",
    176: "정자세에서 하늘보고 안굽히는 다리 구부정"
}

def determine_c4_label_from_description(desc: str, key: int) -> int:
    if key == 145 or '4척추 중립' in desc:
        return 0
    has_spine_defect = '몸통 옆으로 갸우뚱' in desc
    has_cervical_defect = any(k in desc for k in ['왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고'])
    return 1 if (has_spine_defect or has_cervical_defect) else 0

print("Key | Description | c4 Label (0=Normal, 1=Defect)")
print("-" * 75)
defect_keys = []
normal_keys = []
for k in range(145, 177):
    d = keys_desc.get(k, "")
    lbl = determine_c4_label_from_description(d, k)
    status = "DEFECT(1)" if lbl == 1 else "NORMAL(0)"
    if lbl == 1:
        defect_keys.append(k)
    else:
        normal_keys.append(k)
    print(f"Key {k:3d} | {status} | {d}")

print("\nSummary:")
print(f"Normal Keys ({len(normal_keys)}개): {normal_keys}")
print(f"Defect Keys ({len(defect_keys)}개): {defect_keys}")
