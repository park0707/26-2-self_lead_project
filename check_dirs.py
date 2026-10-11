import os
import json

data_dir = 'D:/Data'
try:
    dirs = os.listdir(data_dir)
    res = {'Data': dirs}
    
    for d in dirs:
        if '전처리' in d or 'rtmp' in d.lower() or '' in d: # Just check all to be safe
            try:
                sub_dirs = os.listdir(os.path.join(data_dir, d))
                res[d] = sub_dirs
                for sd in sub_dirs:
                    if 'rtmp' in sd.lower() or '' in sd:
                        try:
                            ssd = os.listdir(os.path.join(data_dir, d, sd))
                            res[d + '/' + sd] = ssd
                        except:
                            pass
            except Exception as e:
                pass
except Exception as e:
    res = str(e)

with open('D:/check_dirs.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
