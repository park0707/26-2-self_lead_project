"""Stage 2: Binary Quality Assessment (Correct vs Incorrect) Models Package"""
import sys
import importlib
import importlib.util
from importlib.abc import MetaPathFinder

# 바벨 스쿼트 관련 모듈이 babel_squat 서브디렉터리로 이동됨에 따른 하위 호환성 메타패스 파인더
_SQUAT_MOVED = {
    'bottom_biomechanics',
    'pilot_squat_dataset',
    'extract_squat_spine_trend',
    'extract_squat_spine_literature',
    'extract_squat_spine_v2',
    'extract_squat_spine_final_25d',
    'eval_spine_v2',
    'eval_spine_v3',
    'eval_spine_v4',
    'generate_spine_refinement_report',
    'test_babel_squat',
}

class _SquatCompatFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        prefix = "src.models.second_models."
        if fullname.startswith(prefix):
            subname = fullname[len(prefix):]
            if subname in _SQUAT_MOVED:
                target_name = f"src.models.second_models.babel_squat.{subname}"
                return importlib.util.find_spec(target_name)
        return None

if not any(isinstance(f, _SquatCompatFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SquatCompatFinder())
