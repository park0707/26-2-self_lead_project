# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: tests/test_render_deployment_readiness.py
Purpose: Render.com 리눅스 컨테이너 배포 무결성 및 하네스 엔지니어링 검증 테스트 스위트
         1. 앱 임포트 시점(import src.web.api.main) 모델 파일(.pkl/.onnx/.pth/.task) 0건 오픈 검증 (Lazy Loading & 512MB RAM 보호)
         2. 임포트된 모든 모듈 소스 코드 내 Windows 전용 PROJECT_ROOT/BUNDLE_PATH/sys.path('D:\\', 'd:/') 하드코딩 제거 검증
         3. BL5, SD2, SD3, DL4, DL5, GM1~GM3 등 수정된 개별 모델 클래스의 하위 호환성, 임포트 순서 독립성(Circular Import 방지) 및 클래스메서드 동작 검증
         4. 8대 운동 33종 결함 x 공식 5대 시점(시점 A~E / view1~view5) 전수 FeatureRegistry & Stage2InferenceEngine 추론 검증
         5. FastAPI Lifespan 기동 및 주요 REST 엔드포인트(/, /api/health, /api/exercises, /api/views, /api/analyze/sequence) 검증
"""

import os
import re
import sys
import subprocess
import unittest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestRenderDeploymentReadiness(unittest.TestCase):
    """Render.com 클라우드 배포 무결성 검증 테스트"""

    def test_01_zero_model_files_opened_at_import_time(self):
        """[Test 1] uvicorn이 src.web.api.main을 임포트할 때 .pkl/.onnx/.pth/.task 가중치 파일을 단 1개도 열지 않아야 함"""
        code = (
            "import builtins, sys; "
            "orig_open = builtins.open; "
            "opened = []; "
            "builtins.open = lambda f, *a, **kw: (opened.append(str(f)), orig_open(f, *a, **kw))[1]; "
            "import src.web.api.main; "
            "model_files = [f for f in opened if f.endswith(('.pkl', '.onnx', '.pth', '.task'))]; "
            "print('MODEL_FILES_COUNT=' + str(len(model_files))); "
            "assert len(model_files) == 0, f'Unexpected model files opened at import: {model_files}'"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        self.assertEqual(res.returncode, 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        self.assertIn("MODEL_FILES_COUNT=0", res.stdout)

    def test_02_no_hardcoded_d_drive_project_root_or_bundle_in_loaded_modules(self):
        """[Test 2] src.web.api.main 임포트 트리에 속한 모든 모듈에 PROJECT_ROOT='d:/', sys.path.insert(0,'d:/'), BUNDLE_PATH='D:\\...' 하드코딩이 없어야 함"""
        import src.web.api.main  # noqa: F401

        forbidden_patterns = [
            re.compile(r'PROJECT_ROOT\s*=\s*r?[\'\'][dD]:[/\\]'),
            re.compile(r'BUNDLE_PATH\s*=\s*r?[\'\'][dD]:[/\\]'),
            re.compile(r'sys\.path\.(?:insert|append)\([^)]*[\'\'][dD]:[/\\]'),
        ]
        violations = []

        for mod_name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and f.endswith(".py") and os.path.abspath(f).startswith(os.path.join(PROJECT_ROOT, "src")):
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    for idx, line in enumerate(fp, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        for pat in forbidden_patterns:
                            if pat.search(stripped):
                                violations.append(f"{f}:{idx} -> {stripped}")

        self.assertEqual(len(violations), 0, f"Hardcoded D: path violations found:\n" + "\n".join(violations))

    def test_03_bundle_paths_exist_and_are_dynamic(self):
        """[Test 3] 수정된 모든 모델 모듈의 기본 번들 경로(BUNDLE_PATH / DEFAULT_BUNDLE_PATH)가 실제 파일로 존재하고 순환 참조가 없어야 함"""
        from src.models.second_models.babel_lunge.bl5_final import BUNDLE_PATH as BL5_BUNDLE, BL5Classifier
        from src.models.second_models.babel_lunge.bl5_hybrid_1 import DEFAULT_BUNDLE_PATH as BL5_HYBRID_BUNDLE
        from src.models.second_models.babel_stiff_deadlift.sd2_model_v3_rtmp import DEFAULT_BUNDLE_PATH as SD2_BUNDLE
        from src.models.second_models.babel_stiff_deadlift.sd3_model_v4_rtmp import DEFAULT_BUNDLE_PATH as SD3_BUNDLE
        from src.models.second_models.dynamic_lunge.dl4_rtmp_1 import MODEL_BUNDLE_PATH as DL4_BUNDLE, DL4RTMP1Classifier
        from src.models.second_models.dynamic_lunge.dl5_final import BUNDLE_PATH as DL5_BUNDLE, DL5Classifier
        from src.models.second_models.babel_good_morning.gm1_rtmp_1 import (
            DEFAULT_WEIGHTS_PATH as GM1_BUNDLE, BabelGoodMorningGM1RTMP1
        )
        from src.models.second_models.babel_good_morning.gm2_rtmp_1 import (
            DEFAULT_WEIGHTS_PATH as GM2_BUNDLE, BabelGoodMorningGM2RTMP1
        )
        from src.models.second_models.babel_good_morning.gm3_rtmp_1 import (
            DEFAULT_WEIGHTS_PATH as GM3_BUNDLE, BabelGoodMorningGM3RTMP1
        )
        from src.web.core.anti_overfitting_predictor import DEFAULT_BUNDLE_PATH as MASTER_BUNDLE
        from src.web.pipeline.stage1_engine import DEFAULT_BUNDLE_PATH as STAGE1_BUNDLE

        self.assertEqual(os.path.dirname(BL5_HYBRID_BUNDLE), os.path.dirname(BL5_BUNDLE))

        for name, path in [
            ("BL5_BUNDLE", BL5_BUNDLE),
            ("SD2_BUNDLE", SD2_BUNDLE),
            ("SD3_BUNDLE", SD3_BUNDLE),
            ("DL4_BUNDLE", DL4_BUNDLE),
            ("DL5_BUNDLE", DL5_BUNDLE),
            ("GM1_BUNDLE", GM1_BUNDLE),
            ("GM2_BUNDLE", GM2_BUNDLE),
            ("GM3_BUNDLE", GM3_BUNDLE),
            ("MASTER_BUNDLE", MASTER_BUNDLE),
            ("STAGE1_BUNDLE", STAGE1_BUNDLE),
        ]:
            self.assertTrue(os.path.exists(path), f"{name} does not exist at {path}")

        # 클래스메서드 피처 추출 동작 및 인스턴스 생성 하위 호환성 확인
        mp33 = np.random.uniform(0.3, 0.7, (64, 33, 3)).astype(np.float32)
        rtm26 = np.random.uniform(0.3, 0.7, (64, 26, 3)).astype(np.float32)

        f_bl5 = BL5Classifier.extract_features(mp33, rtm26)
        self.assertEqual(f_bl5.shape, (79,))

        f_dl4 = DL4RTMP1Classifier.extract_features(rtm26)
        self.assertEqual(f_dl4.ndim, 1)
        self.assertGreater(len(f_dl4), 0)

        f_dl5 = DL5Classifier.extract_features(mp33, rtm26)
        self.assertEqual(f_dl5.ndim, 1)
        self.assertGreater(len(f_dl5), 0)

        f_gm1 = BabelGoodMorningGM1RTMP1.extract_features(rtm26)
        self.assertEqual(f_gm1.shape, (BabelGoodMorningGM1RTMP1.FEATURE_DIM,))

        f_gm2 = BabelGoodMorningGM2RTMP1.extract_features(rtm26)
        self.assertEqual(f_gm2.shape, (BabelGoodMorningGM2RTMP1.FEATURE_DIM,))

        f_gm3 = BabelGoodMorningGM3RTMP1.extract_features(rtm26)
        self.assertEqual(f_gm3.shape, (BabelGoodMorningGM3RTMP1.FEATURE_DIM,))

        # 직접 인스턴스화도 정상 동작하는지 검증
        bl5_inst = BL5Classifier()
        res_bl5 = bl5_inst.predict_sequence(mp33, rtm26, view="시점 A")
        self.assertIn("state_code", res_bl5)

    def test_04_all_33_defects_and_5_views_inference(self):
        """[Test 4] 8대 운동 33종 결함 x 공식 5대 시점(시점 A~E) 전수 추론 및 NaN/Inf 무결성 검증"""
        from src.web.pipeline.stage2_engine import Stage2InferenceEngine

        stage2 = Stage2InferenceEngine.get_instance(prewarm=False)
        mp33 = np.random.uniform(0.3, 0.7, (64, 33, 3)).astype(np.float32)
        rtm26 = np.random.uniform(0.3, 0.7, (64, 26, 3)).astype(np.float32)
        spine9 = np.random.uniform(0.3, 0.7, (64, 9, 3)).astype(np.float32)

        official_views = ["시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]
        total_defect_evals = 0

        for eid in range(8):
            for view_label in official_views:
                diag = stage2.diagnose(
                    exercise=eid,
                    keypoints_seq=mp33,
                    camera_view=view_label,
                    rtmp_seq=rtm26,
                    spine_seq=spine9
                )
                preds = diag["defects"]
                self.assertGreater(len(preds), 0, f"Exercise {eid} returned empty defect predictions")
                for p in preds:
                    self.assertIn(p["state"], (0, 1, 2))
                    self.assertIn(p["state_label"], ("정상", "결함", "주의/보류", "보류"))
                    self.assertTrue(0.0 <= p["probability"] <= 1.0)
                    self.assertFalse(np.isnan(p["probability"]))
                    total_defect_evals += 1

        # 33종 결함 x 5대 시점 = 165건 전수 통과 확인
        self.assertEqual(total_defect_evals, 33 * 5)

    def test_05_fastapi_lifespan_and_endpoints(self):
        """[Test 5] FastAPI 서버 Lifespan 기동 및 핵심 엔드포인트 응답 검증"""
        from fastapi.testclient import TestClient
        from src.web.api.main import app

        with TestClient(app) as client:
            # 1. Root
            r_root = client.get("/")
            self.assertEqual(r_root.status_code, 200)

            # 2. Health
            r_health = client.get("/api/health")
            self.assertEqual(r_health.status_code, 200)
            h_data = r_health.json()
            self.assertEqual(h_data["status"], "healthy")
            self.assertTrue(h_data["stage1_ready"])
            self.assertTrue(h_data["stage2_ready"])
            self.assertEqual(h_data["total_defects_count"], 33)
            self.assertEqual(len(h_data["supported_exercises"]), 8)
            self.assertEqual(len(h_data["camera_views"]), 5)

            # 3. Exercises & Views
            r_ex = client.get("/api/exercises")
            self.assertEqual(r_ex.status_code, 200)
            self.assertEqual(len(r_ex.json()["exercises"]), 8)

            r_views = client.get("/api/views")
            self.assertEqual(r_views.status_code, 200)
            self.assertEqual(len(r_views.json()["views"]), 5)

            # 4. Sequence Analysis
            dummy_seq = np.random.uniform(0.3, 0.7, (64, 33, 3)).round(4).tolist()
            r_seq = client.post(
                "/api/analyze/sequence",
                json={
                    "keypoints": dummy_seq,
                    "exercise_id": 3,
                    "camera_view": "시점 A"
                }
            )
            self.assertEqual(r_seq.status_code, 200)
            s_data = r_seq.json()
            self.assertTrue(s_data["success"])
            self.assertEqual(s_data["exercise"]["exercise_id"], 3)
            self.assertEqual(len(s_data["defects"]), 5)  # 바벨 런지 bl1~bl5


if __name__ == "__main__":
    unittest.main(verbosity=2)
