import json
import tempfile
import unittest
from pathlib import Path

import app as grader_app


class TeacherDashboardTests(unittest.TestCase):
    def setUp(self):
        grader_app.app.config["TESTING"] = True
        self.client = grader_app.app.test_client()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_log_dir = grader_app.LOG_DIR
        grader_app.LOG_DIR = Path(self.tmpdir.name)
        grader_app.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        grader_app.LOG_DIR = self.orig_log_dir
        self.tmpdir.cleanup()

    def _append(self, record):
        with (grader_app.LOG_DIR / "grading.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_dashboard_uses_time_series_review_logic(self):
        rows = [
            {
                "event": "graded", "user_key": "u1", "student_label": "A", "question_key": "q1",
                "score": 4, "max_points": 10, "answer_length": 30,
                "weakness_categories": ["理由"], "ai_review_flag": False, "ts": 1,
            },
            {
                "event": "graded", "user_key": "u1", "student_label": "A", "question_key": "q1",
                "score": 5, "max_points": 10, "answer_length": 28,
                "weakness_categories": ["理由"], "ai_review_flag": False, "ts": 2,
            },
            {
                "event": "graded", "user_key": "u1", "student_label": "A", "question_key": "q1",
                "score": 5, "max_points": 10, "answer_length": 15,
                "weakness_categories": ["理由"], "ai_review_flag": True, "ts": 3,
                "question_preview": "説明しなさい",
            },
        ]
        for r in rows:
            self._append(r)

        dash = grader_app.analyze_teacher_dashboard()
        self.assertEqual(dash["needs_review_count"], 1)
        reasons = dash["needs_review_items"][0]["reasons"]
        self.assertTrue(any("2回以上再提出" in x for x in reasons))
        self.assertTrue(any("同じ弱点が連続" in x for x in reasons))
        self.assertIn("解答が短すぎる", reasons)
        self.assertIn("AI要確認フラグ", reasons)

    def test_teacher_student_endpoint_returns_compare_items(self):
        self._append({
            "event": "graded", "user_key": "u2", "student_label": "B", "question_key": "q2",
            "score": 3, "max_points": 10, "student_answer": "最初の解答", "question_preview": "Q", "ts": 10,
        })
        self._append({
            "event": "graded", "user_key": "u2", "student_label": "B", "question_key": "q2",
            "score": 8, "max_points": 10, "student_answer": "改善した解答", "question_preview": "Q", "ts": 11,
        })

        res = self.client.get("/api/teacher/student/u2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["student"]["student_label"], "B")
        self.assertEqual(body["items"][0]["first_score"], 3)
        self.assertEqual(body["items"][0]["latest_score"], 8)


if __name__ == "__main__":
    unittest.main()
