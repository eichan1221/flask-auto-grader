import unittest
from unittest.mock import patch

import app as grader_app


class GradeAnswerBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.client = grader_app.app.test_client()
        grader_app.app.config["TESTING"] = True

    def _payload(self, difficulty: str = "10点"):
        return {
            "question": "鎌倉幕府が成立した理由を説明しなさい。",
            "student_answer": "武士が力を持ち、源頼朝が政治の中心を鎌倉に置いたから。",
            "difficulty": difficulty,
        }

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_perfect_score_removes_improvement_fields(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": 10,
            "good_points": ["結論が明確", "理由が具体的", "重要語句が適切"],
            "next_step": "次回は具体例を増やす",
            "rewrite_tip": "語尾を整える",
            "next_steps": ["A"],
            "practice_menu": ["B"],
            "weak_tags": ["語句"],
            "short_comment": "とてもよいです",
            "best_sentence": "武士が力を持ったため。",
            "rubric": {"conclusion": 3, "logic": 3, "wording": 3},
        }, "")

        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["is_perfect_score"])
        self.assertEqual(body["score_total"], 10)
        self.assertEqual(body["rewrite_tip"], "")
        self.assertEqual(body["next_steps"], [])
        self.assertEqual(body["practice_menu"], [])
        self.assertEqual(body["weak_tags"], [])
        self.assertGreaterEqual(len(body["good_points"]), 3)

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_non_perfect_keeps_improvement_fields(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": 7,
            "good_points": ["結論がある", "理由が書けている"],
            "next_step": "具体例を入れる",
            "rewrite_tip": "具体例を1つ追加",
            "next_steps": ["次は語句を足す"],
            "practice_menu": ["50字で要約"],
            "weak_tags": ["具体例不足"],
            "short_comment": "あと一歩",
            "best_sentence": "理由は書けている。",
            "rubric": {"conclusion": 2, "logic": 2, "wording": 2},
        }, "")

        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertFalse(body["is_perfect_score"])
        self.assertEqual(body["rewrite_tip"], "具体例を1つ追加")
        self.assertNotEqual(body["next_steps"], [])

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_perfect_score_detected_for_all_max_points(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": 10,
            "good_points": ["良い", "良い", "良い"],
            "next_step": "",
            "short_comment": "",
            "best_sentence": "",
            "rubric": {"conclusion": 3, "logic": 3, "wording": 3},
        }, "")

        for difficulty, expected in (("5点", 5), ("10点", 10), ("満点", 100)):
            res = self.client.post("/api/grade_answer", json=self._payload(difficulty))
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(body["max_score"], expected)
            self.assertEqual(body["score_total"], expected)
            self.assertTrue(body["is_perfect_score"])

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_missing_score_total_returns_error(self, mock_parse, *_):
        mock_parse.return_value = ({"good_points": ["A"]}, "")

        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 502)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "grading_invalid_schema")


if __name__ == "__main__":
    unittest.main()
