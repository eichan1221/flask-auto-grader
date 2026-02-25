import unittest
from unittest.mock import patch, Mock

import app as grader_app


class GradeAnswerBehaviorTests(unittest.TestCase):
    def setUp(self):
        grader_app.app.config["TESTING"] = True
        grader_app._rate_bucket.clear()
        grader_app.FREE_DAILY_MAIN_LIMIT = 9999
        grader_app.FREE_DAILY_AI_LIMIT = 9999
        self.client = grader_app.app.test_client()

    def _payload(self, difficulty: str = "10点", max_score=None, rewrite_count: int = 0):
        payload = {
            "question": "鎌倉幕府が成立した理由を説明しなさい。",
            "student_answer": "武士が力を持ち、源頼朝が政治の中心を鎌倉に置いたから。",
            "difficulty": difficulty,
            "rewrite_count": rewrite_count,
        }
        if max_score is not None:
            payload["max_score"] = max_score
        return payload

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_ten_point_first_grade_success(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": 7,
            "good_points": ["結論がある", "理由が書けている"],
            "next_step": "具体例を入れる",
            "rewrite_tip": "具体例を1つ追加",
            "short_comment": "あと一歩",
            "best_sentence": "理由は書けている。",
            "rubric": {"conclusion": 2, "logic": 2, "wording": 2},
        }, "")

        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["max_score"], 10)
        self.assertEqual(body["score_total"], 7)
        self.assertRegex(body["request_id"], r"^[0-9a-f-]{36}$")

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_rewrite_rescore_success_and_perfect_score_is_safe(self, mock_parse, *_):
        mock_parse.side_effect = [
            ({
                "score_total": 8,
                "good_points": ["結論が明確", "理由が具体的"],
                "next_step": "語句をもう1つ加える",
                "rewrite_tip": "重要語句を増やす",
                "short_comment": "もう少しで満点",
                "best_sentence": "武士が政治の中心になった。",
                "rubric": {"conclusion": 2, "logic": 3, "wording": 2},
            }, ""),
            ({
                "score_total": "10/10",
                "good_points": ["結論が明確", "理由が具体的", "語句が正確"],
                "next_step": "",
                "rewrite_tip": "",
                "short_comment": "満点です",
                "best_sentence": "源頼朝が鎌倉に幕府を開いた。",
                "rubric": {"conclusion": 3, "logic": 3, "wording": 3},
            }, ""),
        ]

        first = self.client.post("/api/grade_answer", json=self._payload("10点", rewrite_count=0))
        self.assertEqual(first.status_code, 200)
        first_body = first.get_json()
        self.assertTrue(first_body["ok"])
        self.assertEqual(first_body["score_total"], 8)

        second = self.client.post("/api/grade_answer", json=self._payload("10点", rewrite_count=1))
        self.assertEqual(second.status_code, 200)
        body = second.get_json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["is_perfect_score"])
        self.assertEqual(body["score_total"], 10)
        self.assertEqual(body["max_score"], 10)
        self.assertEqual(body["improvements"], [])
        self.assertEqual(body["rewrite_tip"], "")

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
    def test_perfect_score_detected_for_supported_max_points(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": 10,
            "good_points": ["良い", "良い", "良い"],
            "next_step": "",
            "short_comment": "",
            "best_sentence": "",
            "rubric": {"conclusion": 3, "logic": 3, "wording": 3},
        }, "")

        for difficulty, expected in (("10点", 10), ("満点", 100)):
            res = self.client.post("/api/grade_answer", json=self._payload(difficulty))
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(body["max_score"], expected)
            self.assertEqual(body["score_total"], expected)
            self.assertTrue(body["is_perfect_score"])

    @patch("app.ensure_openai", return_value=None)
    def test_rejects_five_point_max_score(self, *_):
        res = self.client.post("/api/grade_answer", json=self._payload("10点", max_score=5))
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "unsupported_max_score")
        self.assertIn("5点配点は現在利用できません", body["message"])

    @patch("app.ensure_openai", return_value="openai unavailable")
    def test_grading_failure_returns_ok_false(self, *_):
        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 503)
        body = res.get_json()
        self.assertFalse(body["ok"])

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
        self.assertIn("request_id", body)


    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_rewrite_perfect_rescore_accepts_grade_alias_and_string_lists(self, mock_parse, *_):
        mock_parse.side_effect = [
            ({
                "score_total": 6,
                "good_points": ["要点あり", "方向性OK"],
                "rubric": {"conclusion": 2, "logic": 2, "wording": 1},
            }, ""),
            ({
                "grade": "10/10",
                "good_points": "結論・理由・語句",
                "advice": None,
                "rubric": {"conclusion": 3, "logic": 3, "wording": 3},
                "short_comment": "完成度が高い",
            }, ""),
        ]

        first = self.client.post("/api/grade_answer", json=self._payload("10点", rewrite_count=0))
        self.assertEqual(first.status_code, 200)

        second = self.client.post("/api/grade_answer", json=self._payload("10点", rewrite_count=1))
        self.assertEqual(second.status_code, 200)
        body = second.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["score_total"], 10)
        self.assertTrue(body["is_perfect_score"])
        self.assertEqual(body["improvements"], [])

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry")
    def test_schema_invalid_score_returns_ok_false(self, mock_parse, *_):
        mock_parse.return_value = ({
            "score_total": "N/A",
            "good_points": "結論がある",
            "rubric": {"conclusion": 1, "logic": 1, "wording": 1},
        }, "")

        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 502)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "grading_invalid_schema")

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.parse_grading_response_with_retry", return_value=(None, "not json"))
    def test_parse_failure_returns_ok_false(self, *_):
        res = self.client.post("/api/grade_answer", json=self._payload("10点"))
        self.assertEqual(res.status_code, 502)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "grading_parse_failed")
        self.assertIn("message", body)
        self.assertRegex(body["request_id"], r"^[0-9a-f-]{36}$")

    @patch("app.ensure_openai", return_value=None)
    @patch("app.resolve_model_answer", return_value="")
    @patch("app.append_jsonl")
    @patch("app.parse_grading_response_with_retry", return_value=(None, "raw llm text"))
    def test_parse_failure_emits_structured_error_log(self, *_mocks):
        mock_log = _mocks[1]

        res = self.client.post("/api/grade_answer", json=self._payload("10点", rewrite_count=1))
        self.assertEqual(res.status_code, 502)
        body = res.get_json()
        self.assertFalse(body["ok"])

        records = [call.args[1] for call in mock_log.call_args_list if call.args and call.args[0].name == "grading.jsonl"]
        error_logs = [r for r in records if r.get("event") == "grade_answer_error"]
        self.assertTrue(error_logs)
        event = error_logs[-1]
        self.assertEqual(event.get("error_type"), "grading_parse_failed")
        self.assertEqual(event.get("http_status"), 502)
        self.assertEqual(event.get("max_score"), 10)
        self.assertTrue(event.get("is_rewrite"))
        self.assertIn("user_id_masked", event)
        self.assertEqual(event.get("request_id"), body.get("request_id"))



class GradingParseRobustnessTests(unittest.TestCase):
    def test_parse_json_object_loose_accepts_plain_json(self):
        parsed, source = grader_app.parse_json_object_loose('{"score_total": 8, "good_points": ["A"]}')
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("score_total"), 8)
        self.assertIn(source, {"raw", "embedded_json"})

    def test_parse_json_object_loose_accepts_code_fence(self):
        text = """```json\n{\n  \"score_total\": \"10/10\",\n  \"good_points\": \"結論・理由\"\n}\n```"""
        parsed, _ = grader_app.parse_json_object_loose(text)
        self.assertEqual(parsed.get("score_total"), "10/10")

    def test_parse_json_object_loose_accepts_prefixed_text(self):
        text = "採点結果です。\n{\"score_total\":7,\"good_points\":[\"要点\"]}\n以上です。"
        parsed, source = grader_app.parse_json_object_loose(text)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("score_total"), 7)
        self.assertEqual(source, "embedded_json")

    @patch("app.append_jsonl")
    def test_parse_grading_response_with_retry_retries_once_then_succeeds(self, mock_log):
        first = Mock()
        first.choices = [Mock(message=Mock(content="採点します\n```json\n{\"score_total\": 9"))]
        second = Mock()
        second.choices = [Mock(message=Mock(content='{"score_total": 9, "good_points": ["A"]}'))]

        mock_create = Mock(side_effect=[first, second])
        original_client = grader_app.client
        grader_app.client = Mock(chat=Mock(completions=Mock(create=mock_create)))
        try:
            parsed, raw = grader_app.parse_grading_response_with_retry([{"role": "user", "content": "x"}])
        finally:
            grader_app.client = original_client

        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("score_total"), 9)
        self.assertIn("score_total", raw)
        self.assertEqual(mock_create.call_count, 2)
        events = [call.args[1].get("event") for call in mock_log.call_args_list]
        self.assertIn("grading_parse_failed", events)
        self.assertIn("grading_parse_retry_happened", events)


if __name__ == "__main__":
    unittest.main()
