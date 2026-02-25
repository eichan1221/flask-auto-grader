import unittest

import app as grader_app


class HintQualityTests(unittest.TestCase):
    def test_format_hint_text_has_structured_points(self):
        raw = "まず原因を押さえる。重要語句として鎖国を入れる。結論→理由→結果の順で書く。"
        formatted = grader_app.format_hint_text(raw)
        self.assertIn("- ", formatted)
        self.assertTrue(any(k in formatted for k in ("着眼点", "キーワード", "順番", "注意", "書き出し", "ヒント")))

    def test_hint_similarity_detector_blocks_question_paraphrase(self):
        question = "鎌倉幕府が成立した理由を説明しなさい。"
        paraphrase = "鎌倉幕府が成立した理由を説明する問題です。"
        self.assertTrue(grader_app.hint_is_too_similar_to_question(question, paraphrase))

    def test_generation_prompt_requires_non_paraphrase_hints(self):
        payload = {
            "subject": "社会",
            "category": "歴史",
            "grade": "",
            "difficulty": "10点",
            "unit": "鎌倉時代",
            "genre_hint": "因果を問う",
            "avoid_topics": "",
            "length_hint": "30〜80字",
        }
        msgs = grader_app.build_generation_messages(payload)
        user_prompt = msgs[1]["content"]
        self.assertIn("問題文の言い換え禁止", user_prompt)
        self.assertIn("着眼点", user_prompt)


if __name__ == "__main__":
    unittest.main()
