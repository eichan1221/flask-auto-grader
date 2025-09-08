# app.py
# 社会/理科タブ・難易度・ストック（手動/自動）・説明UIに対応したフル版
# + 安定化拡張: CORS, OpenAI timeout, /ping, ACCESS_CODE ゲート

import os
import json
import random
import time
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify, redirect, make_response
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS  # 追加

# ====== 初期化 ======
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")  # 例: https://xxxx.onrender.com
ACCESS_CODE = os.environ.get("ACCESS_CODE")             # 例: test-2025（未設定なら無効）

app = Flask(__name__)
# /api/* へのCORSのみ許可（ALLOWED_ORIGIN 未設定時は広めに許可）
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

# OpenAI クライアント（timeout 追加）
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30) if OPENAI_API_KEY else None

# ====== 生成パラメータ ======
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
MAX_TOKENS = 850

GRADE_TEMPERATURE = 0.2
GRADE_TOP_P = 0.9
GRADE_MAX_TOKENS = 700

# 形式の連続回避用（セッション内メモ）
RECENT_MAX = 100
recent_formats: List[str] = []

# ストック（自動用）
STOCK_SOCIAL = [
    "問題: 弥生時代に稲作が広がったことが社会に与えた影響を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・因果関係\n・社会構造",
    "問題: 地方自治において住民投票の意義を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・制度の目的\n・民主主義",
    "問題: 環太平洋地域の工業分布の特徴を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・地理的要因\n・産業集積",
]
STOCK_SCIENCE = [
    "問題: 光合成で二酸化炭素が必要な理由を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・反応式の理解\n・物質の出入り",
    "問題: 塩化ナトリウム水溶液の電気伝導性が高い理由を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・イオン\n・電解質",
    "問題: 前線の通過が天気に与える変化を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・気団と前線\n・気象の因果",
]

# ====== ユーティリティ ======
def summarize_recent(n: int = 3) -> str:
    if not recent_formats:
        return "（履歴なし）"
    return " / ".join(recent_formats[-n:])

def _push_recent(tag: str):
    recent_formats.append(tag)
    if len(recent_formats) > RECENT_MAX:
        del recent_formats[: len(recent_formats) - RECENT_MAX]

def detect_format_tag(text: str, hint: str) -> str:
    head = (text or "")[:300]
    body = text or ""
    tag = "不明形式"
    if ("資料" in head) or ("(資料" in body) or ("（資料" in body):
        tag = "資料読解"
    elif ("用語" in head) or ("説明" in head):
        tag = "用語説明"
    elif ("理由" in head) or ("なぜ" in body):
        tag = "背景理由"
    elif ("影響" in body) or ("結果" in body) or ("因果" in body):
        tag = "因果影響"
    _push_recent(f"{tag}-{hint}")
    return tag

def force_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"raw": ""}
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:].strip()
    if "{" in cleaned and "}" in cleaned:
        try:
            s = cleaned.find("{")
            e = cleaned.rfind("}")
            return json.loads(cleaned[s : e + 1])
        except Exception:
            pass
    return {"raw": text}

# ====== プロンプト（社会・理科） ======
SYSTEM_PROMPT = (
    "あなたは宮城県公立高校入試と中学定期テストの出題・採点に詳しい編集者/採点者です。"
    "出題は日本語で、解答は30〜80字程度。形式の連続を避け、多様な観点（資料読解/用語説明/背景理由/因果影響）をローテしてください。"
)

def user_prompt_social(branch: str, unit: str, difficulty: str) -> str:
    points = "5点" if difficulty == "5" else "10点"
    mix = "1:1:1:1"
    return f"""
【出題条件（社会）】
- 分野: {branch}
- 単元・時代: {unit}
- 難易度: {points}
- 解答分量: 30〜80字
- 出題形式バランス: {mix}
- 直近履歴: {summarize_recent()}

【出力フォーマット（厳守）】
問題:
（必要なら資料は文章で要約し、画像は使わない）
想定解答:
出題意図:
模範解答の観点:
・
・

【追加要件】
- 資料問題に偏らない。資料が必要な場合も文章要約で提示。
- {unit} では特定テーマに偏らず、政治・社会・文化・対外関係なども回す。
- 直前の形式や観点の重複を避ける。
""".strip()

def user_prompt_science(grade: str, domain: str, topic: str, difficulty: str) -> str:
    points = "5点" if difficulty == "5" else "10点"
    topic_hint = f"（参考トピック: {topic}）" if topic else ""
    return f"""
【出題条件（理科）】
- 学年: {grade} / 分野: {domain} {topic_hint}
- 難易度: {points}
- 解答分量: 30〜80字
- 出題形式バランス: 1:1:1:1（用語説明/現象の理由/観察・実験の意図/因果・法則の適用）
- 直近履歴: {summarize_recent()}

【出力フォーマット（厳守）】
問題:
（必要なら観察・実験の設定を文章で要約し、図は使わない）
想定解答:
出題意図:
模範解答の観点:
・
・

【追加要件】
- 暗記だけでなく「なぜ」「どうなる」を問う。計算偏重は避け、文章で説明させる。
- {domain} の基礎概念と因果関係をバランスよく問う。
- 直前の形式や観点の重複を避ける。
""".strip()

SYSTEM_PROMPT_GRADER = (
    "あなたは中学の記述答案を公平に採点する採点者です。10点満点で、観点別コメントと模範解答、改善ポイントを述べます。"
)

def user_prompt_grade(question: str, answer: str) -> str:
    return f"""
[問題]
{question}

[生徒の回答]
{answer}

[採点条件]
- 10点満点（整数）
- 字数目安: 30〜80字
- 観点例: 用語の正確性 / 根拠・背景 / 因果関係 / 問題要求への適合
- 必要なら模範解答を簡潔に併記

[出力(JSON)]
{{
  "score": <0～10の整数>,
  "summary": "<総評>",
  "reasons": ["<理由を箇条書き>"],
  "model_answer": "<模範解答（30〜80字）>",
  "points": ["<改善ポイントを箇条書き>"]
}}
""".strip()

# ====== OpenAI クライアント確保（保険） ======
def ensure_client() -> OpenAI:
    global client
    if client is not None:
        return client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)
    return client

# ====== 生成関数 ======
def generate_via_model_for_social(branch: str, unit: str, difficulty: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_social(branch, unit, difficulty)},
    ]
    resp = ensure_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=MAX_TOKENS,
        messages=msgs,
    )
    text = (resp.choices[0].message.content or "").strip()
    detect_format_tag(text, unit)
    return text

def generate_via_model_for_science(grade: str, domain: str, topic: str, difficulty: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_science(grade, domain, topic, difficulty)},
    ]
    resp = ensure_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=MAX_TOKENS,
        messages=msgs,
    )
    text = (resp.choices[0].message.content or "").strip()
    hint = f"{grade}-{domain}" if topic == "" else f"{grade}-{domain}-{topic}"
    detect_format_tag(text, hint)
    return text

# ====== アクセスゲート（ACCESS_CODE が設定されている時だけ有効） ======
@app.before_request
def access_gate():
    if not ACCESS_CODE:
        return  # 無効時は何もしない
    path = request.path or "/"
    # 常時許可
    if path.startswith("/static") or path in ("/ping", "/healthz", "/favicon.ico"):
        return
    # 既に通過済み
    if request.cookies.get("access_code") == ACCESS_CODE:
        return
    # ?code=XXXX で通過 → Cookie 設定
    code = request.args.get("code")
    if code == ACCESS_CODE:
        resp = make_response(redirect(path))
        resp.set_cookie("access_code", ACCESS_CODE, max_age=60*60*24*7, secure=True, httponly=True, samesite="Lax")
        return resp
    return ("Access code required", 403)

# ====== ルーティング ======
@app.route("/")
def index():
    sel = {"mode": "auto"}
    return render_template("index.html", sel=sel)

@app.get("/ping")
def ping():
    return "pong", 200

@app.route("/api/generate_question", methods=["POST"])
def api_generate_question():
    data = request.get_json(silent=True) or request.form
    category = (data.get("category") or "社会").strip()

    mode = (data.get("mode") or "auto").strip()
    if mode == "manual":
        manual_text = (data.get("manual_text") or "").strip()
        if not manual_text:
            return jsonify({"ok": False, "error": "手動プリセットが空です。"}), 400
        return jsonify({"ok": True, "text": manual_text})

    if mode == "stock-auto":
        text = random.choice(STOCK_SOCIAL if category == "社会" else STOCK_SCIENCE)
        return jsonify({"ok": True, "text": text})

    try:
        if category == "社会":
            branch = (data.get("branch") or "歴史").strip()
            unit = (data.get("unit_or_era") or "弥生時代").strip()
            difficulty = (data.get("difficulty") or "5").strip()
            text = generate_via_model_for_social(branch, unit, difficulty)
        else:
            grade = (data.get("grade") or "中1").strip()
            domain = (data.get("domain") or "生物").strip()
            topic = (data.get("topic") or "").strip()
            difficulty = (data.get("difficulty") or "5").strip()
            text = generate_via_model_for_science(grade, domain, topic, difficulty)

        return jsonify({"ok": True, "text": text})
    except Exception as e:
        print("ERROR /api/generate_question:", type(e).__name__, repr(e))
        return jsonify({"ok": False, "error": "問題生成でエラーが発生しました。"}), 500

@app.route("/api/grade", methods=["POST"])
def api_grade():
    data = request.get_json(silent=True) or request.form
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not question or not answer:
        return jsonify({"ok": False, "error": "問題文と生徒の解答を入力してください。"}), 400

    try:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_GRADER},
            {"role": "user", "content": user_prompt_grade(question, answer)},
        ]
        resp = ensure_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=GRADE_TEMPERATURE,
            top_p=GRADE_TOP_P,
            max_tokens=GRADE_MAX_TOKENS,
            messages=msgs,
        )
        result = force_json(resp.choices[0].message.content)
        try:
            result["score"] = int(result.get("score", 0))
        except Exception:
            result["score"] = 0
        return jsonify({"ok": True, **result})
    except Exception as e:
        print("ERROR /api/grade:", type(e).__name__, repr(e))
        return jsonify({"ok": False, "error": "採点でエラーが発生しました。"}), 500

@app.route("/api/reset_session", methods=["POST"])
def api_reset_session():
    try:
        recent_formats.clear()
        return jsonify({"ok": True})
    except Exception as e:
        print("ERROR /api/reset_session:", type(e).__name__, repr(e))
        return jsonify({"ok": False}), 500

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=False)
