# app.py
# 社会/理科タブ・難易度・ストック（手動/自動）・説明UIに対応したフル版
# + 安定化拡張: CORS, OpenAI timeout, /ping, ACCESS_CODE ゲート
# + 追加実装: JSON永続化(stock/history), 上限10件, 重複出題防止, 再挑戦API, 難易度「満点」

import os
import json
import random
import time
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

from flask import Flask, render_template, request, jsonify, redirect, make_response
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS  # 追加

# ====== 初期化 ======
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")  # 例: https://xxxx.onrender.com
ACCESS_CODE = os.environ.get("ACCESS_CODE")             # 例: test-2025（未設定なら無効）
DATA_DIR = os.environ.get("DATA_DIR", "./data")         # 例: ./data

os.makedirs(DATA_DIR, exist_ok=True)
STOCK_PATH = os.path.join(DATA_DIR, "stock.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")

_app_lock = threading.Lock()

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

# 類似度チェック
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_LOOKBACK = 50   # 直近何件を比較するか

# 形式の連続回避用（セッション内メモ）
RECENT_MAX = 100
recent_formats: List[str] = []

# ストック（初期プリセット：自動用のデフォルト）
STOCK_SOCIAL_DEFAULT = [
    "問題: 弥生時代に稲作が広がったことが社会に与えた影響を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・因果関係\n・社会構造",
    "問題: 地方自治において住民投票の意義を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・制度の目的\n・民主主義",
    "問題: 環太平洋地域の工業分布の特徴を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・地理的要因\n・産業集積",
]
STOCK_SCIENCE_DEFAULT = [
    "問題: 光合成で二酸化炭素が必要な理由を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・反応式の理解\n・物質の出入り",
    "問題: 塩化ナトリウム水溶液の電気伝導性が高い理由を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・イオン\n・電解質",
    "問題: 前線の通過が天気に与える変化を説明せよ（30〜60字）。\n想定解答: 〜\n出題意図: 〜\n模範解答の観点:\n・気団と前線\n・気象の因果",
]

# ====== ユーティリティ ======
def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

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

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_json(path: str, default):
    with _app_lock:
        try:
            if not os.path.exists(path):
                return default
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

def save_json(path: str, data) -> None:
    with _app_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

def parse_question_blocks(text: str) -> Dict[str, str]:
    """
    生成テキストから「問題/想定解答/出題意図/模範解答の観点」を素朴に抽出
    """
    res = {"問題": "", "想定解答": "", "出題意図": "", "模範解答の観点": ""}
    if not text:
        return res
    cur = None
    lines = text.splitlines()
    buf = []
    def flush():
        nonlocal buf, cur
        if cur:
            res[cur] = "\n".join(buf).strip()
        buf = []
    for ln in lines:
        if ln.strip().startswith("問題:"):
            flush(); cur = "問題"; buf.append(ln.split("問題:",1)[1].strip())
        elif ln.strip().startswith("想定解答:"):
            flush(); cur = "想定解答"; buf.append(ln.split("想定解答:",1)[1].strip())
        elif ln.strip().startswith("出題意図:"):
            flush(); cur = "出題意図"; buf.append(ln.split("出題意図:",1)[1].strip())
        elif ln.strip().startswith("模範解答の観点:"):
            flush(); cur = "模範解答の観点"; buf.append(ln.split("模範解答の観点:",1)[1].strip())
        else:
            if cur:
                buf.append(ln)
    flush()
    return res

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()

def collect_recent_texts_for_similarity() -> List[str]:
    texts = []
    stock = load_json(STOCK_PATH, {"social": [], "science": []})
    history = load_json(HISTORY_PATH, [])
    for item in stock.get("social", []):
        texts.append(item.get("text",""))
    for item in stock.get("science", []):
        texts.append(item.get("text",""))
    for h in history[-SIMILARITY_LOOKBACK:]:
        if "question_text" in h:
            texts.append(h["question_text"])
    return [t for t in texts if t]

def is_too_similar(text: str, existing: List[str]) -> bool:
    for old in existing[-SIMILARITY_LOOKBACK:]:
        if similarity(text, old) >= SIMILARITY_THRESHOLD:
            return True
    return False

def add_to_stock(category: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    category: "社会" or "理科"
    payload: {id, date, meta..., text, blocks{...}}
    """
    store = load_json(STOCK_PATH, {"social": [], "science": []})
    key = "social" if category == "社会" else "science"
    items = store.get(key, [])
    # 重複（同じtext）を先に除外
    items = [it for it in items if it.get("text") != payload.get("text")]
    # 先頭に追加
    items.insert(0, payload)
    # 上限10件
    items = items[:10]
    store[key] = items
    save_json(STOCK_PATH, store)
    return payload

def list_stock(category: Optional[str] = None) -> Dict[str, Any]:
    store = load_json(STOCK_PATH, None)
    if store is None:
        # 初期化：デフォルトプリセットを保存
        store = {
            "social": [{"id": f"init-s-{i}", "date": now_iso(), "meta": {"category": "社会"}, "text": t, "blocks": parse_question_blocks(t)} for i, t in enumerate(STOCK_SOCIAL_DEFAULT)],
            "science": [{"id": f"init-c-{i}", "date": now_iso(), "meta": {"category": "理科"}, "text": t, "blocks": parse_question_blocks(t)} for i, t in enumerate(STOCK_SCIENCE_DEFAULT)],
        }
        save_json(STOCK_PATH, store)
    if category == "社会":
        return {"social": store.get("social", [])}
    if category == "理科":
        return {"science": store.get("science", [])}
    return store

def delete_from_stock(item_id: str) -> bool:
    store = load_json(STOCK_PATH, {"social": [], "science": []})
    changed = False
    for key in ("social", "science"):
        before = len(store.get(key, []))
        store[key] = [it for it in store.get(key, []) if it.get("id") != item_id]
        if len(store[key]) != before:
            changed = True
    if changed:
        save_json(STOCK_PATH, store)
    return changed

def append_history(entry: Dict[str, Any]) -> None:
    hist = load_json(HISTORY_PATH, [])
    hist.append(entry)
    # ヒストリは肥大化する可能性があるが、ここでは無制限保持（必要なら上限設定）
    save_json(HISTORY_PATH, hist)

def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    hist = load_json(HISTORY_PATH, [])
    return hist[-limit:]

# ====== プロンプト（社会・理科） ======
SYSTEM_PROMPT = (
    "あなたは宮城県公立高校入試と中学定期テストの出題・採点に詳しい編集者/採点者です。"
    "出題は日本語で、解答は30〜80字程度。形式の連続を避け、多様な観点（資料読解/用語説明/背景理由/因果影響）をローテしてください。"
)

def _difficulty_label(difficulty: str) -> str:
    if difficulty in ("満点", "full", "100"):
        return "満点レベル"
    return "5点" if difficulty == "5" else "10点"

def user_prompt_social(branch: str, unit: str, difficulty: str) -> str:
    points = _difficulty_label(difficulty)
    mix = "1:1:1:1"
    rigor = "" if points != "満点レベル" else "\n- 要求水準を上げ、複数観点の統合・根拠の明確化を求める。"
    return f"""
【出題条件（社会）】
- 分野: {branch}
- 単元・時代: {unit}
- 難易度: {points}
- 解答分量: 30〜80字
- 出題形式バランス: {mix}
- 直近履歴: {summarize_recent()}{rigor}

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
    points = _difficulty_label(difficulty)
    topic_hint = f"（参考トピック: {topic}）" if topic else ""
    rigor = "" if points != "満点レベル" else "\n- 要求水準を上げ、観察条件/法則/因果を統合して説明させる。"
    return f"""
【出題条件（理科）】
- 学年: {grade} / 分野: {domain} {topic_hint}
- 難易度: {points}
- 解答分量: 30〜80字
- 出題形式バランス: 1:1:1:1（用語説明/現象の理由/観察・実験の意図/因果・法則の適用）
- 直近履歴: {summarize_recent()}{rigor}

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
def _generate_for_social(branch: str, unit: str, difficulty: str) -> str:
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

def _generate_for_science(grade: str, domain: str, topic: str, difficulty: str) -> str:
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

def generate_with_similarity_guard(category: str, gen_params: Dict[str, str], max_attempts: int = 3) -> str:
    """
    類似度チェックを通しながら最大 max_attempts 回まで生成を試みる
    """
    existing = collect_recent_texts_for_similarity()
    for attempt in range(max_attempts):
        if category == "社会":
            text = _generate_for_social(gen_params["branch"], gen_params["unit"], gen_params["difficulty"])
        else:
            text = _generate_for_science(gen_params["grade"], gen_params["domain"], gen_params.get("topic",""), gen_params["difficulty"])
        # 類似度判定
        if not is_too_similar(text, existing):
            return text
    # どうしても似る場合は最後の生成結果を返す
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

# ====== ルーティング（既存） ======
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
        # 手動は永続化しない（必要なら別APIで保存）
        return jsonify({"ok": True, "text": manual_text})

    if mode == "stock-auto":
        stock = list_stock()
        key = "social" if category == "社会" else "science"
        items = stock.get(key, [])
        if not items:
            # デフォルトから供給
            seed = STOCK_SOCIAL_DEFAULT if category == "社会" else STOCK_SCIENCE_DEFAULT
            choice = random.choice(seed)
            return jsonify({"ok": True, "text": choice})
        choice = random.choice(items)
        return jsonify({"ok": True, "text": choice.get("text","")})

    try:
        if category == "社会":
            branch = (data.get("branch") or "歴史").strip()
            unit = (data.get("unit_or_era") or "弥生時代").strip()
            difficulty = (data.get("difficulty") or "5").strip()
            gen_params = {"branch": branch, "unit": unit, "difficulty": difficulty}
        else:
            grade = (data.get("grade") or "中1").strip()
            domain = (data.get("domain") or "生物").strip()
            topic = (data.get("topic") or "").strip()
            difficulty = (data.get("difficulty") or "5").strip()
            gen_params = {"grade": grade, "domain": domain, "topic": topic, "difficulty": difficulty}

        text = generate_with_similarity_guard(category, gen_params, max_attempts=3)

        # 生成成功 → ストックへ永続化（上限10件）
        meta = {"category": category}
        meta.update(gen_params)
        blocks = parse_question_blocks(text)
        qid = f"q-{int(time.time()*1000)}-{random.randint(1000,9999)}"
        payload = {
            "id": qid,
            "date": now_iso(),
            "meta": meta,
            "text": text,
            "blocks": blocks,
            "prompt_hash": sha1(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        }
        add_to_stock(category, payload)

        return jsonify({"ok": True, "text": text, "id": qid, "meta": meta})
    except Exception as e:
        print("ERROR /api/generate_question:", type(e).__name__, repr(e))
        return jsonify({"ok": False, "error": "問題生成でエラーが発生しました。"}), 500

@app.route("/api/grade", methods=["POST"])
def api_grade():
    data = request.get_json(silent=True) or request.form
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    qid = (data.get("question_id") or "").strip()
    category = (data.get("category") or "").strip() or None
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

        # 採点履歴を保存
        entry = {
            "id": f"h-{int(time.time()*1000)}-{random.randint(1000,9999)}",
            "date": now_iso(),
            "question_id": qid or None,
            "question_text": question,
            "answer": answer,
            "result": result,
            "category": category,
        }
        append_history(entry)

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

# ====== 追加API（互換を保ったまま機能拡張） ======
@app.get("/api/stock/list")
def api_stock_list():
    category = request.args.get("category")  # "社会" or "理科" or None
    return jsonify({"ok": True, "data": list_stock(category)})

@app.post("/api/stock/add")
def api_stock_add():
    """
    手動でストックに追加したいとき用（UIからの『保存』ボタンなど）
    body: {category, text, meta?}
    """
    data = request.get_json(silent=True) or request.form
    category = (data.get("category") or "").strip() or "社会"
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "textが空です。"}), 400
    meta = data.get("meta") or {"category": category}
    blocks = parse_question_blocks(text)
    qid = f"q-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    payload = {
        "id": qid,
        "date": now_iso(),
        "meta": meta,
        "text": text,
        "blocks": blocks,
        "prompt_hash": sha1(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
    }
    add_to_stock(category, payload)
    return jsonify({"ok": True, "id": qid})

@app.post("/api/stock/delete")
def api_stock_delete():
    data = request.get_json(silent=True) or request.form
    item_id = (data.get("id") or "").strip()
    if not item_id:
        return jsonify({"ok": False, "error": "idが空です。"}), 400
    ok = delete_from_stock(item_id)
    return jsonify({"ok": ok})

@app.post("/api/retry")
def api_retry():
    """
    過去ストックから question_id を指定して再挑戦用に問題を取り出す
    body: {id}
    """
    data = request.get_json(silent=True) or request.form
    qid = (data.get("id") or "").strip()
    if not qid:
        return jsonify({"ok": False, "error": "idが空です。"}), 400

    store = list_stock()
    for key in ("social", "science"):
        for it in store.get(key, []):
            if it.get("id") == qid:
                # 新しい attempt を作る想定：ここでは問題文をそのまま返却
                return jsonify({"ok": True, "text": it.get("text",""), "meta": it.get("meta",{}), "id": qid})
    return jsonify({"ok": False, "error": "指定IDの問題が見つかりませんでした。"}), 404

@app.get("/api/history/list")
def api_history_list():
    try:
        limit = int(request.args.get("limit") or "100")
    except Exception:
        limit = 100
    data = get_history(limit)
    return jsonify({"ok": True, "data": data})

# ====== 起動 ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=False)

