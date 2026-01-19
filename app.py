# app.py
# Flask API（個人運用・一般公開想定）
# - ルート(/)で templates/index.html を返す（UIはレイアウト変更なし）
# - CORS: ALLOWED_ORIGIN（カンマ区切り、末尾スラなし）
# - 永続化: DATA_DIR（例: /var/data/eichan）。未設定時は /var/tmp/eichan
# - ACCESS_CODE（任意）で変更系API・情報系の一部を保護
# - EXPORT_CODE（任意）で「ダウンロード/エクスポート系」だけ別コードで保護（講師専用）
# - 重要: 生徒にヒント（模範解答/解説/出題意図）を見せたくない場合、
#         ストック取得APIは講師コードがないとヒント項目を返さない（自動マスク）
# - 機能: 出題自動生成 / 採点 / ストック（上限10・重複対策） / ログ / エクスポート(JSON/CSV) / 健康診断
# - 改善: 入力バリデーション, JSON強制, レート制限, 500/404ハンドラ, ディスク使用状況, メトリクス
# - 互換: X-Access-Code / ?access_code / ?code / Cookie(access_code) を許可
# - 追加: /api/stock/clear /api/stock/import /logs.zip /status に OpenAI 疎通
# - 新規: /api/stock/search /api/stock/random /api/model(POST) / NOVELTY_THRESHOLD / MAX_TOKENS_*
# - 将来課金用: 日次クォータ（FREE_DAILY_LIMIT / PRO_DAILY_LIMIT, /api/usage）
# - 注意: UI 側の fetch は /api/* を呼び出す想定（HTMLは変更不要）

import os
import re
import io
import csv
import json
import time
import uuid
import shutil
import zipfile
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps
from difflib import SequenceMatcher
from datetime import datetime, timezone, timedelta  # ← timedelta を追加

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# --- OpenAI (v1 SDK) ---
try:
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY は環境変数から自動読込
except Exception:
    client = None  # キー未設定やSDK未導入時の保険

# =========================
# アプリ設定
# =========================
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 1_000_000))  # 1MB

# CORS
ALLOWED_ORIGIN_RAW = os.getenv("ALLOWED_ORIGIN", "")
CORS_ORIGINS = [o.strip() for o in ALLOWED_ORIGIN_RAW.split(",") if o.strip()]
# credentials と * は併用不可。未設定時は * / 資格情報オフ、設定済みはピン留め / 資格情報オン
if CORS_ORIGINS:
    CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}}, supports_credentials=True)
else:
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# 永続保存先
DATA_DIR = os.getenv("DATA_DIR", "/var/tmp/eichan")
DATA = Path(DATA_DIR)
STOCK_DIR = DATA / "stocks"     # 1問1ファイル(JSON)
LOG_DIR = DATA / "logs"
UPLOAD_DIR = DATA / "uploads"
USAGE_DIR = DATA / "usage"      # ← 日次クォータ用
for d in (DATA, STOCK_DIR, LOG_DIR, UPLOAD_DIR, USAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 旧パスからの引っ越し（空のときだけ安全にコピー）
_previous_candidates = [Path("/var/data/cyopa"), Path("/var/tmp/cyopa")]
def _dir_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())
if not _dir_nonempty(STOCK_DIR) and not _dir_nonempty(LOG_DIR) and not _dir_nonempty(UPLOAD_DIR):
    for src in _previous_candidates:
        if _dir_nonempty(src):
            try:
                for sub in ("logs", "stocks", "uploads"):
                    shutil.copytree(src / sub, DATA / sub, dirs_exist_ok=True)
                print(f"[migration] copied data from {src} -> {DATA}")
            except Exception as e:
                print(f"[migration] copy from {src} failed: {e}")
            break

ACCESS_CODE = (os.getenv("ACCESS_CODE") or "").strip() or None

# ✅ 追加：エクスポート（DL）専用コード（講師だけ）
# これを設定すると、/api/export/* と /logs.zip はこのコードが必須になる
# 未設定なら従来どおりオープン
EXPORT_CODE = (os.getenv("EXPORT_CODE") or "").strip() or None

MODEL_ID = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
RELEASE = os.getenv("RELEASE", "").strip()
MAX_STOCK = int(os.getenv("MAX_STOCK", "10"))
NOVELTY_THRESHOLD = float(os.getenv("NOVELTY_THRESHOLD", "0.92"))
MAX_TOKENS_GEN = int(os.getenv("MAX_TOKENS_GEN", "600"))
MAX_TOKENS_GRADE = int(os.getenv("MAX_TOKENS_GRADE", "600"))

# レート制限（IPごと/分あたり）
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_rate_bucket: Dict[str, List[float]] = {}

# 日次クォータ（将来の課金モデル用）
# 0 以下なら「そのプランは無制限」
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "0"))  # 例: 5 にすると無料5回/日
PRO_DAILY_LIMIT = int(os.getenv("PRO_DAILY_LIMIT", "0"))    # 例: 1000 にすると有料1000回/日
DEFAULT_PLAN = (os.getenv("DEFAULT_PLAN") or "free").strip().lower()

# 入力のホワイトリスト
ALLOWED_SUBJECTS = {"社会", "理科"}
ALLOWED_CATEGORIES = {"地理", "歴史", "生物", "化学", "地学", "物理", "天体"}
ALLOWED_CAT_BY_SUBJECT = {
    "社会": {"地理", "歴史"},
    "理科": {"生物", "化学", "地学", "物理", "天体"},
}
ALLOWED_GRADES = {"中1", "中2", "中3"}
ALLOWED_DIFFICULTY = {"5点", "10点", "満点"}  # 生成時の難易度ヒント用（採点は常に10点満点表示）

# メトリクス
METRICS = {
    "gen_ok": 0, "gen_err": 0, "gen_avg_ms": 0.0, "gen_n": 0,
    "grade_ok": 0, "grade_err": 0, "grade_avg_ms": 0.0, "grade_n": 0,
}

# OpenAI疎通チェック（5分キャッシュ）
_openai_ping_cache = {"ok": None, "detail": "", "ts": 0}

# =========================
# ユーティリティ
# =========================
def now_ms() -> int:
    return int(time.time() * 1000)

def ms_to_iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone().isoformat()
    except Exception:
        return ""

def atomic_write_json(path: Path, data: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def append_jsonl(path: Path, record: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def is_similar(a: str, b: str, threshold: Optional[float] = None) -> bool:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return False
    t = threshold if isinstance(threshold, (float, int)) else NOVELTY_THRESHOLD
    return SequenceMatcher(None, a, b).ratio() >= float(t)

def list_stock_files() -> List[Path]:
    return sorted([p for p in STOCK_DIR.glob("*.json")], key=lambda p: p.stat().st_mtime)

def load_stock_item(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def current_stocks() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in list_stock_files():
        obj = load_stock_item(p)
        if obj:
            items.append(obj)
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return items

def save_stock_item(item: Dict[str, Any]) -> Dict[str, Any]:
    sid = item.get("id") or uuid.uuid4().hex
    item["id"] = sid
    item["created_at"] = item.get("created_at", now_ms())
    path = STOCK_DIR / f"{sid}.json"
    atomic_write_json(path, item)
    # 上限超過なら古いものから削除
    files = list_stock_files()
    if len(files) > MAX_STOCK:
        for p in files[: len(files) - MAX_STOCK]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    return item

def find_duplicate_in_stocks(question_text: str) -> Optional[Dict[str, Any]]:
    for it in current_stocks():
        if is_similar(it.get("question", ""), question_text):
            return it
    return None

def recent_tags(limit: int = 8) -> List[str]:
    tags: List[str] = []
    for it in current_stocks()[: 30]:  # 直近30件から収集
        for t in (it.get("tags") or []):
            nt = normalize_text(str(t))
            if nt and nt not in tags:
                tags.append(nt)
            if len(tags) >= limit:
                return tags
    return tags

def ensure_openai() -> Optional[str]:
    if client is None:
        return "OpenAI SDKの初期化に失敗しています（openai>=1.x と OPENAI_API_KEY を確認）。"
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY が未設定です。"
    return None

def parse_first_json_block(text: str) -> Dict[str, Any]:
    # モデル出力がJSON以外を含む場合の保険（最初の { ... } を抽出）
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}

def require_access_code(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if ACCESS_CODE:
            provided = (
                request.headers.get("X-Access-Code")
                or request.args.get("access_code")
                or request.args.get("code")             # 互換
                or request.cookies.get("access_code")   # /auth で設定可
            )
            if provided != ACCESS_CODE:
                return jsonify({"error": "Forbidden"}), 403
        return fn(*args, **kwargs)
    return wrapper

# ✅ 追加：エクスポート（DL）専用コード
def require_export_code(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if EXPORT_CODE:
            provided = (
                request.headers.get("X-Export-Code")
                or request.args.get("export_code")
                or request.args.get("exportCode")
                or request.cookies.get("export_code")
            )
            if provided != EXPORT_CODE:
                return jsonify({"error": "forbidden", "message": "EXPORT_CODE required"}), 403
        return fn(*args, **kwargs)
    return wrapper

def _is_teacher_request() -> bool:
    """
    ストックAPI等で「ヒントを返して良いか」を判定。
    ACCESS_CODE を教師コードとして扱う。
    """
    if not ACCESS_CODE:
        return False
    provided = (
        request.headers.get("X-Access-Code")
        or request.args.get("access_code")
        or request.args.get("code")
        or request.cookies.get("access_code")
    )
    return provided == ACCESS_CODE

def _strip_hints(item: Dict[str, Any]) -> Dict[str, Any]:
    # 生徒には見せたくない要素を落とす
    for k in ("model_answer", "explanation", "intention"):
        item.pop(k, None)
    return item

def rate_limit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if RATE_LIMIT_PER_MIN <= 0:
            return fn(*args, **kwargs)
        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        if "," in ip:
            ip = ip.split(",")[0].strip()  # 先頭を採用
        now = time.time()
        bucket = _rate_bucket.setdefault(ip, [])
        while bucket and (now - bucket[0] > 60.0):
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            return jsonify({"ok": False, "error": "Too Many Requests"}), 429
        bucket.append(now)
        return fn(*args, **kwargs)
    return wrapper

def _update_metrics(kind: str, ok: bool, t0: float):
    dt = int((time.time() - t0) * 1000)
    key = "gen" if kind == "generate" else "grade"
    if ok:
        METRICS[f"{key}_ok"] += 1
    else:
        METRICS[f"{key}_err"] += 1
    METRICS[f"{key}_n"] += 1
    prev_n = max(METRICS[f"{key}_n"], 1)
    METRICS[f"{key}_avg_ms"] = (METRICS[f"{key}_avg_ms"] * (prev_n - 1) + dt) / prev_n

# =========================
# 日次クォータ関連（将来課金モデル用）
# =========================
def _today_key() -> str:
    # サーバーローカルタイムの YYYYMMDD を使う
    return datetime.now().astimezone().strftime("%Y%m%d")

def _usage_file_for_today() -> Path:
    return USAGE_DIR / f"{_today_key()}.json"

def _load_usage_for_today() -> Dict[str, Any]:
    p = _usage_file_for_today()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_usage_for_today(data: Dict[str, Any]):
    atomic_write_json(_usage_file_for_today(), data)

def _get_user_key() -> str:
    # 将来: X-User-Id にユーザーID/メールなどを入れる想定
    uid = (
        request.headers.get("X-User-Id")
        or request.headers.get("X-Forwarded-For", "")
        or request.remote_addr
        or "anonymous"
    )
    uid = normalize_text(uid)
    return uid or "anonymous"

def _plan_and_limit() -> Tuple[str, Optional[int]]:
    """
    現時点では:
      - X-User-Plan ヘッダー or Cookie(user_plan) or DEFAULT_PLAN をプラン名として採用
      - free → FREE_DAILY_LIMIT
      - pro  → PRO_DAILY_LIMIT
    0 以下の値なら「そのプランは無制限」として扱う。
    """
    plan = (
        (request.headers.get("X-User-Plan") or "")
        or (request.cookies.get("user_plan") or "")
        or DEFAULT_PLAN
    ).strip().lower() or "free"
    if plan == "pro":
        limit = PRO_DAILY_LIMIT
    else:
        limit = FREE_DAILY_LIMIT
    if limit <= 0:
        return plan, None  # 無制限
    return plan, limit

def _next_reset_iso() -> str:
    # 次の0時（サーバータイム）のISO文字列
    now = datetime.now().astimezone()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.isoformat()

def enforce_quota(kind: str):
    """
    kind: "generate" / "grade"
    FREE_DAILY_LIMIT / PRO_DAILY_LIMIT を超えた場合は 429 を返す。
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            plan, limit = _plan_and_limit()
            # 無制限プランならそのまま通す
            if limit is None:
                return fn(*args, **kwargs)

            user_key = _get_user_key()
            usage = _load_usage_for_today()
            u = usage.get(user_key) or {"total": 0, "generate": 0, "grade": 0}

            if u.get("total", 0) >= limit:
                reset_at = _next_reset_iso()
                payload = {
                    "ok": False,
                    "error": "quota_exceeded",
                    "message": "無料プランの1日あたりの利用回数を超えています。",
                    "plan": plan,
                    "limit": limit,
                    "total": u.get("total", 0),
                    "reset_at": reset_at,
                }
                append_jsonl(LOG_DIR / "usage.jsonl", {
                    "event": "quota_exceeded",
                    "user_key": user_key,
                    "plan": plan,
                    "limit": limit,
                    "usage": u,
                    "ts": now_ms(),
                })
                return jsonify(payload), 429

            # まだ上限未満ならカウントをインクリメントして続行
            u["total"] = u.get("total", 0) + 1
            if kind == "generate":
                u["generate"] = u.get("generate", 0) + 1
            elif kind == "grade":
                u["grade"] = u.get("grade", 0) + 1
            usage[user_key] = u
            _save_usage_for_today(usage)

            append_jsonl(LOG_DIR / "usage.jsonl", {
                "event": "quota_used",
                "user_key": user_key,
                "plan": plan,
                "kind": kind,
                "limit": limit,
                "usage": u,
                "ts": now_ms(),
            })
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# =========================
# バリデーション
# =========================
def validate_generation_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    subject = p.get("subject", "")
    if subject not in ALLOWED_SUBJECTS:
        subject = "社会"

    category = p.get("category", "")
    allowed_cats = ALLOWED_CAT_BY_SUBJECT[subject]
    if category not in allowed_cats:
        category = ("地理" if subject == "社会" else "生物")

    grade = p.get("grade", "")
    if subject == "理科":
        if grade not in ALLOWED_GRADES:
            grade = "中3"
        # 天体は中3のみ許可
        if category == "天体" and grade != "中3":
            category = "地学"
    else:
        grade = ""  # 社会は空でもOK

    difficulty = p.get("difficulty", "10点")
    if difficulty not in ALLOWED_DIFFICULTY:
        difficulty = "10点"

    genre_hint = normalize_text(p.get("genre_hint", ""))[:120]
    avoid = normalize_text(p.get("avoid_topics", ""))[:120]
    length_hint = normalize_text(p.get("length_hint", "30〜80字程度"))[:40]
    include_hints = bool(p.get("include_hints", False))
    return {
        "subject": subject,
        "category": category,
        "grade": grade,
        "difficulty": difficulty,
        "genre_hint": genre_hint,
        "avoid_topics": avoid,
        "length_hint": length_hint,
        "include_hints": include_hints,
    }

def validate_grading_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    question = normalize_text(p.get("question", ""))[:2000]
    student_answer = normalize_text(p.get("student_answer", ""))[:2000]
    model_answer = normalize_text(p.get("model_answer", ""))[:2000]
    difficulty = p.get("difficulty", "10点")
    if difficulty not in ALLOWED_DIFFICULTY:
        difficulty = "10点"
    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": model_answer,
        "difficulty": difficulty,
    }

def check_openai_connectivity(force: bool = False) -> Tuple[Optional[bool], str, int]:
    """OpenAIとの疎通を軽く確認（5分キャッシュ／対象モデルのみ）"""
    ttl_sec = 300
    now = int(time.time())
    if not force and _openai_ping_cache["ts"] and (now - _openai_ping_cache["ts"] < ttl_sec):
        return _openai_ping_cache["ok"], _openai_ping_cache["detail"], _openai_ping_cache["ts"]
    if client is None or not os.getenv("OPENAI_API_KEY"):
        _openai_ping_cache.update({"ok": False, "detail": "SDK未初期化 or APIキー未設定", "ts": now})
        return False, _openai_ping_cache["detail"], now
    try:
        client.models.retrieve(MODEL_ID)
        ok = True
        _openai_ping_cache.update({"ok": ok, "detail": "ok", "ts": now})
        return ok, "ok", now
    except Exception as e:
        _openai_ping_cache.update({"ok": False, "detail": str(e), "ts": now})
        return False, str(e), now

# =========================
# OpenAI プロンプト
# =========================
def build_generation_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    subject = payload["subject"]
    category = payload["category"]
    grade = payload["grade"]
    difficulty = payload["difficulty"]
    genre_hint = payload["genre_hint"]
    avoid = payload["avoid_topics"]
    length_hint = payload["length_hint"]

    recent = recent_tags()
    if recent:
        extra_avoid = "、".join(recent[:8])
        avoid = f"{avoid}（直近の重複回避候補: {extra_avoid}）" if avoid else f"直近の重複回避候補: {extra_avoid}"

    sysprompt = (
        "あなたは中学生向けの記述式問題作成の専門家です。"
        "日本の中学カリキュラムおよび宮城県公立高校入試の傾向に合わせ、"
        "図や資料がなくても解ける良問を作ります。"
        "問題文は簡潔かつ明確に、具体例と条件を適切に与えてください。"
        "外部サイトや画像を参照せず、テキストのみで完結させてください。"
    )
    usr = (
        f"教科: {subject}\n"
        f"分類: {category}\n"
        f"学年: {grade or '（理科以外は空で可）'}\n"
        f"難易度/配点: {difficulty}\n"
        f"出題の方向性（任意）: {genre_hint or '特になし'}\n"
        f"避ける話題（任意）: {avoid or '特になし'}\n"
        f"解答分量の目安: {length_hint}\n\n"
        "要件:\n"
        "1) テキストのみで成立する記述問題を1問。\n"
        "   ・模範解答は30〜80字程度\n"
        "   ・『理由・因果・しくみ』を問う\n"
        "2) JSONで返す: {"
        '"question": str, "model_answer": str, "explanation": str, "intention": str, "tags": [str]'
        "}\n"
        "3) 同一テーマの連発を避ける\n"
    )
    return [{"role": "system", "content": sysprompt},
            {"role": "user", "content": usr}]

def build_grading_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    q = payload["question"]
    sa = payload["student_answer"]
    ma = payload.get("model_answer", "")
    difficulty = payload["difficulty"]

    sysprompt = (
        "あなたは中学生の記述解答を採点する試験官です。"
        "採点は0〜10点の整数。"
        "さらに『入試本番で満点（○）になる確率』を0〜100の整数で推定し、"
        "短い根拠を perfect_probability_note に書いてください。"
        "確率は『満点になる確率』なので、scoreが10でも100とは限りません（表現の曖昧さ・要点漏れの可能性を考慮）。"
        "出力は必ずJSONで、"
        '{"score":int,"perfect_probability":int,"perfect_probability_note":str,"commentary":str,"model_answer":str,"reasons":[str]}。'
        "commentary の冒頭に『10点中◯点』を必ず含める。"
        "reasons は3〜6個、箇条書きの短文。"
        "日本語で丁寧かつ簡潔に。"
    )
    usr = (
        f"問題: {q}\n"
        f"受験者の解答: {sa}\n"
        f"参考解答（任意）: {ma or '（なし）'}\n"
        f"配点設定: {difficulty}\n"
        "評価観点：正確性/要点の網羅/論理性/表現の明瞭さ。"
    )
    return [{"role": "system", "content": sysprompt},
            {"role": "user", "content": usr}]

# =========================
# ルート（UI）
# =========================
@app.get("/")
def root():
    return render_template("index.html")

# =========================
# エラーハンドラ
# =========================
@app.errorhandler(404)
def _404(_e):
    return jsonify({"ok": False, "error": "Not Found"}), 404

@app.errorhandler(500)
def _500(e):
    append_jsonl(LOG_DIR / "errors.jsonl", {
        "event": "server_error",
        "path": request.path,
        "method": request.method,
        "error": str(e),
        "ts": now_ms()
    })
    return jsonify({"ok": False, "error": "Internal Server Error"}), 500

# =========================
# 運用系
# =========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_ms()}, 200

@app.get("/status")
@require_access_code
def status():
    try:
        usage = shutil.disk_usage(DATA)
        disk = {"total": usage.total, "used": usage.used, "free": usage.free}
    except Exception:
        disk = None
    ok, detail, ts = check_openai_connectivity(force=False)

    def preview(v: Optional[str]) -> str:
        if not v:
            return "NOT SET"
        return (v[:4] + "…" + v[-2:]) if len(v) > 8 else "SET"

    return jsonify({
        "ok": True,
        "DATA_DIR": str(DATA),
        "CORS": CORS_ORIGINS,
        "model": MODEL_ID,
        "release": RELEASE,
        "disk": disk,
        "metrics": {
            "generate": {"ok": METRICS["gen_ok"], "err": METRICS["gen_err"], "avg_ms": round(METRICS["gen_avg_ms"], 1), "n": METRICS["gen_n"]},
            "grade":    {"ok": METRICS["grade_ok"], "err": METRICS["grade_err"], "avg_ms": round(METRICS["grade_avg_ms"], 1), "n": METRICS["grade_n"]},
        },
        "openai": {"ok": ok, "detail": detail, "checked_at": ms_to_iso(ts * 1000)},
        "novelty_threshold": NOVELTY_THRESHOLD,
        "max_stock": MAX_STOCK,
        "quota": {
            "free_daily_limit": FREE_DAILY_LIMIT,
            "pro_daily_limit": PRO_DAILY_LIMIT,
            "default_plan": DEFAULT_PLAN,
        },
        "codes": {
            "ACCESS_CODE": preview(ACCESS_CODE),
            "EXPORT_CODE": preview(EXPORT_CODE),
        }
    }), 200

@app.get("/env-check")
@require_access_code
def env_check():
    def preview(v: Optional[str]) -> str:
        if not v:
            return "NOT SET"
        return (v[:4] + "…" + v[-2:]) if len(v) > 8 else "SET"
    return jsonify({
        "ok": True,
        "DATA_DIR": str(DATA),
        "ALLOWED_ORIGIN": CORS_ORIGINS,
        "OPENAI_API_KEY": preview(os.getenv("OPENAI_API_KEY")),
        "ACCESS_CODE": preview(ACCESS_CODE),
        "EXPORT_CODE": preview(EXPORT_CODE),
        "MODEL_ID": MODEL_ID,
        "RELEASE": RELEASE or "NOT SET",
        "MAX_TOKENS_GEN": MAX_TOKENS_GEN,
        "MAX_TOKENS_GRADE": MAX_TOKENS_GRADE,
        "FREE_DAILY_LIMIT": FREE_DAILY_LIMIT,
        "PRO_DAILY_LIMIT": PRO_DAILY_LIMIT,
        "DEFAULT_PLAN": DEFAULT_PLAN,
    }), 200

@app.post("/_probe-write")
@require_access_code
def probe_write():
    p = LOG_DIR / ".probe"
    p.write_text("ok", encoding="utf-8")
    return {"wrote": str(p)}, 200

# ---- 簡易ログイン（Cookie保存・任意） ----
@app.post("/auth")
def auth():
    code = (request.get_json(silent=True) or {}).get("code", "")
    if not ACCESS_CODE:
        return jsonify({"ok": True, "message": "ACCESS_CODE未設定（オープン）"})
    if code == ACCESS_CODE:
        resp = jsonify({"ok": True})
        is_secure = (request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https")
        resp.set_cookie("access_code", code, httponly=True, samesite="Lax", secure=is_secure)
        return resp
    return jsonify({"ok": False, "error": "invalid_code"}), 401

# ---- 日次クォータの確認用API（UIからはまだ未使用）----
@app.get("/api/usage")
def api_usage():
    plan, limit = _plan_and_limit()
    usage = _load_usage_for_today()
    user_key = _get_user_key()
    u = usage.get(user_key) or {"total": 0, "generate": 0, "grade": 0}
    remaining = None
    if limit is not None:
        remaining = max(0, limit - u.get("total", 0))
    return jsonify({
        "ok": True,
        "plan": plan,
        "limit": limit,
        "usage": u,
        "remaining": remaining,
        "reset_at": _next_reset_iso(),
    }), 200

# =========================
# ストック API
# =========================
@app.get("/api/stock/list")
def stock_list():
    items = current_stocks()
    # ✅ 生徒にヒントを見せない（講師コードが無ければヒントを落とす）
    if not _is_teacher_request():
        items = [_strip_hints(dict(it)) for it in items]
    return jsonify({"ok": True, "items": items, "count": len(items), "limit": MAX_STOCK}), 200

@app.get("/api/stock/search")
def stock_search():
    q = normalize_text(request.args.get("q", ""))
    subject = request.args.get("subject", "")
    category = request.args.get("category", "")
    grade = request.args.get("grade", "")
    source = request.args.get("source", "")
    try:
        limit = max(1, min(100, int(request.args.get("limit", "20"))))
    except Exception:
        limit = 20
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except Exception:
        offset = 0

    def match(it: Dict[str, Any]) -> bool:
        if subject and it.get("subject") != subject:
            return False
        if category and it.get("category") != category:
            return False
        if grade and it.get("grade") != grade:
            return False
        if source and it.get("source") != source:
            return False
        if q:
            hay = " ".join([
                it.get("question", ""), " ".join(it.get("tags") or []),
                it.get("model_answer", ""), it.get("explanation", ""),
                it.get("intention", "")
            ])
            if q.lower() not in hay.lower():
                return False
        return True

    items = [it for it in current_stocks() if match(it)]
    total = len(items)
    sliced = items[offset: offset + limit]

    if not _is_teacher_request():
        sliced = [_strip_hints(dict(it)) for it in sliced]

    return jsonify({"ok": True, "total": total, "items": sliced, "limit": limit, "offset": offset}), 200

@app.get("/api/stock/random")
def stock_random():
    q = request.args.get("q", "")
    subject = request.args.get("subject", "")
    category = request.args.get("category", "")
    grade = request.args.get("grade", "")
    source = request.args.get("source", "")
    exclude_ids = set([s for s in (request.args.get("exclude_ids", "") or "").split(",") if s])

    def match(it: Dict[str, Any]) -> bool:
        if it.get("id") in exclude_ids:
            return False
        if subject and it.get("subject") != subject:
            return False
        if category and it.get("category") != category:
            return False
        if grade and it.get("grade") != grade:
            return False
        if source and it.get("source") != source:
            return False
        if q and q.lower() not in (" ".join([
            it.get("question", ""), " ".join(it.get("tags") or []),
            it.get("model_answer", ""), it.get("explanation", ""),
            it.get("intention", "")
        ])).lower():
            return False
        return True

    candidates = [it for it in current_stocks() if match(it)]
    if not candidates:
        return jsonify({"ok": False, "error": "no_candidates"}), 404
    item = random.choice(candidates)

    if not _is_teacher_request():
        item = _strip_hints(dict(item))

    return jsonify({"ok": True, "item": item}), 200

@app.post("/api/stock/add")
@require_access_code
@rate_limit
def stock_add():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(force=True, silent=True) or {}
    question = normalize_text(data.get("question", ""))[:2000]
    if not question:
        return jsonify({"error": "question is required"}), 400

    dup = find_duplicate_in_stocks(question)
    if dup:
        return jsonify({"ok": True, "duplicate_of": dup.get("id"), "item": dup}), 200

    subject = data.get("subject", "")[:20]
    if subject not in ALLOWED_SUBJECTS:
        subject = ""
    category = data.get("category", "")[:20]
    if subject and category and category not in ALLOWED_CAT_BY_SUBJECT[subject]:
        category = ""
    grade = data.get("grade", "")[:10]
    if grade and grade not in ALLOWED_GRADES:
        grade = ""

    item = {
        "id": uuid.uuid4().hex,
        "question": question,
        "subject": subject,
        "category": category,
        "grade": grade,
        "difficulty": (data.get("difficulty", "10点") if data.get("difficulty", "10点") in ALLOWED_DIFFICULTY else "10点"),
        "tags": (data.get("tags", []) if isinstance(data.get("tags", []), list) else [])[:10],
        "source": "manual",
        "created_at": now_ms(),
    }
    saved = save_stock_item(item)
    append_jsonl(LOG_DIR / "stocks.jsonl", {"event": "add", "item": saved, "ts": now_ms()})
    return jsonify({"ok": True, "item": saved}), 200

@app.post("/api/stock/delete")
@require_access_code
@rate_limit
def stock_delete():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(force=True, silent=True) or {}
    sid = (data.get("id") or "").strip()
    if not sid:
        return jsonify({"error": "id is required"}), 400
    path = STOCK_DIR / f"{sid}.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "not found"}), 404
    path.unlink(missing_ok=True)
    append_jsonl(LOG_DIR / "stocks.jsonl", {"event": "delete", "id": sid, "ts": now_ms()})
    return jsonify({"ok": True, "deleted": sid}), 200

@app.post("/api/stock/clear")
@require_access_code
@rate_limit
def stock_clear():
    files = list(STOCK_DIR.glob("*.json"))
    cnt = 0
    for p in files:
        try:
            p.unlink(missing_ok=True)
            cnt += 1
        except Exception:
            pass
    append_jsonl(LOG_DIR / "stocks.jsonl", {"event": "clear", "deleted": cnt, "ts": now_ms()})
    return jsonify({"ok": True, "deleted": cnt}), 200

@app.post("/api/stock/import")
@require_access_code
@rate_limit
def stock_import():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415
    body = request.get_json(force=True, silent=True) or {}
    items = body.get("items")
    if items is None and isinstance(body, list):
        items = body
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "items(list) が必要です"}), 400

    dry = bool(body.get("dry_run", False))
    skip_dup = True if body.get("skip_duplicates", True) else False
    overwrite = True if body.get("overwrite_if_similar", False) else False
    truncate = True if body.get("truncate_before", False) else False

    if truncate and not dry:
        for p in STOCK_DIR.glob("*.json"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    added, skipped, replaced = 0, 0, 0
    saved_ids: List[str] = []
    for raw in items:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        q = normalize_text(raw.get("question", ""))[:2000]
        if not q:
            skipped += 1
            continue
        dup_item = find_duplicate_in_stocks(q)
        if dup_item and skip_dup and not overwrite:
            skipped += 1
            continue

        subject = raw.get("subject", "")
        category = raw.get("category", "")
        grade = raw.get("grade", "")

        if subject not in ALLOWED_SUBJECTS:
            subject = ""
        if subject and category and category not in ALLOWED_CAT_BY_SUBJECT[subject]:
            category = ""
        if grade and grade not in ALLOWED_GRADES:
            grade = ""

        item = {
            "id": raw.get("id") or uuid.uuid4().hex,
            "question": q,
            "subject": subject,
            "category": category,
            "grade": grade,
            "difficulty": (raw.get("difficulty", "10点") if raw.get("difficulty", "10点") in ALLOWED_DIFFICULTY else "10点"),
            "tags": [normalize_text(str(t))[:20] for t in (raw.get("tags") or [])][:10] if isinstance(raw.get("tags"), list) else [],
            "model_answer": normalize_text(raw.get("model_answer", ""))[:2000],
            "explanation": (raw.get("explanation", "") or "").strip(),
            "intention": (raw.get("intention", "") or "").strip(),
            "source": raw.get("source", "import"),
            "created_at": int(raw.get("created_at", now_ms())),
        }

        if dry:
            added += 1
            saved_ids.append(item["id"])
            continue

        if dup_item and overwrite:
            item["id"] = dup_item.get("id")
            path = STOCK_DIR / f"{item['id']}.json"
            atomic_write_json(path, item)
            replaced += 1
            saved_ids.append(item["id"])
        else:
            saved = save_stock_item(item)
            added += 1
            saved_ids.append(saved["id"])

    append_jsonl(LOG_DIR / "stocks.jsonl", {
        "event": "import",
        "dry_run": dry,
        "truncate_before": truncate,
        "added": added,
        "skipped": skipped,
        "replaced": replaced,
        "ts": now_ms()
    })
    return jsonify({"ok": True, "dry_run": dry, "added": added, "skipped": skipped, "replaced": replaced, "ids_sample": saved_ids[:10]}), 200

# =========================
# 生成 API
# =========================
@app.post("/api/generate_question")
@require_access_code
@rate_limit
@enforce_quota("generate")
def generate_question():
    t0 = time.time()
    err = ensure_openai()
    if err:
        _update_metrics("generate", False, t0)
        return jsonify({"ok": False, "error": err}), 503
    if not request.is_json:
        _update_metrics("generate", False, t0)
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload_in = request.get_json(force=True, silent=True) or {}
    payload = validate_generation_payload(payload_in)
    messages = build_generation_messages(payload)

    try:
        rsp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.6,
            max_tokens=MAX_TOKENS_GEN,
        )
        text = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        _update_metrics("generate", False, t0)
        return jsonify({"ok": False, "error": f"OpenAI API error: {e}"}), 502

    obj = parse_first_json_block(text) or {}
    question = normalize_text(obj.get("question", ""))[:2000]
    model_answer = normalize_text(obj.get("model_answer", ""))[:2000]
    explanation = (obj.get("explanation", "") or "").strip()
    intention = (obj.get("intention", "") or "").strip()
    tags = obj.get("tags", []) or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [normalize_text(t)[:20] for t in tags][:10]

    if not question:
        question = normalize_text(text)[:2000]
    if not question:
        _update_metrics("generate", False, t0)
        return jsonify({"ok": False, "error": "generation failed"}), 502

    dup = find_duplicate_in_stocks(question)
    if dup:
        append_jsonl(LOG_DIR / "generation.jsonl", {"event": "generated-duplicate", "question": question, "ts": now_ms()})
        _update_metrics("generate", True, t0)
        generated = {
            "question": question,
            "model_answer": model_answer,
            "explanation": explanation,
            "intention": intention,
            "tags": tags
        }
        return jsonify({"ok": True, "duplicate_of": dup.get("id"), "item": dup, "generated": generated}), 200

    item = {
        "id": uuid.uuid4().hex,
        "source": "auto",
        "question": question,
        "model_answer": model_answer,
        "explanation": explanation,
        "intention": intention,
        "tags": tags,
        "subject": payload["subject"],
        "category": payload["category"],
        "grade": payload["grade"],
        "difficulty": payload["difficulty"],
        "created_at": now_ms(),
    }
    saved = save_stock_item(item)
    append_jsonl(LOG_DIR / "generation.jsonl", {"event": "generated", "item": saved, "ts": now_ms()})

    resp_item = dict(saved)
    if not payload["include_hints"]:
        for k in ("model_answer", "explanation", "intention"):
            resp_item.pop(k, None)

    _update_metrics("generate", True, t0)
    return jsonify({"ok": True, "item": resp_item}), 200

# =========================
# 採点 API
# =========================
@app.post("/api/grade_answer")
@require_access_code
@rate_limit
@enforce_quota("grade")
def grade_answer():
    t0 = time.time()
    err = ensure_openai()
    if err:
        _update_metrics("grade", False, t0)
        return jsonify({"ok": False, "error": err}), 503
    if not request.is_json:
        _update_metrics("grade", False, t0)
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data_in = request.get_json(force=True, silent=True) or {}
    data = validate_grading_payload(data_in)

    if not data["question"] or not data["student_answer"]:
        _update_metrics("grade", False, t0)
        return jsonify({"ok": False, "error": "question と student_answer は必須です"}), 400

    messages = build_grading_messages(data)
    try:
        rsp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_TOKENS_GRADE,
        )
        text = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        _update_metrics("grade", False, t0)
        return jsonify({"ok": False, "error": f"OpenAI API error: {e}"}), 502

    result = parse_first_json_block(text) or {}
    try:
        score = int(result.get("score", 0))
    except Exception:
        score = 0
    score = max(0, min(10, score))
    commentary = (result.get("commentary", "") or "").strip()
    head = f"10点中{score}点"
    if not commentary.startswith(head):
        commentary = f"{head}：{commentary}" if commentary else f"{head}。"
    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [normalize_text(str(x))[:200] for x in reasons][:8]
    model_ans = (result.get("model_answer", "") or data.get("model_answer", "")).strip()

    prob_raw = result.get("perfect_probability", None)
    prob_note = (result.get("perfect_probability_note", "") or "").strip()

    prob: Optional[int] = None
    if prob_raw is not None:
        try:
            digits = re.sub(r"[^0-9]", "", str(prob_raw))
            prob = int(digits) if digits != "" else 0
        except Exception:
            prob = None

    if prob is None:
        prob = int(round(score * 10))
        if not prob_note:
            prob_note = "スコアからの簡易推定"

    prob = max(0, min(100, int(prob)))

    append_jsonl(LOG_DIR / "grading.jsonl", {
        "event": "graded",
        "question": data["question"],
        "student_answer": data["student_answer"],
        "score": score,
        "perfect_probability": prob,
        "ts": now_ms()
    })

    _update_metrics("grade", True, t0)
    return jsonify({
        "ok": True,
        "score": score,
        "commentary": commentary,
        "model_answer": model_ans,
        "reasons": reasons,
        "perfect_probability": prob,
        "perfect_probability_note": prob_note,
        "max_score": 10
    }), 200

# =========================
# エクスポート（✅講師専用コードで保護）
# =========================
@app.get("/api/export/stocks.json")
@require_export_code
def export_stocks_json():
    items = current_stocks()
    tmp = DATA / "stocks_export.json"
    atomic_write_json(tmp, items)
    return send_file(tmp, as_attachment=True, download_name="stocks.json", mimetype="application/json")

@app.get("/api/export/stocks.csv")
@require_export_code
def export_stocks_csv():
    items = current_stocks()
    tmp = DATA / "stocks_export.csv"
    fields = [
        "id", "subject", "category", "grade", "difficulty",
        "question", "model_answer", "explanation", "intention",
        "tags", "source", "created_at_iso"
    ]
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for it in items:
            writer.writerow({
                "id": it.get("id", ""),
                "subject": it.get("subject", ""),
                "category": it.get("category", ""),
                "grade": it.get("grade", ""),
                "difficulty": it.get("difficulty", ""),
                "question": it.get("question", ""),
                "model_answer": it.get("model_answer", ""),
                "explanation": it.get("explanation", ""),
                "intention": it.get("intention", ""),
                "tags": ";".join(it.get("tags", []) or []),
                "source": it.get("source", ""),
                "created_at_iso": ms_to_iso(it.get("created_at", 0)),
            })
    return send_file(tmp, as_attachment=True, download_name="stocks.csv", mimetype="text/csv")

@app.get("/api/export/grading.jsonl")
@require_export_code
def export_grading_jsonl():
    path = LOG_DIR / "grading.jsonl"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return send_file(path, as_attachment=True, download_name="grading.jsonl", mimetype="application/jsonl")

@app.get("/logs.zip")
@require_export_code
def download_logs_zip():
    include = (request.args.get("include") or "").lower()
    with io.BytesIO() as mem:
        with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("errors.jsonl", "generation.jsonl", "grading.jsonl", "stocks.jsonl"):
                p = LOG_DIR / name
                if p.exists():
                    z.write(p, arcname=f"logs/{name}")
            if include == "stocks":
                for p in STOCK_DIR.glob("*.json"):
                    z.write(p, arcname=f"stocks/{p.name}")
        mem.seek(0)
        return send_file(mem, as_attachment=True, download_name="logs.zip", mimetype="application/zip")

# ---- モデル情報 ----
@app.get("/api/model")
def api_model():
    return {"model": MODEL_ID}, 200

@app.post("/api/model")
@require_access_code
def set_model():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    body = request.get_json(force=True, silent=True) or {}
    new_model = normalize_text(body.get("model", ""))
    if not new_model:
        return jsonify({"ok": False, "error": "model is required"}), 400
    global MODEL_ID
    MODEL_ID = new_model
    return jsonify({"ok": True, "model": MODEL_ID}), 200

# ---- エントリポイント ----
if __name__ == "__main__":
    # ✅ ローカルは 5001 をデフォルト（Renderは PORT が入るので影響なし）
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=env_bool("DEBUG"))
