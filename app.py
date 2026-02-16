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
import unicodedata
import io
import csv
import json
import time
import uuid
import shutil
import zipfile
import random
import hmac
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps
from difflib import SequenceMatcher
from datetime import datetime, timezone, timedelta  # ← timedelta を追加

from flask import Flask, request, jsonify, send_file, render_template, session
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
app.secret_key = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

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

# ✅ エクスポート（DL）専用コード（講師だけ）
# これを設定すると、/api/export/* と /logs.zip はこのコードが必須になる
# 未設定の場合は常に403（うっかり公開を防ぐ）
EXPORT_CODE = (os.getenv("EXPORT_CODE") or "").strip() or None

# OpenAIモデルは全機能で必ずこの定数を参照する（直書き禁止）
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_ID = DEFAULT_MODEL
RELEASE = os.getenv("RELEASE", "").strip()
MAX_STOCK = int(os.getenv("MAX_STOCK", "10"))
NOVELTY_THRESHOLD = float(os.getenv("NOVELTY_THRESHOLD", "0.92"))
MAX_TOKENS_GEN = int(os.getenv("MAX_TOKENS_GEN", "600"))
MAX_TOKENS_GRADE = int(os.getenv("MAX_TOKENS_GRADE", "600"))
MAX_TOKENS_ASK_AI = int(os.getenv("MAX_TOKENS_ASK_AI", "1000"))

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
ALLOWED_DIFFICULTY = {"5点", "10点", "満点"}  # 生成時の難易度ヒント用（採点はUI側で配点表示）

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

def normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。,.．・!！?？:：;；（）()「」『』【】［］\[\]<>＜＞\"'’“”]", "", s)
    return s.lower()

def is_similar(a: str, b: str, threshold: Optional[float] = None) -> bool:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return False
    t = threshold if isinstance(threshold, (float, int)) else NOVELTY_THRESHOLD
    return SequenceMatcher(None, a, b).ratio() >= float(t)

def is_high_match(a: str, b: str, threshold: float = 0.98) -> bool:
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold

def list_stock_files() -> List[Path]:
    return sorted([p for p in STOCK_DIR.glob("*.json")], key=lambda p: p.stat().st_mtime)

def load_stock_item(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def enrich_stock_item(item: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(item)
    question = enriched.get("question") or enriched.get("question_text") or ""
    enriched["question"] = question
    enriched["question_text"] = question
    domain = enriched.get("domain") or enriched.get("category") or ""
    enriched["domain"] = domain
    enriched["category"] = enriched.get("category") or domain
    enriched["unit"] = normalize_text(enriched.get("unit", ""))[:60]
    difficulty = enriched.get("difficulty", "10点")
    max_points = enriched.get("max_points")
    try:
        max_points_int = int(max_points)
    except Exception:
        max_points_int = difficulty_max_score(difficulty)
    max_points_int = max(1, max_points_int)
    enriched["max_points"] = max_points_int
    return enriched

def current_stocks() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in list_stock_files():
        obj = load_stock_item(p)
        if obj:
            items.append(enrich_stock_item(obj))
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return items

def save_stock_item(item: Dict[str, Any]) -> Dict[str, Any]:
    item = enrich_stock_item(item)
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

def find_duplicate_in_stocks(
    question_text: str,
    subject: Optional[str] = None,
    category: Optional[str] = None,
    grade: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for it in current_stocks():
        if subject and it.get("subject") != subject:
            continue
        if category and it.get("category") != category:
            continue
        if grade and it.get("grade") != grade:
            continue
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

STOPWORDS = {
    "それ", "これ", "ため", "もの", "こと", "よう", "など", "として", "する",
    "必要", "重要", "的", "について", "なぜ", "どの", "どのよう", "どんな",
    "理由", "原因", "結果", "影響", "比較", "説明", "用語", "資料", "グラフ",
    "社会", "理科", "地理", "歴史", "生物", "化学", "地学", "物理", "天体",
}

ANGLE_OPTIONS = [
    {"key": "cause", "label": "原因・理由", "hint": "原因/理由/背景を問う観点"},
    {"key": "effect", "label": "結果・影響", "hint": "結果/影響/メリット・デメリットを問う観点"},
    {"key": "compare", "label": "比較", "hint": "AとBの違い/共通点を問う観点"},
    {"key": "define", "label": "用語説明", "hint": "用語の説明/しくみを問う観点"},
    {"key": "evidence", "label": "根拠", "hint": "根拠や具体例を挙げる観点"},
]

def extract_keywords(text: str, limit: int = 12) -> List[str]:
    tokens = re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9]+", text or "")
    out: List[str] = []
    for token in tokens:
        t = normalize_text(token)
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out

def keyword_overlap_score(a: str, b: str) -> float:
    ka = set(extract_keywords(a, limit=24))
    kb = set(extract_keywords(b, limit=24))
    if not ka or not kb:
        return 0.0
    inter = ka.intersection(kb)
    union = ka.union(kb)
    return len(inter) / max(len(union), 1)

def detect_angle(text: str) -> Optional[str]:
    t = text or ""
    if re.search(r"比較|違い|共通", t):
        return "compare"
    if re.search(r"原因|理由|背景|なぜ", t):
        return "cause"
    if re.search(r"結果|影響|メリット|デメリット", t):
        return "effect"
    if re.search(r"仕組み|しくみ|定義|説明|用語", t):
        return "define"
    if re.search(r"根拠|資料|データ|事例", t):
        return "evidence"
    return None

def pick_next_angle(recent_qs: List[str]) -> Dict[str, str]:
    recent_angles = [detect_angle(q) for q in recent_qs[:2]]
    recent_angles = [a for a in recent_angles if a]
    for opt in ANGLE_OPTIONS:
        if opt["key"] not in recent_angles:
            return opt
    return random.choice(ANGLE_OPTIONS)

def recent_questions(
    limit: int = 10,
    subject: Optional[str] = None,
    category: Optional[str] = None,
    grade: Optional[str] = None,
    unit: Optional[str] = None,
) -> List[str]:
    questions: List[str] = []
    for it in current_stocks():
        if subject and it.get("subject") != subject:
            continue
        if category and it.get("category") != category:
            continue
        if grade and it.get("grade") != grade:
            continue
        if unit and it.get("unit") != unit:
            continue
        q = normalize_text(it.get("question", ""))
        if q:
            questions.append(q)
        if len(questions) >= limit:
            break
    return questions

def ensure_openai() -> Optional[str]:
    if client is None:
        return "OpenAI SDKの初期化に失敗しています（openai>=1.x と OPENAI_API_KEY を確認）。"
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY が未設定です。"
    return None

def difficulty_max_score(difficulty: str) -> int:
    if difficulty == "5点":
        return 5
    if difficulty == "満点":
        return 100
    return 10

def difficulty_label(difficulty: str) -> str:
    return f"{difficulty_max_score(difficulty)}点"

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


def short_text(s: Any, limit: int = 200) -> str:
    txt = normalize_text(str(s or ""))
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "…"


def summarize_context(context: Any) -> Dict[str, str]:
    if not isinstance(context, dict):
        return {}
    summary: Dict[str, str] = {}
    for key in ("question", "hint", "rubric", "student_answer", "grade_result"):
        if key in context and context.get(key):
            summary[key] = short_text(context.get(key), 220)
    # その他のキーは先頭8件だけ収集（過剰ログ防止）
    for k, v in list(context.items())[:8]:
        ks = normalize_text(str(k))[:40]
        if not ks or ks in summary:
            continue
        if isinstance(v, (dict, list)):
            summary[ks] = short_text(json.dumps(v, ensure_ascii=False), 220)
        else:
            summary[ks] = short_text(v, 220)
    return summary

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
        if not EXPORT_CODE:
            return jsonify({"error": "Forbidden"}), 403
        provided = request.headers.get("X-Export-Code") or request.args.get("code")
        if not provided or not hmac.compare_digest(str(provided), str(EXPORT_CODE)):
            return jsonify({"error": "Forbidden"}), 403
        return fn(*args, **kwargs)
    return wrapper

def _is_teacher_request() -> bool:
    """
    ストックAPI等で「ヒントを返して良いか」を判定。
    EXPORT_CODE があればそれ、なければ ACCESS_CODE を教師コードとして扱う。
    """
    expected = EXPORT_CODE or ACCESS_CODE
    if not expected:
        return False
    provided = (
        request.headers.get("X-Export-Code")
        or request.headers.get("X-Access-Code")
        or request.args.get("access_code")
        or request.args.get("code")
        or request.cookies.get("access_code")
    )
    if not provided:
        return False
    return hmac.compare_digest(str(provided), str(expected))

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

    unit = normalize_text(p.get("unit", ""))[:60]
    genre_hint = normalize_text(p.get("genre_hint", ""))[:120]
    avoid = normalize_text(p.get("avoid_topics", ""))[:120]
    length_hint = normalize_text(p.get("length_hint", "30〜80字程度"))[:40]
    include_hints = bool(p.get("include_hints", False))
    return {
        "subject": subject,
        "category": category,
        "domain": category,
        "grade": grade,
        "difficulty": difficulty,
        "max_points": difficulty_max_score(difficulty),
        "unit": unit,
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
    assist_on = bool(p.get("assist_on", False))
    try:
        rewrite_count = int(p.get("rewrite_count", 0))
    except Exception:
        rewrite_count = 0
    rewrite_count = max(0, min(50, rewrite_count))
    selected_full_score = normalize_text(p.get("selected_full_score", difficulty))[:10]
    last10_scores = p.get("last10_scores", [])
    if not isinstance(last10_scores, list):
        last10_scores = []
    last10_scores = [int(x) for x in last10_scores[:10] if isinstance(x, (int, float, str)) and str(x).isdigit()]
    return {
        "question": question,
        "student_answer": student_answer,
        "model_answer": model_answer,
        "difficulty": difficulty,
        "max_points": difficulty_max_score(difficulty),
        "assist_on": assist_on,
        "rewrite_count": rewrite_count,
        "selected_full_score": selected_full_score,
        "last10_scores": last10_scores,
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
        client.models.retrieve(DEFAULT_MODEL)
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
    unit = payload.get("unit", "")
    genre_hint = payload["genre_hint"]
    avoid = payload["avoid_topics"]
    length_hint = payload["length_hint"]

    recent = recent_tags()
    if recent:
        extra_avoid = "、".join(recent[:8])
        avoid = f"{avoid}（直近の重複回避候補: {extra_avoid}）" if avoid else f"直近の重複回避候補: {extra_avoid}"
    recent_qs = recent_questions(limit=5, subject=subject, category=category, grade=grade, unit=unit)
    recent_keywords: List[str] = []
    for q in recent_qs:
        for kw in extract_keywords(q, limit=8):
            if kw not in recent_keywords:
                recent_keywords.append(kw)
        if len(recent_keywords) >= 10:
            break
    if recent_keywords:
        avoid = f"{avoid}（最近のキーワード: {'・'.join(recent_keywords[:10])}）" if avoid else f"最近のキーワード: {'・'.join(recent_keywords[:10])}"
    angle = pick_next_angle(recent_qs)

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
        f"難易度/配点: {difficulty_label(difficulty)}\n"
        f"単元（任意）: {unit or '特になし'}\n"
        f"出題の方向性（任意）: {genre_hint or '特になし'}\n"
        f"観点ローテ（直近回避）: {angle['label']}（{angle['hint']}）\n"
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
    max_score = difficulty_max_score(difficulty)
    label = difficulty_label(difficulty)

    sysprompt = (
        "あなたは中学生の記述解答を採点する試験官です。"
        "採点は0〜10点の整数。"
        "参考解答と同等の内容であれば10点にしてください（満点例）。"
        "さらに『入試本番で満点（○）になる確率』を0〜100の整数で推定し、"
        "短い根拠を perfect_probability_note に書いてください。"
        "確率は『満点になる確率』なので、score_totalが10でも100とは限りません（表現の曖昧さ・要点漏れの可能性を考慮）。"
        "出力は必ずJSONで、"
        '{"score_total":int,"good_points":[str,str],"next_step":str,'
        '"rubric":{"conclusion":int,"logic":int,"wording":int},'
        '"best_sentence":str,"short_comment":str,"rewrite_tip":str,'
        '"perfect_probability":int,"perfect_probability_note":str,'
        '"model_answer":str,"reasons":[str],'
        '"full_score_criteria":[str],"full_score_example":str,'
        '"weak_tags":[str],"next_steps":[str],"practice_menu":[str]}。'
        "rubric の各項目は0〜3の整数（結論の明確さ/理由の筋道/用語・表現の適切さ）。"
        "good_points は短い文を2つ、next_step は1つ。"
        "short_comment は1〜2行で簡潔に。"
        "best_sentence は受験者の解答から最も良い一文を抜粋。"
        "reasons は3〜6個、箇条書きの短文。"
        "full_score_criteria は満点に必要な要素を3つ程度、短く箇条書きで。"
        "full_score_example は満点になりやすい短い例文（参考解答と同等でも可）。"
        "next_step は「結論→理由→具体例1つ」で、30〜80字の1文にする。"
        "next_steps は1〜3個、達成条件付きの具体的な手順にする。"
        "practice_menu は1〜3個、すぐできる練習メニューにする。"
        "日本語で丁寧かつ簡潔に。"
    )
    usr = (
        f"問題: {q}\n"
        f"受験者の解答: {sa}\n"
        f"参考解答（任意）: {ma or '（なし）'}\n"
        f"配点設定: {difficulty}（表示上の満点: {label} / {max_score}点）\n"
        "評価観点：正確性/要点の網羅/論理性/表現の明瞭さ。"
    )
    return [{"role": "system", "content": sysprompt},
            {"role": "user", "content": usr}]


def build_ask_ai_messages(message: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    ctx = context if isinstance(context, dict) else {}
    ctx_lines: List[str] = []
    key_labels = {
        "question": "問題文",
        "hint": "ヒント",
        "rubric": "採点観点",
        "student_answer": "ユーザー解答",
        "grade_result": "採点結果",
    }
    for key, label in key_labels.items():
        val = short_text(ctx.get(key, ""), 800)
        if val:
            ctx_lines.append(f"- {label}: {val}")

    sysprompt = """あなたは「AI先生」です。中学生〜高校生に向けて、学校や塾の先生のように丁寧に解説します。
口調はやさしく、結論だけで終わらせず「なぜそうなるか」「どう書けば点が取れるか」を必ず説明してください。

【大原則】

* まずは生徒が自力で考えたことを尊重し、前向きに導く。
* 断定しすぎず、必要なら最初に確認質問を1つだけする（曖昧なときのみ）。
* ただの一般論ではなく、与えられた context（問題文/ヒント/採点観点/生徒解答/採点結果）を最優先で使う。
* 個人情報（氏名・学校名など）は入力しないよう、返答の冒頭に1行だけ注意する。

【出力フォーマット（必ずこの順で）】

1. まず要点（1〜2行で結論）
2. わかりやすい解説（3〜8行）：原因→理由→つながり
3. 具体例（1つ）：短い例文・式・言い換えなど
4. 書き方の型（テンプレ）：記述なら「結論→理由→根拠(具体例)」を基本に、字数内で書ける形にする
5. 次の一手（箇条書きで1〜3個）：今日すぐ直せる行動にする
6. 確認（短い質問を1つ）：理解チェック。「どこが一番引っかかった？」など

【禁止】

* 乱暴・極端に短い返答（2〜3行で終わらせない）
* 「とにかく頑張れ」だけで終わる
* contextを無視した一般論

【contextの扱い（必須）】

* /api/ask_ai の入力で context があれば、必ず文章内で参照して説明する（問題文に触れる、採点観点を使う等）
* context が空なら、最初に確認質問を1つだけしてから解説する"""
    user_prompt = f"質問: {message}\n"
    if ctx_lines:
        user_prompt += "\n参考文脈:\n" + "\n".join(ctx_lines)
    else:
        user_prompt += "\n参考文脈: （なし）\n※contextが空です。まず確認質問を1つだけしてから、解説を続けてください。"
    return [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": user_prompt},
    ]

def length_bucket(length: int) -> str:
    if length < 30:
        return "short"
    if length > 80:
        return "long"
    return "ideal"

def build_next_step_templates(length: int) -> Dict[str, Dict[str, Any]]:
    return {
        "length_short": {
            "category": "length",
            "text": "結論→理由→具体例1つで30〜80字に整える（例: 結論／理由／具体例を1文でつなぐ）",
            "steps": [
                "結論を文頭に1文で書く（〜である）。",
                "理由を「〜だから」で1つ足す。",
                "具体例は1つだけ入れて30〜80字に収める。",
            ],
            "practice": [
                "結論だけ10字以内で書く",
                "理由を1つだけ足して30字台にする",
                "具体例を1つ加えて50〜70字にする",
            ],
        },
        "length_long": {
            "category": "length",
            "text": "80字以内に収めるため、具体例は1つに絞り、重複表現を削る",
            "steps": [
                "具体例を1つに絞る。",
                "同じ意味の語を1か所削る。",
                "30〜80字に収まるか数える。",
            ],
            "practice": [
                "具体例を1つだけ残す",
                "主語を1つに絞る",
                "70字以内に要約する",
            ],
        },
        "logic": {
            "category": "logic",
            "text": "理由→結論が1文でつながる形にする（「〜だから、〜である」）",
            "steps": [
                "理由を1つに絞る。",
                "「〜だから、〜である」で1文にする。",
                "具体例は1つだけ添える。",
            ],
            "practice": [
                "理由を1語でメモする",
                "理由＋結論だけで1文を書く",
                "具体例を1つ足して完成させる",
            ],
        },
        "conclusion": {
            "category": "conclusion",
            "text": "文頭で結論を明示する（「〜である。」から始める）",
            "steps": [
                "最初の10字で結論を書く。",
                "理由を「〜から」でつなぐ。",
                "具体例を1つ入れて整える。",
            ],
            "practice": [
                "結論だけを先に書く",
                "理由を1つだけ追加する",
                "具体例を1つ足して30〜80字にする",
            ],
        },
        "wording": {
            "category": "wording",
            "text": "重要語句を1つ入れ、主語を明確にする",
            "steps": [
                "重要語句を1つ選ぶ。",
                "主語を入れて文頭に置く。",
                "理由と具体例を1つずつ足す。",
            ],
            "practice": [
                "重要語句を1語追加する",
                "主語を明確にする",
                "30〜80字で言い切る",
            ],
        },
        "example": {
            "category": "example",
            "text": "具体例を1つ入れる（地名/数字/出来事のどれか1つでOK）",
            "steps": [
                "具体例を1つ決める（地名/数字/出来事）。",
                "結論と理由にその例を結びつける。",
                "30〜80字に整える。",
            ],
            "practice": [
                "具体例だけを書き出す",
                "結論＋理由＋具体例で1文にする",
                "字数を70字前後に調整する",
            ],
        },
    }

def choose_next_step_category(length: int, rubric: Dict[str, int]) -> str:
    if length_bucket(length) == "short":
        return "length_short"
    if length_bucket(length) == "long":
        return "length_long"
    weakest = min(rubric.items(), key=lambda x: x[1])[0] if rubric else "logic"
    if weakest == "conclusion":
        return "conclusion"
    if weakest == "wording":
        return "wording"
    return "logic"

def build_next_step_pack(length: int, rubric: Dict[str, int], history: List[str]) -> Tuple[str, str, List[str], List[str]]:
    templates = build_next_step_templates(length)
    cat = choose_next_step_category(length, rubric)
    recent = history[-2:] if history else []
    if recent and all(h == cat for h in recent):
        for alt in ("example", "logic", "conclusion", "wording", "length_short", "length_long"):
            if alt in templates and alt != cat:
                cat = alt
                break
    tpl = templates.get(cat, templates["logic"])
    return cat, tpl["text"], tpl["steps"], tpl["practice"]

def build_full_score_checks(criteria: List[str], answer: str, rubric: Dict[str, int]) -> List[Dict[str, Any]]:
    answer_norm = normalize_for_match(answer)
    checks: List[Dict[str, Any]] = []
    for c in criteria:
        met = False
        if "結論" in c:
            met = rubric.get("conclusion", 0) >= 2
        elif "理由" in c or "因果" in c:
            met = rubric.get("logic", 0) >= 2
        elif "用語" in c or "語句" in c:
            met = rubric.get("wording", 0) >= 2
        else:
            keys = extract_keywords(c, limit=6)
            met = any(k in answer_norm for k in keys)
        checks.append({"text": c, "met": bool(met)})
    return checks

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
        "model": DEFAULT_MODEL,
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
        "MODEL_ID": DEFAULT_MODEL,
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
# フィードバック
# =========================
@app.post("/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    if not payload and request.form:
        payload = request.form.to_dict()

    f_type = normalize_text(payload.get("type", "")) or "feedback"
    role = normalize_text(payload.get("role", "")) or "unknown"
    email = normalize_text(payload.get("email", ""))[:200]
    message = normalize_text(payload.get("message", ""))
    context = payload.get("context", None)
    if not isinstance(context, (dict, list, str)) and context is not None:
        context = str(context)

    continue_score = payload.get("continue_score", None)
    if continue_score is not None:
        try:
            continue_score = int(continue_score)
        except Exception:
            continue_score = None

    if f_type == "continue_score":
        if continue_score is None or continue_score < 0 or continue_score > 10:
            return jsonify({"ok": False, "error": "continue_score must be 0-10"}), 400
    else:
        if not (200 <= len(message) <= 800):
            return jsonify({"ok": False, "error": "message length must be 200-800 chars"}), 400

    stamp = datetime.now().astimezone()
    record = {
        "timestamp": stamp.isoformat(),
        "type": f_type,
        "role": role,
        "email": email,
        "message": message,
        "continue_score": continue_score,
        "context": context,
        "user_agent": request.headers.get("User-Agent", ""),
    }

    month_key = stamp.strftime("%Y%m")
    path = DATA / f"feedback_{month_key}.jsonl"
    append_jsonl(path, record)

    return jsonify({"ok": True}), 200

# =========================
# ストック API
# =========================
@app.get("/api/stock/list")
def stock_list():
    subject = request.args.get("subject", "")
    category = request.args.get("category", "")
    domain = request.args.get("domain", "") or category
    grade = request.args.get("grade", "")
    unit = request.args.get("unit", "")
    difficulty = request.args.get("difficulty", "")
    max_points = request.args.get("max_points", "")
    order = request.args.get("order", "new")
    try:
        max_points_int = int(max_points) if max_points else None
    except Exception:
        max_points_int = None
    try:
        limit = max(1, min(200, int(request.args.get("limit", str(MAX_STOCK)))))
    except Exception:
        limit = MAX_STOCK
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except Exception:
        offset = 0

    def match(it: Dict[str, Any]) -> bool:
        if subject and it.get("subject") != subject:
            return False
        if domain and it.get("domain") != domain:
            return False
        if grade and it.get("grade") != grade:
            return False
        if unit and it.get("unit") != unit:
            return False
        if difficulty and it.get("difficulty") != difficulty:
            return False
        if max_points_int is not None and int(it.get("max_points") or 0) != max_points_int:
            return False
        return True

    items = [enrich_stock_item(it) for it in current_stocks() if match(it)]
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=(order != "old"))
    total = len(items)
    items = items[offset: offset + limit]
    # ✅ 生徒にヒントを見せない（講師コードが無ければヒントを落とす）
    if not _is_teacher_request():
        items = [_strip_hints(dict(it)) for it in items]
    return jsonify({"ok": True, "items": items, "count": len(items), "limit": limit, "total": total, "offset": offset}), 200

@app.get("/api/stock/get")
def stock_get():
    item_id = normalize_text(request.args.get("id", ""))[:64]
    if not item_id:
        return jsonify({"ok": False, "error": "id is required"}), 400
    path = STOCK_DIR / f"{item_id}.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "not_found"}), 404
    item = load_stock_item(path) or {}
    item = enrich_stock_item(item)
    if not _is_teacher_request():
        item = _strip_hints(dict(item))
    return jsonify({"ok": True, "item": item}), 200

@app.get("/api/stock/search")
def stock_search():
    q = normalize_text(request.args.get("q", ""))
    subject = request.args.get("subject", "")
    category = request.args.get("category", "")
    domain = request.args.get("domain", "") or category
    grade = request.args.get("grade", "")
    unit = request.args.get("unit", "")
    difficulty = request.args.get("difficulty", "")
    max_points = request.args.get("max_points", "")
    source = request.args.get("source", "")
    order = request.args.get("order", "new")
    try:
        max_points_int = int(max_points) if max_points else None
    except Exception:
        max_points_int = None
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
        if domain and it.get("domain") != domain:
            return False
        if grade and it.get("grade") != grade:
            return False
        if unit and it.get("unit") != unit:
            return False
        if difficulty and it.get("difficulty") != difficulty:
            return False
        if max_points_int is not None and int(it.get("max_points") or 0) != max_points_int:
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

    items = [enrich_stock_item(it) for it in current_stocks() if match(it)]
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=(order != "old"))
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
    domain = request.args.get("domain", "") or category
    grade = request.args.get("grade", "")
    unit = request.args.get("unit", "")
    difficulty = request.args.get("difficulty", "")
    max_points = request.args.get("max_points", "")
    source = request.args.get("source", "")
    exclude_ids = set([s for s in (request.args.get("exclude_ids", "") or "").split(",") if s])
    try:
        max_points_int = int(max_points) if max_points else None
    except Exception:
        max_points_int = None

    def match(it: Dict[str, Any]) -> bool:
        if it.get("id") in exclude_ids:
            return False
        if subject and it.get("subject") != subject:
            return False
        if domain and it.get("domain") != domain:
            return False
        if grade and it.get("grade") != grade:
            return False
        if unit and it.get("unit") != unit:
            return False
        if difficulty and it.get("difficulty") != difficulty:
            return False
        if max_points_int is not None and int(it.get("max_points") or 0) != max_points_int:
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

    candidates = [enrich_stock_item(it) for it in current_stocks() if match(it)]
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

    subject = data.get("subject", "")[:20]
    if subject not in ALLOWED_SUBJECTS:
        subject = ""
    category = data.get("category", "")[:20]
    if subject and category and category not in ALLOWED_CAT_BY_SUBJECT[subject]:
        category = ""
    domain = data.get("domain", "")[:20] or category
    grade = data.get("grade", "")[:10]
    if grade and grade not in ALLOWED_GRADES:
        grade = ""
    unit = normalize_text(data.get("unit", ""))[:60]

    dup = find_duplicate_in_stocks(question, subject=subject, category=category, grade=grade)
    if dup:
        return jsonify({"ok": True, "duplicate_of": dup.get("id"), "item": dup}), 200

    item = {
        "id": uuid.uuid4().hex,
        "question": question,
        "question_text": question,
        "subject": subject,
        "category": category,
        "domain": domain,
        "grade": grade,
        "unit": unit,
        "difficulty": (data.get("difficulty", "10点") if data.get("difficulty", "10点") in ALLOWED_DIFFICULTY else "10点"),
        "max_points": difficulty_max_score(data.get("difficulty", "10点")),
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

        subject = raw.get("subject", "")
        category = raw.get("category", "")
        domain = raw.get("domain", "") or category
        grade = raw.get("grade", "")
        unit = normalize_text(raw.get("unit", ""))[:60]

        if subject not in ALLOWED_SUBJECTS:
            subject = ""
        if subject and category and category not in ALLOWED_CAT_BY_SUBJECT[subject]:
            category = ""
        if grade and grade not in ALLOWED_GRADES:
            grade = ""

        dup_item = find_duplicate_in_stocks(q, subject=subject, category=category, grade=grade)
        if dup_item and skip_dup and not overwrite:
            skipped += 1
            continue

        item = {
            "id": raw.get("id") or uuid.uuid4().hex,
            "question": q,
            "question_text": q,
            "subject": subject,
            "category": category,
            "domain": domain,
            "grade": grade,
            "unit": unit,
            "difficulty": (raw.get("difficulty", "10点") if raw.get("difficulty", "10点") in ALLOWED_DIFFICULTY else "10点"),
            "max_points": int(raw.get("max_points") or difficulty_max_score(raw.get("difficulty", "10点"))),
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
    recent_qs = recent_questions(
        limit=10,
        subject=payload["subject"],
        category=payload["category"],
        grade=payload["grade"],
        unit=payload.get("unit"),
    )
    question = ""
    model_answer = ""
    explanation = ""
    intention = ""
    tags: List[str] = []
    obj: Dict[str, Any] = {}

    for attempt in range(2):
        messages = build_generation_messages(payload)
        try:
            # 用途: 問題生成（問題文・模範解答・解説・出題意図の生成）
            rsp = client.chat.completions.create(
                model=DEFAULT_MODEL,
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
            continue

        similar_by_text = any(is_similar(question, prev_q) for prev_q in recent_qs)
        similar_by_keywords = any(keyword_overlap_score(question, prev_q) >= 0.4 for prev_q in recent_qs[:5])
        if similar_by_text or similar_by_keywords:
            if attempt == 0:
                overlap_note = "直近の問題と類似するテーマ・キーワード"
                payload["avoid_topics"] = normalize_text(
                    f"{payload.get('avoid_topics', '')}／{overlap_note}"
                )[:120]
                continue
        break

    if not question:
        _update_metrics("generate", False, t0)
        return jsonify({"ok": False, "error": "generation failed"}), 502

    dup = find_duplicate_in_stocks(
        question,
        subject=payload["subject"],
        category=payload["category"],
        grade=payload["grade"],
    )
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
        return jsonify({"ok": True, "duplicate_of": dup.get("id"), "item": dup, "generated": generated, "model": DEFAULT_MODEL}), 200

    item = {
        "id": uuid.uuid4().hex,
        "source": "auto",
        "question": question,
        "question_text": question,
        "model_answer": model_answer,
        "explanation": explanation,
        "intention": intention,
        "tags": tags,
        "subject": payload["subject"],
        "category": payload["category"],
        "domain": payload["category"],
        "grade": payload["grade"],
        "unit": payload.get("unit", ""),
        "difficulty": payload["difficulty"],
        "max_points": payload["max_points"],
        "created_at": now_ms(),
    }
    saved = save_stock_item(item)
    append_jsonl(LOG_DIR / "generation.jsonl", {"event": "generated", "item": saved, "model": DEFAULT_MODEL, "ts": now_ms()})

    resp_item = dict(saved)
    if not payload["include_hints"]:
        for k in ("model_answer", "explanation", "intention"):
            resp_item.pop(k, None)

    _update_metrics("generate", True, t0)
    return jsonify({"ok": True, "item": resp_item, "model": DEFAULT_MODEL}), 200

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
        # 用途: 採点（採点・講評・改善提案）
        rsp = client.chat.completions.create(
            model=DEFAULT_MODEL,
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
        score_total_raw = int(result.get("score_total", result.get("score", 0)))
    except Exception:
        score_total_raw = 0
    score_total_raw = max(0, min(10, score_total_raw))

    good_points = result.get("good_points", [])
    if not isinstance(good_points, list):
        good_points = [str(good_points)]
    good_points = [normalize_text(str(x))[:120] for x in good_points if str(x).strip()]
    while len(good_points) < 2:
        good_points.append("問いに沿った要素が書けています")
    good_points = good_points[:2]

    rubric_raw = result.get("rubric", {}) if isinstance(result.get("rubric", {}), dict) else {}
    def _rubric_score(key: str) -> int:
        try:
            val = int(rubric_raw.get(key, 0))
        except Exception:
            val = 0
        return max(0, min(3, val))
    rubric = {
        "conclusion": _rubric_score("conclusion"),
        "logic": _rubric_score("logic"),
        "wording": _rubric_score("wording"),
    }

    answer_length = len(data["student_answer"])
    history = session.get("next_step_history") or []
    category, next_step_text, next_steps_fallback, practice_fallback = build_next_step_pack(
        answer_length, rubric, history
    )

    next_step_raw = normalize_text(result.get("next_step", ""))[:160]
    generic_hits = ("具体的", "詳しく", "詳細", "もっと", "より")
    next_step = next_step_raw if next_step_raw and not any(k in next_step_raw for k in generic_hits) else next_step_text

    next_steps = result.get("next_steps", [])
    if not isinstance(next_steps, list):
        next_steps = []
    next_steps = [normalize_text(str(x))[:120] for x in next_steps if str(x).strip()]
    if not next_steps:
        next_steps = next_steps_fallback

    practice_menu = result.get("practice_menu", [])
    if not isinstance(practice_menu, list):
        practice_menu = []
    practice_menu = [normalize_text(str(x))[:120] for x in practice_menu if str(x).strip()]
    if not practice_menu:
        practice_menu = practice_fallback

    weak_tags = result.get("weak_tags", [])
    if not isinstance(weak_tags, list):
        weak_tags = []
    weak_tags = [normalize_text(str(x))[:20] for x in weak_tags if str(x).strip()]

    best_sentence = (result.get("best_sentence", "") or "").strip()
    short_comment = (result.get("short_comment", "") or "").strip()
    rewrite_tip = (result.get("rewrite_tip", "") or "").strip()

    max_score = difficulty_max_score(data["difficulty"])
    score_label = f"{max_score}点"

    model_ans = (data.get("model_answer", "") or result.get("model_answer", "") or "").strip()
    full_score_example = (model_ans or result.get("full_score_example", "") or "").strip()

    if model_ans and is_high_match(data["student_answer"], model_ans, threshold=0.98):
        score_total_raw = 10

    score_total_scaled = int(round(score_total_raw * max_score / 10))
    score_total_scaled = max(0, min(max_score, score_total_scaled))

    commentary_raw = (result.get("commentary", "") or short_comment).strip()
    head = f"{score_label}中{score_total_scaled}点"
    if not commentary_raw.startswith(head):
        commentary = f"{head}：{commentary_raw}" if commentary_raw else f"{head}。"
    else:
        commentary = commentary_raw

    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [normalize_text(str(x))[:200] for x in reasons][:8]

    criteria = result.get("full_score_criteria", [])
    if not isinstance(criteria, list):
        criteria = [str(criteria)]
    criteria = [normalize_text(str(x))[:120] for x in criteria if str(x).strip()]
    if not criteria:
        criteria = ["結論が明確", "理由が1つ以上ある", "重要語句が入っている"]

    full_score_checks = build_full_score_checks(criteria, data["student_answer"], rubric)

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
        prob = int(round(score_total_raw * 10))
        if not prob_note:
            prob_note = "スコアからの簡易推定"

    prob = max(0, min(100, int(prob)))

    append_jsonl(LOG_DIR / "grading.jsonl", {
        "event": "graded",
        "question": data["question"],
        "student_answer": data["student_answer"],
        "score": score_total_scaled,
        "perfect_probability": prob,
        "assist_on": data.get("assist_on", False),
        "rewrite_count": data.get("rewrite_count", 0),
        "selected_full_score": data.get("selected_full_score", data["difficulty"]),
        "last10_scores": data.get("last10_scores", []),
        "answer_length": answer_length,
        "model": DEFAULT_MODEL,
        "ts": now_ms()
    })

    prev = session.get("last_result") or {}
    improvements: List[str] = []
    rubric_diff = {}
    prev_score_total = None
    if prev and prev.get("question") == data["question"]:
        prev_score_total = int(prev.get("score_total_raw", prev.get("score_total", 0)))
        prev_rubric = prev.get("rubric", {}) if isinstance(prev.get("rubric", {}), dict) else {}
        labels = {
            "conclusion": "結論の明確さ",
            "logic": "理由の筋道",
            "wording": "用語・表現の適切さ",
        }
        for key, label in labels.items():
            prev_val = int(prev_rubric.get(key, 0))
            diff = rubric.get(key, 0) - prev_val
            rubric_diff[key] = diff
            if diff > 0:
                improvements.append(f"{label}が前回より良くなりました")
        if score_total_raw > prev_score_total and len(improvements) < 2:
            improvements.append("総合点が前回より上がりました")
        improvements = improvements[:2]
    else:
        rubric_diff = {k: 0 for k in rubric.keys()}

    history = (session.get("next_step_history") or []) + [category]
    session["next_step_history"] = history[-5:]

    session["last_result"] = {
        "question": data["question"],
        "answer": data["student_answer"],
        "score_total": score_total_scaled,
        "score_total_raw": score_total_raw,
        "max_score": max_score,
        "rubric": rubric,
        "good_points": good_points,
        "best_sentence": best_sentence,
        "short_comment": short_comment,
        "ts": now_ms(),
    }

    _update_metrics("grade", True, t0)
    return jsonify({
        "ok": True,
        "score": score_total_scaled,
        "score_total": score_total_scaled,
        "score_total_raw": score_total_raw,
        "commentary": commentary,
        "short_comment": short_comment,
        "good_points": good_points,
        "next_step": next_step,
        "next_step_category": category,
        "next_steps": next_steps,
        "practice_menu": practice_menu,
        "weak_tags": weak_tags,
        "rubric": rubric,
        "best_sentence": best_sentence,
        "rewrite_tip": rewrite_tip,
        "improvements": improvements,
        "rubric_diff": rubric_diff,
        "previous_score_total": prev_score_total,
        "model_answer": model_ans,
        "reasons": reasons,
        "full_score_example": full_score_example,
        "full_score_criteria": criteria,
        "full_score_checks": full_score_checks,
        "perfect_probability": prob,
        "perfect_probability_note": prob_note,
        "max_score": max_score,
        "score_label": score_label,
        "answer_length": answer_length,
        "length_bucket": length_bucket(answer_length),
        "assist_on": data.get("assist_on", False),
        "rewrite_count": data.get("rewrite_count", 0),
        "selected_full_score": data.get("selected_full_score", data["difficulty"]),
        "last10_scores": data.get("last10_scores", []),
        "model": DEFAULT_MODEL,
    }), 200


# =========================
# AI先生チャット API
# =========================
@app.post("/api/ask_ai")
@require_access_code
@rate_limit
def ask_ai():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_content_type", "detail": "Content-Type must be application/json"}), 400

    payload = request.get_json(force=True, silent=True) or {}
    message = normalize_text(payload.get("message", ""))
    context = payload.get("context", {}) if isinstance(payload.get("context", {}), dict) else {}

    if not message:
        return jsonify({"ok": False, "error": "invalid_request", "detail": "message は必須です"}), 400

    err = ensure_openai()
    if err:
        return jsonify({"ok": False, "error": "openai_unavailable", "detail": err}), 500

    log_record = {
        "timestamp": now_ms(),
        "ok": False,
        "message": short_text(message, 400),
        "context_summary": summarize_context(context),
        "model": DEFAULT_MODEL,
    }

    try:
        # 用途: AI先生チャット（文脈重視の丁寧な解説）
        rsp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=build_ask_ai_messages(message, context),
            temperature=0.3,
            max_tokens=MAX_TOKENS_ASK_AI,
        )
        answer = normalize_text((rsp.choices[0].message.content or "").strip())
        if not answer:
            log_record["error"] = "empty_answer"
            append_jsonl(LOG_DIR / "ai_chat.jsonl", log_record)
            return jsonify({"ok": False, "error": "empty_answer", "detail": "AIの応答が空でした"}), 500

        log_record["ok"] = True
        log_record["answer"] = short_text(answer, 500)
        append_jsonl(LOG_DIR / "ai_chat.jsonl", log_record)
        return jsonify({"ok": True, "answer": answer, "model": DEFAULT_MODEL}), 200
    except Exception as e:
        log_record["error"] = f"OpenAI API error: {e}"
        append_jsonl(LOG_DIR / "ai_chat.jsonl", log_record)
        return jsonify({"ok": False, "error": "openai_api_error", "detail": str(e)}), 500

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
        "id", "subject", "domain", "category", "grade", "unit",
        "difficulty", "max_points", "question", "question_text",
        "model_answer", "explanation", "intention",
        "tags", "source", "created_at_iso"
    ]
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for it in items:
            writer.writerow({
                "id": it.get("id", ""),
                "subject": it.get("subject", ""),
                "domain": it.get("domain", ""),
                "category": it.get("category", ""),
                "grade": it.get("grade", ""),
                "unit": it.get("unit", ""),
                "difficulty": it.get("difficulty", ""),
                "max_points": it.get("max_points", ""),
                "question": it.get("question", ""),
                "question_text": it.get("question_text", ""),
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
    return {"model": DEFAULT_MODEL, "available_models": [DEFAULT_MODEL]}, 200

@app.post("/api/model")
@require_access_code
def set_model():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    body = request.get_json(force=True, silent=True) or {}
    requested_model = normalize_text(body.get("model", ""))
    if not requested_model:
        return jsonify({"ok": False, "error": "model is required", "model": DEFAULT_MODEL}), 400
    # モデル切替UI/APIが来ても常に gpt-4o-mini に正規化する
    return jsonify({"ok": True, "requested_model": requested_model, "model": DEFAULT_MODEL}), 200

# ---- エントリポイント ----
if __name__ == "__main__":
    # ✅ ローカルは 5001 をデフォルト（Renderは PORT が入るので影響なし）
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=env_bool("DEBUG"))
