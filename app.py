from flask import Flask, render_template, request, redirect, url_for, session
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import json
from datetime import datetime
from difflib import SequenceMatcher
import time
import tempfile
import re

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-for-beta")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STOCK_FILE = "generated_questions.json"
SEEN_FILE = "seen_questions.json"  # 直近出題の記憶
TERM_CUSTOM_FILE = "terms_custom.json"  # 任意の用語上書き

# 類似度チェックとリトライ
SIM_THRESHOLD = 0.72
NOVELTY_MAX_RETRY = 4
STOCK_MAX = 10  # ★ ストック上限

# ---------- 手動出題プリセット ----------
# 理科（既存）
question_bank_science = {
    "中1": {
        "生物": ["植物の分類と特徴について説明しなさい。"],
        "化学": ["水に溶ける物質と溶けない物質の違いについて述べよ。"],
        "物理": ["光の反射と屈折の違いを説明しなさい。"],
        "地学": ["火山と地震の関係について述べよ。"]
    },
    "中2": {
        "生物": ["人の体のしくみについて説明しなさい。"],
        "化学": ["化学変化と物理変化の違いを説明しなさい。"],
        "物理": ["電流と電圧の関係について述べよ。"],
        "地学": ["地層と化石の関係について説明しなさい。"]
    },
    "中3": {
        "生物": ["遺伝の仕組みについて説明しなさい。"],
        "化学": ["酸とアルカリの性質について述べよ。"],
        "物理": ["力のつり合いと運動の関係について説明しなさい。"],
        "地学": ["天気の変化の仕組みについて述べよ。"],
        "天体": ["太陽と月の動きの違いを説明しなさい。"]
    }
}

# 社会（帝国書院ベース・代表的単元を拡充）
soc_geo_units = [
    "地球儀と世界の姿", "世界の気候と環境", "世界の人々の生活と文化",
    "アジアの地域", "ヨーロッパの地域", "アフリカの地域",
    "北アメリカの地域", "南アメリカの地域", "オセアニアの地域",
    "日本の国土と位置", "日本の地形と気候", "日本の人口・都市",
    "日本の農業", "日本の工業", "日本の資源・エネルギー",
    "日本の情報・交通", "防災と減災", "環境問題と持続可能な社会",
    "日本の諸地域（北海道）", "日本の諸地域（東北）", "日本の諸地域（関東）",
    "日本の諸地域（中部）", "日本の諸地域（近畿）",
    "日本の諸地域（中国・四国）", "日本の諸地域（九州・沖縄）"
]

soc_hist_units = [
    "旧石器・縄文", "弥生", "古墳", "飛鳥", "奈良", "平安",
    "鎌倉", "室町", "安土桃山",
    "江戸前期（幕藩体制の確立）", "江戸中期（社会・文化）", "江戸後期（改革と列強の接近）",
    "開国と幕末", "明治維新と近代国家の成立",
    "日清・日露戦争と帝国日本", "大正デモクラシー",
    "昭和前期（戦争への道）", "第二次世界大戦",
    "戦後の改革と民主化", "高度経済成長と変化", "現代の日本と国際社会"
]

soc_civ_units = [
    "現代社会の特色と課題", "人権と日本国憲法", "基本的人権の保障",
    "国会・内閣・裁判所（三権分立）", "選挙と政治参加", "地方自治",
    "市場経済の仕組み", "金融と日本銀行", "労働・社会保障・福祉",
    "税金と財政", "国際社会と平和", "SDGsと私たちの暮らし"
]

# 社会の手動出題プリセット（各単元1文例）
def build_soc_question_bank():
    d = {"地理": {}, "歴史": {}, "公民": {}}
    for u in soc_geo_units:
        d["地理"][u] = [f"{u}の要点を30〜80字で説明しなさい。"]
    for u in soc_hist_units:
        d["歴史"][u] = [f"{u}の時代（事項）の特色を、因果関係を意識して30〜80字で説明しなさい。"]
    for u in soc_civ_units:
        d["公民"][u] = [f"{u}について、具体例や用語を用いて30〜80字で説明しなさい。"]
    return d

question_bank_social = build_soc_question_bank()

# ---------- サブトピック（新規性のヒント） ----------
subtopic_pool = {
    # 地理（例）
    "社会:地理:日本の農業": ["稲作の地域差", "施設園芸", "酪農地帯", "輸入自由化と食料自給率"],
    "社会:地理:日本の工業": ["臨海工業地帯", "中京工業地帯", "自動車産業の集積", "海外生産と空洞化"],
    "社会:地理:世界の気候と環境": ["気候帯とバイオーム", "サバンナ農牧", "乾燥地域の水問題", "熱帯雨林の減少"],
    "社会:地理:日本の情報・交通": ["新幹線網", "高速道路網", "物流の効率化", "ICTと地域格差"],
    # 歴史（例）
    "社会:歴史:弥生": ["稲作の広がり", "金属器の使用", "ムラからクニへ", "邪馬台国"],
    "社会:歴史:江戸後期（改革と列強の接近）": ["天保の改革", "異国船打払令と通商", "開国要求", "蘭学と洋学の受容"],
    "社会:歴史:明治維新と近代国家の成立": ["廃藩置県", "地租改正", "殖産興業", "帝国議会と憲法"],
    "社会:歴史:戦後の改革と民主化": ["農地改革", "新憲法と三権分立", "教育改革", "GHQの占領政策"],
    # 公民（例）
    "社会:公民:人権と日本国憲法": ["基本的人権の尊重", "法の下の平等", "表現の自由と制限", "新しい人権"],
    "社会:公民:市場経済の仕組み": ["需要供給と価格", "独占と寡占", "公共財と外部性", "物価安定と金融政策"],
}

# ---------- 用語説明モード：用語プール ----------
term_pool_social = {
    "地理": [
        "気候帯", "季節風", "砂漠化", "熱帯雨林", "扇状地", "三角州",
        "人口ピラミッド", "工業団地", "フードマイレージ", "食料自給率"
    ],
    "歴史": [
        "公地公民", "班田収授法", "大政奉還", "地租改正", "廃藩置県",
        "五箇条の御誓文", "富国強兵", "殖産興業", "治安維持法", "高度経済成長"
    ],
    "公民": [
        "国民主権", "三権分立", "基本的人権", "地方自治", "需要と供給",
        "独占禁止法", "消費者主権", "社会保障", "累進課税", "SDGs"
    ],
}

term_pool_science = {
    "生物": ["細胞", "蒸散", "光合成", "呼吸", "食物連鎖", "適応"],
    "化学": ["密度", "溶解度", "状態変化", "酸とアルカリ", "中和", "化学反応式"],
    "物理": ["反射の法則", "屈折", "オームの法則", "電力量", "仕事", "力の分解"],
    "地学": ["プレートテクトニクス", "風化と侵食", "地層", "地震の震源と震央", "火山"],
    "天体": ["日周運動", "年周運動", "月の満ち欠け", "金星の満ち欠け", "太陽高度"]
}

def load_custom_terms():
    """terms_custom.json があれば、既定プールにマージ（上書き優先）"""
    if not os.path.exists(TERM_CUSTOM_FILE):
        return
    try:
        with open(TERM_CUSTOM_FILE, "r", encoding="utf-8") as f:
            custom = json.load(f)
        # 社会
        for fld, lst in (custom.get("社会") or {}).items():
            if isinstance(lst, list) and lst:
                term_pool_social[fld] = lst
        # 理科
        for gen, lst in (custom.get("理科") or {}).items():
            if isinstance(lst, list) and lst:
                term_pool_science[gen] = lst
    except Exception as e:
        print(f"[WARN] load_custom_terms: {e}")

load_custom_terms()

def get_term_list(sel):
    """現在の選択に応じて用語リストを返す"""
    if sel["subject"] == "社会":
        return term_pool_social.get(sel["field"], [])
    else:
        return term_pool_science.get(sel["genre"], [])

def build_glossary_question(term: str, subject: str, field_or_genre: str) -> str:
    """用語説明の出題文を生成（自給自足・短文指示）。"""
    header = f"教科: {subject} / {'分野' if subject=='社会' else '領域'}: {field_or_genre}"
    return (
        f"{header}\n"
        f"次の用語を専門用語を用いて30〜60字で説明しなさい。必要なら具体例を1つ挙げること。\n"
        f"用語：{term}"
    )

# ---------- 汎用ファイルIO（原子的保存） ----------
def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path, data):
    d = json.dumps(data, ensure_ascii=False, indent=2)
    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dir_, delete=False) as tmp:
        tmp.write(d)
        tmp_path = tmp.name
    os.replace(tmp_path, path)  # 原子的に置換

def load_generated_questions():
    return load_json_file(STOCK_FILE, [])

def save_generated_questions_all(data):
    # ★ 上限をかけて保存（新しいもの優先で末尾に溜まる想定なので最後の10件を保持）
    if len(data) > STOCK_MAX:
        data = data[-STOCK_MAX:]
    save_json_file(STOCK_FILE, data)

def save_generated_question(question, subject, detail):
    data = load_generated_questions()
    data.append({
        "question": question,
        "subject": subject,
        "detail": detail,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    # ★ 保存時にトリム
    save_generated_questions_all(data)

def load_seen():
    return load_json_file(SEEN_FILE, {})

def save_seen(seen):
    trimmed = {k: (v[-50:] if isinstance(v, list) else []) for k, v in seen.items()}
    save_json_file(SEEN_FILE, trimmed)

def seen_key(sel):
    if sel["subject"] == "社会":
        return f"社会:{sel['field']}:{sel['era']}"
    else:
        return f"理科:{sel['genre']}"

def add_seen(sel, question_text):
    seen = load_seen()
    key = seen_key(sel)
    arr = seen.get(key, [])
    arr.append(question_text)
    seen[key] = arr
    save_seen(seen)

def get_recent_seen(sel, n=12):
    seen = load_seen()
    key = seen_key(sel)
    arr = seen.get(key, [])
    return arr[-n:]

# ---------- 選択状態（セッション保持） ----------
DEFAULT_SELECTION = {
    "mode": "manual",
    "subject": "社会",  # 社会 or 理科
    "field": "地理",     # 社会の分野：地理/歴史/公民
    "era": "日本の国土と位置",  # 社会の単元（任意文字列OK）
    "grade": "中1",     # 理科
    "genre": "生物",    # 理科
    "qstyle": "auto"    # ★ 出題タイプ（gpt用）：auto/materials/glossary/exam/school
}

def selection_from_form(form):
    subject = form.get("subject") or DEFAULT_SELECTION["subject"]
    mode = form.get("mode") or DEFAULT_SELECTION["mode"]
    qstyle = form.get("qstyle") or DEFAULT_SELECTION["qstyle"]

    if subject == "社会":
        field = form.get("field") or DEFAULT_SELECTION["field"]
        if field == "地理":
            era = form.get("era") or "日本の国土と位置"
        elif field == "歴史":
            era = form.get("era") or "旧石器・縄文"
        else:
            era = form.get("era") or "現代社会の特色と課題"
        grade = DEFAULT_SELECTION["grade"]
        genre = DEFAULT_SELECTION["genre"]
    else:
        grade = form.get("grade") or DEFAULT_SELECTION["grade"]
        genre = form.get("genre") or DEFAULT_SELECTION["genre"]
        field = DEFAULT_SELECTION["field"]  # ダミー
        era = DEFAULT_SELECTION["era"]      # ダミー

    return {
        "mode": mode, "subject": subject, "field": field,
        "era": era, "grade": grade, "genre": genre, "qstyle": qstyle
    }

def get_selection():
    sel = session.get("selection")
    if not sel:
        sel = DEFAULT_SELECTION.copy()
        session["selection"] = sel
    return sel

def set_selection(sel):
    session["selection"] = sel

# ---------- OpenAI呼び出し：簡易リトライ ----------
def _call_openai(func, max_retry=3, base_wait=0.8):
    last_err = None
    for i in range(max_retry):
        try:
            return func()
        except Exception as e:
            last_err = e
            if i < max_retry - 1:
                time.sleep(base_wait * (2 ** i))
    raise last_err

# ---------- プロンプトビルダ：社会（資料読解固定=従来） ----------
def build_concrete_prompt_social(sel):
    # サブトピックキー
    pool_key = f"社会:{sel['field']}:{sel['era']}"
    pool = subtopic_pool.get(pool_key, [])
    recent = get_recent_seen(sel, n=12)

    def not_recent(t): return all((t not in q) for q in recent)
    candidates = [t for t in pool if not_recent(t)] or pool
    novelty_hint = random.choice(candidates) if candidates else ""

    avoid_keywords = [q[:50] for q in recent][:8]
    avoid_str = "、".join(avoid_keywords)

    header = f"教科: 社会 / 分野: {sel['field']} / 単元: {sel['era']}"

    if sel["field"] == "地理":
        example = "資料A（農産物と生産量のミニ表）・資料B（気候帯の分布を箇条書き分類）"
        ask = "資料A・Bから読み取れる特徴を1つ挙げ，その理由を地形・気候・産業の観点から30〜80字で簡潔に説明しなさい。"
    elif sel["field"] == "歴史":
        example = "資料A（年別の年貢量ミニ表）・資料B（農具の普及の簡易表）"
        ask = "資料A・Bをもとに，当時の生活や生産の特色の変化を原因と結果の関係で30〜80字で説明しなさい。"
    else:
        example = "資料A（食品表示の要素のミニ表）・資料B（消費者相談件数の簡易データ）"
        ask = "資料A・Bをふまえ，食品表示が消費者にもたらす利点を30〜80字で説明しなさい。"

    lines = [
        "あなたは中学校の出題者です。以下の厳密条件を満たす、資料読解型の記述式問題を1問だけ日本語で作成してください。",
        header,
        "【厳密条件】",
        "- 問題文だけで完結させること。外部図表や画像の参照を禁止する。",
        "- 『資料A:』『資料B:』を行頭に付け、各3〜6行のミニ表/箇条書きを提示する（例：品目：数値（単位））。",
        "- 少なくとも片方の資料に具体的な数値（整数や%など）を含めること。",
        "- 設問は『〜しなさい。』で終える。回答目安は30〜80字で、一意に答えが定まる内容にする。",
        f"- 出題例：{example}",
        f"- 設問の指示：{ask}",
    ]
    if novelty_hint:
        lines.append(f"- サブトピック焦点：『{novelty_hint}』を中心に出題する。")
    if avoid_str:
        lines.append(f"- 次の語句や観点は含めないこと：{avoid_str}")
    return "\n".join(lines)

# ---------- プロンプトビルダ：理科（スタイル切替対応） ----------
def build_prompt_science(sel, style: str):
    """
    style:
      - materials: 資料読解（従来）
      - glossary: 用語説明（重要語）
      - exam: 入試レベル記述（宮城県公立高校テイスト）
      - school: 定期テスト記述（教科書準拠）
    """
    header = f"教科: 理科 / 学年: {sel['grade']} / 領域: {sel['genre']}"

    if style == "materials":
        # 従来の資料読解（自給自足データ必須）
        example = "資料A（実験条件のミニ表）・資料B（測定結果のミニ表）"
        ask = "資料A・Bから分かる関係を説明し，起こる現象の理由を専門用語で30〜80字で述べなさい。"
        return "\n".join([
            "あなたは中学校の理科の出題者です。資料読解型の記述式問題を1問だけ作成してください。",
            header,
            "【厳密条件】",
            "- 問題文だけで完結させること。外部図表や画像の参照を禁止する。",
            "- 『資料A:』『資料B:』を行頭に付け、各3〜6行のミニ表/箇条書きを提示する（例：条件：値（単位））。",
            "- 少なくとも片方の資料に具体的な数値（整数や%やmmなど）を含めること。",
            "- 設問は『〜しなさい。』で終える。回答目安は30〜80字。",
            f"- 出題例：{example}",
            f"- 設問の指示：{ask}",
        ])

    if style == "glossary":
        # 用語説明：用語はプールから選ぶ（後段で実際の用語を差し込んでもOKだが、ここでは生成側に任せる）
        term_candidates = get_term_list(sel)
        term_hint = random.choice(term_candidates) if term_candidates else sel['genre']
        return "\n".join([
            "あなたは中学校の理科の出題者です。重要用語の説明問題を1問だけ作成してください。",
            header,
            f"- 用語候補の一例：{term_hint}",
            "- 問題文だけで完結させること（資料表は不要）。",
            "- 形式：『次の用語を専門用語を用いて30〜60字で説明しなさい。必要なら具体例を1つ挙げること。用語：◯◯』",
            "- 解答の指示は30〜60字とする。"
        ])

    if style == "exam":
        # 宮城県公立高校入試テイスト（短い記述で因果・用語の正確さ重視）
        return "\n".join([
            "あなたは宮城県公立高校入試の傾向を踏まえた理科の出題者です。入試レベルの記述問題を1問だけ作成してください。",
            header,
            "- 問題文だけで完結させること（必要なら短い前提文や条件を示す）。",
            "- 回答は30〜80字で，一意性・因果関係・用語の正確さを重視させる設問にすること。",
            "- グラフや表は必須ではない（あっても行内の簡単な数値でよい）。",
            "- 例：『○○のとき△△が大きくなるのはなぜか。』など理由説明型。"
        ])

    if style == "school":
        # 学校定期テスト（教科書準拠の頻出観点）
        return "\n".join([
            "あなたは中学校の理科の担当教員です。学校の定期テストで問われやすい短い記述問題を1問だけ作成してください。",
            header,
            "- 問題文だけで完結させること（教科書基本事項に基づく説明問題）。",
            "- 回答は30〜60字を目安とする。定義＋根拠・仕組みの要点を簡潔に問うこと。",
            "- 例：『○○とは何かを説明し，□□との違いも簡潔に述べよ。』"
        ])

    # fallback
    return build_prompt_science(sel, "exam")

# ---------- 生成後チェック：資料が“自給自足”か検査＆補完 ----------
digits_re = re.compile(r"\d")

def _looks_self_contained(q: str) -> bool:
    """資料A/Bがあり、どちらかに数値が含まれているかの簡易判定"""
    if "資料A" not in q or "資料B" not in q:
        return False
    return bool(digits_re.search(q))

def ensure_self_contained_if_needed(question: str) -> str:
    """資料型のときのみ補完。資料型でなければそのまま返す。"""
    is_materials_style = ("資料A" in question) or ("資料B" in question)
    if not is_materials_style:
        return question
    if _looks_self_contained(question):
        return question
    try:
        resp = _call_openai(lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "あなたは中学生向け問題の編集者です。"},
                {"role": "user", "content":
                    "次の問題文は資料が抽象的で解けません。資料A/Bを、各3〜6行のミニ表/箇条書きで、少なくとも一方に数値（整数/％/mmなど）を含める形に書き換えてください。設問文は保持してください。問題文のみ返してください。"},
                {"role": "user", "content": question}
            ]
        ))
        fixed = (resp.choices[0].message.content or "").strip()
        return fixed if _looks_self_contained(fixed) else question
    except Exception:
        return question

# ---------- 出題（新規性＋具体性＋スタイル分岐） ----------
def generate_question_with_gpt(sel):
    """社会は資料読解、理科はqstyleで分岐（auto=比率で混在）"""
    last_texts = get_recent_seen(sel, n=12)

    # スタイル決定
    style = sel.get("qstyle", "auto")
    if sel["subject"] == "理科" and style == "auto":
        # おまかせ比率（資料:用語:入試:定期 = 3:2:3:2）
        style = random.choices(
            ["materials", "glossary", "exam", "school"],
            weights=[3, 2, 3, 2],
            k=1
        )[0]

    for attempt in range(NOVELTY_MAX_RETRY):
        if sel["subject"] == "社会":
            prompt = build_concrete_prompt_social(sel)
        else:
            prompt = build_prompt_science(sel, style)

        try:
            resp = _call_openai(lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.6 if attempt > 0 else 0.35,
                messages=[
                    {"role": "system", "content": "あなたは優秀な中学生向け問題作成AIです。必ず1問のみ、問題文だけを返してください。"},
                    {"role": "user", "content": prompt}
                ]
            ))
            question = (resp.choices[0].message.content or "").strip()
            question = question.lstrip("0123456789. 」）)」]』』『「『").strip()
            question = ensure_self_contained_if_needed(question)
        except Exception as e:
            print(f"[ERROR] generate_question_with_gpt attempt#{attempt+1}: {e}")
            question = None

        if not question:
            continue

        too_similar = any(SequenceMatcher(None, question, old).ratio() >= SIM_THRESHOLD for old in last_texts)
        if not too_similar:
            add_seen(sel, question)
            return question

    # 妥協案：一番似ていないもの
    best_q, best_min = None, 1.0
    for _ in range(2):
        try:
            if sel["subject"] == "社会":
                prompt = build_concrete_prompt_social(sel)
            else:
                prompt = build_prompt_science(sel, style)
            resp = _call_openai(lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "あなたは優秀な中学生向け問題作成AIです。必ず1問のみ、問題文だけを返してください。"},
                    {"role": "user", "content": prompt}
                ]
            ))
            q = (resp.choices[0].message.content or "").strip()
            q = q.lstrip("0123456789. 」）)」]』』『「『").strip()
            q = ensure_self_contained_if_needed(q)
        except Exception:
            q = None
        if not q:
            continue
        if not last_texts:
            best_q = q; break
        min_ratio = min(SequenceMatcher(None, q, old).ratio() for old in last_texts)
        if min_ratio < best_min:
            best_min = min_ratio; best_q = q
    if best_q:
        add_seen(sel, best_q)
    return best_q

# ---------- ユーティリティ：scoreの安全変換 ----------
def _to_int_0_10(x):
    try:
        v = int(float(str(x)))
    except Exception:
        return None
    return max(0, min(10, v))

# ---------- 採点（criteria入りJSON） ----------
def grade_answer_with_gpt(question, answer):
    instruction = (
        "次の問題の解答を採点してください。以下のJSON形式だけで出力してください（日本語）。\n"
        "{\n"
        '  "score": <0から10の整数>,\n'
        '  "comment": "<講評>",\n'
        '  "model_answer": "<模範解答（30〜80字程度）>",\n'
        '  "explanation": "<解説>",\n'
        '  "intention": "<出題意図>",\n'
        '  "criteria": [\n'
        '    {"name":"必須キーワード", "ok": true, "note":"必要語が含まれている／いない など"},\n'
        '    {"name":"理由の明確さ", "ok": true, "note":"因果・根拠の妥当性"},\n'
        '    {"name":"用語の正確さ", "ok": true, "note":"表現の誤りや曖昧さ"},\n'
        '    {"name":"分量（30〜80字）", "ok": true, "note":"過不足があれば具体的に"}\n'
        "  ]\n"
        "}\n"
        "注意：JSON以外の文字（説明文・マークダウン・前置き）は一切出力しないでください。"
    )
    user = f"問題: {question}\n解答: {answer}"
    try:
        resp = _call_openai(lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "あなたは中学生の記述式答案の厳格な採点官です。出力は必ず有効なJSONのみ。"},
                {"role": "user", "content": instruction},
                {"role": "user", "content": user}
            ]
        ))
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        data = json.loads(raw)

        score_val = _to_int_0_10(data.get("score"))
        score = f"10点中{score_val}点" if score_val is not None else "10点中—点"

        comment = data.get("comment") or "講評は生成されませんでした。"
        model_answer = data.get("model_answer") or ""
        explanation = data.get("explanation") or ""
        intention = data.get("intention") or ""

        criteria_in = data.get("criteria") if isinstance(data.get("criteria"), list) else []
        criteria = []
        for item in criteria_in:
            if isinstance(item, dict):
                criteria.append({
                    "name": item.get("name") or "",
                    "ok": bool(item.get("ok")),
                    "note": item.get("note") or ""
                })

        return {
            "score": score,
            "comment": comment,
            "model_answer": model_answer,
            "explanation": explanation,
            "intention": intention,
            "criteria": criteria
        }
    except Exception as e:
        print(f"[ERROR] grade_answer_with_gpt: {e}")
        return {
            "score": "10点中—点",
            "comment": "採点でエラーが発生しました。もう一度お試しください。",
            "model_answer": None,
            "explanation": None,
            "intention": None,
            "criteria": []
        }

# ---------- 画面状態 ----------
def store_view_state(**kwargs):
    session['view'] = kwargs

def load_view_state():
    return session.pop('view', None) or {}

# ---------- ルーティング ----------
@app.route("/", methods=["GET", "POST"])
def index():
    sel = get_selection()
    generated_questions = load_generated_questions()

    question = answer = score = comment = model_answer = explanation = intention = None
    criteria = []

    if request.method == "POST":
        # セレクトの更新を先に反映
        new_sel = selection_from_form(request.form)
        set_selection(new_sel)
        sel = new_sel

        action = request.form.get("action") or request.args.get("action")

        if action == "generate":
            if sel["subject"] == "社会" and sel["mode"] == "manual":
                if sel["field"] == "地理":
                    cands = question_bank_social["地理"].get(sel["era"], [f"{sel['era']}の要点を述べよ。"])
                elif sel["field"] == "歴史":
                    cands = question_bank_social["歴史"].get(sel["era"], [f"{sel['era']}の特色を述べよ。"])
                else:  # 公民
                    cands = question_bank_social["公民"].get(sel["era"], [f"{sel['era']}について説明しなさい。"])
                question = random.choice(cands)
            elif sel["subject"] == "理科" and sel["mode"] == "manual":
                cands = question_bank_science[sel["grade"]].get(sel["genre"], [])
                question = random.choice(cands) if cands else "（エラー）理科の単元が見つかりません。"
            elif sel["mode"] == "glossary":
                # 既存：用語説明モード（任意選択）
                term_from_form = request.form.get("term")
                terms = get_term_list(sel)
                term = term_from_form or (random.choice(terms) if terms else None)
                if not term:
                    question = "（エラー）用語リストが見つかりません。terms_custom.json の追加をご検討ください。"
                else:
                    field_or_genre = sel["field"] if sel["subject"] == "社会" else sel["genre"]
                    question = build_glossary_question(term, sel["subject"], field_or_genre)
            else:
                # ★ GPT自動生成（理科はqstyleで多様化）
                question = generate_question_with_gpt(sel)
                if not question:
                    question = "（エラー）自動生成に失敗しました。もう一度お試しください。"
                else:
                    detail = (
                        f"{sel['field']}-{sel['era']}" if sel["subject"] == "社会"
                        else f"{sel['grade']}-{sel['genre']}-[{sel.get('qstyle','auto')}]"
                    )
                    save_generated_question(question, sel["subject"], detail)

            store_view_state(question=question, answer=None, score=None, comment=None,
                             model_answer=None, explanation=None, intention=None, criteria=[])
            return redirect(url_for("index"))

        elif action == "select_stock":
            idx_str = request.form.get("selected_question_index")
            try:
                idx = int(idx_str)
                if 0 <= idx < len(generated_questions):
                    question = generated_questions[idx]["question"]
                else:
                    question = "（エラー）ストックからの出題に失敗しました。"
            except Exception:
                question = "（エラー）ストックの指定が不正です。"

            store_view_state(question=question, answer=None, score=None, comment=None,
                             model_answer=None, explanation=None, intention=None, criteria=[])
            return redirect(url_for("index"))

        elif action == "delete_stock":
            del_str = request.form.get("delete_index")
            try:
                del_idx = int(del_str)
                if 0 <= del_idx < len(generated_questions):
                    generated_questions.pop(del_idx)
                    save_generated_questions_all(generated_questions)  # ★ ここでも上限維持
                else:
                    store_view_state(question="（エラー）削除対象が見つかりません。")
            except Exception:
                store_view_state(question="（エラー）削除に失敗しました。")
            return redirect(url_for("index"))

        elif action == "submit":
            question = request.form.get("question")
            answer = request.form.get("input_answer")
            if not question:
                score = "10点中—点"; comment = "（エラー）問題文が未設定です。「問題を出題」してから解答してください。"
            elif not (answer and answer.strip()):
                score = "10点中—点"; comment = "（エラー）解答が空です。入力してから送信してください。"
            else:
                result = grade_answer_with_gpt(question, answer)
                score = result["score"]; comment = result["comment"]
                model_answer = result["model_answer"]; explanation = result["explanation"]
                intention = result["intention"]; criteria = result["criteria"]

            store_view_state(question=question, answer=answer, score=score, comment=comment,
                             model_answer=model_answer, explanation=explanation, intention=intention,
                             criteria=criteria)
            return redirect(url_for("index"))

        # actionなしPOSTは選択だけ更新 → GET
        return redirect(url_for("index"))

    # GET：状態復元
    state = load_view_state()
    if state:
        question = state.get("question")
        answer = state.get("answer")
        score = state.get("score")
        comment = state.get("comment")
        model_answer = state.get("model_answer")
        explanation = state.get("explanation")
        intention = state.get("intention")
        criteria = state.get("criteria", [])

    generated_questions = load_generated_questions()

    # 社会の単元リストをビューへ渡す（分野で切替）
    soc_units_map = {"地理": soc_geo_units, "歴史": soc_hist_units, "公民": soc_civ_units}

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        score=score,
        comment=comment,
        model_answer=model_answer,
        explanation=explanation,
        intention=intention,
        criteria=criteria,
        generated_questions=generated_questions,
        sel=sel,
        soc_units_map=soc_units_map,
        term_list=get_term_list(sel),
        request=request
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, host="127.0.0.1", port=port)
