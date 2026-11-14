"""
Mana RAG - همه چیز در یک فایل
  - API: دقیقا همان endpointها و فرمت JSON فایل main.py قبلی (برای React)
  - الگوریتم RAG: rewrite -> filter -> retrieve -> check -> answer (مثل main-(2).py)
  - Embedding و Chroma: مدل E5 با پیشوند passage:/query: و فاصله cosine (مثل code-f.py)

نصب:
  pip install fastapi uvicorn sqlalchemy pydantic-settings python-dotenv \
              chromadb sentence-transformers langchain-groq langchain-text-splitters

اجرا:
  uvicorn app:app --port 8000 --reload

فایل .env:
  GROQ_API_KEY=...
"""
import json
import re
from pathlib import Path
from settings import settings
import chromadb
from dotenv import load_dotenv
import urllib.request
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from db import Messages
from settings import BASE_DIR

load_dotenv()

DOCUMENTS_PATH = BASE_DIR / "documents"
CHROMA_PATH = str(BASE_DIR / "chroma_db_e5")
COLLECTION_NAME = "mana_info_e5"      # اسم جدید چون فضای برداری و مدل عوض شده
EMBED_MODEL = "intfloat/multilingual-e5-small"
N_RESULTS = 15
# فاصله cosine = 1 - شباهت. مقدار قبلی (0.6) برای L2 بود؛ برای E5 باید بر اساس
# تست خودتان تنظیم شود (کمتر = سخت‌گیرانه‌تر).
DISTANCE_THRESHOLD = 0.3
HISTORY_LIMIT = 10                    # تعداد پیام آخر که به LLM داده می‌شود

# متادیتای هر فایل سند
type_metadatas = {
    "company-info.txt": {"category": "company", "type": "company_info"},
    "contact.txt": {"category": "company", "type": "contact"},
    "web-design.txt": {"category": "services", "type": "web_design"},
    "ai.txt": {"category": "services", "type": "ai_services"},
    "app-design.txt": {"category": "services", "type": "app_design"},
    "seo.txt": {"category": "services", "type": "seo_services"},
    "automation.txt": {"category": "services", "type": "automation"},
    "graphic-design.txt": {"category": "services", "type": "graphic_design"},
}

ALLOWED_TYPES = {m["type"] for m in type_metadatas.values()}

# ============================== مدل‌ها و ایندکس ==============================

class _Reply:
    def __init__(self, content):
        self.content = content

class OllamaLLM:
    """مدل محلی از طریق Ollama؛ نیازی به اینترنت یا کلید ندارد."""

    def invoke(self, prompt):
        body = json.dumps({
            "model": settings.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(
            settings.OLLAMA_URL + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        return _Reply(data["message"]["content"])


if settings.LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
elif settings.LLM_PROVIDER == "ollama":
    llm = OllamaLLM()
else:
    llm = None  # حالت none: فقط جستجو، بدون مدل زبانی
embedder = SentenceTransformer(EMBED_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

def embed_passages(texts):
    """E5: برای سند پیشوند passage:"""
    return embedder.encode(
        ["passage: " + t for t in texts], normalize_embeddings=True
    ).tolist()


def embed_query(text):
    """E5: برای سوال پیشوند query:"""
    return embedder.encode("query: " + text, normalize_embeddings=True).tolist()


def build_index():
    """فایل‌های documents/*.txt را چانک کرده و با upsert در Chroma ذخیره می‌کند."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    ids, docs, metas = [], [], []

    for file_path in sorted(DOCUMENTS_PATH.glob("*.txt")):
        info = type_metadatas.get(file_path.name)
        if info is None:
            print(f"[warn] {file_path.name} در type_metadatas نیست؛ رد شد.")
            continue

        text = file_path.read_text(encoding="utf-8")
        for index, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{file_path.stem}_chunk_{index}")
            docs.append(chunk)
            metas.append({
                "source": file_path.name,
                "chunk_index": index,
                "category": info["category"],
                "type": info["type"],
            })

    if not docs:
        print(f"[warn] هیچ سندی در {DOCUMENTS_PATH} پیدا نشد.")
        return

    collection.upsert(
        ids=ids, documents=docs, embeddings=embed_passages(docs), metadatas=metas
    )
    print(f"[rag] {collection.count()} چانک در Chroma ذخیره است.")


# ============================== الگوریتم RAG ==============================

def parse_json(text, default):
    """خروجی JSON مدل را (حتی با ```json) می‌خواند؛ در صورت خطا default برمی‌گرداند."""
    try:
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0) if match else text)
    except Exception:
        return default


def rewrite_query(question, chat_history):
    prompt = f"""
    تو یک بازنویسنده سوال برای سیستم RAG یک وبسایت خدماتی هستی.
    تو باید سوالات کاربر رو که دریافت میکنی به بهترین شکل اون رو کوتاه
    و بهینه ترش کنی برای query زدن داخل collection دیتابیس
    مفهموم سوال اصلا نباید تغیر کنه،فقط کوتاه و قابل فهم تر برای query زدن باید بشه.
    میخوام که سوال بازنویسی شده قابل فهم باشه و یک چیز مبهم نباشه


    برای مثال:
    سوال کاربر:قیمت خدمات طراحی سایت چقدر است؟
    سوال بازنویسی شده:قیمت خدمات طراحی سایت.

    سوال کاربر:شماره تماس شرکت رو بهم بده
    سوال بازنویسی شده:شماره تماس شرکت

    سوال کاربر:طراحی سایت یا خدمات سئو چقدر طول میکشه؟
    سوال بازنویسی شده:زمان طراحی سایت

    دقت کن که سوال بازنویسی شده رو بدون هیچ توضیح اضافه ای فقط برگردونی.

    اگر چتت با کاربر تاریخچه داشت بیا و از اون اطلاعات برای
    بازنویسی سوال استفاده کن.

    برای مثال:
    سوال کاربر داخل تاریخچه:
    ایمیل شرکت چیه؟
    و سپس بعدش میپرسه حالا شمارش چنده
    اینجا با توجه به تاریخچه چت باید تشخیص بدی موضوع رو
    و حالا به عنوان مثال سوالی که تو بازنویسی میکنی:
    شماره تماس شرکت

    سوال کاربر:
    {question}

    تاریخچه چت:
    {chat_history}
    """
    return llm.invoke(prompt).content.strip()


def get_filter_with_llm(question):
    prompt = f"""
    تو یک فیلتر کننده سوال کاربر برای یک سیستم RAG برای وبسایت خدماتی هستی.
    تو وقتی که سوال کاربر رو دریافت میکنی باید اون رو فیلتر کنی و هم تشخیص بدی
    که آیا مربوط به شرکت هست یا یک سوال بی ربطه و نباید به llm داد.

    ساختار مورد نیاز:
    {{
        "type":"company_info",
        "need_retrieval":true
    }}

    هروقت که سوال مربوط به شرکت بود مقدار need_retrieval باید true باشه
    و اگر که سوال کاربر مروبط به شرکت نبود باید مقدارش رو برابر با false قرار بدی.
    دقت کن که اگر سوال کاربر مروبط به شرکت بود و تو نتونستی به type های زیر ربطش بدی،
    باید مقدار type رو null و مقدار need_retrieval رو true قرار بدی.
    و اگر که سوال کاملا غیر مربتط بود با شرکت،مقدار type رو null و need_retrieval رو
    false قرار بده.
    اگر که سوال درباره حدمات شرکت بود ولی نتونستی به هیچکدوم از type های موجود
    ربطش بدی،بیا و مقدار type رو برابر company_info قرار بده.

    فقط حق داری که از تایپ های زیر استفاده کنی برای فیلتر کردن:
    -company_info
    -contact
    -web_design
    -ai_services
    -app_design
    -seo_services
    -automation
    -graphic_design

    برای مثال:
    سوال کاربر:ساعت کاری شرکت کی هست؟
    خروجی:
    {{
        "type":"company_info",
        "need_retrieval":true
    }}

    سوال کاربر:شماره تماس شرکت چنده؟
    خروجی:
    {{
        "type":"contact",
        "need_retrieval":true
    }}

    سوال کاربر:آیا شرکت پشتیبانی بعد از خدمات هم دارد؟
    خروجی:
    {{
        "type":null,
        "need_retrieval":true
    }}

    دقت کن که بدون هیچ توضیح اضافه ای فقط خروجی رو باید برگردونی.

    سوال کاربر:
    {question}
    """
    result = parse_json(
        llm.invoke(prompt).content, {"type": None, "need_retrieval": True}
    )
    if result.get("type") not in ALLOWED_TYPES:
        result["type"] = None
    result["need_retrieval"] = bool(result.get("need_retrieval", True))
    return result


def create_context(question, filter_data, use_filter):
    if collection.count() == 0:
        return ""

    kwargs = {}
    if use_filter and filter_data.get("type") is not None:
        kwargs["where"] = {"type": filter_data["type"]}

    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(N_RESULTS, collection.count()),
        **kwargs,
    )

    nearest = [
        doc
        for doc, dist in zip(results["documents"][0], results["distances"][0])
        if dist <= DISTANCE_THRESHOLD
    ]
    context="\n-".join(nearest)
    return context


def check_context(question, context):
    if not context.strip():
        return {"is_relevant": False}

    prompt = f"""
    تو یک سیستم RAG هوشمند برای یک وبسایت حدماتی هستی.
    بیا سوال کاربر و context رو چک بکن،اگر که با اطلاعاتی که درون
    context وجود داره میتونی به سوال کاربر پاسخ خوب بدی و در کل اطلاعات
    مورد نیاز برای پاسخ به سوال کاربر داری رو بیا و مقدار is_relevant رو
    برابر false قرار بده و اگر هم که اطلاعات context برای پاسخ کافی بود،بیا و
    مقدارش رو برابر true قرار بده.

    ساختار:
    {{
        "is_relevant":true
    }}

    دقت کن که باید بدون هیچ توضیح اضافه دیگه ای مقدار رو برگردونی.

    context:
    {context}

    سوال کاربر:
    {question}
    """
    result = parse_json(llm.invoke(prompt).content, {"is_relevant": False})
    result["is_relevant"] = bool(result.get("is_relevant", False))
    return result


def get_chat_history(db: Session, session_id: int):
    """تاریخچه را مستقیم از دیتابیس می‌خواند (بدون درخواست HTTP به خودش)."""
    messages = (
        db.query(Messages)
        .filter(Messages.session_id == session_id)
        .order_by(Messages.id.desc())
        .limit(HISTORY_LIMIT)
        .all()
    )[::-1]
    return json.dumps(
        [{"question": m.question, "answer": m.answer} for m in messages],
        ensure_ascii=False,
    )


def ask(question, history):
    if llm is None:
        context = create_context(question, {}, use_filter=False)
        if not context.strip():
            return "متاسفانه اطلاعات موجود درباره این سوال را ندارم!"
        return "اطلاعات مرتبط پیدا شده:\n-" + context

    query = rewrite_query(question, history)

    filter_data = get_filter_with_llm(query)
    if not filter_data["need_retrieval"]:
        return "متاسفانه به سوالات غیر مرتبط پاسخی نمی توانم بدم!"

    context = create_context(query, filter_data, use_filter=True)
    context_check = check_context(question, context)

    if not context_check["is_relevant"]:
        new_query = rewrite_query(query, history)
        context = create_context(new_query, filter_data, use_filter=False)
        context_check = check_context(question, context)

        if not context_check["is_relevant"]:
            return "متاسفانه اطلاعات موجود درباره این سوال را ندارم!"

    prompt = f"""
    تو یک RAG هوشمند یک وبسایت خدماتی برای پاسخ به سوالات کاربران هستی.

    قوانینی که باید رعایت کنی:
    -قبل از پاسخگویی به کاربر سلام کن
    -به کاربران جواب گرم و دوستانه بده و اگر که امکانش هست از ایموجی هم استفاده کن
    -فقط از اطلاعاتی که بهت داده شده برای پاسخگویی استفاده کن
    -جواب الکی از خودت در نیار
    -از اطلاعات موجود به بهترین شکل استفاده کن و بهترین و واضح ترین پاسخ رو بده
    -بعد از پاسخگویی حتما بگو اگر سوال دیگری دارید خوشحال میشم کمکتون کنم
    -اگه لازمه یک ساختار مناسب و قابل فهم برای اطلاعات درست کن،مثل ساخت جدول

    اطلاعات موجود:
    {context}

    سوال کاربر:
    {question}
    """
    return llm.invoke(prompt).content

