from contextlib import asynccontextmanager

from fastapi import Cookie, Depends, FastAPI, HTTPException,Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
import uuid
from settings import settings
from app import get_chat_history, ask, build_index
from db import Messages, get_db, Sessions


# ============================== API (سازگار با React) ==============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


app = FastAPI(
    title=settings.API_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    session_id: int | None = Field(default=None, ge=1)


@app.get("/")
def root():
    return {"message": "Mana RAG API"}

@app.get("/init")
def initialize_client(
    response: Response,
    client_id: str | None = Cookie(default=None)
):
    # اگر Cookie وجود نداشت یا معتبر نبود
    try:
        if client_id is None:
            raise ValueError

        uuid.UUID(client_id)

    except ValueError:
        client_id = str(uuid.uuid4())

    # ثبت Cookie در مرورگر
    response.set_cookie(
    key="client_id",
    value=client_id,
    httponly=True,
    max_age=60 * 60 * 24 * 30,
    samesite="lax",
    secure=False
  )

    return {
        "message": "شناسه مرورگر با موفقیت ایجاد شد",
    }

# ============================== ASK ==============================

@app.post("/ask")
def ask_question(
    data: ChatRequest,
    db: Session = Depends(get_db),
    client_id: str | None = Cookie(default=None)
):
    # اگر Cookie وجود نداشته باشد
    if client_id is None:
        raise HTTPException(
            status_code=401,
            detail="شناسه کاربر پیدا نشد!"
        )

    # ۱) نشست را پیدا یا بساز
    if data.session_id is None:

        session = Sessions(
            title=data.question[:20],
            client_id=client_id
        )

        db.add(session)
        db.flush()

        history = "[]"

    else:

        session = (
            db.query(Sessions)
            .filter(
                Sessions.id == data.session_id,
                Sessions.client_id == client_id
            )
            .first()
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="نشست مورد نظر پیدا نشد یا شما به آن دسترسی ندارید!"
            )

        history = get_chat_history(db, session.id)

    # ۲) پاسخ RAG
    try:
        answer = ask(data.question, history)

    except Exception as e:
        import traceback

        traceback.print_exc()

        db.rollback()

        raise HTTPException(
            status_code=502,
            detail=f"خطا در پردازش سوال: {type(e).__name__}: {e}"
        )

    # ۳) ذخیره پیام
    message = Messages(
        question=data.question,
        answer=answer,
        session_id=session.id
    )

    try:
        db.add(message)
        db.commit()
        db.refresh(message)

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="خطا در ذخیره اطلاعات"
        )

    return {
        "session_id": session.id,
        "message_id": message.id,
        "question": data.question,
        "answer": answer,
        "message": "پیام با موفقیت دریافت شد",
    }


# ============================== GET SESSIONS ==============================

@app.get("/sessions")
def get_all_sessions(
    db: Session = Depends(get_db),
    client_id: str | None = Cookie(default=None)
):

    if client_id is None:
        raise HTTPException(
            status_code=401,
            detail="شناسه کاربر پیدا نشد!"
        )

    sessions = (
        db.query(Sessions)
        .filter(Sessions.client_id == client_id)
        .all()
    )

    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title
            }
            for s in sessions
        ]
    }


# ============================== GET MESSAGES ==============================

@app.get("/messages/{session_id}")
def get_all_messages(
    session_id: int,
    db: Session = Depends(get_db),
    client_id: str | None = Cookie(default=None)
):

    if client_id is None:
        raise HTTPException(
            status_code=401,
            detail="شناسه کاربر پیدا نشد!"
        )

    # پیدا کردن Session متعلق به همین Cookie
    session = (
        db.query(Sessions)
        .filter(
            Sessions.id == session_id,
            Sessions.client_id == client_id
        )
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="نشست مورد نظر وجود ندارد یا شما به آن دسترسی ندارید!"
        )

    # پیدا کردن پیام‌های همین Session
    messages = (
        db.query(Messages)
        .filter(Messages.session_id == session_id)
        .all()
    )

    return {
        "session_id": session_id,
        "title": session.title,
        "messages": [
            {
                "id": m.id,
                "number_chat": index + 1,
                "question": m.question,
                "answer": m.answer,
            }
            for index, m in enumerate(messages)
        ],
    }


# ============================== CREATE SESSION ==============================

@app.post("/sessions")
def create_session(
    db: Session = Depends(get_db),
    client_id: str | None = Cookie(default=None)
):

    if client_id is None:
        raise HTTPException(
            status_code=401,
            detail="شناسه کاربر پیدا نشد!"
        )

    new_session = Sessions(
        title="گفتگوی جدید",
        client_id=client_id
    )

    try:
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="خطا در ساخت نشست جدید"
        )

    return {
        "session_id": new_session.id
    }


# ============================== EDIT SESSION ==============================

@app.put("/sessions/{session_id}")
def edit_session(
    session_id: int,
    data: ChatRequest,
    db: Session = Depends(get_db),
    client_id: str | None = Cookie(default=None)
):

    if client_id is None:
        raise HTTPException(
            status_code=401,
            detail="شناسه کاربر پیدا نشد!"
        )

    session = (
        db.query(Sessions)
        .filter(
            Sessions.id == session_id,
            Sessions.client_id == client_id
        )
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="نشست مورد نظر پیدا نشد یا شما به آن دسترسی ندارید!"
        )

    session.title = data.question[:20]

    try:
        db.commit()
        db.refresh(session)

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="خطا در ویرایش نشست"
        )

    return {
        "session_id": session.id,
        "title": session.title
    }