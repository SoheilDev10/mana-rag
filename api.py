from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from settings import settings
from sqlalchemy.orm import Session
from app import get_chat_history,ask,build_index
from db import Messages,get_db,Sessions
from contextlib import asynccontextmanager


# ============================== API (سازگار با React) ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


app = FastAPI(title=settings.API_NAME, debug=settings.DEBUG, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    session_id: int | None = Field(default=None, ge=1)


@app.get("/")
def root():
    return {"message": "Mana RAG API"}


@app.post("/ask")
def ask_question(data: ChatRequest, db: Session = Depends(get_db)):
    # ۱) نشست را پیدا یا بساز (قبل از RAG، چون تاریخچه لازم است)
    if data.session_id is None:
        session = Sessions(title=data.question[:20])
        db.add(session)
        db.flush()
        history = "[]"
    else:
        session = db.query(Sessions).filter(Sessions.id == data.session_id).first()
        if session is None:
            raise HTTPException(status_code=404, detail="نشست مورد نظر پیدا نشد!")
        history = get_chat_history(db, session.id)

    # ۲) پاسخ RAG
    try:
        answer = ask(data.question, history)
    except Exception as e:
        import traceback
        traceback.print_exc()  # خطای اصلی در ترمینال چاپ می‌شود
        db.rollback()
        raise HTTPException(
            status_code=502, detail=f"خطا در پردازش سوال: {type(e).__name__}: {e}"
        )

    # ۳) ذخیره پیام
    message = Messages(
        question=data.question, answer=answer, session_id=session.id
    )
    try:
        db.add(message)
        db.commit()
        db.refresh(message)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="خطا در ذخیره اطلاعات")

    return {
        "session_id": session.id,
        "message_id": message.id,
        "question": data.question,
        "answer": answer,
        "message": "پیام با موفقیت دریافت شد",
    }


@app.get("/sessions")
def get_all_sessions(db: Session = Depends(get_db)):
    sessions = db.query(Sessions).all()
    return {"sessions": [{"id": s.id, "title": s.title} for s in sessions]}


@app.get("/messages/{session_id}")
def get_all_messages(session_id: int, db: Session = Depends(get_db)):
    session = db.query(Sessions).filter(Sessions.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="نشست مورد نظر وجود ندارد!")

    messages = db.query(Messages).filter(Messages.session_id == session_id).all()

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


@app.post("/sessions")
def create_session(db: Session = Depends(get_db)):
    new_session = Sessions(title="گفتگوی جدید")
    try:
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="خطا در ساخت نشست جدید")

    return {"session_id": new_session.id}


@app.put("/sessions/{session_id}")
def edit_session(session_id: int, data: ChatRequest, db: Session = Depends(get_db)):
    session = db.query(Sessions).filter(Sessions.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="نشست مورد نظر پیدا نشد")

    session.title = data.question[:20]
    db.commit()
    db.refresh(session)

    return {"session_id": session.id, "title": session.title}