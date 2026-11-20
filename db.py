from sqlalchemy import Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from settings import settings

# ============================== دیتابیس ==============================

engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Sessions(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(50), nullable=False)
    client_id = Column(String(100), nullable=False, index=True)
    
    message = relationship("Messages", back_populates="session")


class Messages(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True, nullable=False)
    question = Column(String(500), nullable=False)
    answer = Column(String, nullable=False)
    
    session = relationship("Sessions", back_populates="message")


Base.metadata.create_all(bind=engine)