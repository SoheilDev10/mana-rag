from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


# ============================== تنظیمات ==============================

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    # ollama | groq | none
    LLM_PROVIDER: str = "ollama"
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    API_NAME: str = "mana-rag-api"
    DEBUG: bool = False
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'mana_rag_db.db').as_posix()}"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()