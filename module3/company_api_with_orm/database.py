"""
database.py — подключение к PostgreSQL через SQLAlchemy ORM
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# URL подключения к PostgreSQL (порт 5555)
# psycopg3 использует формат postgresql+psycopg://
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://course_user:course_pass@localhost:5555/course_db"
)
# pgsql
# postgresql://course_user:course_pass@localhost:5555/course_db
# jdbc:postgresql://localhost:5555/course_db
# postgresql+psycopg://course_user:course_pass@localhost:5555/course_db

# Создаём движок SQLAlchemy
engine = create_engine(DATABASE_URL, echo=True)

# Фабрика сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()


def get_db():
    """Генератор сессий для использования в FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
