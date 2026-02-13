"""
models.py — SQLAlchemy ORM модели (таблицы в БД)
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship
from database import Base

# Связующая таблица для many-to-many: пользователь ↔ должности
user_position = Table(
    'user_position',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('position_id', Integer, ForeignKey('positions.id'), primary_key=True),
)


class Company(Base):
    """Модель компании"""
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False) # NOT NULL
    description = Column(String, nullable=True)

    # Связь: одна компания → много должностей
    positions = relationship("Position", back_populates="company")


class Position(Base):
    """Модель должности"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)

    # Связь: должность → компания
    company = relationship("Company", back_populates="positions")

    # для примера
    # Связь: должность ↔ пользователи (many-to-many)
    users = relationship("User", secondary=user_position, back_populates="positions")


class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)

    # Связь: пользователь ↔ должности (many-to-many)
    positions = relationship("Position", secondary=user_position, back_populates="users")
