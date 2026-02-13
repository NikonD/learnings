"""Схемы запросов и ответов (Pydantic)."""
from typing import Optional
from pydantic import BaseModel


# --- Компании ---
class Company(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True  # для SQLAlchemy моделей


class CompanyCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


# --- Должности ---
class Position(BaseModel):
    id: int
    title: str
    company_id: int

    class Config:
        from_attributes = True


class PositionCreate(BaseModel):
    title: str
    company_id: int


class PositionUpdate(BaseModel):
    title: Optional[str] = None
    company_id: Optional[int] = None


class PositionWithCompany(BaseModel):
    id: int
    title: str
    company_id: int
    company_name: Optional[str] = None

    class Config:
        from_attributes = True


# --- Пользователи ---
class User(BaseModel):
    id: int
    name: str
    email: str
    position_ids: list[int] = []

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    name: str
    email: str
    position_ids: list[int] = []


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    position_ids: Optional[list[int]] = None


class UserWithPositions(BaseModel):
    id: int
    name: str
    email: str
    position_ids: list[int] = []
    positions: list[PositionWithCompany] = []

    class Config:
        from_attributes = True
