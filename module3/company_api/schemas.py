"""Схемы запросов и ответов (плоские, без наследования)."""
from typing import Optional
from pydantic import BaseModel


# --- Компании ---
class Company(BaseModel):
    id: int
    name: str
    description: Optional[str] = None


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


class PositionCreate(BaseModel):
    title: str
    company_id: int


class PositionUpdate(BaseModel):
    title: Optional[str] = None
    company_id: Optional[int] = None


# --- Пользователи ---
class User(BaseModel):
    id: int
    name: str
    email: str
    position_ids: list[int] = []


class UserCreate(BaseModel):
    name: str
    email: str
    position_ids: list[int] = []


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    position_ids: Optional[list[int]] = None


# --- Ответы с развёрнутыми данными ---
class PositionWithCompany(BaseModel):
    id: int
    title: str
    company_id: int
    company_name: Optional[str] = None


class UserWithPositions(BaseModel):
    id: int
    name: str
    email: str
    position_ids: list[int] = []
    positions: list[PositionWithCompany] = []
