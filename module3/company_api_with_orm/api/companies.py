"""
API для работы с компаниями (CRUD через ORM)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Company as CompanyModel
from schemas import Company, CompanyCreate, CompanyUpdate

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[Company])
def list_companies(db: Session = Depends(get_db)):
    """Получить список всех компаний"""
    companies = db.query(CompanyModel).all()
    return companies


@router.get("/{company_id}", response_model=Company)
def get_company(company_id: int, db: Session = Depends(get_db)):
    """Получить компанию по ID"""
    company = db.query(CompanyModel).filter(CompanyModel.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Компания не найдена")
    return company


@router.post("", response_model=Company, status_code=201)
def create_company(payload: CompanyCreate, db: Session = Depends(get_db)):
    """Создать новую компанию"""
    company = CompanyModel(name=payload.name, description=payload.description)
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


@router.put("/{company_id}", response_model=Company)
def update_company(company_id: int, payload: CompanyUpdate, db: Session = Depends(get_db)):
    """Обновить компанию"""
    company = db.query(CompanyModel).filter(CompanyModel.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Компания не найдена")
    
    if payload.name is not None:
        company.name = payload.name
    if payload.description is not None:
        company.description = payload.description
    
    db.commit()
    db.refresh(company)
    return company


@router.delete("/{company_id}", status_code=204)
def delete_company(company_id: int, db: Session = Depends(get_db)):
    """Удалить компанию"""
    company = db.query(CompanyModel).filter(CompanyModel.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Компания не найдена")
    
    db.delete(company)
    db.commit()
    return
