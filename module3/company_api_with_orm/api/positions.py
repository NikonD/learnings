"""
API для работы с должностями (CRUD через ORM)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from database import get_db
from models import Position as PositionModel, Company as CompanyModel
from schemas import Position, PositionCreate, PositionUpdate, PositionWithCompany

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("", response_model=list[Position])
def list_positions(db: Session = Depends(get_db)):
    """Получить список всех должностей"""
    positions = db.query(PositionModel).all()
    return positions


@router.get("/with-company", response_model=list[PositionWithCompany])
def list_positions_with_company(db: Session = Depends(get_db)):
    """
    Получить должности с названиями компаний.
    Использует JOIN для загрузки компаний одним запросом (избегает N+1).
    """
    # JOIN: positions LEFT OUTER JOIN companies
    positions = db.query(PositionModel).options(
        joinedload(PositionModel.company)  # Eager loading компании через JOIN
    ).all()
    
    return [
        PositionWithCompany(
            id=p.id,
            title=p.title,
            company_id=p.company_id,
            company_name=p.company.name if p.company else None,
        )
        for p in positions
    ]


@router.get("/{position_id}", response_model=Position)
def get_position(position_id: int, db: Session = Depends(get_db)):
    """Получить должность по ID"""
    position = db.query(PositionModel).filter(PositionModel.id == position_id).first()
    if not position:
        raise HTTPException(status_code=404, detail="Должность не найдена")
    return position


@router.post("", response_model=Position, status_code=201)
def create_position(payload: PositionCreate, db: Session = Depends(get_db)):
    """Создать новую должность"""
    # Проверяем, что компания существует
    company = db.query(CompanyModel).filter(CompanyModel.id == payload.company_id).first()
    if not company:
        raise HTTPException(status_code=400, detail=f"Компания с id {payload.company_id} не найдена")
    
    position = PositionModel(title=payload.title, company_id=payload.company_id)
    db.add(position)
    db.commit()
    db.refresh(position)
    return position


@router.put("/{position_id}", response_model=Position)
def update_position(position_id: int, payload: PositionUpdate, db: Session = Depends(get_db)):
    """Обновить должность"""
    position = db.query(PositionModel).filter(PositionModel.id == position_id).first()
    if not position:
        raise HTTPException(status_code=404, detail="Должность не найдена")
    
    if payload.title is not None:
        position.title = payload.title
    if payload.company_id is not None:
        # Проверяем, что новая компания существует
        company = db.query(CompanyModel).filter(CompanyModel.id == payload.company_id).first()
        if not company:
            raise HTTPException(status_code=400, detail=f"Компания с id {payload.company_id} не найдена")
        position.company_id = payload.company_id
    
    db.commit()
    db.refresh(position)
    return position


@router.delete("/{position_id}", status_code=204)
def delete_position(position_id: int, db: Session = Depends(get_db)):
    """Удалить должность"""
    position = db.query(PositionModel).filter(PositionModel.id == position_id).first()
    if not position:
        raise HTTPException(status_code=404, detail="Должность не найдена")
    
    db.delete(position)
    db.commit()
    return
