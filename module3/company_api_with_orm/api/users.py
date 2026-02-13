"""
API для работы с пользователями (CRUD через ORM)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from database import get_db
from models import User as UserModel, Position as PositionModel
from schemas import User, UserCreate, UserUpdate, UserWithPositions, PositionWithCompany

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[User])
def list_users(db: Session = Depends(get_db)):
    """
    Получить список всех пользователей.
    Использует eager loading для загрузки позиций одним запросом (избегает N+1).
    """
    # JOIN: users LEFT OUTER JOIN user_position LEFT OUTER JOIN positions
    users = db.query(UserModel).options(
        joinedload(UserModel.positions)  # Eager loading позиций через JOIN
    ).all()
    
    return [
        User(
            id=u.id,
            name=u.name,
            email=u.email,
            position_ids=[p.id for p in u.positions],
        )
        for u in users
    ]


@router.get("/with-positions", response_model=list[UserWithPositions])
def list_users_with_positions(db: Session = Depends(get_db)):
    """
    Получить пользователей с развёрнутыми данными о должностях.
    Использует JOIN для загрузки позиций и компаний одним запросом (избегает N+1).
    """
    # JOIN: users → positions → companies (все связи загружаются одним запросом)
    users = db.query(UserModel).options(
        joinedload(UserModel.positions).joinedload(PositionModel.company)  # Цепочка JOIN
    ).all()

    # SELECT     
    # *
    # FROM users
    # LEFT JOIN positions ON users.id = positions.user_id
    # LEFT JOIN companies ON positions.company_id = companies.id


    result = []
    for u in users:
        positions_data = [
            PositionWithCompany(
                id=p.id,
                title=p.title,
                company_id=p.company_id,
                company_name=p.company.name if p.company else None,
            )
            for p in u.positions
        ]
        result.append(
            UserWithPositions(
                id=u.id,
                name=u.name,
                email=u.email,
                position_ids=[p.id for p in u.positions],
                positions=positions_data,
            )
        )
    return result


@router.get("/{user_id}", response_model=User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Получить пользователя по ID.
    Использует JOIN для загрузки позиций одним запросом.
    """
    user = db.query(UserModel).options(
        joinedload(UserModel.positions)  # Eager loading позиций
    ).filter(UserModel.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    return User(
        id=user.id,
        name=user.name,
        email=user.email,
        position_ids=[p.id for p in user.positions],
    )


@router.get("/{user_id}/with-positions", response_model=UserWithPositions)
def get_user_with_positions(user_id: int, db: Session = Depends(get_db)):
    """
    Получить пользователя с развёрнутыми данными о должностях.
    Использует JOIN для загрузки позиций и компаний одним запросом.
    """
    # JOIN: user → positions → companies (цепочка JOIN)
    user = db.query(UserModel).options(
        joinedload(UserModel.positions).joinedload(PositionModel.company)
    ).filter(UserModel.id == user_id).first() # WHERE users.id = 1
    
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # user.positions - это список объектов Position (благодаря relationship в модели User)
    # p - это каждый элемент списка (объект Position) в list comprehension
    positions_data = [
        PositionWithCompany(
            id=p.id,                    # p - объект Position из user.positions
            title=p.title,
            company_id=p.company_id,
            company_name=p.company.name if p.company else None,  # p.company - связь к Company
        )
        for p in user.positions  # итерируемся по списку позиций пользователя
    ]
    
    return UserWithPositions(
        id=user.id,
        name=user.name,
        email=user.email,
        position_ids=[p.id for p in user.positions],
        positions=positions_data,
    )


@router.post("", response_model=User, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Создать нового пользователя"""
    # Проверяем, что все должности существуют
    if payload.position_ids:
        positions = db.query(PositionModel).filter(PositionModel.id.in_(payload.position_ids)).all()
        if len(positions) != len(payload.position_ids):
            found_ids = {p.id for p in positions}
            missing = set(payload.position_ids) - found_ids
            raise HTTPException(
                status_code=400,
                detail=f"Должности с id {list(missing)} не найдены"
            )
    
    user = UserModel(name=payload.name, email=payload.email)
    if payload.position_ids:
        user.positions = positions
    
    db.add(user)
    db.commit()
    # Перезагружаем пользователя с позициями через JOIN для ответа
    user = db.query(UserModel).options(joinedload(UserModel.positions)).filter(UserModel.id == user.id).first()
    
    return User(
        id=user.id,
        name=user.name,
        email=user.email,
        position_ids=[p.id for p in user.positions],
    )


@router.put("/{user_id}", response_model=User)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    """
    Обновить пользователя.
    Использует JOIN для загрузки позиций после обновления.
    """
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    if payload.name is not None:
        user.name = payload.name
    if payload.email is not None:
        user.email = payload.email
    if payload.position_ids is not None:
        # Проверяем, что все должности существуют
        positions = db.query(PositionModel).filter(PositionModel.id.in_(payload.position_ids)).all()
        if len(positions) != len(payload.position_ids):
            found_ids = {p.id for p in positions}
            missing = set(payload.position_ids) - found_ids
            raise HTTPException(
                status_code=400,
                detail=f"Должности с id {list(missing)} не найдены"
            )
        user.positions = positions
    
    db.commit()
    # Перезагружаем с позициями через JOIN
    user = db.query(UserModel).options(joinedload(UserModel.positions)).filter(UserModel.id == user_id).first()
    
    return User(
        id=user.id,
        name=user.name,
        email=user.email,
        position_ids=[p.id for p in user.positions],
    )


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Удалить пользователя"""
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    db.delete(user)
    db.commit()
    return
