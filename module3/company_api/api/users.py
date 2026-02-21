from fastapi import APIRouter, HTTPException
from storage import load_json, save_json, next_id, COMPANIES_FILE, POSITIONS_FILE, USERS_FILE
from schemas import User, UserCreate, UserUpdate, UserWithPositions, PositionWithCompany

router = APIRouter(prefix="/users", tags=["users"])


def _positions_with_company(position_ids: list[int]) -> list[PositionWithCompany]:
    positions = load_json(POSITIONS_FILE, [])
    companies = {c["id"]: c["name"] for c in load_json(COMPANIES_FILE, [])}
    result = []
    for p in positions:
        if p["id"] in position_ids:
            result.append(PositionWithCompany(
                id=p["id"],
                title=p["title"],
                company_id=p["company_id"],
                company_name=companies.get(p["company_id"]),
            ))
    return result


@router.get("", response_model=list[User])
def list_users():
    return load_json(USERS_FILE, [])


@router.get("/with-positions", response_model=list[UserWithPositions])
def list_users_with_positions():
    users = load_json(USERS_FILE, [])
    return [
        UserWithPositions(
            id=u["id"],
            name=u["name"],
            email=u["email"],
            position_ids=u.get("position_ids", []),
            positions=_positions_with_company(u.get("position_ids", [])),
        )
        for u in users
    ]


@router.get("/count")
def users_count():
    """Количество пользователей."""
    data = load_json(USERS_FILE, [])
    return {"count": len(data)}


@router.get("/{user_id}", response_model=User)
def get_user(user_id: int):
    data = load_json(USERS_FILE, [])
    for u in data:
        if u["id"] == user_id:
            return u
    raise HTTPException(status_code=404, detail="Пользователь не найден")


@router.get("/{user_id}/with-positions", response_model=UserWithPositions)
def get_user_with_positions(user_id: int):
    data = load_json(USERS_FILE, [])
    for u in data:
        if u["id"] == user_id:
            return UserWithPositions(
                id=u["id"],
                name=u["name"],
                email=u["email"],
                position_ids=u.get("position_ids", []),
                positions=_positions_with_company(u.get("position_ids", [])),
            )
    raise HTTPException(status_code=404, detail="Пользователь не найден")


@router.post("", response_model=User, status_code=201)
def create_user(payload: UserCreate):
    positions_data = load_json(POSITIONS_FILE, [])
    valid_ids = {p["id"] for p in positions_data}
    for pid in payload.position_ids:
        if pid not in valid_ids:
            raise HTTPException(status_code=400, detail=f"Должности с id {pid} не существует")
    data = load_json(USERS_FILE, [])
    new_id = next_id(data)
    new_user = {
        "id": new_id,
        "name": payload.name,
        "email": payload.email,
        "position_ids": payload.position_ids,
    }
    data.append(new_user)
    save_json(USERS_FILE, data)
    return new_user


@router.put("/{user_id}", response_model=User)
def update_user(user_id: int, payload: UserUpdate):
    data = load_json(USERS_FILE, [])
    if payload.position_ids is not None:
        valid_ids = {p["id"] for p in load_json(POSITIONS_FILE, [])}
        for pid in payload.position_ids:
            if pid not in valid_ids:
                raise HTTPException(status_code=400, detail=f"Должности с id {pid} не существует")
    for i, u in enumerate(data):
        if u["id"] == user_id:
            if payload.name is not None:
                data[i]["name"] = payload.name
            if payload.email is not None:
                data[i]["email"] = payload.email
            if payload.position_ids is not None:
                data[i]["position_ids"] = payload.position_ids
            save_json(USERS_FILE, data)
            return data[i]
    raise HTTPException(status_code=404, detail="Пользователь не найден")


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int):
    data = load_json(USERS_FILE, [])
    for i, u in enumerate(data):
        if u["id"] == user_id:
            data.pop(i)
            save_json(USERS_FILE, data)
            return
    raise HTTPException(status_code=404, detail="Пользователь не найден")
