from fastapi import APIRouter, HTTPException
from storage import load_json, save_json, next_id, COMPANIES_FILE, POSITIONS_FILE
from schemas import Position, PositionCreate, PositionUpdate

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("", response_model=list[Position])
def list_positions():
    return load_json(POSITIONS_FILE, [])


@router.get("/{position_id}", response_model=Position)
def get_position(position_id: int):
    data = load_json(POSITIONS_FILE, [])
    for p in data:
        if p["id"] == position_id:
            return p
    raise HTTPException(status_code=404, detail="Должность не найдена")


@router.post("", response_model=Position, status_code=201)
def create_position(payload: PositionCreate):
    companies = load_json(COMPANIES_FILE, [])
    if payload.company_id not in {c["id"] for c in companies}:
        raise HTTPException(status_code=400, detail="Компании с таким id не существует")
    data = load_json(POSITIONS_FILE, [])
    new_id = next_id(data)
    new_pos = {"id": new_id, "title": payload.title, "company_id": payload.company_id}
    data.append(new_pos)
    save_json(POSITIONS_FILE, data)
    return new_pos


@router.put("/{position_id}", response_model=Position)
def update_position(position_id: int, payload: PositionUpdate):
    data = load_json(POSITIONS_FILE, [])
    if payload.company_id is not None:
        companies = load_json(COMPANIES_FILE, [])
        if payload.company_id not in {c["id"] for c in companies}:
            raise HTTPException(status_code=400, detail="Компании с таким id не существует")
    for i, p in enumerate(data):
        if p["id"] == position_id:
            if payload.title is not None:
                data[i]["title"] = payload.title
            if payload.company_id is not None:
                data[i]["company_id"] = payload.company_id
            save_json(POSITIONS_FILE, data)
            return data[i]
    raise HTTPException(status_code=404, detail="Должность не найдена")


@router.delete("/{position_id}", status_code=204)
def delete_position(position_id: int):
    data = load_json(POSITIONS_FILE, [])
    for i, p in enumerate(data):
        if p["id"] == position_id:
            data.pop(i)
            save_json(POSITIONS_FILE, data)
            return
    raise HTTPException(status_code=404, detail="Должность не найдена")
