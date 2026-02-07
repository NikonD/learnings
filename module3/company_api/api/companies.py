from fastapi import APIRouter, HTTPException
from storage import load_json, save_json, next_id, COMPANIES_FILE
from schemas import Company, CompanyCreate, CompanyUpdate

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[Company])
def list_companies():
    return load_json(COMPANIES_FILE, [])


@router.get("/{company_id}", response_model=Company)
def get_company(company_id: int):
    data = load_json(COMPANIES_FILE, [])
    for c in data:
        if c["id"] == company_id:
            return c
    raise HTTPException(status_code=404, detail="Компания не найдена")


@router.post("", response_model=Company, status_code=201)
def create_company(payload: CompanyCreate):
    data = load_json(COMPANIES_FILE, [])
    new_id = next_id(data)
    new_company = {"id": new_id, "name": payload.name, "description": payload.description}
    data.append(new_company)
    save_json(COMPANIES_FILE, data)
    return new_company


@router.put("/{company_id}", response_model=Company)
def update_company(company_id: int, payload: CompanyUpdate):
    data = load_json(COMPANIES_FILE, [])
    for i, c in enumerate(data):
        if c["id"] == company_id:
            if payload.name is not None:
                data[i]["name"] = payload.name
            if payload.description is not None:
                data[i]["description"] = payload.description
            save_json(COMPANIES_FILE, data)
            return data[i]
    raise HTTPException(status_code=404, detail="Компания не найдена")


@router.delete("/{company_id}", status_code=204)
def delete_company(company_id: int):
    data = load_json(COMPANIES_FILE, [])
    for i, c in enumerate(data):
        if c["id"] == company_id:
            data.pop(i)
            save_json(COMPANIES_FILE, data)
            return
    raise HTTPException(status_code=404, detail="Компания не найдена")
