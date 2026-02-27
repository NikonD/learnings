"""
FastAPI CRUD: компании, должности, пользователи.
Данные в JSON-файлах (без БД). У пользователя может быть несколько должностей.
"""
from fastapi import FastAPI
from api.companies import router as companies_router
from api.positions import router as positions_router
from api.users import router as users_router

app = FastAPI(title="Company API", description="CRUD компаний, должностей и пользователей", doc_url="/docs")

app.include_router(companies_router)
app.include_router(positions_router)
app.include_router(users_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
