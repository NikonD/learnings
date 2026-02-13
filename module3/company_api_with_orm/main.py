"""
FastAPI CRUD: компании, должности, пользователи.
Использует PostgreSQL через SQLAlchemy ORM.
"""
from fastapi import FastAPI
from database import engine, Base
from api.companies import router as companies_router
from api.positions import router as positions_router
from api.users import router as users_router

# Создаём таблицы в БД (если их ещё нет)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Company API (with ORM)",
    description="CRUD компаний, должностей и пользователей через PostgreSQL + SQLAlchemy ORM"
)

app.include_router(companies_router)
app.include_router(positions_router)
app.include_router(users_router)


@app.get("/")
def root():
    return {
        "message": "Company API with ORM",
        "docs": "/docs",
        "database": "PostgreSQL on port 5555"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
