"""
FastAPI CRUD: компании, должности, пользователи.
Использует PostgreSQL через SQLAlchemy ORM.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# CORS: разрешаем запросы с фронтендов (native и React на других портах)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5500", "http://127.0.0.1:5500", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
