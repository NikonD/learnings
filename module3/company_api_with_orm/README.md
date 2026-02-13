# Company API with ORM

FastAPI приложение с PostgreSQL и SQLAlchemy ORM для работы с компаниями, должностями и пользователями.

## Структура проекта

```
company_api_with_orm/
├── docker-compose.yml    # PostgreSQL на порту 5555
├── requirements.txt      # Зависимости Python
├── database.py          # Подключение к БД (SQLAlchemy)
├── models.py            # ORM модели (таблицы)
├── schemas.py           # Pydantic схемы (валидация)
├── main.py              # FastAPI приложение
├── import_data.py       # Скрипт импорта данных из JSON
└── api/
    ├── companies.py     # CRUD для компаний
    ├── positions.py     # CRUD для должностей
    └── users.py         # CRUD для пользователей
```

## Установка и запуск

### 1. Запустить PostgreSQL через Docker

```bash
cd module3/company_api_with_orm
docker-compose up -d
```

Проверить, что БД запущена:
```bash
docker ps
# Должен быть контейнер course_module3_db на порту 5555
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Импортировать данные из старого проекта (опционально)

Если нужно перенести данные из `company_api` (JSON файлы) в БД:

```bash
python import_data.py
```

Скрипт прочитает JSON файлы из `../company_api/data/` и создаст записи в PostgreSQL.

### 4. Запустить приложение

```bash
python main.py
```

Или через uvicorn:
```bash
uvicorn main:app --reload
```

Приложение будет доступно на `http://localhost:8000`

## API Endpoints

### Компании
- `GET /companies` — список всех компаний
- `GET /companies/{id}` — получить компанию по ID
- `POST /companies` — создать компанию
- `PUT /companies/{id}` — обновить компанию
- `DELETE /companies/{id}` — удалить компанию

### Должности
- `GET /positions` — список всех должностей
- `GET /positions/with-company` — должности с названиями компаний
- `GET /positions/{id}` — получить должность по ID
- `POST /positions` — создать должность
- `PUT /positions/{id}` — обновить должность
- `DELETE /positions/{id}` — удалить должность

### Пользователи
- `GET /users` — список всех пользователей
- `GET /users/with-positions` — пользователи с развёрнутыми данными о должностях
- `GET /users/{id}` — получить пользователя по ID
- `GET /users/{id}/with-positions` — пользователь с должностями
- `POST /users` — создать пользователя
- `PUT /users/{id}` — обновить пользователя
- `DELETE /users/{id}` — удалить пользователя

## Документация API

После запуска приложения:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## База данных

### Подключение
- **Host**: localhost
- **Port**: 5555
- **Database**: course_db
- **User**: course_user
- **Password**: course_pass
- **URL формат**: `postgresql+psycopg://course_user:course_pass@localhost:5555/course_db`

### Таблицы
- `companies` — компании
- `positions` — должности (связь с компаниями)
- `users` — пользователи
- `user_position` — связующая таблица (many-to-many: пользователи ↔ должности)

### Схема связей
- Компания → много должностей (one-to-many)
- Пользователь ↔ должности (many-to-many)

## Примеры запросов

### Создать компанию
```bash
curl -X POST "http://localhost:8000/companies" \
  -H "Content-Type: application/json" \
  -d '{"name": "ООО Рога и копыта", "description": "Торговля"}'
```

### Создать должность
```bash
curl -X POST "http://localhost:8000/positions" \
  -H "Content-Type: application/json" \
  -d '{"title": "Разработчик", "company_id": 1}'
```

### Создать пользователя
```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{"name": "Иван", "email": "ivan@example.com", "position_ids": [1, 2]}'
```

## Отличия от company_api (без ORM)

- **Хранение**: PostgreSQL вместо JSON файлов
- **ORM**: SQLAlchemy для работы с БД
- **Связи**: автоматические связи между таблицами (relationships)
- **Валидация**: на уровне БД (foreign keys, unique constraints)
- **Производительность**: индексы, оптимизированные запросы
