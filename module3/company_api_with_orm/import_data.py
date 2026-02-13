"""
Скрипт для импорта данных из JSON файлов (старый проект) в PostgreSQL через ORM.

Использование:
    python import_data.py
"""
import json
from pathlib import Path
from database import SessionLocal
from models import Company, Position, User

# Путь к JSON файлам из старого проекта
OLD_PROJECT_DIR = Path(__file__).parent.parent / "company_api" / "data"
COMPANIES_FILE = OLD_PROJECT_DIR / "companies.json"
POSITIONS_FILE = OLD_PROJECT_DIR / "positions.json"
USERS_FILE = OLD_PROJECT_DIR / "users.json"


def load_json(file_path: Path) -> list:
    """Загрузить данные из JSON файла"""
    if not file_path.exists():
        print(f"⚠️  Файл {file_path} не найден!")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_companies(db):
    """Импортировать компании"""
    print("📦 Импорт компаний...")
    companies_data = load_json(COMPANIES_FILE)
    
    for item in companies_data:
        # Проверяем, существует ли уже компания с таким ID
        existing = db.query(Company).filter(Company.id == item["id"]).first()
        if existing:
            print(f"  ⏭️  Компания {item['id']} ({item['name']}) уже существует, пропускаем")
            continue
        
        company = Company(
            id=item["id"],  # Сохраняем оригинальные ID
            name=item["name"],
            description=item.get("description")
        )
        db.add(company)
        print(f"  ✅ Добавлена компания: {item['name']}")
    
    db.commit()
    print(f"✅ Импортировано компаний: {len(companies_data)}\n")


def import_positions(db):
    """Импортировать должности"""
    print("💼 Импорт должностей...")
    positions_data = load_json(POSITIONS_FILE)
    
    for item in positions_data:
        # Проверяем, существует ли уже должность с таким ID
        existing = db.query(Position).filter(Position.id == item["id"]).first()
        if existing:
            print(f"  ⏭️  Должность {item['id']} ({item['title']}) уже существует, пропускаем")
            continue
        
        # Проверяем, что компания существует
        company = db.query(Company).filter(Company.id == item["company_id"]).first()
        if not company:
            print(f"  ⚠️  Компания с id {item['company_id']} не найдена, пропускаем должность {item['title']}")
            continue
        
        position = Position(
            id=item["id"],  # Сохраняем оригинальные ID
            title=item["title"],
            company_id=item["company_id"]
        )
        db.add(position)
        print(f"  ✅ Добавлена должность: {item['title']} (компания: {company.name})")
    
    db.commit()
    print(f"✅ Импортировано должностей: {len(positions_data)}\n")


def import_users(db):
    """Импортировать пользователей"""
    print("👥 Импорт пользователей...")
    users_data = load_json(USERS_FILE)
    
    for item in users_data:
        # Проверяем, существует ли уже пользователь с таким ID
        existing = db.query(User).filter(User.id == item["id"]).first()
        if existing:
            print(f"  ⏭️  Пользователь {item['id']} ({item['name']}) уже существует, пропускаем")
            continue
        
        # Проверяем, что все должности существуют
        position_ids = item.get("position_ids", [])
        if position_ids:
            positions = db.query(Position).filter(Position.id.in_(position_ids)).all()
            found_ids = {p.id for p in positions}
            missing = set(position_ids) - found_ids
            if missing:
                print(f"  ⚠️  Должности с id {list(missing)} не найдены, пропускаем пользователя {item['name']}")
                continue
        
        user = User(
            id=item["id"],  # Сохраняем оригинальные ID
            name=item["name"],
            email=item["email"]
        )
        
        # Устанавливаем связи с должностями
        if position_ids:
            user.positions = positions
        
        db.add(user)
        print(f"  ✅ Добавлен пользователь: {item['name']} (должностей: {len(position_ids)})")
    
    db.commit()
    print(f"✅ Импортировано пользователей: {len(users_data)}\n")


def main():
    """Главная функция импорта"""
    print("🚀 Начало импорта данных из JSON в PostgreSQL\n")
    print(f"📂 Ищем файлы в: {OLD_PROJECT_DIR}\n")
    
    db = SessionLocal()
    try:
        # Импортируем в правильном порядке: компании → должности → пользователи
        import_companies(db)
        import_positions(db)
        import_users(db)
        
        print("✨ Импорт завершён успешно!")
        
        # Показываем статистику
        companies_count = db.query(Company).count()
        positions_count = db.query(Position).count()
        users_count = db.query(User).count()
        
        print(f"\n📊 Статистика в БД:")
        print(f"   Компаний: {companies_count}")
        print(f"   Должностей: {positions_count}")
        print(f"   Пользователей: {users_count}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при импорте: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
