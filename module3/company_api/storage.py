"""Работа с JSON-файлами (без БД)."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COMPANIES_FILE = DATA_DIR / "companies.json"
POSITIONS_FILE = DATA_DIR / "positions.json"
USERS_FILE = DATA_DIR / "users.json"


def load_json(path: Path, default: list) -> list:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def next_id(items: list) -> int:
    if not items:
        return 1
    return max(item["id"] for item in items) + 1
