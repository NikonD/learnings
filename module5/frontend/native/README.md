# Company API — нативный фронт (HTML, CSS, JS)

Клиент **без фреймворков** для бэкенда **module3/company_api_with_orm**. Учебный пример работы с `fetch` и DOM.

## Запуск

1. Запустите бэкенд: `cd module3/company_api_with_orm` → `docker-compose up -d` → `python main.py`
2. Откройте страницу через **статический сервер** (иначе CORS при открытии файла через `file://` может мешать):
   - **Live Server** в VS Code (порт 5500), или
   - из папки `native`: `npx serve -l 5500`
3. В браузере: http://localhost:5500 (или адрес, который показал serve).

## Файлы

- `index.html` — разметка, вкладки, формы
- `css/style.css` — стили
- `js/api.js` — базовый URL и функции запросов к API
- `js/app.js` — загрузка списков, формы создания/редактирования, удаление
